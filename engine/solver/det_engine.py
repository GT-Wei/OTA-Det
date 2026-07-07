"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""


import sys
import math
from typing import Iterable

import torch
import torch.amp
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils


def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)

    cur_iters = epoch * len(data_loader)

    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        for t in targets:
            for key, value in t.items():
                if isinstance(value, str) or isinstance(value, list) or isinstance(value, dict) or isinstance(value, bool) or isinstance(value, int):
                    t[key] = value
                elif value==None:
                    t[key] = value
                else:
                    t[key] = value.to(device, non_blocking=True)

        multi_scale_size = None
        if targets and 'multi_scale_size' in targets[0]:
            multi_scale_size = targets[0].get('multi_scale_size')
            for tg in targets:
                tg.pop('multi_scale_size', None)

        if multi_scale_size is not None:
            # Resize on GPU to avoid CPU-side multi-scale overhead
            resize_size = (multi_scale_size, multi_scale_size) if isinstance(multi_scale_size, int) else multi_scale_size
            samples = F.interpolate(samples, size=resize_size)
            if 'masks' in targets[0]:
                for tg in targets:
                    tg['masks'] = F.interpolate(tg['masks'], size=resize_size, mode='nearest')
                raise NotImplementedError('')
                    
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, cache_enabled=True):
                captions_batch = [
                    target.get('captions') or target.get('class_texts') 
                    for target in targets
                ]
                caption_padding_mask = [
                    torch.ones(len(texts), dtype=torch.bool, device=device) 
                    for texts in captions_batch
                ]
                if targets[0].get('caption_attributes', False):
                    caption_attributes_batch = [
                        target.get('caption_attributes') 
                        for target in targets
                    ]
                    caption_attributes_padding_mask = [
                        torch.ones(len(attributes), dtype=torch.bool, device=device) 
                        for attributes in caption_attributes_batch
                    ]
                    cap_to_attr_map_batch = torch.stack([
                        target['cap_to_attr_map'] 
                        for target in targets
                    ])
                else:
                    caption_attributes_batch = None
                    caption_attributes_padding_mask = None
                    cap_to_attr_map_batch = None
                
                if targets:
                    for idx, target in enumerate(targets):
                        if target and 'caption_padding_mask' in target:
                            caption_padding_mask[idx] = target['caption_padding_mask']
                        if target and 'caption_attributes_padding_mask' in target:
                            caption_attributes_padding_mask[idx] = target['caption_attributes_padding_mask']
                    
                if any(captions_batch):
                    outputs = model(samples, captions_batch=captions_batch, caption_attributes_batch=caption_attributes_batch, targets=targets,
                                    caption_padding_mask=caption_padding_mask, caption_attributes_padding_mask=caption_attributes_padding_mask,
                                    cap_to_attr_map_batch=cap_to_attr_map_batch)
                else:
                    outputs = model(samples, captions_batch=None, caption_attributes_batch=None, targets=targets,
                                    caption_padding_mask=caption_padding_mask, caption_attributes_padding_mask=caption_attributes_padding_mask,
                                    cap_to_attr_map_batch=cap_to_attr_map_batch)

            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                print(outputs['pred_boxes'])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    # Replace 'module' with 'model' in each key
                    new_key = key.replace('module.', '')
                    # Add the updated key-value pair to the state dictionary
                    state[new_key] = value
                new_state['model'] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            captions_batch = [
                target.get('captions') or target.get('class_texts') 
                for target in targets
            ]
            caption_padding_mask = [
                torch.ones(len(texts), dtype=torch.bool, device=device) 
                for texts in captions_batch
            ]
            if targets[0].get('caption_attributes', False):
                caption_attributes_batch = [
                    target.get('caption_attributes') 
                    for target in targets
                ]
                caption_attributes_padding_mask = [
                    torch.ones(len(attributes), dtype=torch.bool, device=device) 
                    for attributes in caption_attributes_batch
                ]
                cap_to_attr_map_batch = torch.stack([
                    target['cap_to_attr_map'] 
                    for target in targets
                ])
            else:
                caption_attributes_batch = None
                caption_attributes_padding_mask = None
                cap_to_attr_map_batch = None
            
            if targets:
                for idx, target in enumerate(targets):
                    if target and 'caption_padding_mask' in target:
                        caption_padding_mask[idx] = target['caption_padding_mask']
                    if target and 'caption_attributes_padding_mask' in target:
                        caption_attributes_padding_mask[idx] = target['caption_attributes_padding_mask']
                
            if any(captions_batch):
                outputs = model(samples, captions_batch=captions_batch, caption_attributes_batch=caption_attributes_batch, targets=targets,
                                caption_padding_mask=caption_padding_mask, caption_attributes_padding_mask=caption_attributes_padding_mask,
                                cap_to_attr_map_batch=cap_to_attr_map_batch)
            else:
                outputs = model(samples, captions_batch=None, caption_attributes_batch=None, targets=targets,
                                caption_padding_mask=caption_padding_mask, caption_attributes_padding_mask=caption_attributes_padding_mask,
                                cap_to_attr_map_batch=cap_to_attr_map_batch)

            loss_dict = criterion(outputs, targets, **metas)

            loss : torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        for t in targets:
            for key, value in t.items():
                if isinstance(value, str) or isinstance(value, list) or isinstance(value, dict) or isinstance(value, bool) or isinstance(value, int):
                    t[key] = value
                elif value==None:
                    t[key] = value
                else:
                    t[key] = value.to(device)
        
        captions_batch = [
            target.get('captions') or target.get('class_texts') 
            for target in targets
        ]
        caption_padding_mask = [
            torch.ones(len(texts), dtype=torch.bool) 
            for texts in captions_batch
        ]
        if targets[0].get('caption_attributes', False):
            caption_attributes_batch = [
                target.get('caption_attributes') 
                for target in targets
            ]
            caption_attributes_padding_mask = [
                torch.ones(len(attributes), dtype=torch.bool) 
                for attributes in caption_attributes_batch
            ]
            cap_to_attr_map_batch = torch.stack([
                target['cap_to_attr_map'] 
                for target in targets
            ])
        else:
            caption_attributes_batch = None
            caption_attributes_padding_mask = None
            cap_to_attr_map_batch = None
        
        if targets:
            for idx, target in enumerate(targets):
                if target and 'caption_padding_mask' in target:
                    caption_padding_mask[idx] = target['caption_padding_mask']
                if target and 'caption_attributes_padding_mask' in target:
                    caption_attributes_padding_mask[idx] = target['caption_attributes_padding_mask']
            
        if any(captions_batch):
            outputs = model(samples, captions_batch=captions_batch, caption_attributes_batch=caption_attributes_batch, targets=targets,
                            caption_padding_mask=caption_padding_mask, caption_attributes_padding_mask=caption_attributes_padding_mask,
                            cap_to_attr_map_batch=cap_to_attr_map_batch)
        else:
            outputs = model(samples, captions_batch=None, caption_attributes_batch=None, targets=targets,
                            caption_padding_mask=caption_padding_mask, caption_attributes_padding_mask=caption_attributes_padding_mask,
                            cap_to_attr_map_batch=cap_to_attr_map_batch)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessor(outputs, orig_target_sizes, cap_to_attr_map_batch)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    if coco_evaluator is not None:
        # COCO-style evaluator
        if hasattr(coco_evaluator, "coco_eval"):
            if 'bbox' in iou_types:
                bbox_eval = coco_evaluator.coco_eval.get('bbox')
                if bbox_eval is not None:
                    if hasattr(bbox_eval, "stats"):
                        stats_val = bbox_eval.stats
                        if hasattr(stats_val, "tolist"):
                            stats_list = stats_val.tolist()
                            if stats_list:  # Only add if not empty
                                stats['coco_eval_bbox'] = stats_list
                        elif isinstance(stats_val, (list, tuple)):
                            stats_list = list(stats_val)
                            if stats_list:  # Only add if not empty
                                stats['coco_eval_bbox'] = stats_list
                    elif isinstance(bbox_eval, (list, tuple)):
                        stats_list = list(bbox_eval)
                        if stats_list:  # Only add if not empty
                            stats['coco_eval_bbox'] = stats_list
            if 'segm' in iou_types and 'segm' in coco_evaluator.coco_eval:
                segm_eval = coco_evaluator.coco_eval.get('segm')
                if segm_eval is not None:
                    if hasattr(segm_eval, "stats"):
                        stats_val = segm_eval.stats
                        if hasattr(stats_val, "tolist"):
                            stats_list = stats_val.tolist()
                            if stats_list:  # Only add if not empty
                                stats['coco_eval_masks'] = stats_list
                        elif isinstance(stats_val, (list, tuple)):
                            stats_list = list(stats_val)
                            if stats_list:  # Only add if not empty
                                stats['coco_eval_masks'] = stats_list
                    elif isinstance(segm_eval, (list, tuple)):
                        stats_list = list(segm_eval)
                        if stats_list:  # Only add if not empty
                            stats['coco_eval_masks'] = stats_list
        # Grounding-only evaluator: pull computed metrics directly
        if hasattr(coco_evaluator, "get_results"):
            results = coco_evaluator.get_results()
            stats.update(results)

    return stats, coco_evaluator
