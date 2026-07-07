"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

import time
import json
import datetime
import numpy as np

import torch

from torch.utils.data import Subset

from ..misc import dist_utils, stats
from ..data import DataLoader, ConcatDataset, get_coco_api_from_dataset

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from ..optim.lr_scheduler import FlatCosineLRScheduler


class DetSolver(BaseSolver):

    def fit(self, ):
        self.train()
        args = self.cfg

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-"*42 + "Start training" + "-"*43)

        for i, (name, param) in enumerate(self.model.named_parameters()):
            if i in [194, 195]:
                print(f"Index {i}: {name} - requires_grad: {param.requires_grad}")

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, args.lr_gamma, iter_per_epoch, total_epochs=args.epoches, 
                                                warmup_iter=args.warmup_iter, flat_epochs=args.flat_epoch, no_aug_epochs=args.no_aug_epoch)
            self.self_lr_scheduler = True
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        n_parameters = sum([p.numel() for p in self.model.parameters() if not p.requires_grad])
        print(f'number of non-trainable parameters: {n_parameters}')

        best_stat = {'epoch': -1, }
        best_primary_by_dataset = {}
        best_primary_avg = float('-inf')

        dataset_names = getattr(self.val_dataloader.dataset, 'dataset_names', None) if isinstance(self.val_dataloader.dataset, ConcatDataset) else None

        def _normalize_metric_values(metric_value):
            """Convert metrics to a list of python floats so scalars won't break logging."""
            if isinstance(metric_value, torch.Tensor):
                metric_value = metric_value.detach().cpu()
            if isinstance(metric_value, np.ndarray):
                metric_value = metric_value.tolist()
            if isinstance(metric_value, (list, tuple)):
                return [float(v) for v in metric_value]
            
            return [float(metric_value)]

        def _split_dataset_and_metric(full_key: str):
            """Extract dataset label and metric name from the logged key."""
            if "::" in full_key:
                ds_label, metric_key = full_key.split("::", 1)
                return ds_label, metric_key
            if full_key.startswith("subdataset_"):
                parts = full_key.split("_", 2)
                if len(parts) >= 3:
                    ds_idx = parts[1]
                    ds_label = f"subdataset_{ds_idx}"
                    if dataset_names and ds_idx.isdigit() and int(ds_idx) < len(dataset_names):
                        ds_label = dataset_names[int(ds_idx)]
                    return ds_label, parts[2]
            return "default", full_key

        def _select_primary_metric(metric_key: str, values):
            """
            Pick the key metric used for model selection.
            - Detection: AP@0.50 (index 1 of coco_eval stats)
            - Grounding: Pr@0.5
            """
            name = metric_key.lower()
            # Prefer explicit AP50 metric if present (produced by coco_eval.py summary).
            if name == "ap50":
                ap50 = values[0] if values else 0.0
                return 0, float(ap50)
            if "coco_eval_bbox" in name or "coco_eval_masks" in name:
                # coco_eval.stats is a list where index 1 corresponds to AP50.
                if isinstance(values, (list, tuple)):
                    if len(values) > 1:
                        ap50 = values[1]
                    elif len(values) == 1:
                        ap50 = values[0]
                    else:
                        ap50 = 0.0
                else:
                    ap50 = values if values is not None else 0.0
                return 0, float(ap50)
            if "pr@0.5" in name:
                # Prefer the default score field, then attribute/averaged variants.
                if "[scores_attr]" in name:
                    priority = 2
                elif "[scores_avg]" in name:
                    priority = 3
                elif "[scores]" in name:
                    priority = 1
                else:
                    priority = 4
                return priority, values[0]
            if name.startswith("global/instance_f1_score"):
                return 0, values[0]
            return None

        def _compute_primary_scores(stats_dict):
            """Collect one primary score per dataset for averaging."""
            primary = {}
            for full_key, values in stats_dict.items():
                dataset_key, metric_key = _split_dataset_and_metric(full_key)
                selected = _select_primary_metric(metric_key, values)
                if selected is None:
                    continue
                priority, primary_val = selected
                if dataset_key not in primary or priority < primary[dataset_key][0]:
                    primary[dataset_key] = (priority, primary_val)
            return {k: v[1] for k, v in primary.items()}

        # evaluate again before resume training
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model

            # Check if evaluator is a list (for ConcatDataset)
            if isinstance(self.evaluator, list):
                coco_list, sub_lens = get_coco_api_from_dataset(self.val_dataloader.dataset)
                start_idx = 0
                test_stats = {}
                dataset_names = getattr(self.val_dataloader.dataset, 'dataset_names', None)

                for ds_idx, (ds_coco, ds_len) in enumerate(zip(coco_list, sub_lens)):
                    ds_name = dataset_names[ds_idx] if dataset_names and ds_idx < len(dataset_names) else f'subdataset_{ds_idx}'
                    print(f'\n===== Evaluating dataset (resume): {ds_name} =====')

                    end_idx = start_idx + ds_len
                    sub_dataset = Subset(self.val_dataloader.dataset, range(start_idx, end_idx))
                    sub_data_loader = DataLoader(
                        sub_dataset,
                        batch_size=self.val_dataloader.batch_size,
                        shuffle=False,
                        num_workers=self.val_dataloader.num_workers,
                        drop_last=False,
                        collate_fn=self.val_dataloader.collate_fn
                    )
                    sub_data_loader = dist_utils.warp_loader(sub_data_loader, shuffle=False)

                    evaluator = self.evaluator[ds_idx]
                    if hasattr(evaluator, 'dataset_name'):
                        evaluator.dataset_name = ds_name
                    else:
                        setattr(evaluator, 'dataset_name', ds_name)

                    sub_test_stats, sub_coco_eval = evaluate(
                        module,
                        self.criterion,
                        self.postprocessor,
                        sub_data_loader,
                        evaluator,
                        self.device
                    )

                    for k, v in sub_test_stats.items():
                        test_stats[f"{ds_name}::{k}"] = v
                    start_idx = end_idx
            else:
                test_stats, coco_evaluator = evaluate(
                    module,
                    self.criterion,
                    self.postprocessor,
                    self.val_dataloader,
                    self.evaluator,
                    self.device
                )

            normalized_stats = {k: _normalize_metric_values(v) for k, v in test_stats.items()}
            primary_scores = _compute_primary_scores(normalized_stats)
            if primary_scores:
                best_primary_avg = float(np.mean(list(primary_scores.values())))
                best_primary_by_dataset = primary_scores
                best_stat['epoch'] = self.last_epoch
            for k, values in normalized_stats.items():
                if values:  # Only update if not empty
                    best_stat[k] = values[0]
            if primary_scores:
                print(f'Initial best (epoch {self.last_epoch}) primary metrics: {primary_scores}, avg={best_primary_avg:.4f}')

        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            train_stats = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer, 
                self.device, 
                epoch, 
                max_norm=args.clip_max_norm, 
                print_freq=args.print_freq, 
                ema=self.ema, 
                scaler=self.scaler, 
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer
            )

            if not self.self_lr_scheduler:  # update by epoch 
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            if isinstance(self.val_dataloader.dataset, ConcatDataset):
                coco_list, sub_lens = get_coco_api_from_dataset(self.val_dataloader.dataset)
                start_idx = 0
                test_stats = {}
                coco_eval_results = []
                dataset_names = getattr(self.val_dataloader.dataset, 'dataset_names', None)

                for ds_idx, (ds_coco, ds_len) in enumerate(zip(coco_list, sub_lens)):
                    ds_name = dataset_names[ds_idx] if dataset_names and ds_idx < len(dataset_names) else f'subdataset_{ds_idx}'
                    print(f'\n===== Evaluating dataset: {ds_name} =====')

                    end_idx = start_idx + ds_len
                    sub_dataset = Subset(self.val_dataloader.dataset, range(start_idx, end_idx))
                    sub_data_loader = DataLoader(
                        sub_dataset,
                        batch_size=self.val_dataloader.batch_size,
                        shuffle=False,
                        num_workers=self.val_dataloader.num_workers,
                        drop_last=False,
                        collate_fn=self.val_dataloader.collate_fn
                    )
                    sub_data_loader = dist_utils.warp_loader(sub_data_loader, shuffle=False)

                    evaluator = self.evaluator[ds_idx] if isinstance(self.evaluator, list) else self.evaluator
                    if hasattr(evaluator, 'dataset_name'):
                        evaluator.dataset_name = ds_name
                    else:
                        setattr(evaluator, 'dataset_name', ds_name)
                    sub_test_stats, sub_coco_eval = evaluate(
                        module,
                        self.criterion,
                        self.postprocessor,
                        sub_data_loader,
                        evaluator,
                        self.device
                    )

                    for k, v in sub_test_stats.items():
                        test_stats[f"{ds_name}::{k}"] = v
                    coco_eval_results.append((ds_idx, sub_coco_eval))
                    start_idx = end_idx
            else:
                test_stats, coco_evaluator = evaluate(
                    module,
                    self.criterion,
                    self.postprocessor,
                    self.val_dataloader,
                    self.evaluator,
                    self.device
                )
                coco_eval_results = [(None, coco_evaluator)]

            normalized_test_stats = {k: _normalize_metric_values(v) for k, v in test_stats.items()}
            if self.writer and dist_utils.is_main_process():
                for k, values in normalized_test_stats.items():
                    for i, v in enumerate(values):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

            for k, values in normalized_test_stats.items():
                current_val = values[0] if values else 0.0
                prev_best = best_stat.get(k, float('-inf'))
                if current_val > prev_best:
                    best_stat[k] = current_val

            primary_scores = _compute_primary_scores(normalized_test_stats)
            current_primary_avg = float(np.mean(list(primary_scores.values()))) if primary_scores else None
            is_new_best = current_primary_avg is not None and current_primary_avg > best_primary_avg

            if is_new_best:
                best_primary_avg = current_primary_avg
                best_primary_by_dataset = primary_scores
                best_stat['epoch'] = epoch
                if self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                    else:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')
            elif self.output_dir and epoch >= self.train_dataloader.collate_fn.stop_epoch:
                best_stage1_path = self.output_dir / 'best_stg1.pth'
                if best_stage1_path.exists():
                    self.ema.decay -= 0.0001
                    self.load_resume_state(str(best_stage1_path))
                    print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            if primary_scores:
                print(f'Primary metrics this epoch: {primary_scores} | avg={current_primary_avg:.4f}')
            else:
                print('Primary metrics this epoch: None')
            if best_primary_by_dataset:
                print(f"Best primary metrics (epoch {best_stat.get('epoch', -1)}): {best_primary_by_dataset} | avg={best_primary_avg:.4f}")

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in normalized_test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                (self.output_dir / 'eval').mkdir(exist_ok=True)
                for ds_idx, coco_eval in coco_eval_results:
                    if coco_eval is not None and hasattr(coco_eval, "coco_eval") and "bbox" in coco_eval.coco_eval:
                        ds_name = None
                        if isinstance(self.val_dataloader.dataset, ConcatDataset):
                            names = getattr(self.val_dataloader.dataset, 'dataset_names', None)
                            if names and ds_idx is not None and ds_idx < len(names):
                                ds_name = names[ds_idx]
                        prefix_label = ds_name if ds_name is not None else (f"subdataset_{ds_idx}" if ds_idx is not None else "")
                        prefix = f"{prefix_label}_" if prefix_label else ""
                        filenames = [f'{prefix}latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{prefix}{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_eval.coco_eval["bbox"].eval,
                                       self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model

        if isinstance(self.val_dataloader.dataset, ConcatDataset):
            coco_list, sub_lens = get_coco_api_from_dataset(self.val_dataloader.dataset)
            start_idx = 0
            dataset_names = getattr(self.val_dataloader.dataset, 'dataset_names', None)

            for ds_idx, (ds_coco, ds_len) in enumerate(zip(coco_list, sub_lens)):
                ds_name = dataset_names[ds_idx] if dataset_names and ds_idx < len(dataset_names) else f'subdataset_{ds_idx}'
                print(f'\n===== Evaluating dataset: {ds_name} =====')

                end_idx = start_idx + ds_len
                sub_dataset = Subset(self.val_dataloader.dataset, range(start_idx, end_idx))
                sub_data_loader = DataLoader(
                    sub_dataset,
                    batch_size=self.val_dataloader.batch_size,
                    shuffle=False,
                    num_workers=self.val_dataloader.num_workers,
                    drop_last=False,
                    collate_fn=self.val_dataloader.collate_fn
                )
                sub_data_loader = dist_utils.warp_loader(sub_data_loader, shuffle=False)

                evaluator = self.evaluator[ds_idx] if isinstance(self.evaluator, list) else self.evaluator
                if hasattr(evaluator, 'dataset_name'):
                    evaluator.dataset_name = ds_name
                else:
                    setattr(evaluator, 'dataset_name', ds_name)
                test_stats, coco_evaluator = evaluate(
                    module,
                    self.criterion,
                    self.postprocessor,
                    sub_data_loader,
                    evaluator,
                    self.device
                )

                if self.output_dir:
                    if coco_evaluator is not None and hasattr(coco_evaluator, "coco_eval") and "bbox" in coco_evaluator.coco_eval:
                        dist_utils.save_on_master(
                            coco_evaluator.coco_eval["bbox"].eval,
                            self.output_dir / f"eval_{ds_name}.pth"
                        )

                start_idx = end_idx
        else:
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )

            if self.output_dir:
                if coco_evaluator is not None and hasattr(coco_evaluator, "coco_eval") and "bbox" in coco_evaluator.coco_eval:
                    dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return