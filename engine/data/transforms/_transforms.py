"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

import PIL
import PIL.Image

# from typing import Any, Dict, List, Optional
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union
from .._misc import convert_to_tv_tensor, _boxes_keys
from .._misc import Image, Video, Mask, BoundingBoxes
from .._misc import SanitizeBoundingBoxes

from torchvision.transforms.v2._utils import _parse_labels_getter, _setup_number_or_seq, _setup_size, get_bounding_boxes, has_any, is_pure_tensor
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision import transforms as _transforms, tv_tensors

from ...core import register
torchvision.disable_beta_transforms_warning()


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes = register(name='SanitizeBoundingBoxes')(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)

@register()
class SanitizeBoundingBoxes_Grounding(T.Transform):
    """Remove degenerate/invalid bounding boxes and their corresponding labels and masks.

    This transform removes bounding boxes and their associated labels/masks that:

    - are below a given ``min_size`` or ``min_area``: by default this also removes degenerate boxes that have e.g. X2 <= X1.
    - have any coordinate outside of their corresponding image. You may want to
      call :class:`~torchvision.transforms.v2.ClampBoundingBoxes` first to avoid undesired removals.

    It can also sanitize other tensors like the "iscrowd" or "area" properties from COCO
    (see ``labels_getter`` parameter).

    It is recommended to call it at the end of a pipeline, before passing the
    input to the models. It is critical to call this transform if
    :class:`~torchvision.transforms.v2.RandomIoUCrop` was called.
    If you want to be extra careful, you may call it after all transforms that
    may modify bounding boxes but once at the end should be enough in most
    cases.

    Args:
        min_size (float, optional): The size below which bounding boxes are removed. Default is 1.
        min_area (float, optional): The area below which bounding boxes are removed. Default is 1.
        labels_getter (callable or str or None, optional): indicates how to identify the labels in the input
            (or anything else that needs to be sanitized along with the bounding boxes).
            By default, this will try to find a "labels" key in the input (case-insensitive), if
            the input is a dict or it is a tuple whose second element is a dict.
            This heuristic should work well with a lot of datasets, including the built-in torchvision datasets.

            It can also be a callable that takes the same input as the transform, and returns either:

            - A single tensor (the labels)
            - A tuple/list of tensors, each of which will be subject to the same sanitization as the bounding boxes.
              This is useful to sanitize multiple tensors like the labels, and the "iscrowd" or "area" properties
              from COCO.

            If ``labels_getter`` is None then only bounding boxes are sanitized.
    """

    def __init__(
        self,
        min_size: float = 1.0,
        min_area: float = 1.0,
        labels_getter: Union[Callable[[Any], Any], str, None] = "default",
    ) -> None:
        super().__init__()

        if min_size < 1:
            raise ValueError(f"min_size must be >= 1, got {min_size}.")
        self.min_size = min_size

        if min_area < 1:
            raise ValueError(f"min_area must be >= 1, got {min_area}.")
        self.min_area = min_area

        self.labels_getter = labels_getter
        self._labels_getter = _parse_labels_getter(labels_getter)

    # def forward(self, *inputs: Any) -> Any:
    #     inputs = inputs if len(inputs) > 1 else inputs[0]

    #     labels = self._labels_getter(inputs)
    #     if labels is not None:
    #         msg = "The labels in the input to forward() must be a tensor or None, got {type} instead."
    #         if isinstance(labels, torch.Tensor):
    #             labels = (labels,)
    #         elif isinstance(labels, (tuple, list)):
    #             for entry in labels:
    #                 if not isinstance(entry, torch.Tensor):
    #                     # TODO: we don't need to enforce tensors, just that entries are indexable as t[bool_mask]
    #                     raise ValueError(msg.format(type=type(entry)))
    #         else:
    #             raise ValueError(msg.format(type=type(labels)))

    #     flat_inputs, spec = tree_flatten(inputs)
    #     boxes = get_bounding_boxes(flat_inputs)

    #     if labels is not None:
    #         for label in labels:
    #             if boxes.shape[0] != label.shape[0]:
    #                 raise ValueError(
    #                     f"Number of boxes (shape={boxes.shape}) and must match the number of labels."
    #                     f"Found labels with shape={label.shape})."
    #                 )

    #     valid = F._misc._get_sanitize_bounding_boxes_mask(
    #         boxes,
    #         format=boxes.format,
    #         canvas_size=boxes.canvas_size,
    #         min_size=self.min_size,
    #         min_area=self.min_area,
    #     )

    #     # ==== Debug 打印：有 boxes 被过滤就打印一条 ====
    #     num_before = int(boxes.shape[0])
    #     num_after = int(valid.sum().item())
    #     if num_after < num_before:
    #         removed = num_before - num_after
    #         # 如需看具体被移除的索引，解开下一行
    #         # removed_idx = torch.nonzero(~valid, as_tuple=False).squeeze(1).tolist()
    #         print(f"[SanitizeBoundingBoxes] removed {removed}/{num_before} boxes "
    #               f"(kept={num_after}, min_size={self.min_size}, min_area={self.min_area})")
    #               # + f", removed_idx={removed_idx}")

    #     params = dict(valid=valid, labels=labels)
    #     flat_outputs = [self._transform(inpt, params) for inpt in flat_inputs]
    #     outputs = tree_unflatten(flat_outputs, spec)
    #     outputs = self._apply_caption_filter(outputs, valid)
    #     return tree_unflatten(flat_outputs, spec)
    def forward(self, *inputs: Any) -> Any:
        inputs = inputs if len(inputs) > 1 else inputs[0]
        if len(inputs[1]['boxes']) == 0:
            return inputs 
        
        labels = self._labels_getter(inputs)
        if labels is not None:
            msg = "The labels in the input to forward() must be a tensor or None, got {type} instead."
            if isinstance(labels, torch.Tensor):
                labels = (labels,)
            elif isinstance(labels, (tuple, list)):
                for entry in labels:
                    if not isinstance(entry, torch.Tensor):
                        # TODO: we don't need to enforce tensors, just that entries are indexable as t[bool_mask]
                        raise ValueError(msg.format(type=type(entry)))
            else:
                raise ValueError(msg.format(type=type(labels)))
            
        flat_inputs, spec = tree_flatten(inputs)
        boxes = get_bounding_boxes(flat_inputs)

        if labels is not None:
            for label in labels:
                if boxes.shape[0] != label.shape[0]:
                    raise ValueError(
                        f"Number of boxes (shape={boxes.shape}) and must match the number of labels."
                        f"Found labels with shape={label.shape})."
                    )

        valid = F._misc._get_sanitize_bounding_boxes_mask(
            boxes,
            format=boxes.format,
            canvas_size=boxes.canvas_size,
            min_size=self.min_size,
            min_area=self.min_area,
        )

        # ==== Debug 打印：有 boxes 被过滤就打印一条 ====
        # num_before = int(boxes.shape[0])
        # num_after = int(valid.sum().item())
        # if num_after < num_before:
        #     removed = num_before - num_after
        #     # 如需看具体被移除的索引，解开下一行
        #     # removed_idx = torch.nonzero(~valid, as_tuple=False).squeeze(1).tolist()
        #     print(f"[SanitizeBoundingBoxes] removed {removed}/{num_before} boxes "
        #         f"(kept={num_after}, min_size={self.min_size}, min_area={self.min_area})")
        #     print(flat_inputs)

        params = dict(valid=valid, labels=labels)
        flat_outputs = [self._transform(inpt, params) for inpt in flat_inputs]
        
        # 修改这里：先 unflatten，再应用 caption filter
        outputs = tree_unflatten(flat_outputs, spec)
        
        if self._has_caption_indices(outputs):
            outputs = self._apply_caption_filter(outputs, valid)
    
        # if num_after < num_before:
        #     print('after filter', outputs)
            
        return outputs 

    def _has_caption_indices(self, outputs) -> bool:
        """检查是否包含 caption_indices_list"""
        if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
            target = outputs[1]
            return isinstance(target, dict) and "caption_indices_list" in target
        return False

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        is_label = params["labels"] is not None and any(inpt is label for label in params["labels"])
        is_bounding_boxes_or_mask = isinstance(inpt, (tv_tensors.BoundingBoxes, tv_tensors.Mask))

        if not (is_label or is_bounding_boxes_or_mask):
            return inpt

        output = inpt[params["valid"]]

        if is_label:
            return output
        else:
            return tv_tensors.wrap(output, like=inpt)

    def _apply_caption_filter(self, outputs, valid: torch.Tensor):
        """过滤掉被移除box对应的caption_indices_list条目"""
        # outputs 通常是 (image, target) 或者单个 dict
        if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
            # (image, target) 结构
            target = outputs[1]
            if isinstance(target, dict) and "caption_indices_list" in target:
                valid_list = valid.tolist()
                cil = target["caption_indices_list"]
                assert len(cil) == len(valid_list), \
                    f"caption_indices_list length ({len(cil)}) != boxes length ({len(valid_list)})"
                
                # 浅拷贝 target，只保留 valid 的条目
                target = dict(target)
                target["caption_indices_list"] = [ci for ci, keep in zip(cil, valid_list) if keep]

                # 同步过滤 area / iscrowd（若存在）
                if "area" in target and isinstance(target["area"], torch.Tensor):
                    target["area"] = target["area"][valid]
                if "iscrowd" in target and isinstance(target["iscrowd"], torch.Tensor):
                    target["iscrowd"] = target["iscrowd"][valid]
                
                # 返回更新后的结构
                outputs = list(outputs)
                outputs[1] = target
                return tuple(outputs)
        
        return outputs


@register()
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode='constant') -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (
        BoundingBoxes,
    )
    def __init__(self, fmt='', normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (
        PIL.Image.Image,
    )
    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == 'float32':
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.

        inpt = Image(inpt)

        return inpt
