"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# from ._dataset import DetDataset
from .coco_dataset import CocoDetection
from .coco_dataset_w_text import CocoDetection_W_Text
from .coco_dataset import (
    mscoco_category2name,
    mscoco_category2label,
    mscoco_label2category,
)
from .coco_eval import CocoEvaluator
from .grounding_evaluator import GroundingEvaluator
from .coco_utils import get_coco_api_from_dataset
from .voc_detection import VOCDetection
from .voc_eval import VOCEvaluator
from .coco_dataset_w_caption import CocoDetection_W_Caption
from .otadet_grounding_dataset import HF_VGDataset
from .otadet_detection_dataset import HF_DetDataset
from .concat_dataset import ConcatDataset
