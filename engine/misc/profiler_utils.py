"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy

from calflops import calculate_flops
import torch
import torch.nn as nn


class _FlopsWrapper(nn.Module):
    def __init__(self, model: nn.Module, use_text: bool, text_arg_name: str = "captions_batch"):
        super().__init__()
        self.model = model
        self.use_text = use_text
        self.text_arg_name = text_arg_name

    def forward(self, images: torch.Tensor):
        if not self.use_text:
            return self.model(images)

        batch_size = images.shape[0]
        captions_batch = [["text"] for _ in range(batch_size)]
        return self.model(images, **{self.text_arg_name: captions_batch})


def stats(cfg, input_shape: tuple = (1, 3, 640, 640)):
    base_size = cfg.train_dataloader.collate_fn.base_size
    input_shape = (1, 3, base_size, base_size)

    use_otadet_contrastive_head = cfg.yaml_cfg["DEIMTransformer"].get(
        "use_otadet_contrastive_head", False
    )

    model_for_info = copy.deepcopy(cfg.model).deploy()
    model_for_info.eval()
    for param in model_for_info.parameters():
        param.requires_grad_(False)

    wrapped = _FlopsWrapper(
        model_for_info,
        use_text=bool(use_otadet_contrastive_head),
        text_arg_name="captions_batch",
    )

    flops, macs, _ = calculate_flops(
        model=wrapped,
        input_shape=input_shape,
        output_as_string=True,
        output_precision=4,
        print_detailed=False,
    )

    params = sum(param.numel() for param in model_for_info.parameters())
    del model_for_info, wrapped

    return params, {"Model FLOPs:%s   MACs:%s   Params:%s" % (flops, macs, params)}
