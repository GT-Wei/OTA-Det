"""
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""
import torch
import torch.nn as nn
from ..core import register


__all__ = ['DEIM', ]


@register()
class DEIM(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
    # cap_to_attr_map_batch: 每条caption对应的属性在caption_attributes_batch中的位置
    def forward(self, x, captions_batch=None, caption_attributes_batch=None, targets=None, caption_padding_mask=None, caption_attributes_padding_mask=None, cap_to_attr_map_batch=None):
        if captions_batch:
            x, text_feat, attributes_feats = self.backbone(x, captions_batch, normalize=True, caption_padding_mask=caption_padding_mask,
                                         caption_attributes_batch=caption_attributes_batch, caption_attributes_padding_mask=caption_attributes_padding_mask,
                                         cap_to_attr_map_batch=cap_to_attr_map_batch)
        else:
            x = self.backbone(x)
        x = self.encoder(x)
        
        if captions_batch:
            x = self.decoder(x, targets, 
                             text_features=text_feat, 
                             caption_padding_mask=caption_padding_mask,
                             attributes_feats=attributes_feats,
                             caption_attributes_padding_mask=caption_attributes_padding_mask,
                             cap_to_attr_map_batch=cap_to_attr_map_batch)
        else:
            x = self.decoder(x, targets)
        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
