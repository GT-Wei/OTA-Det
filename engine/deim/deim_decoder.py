"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import math
import copy
import functools
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ..core import register
from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func_v2, get_activation, inverse_sigmoid, bias_init_with_prob

from .dfine_decoder import MSDeformableAttention, LQE, Integral
from .dfine_utils import weighting_function, distance2bbox
from .deim_utils import RMSNorm, SwiGLUFFN, Gate, MLP
from einops import einsum

__all__ = ['DEIMTransformer']


class OTADetContrastiveHead(nn.Module):
    def __init__(self,
                 image_channels: int,
                 hidden_embed_dim: int,
                 init_logit_scale: float = np.log(1 / 0.07),
                 init_logit_bias: Optional[float] = None,
                 nonscalar_logit_scale: bool = False) -> None:
        super(OTADetContrastiveHead, self).__init__()
        
        self.image_channels = image_channels
        self.hidden_embed_dim = hidden_embed_dim
        
        # self.bias = nn.Parameter(torch.zeros([]))
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else:
            self.logit_bias = None
            
        self.img_proj = nn.Linear(image_channels, hidden_embed_dim)

    def project_image(self, image_features: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize image/query features for contrastive use."""
        return F.normalize(self.img_proj(image_features), dim=-1)

    def normalize_text(self, text_features: torch.Tensor) -> torch.Tensor:
        """L2-normalize text features to keep alignment consistent."""
        return F.normalize(text_features, dim=-1)
        
    def forward(self, image_features, text_features, caption_padding_mask=None):
        image_features = self.project_image(image_features)
        text_features = self.normalize_text(text_features)
        
        logits_similarity = self.logit_scale.exp() * einsum(image_features, text_features, 'b n d, b m d -> b n m')
        if self.logit_bias is not None:
            logits_similarity += self.logit_bias
        
        if caption_padding_mask is not None:
            # Convert a list mask to a tensor.
            if isinstance(caption_padding_mask, list):
                # Each list item is expected to have the same length.
                caption_padding_mask = torch.stack(caption_padding_mask, dim=0)
            
            # Ensure boolean dtype on the logits device.
            col_mask = caption_padding_mask.to(
                dtype=torch.bool, 
                device=logits_similarity.device
            ).unsqueeze(1)  # [B, 1, M]
            
            # Mask invalid positions with -inf.
            logits_similarity = logits_similarity.masked_fill(~col_mask, float('-inf'))

        return logits_similarity


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation='relu',
                 n_levels=4,
                 n_points=4,
                 cross_attn_method='default',
                 layer_scale=None,
                 use_gateway=False,
                 ):
        super(TransformerDecoderLayer, self).__init__()

        if layer_scale is not None:
            print(f"     --- Wide Layer@{layer_scale} ---")
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = RMSNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)

        self.use_gateway = use_gateway
        if use_gateway:
            self.gateway = Gate(d_model, use_rmsnorm=True)
        else:
            self.norm2 = RMSNorm(d_model)

        # ffn
        self.swish_ffn = SwiGLUFFN(d_model, dim_feedforward // 2, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = RMSNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self,
                target,
                reference_points,
                value,
                spatial_shapes,
                attn_mask=None,
                query_pos_embed=None):

        # self attention
        q = k = self.with_pos_embed(target, query_pos_embed)

        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # cross attention
        target2 = self.cross_attn(\
            self.with_pos_embed(target, query_pos_embed),
            reference_points,
            value,
            spatial_shapes)

        if self.use_gateway:
            target = self.gateway(target, self.dropout2(target2))
        else:
            target = target + self.dropout2(target2)
            target = self.norm2(target)

        # ffn
        target2 = self.swish_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target.clamp(min=-65504, max=65504))

        return target


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder implementing Fine-grained Distribution Refinement (FDR).

    This decoder refines object detection predictions through iterative updates across multiple layers,
    utilizing attention mechanisms, location quality estimators, and distribution refinement techniques
    to improve bounding box accuracy and robustness.
    """

    def __init__(self, hidden_dim, decoder_layer, decoder_layer_wide, num_layers, num_head, reg_max, reg_scale, up,
                 eval_idx=-1, layer_scale=2, act='relu', use_otadet_contrastive_head=False):
        super(TransformerDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)] \
                    + [copy.deepcopy(decoder_layer_wide) for _ in range(num_layers - self.eval_idx - 1)])
        self.lqe_layers = nn.ModuleList([copy.deepcopy(LQE(4, 64, 2, reg_max, act=act)) for _ in range(num_layers)])
        self.use_otadet_contrastive_head = use_otadet_contrastive_head

    def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):
        """
        Preprocess values for MSDeformableAttention.
        """
        value = value_proj(memory) if value_proj is not None else memory
        value = F.interpolate(memory, size=value_scale) if value_scale is not None else value
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def convert_to_deploy(self):
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        self.layers = self.layers[:self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.lqe_layers[self.eval_idx]])

    def forward(self,
                target,
                ref_points_unact,
                memory,
                spatial_shapes,
                bbox_head,
                score_head,
                attr_score_head,
                query_pos_head,
                pre_bbox_head,
                integral,
                up,
                reg_scale,
                attn_mask=None,
                memory_mask=None,
                dn_meta=None,
                text_features=None,
                caption_padding_mask=None,
                attributes_feats=None, 
                caption_attributes_padding_mask=None
                ):
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_proj = [] if self.use_otadet_contrastive_head else None
        dec_out_attr_proj = [] if self.use_otadet_contrastive_head else None
        dec_out_attr_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []
        dec_pre_out_proj = None
        dec_pre_attr_out_proj = None
        pre_attr_scores = None
        attr_head = attr_score_head if attr_score_head is not None else score_head
        if not hasattr(self, 'project'):
            project = weighting_function(self.reg_max, up, reg_scale)
        else:
            project = self.project

        ref_points_detach = F.sigmoid(ref_points_unact)
        query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)

            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                value = self.value_op(memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes)
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)

            if i == 0 :
                # Initial bounding box predictions with inverse sigmoid refinement
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                
                if self.use_otadet_contrastive_head and not isinstance(score_head[0], nn.Identity):
                    dec_pre_out_proj = score_head[0].project_image(output)
                    pre_scores = score_head[0](output, text_features, caption_padding_mask=caption_padding_mask)
                    if attributes_feats is not None:
                        if not isinstance(attr_head[0], nn.Identity) and hasattr(attr_head[0], 'project_image'):
                            dec_pre_attr_out_proj = attr_head[0].project_image(output)
                        pre_attr_scores = attr_head[0](output, attributes_feats, caption_padding_mask=caption_attributes_padding_mask)
                else:
                    pre_scores = score_head[0](output)
                    
                ref_points_initial = pre_bboxes.detach()

            # Refine bounding box corners using FDR, integrating previous layer's corrections
            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)

            if self.training or i == self.eval_idx:
                if self.use_otadet_contrastive_head and not isinstance(score_head[i], nn.Identity):
                    proj_feats = score_head[i].project_image(output)
                    scores = score_head[i](output, text_features, caption_padding_mask=caption_padding_mask)
                    attr_scores = None
                    attr_proj_feats = None
                    if attributes_feats is not None:
                        if not isinstance(attr_head[i], nn.Identity) and hasattr(attr_head[i], 'project_image'):
                            attr_proj_feats = attr_head[i].project_image(output)
                        attr_scores = attr_head[i](output, attributes_feats, caption_padding_mask=caption_attributes_padding_mask)
                else:
                    scores = score_head[i](output)
                    proj_feats = None
                    attr_scores = None
                    attr_proj_feats = None
                # Lqe does not affect the performance here.
                scores = self.lqe_layers[i](scores, pred_corners)
                dec_out_logits.append(scores)
                if dec_out_proj is not None and proj_feats is not None:
                    dec_out_proj.append(proj_feats)
                
                if attributes_feats is not None and attr_scores is not None:
                    attr_scores = self.lqe_layers[i](attr_scores, pred_corners)
                    dec_out_attr_logits.append(attr_scores)
                    if dec_out_attr_proj is not None and attr_proj_feats is not None:
                        dec_out_attr_proj.append(attr_proj_feats)
                
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)

                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        if dec_out_proj is not None and len(dec_out_proj) > 0:
            dec_out_proj = torch.stack(dec_out_proj)
        else:
            dec_out_proj = None
        if dec_out_attr_proj is not None and len(dec_out_attr_proj) > 0:
            dec_out_attr_proj = torch.stack(dec_out_attr_proj)
        else:
            dec_out_attr_proj = None

        if attributes_feats is not None:
            return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), \
                torch.stack(dec_out_pred_corners), torch.stack(dec_out_refs), pre_bboxes, pre_scores, \
                torch.stack(dec_out_attr_logits), pre_attr_scores, dec_pre_out_proj, dec_out_proj, \
                dec_pre_attr_out_proj, dec_out_attr_proj
        else:
            return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), \
                torch.stack(dec_out_pred_corners), torch.stack(dec_out_refs), pre_bboxes, pre_scores, \
                None, None, dec_pre_out_proj, dec_out_proj, None, None
            


@register()
class DEIMTransformer(nn.Module):
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_points=4,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learn_query_content=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 aux_loss=True,
                 cross_attn_method='default',
                 query_select_method='default',
                 reg_max=32,
                 reg_scale=4.,
                 layer_scale=1,
                 mlp_act='relu',
                 use_gateway=True,
                 share_bbox_head=False,
                 share_score_head=False,
                 separate_attr_head=False,
                 use_otadet_contrastive_head=False,
                 OTADet_Head_cfg=None,
                 embeddings_dn_sim = False,
                 txt_dim = 768,
                 ):
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)

        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        scaled_dim = round(layer_scale*hidden_dim)
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.reg_max = reg_max
        self.separate_attr_head = separate_attr_head
        head_cfg = OTADet_Head_cfg

        assert query_select_method in ('default', 'one2many', 'agnostic'), ''
        assert cross_attn_method in ('default', 'discrete'), ''
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method
        # -- print the parameters
        print(f"     --- Use Gateway@{use_gateway} ---")
        print(f"     --- Use Share Bbox Head@{share_bbox_head} ---")
        print(f"     --- Use Share Score Head@{share_score_head} ---")

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, \
            activation, num_levels, num_points, cross_attn_method=cross_attn_method, use_gateway=use_gateway)
        decoder_layer_wide = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, \
            activation, num_levels, num_points, cross_attn_method=cross_attn_method, layer_scale=layer_scale, use_gateway=use_gateway)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, decoder_layer_wide, num_layers, nhead,
                                          reg_max, self.reg_scale, self.up, eval_idx, layer_scale, act=activation,
                                          use_otadet_contrastive_head=use_otadet_contrastive_head)
        # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        
        self.embeddings_dn_sim = embeddings_dn_sim
        if num_denoising > 0:
            if self.embeddings_dn_sim:
                self.denoising_class_embed = nn.Linear(txt_dim, hidden_dim)
            else:
                self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)
                init.normal_(self.denoising_class_embed.weight[:-1])

        # decoder embedding
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

        self.use_otadet_contrastive_head = use_otadet_contrastive_head
        if query_select_method == 'agnostic':
            self.enc_score_head = nn.Linear(hidden_dim, 1)
            self.attr_enc_score_head = self.enc_score_head
        else:
            if use_otadet_contrastive_head:
                enc_head = OTADetContrastiveHead(**head_cfg)
            else:
                enc_head = nn.Linear(hidden_dim, num_classes)
            self.enc_score_head = enc_head
            if separate_attr_head and use_otadet_contrastive_head:
                self.attr_enc_score_head = OTADetContrastiveHead(**head_cfg)
            else:
                self.attr_enc_score_head = self.enc_score_head
            
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)

        self.query_pos_head = MLP(4, hidden_dim, hidden_dim, 3, act=mlp_act)

        # decoder head
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)
        self.integral = Integral(self.reg_max)

        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        
        if use_otadet_contrastive_head:
            dec_score_head = OTADetContrastiveHead(**head_cfg)
            self.dec_score_head = nn.ModuleList(
                [dec_score_head if share_score_head else copy.deepcopy(dec_score_head) for _ in range(self.eval_idx + 1)]
            + [copy.deepcopy(dec_score_head) for _ in range(num_layers - self.eval_idx - 1)])
            if separate_attr_head:
                attr_dec_score_head = OTADetContrastiveHead(**head_cfg)
                self.attr_dec_score_head = nn.ModuleList(
                    [attr_dec_score_head if share_score_head else copy.deepcopy(attr_dec_score_head) for _ in range(self.eval_idx + 1)]
                + [copy.deepcopy(attr_dec_score_head) for _ in range(num_layers - self.eval_idx - 1)])
            else:
                self.attr_dec_score_head = self.dec_score_head

        else:
            dec_score_head = nn.Linear(hidden_dim, num_classes)
            self.dec_score_head = nn.ModuleList(
                [dec_score_head if share_score_head else copy.deepcopy(dec_score_head) for _ in range(self.eval_idx + 1)]
            + [copy.deepcopy(dec_score_head) for _ in range(num_layers - self.eval_idx - 1)])
            self.attr_dec_score_head = self.dec_score_head

        # Share the same bbox head for all layers
        dec_bbox_head = MLP(hidden_dim, hidden_dim, 4 * (self.reg_max+1), 3, act=mlp_act)
        self.dec_bbox_head = nn.ModuleList(
            [dec_bbox_head if share_bbox_head else copy.deepcopy(dec_bbox_head) for _ in range(self.eval_idx + 1)]
          + [MLP(scaled_dim, scaled_dim, 4 * (self.reg_max+1), 3, act=mlp_act) for _ in range(num_layers - self.eval_idx - 1)])

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors)
            self.register_buffer('valid_mask', valid_mask)
        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()


        self._reset_parameters(feat_channels)

    def convert_to_deploy(self):
        self.dec_score_head = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.dec_score_head[self.eval_idx]])
        self.dec_bbox_head = nn.ModuleList(
            [self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity() for i in range(len(self.dec_bbox_head))]
        )
        if self.attr_dec_score_head is not self.dec_score_head:
            self.attr_dec_score_head = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.attr_dec_score_head[self.eval_idx]])

    def _reset_parameters(self, feat_channels):
        bias = bias_init_with_prob(0.01)
        
        if not self.use_otadet_contrastive_head:
            init.constant_(self.enc_score_head.bias, bias)
            if self.separate_attr_head and self.attr_enc_score_head is not self.enc_score_head:
                init.constant_(self.attr_enc_score_head.bias, bias)
            
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            if not self.use_otadet_contrastive_head:
                init.constant_(cls_.bias, bias)
            if hasattr(reg_, 'layers'):
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)
        if self.separate_attr_head and self.attr_dec_score_head is not self.dec_score_head:
            for cls_ in self.attr_dec_score_head:
                if not self.use_otadet_contrastive_head:
                    init.constant_(cls_.bias, bias)

        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        init.xavier_uniform_(self.query_pos_head.layers[-1].weight)
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                        ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                    )
                )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                        ('norm', nn.BatchNorm2d(self.hidden_dim))])
                    )
                )
                in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask


    def _get_decoder_input(self,
                           memory: torch.Tensor,
                           spatial_shapes,
                           denoising_logits=None,
                           denoising_bbox_unact=None,
                           text_features=None,
                           caption_padding_mask=None,
                           attributes_feats=None, 
                           caption_attributes_padding_mask=None,
                           attr_score_head=None):

        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        # memory = torch.where(valid_mask, memory, 0)
        memory = valid_mask.to(memory.dtype) * memory

        if self.use_otadet_contrastive_head and text_features != None:
            enc_outputs_logits :torch.Tensor = self.enc_score_head(memory, text_features, caption_padding_mask=caption_padding_mask)
            if attributes_feats != None:
                enc_attr_head = attr_score_head if attr_score_head is not None else self.enc_score_head
                enc_attr_outputs_logits :torch.Tensor = enc_attr_head(memory, attributes_feats, caption_padding_mask=caption_attributes_padding_mask)
            else:
                enc_attr_outputs_logits = None
        else:
            enc_outputs_logits :torch.Tensor = self.enc_score_head(memory)
            enc_attr_outputs_logits = None

        # select topk queries
        enc_topk_memory, enc_topk_logits, enc_topk_anchors, enc_topk_attr_logits = \
            self._select_topk(memory, enc_outputs_logits, anchors, self.num_queries, enc_attr_outputs_logits)

        enc_topk_bbox_unact :torch.Tensor = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors

        enc_topk_bboxes_list, enc_topk_logits_list, enc_topk_attr_logits_list = [], [], []
        enc_topk_proj_list = []
        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)
            if self.use_otadet_contrastive_head:
                enc_topk_proj_list.append(self.enc_score_head.project_image(enc_topk_memory))
            if enc_topk_attr_logits != None:
                enc_topk_attr_logits_list.append(enc_topk_attr_logits)

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)

        if not self.use_otadet_contrastive_head or len(enc_topk_proj_list) == 0:
            enc_topk_proj_list = None
        if attributes_feats is None:
            enc_topk_attr_logits_list = None
        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list, enc_topk_attr_logits_list, enc_topk_proj_list

    def _select_topk(self, memory: torch.Tensor, outputs_logits: torch.Tensor, outputs_anchors_unact: torch.Tensor, topk: int, attr_outputs_logits):
        if self.query_select_method == 'default':
            base_scores = outputs_logits.max(-1).values  # [B, L]
            _, topk_ind = torch.topk(base_scores, topk, dim=-1)


        elif self.query_select_method == 'one2many':
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes

        elif self.query_select_method == 'agnostic':
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)

        topk_ind: torch.Tensor

        topk_anchors = outputs_anchors_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_anchors_unact.shape[-1]))

        topk_logits = outputs_logits.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1])) if self.training else None

        if attr_outputs_logits != None:
            topk_attr_logits = attr_outputs_logits.gather(dim=1, \
                index=topk_ind.unsqueeze(-1).repeat(1, 1, attr_outputs_logits.shape[-1])) if self.training else None
        else:
            topk_attr_logits = None
        
        topk_memory = memory.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))

        return topk_memory, topk_logits, topk_anchors, topk_attr_logits

    def forward(self, feats, targets=None, text_features=None, caption_padding_mask=None,
                attributes_feats=None, caption_attributes_padding_mask=None, cap_to_attr_map_batch=None):
        # input projection and embedding
        memory, spatial_shapes = self._get_encoder_input(feats)

        # prepare denoising training
        if self.training and self.num_denoising > 0:
            if self.embeddings_dn_sim:
                denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                    get_contrastive_denoising_training_group(targets, \
                        -1,
                        self.num_queries,
                        self.denoising_class_embed,
                        num_denoising=self.num_denoising,
                        label_noise_ratio=self.label_noise_ratio,
                        box_noise_scale=1.0,
                        text_features = text_features
                    )
            else:
                denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                    get_contrastive_denoising_training_group(targets, \
                        self.num_classes,
                        self.num_queries,
                        self.denoising_class_embed,
                        num_denoising=self.num_denoising,
                        label_noise_ratio=self.label_noise_ratio,
                        box_noise_scale=1.0,
                    )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        if self.use_otadet_contrastive_head and text_features != None:
            init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list, enc_topk_attr_logits_list, enc_topk_proj_list = \
                self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact,
                                        text_features=text_features, caption_padding_mask=caption_padding_mask,
                                        attributes_feats=attributes_feats, caption_attributes_padding_mask=caption_attributes_padding_mask,
                                        attr_score_head=self.attr_enc_score_head)
        else:
            init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list, enc_topk_attr_logits_list, enc_topk_proj_list = \
                self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits, out_attr_logits, attr_pre_logits, pre_out_proj, out_proj, \
            pre_attr_out_proj, out_attr_proj = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.attr_dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.up,
            self.reg_scale,
            attn_mask=attn_mask,
            dn_meta=dn_meta,
            text_features=text_features,
            caption_padding_mask=caption_padding_mask,
            attributes_feats=attributes_feats, 
            caption_attributes_padding_mask=caption_attributes_padding_mask)

        if self.training and dn_meta is not None:
            # the output from the first decoder layer, only one
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta['dn_num_split'], dim=1)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            
            if out_attr_logits != None:
                dn_pre_attr_logits, attr_pre_logits = torch.split(attr_pre_logits, dn_meta['dn_num_split'], dim=1)
                dn_out_attr_logits, out_attr_logits = torch.split(out_attr_logits, dn_meta['dn_num_split'], dim=2)
            else:
                dn_pre_attr_logits, attr_pre_logits = None, None
                dn_out_attr_logits, out_attr_logits = None, None
                
            if out_proj is not None:
                dn_pre_out_proj, pre_out_proj = torch.split(pre_out_proj, dn_meta['dn_num_split'], dim=1)
                dn_out_proj, out_proj = torch.split(out_proj, dn_meta['dn_num_split'], dim=2)
            else:
                dn_pre_out_proj, pre_out_proj = None, None
                dn_out_proj, out_proj = None, None
            if out_attr_proj is not None:
                dn_attr_out_proj, out_attr_proj = torch.split(out_attr_proj, dn_meta['dn_num_split'], dim=2)
            else:
                dn_attr_out_proj, out_attr_proj = None, None
            if pre_attr_out_proj is not None:
                dn_pre_attr_out_proj, pre_attr_out_proj = torch.split(pre_attr_out_proj, dn_meta['dn_num_split'], dim=1)
            else:
                dn_pre_attr_out_proj, pre_attr_out_proj = None, None
                
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta['dn_num_split'], dim=1)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_corners, out_corners = torch.split(out_corners, dn_meta['dn_num_split'], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta['dn_num_split'], dim=2)

        if self.training:
            # pre_logits correspond to the final decoder-layer output.
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1], 'pred_corners': out_corners[-1],
                   'ref_points': out_refs[-1], 'up': self.up, 'reg_scale': self.reg_scale,}
            if out_attr_logits != None:
                out['pred_attr_logits'] = out_attr_logits[-1]
            if out_proj is not None:
                out['proj_queries'] = out_proj[-1]
            if out_attr_proj is not None:
                out['attr_proj_queries'] = out_attr_proj[-1]
        else:
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}
            if out_attr_logits != None:
                out['pred_attr_logits'] = out_attr_logits[-1]

        if self.training and self.aux_loss:
            if out_attr_logits != None:
                out['aux_outputs'] = self._set_aux_loss2(out_logits[:-1], out_bboxes[:-1], out_corners[:-1], out_refs[:-1],
                                                        out_corners[-1], out_logits[-1], 
                                                        out_attr_logits[:-1], out_attr_logits[-1],
                                                        out_proj[:-1] if out_proj is not None else None,
                                                        out_attr_proj[:-1] if out_attr_proj is not None else None)
            else:
                out['aux_outputs'] = self._set_aux_loss2(out_logits[:-1], out_bboxes[:-1], out_corners[:-1], out_refs[:-1],
                                                        out_corners[-1], out_logits[-1],
                                                        outputs_proj=out_proj[:-1] if out_proj is not None else None,
                                                        outputs_attr_proj=out_attr_proj[:-1] if out_attr_proj is not None else None)
            
            out['enc_aux_outputs'] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list, enc_topk_attr_logits_list,
                                                        enc_topk_proj_list)

           
            if attr_pre_logits != None:
                out['pre_outputs'] = {'pred_logits': pre_logits, 'pred_boxes': pre_bboxes,
                                      'pred_attr_logits': attr_pre_logits,
                                      'proj_queries': pre_out_proj}
                if pre_attr_out_proj is not None:
                    out['pre_outputs']['attr_proj_queries'] = pre_attr_out_proj
            else:
                out['pre_outputs'] = {'pred_logits': pre_logits, 'pred_boxes': pre_bboxes}
                if pre_out_proj is not None:
                    out['pre_outputs']['proj_queries'] = pre_out_proj
                if pre_attr_out_proj is not None:
                    out['pre_outputs']['attr_proj_queries'] = pre_attr_out_proj
            
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}

            if dn_meta is not None:
                if dn_out_attr_logits != None:
                    out['dn_outputs'] = self._set_aux_loss2(dn_out_logits, dn_out_bboxes, dn_out_corners, dn_out_refs,
                                                        dn_out_corners[-1], dn_out_logits[-1],
                                                        dn_out_attr_logits, dn_out_attr_logits[-1],
                                                        dn_out_proj,
                                                        dn_attr_out_proj)
                else:
                    out['dn_outputs'] = self._set_aux_loss2(dn_out_logits, dn_out_bboxes, dn_out_corners, dn_out_refs,
                                                        dn_out_corners[-1], dn_out_logits[-1],
                                                        outputs_proj=dn_out_proj,
                                                        outputs_attr_proj=dn_attr_out_proj)
                
                if attr_pre_logits != None:
                    out['dn_pre_outputs'] = {'pred_logits': dn_pre_logits, 'pred_boxes': dn_pre_bboxes,
                                             'pred_attr_logits': dn_pre_attr_logits,
                                             'proj_queries': dn_pre_out_proj}
                    if dn_pre_attr_out_proj is not None:
                        out['dn_pre_outputs']['attr_proj_queries'] = dn_pre_attr_out_proj
                else:
                    out['dn_pre_outputs'] = {'pred_logits': dn_pre_logits, 'pred_boxes': dn_pre_bboxes}
                    if dn_pre_out_proj is not None:
                        out['dn_pre_outputs']['proj_queries'] = dn_pre_out_proj
                    if dn_pre_attr_out_proj is not None:
                        out['dn_pre_outputs']['attr_proj_queries'] = dn_pre_attr_out_proj
                out['dn_meta'] = dn_meta

        # build caption+attribute groups for OT alignment (caption slot + per-caption attributes)
        # OT alignment is disabled by default for release configs.
        # caption_attr_groups = None
        # caption_attr_mask = None
        # if attributes_feats is not None and text_features is not None and cap_to_attr_map_batch is not None:
        #     caption_attr_groups, caption_attr_mask = self._build_caption_attr_groups(
        #         text_features, attributes_feats, cap_to_attr_map_batch,
        #         caption_padding_mask=caption_padding_mask,
        #         caption_attributes_padding_mask=caption_attributes_padding_mask,
        #     )
        #     if caption_attr_groups is not None:
        #         cap_map_device = cap_to_attr_map_batch.to(caption_attr_groups.device)
        #         out['caption_attr_groups'] = caption_attr_groups
        #         out['caption_attr_mask'] = caption_attr_mask
        #         out['cap_to_attr_map'] = cap_map_device
        #         for aux in out.get('aux_outputs', []):
        #             aux['caption_attr_groups'] = caption_attr_groups
        #             aux['caption_attr_mask'] = caption_attr_mask
        #             aux['cap_to_attr_map'] = cap_map_device
        #         for aux in out.get('enc_aux_outputs', []):
        #             aux['caption_attr_groups'] = caption_attr_groups
        #             aux['caption_attr_mask'] = caption_attr_mask
        #             aux['cap_to_attr_map'] = cap_map_device
        #         if 'pre_outputs' in out:
        #             out['pre_outputs']['caption_attr_groups'] = caption_attr_groups
        #             out['pre_outputs']['caption_attr_mask'] = caption_attr_mask
        #             out['pre_outputs']['cap_to_attr_map'] = cap_map_device
        #         if 'dn_outputs' in out:
        #             for aux in out['dn_outputs']:
        #                 aux['caption_attr_groups'] = caption_attr_groups
        #                 aux['caption_attr_mask'] = caption_attr_mask
        #                 aux['cap_to_attr_map'] = cap_map_device
        #         if 'dn_pre_outputs' in out:
        #             out['dn_pre_outputs']['caption_attr_groups'] = caption_attr_groups
        #             out['dn_pre_outputs']['caption_attr_mask'] = caption_attr_mask
        #             out['dn_pre_outputs']['cap_to_attr_map'] = cap_map_device

        return out


    @torch.jit.unused
    def _to_bool_mask(self, mask, ref_tensor):
        if mask is None:
            return torch.ones(ref_tensor.shape[:2], dtype=torch.bool, device=ref_tensor.device)
        if isinstance(mask, list):
            mask = torch.stack(mask, dim=0)
        return mask.to(device=ref_tensor.device, dtype=torch.bool)

    @torch.jit.unused
    def _build_caption_attr_groups(self,
                                   caption_feats: torch.Tensor,
                                   attr_feats: torch.Tensor,
                                   cap_to_attr_map: torch.Tensor,
                                   caption_padding_mask=None,
                                   caption_attributes_padding_mask=None):
        """
        Build per-caption text slots: [caption, attr_1, ..., attr_K]
        Returns:
            groups: [B, num_caps, K+1, D]
            mask:   [B, num_caps, K+1]  (True for valid slots)
        """
        if cap_to_attr_map is None:
            return None, None
        if isinstance(cap_to_attr_map, (list, tuple)):
            cap_to_attr_map = torch.stack([m.to(caption_feats.device) for m in cap_to_attr_map], dim=0)
        else:
            cap_to_attr_map = cap_to_attr_map.to(caption_feats.device)

        caption_mask = self._to_bool_mask(caption_padding_mask, caption_feats)
        attr_mask = self._to_bool_mask(caption_attributes_padding_mask, attr_feats)
        if attr_mask.shape[1] != attr_feats.shape[1]:
            raise ValueError('attr_mask.shape[1] != attr_feats.shape[1], check')

        bsz, num_caps, dim = caption_feats.shape
        max_attrs_per_cap = cap_to_attr_map.shape[-1]
        # cap_to_attr_map = cap_to_attr_map[:, :num_caps, :max_attrs_per_cap]

        groups = caption_feats.new_zeros((bsz, num_caps, max_attrs_per_cap + 1, dim))
        mask = torch.zeros((bsz, num_caps, max_attrs_per_cap + 1), dtype=torch.bool, device=caption_feats.device)

        groups[:, :, 0] = caption_feats
        mask[:, :, 0] = caption_mask[:, :num_caps]

        if attr_feats is None or attr_feats.numel() == 0:
            return groups, mask

        max_attr_idx = max(attr_feats.shape[1] - 1, 0)
        valid_attr_idx = (cap_to_attr_map >= 0) & (cap_to_attr_map <= max_attr_idx)
        safe_idx = cap_to_attr_map.clamp(min=0, max=max_attr_idx)

        attr_feats_exp = attr_feats.unsqueeze(1).expand(-1, num_caps, -1, -1)
        gather_idx = safe_idx.unsqueeze(-1).expand(-1, -1, -1, attr_feats.shape[-1])
        gathered_attr = torch.gather(attr_feats_exp, 2, gather_idx)
        groups[:, :, 1:] = gathered_attr * valid_attr_idx.unsqueeze(-1)

        attr_mask_exp = attr_mask.unsqueeze(1).expand(-1, num_caps, -1)
        gathered_attr_mask = torch.gather(attr_mask_exp, 2, safe_idx)
        valid_attr = valid_attr_idx & gathered_attr_mask
        mask[:, :, 1:] = valid_attr & mask[:, :, 0].unsqueeze(-1)
        return groups, mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_attr_class=None, outputs_proj_class=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        results = []
        for idx in range(len(outputs_class)):
            item = {'pred_logits': outputs_class[idx], 'pred_boxes': outputs_coord[idx]}
            if outputs_attr_class is not None:
                item['pred_attr_logits'] = outputs_attr_class[idx]
            if outputs_proj_class is not None:
                item['proj_queries'] = outputs_proj_class[idx]
            results.append(item)
        return results


    @torch.jit.unused
    def _set_aux_loss2(self, outputs_class, outputs_coord, outputs_corners, outputs_ref,
                       teacher_corners=None, teacher_logits=None,
                       outputs_attr_class=None, teacher_attr_logtis=None,
                       outputs_proj=None, outputs_attr_proj=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        results = []
        for idx in range(len(outputs_class)):
            item = {
                'pred_logits': outputs_class[idx],
                'pred_boxes': outputs_coord[idx],
                'pred_corners': outputs_corners[idx],
                'ref_points': outputs_ref[idx],
                'teacher_corners': teacher_corners,
                'teacher_logits': teacher_logits,
            }
            if outputs_attr_class is not None:
                item['pred_attr_logits'] = outputs_attr_class[idx]
                item['teacher_attr_logtis'] = teacher_attr_logtis
            if outputs_proj is not None:
                item['proj_queries'] = outputs_proj[idx]
            if outputs_attr_proj is not None:
                item['attr_proj_queries'] = outputs_attr_proj[idx]
            results.append(item)
        return results
