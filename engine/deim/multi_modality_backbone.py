# Copyright (c) Tencent Inc. All rights reserved.
import itertools
from typing import List, Sequence, Tuple, Optional
import torch, os
from pathlib import Path
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from torch import nn
import torch.nn.functional as F

from ..core import register
from ..backbone.CLIP_encoder.openclip_hf_model import HFTextEncoder
from ..backbone.CLIP_encoder.openclip_transformer import TextTransformer
from ..backbone.presnet import PResNet
from ..backbone.dinov3_adapter import DINOv3STAs

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
from ..backbone.CLIP_encoder.openclip_transformer import (
    LayerNormFp32,
    LayerNorm,
    QuickGELU,
    Attention,
    VisionTransformer,
    TextTransformer,
    text_global_pool,
    lock_text_tower,
)
from ..backbone.CLIP_encoder.openclip_utils import to_2tuple
from ..backbone.CLIP_encoder.openai_factory import get_tokenizer
from itertools import chain

__all__ = ['Multi_modality_Backbone']

@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    hf_tokenizer_name: Optional[str] = None
    tokenizer_mode: Optional[str] = None
    tokenizer_kwargs: Optional[dict] = None

    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    ls_init_value: Optional[float] = None  # layer scale initial value
    embed_cls: bool = False
    pad_id: int = 0
    eos_id: int = 2  # only used for when pool_type == 'eos', must match tokenizer eos
    no_causal_mask: bool = False  # disable causal masking
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'argmax'
    proj_bias: bool = False
    proj_type: str = 'linear'  # control final text projection, 'none' forces no projection
    output_tokens: bool = False
    act_kwargs: dict = None
    norm_kwargs: dict = None

    # Custom attention block settings
    block_type: Optional[str] = None  # attention block type ('default', 'custom'), auto-selects 'custom' if any custom features enabled
    qk_norm: bool = False  # apply layer norm to q and k in attention
    scaled_cosine_attn: bool = False  # use scaled cosine attention
    scale_heads: bool = False  # learnable head-specific scale applied to attention logits
    scale_attn_inner: bool = False  # apply layer norm on attention context, before output projection
    scale_attn: bool = False  # apply layer norm after full attention block
    scale_fc: bool = False  # apply layer norm in MLP block

    # HuggingFace specific text tower config
    hf_model_name: Optional[str] = None
    hf_model_pretrained: bool = True
    hf_proj_type: str = 'mlp'
    hf_pooler_type: str = 'mean_pooler'  # attentional pooling for HF models
    
    weights_path: Optional[str] = None  # ckpt path



@register()
class Multi_modality_Backbone(nn.Module):
    def __init__(self,
                 image_model_cfg,
                 text_model_cfg,
                 tokenizer_model,
                 image_model_type,
                 text_model_type,
                 embed_dim: int,
                 custom_text,
                 quick_gelu: bool = False,
                 cast_dtype: Optional[torch.dtype] = None,
                 lock_text_tower_flag: bool = True,
                 attribute_embed_method: str = 'separate') -> None:  
        super(Multi_modality_Backbone, self).__init__()
        assert attribute_embed_method in ('separate', 'caption_token_mean'), \
            f'Unsupported attribute_embed_method: {attribute_embed_method}'
        
        if text_model_type != None:
            self.with_text_model = True
        else:
            self.with_text_model = False
        
        if image_model_type == 'DINOv3STAs':
            self.image_model = DINOv3STAs(**image_model_cfg)
        elif image_model_type == 'PResNet':
            self.image_model = PResNet(**image_model_cfg)
        else:
            self.image_model = None
            
        if text_model_type == 'CLIP_Text_Encoder':
            self.custom_text = custom_text
            text_model_cfg = CLIPTextCfg(**text_model_cfg) if isinstance(text_model_cfg, dict) else text_model_cfg
            self.text_out_dim = text_model_cfg.width
            self.text_model = _build_text_tower(embed_dim, text_model_cfg, quick_gelu, cast_dtype)
            self.transformer = self.text_model.transformer
            self.context_length = self.text_model.context_length
            self.vocab_size = self.text_model.vocab_size
            self.token_embedding = self.text_model.token_embedding
            self.positional_embedding = self.text_model.positional_embedding
            self.ln_final = self.text_model.ln_final
            self.text_projection = self.text_model.text_projection
            self.text_pool_type = self.text_model.pool_type
            self.text_eos_id = self.text_model.eos_id
            self.register_buffer('attn_mask', self.text_model.attn_mask, persistent=False)
            self.tokenizer = get_tokenizer(tokenizer_model)

            if text_model_cfg.weights_path is not None and os.path.exists(text_model_cfg.weights_path):
                self._load_text_encoder_weights(text_model_cfg.weights_path)
            else:
                print('Training text encoder from scratch...')

        else:
            self.text_model = None
            
        self.lock_text_tower_flag = lock_text_tower_flag
        if self.lock_text_tower_flag:
            self.lock_text_tower()

        self.attribute_embed_method = attribute_embed_method
        # Text cache controls for skipping text encoder during profiling/FPS.
        self.text_cache_mode = "off"        # off | record | reuse
        self.text_cache_path: Optional[Path] = None
        self.text_cache_max_entries: Optional[int] = None
        self.text_feat_cache: dict = {}

    def _load_text_encoder_weights(self, weights_path: str):
        print(f'Loading text encoder weights from {weights_path}...')

        checkpoint_path = weights_path
        print(f'Using checkpoint file: {checkpoint_path}')

        # 加载state_dict
        # state_dict = self._load_text_state_dict(checkpoint_path)
        if str(checkpoint_path).endswith('.safetensors'):
            try:
                from safetensors.torch import load_file
                print('Loading from safetensors format...')
                state_dict = load_file(checkpoint_path, device='cpu')
            except ImportError:
                raise ImportError("safetensors not installed. Please install: pip install safetensors")
        else:
            # PyTorch格式
            print('Loading from PyTorch format...')
            try:
                state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            except TypeError:
                # 旧版本PyTorch不支持weights_only参数
                state_dict = torch.load(checkpoint_path, map_location='cpu')


        # 处理不同的checkpoint格式
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # 处理module前缀
        if next(iter(state_dict.items()))[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        # 转换为custom text格式（如果需要）
        # state_dict = convert_to_custom_text_state_dict(state_dict)

        # 提取text相关的权重
        text_model_dict = self._extract_text_weights(state_dict)

        # 加载权重，允许部分匹配
        missing_keys, unexpected_keys = self.text_model.load_state_dict(text_model_dict, strict=False)

        if missing_keys:
            print(f"Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")  # 只显示前5个
        if unexpected_keys:
            print(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")

        print(f"Successfully loaded {len(text_model_dict)} parameters for text encoder")

    def _extract_text_weights(self, state_dict: dict):
        """
        Extract text encoder weights from full state dict.

        Args:
            state_dict: Full state dictionary

        Returns:
            text_model_dict: Filtered dictionary containing only text encoder weights
        """
        text_model_dict = {}

        for k, v in state_dict.items():
            # 移除text_model或text前缀
            if k.startswith('text_model.') or k.startswith('text.'):
                new_k = k.replace('text_model.', '').replace('text.', '')
                text_model_dict[new_k] = v
            # 直接包含的text相关键
            elif any(k.startswith(prefix) for prefix in [
                'transformer.',
                'token_embedding.',
                'positional_embedding',
                'ln_final.',
                'text_projection',
                'attn_mask'
            ]):
                text_model_dict[k] = v

        return text_model_dict

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        if self.text_model is not None:
            self.text_model.lock(unlocked_layers, freeze_layer_norm)
            print('Freeze text encoder...')

    # ---------------------- Text cache helpers ---------------------- #
    def _make_cache_key(self, captions_batch: List[List[str]]):
        """Normalize captions into a hashable structure (tuple of tuples of strings)."""
        def normalize_caption(c):
            if isinstance(c, (list, tuple)) and len(c) > 0:
                return str(c[0])
            return str(c)
        return tuple(tuple(normalize_caption(c) for c in caps) for caps in captions_batch)

    def set_text_cache(self, path: Optional[str], mode: str = "reuse", max_entries: Optional[int] = None):
        """Configure text feature cache. mode: off | record | reuse."""
        mode = mode.lower()
        if mode not in ("off", "record", "reuse"):
            raise ValueError(f"Unsupported text cache mode: {mode}")
        self.text_cache_mode = mode
        self.text_cache_max_entries = max_entries
        self.text_cache_path = Path(path) if path else None

        if mode == "reuse":
            if self.text_cache_path and self.text_cache_path.exists():
                data = torch.load(self.text_cache_path, map_location="cpu")
                self.text_feat_cache = data.get("text_feats", {})
                print(f"[TextCache] Loaded {len(self.text_feat_cache)} entries from {self.text_cache_path}")
            else:
                print(f"[TextCache] reuse requested but path missing: {self.text_cache_path}, disabling cache.")
                self.text_cache_mode = "off"

    def save_text_cache(self):
        if self.text_cache_mode != "record":
            return
        if not self.text_feat_cache or not self.text_cache_path:
            return
        self.text_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"text_feats": self.text_feat_cache}, self.text_cache_path)
        print(f"[TextCache] Saved {len(self.text_feat_cache)} entries to {self.text_cache_path}")

    def _maybe_get_cached_text(self, captions_batch: List[List[str]], device: torch.device):
        if self.text_cache_mode != "reuse":
            return None
        key = self._make_cache_key(captions_batch)
        cached = self.text_feat_cache.get(key)
        if cached is None:
            return None
        return cached.to(device)

    def _maybe_store_cached_text(self, captions_batch: List[List[str]], txt_feats: torch.Tensor):
        if self.text_cache_mode != "record":
            return
        if self.text_cache_max_entries is not None and len(self.text_feat_cache) >= self.text_cache_max_entries:
            return
        if self.text_cache_path is None:
            return
        key = self._make_cache_key(captions_batch)
        if key in self.text_feat_cache:
            return
        self.text_feat_cache[key] = txt_feats.detach().cpu()

    def forward(self, image: Tensor,
                captions_batch: List[List[str]] = None,
                normalize = True,
                caption_padding_mask=None,
                caption_attributes_batch = None,
                caption_attributes_padding_mask = None,
                cap_to_attr_map_batch: Optional[Tensor] = None) -> Tuple[Tuple[Tensor], Tensor]:
        img_feats = self.forward_image(image)
        attributes_feats = None
        if self.with_text_model and captions_batch != None:
            need_tokens = self.attribute_embed_method == 'caption_token_mean'
            can_use_cache = (not need_tokens) and caption_attributes_batch is None
            txt_feats = None

            if can_use_cache:
                cached = self._maybe_get_cached_text(captions_batch, img_feats[0].device)
                if cached is not None:
                    # print('use cached')
                    txt_feats = cached

            if txt_feats is None:
                # print('text encoder')
                if self.custom_text:
                    if need_tokens:
                        txt_feats, caption_tokens, caption_token_ids = self.forward_custom_text(
                            captions_batch, normalize=normalize, device=img_feats[0].device,
                            caption_padding_mask=caption_padding_mask, return_projected_tokens=True, return_token_ids=True)
                    else:
                        txt_feats = self.forward_custom_text(captions_batch, normalize=normalize, device=img_feats[0].device, caption_padding_mask=caption_padding_mask)
                else:
                    if need_tokens:
                        txt_feats, caption_tokens, caption_token_ids = self.forward_text(
                            captions_batch, normalize=normalize, device=img_feats[0].device,
                            caption_padding_mask=caption_padding_mask, return_projected_tokens=True, return_token_ids=True)
                    else:
                        txt_feats = self.forward_text(captions_batch, normalize=normalize, device=img_feats[0].device, caption_padding_mask=caption_padding_mask)

                if can_use_cache:
                    self._maybe_store_cached_text(captions_batch, txt_feats)

            if caption_attributes_batch != None:
                if self.attribute_embed_method == 'caption_token_mean':
                    if not need_tokens or caption_tokens is None or caption_token_ids is None:
                        raise ValueError("caption_token_mean attribute mode requires text_cfg.output_tokens=True to return token features.")
                    attributes_feats, found_mask = self._pool_attributes_from_captions(
                        caption_attributes_batch,
                        caption_attributes_padding_mask,
                        caption_tokens,
                        caption_token_ids,
                        caption_padding_mask,
                        captions_batch,
                        cap_to_attr_map_batch,
                        normalize=normalize,
                        device=img_feats[0].device
                    )
                    if found_mask is not None and not found_mask.all():
                        if self.custom_text:
                            fallback = self.forward_custom_text(caption_attributes_batch, normalize=normalize, device=img_feats[0].device, caption_padding_mask=caption_attributes_padding_mask)
                        else:
                            fallback = self.forward_text(caption_attributes_batch, normalize=normalize, device=img_feats[0].device, caption_padding_mask=caption_attributes_padding_mask)
                        # 仅对未匹配的属性使用回退特征
                        mask = found_mask.to(attributes_feats.dtype).unsqueeze(-1)
                        attributes_feats = attributes_feats * mask + fallback * (1 - mask)
                else:
                    if self.custom_text:
                        attributes_feats = self.forward_custom_text(caption_attributes_batch, normalize=normalize, device=img_feats[0].device, caption_padding_mask=caption_attributes_padding_mask)
                    else:
                        attributes_feats = self.forward_text(caption_attributes_batch, normalize=normalize, device=img_feats[0].device, caption_padding_mask=caption_attributes_padding_mask)
            return img_feats, txt_feats, attributes_feats
        else:
            return img_feats

    def forward_text(self, captions_batch: List[List[str]], normalize=True, device='cuda', caption_padding_mask=None,
                    return_projected_tokens: bool = False, return_token_ids: bool = False):
        # 展平 texts：B_total = sum_i num_classes_i
        num_batch_classes = len(captions_batch)
        num_classes = len(captions_batch[0])
        all_texts = [text_list[0] for batch in captions_batch for text_list in batch]  # len = B_total

        # 若提供了 padding mask（布尔；True=有效，False=padding），展开成与 all_texts 对齐的一维 mask
        if caption_padding_mask is not None:
            flat_mask = torch.cat([m.to(torch.bool) for m in caption_padding_mask], dim=0)
        else:
            # 没有 mask 就默认都有效
            flat_mask = torch.ones(len(all_texts), dtype=torch.bool)

        # 取有效索引
        if flat_mask.ndim != 1 or flat_mask.numel() != len(all_texts):
            raise ValueError(f"caption_padding_mask shape mismatch: expect {len(all_texts)}, got {tuple(flat_mask.shape)}")
        valid_idx = torch.nonzero(flat_mask, as_tuple=False).squeeze(1)
        has_valid = valid_idx.numel() > 0

        # 只对有效 captions 做 tokenize + encoder
        out_dim = self.text_projection.weight.shape[1] if isinstance(self.text_projection, nn.Linear) else self.text_out_dim
        out = torch.zeros((len(all_texts), out_dim), device=device, dtype=self.transformer.get_cast_dtype())
        seq_len = self.context_length
        tokens_proj_out = torch.zeros((len(all_texts), seq_len, out_dim), device=device, dtype=self.transformer.get_cast_dtype()) if return_projected_tokens else None
        token_ids_out = torch.zeros((len(all_texts), seq_len), device=device, dtype=torch.long) if return_token_ids else None

        if has_valid:
            valid_texts = [all_texts[i] for i in valid_idx.tolist()]
            texts_tok = self.tokenizer(valid_texts).to(device)
            cast_dtype = self.transformer.get_cast_dtype()

            # 标准 CLIP 流程
            x = self.token_embedding(texts_tok).to(cast_dtype)
            x = x + self.positional_embedding.to(cast_dtype)
            x = self.transformer(x, attn_mask=self.attn_mask)
            x = self.ln_final(x)
            tokens_proj = self._project_tokens(x) if return_projected_tokens else None
            x = text_global_pool(x, texts_tok, self.text_pool_type, eos_token_id=getattr(self, "text_eos_id", None))

            if self.text_projection is not None:
                if isinstance(self.text_projection, nn.Linear):
                    x = self.text_projection(x)
                else:
                    x = x @ self.text_projection

            out[valid_idx] = F.normalize(x, dim=-1) if normalize else x
            if return_projected_tokens and tokens_proj is not None:
                tokens_proj_out[valid_idx] = tokens_proj.to(dtype=tokens_proj_out.dtype)
            if return_token_ids:
                token_ids_out[valid_idx] = texts_tok

        # 还原为 [batch, num_classes, dim]
        out = out.view(num_batch_classes, num_classes, -1)

        if return_projected_tokens or return_token_ids:
            tokens_proj_out = tokens_proj_out.view(num_batch_classes, num_classes, -1, tokens_proj_out.shape[-1]) if tokens_proj_out is not None else None
            token_ids_out = token_ids_out.view(num_batch_classes, num_classes, -1) if token_ids_out is not None else None
            return out, tokens_proj_out, token_ids_out

        return out

    def forward_custom_text(self, captions_batch: List[List[str]], normalize=True, device='cuda', caption_padding_mask=None,
                            return_projected_tokens: bool = False, return_token_ids: bool = False):
        num_batch_classes = len(captions_batch)
        num_classes = len(captions_batch[0])

        flat_texts = list(chain.from_iterable(captions_batch))  # len = B_total

        if caption_padding_mask is not None:
            flat_mask = torch.cat([m.to(torch.bool) for m in caption_padding_mask], dim=0)
        else:
            flat_mask = torch.ones(len(flat_texts), dtype=torch.bool)

        if flat_mask.ndim != 1 or flat_mask.numel() != len(flat_texts):
            raise ValueError(f"caption_padding_mask shape mismatch: expect {len(flat_texts)}, got {tuple(flat_mask.shape)}")

        valid_idx = torch.nonzero(flat_mask, as_tuple=False).squeeze(1)
        has_valid = valid_idx.numel() > 0

        out_dim = self.text_projection.weight.shape[1] if isinstance(self.text_projection, nn.Linear) else self.text_out_dim
        out = torch.zeros((len(flat_texts), out_dim), device=device, dtype=self.transformer.get_cast_dtype())
        seq_len = self.context_length
        tokens_proj_out = torch.zeros((len(flat_texts), seq_len, out_dim), device=device, dtype=self.transformer.get_cast_dtype()) if return_projected_tokens else None
        token_ids_out = torch.zeros((len(flat_texts), seq_len), device=device, dtype=torch.long) if return_token_ids else None

        if has_valid:
            valid_texts = [flat_texts[i] for i in valid_idx.tolist()]
            # if not (isinstance(valid_texts, list) and all(isinstance(x, str) and x.strip() for x in valid_texts)): raise ValueError("valid_texts 含 None/非 str/空串")
            texts_tok = self.tokenizer(valid_texts).to(device)
            model_out = self.text_model(texts_tok)

            if isinstance(model_out, tuple):
                x, tokens_raw = model_out # x为pool的global token值，tokens_raw是细粒度tokens值[32， 768]， [32, 64, 768]
            else:
                x, tokens_raw = model_out, None

            tokens_proj = None
            if return_projected_tokens:  # 这是因为tokens_raw相比pool,缺少一次proj，可以查看engine/backbone/CLIP_encoder/openclip_transformer.py
                if tokens_raw is None:
                    raise ValueError("text_model must be created with output_tokens=True to return token-level embeddings.")
                tokens_proj = self._project_tokens(tokens_raw)

            out[valid_idx] = F.normalize(x, dim=-1) if normalize else x

            if return_projected_tokens and tokens_proj is not None:
                tokens_proj_out[valid_idx] = tokens_proj.to(dtype=tokens_proj_out.dtype)  # 把sentence embed映射回对应位置 
            if return_token_ids:
                token_ids_out[valid_idx] = texts_tok

        out = out.view(num_batch_classes, num_classes, -1)

        if return_projected_tokens or return_token_ids:
            tokens_proj_out = tokens_proj_out.view(num_batch_classes, num_classes, -1, tokens_proj_out.shape[-1]) if tokens_proj_out is not None else None
            token_ids_out = token_ids_out.view(num_batch_classes, num_classes, -1) if token_ids_out is not None else None
            return out, tokens_proj_out, token_ids_out

        return out

    def _project_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.text_projection is None:
            return tokens
        if isinstance(self.text_projection, nn.Linear):
            return self.text_projection(tokens)
        return tokens @ self.text_projection

    def _valid_token_length(self, token_ids: torch.Tensor, eot_id: Optional[int], pad_id: int) -> int:
        token_list = token_ids.tolist()
        for i, tok in enumerate(token_list):
            if (eot_id is not None and tok == eot_id) or tok == pad_id:
                return i
        return len(token_list)

    def _trim_token_ids(self, token_ids: torch.Tensor, eot_id: Optional[int], pad_id: int, drop_sot: bool) -> Tuple[int, List[int]]:
        valid_len = self._valid_token_length(token_ids, eot_id, pad_id)
        trimmed = token_ids[:valid_len].tolist()
        if drop_sot and trimmed:
            trimmed = trimmed[1:]
        return len(trimmed), trimmed

    def _find_subsequence(self, sequence: List[int], subseq: List[int]) -> List[int]:
        if not subseq or not sequence or len(subseq) > len(sequence):
            return []
        matches = []
        target_len = len(subseq)
        for start in range(len(sequence) - target_len + 1):
            if sequence[start:start + target_len] == subseq:
                matches.append(start)
        return matches

    def _get_token_offsets(self, text: str):
        # 仅在 tokenizer 支持 offset mapping 时可用
        tok = getattr(self.tokenizer, 'tokenizer', None)
        if tok is None or not hasattr(tok, 'batch_encode_plus'):
            return None
        try:
            enc = tok.batch_encode_plus(
                [text],
                return_offsets_mapping=True,
                max_length=self.context_length,
                padding='max_length',
                truncation=True,
            )
            offsets = enc.get('offset_mapping', None)
            if offsets is None:
                return None
            offsets = offsets[0]
            if hasattr(offsets, 'tolist'):
                offsets = offsets.tolist()
            return offsets
        except Exception:
            raise ValueError("tokenizer no has offset.")

    def _pool_attributes_from_captions(self,
                                       caption_attributes_batch: List[List[str]],
                                       caption_attributes_padding_mask: Optional[Tensor],
                                       caption_tokens: Tensor,
                                       caption_token_ids: Tensor,
                                       caption_padding_mask,
                                       captions_batch,
                                       cap_to_attr_map_batch: Optional[Tensor],
                                       normalize: bool = True,
                                       device='cuda'):
        if caption_tokens is None or caption_token_ids is None:
            return None, None

        batch_size, num_captions, seq_len, dim = caption_tokens.shape
        num_attrs = len(caption_attributes_batch[0])
        attr_feats = torch.zeros((batch_size, num_attrs, dim), device=device, dtype=caption_tokens.dtype)
        found_mask = torch.zeros((batch_size, num_attrs), device=device, dtype=torch.bool)

        attr_pad = caption_attributes_padding_mask
        if attr_pad is not None:
            if isinstance(attr_pad, (list, tuple)):
                attr_pad = torch.stack([m.to(device=device, dtype=torch.bool) for m in attr_pad], dim=0)
            else:
                attr_pad = attr_pad.to(device=device, dtype=torch.bool)
        else:
            attr_pad = torch.ones((batch_size, num_attrs), device=device, dtype=torch.bool)

        cap_valid = caption_padding_mask
        if cap_valid is not None:
            if isinstance(cap_valid, (list, tuple)):
                cap_valid = torch.stack([m.to(device=device, dtype=torch.bool) for m in cap_valid], dim=0)
            else:
                cap_valid = cap_valid.to(device=device, dtype=torch.bool)
        else:
            cap_valid = torch.ones((batch_size, num_captions), device=device, dtype=torch.bool)
        if cap_to_attr_map_batch is not None:
            if isinstance(cap_to_attr_map_batch, (list, tuple)):
                cap_to_attr_map_batch = torch.stack([m.to(device=device) for m in cap_to_attr_map_batch], dim=0)
            else:
                cap_to_attr_map_batch = cap_to_attr_map_batch.to(device=device)

        eot_id = getattr(self.tokenizer, 'eot_token_id', getattr(self, "text_eos_id", None))
        pad_id = getattr(self.tokenizer.tokenizer, 'pad_token_id', 0)

        for b in range(batch_size):
            collected_map = {}
            cap_offsets_cache = {}
            for cap_idx in range(num_captions):
                if not cap_valid[b, cap_idx]:
                    continue

                cap_text_raw = None
                if cap_idx < len(captions_batch[b]):
                    cap_text_raw = captions_batch[b][cap_idx][0] if isinstance(captions_batch[b][cap_idx], (list, tuple)) else captions_batch[b][cap_idx]
                cap_offsets = None
                if isinstance(cap_text_raw, str):
                    if cap_idx not in cap_offsets_cache:
                        cap_offsets_cache[cap_idx] = self._get_token_offsets(cap_text_raw)
                    cap_offsets = cap_offsets_cache.get(cap_idx, None)

                if cap_to_attr_map_batch is not None:
                    attr_indices = cap_to_attr_map_batch[b, cap_idx]
                    attr_indices = attr_indices[(attr_indices >= 0) & (attr_indices < num_attrs)].unique()
                else:
                    raise ValueError('miss cap_to_attr_map_batch')

                for attr_idx in attr_indices.tolist():
                    if attr_idx >= num_attrs or not attr_pad[b, attr_idx]:
                        continue

                    if cap_offsets is None or not isinstance(caption_attributes_batch[b][attr_idx], str) or not isinstance(cap_text_raw, str):
                        continue

                    attr_text_raw = caption_attributes_batch[b][attr_idx]
                    pos = cap_text_raw.lower().find(attr_text_raw.lower())
                    if pos < 0:
                        continue
                    end_char = pos + len(attr_text_raw)
                    token_idxs = [i for i, (s, e) in enumerate(cap_offsets)
                                  if e > pos and s < end_char and not (s == 0 and e == 0)]
                    if not token_idxs:
                        continue
                    token_start = min(token_idxs)
                    token_end = min(max(token_idxs) + 1, caption_tokens.shape[2])
                    collected_map.setdefault(attr_idx, []).append(
                        caption_tokens[b, cap_idx, token_start:token_end, :].mean(dim=0)
                    )

            for attr_idx, embeds in collected_map.items():
                pooled = torch.stack(embeds, dim=0).mean(dim=0)
                attr_feats[b, attr_idx] = F.normalize(pooled, dim=-1) if normalize else pooled
                found_mask[b, attr_idx] = True

            # 调试：打印未匹配到的属性名称及对应 caption 文本
            # unmatched = (attr_pad[b] & (~found_mask[b])).nonzero(as_tuple=False).squeeze(1)
            # if unmatched.numel() > 0:
            #     try:
            #         names = [caption_attributes_batch[b][idx] for idx in unmatched.tolist()]
            #         cap_map = {}
            #         if cap_to_attr_map_batch is not None:
            #             for cap_idx in range(num_captions):
            #                 if cap_idx >= len(captions_batch[b]):
            #                     continue
            #                 mapped = cap_to_attr_map_batch[b, cap_idx]
            #                 mapped = mapped[(mapped >= 0) & (mapped < num_attrs)].tolist()
            #                 for a_idx in mapped:
            #                     cap_map.setdefault(a_idx, []).append(captions_batch[b][cap_idx][0] if isinstance(captions_batch[b][cap_idx], (list, tuple)) else captions_batch[b][cap_idx])
            #         captions_for_attr = {idx: cap_map.get(idx, []) for idx in unmatched.tolist()}
            #         print(f"[attr_pool] sample {b} unmatched attributes: {names}; captions={captions_for_attr}")
            #     except Exception:
            #         print(f"[attr_pool] sample {b} unmatched attribute indices: {unmatched.tolist()}")

        return attr_feats, found_mask


    def forward_image(self, image: Tensor) -> Tuple[Tensor]:
        return self.image_model(image)


def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj_type=text_cfg.hf_proj_type,
            pooler_type=text_cfg.hf_pooler_type,
            pretrained=text_cfg.hf_model_pretrained,
            output_tokens=text_cfg.output_tokens,
        )
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        if text_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **text_cfg.norm_kwargs)
        if text_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **text_cfg.act_kwargs)

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            mlp_ratio=text_cfg.mlp_ratio,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            no_causal_mask=text_cfg.no_causal_mask,
            pad_id=text_cfg.pad_id,
            eos_id=text_cfg.eos_id,
            pool_type=text_cfg.pool_type,
            proj_type=text_cfg.proj_type,
            proj_bias=text_cfg.proj_bias,
            output_tokens=text_cfg.output_tokens,
            act_layer=act_layer,
            norm_layer=norm_layer,
            block_type=text_cfg.block_type,
            qk_norm=text_cfg.qk_norm,
            scaled_cosine_attn=text_cfg.scaled_cosine_attn,
            scale_heads=text_cfg.scale_heads,
            scale_attn_inner=text_cfg.scale_attn_inner,
            scale_attn=text_cfg.scale_attn,
            scale_fc=text_cfg.scale_fc,
        )
    return text


# used to maintain checkpoint compatibility
def convert_to_custom_text_state_dict(state_dict: dict):
    if 'text_projection' in state_dict:
        # old format state_dict, move text tower -> .text
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in (
                'text_projection',
                'positional_embedding',
                'token_embedding',
                'transformer',
                'ln_final',
            )):
                k = 'text.' + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict
