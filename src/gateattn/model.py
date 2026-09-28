"""Prune timm ViT tokens without replacing pretrained block computations."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn


@dataclass
class AttentionSnapshot:
    block: int
    patch_indices: torch.Tensor
    scores: torch.Tensor
    retained_indices: torch.Tensor


def attention_keep(scores: torch.Tensor, drop_rate: float, policy: str = 'quantile') -> torch.Tensor:
    """Select tokens, always preserving CLS at index 0.

    quantile includes CLS in the threshold, matching original notebooks.
    topk ranks patch tokens only and keeps an exact count, with stable ties.
    """
    if scores.ndim != 1 or scores.numel() < 2 or not torch.isfinite(scores).all():
        raise ValueError('Expected finite CLS attention scores including at least one patch.')
    if not math.isfinite(drop_rate) or not 0 <= drop_rate < 1:
        raise ValueError('Require 0 <= attention drop rate < 1.')
    if policy == 'quantile':
        mask = scores >= torch.quantile(scores.float(), drop_rate)
    elif policy == 'topk':
        k = max(1, int((scores.numel() - 1) * (1 - drop_rate)))
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[1 + torch.argsort(scores[1:], descending=True, stable=True)[:k]] = True
    else:
        raise ValueError(f'Unknown attention policy: {policy}')
    mask[0] = True
    if not mask[1:].any():
        mask[1 + torch.argmax(scores[1:])] = True
    return mask


class GateAttnViT(nn.Module):
    """Single-image inference with EDPG before block 0 and ATS after chosen blocks.

    Block indices are zero-based. Original spatial indices survive each pruning step.
    Only standard, fixed-resolution timm ViTs with one CLS prefix are supported.
    """
    def __init__(self, backbone: nn.Module, attention_blocks=(), attention_drop=0.0,
                 attention_policy='quantile', capture_attention=False):
        super().__init__()
        if getattr(backbone, 'num_prefix_tokens', None) != 1 or getattr(backbone, 'dynamic_img_size', False):
            raise ValueError('Require a fixed-resolution ViT with exactly one CLS token.')
        if not hasattr(backbone, 'blocks') or not hasattr(backbone, '_pos_embed'):
            raise ValueError('Expected a timm VisionTransformer backbone.')
        blocks = tuple(attention_blocks)
        if len(blocks) != len(set(blocks)) or any(type(i) is not int or i < 0 or i >= len(backbone.blocks) for i in blocks):
            raise ValueError('ATS block indices must be unique, valid zero-based integers.')
        rates = [attention_drop] * len(blocks) if isinstance(attention_drop, (int, float)) else list(attention_drop)
        if len(rates) != len(blocks) or any(not math.isfinite(r) or not 0 <= r < 1 for r in rates):
            raise ValueError('Provide one valid attention drop rate per ATS block.')
        if attention_policy not in {'quantile', 'topk'}:
            raise ValueError('Attention policy must be quantile or topk.')
        self.backbone = backbone
        self.rates = dict(zip(blocks, rates))
        self.attention_policy = attention_policy
        self.capture_attention = capture_attention
        self.snapshots = []
        self.token_counts = []
        self._scores = None
        self._handles = []
        for index, block in enumerate(backbone.blocks):
            if index in self.rates or capture_attention:
                if not hasattr(block.attn, 'attn_drop'):
                    raise ValueError('Backbone attention has no supported attention probability hook.')
                block.attn.fused_attn = False
                self._handles.append(block.attn.attn_drop.register_forward_hook(self._capture))

    def _capture(self, module, inputs, output):
        self._scores = output.mean(dim=1)[:, 0, :].detach()

    def forward(self, image, patch_mask=None):
        if self.training:
            raise RuntimeError('This is an inference implementation; call model.eval().')
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError('Use batch size 1: threshold pruning can retain different counts per image.')
        model = self.backbone
        x = model.patch_embed(image)
        if x.ndim != 3:
            raise ValueError('Expected flattened patch embeddings.')
        count = x.shape[1]
        indices = torch.arange(count, device=x.device)
        x = model._pos_embed(x)
        x = model.patch_drop(x)
        x = model.norm_pre(x)
        if patch_mask is not None:
            mask = torch.as_tensor(patch_mask, device=x.device, dtype=torch.bool)
            if mask.ndim != 1 or mask.numel() != count or not mask.any():
                raise ValueError('Mask must contain one Boolean per patch and retain at least one patch.')
            indices = indices[mask]
            x = torch.cat((x[:, :1], x[:, 1:][:, mask]), dim=1)
        self.snapshots, self.token_counts = [], []
        for index, block in enumerate(model.blocks):
            self.token_counts.append(x.shape[1])
            self._scores = None
            x = block(x)
            before = indices.clone()
            scores = self._scores
            if index in self.rates:
                if scores is None:
                    raise RuntimeError('Attention hook did not run.')
                keep = attention_keep(scores[0], self.rates[index], self.attention_policy)
                x = x[:, keep]
                indices = indices[keep[1:]]
            if self.capture_attention:
                if scores is None:
                    raise RuntimeError('Attention hook did not run.')
                self.snapshots.append(AttentionSnapshot(index, before.cpu(), scores[0, 1:].cpu(), indices.cpu().clone()))
        self.retained_indices = indices.detach().cpu()
        return model.forward_head(model.norm(x))
