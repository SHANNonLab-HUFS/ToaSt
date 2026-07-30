"""Patch a timm VisionTransformer to run Token Channel Selection in its FFN blocks.

`apply_toast` swaps the class of each `Block` and `Attention` in place, so the model keeps its
parameter names and any checkpoint stays loadable. Two things change:

* attention returns its attention map alongside the output, because TCS needs the class
  token's attention row to weight patch statistics;
* the FFN selects channels for `fc1` and `fc2` per forward pass.

This module implements the *masked* formulation: rejected channels are zeroed and the matmuls
keep their original shapes. That is what training and accuracy evaluation use. For latency
measurement the same selection is realised as physically smaller matmuls -- see
:mod:`toast.dense`.
"""

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Block, VisionTransformer

from .token_channel import (
    DEFAULT_CLS_WEIGHT,
    DEFAULT_SAMPLE_RATIO,
    select_channels_masked,
)

__all__ = ["apply_toast", "ToastBlock", "ToastAttention"]

_IDENTITY = nn.Identity()


def _part(module: nn.Module, *names: str) -> nn.Module:
    """First attribute of `module` present among `names`, else a shared Identity.

    timm renamed several block internals across versions (`drop_path` became
    `drop_path1`/`drop_path2`, `Mlp.drop` became `drop1`/`drop2`, LayerScale `ls1`/`ls2`
    appeared). Resolving by name keeps one implementation working across them.
    """
    for name in names:
        part = getattr(module, name, None)
        if part is not None:
            return part
    return _IDENTITY


class ToastAttention(Attention):
    """Attention that also returns its attention map.

    Deliberately does not use fused/SDPA attention: the map is an output, not an
    implementation detail we can let the kernel discard.
    """

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        # No-ops on DeiT; present in newer timm.
        q = _part(self, "q_norm")(q)
        k = _part(self, "k_norm")(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class ToastBlock(Block):
    """Transformer block whose FFN applies Token Channel Selection.

    Set up by :func:`apply_toast`; `fc1_prune_ratios` and `fc2_prune_ratios` are indexed by
    `block_index`, and a ratio of ``0.0`` leaves that projection untouched.
    """

    block_index: int = 0
    fc1_prune_ratios: Sequence[float] = ()
    fc2_prune_ratios: Sequence[float] = ()
    cls_weight: float = DEFAULT_CLS_WEIGHT
    sample_ratio: float = DEFAULT_SAMPLE_RATIO
    tcs_generator: Optional[torch.Generator] = None

    def configure_tcs(
        self,
        block_index: int,
        fc1_prune_ratios: Sequence[float],
        fc2_prune_ratios: Sequence[float],
        cls_weight: float = DEFAULT_CLS_WEIGHT,
        sample_ratio: float = DEFAULT_SAMPLE_RATIO,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.block_index = block_index
        self.fc1_prune_ratios = fc1_prune_ratios
        self.fc2_prune_ratios = fc2_prune_ratios
        self.cls_weight = cls_weight
        self.sample_ratio = sample_ratio
        self.tcs_generator = generator

    @property
    def fc1_ratio(self) -> float:
        return 0.0 if self.block_index == 0 else float(self.fc1_prune_ratios[self.block_index])

    @property
    def fc2_ratio(self) -> float:
        return 0.0 if self.block_index == 0 else float(self.fc2_prune_ratios[self.block_index])

    def _select(self, tokens: torch.Tensor, ratio: float, cls_attn: Optional[torch.Tensor]):
        selected, _ = select_channels_masked(
            tokens,
            prune_ratio=ratio,
            attn_weights=cls_attn,
            cls_weight=self.cls_weight,
            sample_ratio=self.sample_ratio,
            generator=self.tcs_generator,
        )
        return selected.to(dtype=tokens.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_attn, attn = self.attn(self.norm1(x))
        x = x + _part(self, "drop_path1", "drop_path")(_part(self, "ls1")(x_attn))

        # Class-token attention row, averaged over heads: (B, 1, N).
        cls_attn = attn.mean(dim=1)[:, 0:1, :] if attn is not None and attn.dim() == 4 else None

        mlp = self.mlp
        fc1_ratio = self.fc1_ratio

        # The residual carries the selected activations too, so a channel dropped before fc1
        # is dropped from this block's contribution to the residual stream as well.
        residual = x if fc1_ratio == 0.0 else self._select(x, fc1_ratio, cls_attn)

        h = mlp.fc1(self.norm2(residual))
        h = mlp.act(h)
        h = _part(mlp, "drop1", "drop")(h)

        fc2_ratio = self.fc2_ratio
        if fc2_ratio != 0.0:
            h = self._select(h, fc2_ratio, cls_attn)

        h = mlp.fc2(h)
        h = _part(mlp, "drop2", "drop")(h)

        return residual + _part(self, "drop_path2", "drop_path")(_part(self, "ls2")(h.to(residual.dtype)))


def apply_toast(
    model: VisionTransformer,
    fc1_prune_ratios: Optional[Sequence[float]] = None,
    fc2_prune_ratios: Optional[Sequence[float]] = None,
    cls_weight: float = DEFAULT_CLS_WEIGHT,
    sample_ratio: float = DEFAULT_SAMPLE_RATIO,
    generator: Optional[torch.Generator] = None,
) -> VisionTransformer:
    """Enable Token Channel Selection on `model`, in place.

    Args:
        model: a timm VisionTransformer (DeiT included).
        fc1_prune_ratios: per-block fraction of `fc1` *input* channels to drop, indexed by
            block. ``None`` means no pruning anywhere.
        fc2_prune_ratios: per-block fraction of `fc2` input channels (i.e. hidden units) to
            drop.
        cls_weight, sample_ratio, generator: forwarded to
            :func:`toast.token_channel.channel_importance`.

    Returns the same model object.
    """
    blocks: List[Block] = list(model.blocks)
    n = len(blocks)
    fc1 = list(fc1_prune_ratios) if fc1_prune_ratios is not None else [0.0] * n
    fc2 = list(fc2_prune_ratios) if fc2_prune_ratios is not None else [0.0] * n

    for label, ratios in (("fc1_prune_ratios", fc1), ("fc2_prune_ratios", fc2)):
        if len(ratios) != n:
            raise ValueError(f"{label} has {len(ratios)} entries but the model has {n} blocks")
        if any(not 0.0 <= r < 1.0 for r in ratios):
            raise ValueError(f"{label} entries must lie in [0, 1); got {ratios}")

    for module in model.modules():
        if isinstance(module, Attention):
            module.__class__ = ToastAttention

    for index, block in enumerate(blocks):
        block.__class__ = ToastBlock
        block.configure_tcs(index, fc1, fc2, cls_weight, sample_ratio, generator)

    return model
