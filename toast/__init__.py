"""ToaST: coupled weight pruning and token-channel selection for vision transformers.

Two compression stages that target the two halves of a transformer block:

* :mod:`toast.weight_pruning` -- **Structured Coupled Weight Pruning (SCWP)** removes head
  dimensions from multi-head self-attention, in Q/K and V/O pairs, with a uniform per-head
  budget so the result re-packs into dense matrices.
* :mod:`toast.token_channel` -- **Token Channel Selection (TCS)** picks which feed-forward
  channels to compute per forward pass, scored from class-token-weighted activations.

Typical use::

    model = timm.create_model("deit_small_patch16_224", pretrained=True)
    apply_toast(model, fc1_prune_ratios=..., fc2_prune_ratios=...)
    pruner = StructuredCoupledPruner(model, head_sparsity=90.0)
    # ... fine-tune, calling reapply_masks(model, pruner.masks) after each optimiser step

`toast.dense` re-packs a pruned model into smaller matmuls for latency measurement.
"""

from .config import (
    DEFAULT_CONFIG_PATH,
    TcsConfig,
    available_targets,
    load_tcs_config,
    resolve_config,
)
from .dense import DenseCoupledAttention, DenseToastBlock, densify, extract_dense_heads
from .flops import FlopsBreakdown, ViTSpec, spec_from_model, vit_flops
from .importance import COUPLINGS, SCORES, head_importance
from .patch import ToastAttention, ToastBlock, apply_toast
from .token_channel import (
    channel_importance,
    select_channels_dense,
    select_channels_masked,
)
from .weight_pruning import StructuredCoupledPruner, attention_layers, reapply_masks

__all__ = [
    "COUPLINGS",
    "DEFAULT_CONFIG_PATH",
    "SCORES",
    "DenseCoupledAttention",
    "DenseToastBlock",
    "FlopsBreakdown",
    "StructuredCoupledPruner",
    "TcsConfig",
    "ToastAttention",
    "ToastBlock",
    "ViTSpec",
    "apply_toast",
    "attention_layers",
    "available_targets",
    "channel_importance",
    "densify",
    "extract_dense_heads",
    "head_importance",
    "load_tcs_config",
    "reapply_masks",
    "resolve_config",
    "select_channels_dense",
    "select_channels_masked",
    "spec_from_model",
    "vit_flops",
]
