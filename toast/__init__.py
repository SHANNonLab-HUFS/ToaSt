"""ToaST: coupled weight pruning and token-channel selection for vision transformers.

Two compression stages that target the two halves of a transformer block:

* :mod:`toast.weight_pruning` -- **Structured Coupled Weight Pruning (SCWP)** removes head
  dimensions from multi-head self-attention, in Q/K and V/O pairs, with a uniform per-head
  budget so the result re-packs into dense matrices.
* :mod:`toast.token_channel` -- **Token Channel Selection (TCS)** picks which feed-forward
  channels to compute per forward pass, scored from class-token-weighted activations.

Both work on ViT and on Swin. :mod:`toast.arch` flattens Swin's stage-nested blocks into one
globally indexed list, so a schedule is a per-block vector either way; :mod:`toast.swin` holds
the parts that genuinely differ.

Typical use::

    model = timm.create_model("deit_small_patch16_224", pretrained=True)
    apply_toast(model, fc1_prune_ratios=..., fc2_prune_ratios=...)
    pruner = StructuredCoupledPruner(model, head_sparsity=90.0)
    # ... fine-tune, calling reapply_masks(model, pruner.masks) after each optimiser step

`toast.dense` re-packs a pruned model into smaller matmuls for latency measurement.
"""

from .arch import arch_of, blocks_of, is_swin, iter_blocks, num_blocks, stage_depths
from .config import (
    DEFAULT_CONFIG_PATH,
    TcsConfig,
    available_targets,
    load_tcs_config,
    resolve_config,
)
from .dense import (
    DenseCoupledAttention,
    DenseSwinAttention,
    DenseSwinBlock,
    DenseToastBlock,
    densify,
    extract_dense_heads,
)
from .flops import (
    FlopsBreakdown,
    SwinSpec,
    ViTSpec,
    spec_for_model,
    spec_from_model,
    swin_flops,
    swin_spec_from_model,
    toast_flops,
    vit_flops,
)
from .importance import COUPLINGS, SCORES, head_importance
from .patch import ToastAttention, ToastBlock, apply_toast
from .swin import (
    SwinToastBlock,
    SwinToastWindowAttention,
    apply_toast_swin,
    swin_channel_importance,
    window_attention_received,
)
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
    "DenseSwinAttention",
    "DenseSwinBlock",
    "DenseToastBlock",
    "FlopsBreakdown",
    "StructuredCoupledPruner",
    "SwinSpec",
    "SwinToastBlock",
    "SwinToastWindowAttention",
    "TcsConfig",
    "ToastAttention",
    "ToastBlock",
    "ViTSpec",
    "apply_toast",
    "apply_toast_swin",
    "arch_of",
    "attention_layers",
    "available_targets",
    "blocks_of",
    "channel_importance",
    "densify",
    "extract_dense_heads",
    "head_importance",
    "is_swin",
    "iter_blocks",
    "load_tcs_config",
    "num_blocks",
    "reapply_masks",
    "resolve_config",
    "select_channels_dense",
    "select_channels_masked",
    "spec_for_model",
    "spec_from_model",
    "stage_depths",
    "swin_channel_importance",
    "swin_flops",
    "swin_spec_from_model",
    "toast_flops",
    "vit_flops",
    "window_attention_received",
]
