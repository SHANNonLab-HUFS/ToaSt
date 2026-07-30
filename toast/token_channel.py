"""Token Channel Selection (TCS) for the feed-forward block.

SCWP prunes attention weights once, offline. The FFN is handled differently: which of its
channels matter depends on the image, so TCS scores channels from the *activations* at run
time and keeps the top fraction for that batch.

The score of channel `c` mixes two terms:

* how strongly the class token uses it, ``|x_cls[c]|``, up-weighted by ``cls_weight`` because
  the class token is the one that reaches the classifier;
* how strongly the patch tokens use it, ``|x_patch[c]|``, weighted by each patch's attention
  from the class token so that background patches count for less.

The two are combined with weights ``1/(1+P)`` and ``P/(1+P)`` for `P` patches, i.e. the class
token counts as one token among many *after* its ``cls_weight`` boost.

Patch statistics are estimated from a small random subsample (2 % by default). Averaging
``|x|`` over patches converges quickly, and this keeps the selection cost negligible next to
the matmul it is saving.

Two ways to use a selection:

* :func:`select_channels_masked` zeroes the rejected channels and keeps the tensor's width.
  Mathematically what the paper describes, and what training uses -- gradients still flow to
  every weight, and the shape is static.
* :func:`select_channels_dense` physically narrows the tensor. Used for latency measurement,
  where the point is to make the following matmul smaller (see :mod:`toast.dense`).
"""

from typing import Dict, Optional, Tuple

import torch

__all__ = [
    "channel_importance",
    "select_channels_masked",
    "select_channels_dense",
    "DEFAULT_CLS_WEIGHT",
    "DEFAULT_SAMPLE_RATIO",
]

DEFAULT_CLS_WEIGHT = 2.0
DEFAULT_SAMPLE_RATIO = 0.02


def channel_importance(
    tokens: torch.Tensor,
    attn_weights: Optional[torch.Tensor] = None,
    cls_weight: float = DEFAULT_CLS_WEIGHT,
    sample_ratio: float = DEFAULT_SAMPLE_RATIO,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Per-channel importance of an activation tensor.

    Args:
        tokens: ``(B, N, C)`` activations, class token at position 0.
        attn_weights: ``(B, 1, N)`` class-token attention, as produced by averaging the
            attention map over heads and taking row 0. If ``None``, patches are weighted
            uniformly.
        cls_weight: multiplier on the class token's contribution.
        sample_ratio: fraction of patch tokens to sample. ``>= 1`` uses all of them.
        generator: RNG for the subsample. Pass one for reproducible selections; the default
            draws from the ambient RNG.

    Returns:
        ``(C,)`` importance, larger means more important.
    """
    B, N, C = tokens.shape

    cls_importance = tokens[:, 0, :].abs().mean(dim=0) * cls_weight
    if N == 1:
        return cls_importance

    num_patches = N - 1
    num_samples = max(1, int(num_patches * sample_ratio))

    if num_samples < num_patches:
        indices = torch.randint(1, N, (num_samples,), device=tokens.device, generator=generator)
    else:
        indices = torch.arange(1, N, device=tokens.device)

    patches = tokens[:, indices, :]
    if attn_weights is not None:
        # Average the per-patch attention over the batch, then weight |x| by it.
        attn_mean = attn_weights[:, 0, indices].mean(dim=0, keepdim=True)
        patch_importance = (patches.abs() * attn_mean.unsqueeze(-1)).mean(dim=(0, 1))
    else:
        patch_importance = patches.abs().mean(dim=(0, 1))

    total = 1.0 + num_patches
    return cls_importance * (1.0 / total) + patch_importance * (num_patches / total)


def _top_channels(importance: torch.Tensor, prune_ratio: float) -> torch.Tensor:
    """Indices of the channels to keep, sorted ascending."""
    C = importance.numel()
    num_keep = min(max(1, int(C * (1.0 - prune_ratio))), C)
    return torch.topk(importance, num_keep).indices.sort()[0]


def select_channels_masked(
    tokens: torch.Tensor,
    prune_ratio: float = 0.5,
    attn_weights: Optional[torch.Tensor] = None,
    cls_weight: float = DEFAULT_CLS_WEIGHT,
    sample_ratio: float = DEFAULT_SAMPLE_RATIO,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict]:
    """Zero the rejected channels, keeping the tensor's width.

    Returns ``(tokens, info)`` where ``info`` carries ``kept_indices``, ``mask`` and the
    realised ratio.
    """
    importance = channel_importance(tokens, attn_weights, cls_weight, sample_ratio, generator)
    kept = _top_channels(importance, prune_ratio)

    mask = torch.zeros(tokens.shape[-1], device=tokens.device, dtype=tokens.dtype)
    mask[kept] = 1.0

    info = {
        "mode": "masked",
        "total_channels": tokens.shape[-1],
        "kept_channels": kept.numel(),
        "prune_ratio": prune_ratio,
        "actual_ratio": 1.0 - kept.numel() / tokens.shape[-1],
        "kept_indices": kept,
        "mask": mask,
    }
    return tokens * mask.unsqueeze(0).unsqueeze(0), info


def select_channels_dense(
    tokens: torch.Tensor,
    prune_ratio: float = 0.5,
    attn_weights: Optional[torch.Tensor] = None,
    cls_weight: float = DEFAULT_CLS_WEIGHT,
    sample_ratio: float = DEFAULT_SAMPLE_RATIO,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict]:
    """Physically narrow the tensor to the kept channels.

    Returns ``(tokens[..., kept], info)``. The caller is responsible for slicing whatever
    weight matrix consumes this tensor by the same ``kept_indices``.
    """
    importance = channel_importance(tokens, attn_weights, cls_weight, sample_ratio, generator)
    kept = _top_channels(importance, prune_ratio)

    info = {
        "mode": "dense",
        "total_channels": tokens.shape[-1],
        "kept_channels": kept.numel(),
        "prune_ratio": prune_ratio,
        "actual_ratio": 1.0 - kept.numel() / tokens.shape[-1],
        "kept_indices": kept,
        "original_dim": tokens.shape[-1],
    }
    return tokens[:, :, kept], info
