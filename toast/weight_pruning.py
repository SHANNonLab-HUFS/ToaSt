"""Structured Coupled Weight Pruning (SCWP) for multi-head self-attention.

Within each head we drop whole *dimensions* rather than individual weights, and we drop them
in coupled pairs:

* a Q/K dimension is removed from `W_Q` and `W_K` together -- the attention logit
  `q_i · k_i` contributes nothing if either side is zero, so pruning one without the other
  wastes the other's parameters;
* a V/O dimension is removed from `W_V`'s rows and `W_O`'s columns together, for the same
  reason on the value path.

Every head keeps exactly ``round_down(head_dim * (1 - sparsity))`` dimensions, so the
surviving weights of a block re-pack into smaller *dense* matrices with no gather and no
sparse kernel -- see :mod:`toast.dense`. That uniformity is what turns the sparsity into
wall-clock speedup, and it is why the sparsity budget is per-head rather than global.

The first block is left dense by default: its attention is the only one operating on raw
patch embeddings, and pruning it costs disproportionate accuracy.

Masks follow the convention used throughout this repo: **True means pruned**.
"""

from typing import List, Sequence, Union

import torch
import torch.nn as nn

from .importance import head_importance

__all__ = ["StructuredCoupledPruner", "reapply_masks", "attention_layers"]


def attention_layers(model: nn.Module, skip_first_block: bool = True):
    """Yield ``(block_idx, qkv_linear, proj_linear)`` for each prunable attention block.

    Blocks are visited in index order, and the two linears are yielded in the same order in
    which :func:`reapply_masks` walks the model, so a flat mask list stays aligned.
    """
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise AttributeError("model has no `.blocks`; SCWP expects a ViT-style backbone")

    for idx, block in enumerate(blocks):
        if skip_first_block and idx == 0:
            continue
        attn = getattr(block, "attn", None)
        qkv = getattr(attn, "qkv", None)
        proj = getattr(attn, "proj", None)
        if isinstance(qkv, nn.Linear) and isinstance(proj, nn.Linear):
            yield idx, qkv, proj


def reapply_masks(
    model: nn.Module,
    masks: Sequence[torch.Tensor],
    skip_first_block: bool = True,
) -> int:
    """Project every pruned weight back onto its mask, in place.

    Call before any evaluation, and optionally after each optimiser step. Runs on whatever
    device the weights already live on.

    Returns the number of layers touched.
    """
    layers = [lin for _, qkv, proj in attention_layers(model, skip_first_block) for lin in (qkv, proj)]
    if len(masks) != len(layers):
        raise ValueError(
            f"got {len(masks)} masks for {len(layers)} prunable layers; "
            "if this is an older checkpoint, load it through "
            "`experiments.common.load_toast_checkpoint`, which normalises the mask list"
        )
    for layer, mask in zip(layers, masks):
        layer.weight.data.masked_fill_(mask.to(layer.weight.device), 0.0)
    return len(layers)


class StructuredCoupledPruner:
    """Compute and apply SCWP masks for every attention block of a ViT.

    Args:
        model: ViT-style backbone exposing ``.blocks[i].attn.{qkv,proj}``.
        head_sparsity: percentage (0-100) of each head's dimensions to drop. Either a single
            value for all blocks, or one value per block -- in which case index 0 corresponds
            to block 0 and is ignored when ``skip_first_block`` is set.
        score: row score, see :data:`toast.importance.SCORES`. ``gm`` is the paper default.
        coupling: how the pair is scored, see :data:`toast.importance.COUPLINGS`.
            ``coupled`` is the paper default; the rest are ablations.
        skip_first_block: leave block 0 dense.
        verbose: print the per-block sparsity table.

    Attributes:
        masks: flat list of boolean masks, two per pruned block (qkv then proj), aligned with
            :func:`attention_layers`.
    """

    def __init__(
        self,
        model: nn.Module,
        head_sparsity: Union[float, Sequence[float]],
        score: str = "gm",
        coupling: str = "coupled",
        skip_first_block: bool = True,
        verbose: bool = True,
    ):
        self.model = model
        self.score = score
        self.coupling = coupling
        self.skip_first_block = skip_first_block

        self.masks: List[torch.Tensor] = []
        self._block_ids: List[int] = []

        for idx, qkv, proj in attention_layers(model, skip_first_block):
            qkv_mask, proj_mask = self._block_masks(
                qkv, proj, self._sparsity_for(head_sparsity, idx), self._num_heads(model, idx)
            )
            self.masks += [qkv_mask, proj_mask]
            self._block_ids.append(idx)

        if not self.masks:
            raise ValueError("no prunable attention blocks found")

        if verbose:
            print(self.report())

        reapply_masks(model, self.masks, skip_first_block)

    # -- mask construction ---------------------------------------------------------------

    @staticmethod
    def _sparsity_for(head_sparsity, block_idx: int) -> float:
        if isinstance(head_sparsity, (int, float)):
            return float(head_sparsity)
        if block_idx >= len(head_sparsity):
            raise ValueError(
                f"head_sparsity has {len(head_sparsity)} entries but block {block_idx} needs one"
            )
        return float(head_sparsity[block_idx])

    @staticmethod
    def _num_heads(model: nn.Module, block_idx: int) -> int:
        heads = getattr(model.blocks[block_idx].attn, "num_heads", None)
        if not heads:
            raise AttributeError(f"block {block_idx}: attn has no `num_heads`")
        return int(heads)

    def _block_masks(self, qkv: nn.Linear, proj: nn.Linear, sparsity: float, num_heads: int):
        """Masks for one block. Returns ``(qkv_mask, proj_mask)``, True = pruned."""
        qkv_weight = qkv.weight.data
        proj_weight = proj.weight.data

        embed_dim = qkv_weight.shape[0] // 3
        if embed_dim % num_heads:
            raise ValueError(f"embed_dim {embed_dim} not divisible by num_heads {num_heads}")
        head_dim = embed_dim // num_heads
        keep = int(head_dim * (100.0 - sparsity) / 100.0)
        if keep < 1:
            raise ValueError(
                f"sparsity {sparsity}% leaves {keep} of {head_dim} dimensions per head; "
                "at least one must survive"
            )

        Q, K, V = qkv_weight.split(embed_dim, dim=0)

        # True = keep, per embed_dim position. Built per head so every head keeps exactly
        # `keep` dimensions -- the property toast.dense relies on.
        qk_keep = torch.zeros(embed_dim, dtype=torch.bool)
        vo_keep = torch.zeros(embed_dim, dtype=torch.bool)

        for h in range(num_heads):
            lo, hi = h * head_dim, (h + 1) * head_dim
            qk_imp, vo_imp = head_importance(
                Q[lo:hi], K[lo:hi], V[lo:hi], proj_weight[:, lo:hi].T,
                score=self.score, coupling=self.coupling,
            )
            qk_keep[lo:hi][torch.topk(qk_imp, keep).indices] = True
            vo_keep[lo:hi][torch.topk(vo_imp, keep).indices] = True

        # Q and K share the attention-pair mask; V's rows and O's columns share the value-pair
        # mask. Rows of qkv, columns of proj.
        qkv_mask = torch.cat([~qk_keep, ~qk_keep, ~vo_keep]).unsqueeze(1).expand_as(qkv_weight)
        proj_mask = (~vo_keep).unsqueeze(0).expand_as(proj_weight)
        return qkv_mask.contiguous(), proj_mask.contiguous()

    # -- reporting -----------------------------------------------------------------------

    def report(self) -> str:
        lines = [
            "Structured Coupled Weight Pruning",
            f"  score={self.score}  coupling={self.coupling}",
            f"  {'block':>7}  {'layer':<5}  {'pruned':>12} / {'total':<12}  ratio",
        ]
        pruned_total = elem_total = 0
        for i, mask in enumerate(self.masks):
            pruned, total = int(mask.sum()), mask.numel()
            pruned_total += pruned
            elem_total += total
            lines.append(
                f"  {self._block_ids[i // 2]:>7}  {'qkv' if i % 2 == 0 else 'proj':<5}  "
                f"{pruned:>12,} / {total:<12,}  {pruned / total:6.1%}"
            )
        lines.append(
            f"  {'total':>7}  {'':<5}  {pruned_total:>12,} / {elem_total:<12,}  "
            f"{pruned_total / elem_total:6.1%}"
        )
        return "\n".join(lines)

    @property
    def sparsity(self) -> float:
        """Overall fraction of pruned weights across all masked layers."""
        pruned = sum(int(m.sum()) for m in self.masks)
        return pruned / sum(m.numel() for m in self.masks)
