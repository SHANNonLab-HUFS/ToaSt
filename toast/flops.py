"""Analytic FLOPs accounting for ToaST-compressed vision transformers.

Counts multiply-accumulates, matching the convention the token-compression literature uses
(ToMe, DiffRate), so the numbers here line up with theirs. For an uncompressed block:

    MHSA = 4*N*C^2 + 2*N^2*C          qkv and output projections, then the two attention matmuls
    FFN  = 2*N*C*(mlp_ratio*C)        fc1 and fc2

Both compression stages enter as multipliers on those terms.

**SCWP** keeps a fraction `k = keep / head_dim` of every head's dimensions. Q, K, V and the
output projection all narrow by `k`, and so does the per-head dimension inside the attention
matmuls, so the whole MHSA term scales by `k`. Because the budget is uniform across heads this
is exact, not an estimate -- the pruned model really is this size once re-packed
(:mod:`toast.dense`).

**TCS** drops a fraction `r1` of fc1's input channels and `r2` of fc2's input channels, so the
FFN term becomes `N*C*(mlp_ratio*C)*((1 - r1) + (1 - r2))`. This counts the compute of the
re-packed model; the masked model in :mod:`toast.patch` still multiplies by zeros and so has
baseline cost, which is the point of measuring latency separately.

Patch embedding and classifier are counted as in DiffRate, and are unaffected by either stage.

Run as a module to print a breakdown:

    python -m toast.flops --model deit_small_patch16_224 --head-sparsity 90 \\
        --fc2-prune-ratio 0 0 0 0 0 0 0 0 0 0 0.9 0.9
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

__all__ = ["ViTSpec", "FlopsBreakdown", "spec_from_model", "vit_flops"]


@dataclass
class ViTSpec:
    """The shape parameters FLOPs depend on. Build one with :func:`spec_from_model`."""

    embed_dim: int
    depth: int
    num_heads: int
    num_patches: int
    patch_size: int
    mlp_ratio: float = 4.0
    in_chans: int = 3
    num_classes: int = 1000
    num_prefix_tokens: int = 1  # class token, and the distillation token if present

    @property
    def num_tokens(self) -> int:
        return self.num_patches + self.num_prefix_tokens

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads


@dataclass
class FlopsBreakdown:
    """MAC counts, in units of one multiply-accumulate."""

    patch_embed: float
    classifier: float
    mhsa: List[float] = field(default_factory=list)
    ffn: List[float] = field(default_factory=list)

    @property
    def blocks(self) -> float:
        return sum(self.mhsa) + sum(self.ffn)

    @property
    def total(self) -> float:
        return self.patch_embed + self.blocks + self.classifier

    @property
    def gflops(self) -> float:
        return self.total / 1e9

    def summary(self, baseline: Optional["FlopsBreakdown"] = None) -> str:
        lines = [f"{'block':>6}  {'MHSA (M)':>10}  {'FFN (M)':>10}  {'total (M)':>10}"]
        for i, (mhsa, ffn) in enumerate(zip(self.mhsa, self.ffn)):
            lines.append(f"{i:>6}  {mhsa / 1e6:>10.1f}  {ffn / 1e6:>10.1f}  {(mhsa + ffn) / 1e6:>10.1f}")
        lines.append(
            f"{'':>6}  patch_embed {self.patch_embed / 1e6:.1f} M   "
            f"classifier {self.classifier / 1e6:.2f} M"
        )
        line = f"{'total':>6}  {self.gflops:.3f} GFLOPs"
        if baseline is not None:
            reduction = 100.0 * (1.0 - self.total / baseline.total)
            line += f"  ({reduction:.1f}% below the {baseline.gflops:.3f} G baseline)"
        lines.append(line)
        return "\n".join(lines)


def spec_from_model(model, num_classes: Optional[int] = None) -> ViTSpec:
    """Read a :class:`ViTSpec` off a timm VisionTransformer.

    Works on an already-patched model: `apply_toast` and SCWP change classes and zero weights
    but not shapes.
    """
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise AttributeError("expected a ViT-style model with `.blocks`")

    block = blocks[0]
    embed_dim = block.attn.qkv.weight.shape[1]
    patch_size = model.patch_embed.patch_size
    patch_size = patch_size[0] if isinstance(patch_size, (tuple, list)) else patch_size

    prefix = getattr(model, "num_prefix_tokens", None)
    if prefix is None:
        prefix = int(getattr(model, "cls_token", None) is not None)

    return ViTSpec(
        embed_dim=embed_dim,
        depth=len(blocks),
        num_heads=int(block.attn.num_heads),
        num_patches=int(model.patch_embed.num_patches),
        patch_size=int(patch_size),
        mlp_ratio=block.mlp.fc1.out_features / embed_dim,
        in_chans=model.patch_embed.proj.weight.shape[1],
        num_classes=num_classes if num_classes is not None else model.num_classes,
        num_prefix_tokens=int(prefix),
    )


def _keep_fraction(spec: ViTSpec, head_sparsity: float) -> float:
    """Fraction of each head's dimensions SCWP retains, after integer truncation."""
    keep = int(spec.head_dim * (100.0 - head_sparsity) / 100.0)
    return keep / spec.head_dim


def _per_block(values: Optional[Union[float, Sequence[float]]], depth: int, name: str) -> List[float]:
    if values is None:
        return [0.0] * depth
    if isinstance(values, (int, float)):
        return [float(values)] * depth
    values = [float(v) for v in values]
    if len(values) == 1:
        return values * depth
    if len(values) != depth:
        raise ValueError(f"{name} has {len(values)} entries but the model has {depth} blocks")
    return values


def vit_flops(
    spec: ViTSpec,
    head_sparsity: Optional[Union[float, Sequence[float]]] = None,
    fc1_prune_ratios: Optional[Union[float, Sequence[float]]] = None,
    fc2_prune_ratios: Optional[Union[float, Sequence[float]]] = None,
    skip_first_block: bool = True,
) -> FlopsBreakdown:
    """FLOPs of a ViT under a given ToaST configuration.

    Args:
        spec: model shape, from :func:`spec_from_model`.
        head_sparsity: SCWP sparsity in percent, scalar or per block. ``None`` means no
            weight pruning.
        fc1_prune_ratios, fc2_prune_ratios: TCS ratios as fractions, scalar or per block.
        skip_first_block: block 0 is left dense, matching
            :class:`toast.weight_pruning.StructuredCoupledPruner`.

    Returns a :class:`FlopsBreakdown`; ``.gflops`` is the headline number.
    """
    N, C = spec.num_tokens, spec.embed_dim
    hidden = spec.mlp_ratio * C

    mhsa_dense = 4.0 * N * C * C + 2.0 * N * N * C
    ffn_unit = N * C * hidden  # cost of one of the two FFN projections

    sparsity = _per_block(head_sparsity, spec.depth, "head_sparsity")
    fc1 = _per_block(fc1_prune_ratios, spec.depth, "fc1_prune_ratios")
    fc2 = _per_block(fc2_prune_ratios, spec.depth, "fc2_prune_ratios")

    breakdown = FlopsBreakdown(
        patch_embed=N * C * (spec.patch_size**2 * spec.in_chans),
        classifier=C * spec.num_classes,
    )

    for block in range(spec.depth):
        pruned = head_sparsity is not None and not (skip_first_block and block == 0)
        keep = _keep_fraction(spec, sparsity[block]) if pruned else 1.0
        breakdown.mhsa.append(mhsa_dense * keep)
        breakdown.ffn.append(ffn_unit * ((1.0 - fc1[block]) + (1.0 - fc2[block])))

    return breakdown


def _main():
    import argparse

    import timm

    parser = argparse.ArgumentParser(description="Print the FLOPs breakdown of a ToaST config.")
    parser.add_argument("--model", default="deit_small_patch16_224")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--head-sparsity", type=float, nargs="+", default=None)
    parser.add_argument("--fc1-prune-ratio", type=float, nargs="+", default=None)
    parser.add_argument("--fc2-prune-ratio", type=float, nargs="+", default=None)
    args = parser.parse_args()

    spec = spec_from_model(
        timm.create_model(args.model, pretrained=False, num_classes=args.num_classes),
        num_classes=args.num_classes,
    )
    print(
        f"{args.model}: C={spec.embed_dim} depth={spec.depth} heads={spec.num_heads} "
        f"tokens={spec.num_tokens} mlp_ratio={spec.mlp_ratio:g}"
    )
    baseline = vit_flops(spec)
    print(f"\nbaseline: {baseline.gflops:.3f} GFLOPs\n")
    compressed = vit_flops(
        spec,
        head_sparsity=args.head_sparsity,
        fc1_prune_ratios=args.fc1_prune_ratio,
        fc2_prune_ratios=args.fc2_prune_ratio,
    )
    print(compressed.summary(baseline))


if __name__ == "__main__":
    _main()
