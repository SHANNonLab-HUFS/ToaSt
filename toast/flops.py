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

**Swin** is the same accounting per block, with two differences that come from the hierarchy.
Each stage has its own token count and channel width, and attention is confined to a window,
so the two attention matmuls cost `2*N*window_area*C` rather than `2*N^2*C`. The patch-merging
layer between stages is counted as the linear it is. Use :func:`swin_flops` with a
:class:`SwinSpec`, or :func:`toast_flops` to dispatch on whichever spec you have.

Run as a module to print a breakdown:

    python -m toast.flops --model deit_small_patch16_224 --head-sparsity 90 \\
        --fc2-prune-ratio 0 0 0 0 0 0 0 0 0 0 0.9 0.9
    python -m toast.flops --model swin_tiny_patch4_window7_224 --head-sparsity 90
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

from .arch import is_swin, stage_depths

__all__ = [
    "ViTSpec",
    "SwinSpec",
    "FlopsBreakdown",
    "spec_from_model",
    "swin_spec_from_model",
    "spec_for_model",
    "vit_flops",
    "swin_flops",
    "toast_flops",
]


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
class SwinSpec:
    """The shape parameters a Swin's FLOPs depend on. Build one with
    :func:`swin_spec_from_model`.

    `embed_dim` is the first stage's width; stage `s` is twice as wide and has a quarter as
    many tokens as stage `s-1`.
    """

    embed_dim: int
    depths: List[int]
    num_heads: List[int]
    grid_size: int  # patch grid of the first stage, per side
    patch_size: int
    window_size: int = 7
    mlp_ratio: float = 4.0
    in_chans: int = 3
    num_classes: int = 1000

    @property
    def depth(self) -> int:
        """Total blocks, the length of a ratio vector."""
        return sum(self.depths)

    @property
    def num_stages(self) -> int:
        return len(self.depths)

    def stage_of(self, block: int) -> int:
        """Which stage a global block index falls in."""
        seen = 0
        for stage, depth in enumerate(self.depths):
            seen += depth
            if block < seen:
                return stage
        raise IndexError(f"block {block} beyond the model's {self.depth} blocks")

    def dim(self, stage: int) -> int:
        return self.embed_dim * 2**stage

    def tokens(self, stage: int) -> int:
        side = self.grid_size // 2**stage
        return side * side

    def window_area(self, stage: int) -> int:
        """Tokens per attention window, capped by a stage smaller than the window."""
        side = min(self.window_size, self.grid_size // 2**stage)
        return side * side

    def head_dim(self, stage: int) -> int:
        return self.dim(stage) // self.num_heads[stage]


@dataclass
class FlopsBreakdown:
    """MAC counts, in units of one multiply-accumulate."""

    patch_embed: float
    classifier: float
    downsample: float = 0.0  # Swin's patch merging between stages; zero for a ViT
    mhsa: List[float] = field(default_factory=list)
    ffn: List[float] = field(default_factory=list)

    @property
    def blocks(self) -> float:
        return sum(self.mhsa) + sum(self.ffn)

    @property
    def total(self) -> float:
        return self.patch_embed + self.blocks + self.downsample + self.classifier

    @property
    def gflops(self) -> float:
        return self.total / 1e9

    def summary(self, baseline: Optional["FlopsBreakdown"] = None) -> str:
        lines = [f"{'block':>6}  {'MHSA (M)':>10}  {'FFN (M)':>10}  {'total (M)':>10}"]
        for i, (mhsa, ffn) in enumerate(zip(self.mhsa, self.ffn)):
            lines.append(f"{i:>6}  {mhsa / 1e6:>10.1f}  {ffn / 1e6:>10.1f}  {(mhsa + ffn) / 1e6:>10.1f}")
        tail = (
            f"{'':>6}  patch_embed {self.patch_embed / 1e6:.1f} M   "
            f"classifier {self.classifier / 1e6:.2f} M"
        )
        if self.downsample:
            tail += f"   patch_merging {self.downsample / 1e6:.1f} M"
        lines.append(tail)
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


def swin_spec_from_model(model, num_classes: Optional[int] = None) -> SwinSpec:
    """Read a :class:`SwinSpec` off a timm SwinTransformer.

    Works on an already-patched model, for the same reason :func:`spec_from_model` does.
    """
    layers = getattr(model, "layers", None)
    if layers is None:
        raise AttributeError("expected a Swin-style model with `.layers`")

    first = layers[0].blocks[0]
    embed_dim = int(first.attn.qkv.weight.shape[1])
    grid = model.patch_embed.grid_size
    patch_size = model.patch_embed.patch_size

    return SwinSpec(
        embed_dim=embed_dim,
        depths=stage_depths(model),
        num_heads=[int(stage.blocks[0].attn.num_heads) for stage in layers],
        grid_size=int(grid[0] if isinstance(grid, (tuple, list)) else grid),
        patch_size=int(patch_size[0] if isinstance(patch_size, (tuple, list)) else patch_size),
        window_size=int(first.window_size[0]),
        mlp_ratio=first.mlp.fc1.out_features / embed_dim,
        in_chans=int(model.patch_embed.proj.weight.shape[1]),
        num_classes=num_classes if num_classes is not None else model.num_classes,
    )


def spec_for_model(model, num_classes: Optional[int] = None) -> Union[ViTSpec, SwinSpec]:
    """The right spec for whichever backbone this is."""
    if is_swin(model):
        return swin_spec_from_model(model, num_classes)
    return spec_from_model(model, num_classes)


def _keep_fraction(head_dim: int, head_sparsity: float) -> float:
    """Fraction of a head's dimensions SCWP retains, after integer truncation."""
    keep = int(head_dim * (100.0 - head_sparsity) / 100.0)
    return keep / head_dim


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
        keep = _keep_fraction(spec.head_dim, sparsity[block]) if pruned else 1.0
        breakdown.mhsa.append(mhsa_dense * keep)
        breakdown.ffn.append(ffn_unit * ((1.0 - fc1[block]) + (1.0 - fc2[block])))

    return breakdown


def swin_flops(
    spec: SwinSpec,
    head_sparsity: Optional[Union[float, Sequence[float]]] = None,
    fc1_prune_ratios: Optional[Union[float, Sequence[float]]] = None,
    fc2_prune_ratios: Optional[Union[float, Sequence[float]]] = None,
    skip_first_block: bool = True,
) -> FlopsBreakdown:
    """FLOPs of a Swin under a given ToaST configuration.

    Same arguments and same conventions as :func:`vit_flops`; the ratio vectors are indexed by
    the global block index, counting through the stages. Attention is windowed, so its two
    matmuls cost `2*N*window_area*C` instead of the ViT's `2*N^2*C`, and the patch-merging
    linears between stages are counted separately, as neither stage compresses them.
    """
    sparsity = _per_block(head_sparsity, spec.depth, "head_sparsity")
    fc1 = _per_block(fc1_prune_ratios, spec.depth, "fc1_prune_ratios")
    fc2 = _per_block(fc2_prune_ratios, spec.depth, "fc2_prune_ratios")

    first_tokens = spec.tokens(0)
    breakdown = FlopsBreakdown(
        patch_embed=first_tokens * spec.embed_dim * (spec.patch_size**2 * spec.in_chans),
        classifier=spec.dim(spec.num_stages - 1) * spec.num_classes,
        # Patch merging concatenates 2x2 neighbours and halves the width again:
        # Linear(4 * C_prev -> C_stage) applied to the smaller grid.
        downsample=sum(
            spec.tokens(s) * (4 * spec.dim(s - 1)) * spec.dim(s) for s in range(1, spec.num_stages)
        ),
    )

    for block in range(spec.depth):
        stage = spec.stage_of(block)
        N, C = spec.tokens(stage), spec.dim(stage)

        mhsa_dense = 4.0 * N * C * C + 2.0 * N * spec.window_area(stage) * C
        pruned = head_sparsity is not None and not (skip_first_block and block == 0)
        keep = _keep_fraction(spec.head_dim(stage), sparsity[block]) if pruned else 1.0
        breakdown.mhsa.append(mhsa_dense * keep)

        ffn_unit = N * C * (spec.mlp_ratio * C)
        breakdown.ffn.append(ffn_unit * ((1.0 - fc1[block]) + (1.0 - fc2[block])))

    return breakdown


def toast_flops(
    spec: Union[ViTSpec, SwinSpec],
    head_sparsity: Optional[Union[float, Sequence[float]]] = None,
    fc1_prune_ratios: Optional[Union[float, Sequence[float]]] = None,
    fc2_prune_ratios: Optional[Union[float, Sequence[float]]] = None,
    skip_first_block: bool = True,
) -> FlopsBreakdown:
    """:func:`vit_flops` or :func:`swin_flops`, whichever the spec calls for."""
    count = swin_flops if isinstance(spec, SwinSpec) else vit_flops
    return count(spec, head_sparsity, fc1_prune_ratios, fc2_prune_ratios, skip_first_block)


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

    spec = spec_for_model(
        timm.create_model(args.model, pretrained=False, num_classes=args.num_classes),
        num_classes=args.num_classes,
    )
    if isinstance(spec, SwinSpec):
        print(
            f"{args.model}: C={spec.embed_dim} depths={spec.depths} heads={spec.num_heads} "
            f"grid={spec.grid_size} window={spec.window_size} mlp_ratio={spec.mlp_ratio:g}"
        )
    else:
        print(
            f"{args.model}: C={spec.embed_dim} depth={spec.depth} heads={spec.num_heads} "
            f"tokens={spec.num_tokens} mlp_ratio={spec.mlp_ratio:g}"
        )
    baseline = toast_flops(spec)
    print(f"\nbaseline: {baseline.gflops:.3f} GFLOPs\n")
    compressed = toast_flops(
        spec,
        head_sparsity=args.head_sparsity,
        fc1_prune_ratios=args.fc1_prune_ratio,
        fc2_prune_ratios=args.fc2_prune_ratio,
    )
    print(compressed.summary(baseline))


if __name__ == "__main__":
    _main()
