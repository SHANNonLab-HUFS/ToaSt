"""Where a backbone keeps its transformer blocks.

ToaST is written against a flat list of blocks: the ratio vectors, the SCWP mask list and the
FLOPs breakdown are all indexed by block. A timm `VisionTransformer` already is that list
(`model.blocks`); a `SwinTransformer` nests its blocks in stages
(`model.layers[i].blocks[j]`). This module flattens the second case so the rest of the package
does not have to care, and so a Swin-T schedule is a twelve-element vector exactly like a
DeiT one.

The global index counts straight through the stages, in forward order. It is the index the
config's `fc1`/`fc2` vectors use, and the one `skip_first_block` refers to.

Duck-typed on purpose -- no timm import here, so `toast.flops` and `toast.config` stay usable
without a model instantiated.
"""

from typing import Iterator, List, Tuple

import torch.nn as nn

__all__ = [
    "VIT",
    "SWIN",
    "arch_of",
    "is_swin",
    "iter_blocks",
    "blocks_of",
    "num_blocks",
    "stage_depths",
    "part",
]

VIT = "vit"
SWIN = "swin"

_IDENTITY = nn.Identity()


def part(module: nn.Module, *names: str) -> nn.Module:
    """First attribute of `module` present among `names`, else a shared Identity.

    timm renamed several block internals across versions (`drop_path` became
    `drop_path1`/`drop_path2`, `Mlp.drop` became `drop1`/`drop2`, LayerScale `ls1`/`ls2`
    appeared). Resolving by name keeps one implementation working across them.
    """
    for name in names:
        found = getattr(module, name, None)
        if found is not None:
            return found
    return _IDENTITY


def arch_of(model: nn.Module) -> str:
    """``"vit"`` or ``"swin"``, from the module layout."""
    if getattr(model, "blocks", None) is not None:
        return VIT
    if getattr(model, "layers", None) is not None:
        return SWIN
    raise AttributeError(
        "model has no `.blocks` (ViT) and no `.layers` (Swin), so ToaST cannot find its "
        "transformer blocks"
    )


def is_swin(model: nn.Module) -> bool:
    return arch_of(model) == SWIN


def iter_blocks(model: nn.Module) -> Iterator[Tuple[int, nn.Module]]:
    """Yield ``(global_index, block)`` in forward order, for either architecture."""
    if arch_of(model) == VIT:
        yield from enumerate(model.blocks)
        return
    index = 0
    for stage in model.layers:
        for block in stage.blocks:
            yield index, block
            index += 1


def blocks_of(model: nn.Module) -> List[nn.Module]:
    return [block for _, block in iter_blocks(model)]


def num_blocks(model: nn.Module) -> int:
    """Total number of transformer blocks -- the length every ratio vector must have."""
    return sum(1 for _ in iter_blocks(model))


def stage_depths(model: nn.Module) -> List[int]:
    """Blocks per stage. A ViT is one stage of `depth` blocks."""
    if arch_of(model) == VIT:
        return [len(model.blocks)]
    return [len(stage.blocks) for stage in model.layers]
