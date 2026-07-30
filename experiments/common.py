"""Shared plumbing for the analysis scripts in this directory.

The five original analysis scripts each carried their own copy of "load a checkpoint, figure
out its pruning configuration, build a validation loader, evaluate". This module is that copy,
once.
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from timm.models import create_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils  # noqa: E402
from datasets import build_dataset  # noqa: E402
from engine import evaluate  # noqa: E402
from toast import StructuredCoupledPruner, apply_toast, attention_layers, reapply_masks  # noqa: E402
from utils import MultiEpochsDataLoader  # noqa: E402

# Earlier spellings of the same arguments, kept so older checkpoints stay readable.
LEGACY_ARG_ALIASES = {
    "all_pr": "head_sparsity",
    "fc1_pr": "fc1_prune_ratio",
    "fc2_pr": "fc2_prune_ratio",
    "importance_strategy": "coupling",
}
LEGACY_SCORE_TYPES = {1: "l1", 2: "l2", 3: "gm"}

_MISSING = object()


class PrintLogger:
    """Minimal stand-in for the logger `engine.evaluate` expects."""

    def info(self, msg):
        print(msg)

    debug = info


def eval_args(
    data_path: str,
    data_set: str = "IMNET",
    batch_size: int = 256,
    num_workers: int = 10,
    input_size: int = 224,
) -> argparse.Namespace:
    """Namespace with the fields `build_dataset` needs for a clean validation transform."""
    return argparse.Namespace(
        data_path=data_path,
        data_set=data_set,
        inat_category="name",
        input_size=input_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        color_jitter=0.0,
        aa=None,
        train_interpolation="bicubic",
        reprob=0.0,
        remode="pixel",
        recount=1,
        resplit=False,
    )


def build_val_loader(args) -> Tuple[MultiEpochsDataLoader, torch.utils.data.Dataset]:
    """Sequential validation loader, so results do not depend on process count."""
    dataset, _ = build_dataset(is_train=False, args=args)
    loader = MultiEpochsDataLoader(
        dataset,
        sampler=torch.utils.data.SequentialSampler(dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    return loader, dataset


def _get(saved, name, default=None):
    """Read `name` from a checkpoint's saved args, falling back to an earlier spelling.

    Accepts both the dict this revision writes and the `argparse.Namespace` older ones did.
    """
    if saved is None:
        return default

    def lookup(key):
        if isinstance(saved, dict):
            return saved[key] if key in saved else _MISSING
        return getattr(saved, key, _MISSING)

    value = lookup(name)
    if value is not _MISSING:
        return value
    for old, new in LEGACY_ARG_ALIASES.items():
        if new == name:
            value = lookup(old)
            if value is not _MISSING:
                return value
    return default


def describe_saved_config(saved) -> str:
    """One-line summary of how a checkpoint was compressed, for the load log."""
    if saved is None:
        return "no args recorded"

    score = _get(saved, "importance")
    if score is None:
        # Older checkpoints recorded the score as an integer.
        score = LEGACY_SCORE_TYPES.get(_get(saved, "score_type"), "unknown")

    fields = [
        f"head_sparsity={_get(saved, 'head_sparsity', 'n/a')}",
        f"importance={score}",
        f"coupling={_get(saved, 'coupling', 'coupled')}",
    ]
    return "  ".join(fields)


def resolve_saved_ratios(saved, name: str, num_blocks: int) -> List[float]:
    """Per-block TCS ratios recorded in a checkpoint, expanded to one entry per block."""
    values = _get(saved, name)
    if values is None:
        return [0.0] * num_blocks
    values = [float(v) for v in values]
    return values * num_blocks if len(values) == 1 else values


def build_toast_model(
    model_name: str = "deit_small_patch16_224",
    num_classes: int = 1000,
    pretrained: bool = True,
    fc1_ratios: Optional[Sequence[float]] = None,
    fc2_ratios: Optional[Sequence[float]] = None,
    head_sparsity: Optional[float] = None,
    score: str = "gm",
    coupling: str = "coupled",
    verbose: bool = False,
) -> Tuple[nn.Module, Optional[List[torch.Tensor]]]:
    """Create a model with TCS installed and, optionally, SCWP applied.

    Returns ``(model, masks)``; `masks` is ``None`` when `head_sparsity` is ``None``.
    """
    model = create_model(
        model_name, pretrained=pretrained, num_classes=num_classes,
        drop_rate=0.0, drop_path_rate=0.0,
    )
    n = len(model.blocks)
    apply_toast(
        model,
        fc1_prune_ratios=list(fc1_ratios) if fc1_ratios is not None else [0.0] * n,
        fc2_prune_ratios=list(fc2_ratios) if fc2_ratios is not None else [0.0] * n,
    )
    masks = None
    if head_sparsity is not None:
        masks = StructuredCoupledPruner(
            model, head_sparsity=head_sparsity, score=score, coupling=coupling,
            verbose=verbose,
        ).masks
    return model, masks


def load_toast_checkpoint(
    path: str,
    model_name: Optional[str] = None,
    num_classes: int = 1000,
) -> Tuple[nn.Module, Optional[List[torch.Tensor]], Optional[argparse.Namespace]]:
    """Rebuild the model a checkpoint was saved from, and apply its masks.

    Reads the pruning configuration out of the checkpoint's saved args, accepting both the
    current flag names and the older ones. Mask lists from older revisions contain one entry
    per *submodule* rather than per block; those are trimmed to the layers they were actually
    applied to, so the model comes back exactly as it was evaluated.

    Returns ``(model, masks, saved_args)``.
    """
    checkpoint = utils.load_checkpoint(path)
    saved = checkpoint.get("args")

    model_name = model_name or _get(saved, "model", "deit_small_patch16_224")
    num_classes = _get(saved, "nb_classes", num_classes)
    print(f"loading {path}  (model={model_name}, {num_classes} classes)")
    print(f"  recorded config: {describe_saved_config(saved)}")

    model = create_model(
        model_name, pretrained=False, num_classes=num_classes, drop_rate=0.0, drop_path_rate=0.0
    )
    n = len(model.blocks)

    apply_toast(
        model,
        fc1_prune_ratios=resolve_saved_ratios(saved, "fc1_prune_ratio", n),
        fc2_prune_ratios=resolve_saved_ratios(saved, "fc2_prune_ratio", n),
        cls_weight=_get(saved, "cls_weight", 2.0),
        sample_ratio=_get(saved, "sample_ratio", 0.02),
    )

    state_dict = checkpoint.get("model", checkpoint)
    msg = model.load_state_dict(state_dict, strict=False)
    if msg.missing_keys:
        print(f"  missing keys: {msg.missing_keys}")
    if msg.unexpected_keys:
        print(f"  unexpected keys: {msg.unexpected_keys}")

    masks = checkpoint.get("masks", checkpoint.get("pruned_mask"))
    if masks is not None:
        masks = normalise_masks(model, list(masks))
        reapply_masks(model, masks)
        print(f"  masks: {len(masks)} layers, overall sparsity {mask_sparsity(masks):.2%}")
    else:
        print("  no masks in checkpoint")

    return model, masks, saved


def normalise_masks(model: nn.Module, masks: List[torch.Tensor]) -> List[torch.Tensor]:
    """Trim a saved mask list to one entry per prunable layer.

    Some checkpoints store more masks than there are layers, in which case the leading entries
    are the ones that were applied.
    """
    expected = 2 * sum(1 for _ in attention_layers(model, skip_first_block=True))
    if len(masks) == expected:
        return masks
    if len(masks) > expected:
        print(
            f"  checkpoint stores {len(masks)} masks for {expected} layers; "
            "keeping the leading entries, which are the ones that were applied"
        )
        return masks[:expected]
    raise ValueError(f"checkpoint has {len(masks)} masks but the model needs {expected}")


def mask_sparsity(masks: Sequence[torch.Tensor]) -> float:
    """Fraction of pruned weights across all masked layers."""
    return sum(int(m.sum()) for m in masks) / sum(m.numel() for m in masks)


def mask_report(masks: Sequence[torch.Tensor]) -> str:
    lines = [f"{'layer':>6}  {'pruned':>12} / {'total':<12}  ratio"]
    for i, mask in enumerate(masks):
        pruned, total = int(mask.sum()), mask.numel()
        lines.append(f"{i:>6}  {pruned:>12,} / {total:<12,}  {pruned / total:6.1%}")
    lines.append(f"{'total':>6}  overall sparsity {mask_sparsity(masks):.2%}")
    return "\n".join(lines)


def run_eval(
    model: nn.Module,
    loader,
    masks: Optional[Sequence[torch.Tensor]] = None,
    device: str = "cuda",
) -> Dict[str, float]:
    """Evaluate and return ``{'acc1', 'acc5', 'loss'}``."""
    model.to(device).eval()
    return evaluate(loader, model, torch.device(device), logger=PrintLogger(), masks=masks)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Flags every analysis script needs."""
    parser.add_argument("--data-path", required=True, help="dataset root")
    parser.add_argument("--data-set", default="IMNET", choices=["CIFAR", "IMNET", "INAT", "INAT19"])
    parser.add_argument("--model", default="deit_small_patch16_224")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="./results")
    return parser
