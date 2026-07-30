#!/usr/bin/env python3
"""Accuracy of SCWP-compressed checkpoints across head sparsities and batch sizes.

Point `--checkpoint-pattern` at a set of fine-tuned checkpoints named by their sparsity, e.g.
``/ckpt/deit_s_{sparsity}.pth``, and this evaluates each at each batch size. Batch size is
swept because TCS scores channels from batch statistics, so its selection -- and therefore
accuracy -- depends on how many images share a forward pass.

    python -m experiments.sparsity_sweep \
        --data-path $IMAGENET \
        --checkpoint-pattern '/ckpt/deit_s_{sparsity}.pth' \
        --sparsities 60 70 80 90 --batch-sizes 256 64 1
"""

import argparse
import csv
import os
import traceback
from datetime import datetime

import torch

from experiments.common import (
    add_common_args,
    build_val_loader,
    eval_args,
    load_toast_checkpoint,
    mask_sparsity,
    run_eval,
)

FIELDS = [
    "sparsity", "batch_size", "acc1", "acc5", "loss",
    "model", "mask_sparsity", "checkpoint", "timestamp",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument(
        "--checkpoint-pattern", required=True,
        help="path template containing '{sparsity}', e.g. '/ckpt/deit_s_{sparsity}.pth'",
    )
    p.add_argument("--sparsities", type=int, nargs="+", default=[20, 30, 40, 50, 60, 70, 80, 90])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[256, 128, 64, 32, 1])
    return p.parse_args()


def run_one(args, sparsity, batch_size):
    path = args.checkpoint_pattern.format(sparsity=sparsity)
    if not os.path.exists(path):
        print(f"  skipping, no checkpoint at {path}")
        return None

    model, masks, saved = load_toast_checkpoint(path, model_name=args.model)
    loader, dataset = build_val_loader(
        eval_args(args.data_path, args.data_set, batch_size, args.num_workers)
    )
    stats = run_eval(model, loader, masks, args.device)

    result = {
        "sparsity": sparsity,
        "batch_size": batch_size,
        "acc1": stats["acc1"],
        "acc5": stats["acc5"],
        "loss": stats["loss"],
        "model": getattr(saved, "model", args.model) if saved else args.model,
        "mask_sparsity": f"{mask_sparsity(masks):.4f}" if masks else "",
        "checkpoint": path,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    print(f"  Acc@1 {stats['acc1']:.3f}  Acc@5 {stats['acc5']:.3f}  loss {stats['loss']:.4f}")

    del model, loader, dataset
    torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir, f"sparsity_sweep_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )

    results, failures = [], []
    total = len(args.sparsities) * len(args.batch_sizes)
    n = 0

    for sparsity in args.sparsities:
        for batch_size in args.batch_sizes:
            n += 1
            print(f"\n[{n}/{total}] sparsity {sparsity}%, batch {batch_size}")
            try:
                result = run_one(args, sparsity, batch_size)
            except Exception:
                traceback.print_exc()
                result = None
            (results if result else failures).append(result or (sparsity, batch_size))

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{len(results)} succeeded, {len(failures)} failed -> {out_path}")
    for sparsity, batch_size in failures:
        print(f"  failed: sparsity {sparsity}%, batch {batch_size}")
    if results:
        best = max(results, key=lambda r: r["acc1"])
        print(f"  best Acc@1 {best['acc1']:.3f} at sparsity {best['sparsity']}%, batch {best['batch_size']}")


if __name__ == "__main__":
    main()
