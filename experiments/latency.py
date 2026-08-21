#!/usr/bin/env python3
"""Wall-clock latency of a masked model versus its dense re-packed equivalent.

A masked model still multiplies by zeros, so its latency is that of the dense baseline. The
point of SCWP's uniform per-head budget is that the surviving weights re-pack into smaller
matrices, and this script measures what that is worth. It also re-evaluates accuracy on the
re-packed model, because the re-packing is not bit-exact -- `toast.dense` explains why.

    python -m experiments.latency --checkpoint /ckpt/deit_s_90.pth --data-path $IMAGENET
    python -m experiments.latency --checkpoint /ckpt/deit_s_90.pth --skip-eval   # timing only
"""

import argparse
import copy
import json
import os
import time

import torch

from experiments.common import (
    add_common_args,
    build_val_loader,
    eval_args,
    load_toast_checkpoint,
    resolve_saved_ratios,
    run_eval,
)
from toast import blocks_of, densify
from toast import num_blocks as count_blocks


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--checkpoint", required=True, help="ToaST checkpoint with masks")
    p.add_argument("--bench-batch-sizes", type=int, nargs="+", default=[1, 32, 128, 256])
    p.add_argument("--runs", type=int, default=100, help="timed forward passes per batch size")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--skip-eval", action="store_true", help="time only, do not evaluate accuracy")
    return p.parse_args()


@torch.no_grad()
def benchmark(model, batch_size, device, runs, warmup, input_size=224):
    """Mean seconds per forward pass."""
    model.eval().to(device)
    x = torch.randn(batch_size, 3, input_size, input_size, device=device)

    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / runs


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    masked, masks, saved = load_toast_checkpoint(args.checkpoint, model_name=args.model)
    if masks is None:
        raise SystemExit("checkpoint has no masks; nothing to re-pack")

    n = count_blocks(masked)
    fc1 = resolve_saved_ratios(saved, "fc1_prune_ratio", n)
    fc2 = resolve_saved_ratios(saved, "fc2_prune_ratio", n)

    # Score the dense model exactly as the masked one is scored, or the two will not select
    # the same channels. The settings are read off a patched block rather than re-derived.
    probe = blocks_of(masked)[-1]
    dense = densify(
        copy.deepcopy(masked),
        fc1_prune_ratios=fc1,
        fc2_prune_ratios=fc2,
        cls_weight=getattr(probe, "cls_weight", 2.0),
        sample_ratio=getattr(probe, "sample_ratio", None),
        swin_attn_weighting=getattr(probe, "attn_weighting", False),
    )

    block = blocks_of(dense)[n // 2]
    print(
        f"\nre-packed per-head dims: qk={block.attn.qk_head_dim} vo={block.attn.vo_head_dim} "
        f"(dense qkv weight {tuple(block.attn.qkv_weight.shape)})"
    )

    results = {"checkpoint": args.checkpoint, "timing": {}}
    print(f"\n{'batch':>6} {'masked (ms)':>13} {'dense (ms)':>12} {'speedup':>9}")
    for batch_size in args.bench_batch_sizes:
        t_masked = benchmark(masked, batch_size, device, args.runs, args.warmup)
        t_dense = benchmark(dense, batch_size, device, args.runs, args.warmup)
        results["timing"][batch_size] = {
            "masked_ms": t_masked * 1e3,
            "dense_ms": t_dense * 1e3,
            "speedup": t_masked / t_dense,
            "masked_img_per_s": batch_size / t_masked,
            "dense_img_per_s": batch_size / t_dense,
        }
        print(
            f"{batch_size:>6} {t_masked * 1e3:>13.2f} {t_dense * 1e3:>12.2f} "
            f"{t_masked / t_dense:>8.2f}x"
        )

    if not args.skip_eval:
        loader, dataset = build_val_loader(
            eval_args(args.data_path, args.data_set, args.batch_size, args.num_workers)
        )
        print(f"\nevaluating masked model on {len(dataset)} images")
        results["masked_acc"] = run_eval(masked, loader, masks, args.device)
        print(f"\nevaluating dense re-packed model on {len(dataset)} images")
        results["dense_acc"] = run_eval(dense, loader, None, args.device)
        print(
            f"\nAcc@1  masked {results['masked_acc']['acc1']:.3f}  "
            f"dense {results['dense_acc']['acc1']:.3f}  "
            f"(delta {results['dense_acc']['acc1'] - results['masked_acc']['acc1']:+.3f})"
        )

    out_path = os.path.join(args.output_dir, "latency.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
