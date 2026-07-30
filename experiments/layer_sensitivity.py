#!/usr/bin/env python3
"""How much accuracy each block loses when only that block's FFN is pruned.

Prunes one block at a time, at several ratios, leaving every other block untouched. The result
is the per-block sensitivity curve that motivates ToaST's non-uniform ratio schedule: later
blocks tolerate far more channel pruning than early ones. No fine-tuning -- this measures the
off-the-shelf sensitivity of the pretrained model.

    python -m experiments.layer_sensitivity --data-path $IMAGENET \
        --ratios 10 20 30 50 70 --targets fc1 fc2
"""

import argparse
import json
import os

from experiments.common import add_common_args, build_toast_model, build_val_loader, eval_args, run_eval


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--ratios", type=float, nargs="+", default=[10, 20, 30, 40, 50, 60, 70, 80, 90],
                   help="prune ratios to test, in percent")
    p.add_argument("--targets", nargs="+", default=["fc1", "fc2"], choices=["fc1", "fc2", "both"])
    p.add_argument("--blocks", type=int, nargs="+", default=None,
                   help="block indices to test (default: all)")
    p.add_argument("--head-sparsity", type=float, default=None,
                   help="also apply SCWP at this sparsity, to measure sensitivity of the "
                        "already weight-pruned model")
    p.add_argument("--plot", action="store_true", help="write a heatmap alongside the JSON")
    return p.parse_args()


def ratio_lists(num_blocks, block_idx, target, ratio):
    """(fc1, fc2) ratio lists with only `block_idx` pruned, as fractions."""
    fc1 = [0.0] * num_blocks
    fc2 = [0.0] * num_blocks
    if target in ("fc1", "both"):
        fc1[block_idx] = ratio / 100.0
    if target in ("fc2", "both"):
        fc2[block_idx] = ratio / 100.0
    return fc1, fc2


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    loader, dataset = build_val_loader(
        eval_args(args.data_path, args.data_set, args.batch_size, args.num_workers)
    )
    num_classes = 100 if args.data_set == "CIFAR" else 1000

    # Reference point: nothing pruned in the FFN.
    baseline_model, masks = build_toast_model(
        args.model, num_classes, head_sparsity=args.head_sparsity
    )
    num_blocks = len(baseline_model.blocks)
    baseline = run_eval(baseline_model, loader, masks, args.device)
    print(f"\nbaseline Acc@1 {baseline['acc1']:.3f}")
    del baseline_model

    blocks = args.blocks if args.blocks is not None else list(range(num_blocks))
    results = {"baseline": baseline, "num_blocks": num_blocks, "ratios": args.ratios, "runs": []}

    for target in args.targets:
        for block_idx in blocks:
            for ratio in args.ratios:
                print(f"\nblock {block_idx}, {target}, ratio {ratio}%")
                fc1, fc2 = ratio_lists(num_blocks, block_idx, target, ratio)
                model, masks = build_toast_model(
                    args.model, num_classes, fc1_ratios=fc1, fc2_ratios=fc2,
                    head_sparsity=args.head_sparsity,
                )
                stats = run_eval(model, loader, masks, args.device)
                drop = baseline["acc1"] - stats["acc1"]
                print(f"  Acc@1 {stats['acc1']:.3f}  (drop {drop:+.3f})")
                results["runs"].append(
                    {"block": block_idx, "target": target, "ratio": ratio,
                     "acc1": stats["acc1"], "acc5": stats["acc5"], "acc1_drop": drop}
                )
                del model

    out_path = os.path.join(args.output_dir, "layer_sensitivity.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")

    summarise(results, args.targets, blocks)
    if args.plot:
        plot(results, args.targets, blocks, args.ratios, args.output_dir)


def summarise(results, targets, blocks):
    """Rank blocks by mean accuracy drop -- the ordering the ratio schedule should follow."""
    for target in targets:
        print(f"\n{target}: mean Acc@1 drop per block (higher = more sensitive)")
        means = []
        for block_idx in blocks:
            drops = [r["acc1_drop"] for r in results["runs"]
                     if r["block"] == block_idx and r["target"] == target]
            if drops:
                means.append((block_idx, sum(drops) / len(drops)))
        for block_idx, mean in sorted(means, key=lambda kv: -kv[1]):
            print(f"  block {block_idx:2d}: {mean:+.3f}")


def plot(results, targets, blocks, ratios, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 4.5), squeeze=False)
    for ax, target in zip(axes[0], targets):
        grid = np.full((len(blocks), len(ratios)), np.nan)
        for run in results["runs"]:
            if run["target"] != target:
                continue
            grid[blocks.index(run["block"]), ratios.index(run["ratio"])] = run["acc1_drop"]
        im = ax.imshow(grid, aspect="auto", cmap="magma_r")
        ax.set_xticks(range(len(ratios)), [f"{r:g}" for r in ratios])
        ax.set_yticks(range(len(blocks)), [str(b) for b in blocks])
        ax.set_xlabel("prune ratio (%)")
        ax.set_ylabel("block")
        ax.set_title(f"{target}: Acc@1 drop")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path = os.path.join(output_dir, "layer_sensitivity.png")
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
