#!/usr/bin/env python3
"""Compare ways of distributing a fixed average prune ratio across blocks.

Every pattern here spends the same average budget (`--fc1-avg`, `--fc2-avg`); they differ only
in how it is allocated over depth. Together with `layer_sensitivity.py` this is the evidence
for ToaST's schedule: back-loaded patterns keep accuracy that uniform and front-loaded ones
lose, at identical cost. `inverse` is the negative control.

    python -m experiments.ratio_patterns --data-path $IMAGENET --fc1-avg 10 --fc2-avg 30
"""

import argparse
import json
import os

import numpy as np

from experiments.common import add_common_args, build_toast_model, build_val_loader, eval_args, run_eval
from toast import num_blocks as count_blocks


def _rescale(values, target_avg):
    """Scale `values` so their mean is exactly `target_avg`."""
    values = np.asarray(values, dtype=float)
    return values * (target_avg * len(values) / values.sum())


def generate_patterns(num_blocks=12, fc1_avg=10.0, fc2_avg=30.0):
    """Per-block prune ratios (percent) for each pattern, all at the same average."""
    patterns = {}

    patterns["stepwise"] = {
        "fc1": [0.0] * 8 + [20.0, 30.0, 30.0, 40.0],
        "fc2": [0.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        "description": "hand-tuned steps, early blocks untouched (the schedule ToaST uses)",
    }

    patterns["linear"] = {
        "fc1": _rescale(np.linspace(0, 2 * fc1_avg, num_blocks), fc1_avg).tolist(),
        "fc2": _rescale(np.linspace(0, 2 * fc2_avg, num_blocks), fc2_avg).tolist(),
        "description": "linear ramp from first to last block",
    }

    x = np.linspace(0, 1, num_blocks)
    patterns["exponential"] = {
        "fc1": _rescale((np.exp(4.0 * x) - 1) / (np.exp(4.0) - 1), fc1_avg).tolist(),
        "fc2": _rescale((np.exp(2.5 * x) - 1) / (np.exp(2.5) - 1), fc2_avg).tolist(),
        "description": "exponential: slow start, aggressive end",
    }

    patterns["inverse"] = {
        "fc1": _rescale(np.linspace(2 * fc1_avg, 0, num_blocks), fc1_avg).tolist(),
        "fc2": _rescale(np.linspace(2 * fc2_avg, 0, num_blocks), fc2_avg).tolist(),
        "description": "reversed ramp, early blocks hit hardest (negative control)",
    }

    sigmoid = 1.0 / (1.0 + np.exp(-np.linspace(-6, 6, num_blocks)))
    patterns["sigmoid"] = {
        "fc1": _rescale(sigmoid, fc1_avg).tolist(),
        "fc2": _rescale(sigmoid, fc2_avg).tolist(),
        "description": "S-curve, transition in the middle blocks",
    }

    patterns["uniform"] = {
        "fc1": [fc1_avg] * num_blocks,
        "fc2": [fc2_avg] * num_blocks,
        "description": "same ratio everywhere (baseline)",
    }

    patterns["conservative"] = {
        "fc1": _rescale([0.0, 0.0, 0.0] + list(np.linspace(5, 25, num_blocks - 3)), fc1_avg).tolist(),
        "fc2": _rescale([0.0, 0.0, 0.0] + list(np.linspace(10, 65, num_blocks - 3)), fc2_avg).tolist(),
        "description": "first three blocks fully protected",
    }

    return patterns


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--fc1-avg", type=float, default=10.0, help="average fc1 prune ratio (percent)")
    p.add_argument("--fc2-avg", type=float, default=30.0, help="average fc2 prune ratio (percent)")
    p.add_argument("--patterns", nargs="+", default=None, help="subset of patterns to run")
    p.add_argument("--head-sparsity", type=float, default=None, help="also apply SCWP at this sparsity")
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    loader, _ = build_val_loader(
        eval_args(args.data_path, args.data_set, args.batch_size, args.num_workers)
    )
    num_classes = 100 if args.data_set == "CIFAR" else 1000

    probe, masks = build_toast_model(args.model, num_classes, head_sparsity=args.head_sparsity)
    num_blocks = count_blocks(probe)
    baseline = run_eval(probe, loader, masks, args.device)
    print(f"\nbaseline (no FFN pruning) Acc@1 {baseline['acc1']:.3f}")
    del probe

    patterns = generate_patterns(num_blocks, args.fc1_avg, args.fc2_avg)
    selected = args.patterns or list(patterns)
    unknown = set(selected) - set(patterns)
    if unknown:
        raise SystemExit(f"unknown patterns: {sorted(unknown)}; choose from {sorted(patterns)}")

    results = {"baseline": baseline, "fc1_avg": args.fc1_avg, "fc2_avg": args.fc2_avg, "runs": {}}

    for name in selected:
        pattern = patterns[name]
        print(f"\n=== {name}: {pattern['description']} ===")
        print(f"  fc1 mean {np.mean(pattern['fc1']):.2f}%  fc2 mean {np.mean(pattern['fc2']):.2f}%")
        model, masks = build_toast_model(
            args.model, num_classes,
            fc1_ratios=[r / 100.0 for r in pattern["fc1"]],
            fc2_ratios=[r / 100.0 for r in pattern["fc2"]],
            head_sparsity=args.head_sparsity,
        )
        stats = run_eval(model, loader, masks, args.device)
        print(f"  Acc@1 {stats['acc1']:.3f}  (drop {baseline['acc1'] - stats['acc1']:+.3f})")
        results["runs"][name] = {
            **stats,
            "acc1_drop": baseline["acc1"] - stats["acc1"],
            "fc1": pattern["fc1"],
            "fc2": pattern["fc2"],
            "description": pattern["description"],
        }
        del model

    out_path = os.path.join(args.output_dir, "ratio_patterns.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")

    print(f"\n{'pattern':<14} {'Acc@1':>8} {'drop':>8}")
    for name, run in sorted(results["runs"].items(), key=lambda kv: -kv[1]["acc1"]):
        print(f"{name:<14} {run['acc1']:>8.3f} {run['acc1_drop']:>+8.3f}")

    if args.plot:
        plot(results, args.output_dir)


def plot(results, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_sched, ax_acc) = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, run in results["runs"].items():
        ax_sched.plot(run["fc2"], marker="o", markersize=3, label=name)
    ax_sched.set_xlabel("block")
    ax_sched.set_ylabel("fc2 prune ratio (%)")
    ax_sched.set_title("allocation over depth (equal average)")
    ax_sched.legend(fontsize=8)

    names = sorted(results["runs"], key=lambda n: -results["runs"][n]["acc1"])
    ax_acc.barh(names, [results["runs"][n]["acc1"] for n in names])
    ax_acc.axvline(results["baseline"]["acc1"], color="k", ls="--", lw=1, label="no FFN pruning")
    ax_acc.set_xlabel("Acc@1 (%)")
    ax_acc.set_title("accuracy at equal cost")
    ax_acc.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(output_dir, "ratio_patterns.png")
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
