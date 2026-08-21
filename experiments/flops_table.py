#!/usr/bin/env python3
"""Print the results tables from `configs/tcs.json`.

One table per model: the dense baseline, the prior work it is compared against, and ToaST at
each FLOPs budget the config holds. Needs no dataset and no GPU.

    python -m experiments.flops_table
    python -m experiments.flops_table --markdown        # paste into the README
    python -m experiments.flops_table --show-computed   # add toast.flops' own FLOPs count
"""

import argparse
import sys

import timm

from toast import load_tcs_config, spec_for_model, toast_flops


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="path to tcs.json (default: configs/tcs.json)")
    p.add_argument("--markdown", action="store_true", help="emit markdown tables")
    p.add_argument("--models", nargs="+", default=None, help="restrict to these models")
    p.add_argument(
        "--show-computed", action="store_true",
        help="add a column with the FLOPs toast.flops derives from each schedule",
    )
    return p.parse_args()


def spec_for(model_name, num_classes=1000):
    return spec_for_model(
        timm.create_model(model_name, pretrained=False, num_classes=num_classes),
        num_classes=num_classes,
    )


def rows_for(model_name, entry):
    """One row per config entry, plus the baseline and reference rows."""
    spec = spec_for(model_name)
    baseline_gflops = toast_flops(spec).gflops
    rows = [{
        "method": "baseline (dense)",
        "gflops": entry["baseline"].get("gflops"),
        "computed": baseline_gflops,
        "top1": entry["baseline"].get("top1"),
        "top5": entry["baseline"].get("top5"),
        "throughput": entry["baseline"].get("throughput_img_s"),
        "speedup": 1.0,
    }]
    for ref in entry.get("references", []):
        rows.append({
            "method": ref["method"],
            "gflops": ref.get("gflops"),
            "computed": None,
            "top1": ref.get("top1"),
            "top5": ref.get("top5"),
            "throughput": ref.get("throughput_img_s"),
            "speedup": ref.get("speedup"),
        })
    for target, spec_entry in sorted(entry["configs"].items(), key=lambda kv: -float(kv[0])):
        reported = spec_entry.get("reported", {})
        computed = toast_flops(
            spec,
            head_sparsity=spec_entry["head_sparsity"],
            fc1_prune_ratios=spec_entry["fc1"],
            fc2_prune_ratios=spec_entry["fc2"],
        ).gflops
        rows.append({
            "method": f"ToaST @ {target}G",
            "gflops": reported.get("gflops", float(target)),
            "computed": computed,
            "top1": reported.get("top1"),
            "top5": reported.get("top5"),
            "throughput": reported.get("throughput_img_s"),
            "speedup": reported.get("speedup"),
            "epochs": reported.get("epochs"),
            "baseline_gflops": baseline_gflops,
        })
    return rows


def fmt(value, spec="{:.2f}"):
    return "-" if value is None else spec.format(value)


def main():
    args = parse_args()
    config = load_tcs_config(args.config)
    models = args.models or [
        name for name, entry in config["models"].items() if entry.get("supported", True)
    ]

    for model_name in models:
        entry = config["models"].get(model_name)
        if entry is None:
            print(f"skipping unknown model {model_name}", file=sys.stderr)
            continue
        if not entry.get("supported", True):
            reason = entry.get("unsupported_reason", "unsupported architecture")
            print(f"\n## {model_name}  (recorded only: {reason})")
            continue

        rows = rows_for(model_name, entry)
        header = ["method", "GFLOPs", "reduction", "Acc@1", "Acc@5", "img/s", "speedup", "epochs"]
        widths = [20, 8, 11, 8, 8, 9, 9, 8]
        if args.show_computed:
            header.insert(2, "computed")
            widths.insert(2, 10)

        print(f"\n## {model_name}")
        if args.markdown:
            print("| " + " | ".join(header) + " |")
            print("|" + "---|" * len(header))
        else:
            print(f"{header[0]:<{widths[0]}}"
                  + "".join(f"{h:>{w}}" for h, w in zip(header[1:], widths[1:])))

        for row in rows:
            reduction = "-"
            if row.get("baseline_gflops") and row["computed"]:
                reduction = f"{100 * (1 - row['computed'] / row['baseline_gflops']):.1f}%"
            cells = [
                row["method"],
                fmt(row["gflops"], "{:.2f}"),
                reduction,
                fmt(row["top1"]),
                fmt(row["top5"]),
                fmt(row["throughput"], "{:.0f}"),
                fmt(row["speedup"], "{:.2f}x"),
                fmt(row.get("epochs"), "{:.0f}"),
            ]
            if args.show_computed:
                cells.insert(2, fmt(row["computed"], "{:.3f}"))
            if args.markdown:
                print("| " + " | ".join(cells) + " |")
            else:
                print(f"{cells[0]:<{widths[0]}}"
                      + "".join(f"{c:>{w}}" for c, w in zip(cells[1:], widths[1:])))


if __name__ == "__main__":
    main()
