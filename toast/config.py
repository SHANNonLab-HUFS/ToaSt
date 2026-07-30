"""Load per-model compression schedules from `configs/tcs.json`.

Rather than pasting twelve-element ratio vectors onto every command line, name the model and
the FLOPs budget:

    python main.py --eval --model deit_small_patch16_224 --target-flops 2.9 ...

The config records, per model and per budget, the SCWP sparsity and the per-block TCS ratios,
alongside the accuracy and latency the paper reports for that setting.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = ["TcsConfig", "DEFAULT_CONFIG_PATH", "load_tcs_config", "resolve_config"]

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "tcs.json"
)


@dataclass
class TcsConfig:
    """One model at one FLOPs budget."""

    model: str
    target_flops: str
    head_sparsity: float
    fc1_prune_ratios: List[float]
    fc2_prune_ratios: List[float]
    importance: str = "gm"
    coupling: str = "coupled"
    computed_gflops: Optional[float] = None
    reported: Dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"{self.model} @ {self.target_flops}G  "
            f"head_sparsity={self.head_sparsity:g}%  importance={self.importance}  "
            f"coupling={self.coupling}",
            f"  fc1 {[round(r, 2) for r in self.fc1_prune_ratios]}",
            f"  fc2 {[round(r, 2) for r in self.fc2_prune_ratios]}",
        ]
        if self.computed_gflops is not None:
            lines.append(f"  computed {self.computed_gflops:.3f} GFLOPs")
        if self.reported:
            top1 = self.reported.get("top1")
            reported_bits = []
            if top1 is not None:
                reported_bits.append(f"Acc@1 {top1}")
            for key, label in (("gflops", "G"), ("throughput_img_s", "img/s"), ("speedup", "x")):
                if key in self.reported:
                    reported_bits.append(f"{self.reported[key]}{label}")
            if self.reported.get("epochs") is not None:
                reported_bits.append(f"{self.reported['epochs']} epochs")
            lines.append("  reported in the paper: " + ", ".join(reported_bits))
        return "\n".join(lines)


def load_tcs_config(path: Optional[str] = None) -> Dict:
    """Parse the config JSON."""
    path = path or DEFAULT_CONFIG_PATH
    with open(path) as f:
        return json.load(f)


def _find_model(config: Dict, model: str) -> tuple:
    """Resolve `model` against config keys and their aliases."""
    models = config["models"]
    if model in models:
        return model, models[model]
    for name, entry in models.items():
        if model in entry.get("aliases", ()):
            return name, entry
    known = sorted(models)
    raise KeyError(f"no config for model {model!r}; the config knows about {known}")


def resolve_config(
    model: str,
    target_flops,
    path: Optional[str] = None,
    config: Optional[Dict] = None,
) -> TcsConfig:
    """Look up one model's schedule at one FLOPs budget.

    `target_flops` may be a string or a number; ``2.9``, ``"2.9"`` and ``"2.90"`` all match.
    """
    config = config if config is not None else load_tcs_config(path)
    name, entry = _find_model(config, model)

    if not entry.get("supported", True):
        raise NotImplementedError(
            f"{name} is recorded in the config but not runnable from this release: "
            f"{entry.get('unsupported_reason', 'unsupported architecture')}"
        )

    key = str(target_flops)
    configs = entry["configs"]
    if key not in configs:
        # Tolerate 2.9 vs "2.90" and similar spellings.
        matches = [k for k in configs if float(k) == float(key)]
        if not matches:
            available = ", ".join(sorted(configs, key=float))
            raise KeyError(
                f"{name} has no config at {key}G; available budgets: {available}"
            )
        key = matches[0]

    spec = configs[key]
    return TcsConfig(
        model=name,
        target_flops=key,
        head_sparsity=float(spec["head_sparsity"]),
        fc1_prune_ratios=[float(r) for r in spec["fc1"]],
        fc2_prune_ratios=[float(r) for r in spec["fc2"]],
        importance=spec.get("importance", "gm"),
        coupling=spec.get("coupling", "coupled"),
        computed_gflops=spec.get("computed_gflops"),
        reported=spec.get("reported", {}),
    )


def available_targets(model: str, path: Optional[str] = None, config: Optional[Dict] = None) -> List[str]:
    """FLOPs budgets the config holds for `model`, ascending."""
    config = config if config is not None else load_tcs_config(path)
    _, entry = _find_model(config, model)
    return sorted(entry["configs"], key=float)
