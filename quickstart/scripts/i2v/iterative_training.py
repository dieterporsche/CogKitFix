import subprocess
import yaml
import re
from pathlib import Path
from typing import Dict, Tuple

# We want the repository root so metrics and output paths resolve correctly.
# __file__ = quickstart/scripts/i2v/hyperparam_search.py
# parents[0] -> quickstart/scripts/i2v
# parents[1] -> quickstart/scripts
# parents[2] -> quickstart
# parents[3] -> repo root
ROOT = Path(__file__).resolve().parents[3]
CONFIG_TEMPLATE = Path(__file__).parent / "config.yaml"
TRAIN_SCRIPT = Path(__file__).parent.parent / "train.py"
METRICS_SCRIPT = ROOT / "metrics" / "compute_metrics.py"
OUTPUT_ROOT = ROOT / "output"


def _write_config(lr: float, batch_size: int, epochs: int) -> Path:
    with open(CONFIG_TEMPLATE, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["learning_rate"] = lr
    cfg["batch_size"] = batch_size
    cfg["train_epochs"] = epochs
    tmp = CONFIG_TEMPLATE.parent / f"tmp_{lr}_{batch_size}.yaml"
    with open(tmp, "w") as f:
        yaml.safe_dump(cfg, f)
    return tmp


def _run_training(config: Path) -> None:
    cmd = [
        "torchrun",
        "--nproc_per_node=1",
        "--master_port=29501",
        TRAIN_SCRIPT.as_posix(),
        "--yaml",
        str(config),
    ]
    subprocess.run(cmd, check=True)


def _run_metrics() -> None:
    subprocess.run(["python", METRICS_SCRIPT.as_posix()], check=True)


def _parse_metrics_file(path: Path) -> Dict[str, Tuple[float, float, float]]:
    metrics: Dict[str, Tuple[float, float, float]] = {}
    epoch = None
    with open(path, "r") as f:
        lines = [line.strip() for line in f]
    try:
        idx = lines.index("#### Average ALL: ####") + 1
    except ValueError:
        return metrics
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("Epoch_"):
            epoch = line.rstrip(":")
            if idx + 2 < len(lines):
                vals = lines[idx + 2].split()
                if len(vals) >= 3:
                    metrics[epoch] = (
                        float(vals[0]),
                        float(vals[1]),
                        float(vals[2]),
                    )
            idx += 3
        else:
            idx += 1
    return metrics


def _select_best(metrics: Dict[str, Tuple[float, float, float]]) -> Tuple[str, Tuple[float, float, float]]:
    if not metrics:
        return "", (float("inf"), float("inf"), float("inf"))
    mins = [
        min(v[i] for v in metrics.values())
        for i in range(3)
    ]
    best_epoch = None
    best_vals = None
    best_count = -1
    best_sum = float("inf")
    for epoch, vals in metrics.items():
        count = sum(vals[i] == mins[i] for i in range(3))
        total = sum(vals)
        if count > best_count or (count == best_count and total < best_sum):
            best_epoch = epoch
            best_vals = vals
            best_count = count
            best_sum = total
    return best_epoch or "", best_vals or (float("inf"), float("inf"), float("inf"))


def _extract_lr_bs(name: str) -> Tuple[float, int]:
    m = re.search(r"LR_([\d.eE-]+)__BS_(\d+)", name)
    if not m:
        return 0.0, 0
    return float(m.group(1)), int(m.group(2))


def _choose_best_lr() -> float:
    results = {}
    for p in OUTPUT_ROOT.glob("metrics_Output*.txt"):
        lr, bs = _extract_lr_bs(p.stem)
        if bs != 4:
            continue
        metrics = _parse_metrics_file(p)
        _, vals = _select_best(metrics)
        results[lr] = vals
    if not results:
        return 0.0
    mins = [min(v[i] for v in results.values()) for i in range(3)]
    best_lr = None
    best_count = -1
    best_sum = float("inf")
    for lr, vals in results.items():
        count = sum(vals[i] == mins[i] for i in range(3))
        total = sum(vals)
        if count > best_count or (count == best_count and total < best_sum):
            best_lr = lr
            best_count = count
            best_sum = total
    return best_lr or 0.0


def _choose_best_overall() -> Tuple[float, int]:
    results = {}
    for p in OUTPUT_ROOT.glob("metrics_Output*.txt"):
        lr, bs = _extract_lr_bs(p.stem)
        metrics = _parse_metrics_file(p)
        _, vals = _select_best(metrics)
        results[(lr, bs)] = vals
    if not results:
        return 0.0, 0
    mins = [min(v[i] for v in results.values()) for i in range(3)]
    best = None
    best_count = -1
    best_sum = float("inf")
    for key, vals in results.items():
        count = sum(vals[i] == mins[i] for i in range(3))
        total = sum(vals)
        if count > best_count or (count == best_count and total < best_sum):
            best = key
            best_count = count
            best_sum = total
    return best if best else (0.0, 0)


def main() -> None:
    first_lrs = [5.0e-6, 2.0e-5, 5.0e-5, 1.0e-4]
    for lr in first_lrs:
        cfg = _write_config(lr, 4, 7)
        _run_training(cfg)
    _run_metrics()
    best_lr = _choose_best_lr()
    print(f"Best learning rate: {best_lr}")
    for bs in [2, 8]:
        cfg = _write_config(best_lr, bs, 7)
        _run_training(cfg)
    _run_metrics()
    lr, bs = _choose_best_overall()
    print(f"Best configuration: lr={lr}, batch_size={bs}")


if __name__ == "__main__":
    main()