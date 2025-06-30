import subprocess
import yaml
import logging
import time
from pathlib import Path
from typing import Dict
import torch

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

OUTPUT_ROOT.mkdir(exist_ok=True)
LOG_FILE = OUTPUT_ROOT / "iterative_training.log"
GPU_COUNT = torch.cuda.device_count()

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("iterative_training")
logger.addHandler(logging.StreamHandler())
logger.info("GPUs available: %s", GPU_COUNT)


def _count_training_data(data_root: Path) -> int:
    meta = data_root / "train" / "metadata.jsonl"
    if meta.exists():
        with open(meta, "r") as f:
            return sum(1 for _ in f)
    return 0


def _write_config(lr: float, batch_size: int, epochs: int) -> Path:
    with open(CONFIG_TEMPLATE, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["learning_rate"] = lr
    cfg["batch_size"] = batch_size
    cfg["train_epochs"] = epochs
    tmp = CONFIG_TEMPLATE.parent / f"tmp_{lr}_{batch_size}_{epochs}.yaml"
    with open(tmp, "w") as f:
        yaml.safe_dump(cfg, f)
    logger.info(
        "Created config %s (lr=%s, batch_size=%s, epochs=%s)",
        tmp.name,
        lr,
        batch_size,
        epochs,
    )
    data_count = _count_training_data(Path(cfg.get("data_root", ".")))
    logger.info("Training data count: %s", data_count)
    return tmp


def _run_training(config: Path) -> float:
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)
    nproc = max(GPU_COUNT, 1)
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "--master_port=29501",
        TRAIN_SCRIPT.as_posix(),
        "--yaml",
        str(config),
    ]
    logger.info(
        "Start training %s with nproc_per_node=%s", config.name, nproc
    )
    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    duration = time.perf_counter() - start
    logger.info(
        "Finished training %s in %.2fs", config.name, duration
    )
    return duration


def _run_metrics() -> None:
    logger.info("Computing metrics...")
    start = time.perf_counter()
    subprocess.run(["python", METRICS_SCRIPT.as_posix()], check=True)
    duration = time.perf_counter() - start
    logger.info("Metrics computed in %.2fs", duration)
    for p in OUTPUT_ROOT.glob("metrics_Output*.txt"):
        metrics = _parse_metrics_file(p)
        for epoch, mse in metrics.items():
            logger.info("%s %s -> MSE=%.6f", p.name, epoch, mse)

def _parse_metrics_file(path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
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
            idx += 1
            if idx < len(lines) and lines[idx].lower() == "mse":
                idx += 1
            if idx < len(lines):
                try:
                    metrics[epoch] = float(lines[idx].split()[0])
                except ValueError:
                    pass
            idx += 1
        else:
            idx += 1
    return metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    args = parser.parse_args()

    cfg = _write_config(args.learning_rate, args.batch_size, args.epochs)
    _run_training(cfg)
    _run_metrics()


if __name__ == "__main__":
    main()
