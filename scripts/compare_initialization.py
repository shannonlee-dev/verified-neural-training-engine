from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "verified-neural-engine-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neural_engine.config import DEFAULT_SEED
from neural_engine.experiments import train_xor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare XOR initializations")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--csv-file", type=Path, default=Path("logs/initialization_comparison.csv")
    )
    parser.add_argument(
        "--figure-file", type=Path, default=Path("figures/initialization_loss.png")
    )
    args = parser.parse_args(argv)

    histories = {
        initialization: train_xor(
            initialization,
            epochs=args.epochs,
            seed=args.seed,
            learning_rate=args.learning_rate,
        )
        for initialization in ("zero", "random", "he")
    }

    args.csv_file.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "loss", "accuracy", "initialization", "seed"],
            lineterminator="\n",
        )
        writer.writeheader()
        for initialization in ("zero", "random", "he"):
            for item in histories[initialization]:
                writer.writerow(
                    {
                        "epoch": item.epoch,
                        "loss": f"{item.loss:.10f}",
                        "accuracy": f"{item.accuracy:.4f}",
                        "initialization": item.initialization,
                        "seed": item.seed,
                    }
                )

    args.figure_file.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    for initialization, history in histories.items():
        axis.plot(
            [item.epoch for item in history],
            [item.loss for item in history],
            label=initialization.capitalize(),
        )
    axis.set(title="XOR Loss by Weight Initialization", xlabel="Epoch", ylabel="Loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.figure_file, dpi=150)
    plt.close(figure)

    for initialization, history in histories.items():
        final = history[-1]
        print(
            f"{initialization:>6}: loss={final.loss:.6f}, "
            f"accuracy={final.accuracy:.2%}"
        )
    print(f"CSV: {args.csv_file}")
    print(f"Figure: {args.figure_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
