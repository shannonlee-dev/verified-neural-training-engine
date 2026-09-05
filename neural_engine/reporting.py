from __future__ import annotations

from pathlib import Path

from neural_engine.experiments import EpochMetrics


def format_xor_history(history: list[EpochMetrics]) -> str:
    lines = ["epoch,loss,accuracy,initialization,seed"]
    lines.extend(
        f"{item.epoch},{item.loss:.10f},{item.accuracy:.4f},{item.initialization},{item.seed}"
        for item in history
    )
    final = history[-1]
    lines.append(
        f"Final: loss={final.loss:.6f}, accuracy={final.accuracy:.2%}, "
        f"initialization={final.initialization}, seed={final.seed}"
    )
    return "\n".join(lines) + "\n"


def write_report(output: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output, encoding="utf-8")


def default_xor_log_path(initialization: str) -> Path:
    return Path(f"logs/xor_{initialization}.log")


def write_initialization_comparison(
    histories: dict[str, list[EpochMetrics]], csv_path: Path, figure_path: Path
) -> None:
    import csv
    import os
    import tempfile

    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "verified-neural-engine-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "loss", "accuracy", "initialization", "seed"],
            lineterminator="\n",
        )
        writer.writeheader()
        for initialization, history in histories.items():
            for item in history:
                writer.writerow(
                    {
                        "epoch": item.epoch,
                        "loss": f"{item.loss:.10f}",
                        "accuracy": f"{item.accuracy:.4f}",
                        "initialization": initialization,
                        "seed": item.seed,
                    }
                )

    figure_path.parent.mkdir(parents=True, exist_ok=True)
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
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)