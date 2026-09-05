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