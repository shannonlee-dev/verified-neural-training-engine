from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neural_engine.config import DEFAULT_SEED
from neural_engine.experiments import train_xor
from neural_engine.nn.initialization import INITIALIZATIONS


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an XOR network")
    parser.add_argument("--initialization", choices=INITIALIZATIONS, default="he")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    history = train_xor(
        initialization=args.initialization,
        epochs=args.epochs,
        seed=args.seed,
        learning_rate=args.learning_rate,
    )
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
    output = "\n".join(lines) + "\n"
    print(output, end="")

    log_file = args.log_file or Path(f"logs/xor_{args.initialization}.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
