from __future__ import annotations

import argparse
from pathlib import Path

from neural_engine.config import DEFAULT_SEED
from neural_engine.experiments import train_xor
from neural_engine.nn.initialization import INITIALIZATIONS
from neural_engine.reporting import default_xor_log_path, format_xor_history, write_report


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--initialization", choices=INITIALIZATIONS, default="he")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--log-file", type=Path)


def run(args: argparse.Namespace) -> int:
    history = train_xor(
        initialization=args.initialization,
        epochs=args.epochs,
        seed=args.seed,
        learning_rate=args.learning_rate,
    )
    output = format_xor_history(history)
    print(output, end="")
    log_file = args.log_file or default_xor_log_path(args.initialization)
    write_report(output, log_file)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train an XOR network")
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())