from __future__ import annotations

import argparse
from pathlib import Path

from neural_engine.config import DEFAULT_SEED
from neural_engine.experiments import compare_xor_initializations
from neural_engine.reporting import write_initialization_comparison


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--csv-file", type=Path, default=Path("logs/initialization_comparison.csv")
    )
    parser.add_argument(
        "--figure-file", type=Path, default=Path("figures/initialization_loss.png")
    )


def run(args: argparse.Namespace) -> int:
    histories = compare_xor_initializations(
        epochs=args.epochs, seed=args.seed, learning_rate=args.learning_rate
    )
    write_initialization_comparison(histories, args.csv_file, args.figure_file)
    for initialization, history in histories.items():
        final = history[-1]
        print(f"{initialization:>6}: loss={final.loss:.6f}, accuracy={final.accuracy:.2%}")
    print(f"CSV: {args.csv_file}")
    print(f"Figure: {args.figure_file}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare XOR initializations")
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())