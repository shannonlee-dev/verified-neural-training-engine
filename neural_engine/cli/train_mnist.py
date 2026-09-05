from __future__ import annotations

import argparse
from pathlib import Path

from neural_engine.config import DEFAULT_MNIST_HIDDEN_FEATURES, DEFAULT_SEED
from neural_engine.data.mnist import MNIST_FILES, load_mnist
from neural_engine.mnist_training import train_mnist


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--hidden-features", type=int, default=DEFAULT_MNIST_HIDDEN_FEATURES
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--log-file", type=Path, default=Path("logs/mnist.log"))


def run(args: argparse.Namespace) -> int:
    cache_exists = all((args.data_dir / name).exists() for name in MNIST_FILES.values())
    print(
        "Using cached MNIST IDX gzip files."
        if cache_exists
        else "Downloading MNIST IDX gzip files..."
    )
    x_train, y_train, x_test, y_test = load_mnist(args.data_dir)
    if args.train_limit is not None:
        x_train, y_train = x_train[: args.train_limit], y_train[: args.train_limit]
    if args.test_limit is not None:
        x_test, y_test = x_test[: args.test_limit], y_test[: args.test_limit]

    history = train_mnist(
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        hidden_features=args.hidden_features,
        class_count=10,
    )
    lines = ["epoch,loss,accuracy,seed"]
    for item in history:
        lines.append(f"{item.epoch},{item.loss:.10f},{item.accuracy:.4f},{item.seed}")
        print(
            f"Epoch {item.epoch}/{args.epochs}: loss={item.loss:.6f}, "
            f"test_accuracy={item.accuracy:.2%}"
        )
    final = history[-1]
    lines.append(
        f"Final: loss={final.loss:.6f}, test_accuracy={final.accuracy:.2%}, seed={final.seed}"
    )
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Log: {args.log_file}")

    full_run = args.train_limit is None and args.test_limit is None
    return 1 if full_run and final.accuracy < 0.95 else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the NumPy engine on MNIST")
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())