from __future__ import annotations

import argparse

from neural_engine.cli import compare_initialization, gradient_check, train_mnist, train_xor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verified Neural Training Engine")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="run a training experiment")
    train_commands = train.add_subparsers(dest="training_command", required=True)
    xor = train_commands.add_parser("xor", help="train an XOR network")
    train_xor.add_arguments(xor)
    xor.set_defaults(run=train_xor.run)
    mnist = train_commands.add_parser("mnist", help="train on MNIST")
    train_mnist.add_arguments(mnist)
    mnist.set_defaults(run=train_mnist.run)

    verify = commands.add_parser("verify", help="run verification checks")
    verify_commands = verify.add_subparsers(dest="verification_command", required=True)
    gradients = verify_commands.add_parser("gradients", help="check analytic gradients")
    gradient_check.add_arguments(gradients)
    gradients.set_defaults(run=gradient_check.run)

    compare = commands.add_parser("compare", help="run comparison experiments")
    compare_commands = compare.add_subparsers(dest="comparison_command", required=True)
    initialization = compare_commands.add_parser(
        "initialization", help="compare XOR initializations"
    )
    compare_initialization.add_arguments(initialization)
    initialization.set_defaults(run=compare_initialization.run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = vars(build_parser().parse_args(argv))
    runner = args.pop("run")
    args.pop("command", None)
    args.pop("training_command", None)
    args.pop("verification_command", None)
    args.pop("comparison_command", None)
    return runner(argparse.Namespace(**args))


if __name__ == "__main__":
    raise SystemExit(main())