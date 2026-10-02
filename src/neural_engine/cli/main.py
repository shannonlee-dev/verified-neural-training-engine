from __future__ import annotations

import argparse

from neural_engine.cli import (
    compare_initialization,
    gradient_check,
    train_mnist,
    train_xor,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="검증 가능한 신경망 학습 엔진")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="학습 실험 실행")
    train_commands = train.add_subparsers(dest="training_command", required=True)
    xor = train_commands.add_parser("xor", help="XOR 신경망 학습")
    train_xor.add_arguments(xor)
    xor.set_defaults(run=train_xor.run)
    mnist = train_commands.add_parser("mnist", help="MNIST 학습")
    train_mnist.add_arguments(mnist)
    mnist.set_defaults(run=train_mnist.run)

    verify = commands.add_parser("verify", help="수치 검증 실행")
    verify_commands = verify.add_subparsers(dest="verification_command", required=True)
    gradients = verify_commands.add_parser("gradients", help="해석적 기울기 검증")
    gradient_check.add_arguments(gradients)
    gradients.set_defaults(run=gradient_check.run)

    compare = commands.add_parser("compare", help="비교 실험 실행")
    compare_commands = compare.add_subparsers(dest="comparison_command", required=True)
    initialization = compare_commands.add_parser(
        "initialization", help="XOR 초기화 비교"
    )
    compare_initialization.add_arguments(initialization)
    initialization.set_defaults(run=compare_initialization.run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
