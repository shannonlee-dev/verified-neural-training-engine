from __future__ import annotations

import argparse
from pathlib import Path

from neural_engine.reporting import format_gradient_checks, write_report
from neural_engine.verification import GRADIENT_THRESHOLD, run_gradient_checks


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-file", type=Path, default=Path("logs/gradient_check.log")
    )


def run(args: argparse.Namespace) -> int:
    results = run_gradient_checks()
    output, maximum = format_gradient_checks(results, GRADIENT_THRESHOLD)
    print(output, end="")
    write_report(output, args.log_file)
    return 0 if maximum <= GRADIENT_THRESHOLD else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify analytic AutoGrad gradients")
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
