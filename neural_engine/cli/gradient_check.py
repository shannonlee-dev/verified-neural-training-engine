from __future__ import annotations

import argparse
from pathlib import Path

from neural_engine.verification import GRADIENT_THRESHOLD, run_gradient_checks


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-file", type=Path, default=Path("logs/gradient_check.log"))


def run(args: argparse.Namespace) -> int:
    results = run_gradient_checks()
    lines = []
    for result in results:
        status = "PASS" if result.relative_error <= GRADIENT_THRESHOLD else "FAIL"
        lines.append(f"[{status}] {result.name}: relative_error={result.relative_error:.3e}")
    maximum = max(result.relative_error for result in results)
    lines.append(
        f"Maximum relative error: {maximum:.3e} (threshold: {GRADIENT_THRESHOLD:.1e})"
    )
    lines.append(
        "All gradient checks passed."
        if maximum <= GRADIENT_THRESHOLD
        else "Gradient checks failed."
    )
    output = "\n".join(lines) + "\n"
    print(output, end="")
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.write_text(output, encoding="utf-8")
    return 0 if maximum <= GRADIENT_THRESHOLD else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify analytic AutoGrad gradients")
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())