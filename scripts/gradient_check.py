from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neural_engine.verification import GRADIENT_THRESHOLD, run_gradient_checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify analytic AutoGrad gradients")
    parser.add_argument("--log-file", type=Path, default=Path("logs/gradient_check.log"))
    args = parser.parse_args()

    results = run_gradient_checks()
    lines = []
    for result in results:
        status = "PASS" if result.relative_error <= GRADIENT_THRESHOLD else "FAIL"
        lines.append(
            f"[{status}] {result.name}: relative_error={result.relative_error:.3e}"
        )
    maximum = max(result.relative_error for result in results)
    lines.append(
        f"Maximum relative error: {maximum:.3e} (threshold: {GRADIENT_THRESHOLD:.1e})"
    )
    lines.append("All gradient checks passed." if maximum <= GRADIENT_THRESHOLD else "Gradient checks failed.")
    output = "\n".join(lines) + "\n"
    print(output, end="")
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.write_text(output, encoding="utf-8")
    return 0 if maximum <= GRADIENT_THRESHOLD else 1


if __name__ == "__main__":
    raise SystemExit(main())
