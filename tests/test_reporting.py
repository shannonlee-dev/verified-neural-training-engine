import tempfile
import unittest
from pathlib import Path

from neural_engine.experiments import EpochMetrics
from neural_engine.reporting import (
    default_xor_log_path,
    format_xor_history,
    write_report,
)


class ReportingTests(unittest.TestCase):
    def test_format_xor_history_preserves_report_format(self):
        history = [
            EpochMetrics(
                epoch=1,
                loss=0.123456789,
                accuracy=0.75,
                initialization="he",
                seed=42,
            ),
            EpochMetrics(
                epoch=2,
                loss=0.0123456789,
                accuracy=1.0,
                initialization="he",
                seed=42,
            ),
        ]

        self.assertEqual(
            format_xor_history(history),
            "epoch,loss,accuracy,initialization,seed\n"
            "1,0.1234567890,0.7500,he,42\n"
            "2,0.0123456789,1.0000,he,42\n"
            "Final: loss=0.012346, accuracy=100.00%, initialization=he, seed=42\n",
        )

    def test_write_report_creates_parent_and_writes_utf8(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "report.log"
            output = "정확도\n"

            write_report(output, path)

            self.assertEqual(path.read_text(encoding="utf-8"), output)

    def test_default_xor_log_path_preserves_existing_location(self):
        self.assertEqual(default_xor_log_path("he"), Path("logs/xor_he.log"))


if __name__ == "__main__":
    unittest.main()