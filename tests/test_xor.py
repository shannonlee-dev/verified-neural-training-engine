import unittest
import tempfile
from pathlib import Path

from neural_engine.experiments import train_xor
from scripts.compare_initialization import main as compare_initialization_main


class XorExperimentTests(unittest.TestCase):
    def test_he_initialization_solves_xor_within_100_epochs(self):
        history = train_xor("he", epochs=100, seed=42)

        self.assertTrue(
            history[-1].loss < 0.1 or history[-1].accuracy >= 0.95,
            f"final metrics were {history[-1]}",
        )

    def test_zero_initialization_stays_unsolved_for_50_epochs(self):
        history = train_xor("zero", epochs=50, seed=42)

        self.assertLess(history[-1].accuracy, 0.95)
        self.assertGreater(history[-1].loss, 0.1)
        self.assertLess(abs(history[0].loss - history[-1].loss), 1e-8)

    def test_same_seed_reproduces_identical_history(self):
        first = train_xor("he", epochs=5, seed=7)
        second = train_xor("he", epochs=5, seed=7)

        self.assertEqual(first, second)

    def test_comparison_csv_uses_portable_line_endings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_file = root / "comparison.csv"
            figure_file = root / "comparison.png"

            status = compare_initialization_main(
                [
                    "--epochs",
                    "1",
                    "--csv-file",
                    str(csv_file),
                    "--figure-file",
                    str(figure_file),
                ]
            )

            self.assertEqual(status, 0)
            self.assertNotIn(b"\r\n", csv_file.read_bytes())
            self.assertTrue(figure_file.stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
