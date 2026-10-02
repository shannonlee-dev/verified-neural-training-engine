from pathlib import Path

import numpy as np

from neural_engine.cli.compare_initialization import main as compare_initialization_main
from neural_engine.experiments import build_xor_model, train_xor


def test_he_initialization_solves_xor_within_100_epochs():
    history = train_xor("he", epochs=100, seed=42)
    assert history[-1].loss < 0.1 or history[-1].accuracy >= 0.95, (
        f"final metrics were {history[-1]}"
    )


def test_zero_initialization_stays_unsolved_for_50_epochs():
    history = train_xor("zero", epochs=50, seed=42)
    assert history[-1].accuracy < 0.95
    assert history[-1].loss > 0.1
    assert abs(history[0].loss - history[-1].loss) < 1e-08


def test_same_seed_reproduces_identical_history():
    first = train_xor("he", epochs=5, seed=7)
    second = train_xor("he", epochs=5, seed=7)
    assert first == second


def test_model_seed_controls_weight_initialization():
    first = build_xor_model(seed=7).parameters()
    second = build_xor_model(seed=7).parameters()
    different = build_xor_model(seed=8).parameters()
    for first_parameter, second_parameter in zip(first, second):
        np.testing.assert_array_equal(first_parameter.data, second_parameter.data)
    assert any(
        (
            not np.array_equal(first_parameter.data, different_parameter.data)
            for first_parameter, different_parameter in zip(first, different)
        )
    )


def test_comparison_csv_uses_portable_line_endings(tmp_path):
    directory = tmp_path
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
    assert status == 0
    assert b"\r\n" not in csv_file.read_bytes()
    assert figure_file.stat().st_size > 0
