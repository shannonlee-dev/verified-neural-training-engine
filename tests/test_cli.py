"""설치된 신경망 CLI의 진입점을 임시 작업 디렉토리에서 검증한다."""

import subprocess
import sys
from importlib.metadata import distribution

import pytest


@pytest.mark.smoke
def test_installed_console_and_module(tmp_path):
    entries = {
        entry.name: entry.value
        for entry in distribution("verified-neural-training-engine").entry_points
    }
    assert entries["neural-engine"] == "neural_engine.cli.main:main"
    result = subprocess.run(
        [sys.executable, "-I", "-m", "neural_engine", "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "verify" in result.stdout and "train" in result.stdout
    assert list(tmp_path.iterdir()) == []
