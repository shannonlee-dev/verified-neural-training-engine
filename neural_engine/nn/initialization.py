from __future__ import annotations

import numpy as np

from neural_engine.config import DEFAULT_SEED


INITIALIZATIONS = ("zero", "random", "he", "xavier")


def initialize_weights(
    in_features: int,
    out_features: int,
    initialization: str,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if in_features <= 0 or out_features <= 0:
        raise ValueError("in_features and out_features must be positive")
    if initialization not in INITIALIZATIONS:
        raise ValueError(
            f"initialization must be one of {INITIALIZATIONS}, got {initialization!r}"
        )
    if initialization == "zero":
        return np.zeros((in_features, out_features), dtype=np.float64)

    generator = rng or np.random.default_rng(DEFAULT_SEED)
    scales = {
        "random": 1.0,
        "he": np.sqrt(2.0 / in_features),
        "xavier": np.sqrt(2.0 / (in_features + out_features)),
    }
    return generator.normal(
        loc=0.0,
        scale=scales[initialization],
        size=(in_features, out_features),
    )
