from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neural_engine.config import DEFAULT_SEED
from neural_engine.core.tensor import Tensor
from neural_engine.nn.activations import ReLU, Sigmoid
from neural_engine.nn.initialization import INITIALIZATIONS
from neural_engine.nn.layers import Linear
from neural_engine.nn.losses import binary_cross_entropy
from neural_engine.nn.module import Sequential
from neural_engine.optim.adam import Adam


XOR_INPUTS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_TARGETS = np.array([[0.0], [1.0], [1.0], [0.0]])


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    loss: float
    accuracy: float
    initialization: str
    seed: int


def build_xor_model(
    initialization: str = "he",
    seed: int = DEFAULT_SEED,
) -> Sequential:
    if initialization not in INITIALIZATIONS:
        raise ValueError(
            f"initialization must be one of {INITIALIZATIONS}, got {initialization!r}"
        )
    rng = np.random.default_rng(seed)
    return Sequential(
        Linear(2, 8, initialization=initialization, rng=rng),
        ReLU(),
        Linear(8, 1, initialization=initialization, rng=rng),
        Sigmoid(),
    )


def _xor_metrics(model: Sequential) -> tuple[float, float]:
    probabilities = model(Tensor(XOR_INPUTS))
    loss = binary_cross_entropy(probabilities, XOR_TARGETS)
    predictions = (probabilities.data >= 0.5).astype(np.float64)
    accuracy = float(np.mean(predictions == XOR_TARGETS))
    return float(loss.data), accuracy


def train_xor(
    initialization: str = "he",
    epochs: int = 100,
    seed: int = DEFAULT_SEED,
    learning_rate: float = 0.05,
) -> list[EpochMetrics]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    model = build_xor_model(initialization, seed)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    inputs = Tensor(XOR_INPUTS)
    history: list[EpochMetrics] = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        probabilities = model(inputs)
        loss = binary_cross_entropy(probabilities, XOR_TARGETS)
        loss.backward()
        optimizer.step()

        measured_loss, accuracy = _xor_metrics(model)
        history.append(
            EpochMetrics(
                epoch=epoch,
                loss=measured_loss,
                accuracy=accuracy,
                initialization=initialization,
                seed=seed,
            )
        )
    return history


def compare_xor_initializations(
    epochs: int = 100,
    seed: int = DEFAULT_SEED,
    learning_rate: float = 0.05,
) -> dict[str, list[EpochMetrics]]:
    return {
        initialization: train_xor(
            initialization,
            epochs=epochs,
            seed=seed,
            learning_rate=learning_rate,
        )
        for initialization in ("zero", "random", "he")
    }
