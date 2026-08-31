from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neural_engine.config import DEFAULT_SEED
from neural_engine.core.tensor import Tensor
from neural_engine.data.mnist import batch_iterator
from neural_engine.nn.activations import ReLU
from neural_engine.nn.layers import Linear
from neural_engine.nn.losses import cross_entropy
from neural_engine.nn.module import Sequential
from neural_engine.optim.adam import Adam


@dataclass(frozen=True)
class MnistEpochMetrics:
    epoch: int
    loss: float
    accuracy: float
    seed: int


def build_mnist_model(
    input_features: int = 784,
    hidden_features: int = 128,
    class_count: int = 10,
    seed: int = DEFAULT_SEED,
) -> Sequential:
    rng = np.random.default_rng(seed)
    return Sequential(
        Linear(input_features, hidden_features, initialization="he", rng=rng),
        ReLU(),
        Linear(hidden_features, class_count, initialization="he", rng=rng),
    )


def predict_mnist(
    model: Sequential,
    inputs: np.ndarray,
    batch_size: int = 1024,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if (
        len(model.layers) != 3
        or not isinstance(model.layers[0], Linear)
        or not isinstance(model.layers[1], ReLU)
        or not isinstance(model.layers[2], Linear)
    ):
        raise ValueError("MNIST prediction expects Linear → ReLU → Linear")
    first = model.layers[0]
    second = model.layers[2]
    predictions = []
    for start in range(0, len(inputs), batch_size):
        batch = np.asarray(inputs[start : start + batch_size], dtype=np.float64)
        hidden = np.maximum(batch @ first.weight.data + first.bias.data, 0.0)
        logits = hidden @ second.weight.data + second.bias.data
        predictions.append(np.argmax(logits, axis=1))
    if not predictions:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(predictions).astype(np.int64, copy=False)


def train_mnist(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    test_inputs: np.ndarray,
    test_targets: np.ndarray,
    *,
    epochs: int = 1,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    seed: int = DEFAULT_SEED,
    hidden_features: int = 128,
    class_count: int = 10,
) -> list[MnistEpochMetrics]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if train_inputs.ndim != 2 or test_inputs.ndim != 2:
        raise ValueError("MNIST inputs must be flattened 2D arrays")
    if train_inputs.shape[1] != test_inputs.shape[1]:
        raise ValueError("train and test feature counts must match")
    if len(train_inputs) == 0 or len(test_inputs) == 0:
        raise ValueError("train and test splits must not be empty")
    if len(train_inputs) != len(train_targets) or len(test_inputs) != len(test_targets):
        raise ValueError("inputs and targets must have the same length")

    model = build_mnist_model(
        input_features=train_inputs.shape[1],
        hidden_features=hidden_features,
        class_count=class_count,
        seed=seed,
    )
    optimizer = Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    history: list[MnistEpochMetrics] = []

    for epoch in range(1, epochs + 1):
        weighted_loss = 0.0
        sample_count = 0
        for batch_inputs, batch_targets in batch_iterator(
            train_inputs, train_targets, batch_size, rng, shuffle=True
        ):
            optimizer.zero_grad()
            logits = model(Tensor(batch_inputs))
            loss = cross_entropy(logits, batch_targets)
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.data) * len(batch_inputs)
            sample_count += len(batch_inputs)

        predictions = predict_mnist(model, test_inputs)
        accuracy = float(np.mean(predictions == test_targets))
        average_loss = weighted_loss / sample_count
        if not np.isfinite(average_loss) or not np.isfinite(accuracy):
            raise FloatingPointError("training produced a non-finite loss or accuracy")
        history.append(
            MnistEpochMetrics(
                epoch=epoch,
                loss=average_loss,
                accuracy=accuracy,
                seed=seed,
            )
        )
    return history
