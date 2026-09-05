from __future__ import annotations

import numpy as np

from neural_engine.core.tensor import Tensor


def binary_cross_entropy(
    probabilities: Tensor,
    targets: np.ndarray,
    epsilon: float = 1e-12,
) -> Tensor:
    target_values = np.asarray(targets, dtype=np.float64)
    if probabilities.shape != target_values.shape:
        raise ValueError(
            f"probability shape {probabilities.shape} does not match target shape {target_values.shape}"
        )
    if np.any(probabilities.data < 0.0) or np.any(probabilities.data > 1.0):
        raise ValueError("binary_cross_entropy probabilities must be in [0, 1]")
    clipped = np.clip(probabilities.data, epsilon, 1.0 - epsilon)
    loss_value = -np.mean(
        target_values * np.log(clipped)
        + (1.0 - target_values) * np.log(1.0 - clipped)
    )
    output = Tensor(
        loss_value,
        probabilities.requires_grad,
        _parents=(probabilities,),
        _op="binary_cross_entropy",
    )

    def backward() -> None:
        derivative = (
            -target_values / clipped + (1.0 - target_values) / (1.0 - clipped)
        ) / target_values.size
        interior = (probabilities.data > epsilon) & (
            probabilities.data < 1.0 - epsilon
        )
        derivative = np.where(interior, derivative, 0.0)
        probabilities._accumulate(output.grad * derivative)

    if output.requires_grad:
        output._backward = backward
    return output


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    target_values = np.asarray(targets, dtype=np.int64)
    if logits.ndim != 2:
        raise ValueError(f"cross_entropy expects 2D logits, got {logits.shape}")
    if target_values.shape != (logits.shape[0],):
        raise ValueError(
            f"targets must have shape ({logits.shape[0]},), got {target_values.shape}"
        )
    if np.any(target_values < 0) or np.any(target_values >= logits.shape[1]):
        raise ValueError("target class is outside the logits class range")

    shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    partition = exponentials.sum(axis=1, keepdims=True)
    probabilities = exponentials / partition
    batch_indices = np.arange(logits.shape[0])
    target_logits = shifted[batch_indices, target_values]
    loss_value = (-target_logits + np.log(partition[:, 0])).mean()
    output = Tensor(
        loss_value,
        logits.requires_grad,
        _parents=(logits,),
        _op="cross_entropy",
    )

    def backward() -> None:
        gradient = probabilities.copy()
        gradient[batch_indices, target_values] -= 1.0
        gradient /= logits.shape[0]
        logits._accumulate(output.grad * gradient)

    if output.requires_grad:
        output._backward = backward
    return output
