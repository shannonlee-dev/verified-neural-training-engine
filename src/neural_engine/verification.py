from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from neural_engine.core.tensor import Tensor
from neural_engine.nn.activations import ReLU, Sigmoid, Softmax
from neural_engine.nn.layers import Linear
from neural_engine.nn.losses import binary_cross_entropy, cross_entropy

GRADIENT_EPSILON = 1e-5
GRADIENT_THRESHOLD = 1e-7


@dataclass(frozen=True)
class CheckResult:
    name: str
    relative_error: float


def numerical_gradient(
    function: Callable[[np.ndarray], float],
    values: np.ndarray,
    epsilon: float = GRADIENT_EPSILON,
) -> np.ndarray:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    array = np.asarray(values, dtype=np.float64)
    gradient = np.zeros_like(array)
    iterator = np.nditer(array, flags=["multi_index"], op_flags=["readwrite"])
    while not iterator.finished:
        index = iterator.multi_index
        original = float(array[index])
        try:
            array[index] = original + epsilon
            positive = float(function(array))
            array[index] = original - epsilon
            negative = float(function(array))
        finally:
            array[index] = original
        gradient[index] = (positive - negative) / (2.0 * epsilon)
        iterator.iternext()
    return gradient


def relative_error(
    analytical_gradient: np.ndarray, numerical_gradient_values: np.ndarray
) -> float:
    analytical_values = np.asarray(analytical_gradient, dtype=np.float64)
    numerical_values = np.asarray(numerical_gradient_values, dtype=np.float64)
    if analytical_values.shape != numerical_values.shape:
        raise ValueError("gradient shapes must match")
    denominator = np.maximum(
        1e-12, np.abs(analytical_values) + np.abs(numerical_values)
    )
    return float(np.max(np.abs(analytical_values - numerical_values) / denominator))


def compare_gradients(
    name: str,
    analytical_gradient: np.ndarray,
    numerical_function: Callable[[np.ndarray], float],
    numerical_values: np.ndarray,
) -> CheckResult:
    numerical_gradient_values = numerical_gradient(numerical_function, numerical_values)
    return CheckResult(
        name, relative_error(analytical_gradient, numerical_gradient_values)
    )


def run_gradient_checks() -> list[CheckResult]:
    results: list[CheckResult] = []

    base = np.array([[0.2, -0.4, 0.7], [1.1, 0.3, -0.8]])
    numerical_values = np.array([0.1, -0.2, 0.4])
    upstream = np.array([[0.7, -1.1, 0.3], [-0.2, 0.5, 1.3]])
    bias = Tensor(numerical_values.copy(), requires_grad=True)
    (((Tensor(base) + bias) * upstream).sum()).backward()
    analytical_gradient = bias.grad.copy()

    def numerical_function(values):
        return np.sum((base + values) * upstream)

    results.append(
        compare_gradients(
            "add_broadcast",
            analytical_gradient,
            numerical_function,
            numerical_values,
        )
    )

    numerical_values = np.array([-0.7, 0.2, 1.3])
    multiplier = np.array([0.4, -1.1, 0.8])
    multiply_upstream = np.array([1.2, -0.5, 0.9])
    multiply_tensor = Tensor(numerical_values.copy(), requires_grad=True)
    ((multiply_tensor * multiplier) * multiply_upstream).sum().backward()
    analytical_gradient = multiply_tensor.grad.copy()

    def numerical_function(values):
        return np.sum(values * multiplier * multiply_upstream)

    results.append(
        compare_gradients(
            "multiply",
            analytical_gradient,
            numerical_function,
            numerical_values,
        )
    )

    numerator = np.array([1.2, -0.8, 2.1])
    numerical_values = np.array([0.7, 1.3, -0.9])
    divide_upstream = np.array([0.4, -1.2, 0.6])
    denominator = Tensor(numerical_values.copy(), requires_grad=True)
    ((Tensor(numerator) / denominator) * divide_upstream).sum().backward()
    analytical_gradient = denominator.grad.copy()

    def numerical_function(values):
        return np.sum(numerator / values * divide_upstream)

    results.append(
        compare_gradients(
            "divide",
            analytical_gradient,
            numerical_function,
            numerical_values,
        )
    )

    numerical_values = np.array([[0.2, -0.3, 0.7], [1.1, 0.5, -0.4]])
    right_values = np.array([[0.4, -0.2], [0.8, 0.3], [-0.6, 1.2]])
    matmul_upstream = np.array([[0.7, -1.1], [0.2, 0.9]])
    left = Tensor(numerical_values.copy(), requires_grad=True)
    ((left @ Tensor(right_values)) * matmul_upstream).sum().backward()
    analytical_gradient = left.grad.copy()

    def numerical_function(values):
        return np.sum(values @ right_values * matmul_upstream)

    results.append(
        compare_gradients(
            "matmul",
            analytical_gradient,
            numerical_function,
            numerical_values,
        )
    )

    numerical_values = np.array([[0.4, -0.2, 1.1], [-0.7, 0.3, 0.8]])
    reduction = Tensor(numerical_values.copy(), requires_grad=True)
    reduction.sum(axis=1).mean().backward()
    analytical_gradient = reduction.grad.copy()

    def numerical_function(values):
        return np.mean(np.sum(values, axis=1))

    results.append(
        compare_gradients(
            "sum_mean",
            analytical_gradient,
            numerical_function,
            numerical_values,
        )
    )

    layer = Linear(3, 2, initialization="zero")
    layer.weight.data[:] = np.array([[0.2, -0.4], [0.7, 0.3], [-0.5, 0.8]])
    layer.bias.data[:] = np.array([0.1, -0.2])
    linear_input_values = np.array([[0.4, -0.3, 0.9], [1.2, 0.5, -0.7]])
    linear_upstream = np.array([[0.6, -1.1], [0.2, 0.8]])
    linear_input = Tensor(linear_input_values.copy(), requires_grad=True)
    (layer(linear_input) * linear_upstream).sum().backward()
    analytical_gradient = linear_input.grad.copy()

    def numerical_function(values):
        return np.sum((values @ layer.weight.data + layer.bias.data) * linear_upstream)

    numerical_values = linear_input_values
    results.append(
        compare_gradients(
            "Linear.input",
            analytical_gradient,
            numerical_function,
            numerical_values,
        )
    )

    analytical_gradient = layer.weight.grad.copy()

    def numerical_function(values):
        return np.sum(
            (linear_input_values @ values + layer.bias.data) * linear_upstream
        )

    numerical_values = layer.weight.data
    results.append(
        compare_gradients(
            "Linear.weight",
            analytical_gradient,
            numerical_function,
            numerical_values,
        )
    )

    analytical_gradient = layer.bias.grad.copy()

    def numerical_function(values):
        return np.sum(
            (linear_input_values @ layer.weight.data + values) * linear_upstream
        )

    numerical_values = layer.bias.data
    results.append(
        compare_gradients(
            "Linear.bias",
            analytical_gradient,
            numerical_function,
            numerical_values,
        )
    )

    numerical_values = np.array([-1.2, -0.3, 0.4, 1.5])
    activation_upstream = np.array([0.7, -1.1, 0.4, 0.9])
    relu_input = Tensor(numerical_values.copy(), requires_grad=True)
    (ReLU()(relu_input) * activation_upstream).sum().backward()
    analytical_gradient = relu_input.grad.copy()

    def numerical_function(values):
        return np.sum(np.maximum(values, 0.0) * activation_upstream)

    results.append(
        compare_gradients(
            "ReLU",
            analytical_gradient,
            numerical_function,
            numerical_values,
        )
    )

    numerical_values = np.array([-1.3, -0.2, 0.6, 1.7])
    sigmoid_input = Tensor(numerical_values.copy(), requires_grad=True)
    (Sigmoid()(sigmoid_input) * activation_upstream).sum().backward()

    def numerical_function(values: np.ndarray) -> float:
        probabilities = 1.0 / (1.0 + np.exp(-values))
        return float(np.sum(probabilities * activation_upstream))

    analytical_gradient = sigmoid_input.grad.copy()
    results.append(
        compare_gradients(
            "Sigmoid",
            analytical_gradient,
            numerical_function,
            numerical_values,
        )
    )

    numerical_values = np.array([[0.3, -0.8, 1.2], [1.1, 0.4, -0.5]])
    softmax_upstream = np.array([[0.7, -0.2, 1.3], [-0.6, 0.9, 0.4]])
    softmax_input = Tensor(numerical_values.copy(), requires_grad=True)
    (Softmax()(softmax_input) * softmax_upstream).sum().backward()

    def numerical_function(values: np.ndarray) -> float:
        shifted = values - np.max(values, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
        return float(np.sum(probabilities * softmax_upstream))

    analytical_gradient = softmax_input.grad.copy()
    results.append(
        compare_gradients(
            "Softmax",
            analytical_gradient,
            numerical_function,
            numerical_values,
        )
    )

    numerical_values = np.array([[0.2], [0.7], [0.4]])
    binary_targets = np.array([[0.0], [1.0], [1.0]])
    probabilities = Tensor(numerical_values.copy(), requires_grad=True)
    binary_cross_entropy(probabilities, binary_targets).backward()

    def numerical_function(values: np.ndarray) -> float:
        clipped = np.clip(values, 1e-12, 1.0 - 1e-12)
        return float(
            -np.mean(
                binary_targets * np.log(clipped)
                + (1.0 - binary_targets) * np.log(1.0 - clipped)
            )
        )

    analytical_gradient = probabilities.grad.copy()
    results.append(
        compare_gradients(
            "BinaryCrossEntropy",
            analytical_gradient,
            numerical_function,
            numerical_values,
        )
    )

    numerical_values = np.array([[0.2, -0.7, 1.1], [1.4, 0.3, -0.5]])
    class_targets = np.array([2, 0])
    logits = Tensor(numerical_values.copy(), requires_grad=True)
    cross_entropy(logits, class_targets).backward()

    def numerical_function(values: np.ndarray) -> float:
        shifted = values - np.max(values, axis=1, keepdims=True)
        log_partition = np.log(np.exp(shifted).sum(axis=1))
        target_values = shifted[np.arange(len(values)), class_targets]
        return float(np.mean(-target_values + log_partition))

    analytical_gradient = logits.grad.copy()
    results.append(
        compare_gradients(
            "CrossEntropy",
            analytical_gradient,
            numerical_function,
            numerical_values,
        )
    )
    return results
