from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from neural_engine.core.tensor import Tensor
from neural_engine.nn.activations import ReLU, Sigmoid, Softmax
from neural_engine.nn.layers import Linear


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
        array[index] = original + epsilon
        positive = float(function(array))
        array[index] = original - epsilon
        negative = float(function(array))
        array[index] = original
        gradient[index] = (positive - negative) / (2.0 * epsilon)
        iterator.iternext()
    return gradient


def relative_error(analytic: np.ndarray, numerical: np.ndarray) -> float:
    analytic_values = np.asarray(analytic, dtype=np.float64)
    numerical_values = np.asarray(numerical, dtype=np.float64)
    if analytic_values.shape != numerical_values.shape:
        raise ValueError("gradient shapes must match")
    denominator = np.maximum(
        1e-12, np.abs(analytic_values) + np.abs(numerical_values)
    )
    return float(np.max(np.abs(analytic_values - numerical_values) / denominator))


def _check(
    name: str,
    analytic: np.ndarray,
    function: Callable[[np.ndarray], float],
    values: np.ndarray,
) -> CheckResult:
    numerical = numerical_gradient(function, values)
    return CheckResult(name, relative_error(analytic, numerical))


def run_gradient_checks() -> list[CheckResult]:
    results: list[CheckResult] = []

    base = np.array([[0.2, -0.4, 0.7], [1.1, 0.3, -0.8]])
    bias_values = np.array([0.1, -0.2, 0.4])
    upstream = np.array([[0.7, -1.1, 0.3], [-0.2, 0.5, 1.3]])
    bias = Tensor(bias_values.copy(), requires_grad=True)
    (((Tensor(base) + bias) * upstream).sum()).backward()
    results.append(
        _check(
            "add_broadcast",
            bias.grad.copy(),
            lambda value: np.sum((base + value) * upstream),
            bias_values,
        )
    )

    multiply_values = np.array([-0.7, 0.2, 1.3])
    multiplier = np.array([0.4, -1.1, 0.8])
    multiply_upstream = np.array([1.2, -0.5, 0.9])
    multiply_tensor = Tensor(multiply_values.copy(), requires_grad=True)
    ((multiply_tensor * multiplier) * multiply_upstream).sum().backward()
    results.append(
        _check(
            "multiply",
            multiply_tensor.grad.copy(),
            lambda value: np.sum(value * multiplier * multiply_upstream),
            multiply_values,
        )
    )

    numerator = np.array([1.2, -0.8, 2.1])
    denominator_values = np.array([0.7, 1.3, -0.9])
    divide_upstream = np.array([0.4, -1.2, 0.6])
    denominator = Tensor(denominator_values.copy(), requires_grad=True)
    ((Tensor(numerator) / denominator) * divide_upstream).sum().backward()
    results.append(
        _check(
            "divide",
            denominator.grad.copy(),
            lambda value: np.sum((numerator / value) * divide_upstream),
            denominator_values,
        )
    )

    left_values = np.array([[0.2, -0.3, 0.7], [1.1, 0.5, -0.4]])
    right_values = np.array([[0.4, -0.2], [0.8, 0.3], [-0.6, 1.2]])
    matmul_upstream = np.array([[0.7, -1.1], [0.2, 0.9]])
    left = Tensor(left_values.copy(), requires_grad=True)
    ((left @ Tensor(right_values)) * matmul_upstream).sum().backward()
    results.append(
        _check(
            "matmul",
            left.grad.copy(),
            lambda value: np.sum((value @ right_values) * matmul_upstream),
            left_values,
        )
    )

    reduction_values = np.array([[0.4, -0.2, 1.1], [-0.7, 0.3, 0.8]])
    reduction = Tensor(reduction_values.copy(), requires_grad=True)
    reduction.sum(axis=1).mean().backward()
    results.append(
        _check(
            "sum_mean",
            reduction.grad.copy(),
            lambda value: np.mean(np.sum(value, axis=1)),
            reduction_values,
        )
    )

    layer = Linear(3, 2, initialization="zero")
    layer.weight.data[:] = np.array([[0.2, -0.4], [0.7, 0.3], [-0.5, 0.8]])
    layer.bias.data[:] = np.array([0.1, -0.2])
    linear_input_values = np.array([[0.4, -0.3, 0.9], [1.2, 0.5, -0.7]])
    linear_upstream = np.array([[0.6, -1.1], [0.2, 0.8]])
    linear_input = Tensor(linear_input_values.copy(), requires_grad=True)
    (layer(linear_input) * linear_upstream).sum().backward()
    results.extend(
        [
            _check(
                "Linear.input",
                linear_input.grad.copy(),
                lambda value: np.sum(
                    (value @ layer.weight.data + layer.bias.data) * linear_upstream
                ),
                linear_input_values,
            ),
            _check(
                "Linear.weight",
                layer.weight.grad.copy(),
                lambda value: np.sum(
                    (linear_input_values @ value + layer.bias.data) * linear_upstream
                ),
                layer.weight.data,
            ),
            _check(
                "Linear.bias",
                layer.bias.grad.copy(),
                lambda value: np.sum(
                    (linear_input_values @ layer.weight.data + value) * linear_upstream
                ),
                layer.bias.data,
            ),
        ]
    )

    relu_values = np.array([-1.2, -0.3, 0.4, 1.5])
    activation_upstream = np.array([0.7, -1.1, 0.4, 0.9])
    relu_input = Tensor(relu_values.copy(), requires_grad=True)
    (ReLU()(relu_input) * activation_upstream).sum().backward()
    results.append(
        _check(
            "ReLU",
            relu_input.grad.copy(),
            lambda value: np.sum(np.maximum(value, 0.0) * activation_upstream),
            relu_values,
        )
    )

    sigmoid_values = np.array([-1.3, -0.2, 0.6, 1.7])
    sigmoid_input = Tensor(sigmoid_values.copy(), requires_grad=True)
    (Sigmoid()(sigmoid_input) * activation_upstream).sum().backward()

    def sigmoid_function(value: np.ndarray) -> float:
        probabilities = 1.0 / (1.0 + np.exp(-value))
        return float(np.sum(probabilities * activation_upstream))

    results.append(
        _check(
            "Sigmoid",
            sigmoid_input.grad.copy(),
            sigmoid_function,
            sigmoid_values,
        )
    )

    softmax_values = np.array([[0.3, -0.8, 1.2], [1.1, 0.4, -0.5]])
    softmax_upstream = np.array([[0.7, -0.2, 1.3], [-0.6, 0.9, 0.4]])
    softmax_input = Tensor(softmax_values.copy(), requires_grad=True)
    (Softmax()(softmax_input) * softmax_upstream).sum().backward()

    def softmax_function(value: np.ndarray) -> float:
        shifted = value - np.max(value, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
        return float(np.sum(probabilities * softmax_upstream))

    results.append(
        _check(
            "Softmax",
            softmax_input.grad.copy(),
            softmax_function,
            softmax_values,
        )
    )
    return results
