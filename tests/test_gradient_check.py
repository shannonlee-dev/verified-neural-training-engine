import numpy as np
import pytest

from neural_engine import verification
from neural_engine.verification import (
    GRADIENT_THRESHOLD,
    numerical_gradient,
    relative_error,
    run_gradient_checks,
)


def test_compare_gradients_returns_named_relative_error():
    analytical_gradient = np.array([2.0, -4.0])

    def numerical_function(values):
        return float((values * values).sum())

    numerical_values = np.array([1.0, -2.0])
    assert hasattr(verification, "compare_gradients")
    result = verification.compare_gradients(
        "quadratic", analytical_gradient, numerical_function, numerical_values
    )
    assert result.name == "quadratic"
    assert result.relative_error < 1e-09


def test_numerical_gradient_matches_quadratic():
    values = np.array([-2.0, 3.0])
    original = values.copy()
    gradient = numerical_gradient(lambda x: float((x * x).sum()), values)
    np.testing.assert_allclose(gradient, 2 * values, rtol=1e-09, atol=1e-09)
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("fail_on_call", [1, 2, 3, 4])
def test_numerical_gradient_restores_input_on_evaluation_failure(fail_on_call):
    values = np.array([1.0, 2.0])
    original = values.copy()
    calls = 0

    def objective(array):
        nonlocal calls
        calls += 1
        if calls == fail_on_call:
            raise RuntimeError("evaluation failed")
        return float((array * array).sum())

    with pytest.raises(RuntimeError, match="evaluation failed"):
        numerical_gradient(objective, values)
    np.testing.assert_array_equal(values, original)


def test_relative_error_is_symmetric():
    analytic = np.array([1.0, -2.0])
    numerical = np.array([1.0 + 1e-08, -2.0])
    assert (
        relative_error(analytic, numerical) == relative_error(numerical, analytic)
        or round(
            abs(
                relative_error(analytic, numerical)
                - relative_error(numerical, analytic)
            ),
            7,
        )
        == 0
    )


@pytest.mark.smoke
def test_required_checks_meet_threshold():
    results = run_gradient_checks()
    required = {
        "add_broadcast",
        "multiply",
        "divide",
        "matmul",
        "sum_mean",
        "Linear.input",
        "Linear.weight",
        "Linear.bias",
        "ReLU",
        "Sigmoid",
        "Softmax",
        "BinaryCrossEntropy",
        "CrossEntropy",
    }
    assert required.issubset({result.name for result in results})
    assert max((result.relative_error for result in results)) <= GRADIENT_THRESHOLD
