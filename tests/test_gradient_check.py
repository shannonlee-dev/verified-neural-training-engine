import unittest

import numpy as np

from neural_engine.verification import (
    GRADIENT_THRESHOLD,
    numerical_gradient,
    relative_error,
    run_gradient_checks,
)


class GradientCheckTests(unittest.TestCase):
    def test_numerical_gradient_matches_quadratic(self):
        values = np.array([-2.0, 3.0])

        gradient = numerical_gradient(lambda x: float((x * x).sum()), values)

        np.testing.assert_allclose(gradient, 2 * values, rtol=1e-9, atol=1e-9)

    def test_relative_error_is_symmetric(self):
        analytic = np.array([1.0, -2.0])
        numerical = np.array([1.0 + 1e-8, -2.0])

        self.assertAlmostEqual(
            relative_error(analytic, numerical),
            relative_error(numerical, analytic),
        )

    def test_required_checks_meet_threshold(self):
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
        }

        self.assertTrue(required.issubset({result.name for result in results}))
        self.assertLessEqual(
            max(result.relative_error for result in results), GRADIENT_THRESHOLD
        )


if __name__ == "__main__":
    unittest.main()
