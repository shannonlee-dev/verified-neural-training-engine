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
        original = values.copy()

        gradient = numerical_gradient(lambda x: float((x * x).sum()), values)

        np.testing.assert_allclose(gradient, 2 * values, rtol=1e-9, atol=1e-9)
        np.testing.assert_array_equal(values, original)

    def test_numerical_gradient_restores_input_on_evaluation_failure(self):
        for fail_on_call in (1, 2, 3, 4):
            with self.subTest(fail_on_call=fail_on_call):
                values = np.array([1.0, 2.0])
                original = values.copy()
                calls = 0

                def objective(array):
                    nonlocal calls
                    calls += 1
                    if calls == fail_on_call:
                        raise RuntimeError("evaluation failed")
                    return float((array * array).sum())

                with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                    numerical_gradient(objective, values)
                np.testing.assert_array_equal(values, original)

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
            "BinaryCrossEntropy",
            "CrossEntropy",
        }

        self.assertTrue(required.issubset({result.name for result in results}))
        self.assertLessEqual(
            max(result.relative_error for result in results), GRADIENT_THRESHOLD
        )


if __name__ == "__main__":
    unittest.main()
