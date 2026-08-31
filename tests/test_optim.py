import unittest

import numpy as np

from neural_engine.core.tensor import Tensor
from neural_engine.optim.adam import Adam
from neural_engine.optim.sgd import SGD


class OptimizerTests(unittest.TestCase):
    def test_sgd_updates_parameter_and_zeroes_gradient(self):
        parameter = Tensor([2.0], requires_grad=True)
        parameter.grad = np.array([0.5])
        optimizer = SGD([parameter], lr=0.1)

        optimizer.step()
        np.testing.assert_allclose(parameter.data, [1.95])

        optimizer.zero_grad()
        np.testing.assert_allclose(parameter.grad, [0.0])

    def test_first_adam_step_matches_bias_corrected_formula(self):
        parameter = Tensor([2.0], requires_grad=True)
        parameter.grad = np.array([0.5])

        Adam([parameter], lr=0.1).step()

        np.testing.assert_allclose(parameter.data, [1.9], rtol=1e-7, atol=1e-8)

    def test_adam_keeps_independent_state_for_each_parameter(self):
        first = Tensor([1.0], requires_grad=True)
        second = Tensor([1.0], requires_grad=True)
        first.grad = np.array([1.0])
        second.grad = np.array([-1.0])

        Adam([first, second], lr=0.01).step()

        self.assertLess(first.data.item(), 1.0)
        self.assertGreater(second.data.item(), 1.0)

    def test_optimizers_reject_nonpositive_learning_rate(self):
        parameter = Tensor([1.0], requires_grad=True)

        with self.assertRaises(ValueError):
            SGD([parameter], lr=0.0)
        with self.assertRaises(ValueError):
            Adam([parameter], lr=-0.1)


if __name__ == "__main__":
    unittest.main()
