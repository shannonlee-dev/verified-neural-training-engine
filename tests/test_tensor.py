import unittest

import numpy as np

from neural_engine.core.tensor import Tensor


class TensorTests(unittest.TestCase):
    def test_graph_parents_preserve_operation_input_order(self):
        left = Tensor([1.0], requires_grad=True)
        right = Tensor([2.0], requires_grad=True)

        output = left * right

        self.assertIsInstance(output._prev, tuple)
        self.assertEqual(output._prev, (left, right))

    def test_backward_accumulates_through_shared_graph(self):
        x = Tensor([2.0], requires_grad=True)
        y = x * x + x

        y.backward()
        np.testing.assert_allclose(x.grad, [5.0])

        y.backward()
        np.testing.assert_allclose(x.grad, [10.0])

    def test_broadcast_gradient_returns_original_shape(self):
        x = Tensor(np.ones((2, 3)), requires_grad=True)
        bias = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)

        (x + bias).sum().backward()

        np.testing.assert_allclose(x.grad, np.ones((2, 3)))
        np.testing.assert_allclose(bias.grad, [2.0, 2.0, 2.0])

    def test_required_tensor_operations_propagate_gradients(self):
        left = Tensor([[1.0, 2.0]], requires_grad=True)
        right = Tensor([[3.0], [4.0]], requires_grad=True)

        ((left @ right) / 2.0).mean().backward()

        np.testing.assert_allclose(left.grad, [[1.5, 2.0]])
        np.testing.assert_allclose(right.grad, [[0.5], [1.0]])

    def test_exp_log_and_indexing_propagate(self):
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)

        x.exp().log()[[0, 2]].sum().backward()

        np.testing.assert_allclose(x.grad, [1.0, 0.0, 1.0])

    def test_non_scalar_backward_requires_gradient(self):
        with self.assertRaises(ValueError):
            Tensor([1.0, 2.0], requires_grad=True).backward()

    def test_backward_rejects_wrong_gradient_shape(self):
        x = Tensor([1.0, 2.0], requires_grad=True)

        with self.assertRaises(ValueError):
            x.backward(np.array([1.0]))

    def test_power_and_subtraction_gradients(self):
        x = Tensor([2.0, 3.0], requires_grad=True)

        ((x - 1.0) ** 2).sum().backward()

        np.testing.assert_allclose(x.grad, [2.0, 4.0])

    def test_reshape_propagates_original_shape(self):
        x = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)

        x.reshape(3, 2).sum().backward()

        np.testing.assert_allclose(x.grad, np.ones((2, 3)))


if __name__ == "__main__":
    unittest.main()
