import unittest

import numpy as np

from neural_engine import no_grad
from neural_engine.core.tensor import Tensor
from neural_engine.nn.activations import ReLU, Sigmoid, Softmax
from neural_engine.nn.losses import binary_cross_entropy, cross_entropy


class TensorTests(unittest.TestCase):
    def test_no_grad_detaches_results_and_restores_tracking(self):
        x = Tensor([2.0], requires_grad=True)

        with no_grad():
            y = x * x
            self.assertFalse(y.requires_grad)
            self.assertIsNone(y.grad)
            self.assertEqual(y._parents, ())
            self.assertIsNone(y._backward.__closure__)

        self.assertTrue(x.requires_grad)
        (x * x).sum().backward()
        np.testing.assert_array_equal(x.grad, [4.0])

    def test_no_grad_restores_outer_state_after_exception(self):
        x = Tensor([2.0], requires_grad=True)

        with no_grad():
            with self.assertRaisesRegex(RuntimeError, "probe"):
                with no_grad():
                    raise RuntimeError("probe")
            self.assertFalse((x + 1).requires_grad)

        self.assertTrue((x + 1).requires_grad)

    def test_no_grad_preserves_explicit_leaf_requires_grad(self):
        with no_grad():
            leaf = Tensor([2.0], requires_grad=True)
            result = leaf + 1.0

        self.assertTrue(leaf.requires_grad)
        self.assertIsNotNone(leaf.grad)
        self.assertFalse(result.requires_grad)
        self.assertEqual(result._parents, ())

    def test_no_grad_detaches_tensor_operations_and_nn_outputs(self):
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        matrix = Tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)

        with no_grad():
            outputs = [
                x + 1.0,
                x - 1.0,
                -x,
                x * 2.0,
                x / 2.0,
                x**2,
                x @ matrix,
                x.sum(),
                x.mean(),
                x.exp(),
                x.log(),
                x.reshape(4),
                x[[0]],
                ReLU()(x),
                Sigmoid()(x),
                Softmax()(x),
                binary_cross_entropy(Sigmoid()(x), np.ones_like(x.data)),
                cross_entropy(x, np.array([0, 1])),
            ]

        for output in outputs:
            with self.subTest(operation=output._op):
                self.assertFalse(output.requires_grad)
                self.assertIsNone(output.grad)
                self.assertEqual(output._parents, ())
                self.assertIsNone(output._backward.__closure__)

    def test_zero_power_has_zero_gradient_at_zero(self):
        x = Tensor([0.0, -2.0, 3.0], requires_grad=True)

        with np.errstate(divide="raise", invalid="raise"):
            y = x**0
            np.testing.assert_array_equal(y.data, np.ones(3))
            y.backward(np.array([2.0, -1.0, 4.0]))

        np.testing.assert_array_equal(x.grad, np.zeros(3))

    def test_zero_power_preserves_accumulated_leaf_gradient(self):
        x = Tensor([0.0], requires_grad=True)

        x.sum().backward()
        (x**0).sum().backward()

        np.testing.assert_array_equal(x.grad, [1.0])

    def test_zero_grad_reuses_existing_gradient_array(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        gradient = x.grad
        x.grad[:] = [3.0, 4.0]

        x.zero_grad()

        self.assertIs(x.grad, gradient)
        np.testing.assert_array_equal(x.grad, [0.0, 0.0])

    def test_graph_parents_preserve_operation_input_order(self):
        left = Tensor([1.0], requires_grad=True)
        right = Tensor([2.0], requires_grad=True)

        output = left * right

        self.assertIsInstance(output._parents, tuple)
        self.assertEqual(output._parents, (left, right))

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
