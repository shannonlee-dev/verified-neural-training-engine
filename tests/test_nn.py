import unittest

import numpy as np

from neural_engine.core.tensor import Tensor
from neural_engine.nn.activations import ReLU, Sigmoid, Softmax
from neural_engine.nn.initialization import initialize_weights
from neural_engine.nn.layers import Linear
from neural_engine.nn.losses import binary_cross_entropy, cross_entropy
from neural_engine.nn.module import Sequential


class NeuralNetworkTests(unittest.TestCase):
    def test_initializers_have_expected_values_and_scales(self):
        rng = np.random.default_rng(42)

        zero = initialize_weights(1000, 1000, "zero", rng)
        random = initialize_weights(1000, 1000, "random", rng)
        he = initialize_weights(1000, 1000, "he", rng)
        xavier = initialize_weights(1000, 1000, "xavier", rng)

        self.assertTrue(np.all(zero == 0))
        self.assertAlmostEqual(float(random.std()), 1.0, delta=0.01)
        self.assertAlmostEqual(float(he.std()), np.sqrt(2 / 1000), delta=0.001)
        self.assertAlmostEqual(float(xavier.std()), np.sqrt(2 / 2000), delta=0.001)

    def test_invalid_initializer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "initialization"):
            initialize_weights(2, 3, "unknown", np.random.default_rng(42))

    def test_sequential_collects_linear_parameters_and_runs_forward(self):
        rng = np.random.default_rng(42)
        model = Sequential(
            Linear(2, 3, rng=rng),
            ReLU(),
            Linear(3, 1, rng=rng),
        )

        output = model(Tensor([[1.0, -1.0]]))

        self.assertEqual(output.shape, (1, 1))
        self.assertEqual(len(model.parameters()), 4)

    def test_linear_propagates_input_weight_and_bias_gradients(self):
        layer = Linear(2, 1, initialization="zero")
        layer.weight.data[:] = [[2.0], [3.0]]
        layer.bias.data[:] = [1.0]
        x = Tensor([[4.0, 5.0]], requires_grad=True)

        layer(x).sum().backward()

        np.testing.assert_allclose(x.grad, [[2.0, 3.0]])
        np.testing.assert_allclose(layer.weight.grad, [[4.0], [5.0]])
        np.testing.assert_allclose(layer.bias.grad, [1.0])

    def test_relu_uses_zero_gradient_for_negative_values(self):
        x = Tensor([-1.2, -0.3, 0.4, 1.5], requires_grad=True)

        ReLU()(x).sum().backward()

        np.testing.assert_allclose(x.grad, [0.0, 0.0, 1.0, 1.0])

    def test_sigmoid_matches_known_value_and_gradient(self):
        x = Tensor([0.0], requires_grad=True)

        output = Sigmoid()(x)
        output.backward()

        np.testing.assert_allclose(output.data, [0.5])
        np.testing.assert_allclose(x.grad, [0.25])

    def test_softmax_rows_are_probabilities_and_backpropagate(self):
        x = Tensor([[1000.0, 1001.0, 1002.0]], requires_grad=True)

        probabilities = Softmax()(x)
        probabilities.backward(np.array([[1.0, 2.0, 3.0]]))

        np.testing.assert_allclose(probabilities.data.sum(axis=1), [1.0])
        self.assertTrue(np.isfinite(probabilities.data).all())
        np.testing.assert_allclose(x.grad.sum(axis=1), [0.0], atol=1e-12)

    def test_cross_entropy_is_stable_and_backpropagates(self):
        logits = Tensor([[1000.0, 1001.0, 1002.0]], requires_grad=True)

        loss = cross_entropy(logits, np.array([2]))
        loss.backward()

        self.assertAlmostEqual(float(loss.data), 0.4076059644, places=9)
        self.assertTrue(np.isfinite(logits.grad).all())
        np.testing.assert_allclose(logits.grad.sum(axis=1), [0.0], atol=1e-12)

    def test_cross_entropy_remains_finite_for_underflowing_target_probability(self):
        logits = Tensor([[-1000.0, 1000.0]], requires_grad=True)

        loss = cross_entropy(logits, np.array([0]))
        loss.backward()

        self.assertAlmostEqual(float(loss.data), 2000.0)
        np.testing.assert_allclose(logits.grad, [[-1.0, 1.0]])

    def test_binary_cross_entropy_matches_hand_calculation(self):
        probabilities = Tensor([[0.8], [0.25]], requires_grad=True)
        targets = np.array([[1.0], [0.0]])

        loss = binary_cross_entropy(probabilities, targets)
        loss.backward()

        expected = -(np.log(0.8) + np.log(0.75)) / 2
        self.assertAlmostEqual(float(loss.data), expected)
        np.testing.assert_allclose(probabilities.grad, [[-0.625], [2 / 3]])


if __name__ == "__main__":
    unittest.main()
