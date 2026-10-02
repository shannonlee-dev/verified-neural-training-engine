import numpy as np
import pytest

from neural_engine.core.tensor import Tensor
from neural_engine.nn.activations import ReLU, Sigmoid, Softmax
from neural_engine.nn.initialization import initialize_weights
from neural_engine.nn.layers import Linear
from neural_engine.nn.losses import binary_cross_entropy, cross_entropy
from neural_engine.nn.module import Sequential


def test_initializers_have_expected_values_and_scales():
    rng = np.random.default_rng(42)
    zero = initialize_weights(1000, 1000, "zero", rng)
    random = initialize_weights(1000, 1000, "random", rng)
    he = initialize_weights(1000, 1000, "he", rng)
    xavier = initialize_weights(1000, 1000, "xavier", rng)
    assert np.all(zero == 0)
    assert abs(float(random.std()) - 1.0) <= 0.01
    assert abs(float(he.std()) - np.sqrt(2 / 1000)) <= 0.001
    assert abs(float(xavier.std()) - np.sqrt(2 / 2000)) <= 0.001


def test_invalid_initializer_is_rejected():
    with pytest.raises(ValueError, match="initialization"):
        initialize_weights(2, 3, "unknown", np.random.default_rng(42))


def test_initializer_without_rng_does_not_restart_a_fixed_sequence():
    first = initialize_weights(8, 8, "he")
    second = initialize_weights(8, 8, "he")
    assert not np.array_equal(first, second)


def test_sequential_collects_linear_parameters_and_runs_forward():
    rng = np.random.default_rng(42)
    model = Sequential(Linear(2, 3, rng=rng), ReLU(), Linear(3, 1, rng=rng))
    output = model(Tensor([[1.0, -1.0]]))
    assert output.shape == (1, 1)
    assert len(model.parameters()) == 4


def test_linear_propagates_input_weight_and_bias_gradients():
    layer = Linear(2, 1, initialization="zero")
    layer.weight.data[:] = [[2.0], [3.0]]
    layer.bias.data[:] = [1.0]
    x = Tensor([[4.0, 5.0]], requires_grad=True)
    layer(x).sum().backward()
    np.testing.assert_allclose(x.grad, [[2.0, 3.0]])
    np.testing.assert_allclose(layer.weight.grad, [[4.0], [5.0]])
    np.testing.assert_allclose(layer.bias.grad, [1.0])


def test_relu_uses_zero_gradient_for_negative_values():
    x = Tensor([-1.2, -0.3, 0.4, 1.5], requires_grad=True)
    ReLU()(x).sum().backward()
    np.testing.assert_allclose(x.grad, [0.0, 0.0, 1.0, 1.0])


def test_sigmoid_matches_known_value_and_gradient():
    x = Tensor([0.0], requires_grad=True)
    output = Sigmoid()(x)
    output.backward()
    np.testing.assert_allclose(output.data, [0.5])
    np.testing.assert_allclose(x.grad, [0.25])


def test_softmax_rows_are_probabilities_and_backpropagate():
    x = Tensor([[1000.0, 1001.0, 1002.0]], requires_grad=True)
    probabilities = Softmax()(x)
    probabilities.backward(np.array([[1.0, 2.0, 3.0]]))
    np.testing.assert_allclose(probabilities.data.sum(axis=1), [1.0])
    assert np.isfinite(probabilities.data).all()
    np.testing.assert_allclose(x.grad.sum(axis=1), [0.0], atol=1e-12)


def test_cross_entropy_is_stable_and_backpropagates():
    logits = Tensor([[1000.0, 1001.0, 1002.0]], requires_grad=True)
    loss = cross_entropy(logits, np.array([2]))
    loss.backward()
    assert (
        float(loss.data) == 0.4076059644
        or round(abs(float(loss.data) - 0.4076059644), 9) == 0
    )
    assert np.isfinite(logits.grad).all()
    np.testing.assert_allclose(logits.grad.sum(axis=1), [0.0], atol=1e-12)


def test_cross_entropy_remains_finite_for_underflowing_target_probability():
    logits = Tensor([[-1000.0, 1000.0]], requires_grad=True)
    loss = cross_entropy(logits, np.array([0]))
    loss.backward()
    assert float(loss.data) == 2000.0 or round(abs(float(loss.data) - 2000.0), 7) == 0
    np.testing.assert_allclose(logits.grad, [[-1.0, 1.0]])


def test_binary_cross_entropy_matches_hand_calculation():
    probabilities = Tensor([[0.8], [0.25]], requires_grad=True)
    targets = np.array([[1.0], [0.0]])
    loss = binary_cross_entropy(probabilities, targets)
    loss.backward()
    expected = -(np.log(0.8) + np.log(0.75)) / 2
    assert (
        float(loss.data) == expected or round(abs(float(loss.data) - expected), 7) == 0
    )
    np.testing.assert_allclose(probabilities.grad, [[-0.625], [2 / 3]])


def test_binary_cross_entropy_clipping_has_consistent_boundary_gradient():
    probabilities = Tensor([[0.0], [1.0]], requires_grad=True)
    targets = np.array([[0.0], [1.0]])
    loss = binary_cross_entropy(probabilities, targets)
    loss.backward()
    assert np.isfinite(loss.data)
    np.testing.assert_allclose(probabilities.grad, [[0.0], [0.0]])


def test_binary_cross_entropy_rejects_values_outside_probability_range():
    probabilities = Tensor([[-0.1], [1.1]], requires_grad=True)
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        binary_cross_entropy(probabilities, np.array([[0.0], [1.0]]))
