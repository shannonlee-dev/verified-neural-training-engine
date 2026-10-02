import numpy as np
import pytest

from neural_engine import no_grad
from neural_engine.core.tensor import Tensor
from neural_engine.nn.activations import ReLU, Sigmoid, Softmax
from neural_engine.nn.losses import binary_cross_entropy, cross_entropy


def test_no_grad_detaches_results_and_restores_tracking():
    x = Tensor([2.0], requires_grad=True)
    with no_grad():
        y = x * x
        assert not y.requires_grad
        assert y.grad is None
        assert y._parents == ()
        assert y._backward.__closure__ is None
    assert x.requires_grad
    (x * x).sum().backward()
    np.testing.assert_array_equal(x.grad, [4.0])


def test_no_grad_restores_outer_state_after_exception():
    x = Tensor([2.0], requires_grad=True)
    with no_grad():
        with pytest.raises(RuntimeError, match="probe"):
            with no_grad():
                raise RuntimeError("probe")
        assert not (x + 1).requires_grad
    assert (x + 1).requires_grad


def test_no_grad_preserves_explicit_leaf_requires_grad():
    with no_grad():
        leaf = Tensor([2.0], requires_grad=True)
        result = leaf + 1.0
    assert leaf.requires_grad
    assert leaf.grad is not None
    assert not result.requires_grad
    assert result._parents == ()


def test_no_grad_detaches_tensor_operations_and_nn_outputs():
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
        assert not output.requires_grad
        assert output.grad is None
        assert output._parents == ()
        assert output._backward.__closure__ is None


def test_zero_power_has_zero_gradient_at_zero():
    x = Tensor([0.0, -2.0, 3.0], requires_grad=True)
    with np.errstate(divide="raise", invalid="raise"):
        y = x**0
        np.testing.assert_array_equal(y.data, np.ones(3))
        y.backward(np.array([2.0, -1.0, 4.0]))
    np.testing.assert_array_equal(x.grad, np.zeros(3))


def test_zero_power_preserves_accumulated_leaf_gradient():
    x = Tensor([0.0], requires_grad=True)
    x.sum().backward()
    (x**0).sum().backward()
    np.testing.assert_array_equal(x.grad, [1.0])


def test_zero_grad_reuses_existing_gradient_array():
    x = Tensor([1.0, 2.0], requires_grad=True)
    gradient = x.grad
    x.grad[:] = [3.0, 4.0]
    x.zero_grad()
    assert x.grad is gradient
    np.testing.assert_array_equal(x.grad, [0.0, 0.0])


def test_graph_parents_preserve_operation_input_order():
    left = Tensor([1.0], requires_grad=True)
    right = Tensor([2.0], requires_grad=True)
    output = left * right
    assert isinstance(output._parents, tuple)
    assert output._parents == (left, right)


def test_backward_accumulates_through_shared_graph():
    x = Tensor([2.0], requires_grad=True)
    y = x * x + x
    y.backward()
    np.testing.assert_allclose(x.grad, [5.0])
    y.backward()
    np.testing.assert_allclose(x.grad, [10.0])


def test_broadcast_gradient_returns_original_shape():
    x = Tensor(np.ones((2, 3)), requires_grad=True)
    bias = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    (x + bias).sum().backward()
    np.testing.assert_allclose(x.grad, np.ones((2, 3)))
    np.testing.assert_allclose(bias.grad, [2.0, 2.0, 2.0])


def test_required_tensor_operations_propagate_gradients():
    left = Tensor([[1.0, 2.0]], requires_grad=True)
    right = Tensor([[3.0], [4.0]], requires_grad=True)
    (left @ right / 2.0).mean().backward()
    np.testing.assert_allclose(left.grad, [[1.5, 2.0]])
    np.testing.assert_allclose(right.grad, [[0.5], [1.0]])


def test_exp_log_and_indexing_propagate():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    x.exp().log()[[0, 2]].sum().backward()
    np.testing.assert_allclose(x.grad, [1.0, 0.0, 1.0])


def test_non_scalar_backward_requires_gradient():
    with pytest.raises(ValueError):
        Tensor([1.0, 2.0], requires_grad=True).backward()


def test_backward_rejects_wrong_gradient_shape():
    x = Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(ValueError):
        x.backward(np.array([1.0]))


def test_power_and_subtraction_gradients():
    x = Tensor([2.0, 3.0], requires_grad=True)
    ((x - 1.0) ** 2).sum().backward()
    np.testing.assert_allclose(x.grad, [2.0, 4.0])


def test_reshape_propagates_original_shape():
    x = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
    x.reshape(3, 2).sum().backward()
    np.testing.assert_allclose(x.grad, np.ones((2, 3)))
