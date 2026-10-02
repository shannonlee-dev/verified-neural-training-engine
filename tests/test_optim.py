import numpy as np
import pytest

from neural_engine.core.tensor import Tensor
from neural_engine.optim.adam import Adam
from neural_engine.optim.sgd import SGD


def test_sgd_updates_parameter_and_zeroes_gradient():
    parameter = Tensor([2.0], requires_grad=True)
    parameter.grad = np.array([0.5])
    optimizer = SGD([parameter], lr=0.1)
    optimizer.step()
    np.testing.assert_allclose(parameter.data, [1.95])
    optimizer.zero_grad()
    np.testing.assert_allclose(parameter.grad, [0.0])


def test_first_adam_step_matches_bias_corrected_formula():
    parameter = Tensor([2.0], requires_grad=True)
    parameter.grad = np.array([0.5])
    Adam([parameter], lr=0.1).step()
    np.testing.assert_allclose(parameter.data, [1.9], rtol=1e-07, atol=1e-08)


def test_adam_keeps_independent_state_for_each_parameter():
    first = Tensor([1.0], requires_grad=True)
    second = Tensor([1.0], requires_grad=True)
    first.grad = np.array([1.0])
    second.grad = np.array([-1.0])
    Adam([first, second], lr=0.01).step()
    assert first.data.item() < 1.0
    assert second.data.item() > 1.0


def test_optimizers_reject_nonpositive_learning_rate():
    parameter = Tensor([1.0], requires_grad=True)
    with pytest.raises(ValueError):
        SGD([parameter], lr=0.0)
    with pytest.raises(ValueError):
        Adam([parameter], lr=-0.1)
