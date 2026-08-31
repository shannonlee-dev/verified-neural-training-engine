from __future__ import annotations

import numpy as np

from neural_engine.core.tensor import Tensor
from neural_engine.nn.module import Module


class ReLU(Module):
    def forward(self, inputs: Tensor) -> Tensor:
        output = Tensor(
            np.maximum(inputs.data, 0.0),
            inputs.requires_grad,
            _children=(inputs,),
            _op="relu",
        )

        def backward() -> None:
            inputs._accumulate(output.grad * (inputs.data > 0.0))

        output._backward = backward
        return output


class Sigmoid(Module):
    def forward(self, inputs: Tensor) -> Tensor:
        values = np.empty_like(inputs.data)
        positive = inputs.data >= 0
        values[positive] = 1.0 / (1.0 + np.exp(-inputs.data[positive]))
        negative_exp = np.exp(inputs.data[~positive])
        values[~positive] = negative_exp / (1.0 + negative_exp)
        output = Tensor(
            values,
            inputs.requires_grad,
            _children=(inputs,),
            _op="sigmoid",
        )

        def backward() -> None:
            inputs._accumulate(output.grad * values * (1.0 - values))

        output._backward = backward
        return output


class Softmax(Module):
    def __init__(self, axis: int = -1) -> None:
        self.axis = axis

    def forward(self, inputs: Tensor) -> Tensor:
        shifted = inputs.data - np.max(inputs.data, axis=self.axis, keepdims=True)
        exponentials = np.exp(shifted)
        probabilities = exponentials / exponentials.sum(axis=self.axis, keepdims=True)
        output = Tensor(
            probabilities,
            inputs.requires_grad,
            _children=(inputs,),
            _op="softmax",
        )

        def backward() -> None:
            dot = np.sum(output.grad * probabilities, axis=self.axis, keepdims=True)
            inputs._accumulate(probabilities * (output.grad - dot))

        output._backward = backward
        return output
