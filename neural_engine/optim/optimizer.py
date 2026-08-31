from __future__ import annotations

from collections.abc import Iterable

from neural_engine.core.tensor import Tensor


class Optimizer:
    def __init__(self, parameters: Iterable[Tensor], lr: float) -> None:
        if lr <= 0:
            raise ValueError("learning rate must be positive")
        self.parameters = list(parameters)
        if not all(isinstance(parameter, Tensor) for parameter in self.parameters):
            raise TypeError("optimizer parameters must be Tensor instances")
        self.lr = float(lr)

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self) -> None:
        raise NotImplementedError
