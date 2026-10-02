from __future__ import annotations

from collections.abc import Iterable

from neural_engine.core.tensor import Tensor
from neural_engine.optim.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, parameters: Iterable[Tensor], lr: float = 0.01) -> None:
        super().__init__(parameters, lr)

    def step(self) -> None:
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.data -= self.lr * parameter.grad
