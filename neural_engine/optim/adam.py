from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from neural_engine.core.tensor import Tensor
from neural_engine.optim.optimizer import Optimizer


class Adam(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        super().__init__(parameters, lr)
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("Adam betas must be in [0, 1)")
        if eps <= 0:
            raise ValueError("Adam epsilon must be positive")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.step_count = 0
        self.first_moments = [np.zeros_like(item.data) for item in self.parameters]
        self.second_moments = [np.zeros_like(item.data) for item in self.parameters]

    def step(self) -> None:
        self.step_count += 1
        for index, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue
            gradient = parameter.grad
            first = self.first_moments[index]
            second = self.second_moments[index]
            first *= self.beta1
            first += (1.0 - self.beta1) * gradient
            second *= self.beta2
            second += (1.0 - self.beta2) * gradient**2

            corrected_first = first / (1.0 - self.beta1**self.step_count)
            corrected_second = second / (1.0 - self.beta2**self.step_count)
            parameter.data -= self.lr * corrected_first / (
                np.sqrt(corrected_second) + self.eps
            )
