from __future__ import annotations

import numpy as np

from neural_engine.core.tensor import Tensor
from neural_engine.nn.initialization import initialize_weights
from neural_engine.nn.module import Module


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        initialization: str = "he",
        rng: np.random.Generator | None = None,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(
            initialize_weights(in_features, out_features, initialization, rng),
            requires_grad=True,
        )
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.in_features:
            raise ValueError(
                f"Linear expected shape (batch, {self.in_features}), got {inputs.shape}"
            )
        return inputs @ self.weight + self.bias
