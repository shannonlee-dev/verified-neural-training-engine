from __future__ import annotations

from collections.abc import Iterator

from neural_engine.core.tensor import Tensor


class Module:
    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)

    def parameters(self) -> list[Tensor]:
        parameters: list[Tensor] = []
        seen: set[int] = set()

        def collect(value: object) -> Iterator[Tensor]:
            if isinstance(value, Tensor) and value.requires_grad:
                yield value
            elif isinstance(value, Module):
                for nested in value.__dict__.values():
                    yield from collect(nested)
            elif isinstance(value, (list, tuple)):
                for nested in value:
                    yield from collect(nested)

        for value in self.__dict__.values():
            for parameter in collect(value):
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    parameters.append(parameter)
        return parameters


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        self.layers = list(layers)

    def forward(self, inputs: Tensor) -> Tensor:
        output = inputs
        for layer in self.layers:
            output = layer(output)
        return output
