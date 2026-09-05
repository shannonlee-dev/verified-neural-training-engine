from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from neural_engine.core.grad_mode import is_grad_enabled
from neural_engine.core.utils import sum_to_shape


ArrayLike = Any


class Tensor:
    """A NumPy tensor that records a dynamic reverse-mode AutoGrad graph."""

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        *,
        _parents: tuple["Tensor", ...] = (),
        _op: str = "",
    ) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = bool(requires_grad) and (
            not _parents or is_grad_enabled()
        )
        self.grad = np.zeros_like(self.data) if self.requires_grad else None
        self._parents = tuple(_parents) if self.requires_grad else ()
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def _accumulate(self, gradient: np.ndarray) -> None:
        if self.requires_grad:
            self.grad += sum_to_shape(gradient, self.shape)

    @staticmethod
    def _coerce(value: ArrayLike) -> "Tensor":
        return value if isinstance(value, Tensor) else Tensor(value)

    def backward(self, gradient: ArrayLike | None = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("cannot call backward() on a tensor without gradients")
        if gradient is None:
            if self.data.size != 1:
                raise ValueError("gradient is required for non-scalar tensors")
            seed_gradient = np.ones_like(self.data)
        else:
            seed_gradient = np.asarray(gradient, dtype=np.float64)
            if seed_gradient.shape != self.shape:
                raise ValueError(
                    f"gradient shape {seed_gradient.shape} does not match tensor shape {self.shape}"
                )

        topological: list[Tensor] = []
        visited: set[Tensor] = set()

        def visit(node: Tensor) -> None:
            if node in visited:
                return
            visited.add(node)
            for parent in node._parents:
                visit(parent)
            topological.append(node)

        visit(self)
        for node in topological:
            if node._parents and node.requires_grad:
                node.grad = np.zeros_like(node.data)

        if self._parents:
            self.grad = seed_gradient.copy()
        else:
            self.grad += seed_gradient
        for node in reversed(topological):
            if node.requires_grad:
                node._backward()

    def __add__(self, other: ArrayLike) -> "Tensor":
        other = self._coerce(other)
        output = Tensor(
            self.data + other.data,
            self.requires_grad or other.requires_grad,
            _parents=(self, other),
            _op="add",
        )

        def backward() -> None:
            self._accumulate(output.grad)
            other._accumulate(output.grad)

        if output.requires_grad:
            output._backward = backward
        return output

    def __radd__(self, other: ArrayLike) -> "Tensor":
        return self + other

    def __neg__(self) -> "Tensor":
        output = Tensor(
            -self.data,
            self.requires_grad,
            _parents=(self,),
            _op="neg",
        )

        def backward() -> None:
            self._accumulate(-output.grad)

        if output.requires_grad:
            output._backward = backward
        return output

    def __sub__(self, other: ArrayLike) -> "Tensor":
        return self + -self._coerce(other)

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        return self._coerce(other) - self

    def __mul__(self, other: ArrayLike) -> "Tensor":
        other = self._coerce(other)
        output = Tensor(
            self.data * other.data,
            self.requires_grad or other.requires_grad,
            _parents=(self, other),
            _op="multiply",
        )

        def backward() -> None:
            self._accumulate(output.grad * other.data)
            other._accumulate(output.grad * self.data)

        if output.requires_grad:
            output._backward = backward
        return output

    def __rmul__(self, other: ArrayLike) -> "Tensor":
        return self * other

    def __pow__(self, exponent: float) -> "Tensor":
        if not np.isscalar(exponent):
            raise TypeError("Tensor powers require a scalar exponent")
        output = Tensor(
            self.data**exponent,
            self.requires_grad,
            _parents=(self,),
            _op="power",
        )

        def backward() -> None:
            if exponent == 0:
                return
            self._accumulate(output.grad * exponent * self.data ** (exponent - 1))

        if output.requires_grad:
            output._backward = backward
        return output

    def __truediv__(self, other: ArrayLike) -> "Tensor":
        other = self._coerce(other)
        output = Tensor(
            self.data / other.data,
            self.requires_grad or other.requires_grad,
            _parents=(self, other),
            _op="divide",
        )

        def backward() -> None:
            self._accumulate(output.grad / other.data)
            other._accumulate(-output.grad * self.data / (other.data**2))

        if output.requires_grad:
            output._backward = backward
        return output

    def __rtruediv__(self, other: ArrayLike) -> "Tensor":
        return self._coerce(other) / self

    def __matmul__(self, other: ArrayLike) -> "Tensor":
        other = self._coerce(other)
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError("matmul currently supports two-dimensional tensors")
        output = Tensor(
            self.data @ other.data,
            self.requires_grad or other.requires_grad,
            _parents=(self, other),
            _op="matmul",
        )

        def backward() -> None:
            self._accumulate(output.grad @ other.data.T)
            other._accumulate(self.data.T @ output.grad)

        if output.requires_grad:
            output._backward = backward
        return output

    def sum(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> "Tensor":
        output = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            self.requires_grad,
            _parents=(self,),
            _op="sum",
        )

        def backward() -> None:
            gradient = output.grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else axis
                normalized = tuple(item if item >= 0 else item + self.ndim for item in axes)
                for item in sorted(normalized):
                    gradient = np.expand_dims(gradient, axis=item)
            self._accumulate(np.broadcast_to(gradient, self.shape))

        if output.requires_grad:
            output._backward = backward
        return output

    def mean(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> "Tensor":
        if axis is None:
            count = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            count = int(np.prod([self.shape[item] for item in axes]))
        return self.sum(axis=axis, keepdims=keepdims) / count

    def exp(self) -> "Tensor":
        data = np.exp(self.data)
        output = Tensor(data, self.requires_grad, _parents=(self,), _op="exp")

        def backward() -> None:
            self._accumulate(output.grad * data)

        if output.requires_grad:
            output._backward = backward
        return output

    def log(self) -> "Tensor":
        output = Tensor(
            np.log(self.data),
            self.requires_grad,
            _parents=(self,),
            _op="log",
        )

        def backward() -> None:
            self._accumulate(output.grad / self.data)

        if output.requires_grad:
            output._backward = backward
        return output

    def reshape(self, *shape: int | tuple[int, ...]) -> "Tensor":
        target = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        output = Tensor(
            self.data.reshape(target),
            self.requires_grad,
            _parents=(self,),
            _op="reshape",
        )

        def backward() -> None:
            self._accumulate(output.grad.reshape(self.shape))

        if output.requires_grad:
            output._backward = backward
        return output

    def __getitem__(self, index: Any) -> "Tensor":
        output = Tensor(
            self.data[index],
            self.requires_grad,
            _parents=(self,),
            _op="slice",
        )

        def backward() -> None:
            if self.requires_grad:
                gradient = np.zeros_like(self.data)
                np.add.at(gradient, index, output.grad)
                self.grad += gradient

        if output.requires_grad:
            output._backward = backward
        return output

    def __repr__(self) -> str:
        return f"Tensor(data={self.data!r}, requires_grad={self.requires_grad})"
