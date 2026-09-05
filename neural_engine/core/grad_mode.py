from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar


_grad_enabled: ContextVar[bool] = ContextVar("grad_enabled", default=True)


def is_grad_enabled() -> bool:
    return _grad_enabled.get()


@contextmanager
def no_grad() -> Generator[None, None, None]:
    token = _grad_enabled.set(False)
    try:
        yield
    finally:
        _grad_enabled.reset(token)
