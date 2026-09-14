from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")
E = TypeVar("E")


@dataclass(frozen=True)
class Ok(Generic[T]):
    """Success result containing a value."""

    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False


@dataclass(frozen=True)
class Err(Generic[E]):
    """Error result containing an error."""

    error: E

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True


Result = Ok[T] | Err[E]


def unwrap(result: Result[T, E]) -> T:
    """Unwrap a result, raising if error."""
    if isinstance(result, Err):
        raise ValueError(f"Result contains error: {result.error}")

    return result.value


def unwrap_or(result: Result[T, E], default: T) -> T:
    """Unwrap a result or return default if error."""
    if isinstance(result, Ok):
        return result.value
    return default


def unwrap_or_else(result: Result[T, E], f: Callable[[], T]) -> T:
    """Unwrap a result or compute default via function if error."""
    if isinstance(result, Ok):
        return result.value
    return f()
