from __future__ import annotations

from typing import Iterable, Tuple, TypeVar

T = TypeVar("T")


def to_2tuple(x: T | Iterable[T]) -> Tuple[T, T]:
    if isinstance(x, tuple) and len(x) == 2:
        return x[0], x[1]
    if isinstance(x, list) and len(x) == 2:
        return x[0], x[1]
    return x, x

