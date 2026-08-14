"""Shared cooperative-cancellation contract for background operations."""

from __future__ import annotations

from collections.abc import Callable

CancellationCheck = Callable[[], bool]


class OperationCancelledError(RuntimeError):
    """Raised at a safe boundary when a caller requests cancellation."""


def raise_if_cancelled(
    cancelled: CancellationCheck | None,
    *,
    message: str,
) -> None:
    """Raise the dedicated cancellation result without colliding with I/O errors."""

    if cancelled is not None and cancelled():
        raise OperationCancelledError(message)
