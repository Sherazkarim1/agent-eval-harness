"""Clock port, so tests can measure latency deterministically."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Time source for timestamps and latency measurement."""

    def now(self) -> datetime:
        """Wall-clock timestamp (used for human-facing metadata only)."""
        ...

    def monotonic(self) -> float:
        """Monotonic timestamp in seconds (used for latency measurement)."""
        ...


class SystemClock:
    """Default clock backed by the standard library."""

    def now(self) -> datetime:
        """Current local time with timezone information."""
        return datetime.now().astimezone()

    def monotonic(self) -> float:
        """Current value of :func:`time.perf_counter`."""
        return time.perf_counter()
