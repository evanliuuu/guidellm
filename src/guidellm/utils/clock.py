"""Unix timestamps measured on a monotonic clock.

``time.time()`` moves when the system clock steps, so a step during a
request is reported as extra latency. ``time.monotonic()`` does not.
One wall-clock reading anchors that monotonic clock on the Unix epoch.
Later timestamps are the anchor plus elapsed monotonic time, so durations
ignore a later step and reported values stay in Unix seconds.

CPython's monotonic clock is the machine clock. Anchors taken with no
step between them therefore match across processes. :func:`capture_wall_clock`
records one anchor before workers start and publishes it, so a step during
startup cannot split scheduler timestamps compared across those processes.
"""

from __future__ import annotations

import os
import time

__all__ = ["capture_wall_clock", "wall_time"]

_OFFSET_ENV = "_GUIDELLM_WALL_CLOCK_OFFSET"


class _ClockAnchor:
    offset: float | None = None


_anchor = _ClockAnchor()


def capture_wall_clock() -> float:
    """Anchor this process to the wall clock and publish that anchor for children.

    :return: ``time.time() - time.monotonic()`` at capture, in seconds
    """
    offset = time.time() - time.monotonic()
    _anchor.offset = offset
    os.environ[_OFFSET_ENV] = repr(offset)
    return offset


def wall_time() -> float:
    """Return the current Unix time, ignoring wall-clock steps after the anchor.

    A child process uses the anchor published by :func:`capture_wall_clock`
    when that variable is set. Otherwise the first call anchors this process.

    :return: Unix timestamp in seconds
    """
    offset = _anchor.offset
    if offset is None:
        raw = os.environ.get(_OFFSET_ENV)
        offset = float(raw) if raw is not None else time.time() - time.monotonic()
        _anchor.offset = offset
    return offset + time.monotonic()
