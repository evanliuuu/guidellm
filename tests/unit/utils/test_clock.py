"""Tests for monotonic wall-clock timestamps."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from multiprocessing.connection import Connection

import pytest

from guidellm.utils import clock
from guidellm.utils.clock import capture_wall_clock, wall_time


@pytest.fixture(autouse=True)
def _fresh_clock():
    """Give each test its own anchor and restore the process clock afterwards."""
    previous_offset = clock._anchor.offset
    previous_env = os.environ.get(clock._OFFSET_ENV)
    clock._anchor.offset = None
    os.environ.pop(clock._OFFSET_ENV, None)
    yield
    clock._anchor.offset = previous_offset
    if previous_env is None:
        os.environ.pop(clock._OFFSET_ENV, None)
    else:
        os.environ[clock._OFFSET_ENV] = previous_env


@pytest.mark.smoke
def test_wall_time_matches_unix_clock():
    """The anchored timestamp stays on the Unix clock.

    ## WRITTEN BY AI ##
    """
    assert abs(wall_time() - time.time()) < 0.05


@pytest.mark.regression
def test_wall_clock_step_does_not_change_elapsed(monkeypatch: pytest.MonkeyPatch):
    """A later step in time.time() does not move wall_time().

    ## WRITTEN BY AI ##
    """
    start = wall_time()
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 3.0)

    elapsed = wall_time() - start

    assert elapsed < 1.0
    assert abs(wall_time() - real_time()) < 1.0


def _child_wall_time(conn: Connection, step: bool) -> None:
    if step:
        real_time = time.time

        def stepped_time() -> float:
            return real_time() + 3.0

        time.time = stepped_time
    conn.send(wall_time())
    conn.close()


@pytest.mark.regression
@pytest.mark.parametrize("step", [False, True])
def test_spawned_process_uses_published_anchor(step: bool):
    """A spawned process reports the parent's anchor, not its own stepped clock.

    ## WRITTEN BY AI ##
    """
    capture_wall_clock()
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child_wall_time, args=(child, step))
    proc.start()
    child.close()
    try:
        stamp = parent.recv()
    finally:
        proc.join(timeout=30)

    assert proc.exitcode == 0
    assert abs(stamp - time.time()) < 1.0, (
        f"child timestamp {stamp:.3f} diverged from the wall clock"
    )
