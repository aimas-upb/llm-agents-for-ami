"""
Non-blocking I/O for behaviour-tree leaves.

A tick must not block. When execution moves into a SPADE behaviour that ticks
once per `run()`, any node that waits inline for an HTTP round trip stalls the
whole agent for the duration: no other request is served, no status query is
answered, no cancel is honoured. The tree cannot even be cancelled *between*
ticks if a single tick lasts three seconds.

The fix is the idiom py_trees is built around — a leaf that has not finished
returns RUNNING and is asked again on the next tick:

    def update(self):
        self._start(self._blocking_io)     # submits once; a no-op afterwards
        done, result, exc = self._poll()
        if not done:
            return Status.RUNNING          # <- the whole point
        self._reset()
        ...                                # interpret result -> SUCCESS/FAILURE

The work itself still happens on a thread, because the HTTP client underneath
is synchronous. What changes is who waits: the executor thread does, and the
tick returns immediately.

`shutdown_executor()` exists for tests and for a clean process exit; the pool is
a daemon-threaded module singleton otherwise, so it never keeps the interpreter
alive on its own.
"""

from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional, Tuple

from ...shared.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger("BTNodes")

# Sized for concurrent maintenance plans, each polling a property or two. Too
# small and plans starve each other -- a tick that cannot get a worker just
# stays RUNNING, so the symptom is a tree that mysteriously makes no progress
# rather than an error. Override with BT_EXECUTOR_MAX_WORKERS.
_DEFAULT_MAX_WORKERS = 16

_executor: Optional[ThreadPoolExecutor] = None


def get_executor() -> ThreadPoolExecutor:
    """The shared pool that node I/O runs on, created on first use."""
    global _executor
    if _executor is None:
        max_workers = int(os.getenv("BT_EXECUTOR_MAX_WORKERS",
                                    _DEFAULT_MAX_WORKERS))
        _executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="bt-node")
        logger.debug(f"BT node executor started (max_workers={max_workers})")
    return _executor


def shutdown_executor(wait: bool = True) -> None:
    """Tear the pool down. Safe to call when it was never started."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=wait)
        _executor = None


class FuturePollingMixin:
    """Submit blocking work once, then poll it across ticks.

    Mixed into a `py_trees.behaviour.Behaviour`. The contract is three calls:

    - `_start(fn, *args)` submits `fn` unless something is already in flight,
      so calling it on every tick is correct and idempotent.
    - `_poll()` returns `(done, result, exception)` without blocking.
    - `_reset()` clears the slot, readying the node for its next attempt --
      which is what a polling node does between attempts, and what
      `initialise()` does between runs.
    """

    _future: Optional[Future] = None

    def _start(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Submit `fn` if nothing is in flight. Idempotent per attempt."""
        if self._future is None:
            self._future = get_executor().submit(fn, *args, **kwargs)

    def _poll(self) -> Tuple[bool, Any, Optional[BaseException]]:
        """Has it finished? Never blocks.

        Returns `(done, result, exception)`. While pending: `(False, None,
        None)`. On completion the result or the exception is handed back, and
        the exception is returned rather than raised so the caller decides
        whether it means FAILURE or another attempt.
        """
        future = self._future
        if future is None or not future.done():
            return False, None, None
        try:
            return True, future.result(), None
        except BaseException as exc:  # noqa: BLE001 - the caller interprets it
            return True, None, exc

    def _reset(self) -> None:
        """Drop the finished (or abandoned) attempt."""
        self._future = None

    def _cancel_pending(self) -> None:
        """Best-effort cancel, for `terminate()`.

        A future already running cannot be cancelled -- the HTTP call will
        finish and its result be discarded, which is the right trade: no
        request is left half-sent, and the node stops caring about the answer.
        """
        if self._future is not None:
            self._future.cancel()
            self._future = None

    @property
    def _io_in_flight(self) -> bool:
        return self._future is not None and not self._future.done()
