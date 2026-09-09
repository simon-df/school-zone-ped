"""Threading utilities: Worker wrapper and Queue-based UI communication."""

from __future__ import annotations

import queue
import threading
import traceback
from typing import Any, Callable

import logging
logger = logging.getLogger(__name__)

# Sentinel object used to signal that a worker has finished.
_SENTINEL = object()


class WorkerTask:
    """Run a callable in a background thread and relay results via a queue.

    The queue receives dictionaries with the following keys:

    * ``"type"`` – one of ``"progress"``, ``"preview"``, ``"status"``,
      ``"result"``, ``"error"``, ``"done"``
    * ``"payload"`` – the value associated with the message type.

    UI code should poll or bind to this queue and handle messages **only in
    the main thread** (e.g. via ``root.after``).

    Args:
        target: The callable to run in the background.
        args: Positional arguments for *target*.
        kwargs: Keyword arguments for *target*.
        result_queue: Queue to receive status/progress/result messages.
    """

    def __init__(
        self,
        target: Callable[..., Any],
        args: tuple = (),
        kwargs: dict | None = None,
        result_queue: queue.Queue | None = None,
    ) -> None:
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self.queue: queue.Queue = result_queue if result_queue is not None else queue.Queue()
        self._thread: threading.Thread | None = None
        self._cancelled = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the background thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        """Request cancellation.  The worker function must honour ``is_cancelled``."""
        self._cancelled.set()

    def is_cancelled(self) -> bool:
        """Return ``True`` when cancellation has been requested."""
        return self._cancelled.is_set()

    def is_alive(self) -> bool:
        """Return ``True`` when the background thread is still running."""
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            result = self._target(*self._args, **self._kwargs)
            self.queue.put({"type": "result", "payload": result})
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("Worker error: %s\n%s", exc, tb)
            self.queue.put({"type": "error", "payload": (exc, tb)})
        finally:
            self.queue.put({"type": "done", "payload": None})


def send_status(q: queue.Queue, message: str) -> None:
    """Put a status message into *q*.

    Args:
        q: The result queue of a :class:`WorkerTask`.
        message: Human-readable status string.
    """
    q.put({"type": "status", "payload": message})


def send_progress(q: queue.Queue, value: float) -> None:
    """Put a progress value (0.0–1.0) into *q*.

    Args:
        q: The result queue of a :class:`WorkerTask`.
        value: Fractional progress between 0.0 and 1.0.
    """
    q.put({"type": "progress", "payload": float(value)})


def send_preview(q: queue.Queue, frame: Any) -> None:
    """Put a preview frame into *q*.

    Args:
        q: The result queue of a :class:`WorkerTask`.
        frame: The frame object (e.g. a NumPy array) to display.
    """
    q.put({"type": "preview", "payload": frame})


def poll_queue(q: queue.Queue, handler: Callable[[dict], None], max_items: int = 20) -> None:
    """Drain up to *max_items* items from *q* and call *handler* for each.

    This function is intended to be called from the Tkinter main thread using
    ``root.after(interval, ...)`` so that the UI stays responsive.

    Args:
        q: Queue to drain.
        handler: Callable receiving each message dict.
        max_items: Maximum number of items to process per call (prevents
            blocking the main thread for too long).
    """
    for _ in range(max_items):
        try:
            msg = q.get_nowait()
            handler(msg)
        except queue.Empty:
            break
