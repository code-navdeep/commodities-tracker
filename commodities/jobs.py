"""Single-slot background job runner.

Dash callbacks must return fast, so long work (a refresh) runs on a daemon
thread: start() kicks it off and rejects a second start while one is live;
a polling callback reads status() until the result appears. One generic
implementation — anything needing background work uses this.
"""

import logging
import threading

logger = logging.getLogger(__name__)


class JobRunner:
    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._result = None

    def start(self, fn) -> bool:
        """Run fn() in the background. False if a job is already running."""
        with self._lock:
            if self._running:
                return False
            self._running = True
            self._result = None

        def _run():
            try:
                result = fn()
            except Exception as e:
                logger.error(f"Background job failed: {e}", exc_info=True)
                result = e
            with self._lock:
                self._running = False
                self._result = result

        threading.Thread(target=_run, daemon=True).start()
        return True

    def status(self) -> tuple[bool, object]:
        """(running, result). The result is returned once and then cleared."""
        with self._lock:
            if self._running:
                return True, None
            result, self._result = self._result, None
            return False, result
