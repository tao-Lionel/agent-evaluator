from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TaskRunner:
    """Async task executor using thread pool with completion callbacks."""

    def __init__(self, max_workers: int = 2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit(
        self,
        task_id: str,
        work_fn: Callable[[], Any],
        on_done: Callable[[str, Any], None],
    ) -> None:
        future = self.executor.submit(work_fn)

        def callback(fut):
            try:
                result = fut.result()
                on_done(task_id, result)
            except Exception as e:
                logger.error("Task %s failed: %s", task_id, e)
                on_done(task_id, e)

        future.add_done_callback(callback)

    def shutdown(self):
        self.executor.shutdown(wait=True)
