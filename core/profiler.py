"""Lightweight performance profiler for evaluation phases."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any


class Profiler:
    """Accumulates timing for named phases."""

    def __init__(self):
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._start_time: float = time.time()

    @contextmanager
    def phase(self, name: str):
        """Time a named phase. Accumulates if called multiple times with same name."""
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            self._totals[name] = self._totals.get(name, 0.0) + elapsed
            self._counts[name] = self._counts.get(name, 0) + 1

    def summary(self) -> dict[str, Any]:
        """Return timing summary as a dict."""
        result: dict[str, Any] = {}
        for name, total in self._totals.items():
            result[name] = round(total, 3)
            count = self._counts[name]
            if count > 1:
                result[f"{name}_count"] = count
        result["total"] = round(time.time() - self._start_time, 3)
        return result

    def format_report(self) -> str:
        """Return a human-readable timing report."""
        summary = self.summary()
        total = summary.pop("total", 0)
        lines = []
        for key, val in sorted(summary.items()):
            if key.endswith("_count"):
                continue
            count = summary.get(f"{key}_count", 1)
            pct = (val / total * 100) if total > 0 else 0
            count_str = f" x{count}" if count > 1 else ""
            lines.append(f"  {key:20s}: {val:6.2f}s ({pct:4.1f}%){count_str}")
        lines.append(f"  {'total':20s}: {total:6.2f}s")
        return "\n".join(lines)
