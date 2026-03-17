from __future__ import annotations

from typing import Any


class Registry:
    """Central registry for pluggable components."""

    def __init__(self):
        self._adapters: dict[str, type] = {}
        self._environments: dict[str, type] = {}
        self._evaluators: dict[str, type] = {}

    # ── decorators ──

    def adapter(self, name: str):
        def wrap(cls):
            self._adapters[name] = cls
            return cls
        return wrap

    def environment(self, name: str):
        def wrap(cls):
            self._environments[name] = cls
            return cls
        return wrap

    def evaluator(self, name: str):
        def wrap(cls):
            self._evaluators[name] = cls
            return cls
        return wrap

    # ── lookups ──

    def get_adapter(self, name: str) -> type:
        if name not in self._adapters:
            raise KeyError(f"Adapter '{name}' not registered. Available: {list(self._adapters)}")
        return self._adapters[name]

    def get_environment(self, name: str) -> type:
        if name not in self._environments:
            raise KeyError(f"Environment '{name}' not registered. Available: {list(self._environments)}")
        return self._environments[name]

    def get_evaluator(self, name: str) -> type:
        if name not in self._evaluators:
            raise KeyError(f"Evaluator '{name}' not registered. Available: {list(self._evaluators)}")
        return self._evaluators[name]

    def list_all(self) -> dict[str, list[str]]:
        return {
            "adapters": list(self._adapters),
            "environments": list(self._environments),
            "evaluators": list(self._evaluators),
        }


registry = Registry()
