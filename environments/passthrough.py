from __future__ import annotations

import hashlib

from core.types import Task, StepResult, ToolCall
from core.base import Environment
from core.registry import registry


@registry.environment("passthrough")
class PassthroughEnvironment(Environment):
    """No-op environment for black-box agents that handle everything internally."""

    def reset(self, task: Task) -> str:
        return "Black-box agent mode. No environment tools available."

    def step(self, tool_call: ToolCall) -> StepResult:
        return StepResult(observation="No environment available.", done=False)

    def get_state_hash(self) -> str:
        return hashlib.md5(b"passthrough").hexdigest()

    def get_tool_schemas(self) -> list[dict]:
        return []
