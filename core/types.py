from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(Enum):
    SYSTEM = "system"
    AGENT = "agent"
    USER = "user"
    ENV = "env"


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str | None = None


@dataclass
class ToolResult:
    tool_call_id: str | None
    name: str
    output: str
    error: bool = False


@dataclass
class Message:
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_results: list[ToolResult] | None = None

    def __post_init__(self):
        if self.role in (Role.AGENT, Role.USER):
            has_content = self.content is not None
            has_tools = self.tool_calls is not None and len(self.tool_calls) > 0
            if has_content and has_tools:
                pass  # some models return both, allow it


@dataclass
class Task:
    id: str
    description: str
    initial_message: str
    initial_state: dict[str, Any]
    max_steps: int = 20
    expected_actions: list[dict[str, Any]] = field(default_factory=list)
    expected_state: dict[str, Any] = field(default_factory=dict)
    required_info: list[str] = field(default_factory=list)
    difficulty: str = "medium"

    @classmethod
    def from_dict(cls, data: dict) -> Task:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class StepResult:
    observation: str
    reward: float = 0.0
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class TerminationReason(Enum):
    SUCCESS = "success"
    MAX_STEPS = "max_steps"
    AGENT_ERROR = "agent_error"
    ENV_ERROR = "env_error"


@dataclass
class EvalResult:
    task_id: str
    terminated: TerminationReason
    trajectory: list[Message]
    scores: dict[str, float]
    overall_score: float
    steps_taken: int
    diagnosis: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "terminated": self.terminated.value,
            "steps_taken": self.steps_taken,
            "scores": self.scores,
            "overall_score": self.overall_score,
            "trajectory": self._serialize_trajectory(),
        }

    def _serialize_trajectory(self) -> list[dict[str, Any]]:
        result = []
        for msg in self.trajectory:
            entry: dict[str, Any] = {"role": msg.role.value}
            if msg.content:
                entry["content"] = msg.content
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in msg.tool_calls
                ]
            if msg.tool_results:
                entry["tool_results"] = [
                    {"name": tr.name, "output": tr.output, "error": tr.error}
                    for tr in msg.tool_results
                ]
            result.append(entry)
        return result
