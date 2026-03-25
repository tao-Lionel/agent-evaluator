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
    content: Any = None  # str, dict (structured request), or None
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
    initial_message: Any  # str or dict (structured request body)
    initial_state: dict[str, Any]
    max_steps: int = 20
    expected_actions: list[dict[str, Any]] = field(default_factory=list)
    expected_state: dict[str, Any] = field(default_factory=dict)
    required_info: list[str] = field(default_factory=list)
    difficulty: str = "medium"
    single_turn: bool = False
    user_scenario: dict[str, Any] | None = None
    nl_assertions: list[str] = field(default_factory=list)

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
    USER_STOP = "user_stop"


@dataclass
class EvalResult:
    task_id: str
    terminated: TerminationReason
    trajectory: list[Message]
    scores: dict[str, float]
    overall_score: float
    steps_taken: int
    score_details: dict[str, str] = field(default_factory=dict)
    diagnosis: str = ""
    elapsed_seconds: float = 0.0
    step_durations: list[float] = field(default_factory=list)
    step_rewards: list[float] = field(default_factory=list)
    progress_rate: float = 0.0

    def summary(self) -> dict[str, Any]:
        result = {
            "task_id": self.task_id,
            "terminated": self.terminated.value,
            "steps_taken": self.steps_taken,
            "scores": self.scores,
            "overall_score": self.overall_score,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "trajectory": self._serialize_trajectory(),
        }
        if self.score_details:
            result["score_details"] = self.score_details
        if self.step_durations:
            result["step_durations"] = [round(d, 3) for d in self.step_durations]
        if self.step_rewards:
            result["step_rewards"] = [round(r, 3) for r in self.step_rewards]
            result["progress_rate"] = round(self.progress_rate, 3)
        return result

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
