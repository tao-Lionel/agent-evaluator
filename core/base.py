from __future__ import annotations

from abc import ABC, abstractmethod

from core.types import Message, Task, StepResult, ToolCall, Role


class AgentAdapter(ABC):
    """Adapter interface for agents being evaluated."""

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def act(self, messages: list[Message]) -> Message:
        """Given conversation history, return the agent's next message."""
        ...

    @property
    def capabilities(self) -> set[str]:
        return set()


class Environment(ABC):
    """Task environment interface."""

    @abstractmethod
    def reset(self, task: Task) -> str:
        """Reset environment to task's initial state. Return initial observation."""
        ...

    @abstractmethod
    def step(self, tool_call: ToolCall) -> StepResult:
        """Execute a tool call, return observation."""
        ...

    @abstractmethod
    def get_state_hash(self) -> str:
        """Hash of current environment state, for evaluation."""
        ...

    @abstractmethod
    def get_tool_schemas(self) -> list[dict]:
        """Return OpenAI-compatible tool/function JSON schemas."""
        ...


class Evaluator(ABC):
    """Scoring interface."""

    @abstractmethod
    def evaluate(self, task: Task, trajectory: list[Message], env: Environment) -> float:
        """Return a score between 0.0 and 1.0."""
        ...


class UserSimulator(ABC):
    """Simulated user for multi-turn conversations."""

    @abstractmethod
    def reset(self, task: Task) -> None:
        """Reset state for a new task."""
        ...

    @abstractmethod
    def respond(self, task: Task, trajectory: list[Message]) -> Message | None:
        """Generate user reply based on conversation so far.

        Returns a Message with role=USER, or None if the user is satisfied
        and the conversation should end.
        """
        ...
