from __future__ import annotations

import logging

from core.types import Role, Message, Task
from core.base import UserSimulator
from core.registry import registry

logger = logging.getLogger(__name__)


@registry.user("scripted")
class ScriptedUserSimulator(UserSimulator):
    """Condition-branch user simulator driven by task JSON scripts.

    Each branch in the script has:
      - "if_contains": list of keywords to match in agent's last reply
      - "default": true for fallback branch
      - "reply": string to send, or null to signal task done
    """

    def reset(self, task: Task) -> None:
        pass

    def respond(self, task: Task, trajectory: list[Message]) -> Message | None:
        if not task.user_scenario:
            return None

        script = task.user_scenario.get("script", [])
        if not script:
            return None

        agent_reply = self._get_last_agent_text(trajectory)

        # Keyword match branches (first match wins)
        for branch in script:
            if "if_contains" in branch:
                keywords = branch["if_contains"]
                if any(kw.lower() in agent_reply.lower() for kw in keywords):
                    return self._make_reply(branch["reply"])

        # Default branch
        for branch in script:
            if branch.get("default"):
                return self._make_reply(branch["reply"])

        # No match -> end conversation
        return None

    @staticmethod
    def _get_last_agent_text(trajectory: list[Message]) -> str:
        for msg in reversed(trajectory):
            if msg.role == Role.AGENT and msg.content:
                return msg.content
        return ""

    @staticmethod
    def _make_reply(reply: str | None) -> Message | None:
        if reply is None:
            return None
        return Message(role=Role.USER, content=reply)
