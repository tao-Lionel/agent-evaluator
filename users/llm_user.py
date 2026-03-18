from __future__ import annotations

import logging

from openai import OpenAI

from core.types import Role, Message, Task
from core.base import UserSimulator
from core.registry import registry

logger = logging.getLogger(__name__)

USER_SYSTEM_TEMPLATE = """\
You are role-playing as a user talking to a customer service agent.

Your character: {persona}
Your goal: {goal}

Rules:
- Act naturally like a real user. Do not reveal you are a simulator.
- If the agent has fully satisfied your goal, reply with exactly [TASK_DONE] and nothing else.
- If the agent asks irrelevant questions or stalls, insist on your goal.
- Keep replies short, 1-2 sentences."""


@registry.user("llm")
class LLMUserSimulator(UserSimulator):
    """LLM-powered user simulator that role-plays a persona."""

    def __init__(
        self,
        model: str = "glm-4-flash",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        **kwargs,
    ):
        import os
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(
            api_key=api_key or os.getenv("ZHIPU_API_KEY", ""),
            base_url=base_url or "https://open.bigmodel.cn/api/paas/v4",
        )

    def reset(self, task: Task) -> None:
        pass

    def respond(self, task: Task, trajectory: list[Message]) -> Message | None:
        if not task.user_scenario:
            return None

        messages = self._build_messages(task, trajectory)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=256,
            )
            reply = response.choices[0].message.content or ""
        except Exception as e:
            logger.error("LLMUser API error: %s", e)
            reply = "[TASK_DONE]"

        logger.info("LLMUser reply: %s", reply[:100])

        if self._is_done(reply):
            return None

        return Message(role=Role.USER, content=reply)

    def _build_messages(self, task: Task, trajectory: list[Message]) -> list[dict]:
        scenario = task.user_scenario or {}
        system_prompt = USER_SYSTEM_TEMPLATE.format(
            persona=scenario.get("persona", "a regular customer"),
            goal=scenario.get("goal", task.description),
        )

        messages = [{"role": "system", "content": system_prompt}]

        for msg in trajectory:
            if msg.role == Role.USER:
                messages.append({"role": "user", "content": msg.content or ""})
            elif msg.role == Role.AGENT and msg.content:
                # From the user-simulator's perspective, agent replies are
                # the "assistant" it is talking to, so map to assistant role
                messages.append({"role": "assistant", "content": msg.content})
            # Skip ENV/SYSTEM messages — user doesn't see tool calls

        return messages

    @staticmethod
    def _is_done(text: str) -> bool:
        return "[TASK_DONE]" in text
