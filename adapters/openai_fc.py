from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from core.types import Role, Message, ToolCall, ToolResult
from core.base import AgentAdapter
from core.registry import registry

logger = logging.getLogger(__name__)


@registry.adapter("openai_fc")
class OpenAIFCAdapter(AgentAdapter):
    """Adapter for any OpenAI-compatible API with function calling."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.tools = tools or []
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt_override = system_prompt
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def reset(self) -> None:
        pass

    def act(self, messages: list[Message]) -> Message:
        oai_messages = self._to_openai_messages(messages)

        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=oai_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if self.tools:
            kwargs["tools"] = self.tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0].message

        # Parse tool calls
        tool_calls: list[ToolCall] | None = None
        if choice.tool_calls:
            tool_calls = []
            for tc in choice.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(
                    name=tc.function.name,
                    arguments=args,
                    id=tc.id,
                ))

        return Message(
            role=Role.AGENT,
            content=choice.content,
            tool_calls=tool_calls,
        )

    @property
    def capabilities(self) -> set[str]:
        return {"chat", "tool_call"}

    def _to_openai_messages(self, messages: list[Message]) -> list[dict]:
        """Convert internal Message list to OpenAI API format."""
        oai: list[dict] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                oai.append({"role": "system", "content": msg.content or ""})

            elif msg.role == Role.USER:
                oai.append({"role": "user", "content": msg.content or ""})

            elif msg.role == Role.AGENT:
                entry: dict[str, Any] = {"role": "assistant"}
                if msg.content:
                    entry["content"] = msg.content
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.id or f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                            },
                        }
                        for i, tc in enumerate(msg.tool_calls)
                    ]
                    if "content" not in entry:
                        entry["content"] = None
                oai.append(entry)

            elif msg.role == Role.ENV:
                if msg.tool_results:
                    for tr in msg.tool_results:
                        oai.append({
                            "role": "tool",
                            "tool_call_id": tr.tool_call_id or "call_0",
                            "content": tr.output,
                        })

        return oai
