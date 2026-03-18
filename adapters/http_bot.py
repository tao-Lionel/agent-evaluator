from __future__ import annotations

import logging
from typing import Any

import httpx

from core.types import Role, Message
from core.base import AgentAdapter
from core.registry import registry

logger = logging.getLogger(__name__)


@registry.adapter("http_bot")
class HttpBotAdapter(AgentAdapter):
    """Adapter for any chatbot that exposes an HTTP POST interface.

    Expected bot API:
        POST bot_url
        Request:  {"message": "user input", ...headers}
        Response: {"reply": "bot response"}

    Field names are configurable via message_field / reply_field.
    """

    def __init__(
        self,
        bot_url: str,
        message_field: str = "message",
        reply_field: str = "reply",
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        **kwargs,
    ):
        self.bot_url = bot_url
        self.message_field = message_field
        self.reply_field = reply_field
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def reset(self) -> None:
        pass

    def act(self, messages: list[Message]) -> Message:
        # Extract the last user message to send to the bot
        user_message = ""
        for msg in reversed(messages):
            if msg.role == Role.USER and msg.content:
                user_message = msg.content
                break

        payload = {self.message_field: user_message}

        response = self.client.post(
            self.bot_url,
            json=payload,
            headers=self.headers,
        )
        response.raise_for_status()
        data = response.json()

        reply = data.get(self.reply_field, "")
        logger.debug("HttpBot reply: %s", reply[:200])

        return Message(role=Role.AGENT, content=reply)

    @property
    def capabilities(self) -> set[str]:
        return {"chat"}
