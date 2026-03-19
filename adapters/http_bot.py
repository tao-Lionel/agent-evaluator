from __future__ import annotations

import uuid
import logging
import time
from typing import Any

import httpx

from core.types import Role, Message
from core.base import AgentAdapter
from core.registry import registry

logger = logging.getLogger(__name__)


@registry.adapter("http_bot")
class HttpBotAdapter(AgentAdapter):
    """Adapter for any chatbot that exposes an HTTP POST interface.

    Supports three modes via `history_mode`:
      - "last"    : send only the last user message as a string (default, simplest)
      - "history" : send full conversation history as a messages array
      - "session" : send last user message + conversation_id for server-side history

    Request/response field names are configurable.

    Example configs:

      # Simple bot (single message in, single reply out)
      agent:
        adapter: http_bot
        bot_url: "http://localhost:8000/chat"

      # Bot that accepts conversation history
      agent:
        adapter: http_bot
        bot_url: "http://localhost:8000/chat"
        history_mode: history
        message_field: messages
        reply_field: reply

      # Bot with server-side session
      agent:
        adapter: http_bot
        bot_url: "http://localhost:8000/chat"
        history_mode: session
        session_field: conversation_id
    """

    def __init__(
        self,
        bot_url: str,
        history_mode: str = "last",
        message_field: str = "message",
        reply_field: str = "reply",
        session_field: str = "conversation_id",
        headers: dict[str, str] | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        retry_delay: float = 2.0,
        extra_body: dict[str, Any] | None = None,
        **kwargs,
    ):
        self.bot_url = bot_url
        self.history_mode = history_mode
        self.message_field = message_field
        self.reply_field = reply_field
        self.session_field = session_field
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.extra_body = extra_body or {}
        self.conversation_id: str | None = None
        self.client = httpx.Client(timeout=timeout)

    def reset(self) -> None:
        self.conversation_id = str(uuid.uuid4())

    def act(self, messages: list[Message]) -> Message:
        payload = self._build_payload(messages)
        reply_text = self._send_with_retry(payload)
        logger.debug("HttpBot reply: %s", reply_text[:200] if reply_text else "")
        return Message(role=Role.AGENT, content=reply_text)

    def _build_payload(self, messages: list[Message]) -> dict:
        payload = dict(self.extra_body)

        if self.history_mode == "history":
            payload[self.message_field] = self._to_history(messages)
        else:
            last_user_msg = self._extract_last_user_message(messages)
            payload[self.message_field] = last_user_msg

        if self.history_mode == "session" and self.conversation_id:
            payload[self.session_field] = self.conversation_id

        return payload

    def _send_with_retry(self, payload: dict) -> str:
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.post(
                    self.bot_url,
                    json=payload,
                    headers=self.headers,
                )
                response.raise_for_status()
                data = response.json()
                return self._extract_reply(data)
            except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout) as e:
                last_error = e
                logger.warning(
                    "HttpBot request failed (attempt %d/%d): %s",
                    attempt, self.max_retries, e,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
        raise last_error  # type: ignore[misc]

    def _extract_reply(self, data: dict) -> str:
        """Extract reply from response, supporting nested paths like 'data.reply'."""
        parts = self.reply_field.split(".")
        current = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part, "")
            else:
                return ""
        return str(current) if current else ""

    @staticmethod
    def _extract_last_user_message(messages: list[Message]) -> str:
        for msg in reversed(messages):
            if msg.role == Role.USER and msg.content:
                return msg.content
        return ""

    @staticmethod
    def _to_history(messages: list[Message]) -> list[dict]:
        """Convert trajectory to a portable messages array."""
        history = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                history.append({"role": "system", "content": msg.content or ""})
            elif msg.role == Role.USER:
                history.append({"role": "user", "content": msg.content or ""})
            elif msg.role == Role.AGENT:
                history.append({"role": "assistant", "content": msg.content or ""})
            elif msg.role == Role.ENV and msg.tool_results:
                for tr in msg.tool_results:
                    history.append({"role": "tool", "content": tr.output or "", "name": tr.name})
        return history

    @property
    def capabilities(self) -> set[str]:
        return {"chat"}
