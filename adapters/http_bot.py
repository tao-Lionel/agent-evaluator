from __future__ import annotations

import json
import re
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
    """Universal adapter for any HTTP API Agent.

    Two request modes:

      1. request_template (recommended) — config-driven, no code needed:

         agent:
           adapter: http_bot
           bot_url: "http://localhost:8000/api/generate"
           request_template:
             topic: "${initial_message}"
             page_count: 10
           reply_field: "."

      2. message_field (legacy) — simple chat bots:

         agent:
           adapter: http_bot
           bot_url: "http://localhost:8000/chat"
           reply_field: "reply"

    reply_field supports:
      "reply"       → response["reply"]
      "data.reply"  → response["data"]["reply"]
      "."           → entire response body as JSON string

    history_mode (legacy, chat bots only):
      "last"    — send only the last user message (default)
      "history" — send full conversation history array
      "session" — send last message + conversation_id
    """

    def __init__(
        self,
        bot_url: str,
        request_template: dict[str, Any] | None = None,
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
        self.request_template = request_template
        self.history_mode = history_mode
        self.message_field = message_field
        self.reply_field = reply_field
        self.session_field = session_field
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.extra_body = extra_body or {}
        self.conversation_id: str | None = None
        self.client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self.client.close()

    def reset(self) -> None:
        self.conversation_id = str(uuid.uuid4())

    def act(self, messages: list[Message]) -> Message:
        payload = self._build_payload(messages)
        reply_text = self._send_with_retry(payload)
        logger.debug("HttpBot reply: %s", reply_text[:200] if reply_text else "")
        return Message(role=Role.AGENT, content=reply_text)

    def _build_payload(self, messages: list[Message]) -> dict:
        # Template mode: render request_template with variable substitution
        if self.request_template:
            return self._render_template(messages)

        # Legacy mode: message_field + history_mode
        payload = dict(self.extra_body)

        if self.history_mode == "history":
            payload[self.message_field] = self._to_history(messages)
        else:
            last_user_msg = self._extract_last_user_message(messages)
            payload[self.message_field] = last_user_msg

        if self.history_mode == "session" and self.conversation_id:
            payload[self.session_field] = self.conversation_id

        return payload

    def _render_template(self, messages: list[Message]) -> dict:
        """Render request_template by substituting ${variable} placeholders.

        If the last user message is already a dict (structured request body from
        scenario JSON), use it directly as the payload — this allows per-scenario
        customization of all request fields.
        """
        last_msg = self._extract_last_user_content(messages)

        # If user message is a dict, use it directly as request body
        if isinstance(last_msg, dict):
            return last_msg

        # Otherwise, do template variable substitution
        variables = {
            "initial_message": last_msg if isinstance(last_msg, str) else str(last_msg),
            "description": self._extract_system_description(messages),
            "task_id": "",
        }

        def substitute(value: Any) -> Any:
            if isinstance(value, str):
                def replacer(match: re.Match) -> str:
                    var_name = match.group(1)
                    return variables.get(var_name, match.group(0))
                return re.sub(r"\$\{(\w+)\}", replacer, value)
            elif isinstance(value, dict):
                return {k: substitute(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute(item) for item in value]
            return value

        return substitute(dict(self.request_template))

    @staticmethod
    def _extract_system_description(messages: list[Message]) -> str:
        for msg in messages:
            if msg.role == Role.SYSTEM and msg.content:
                return msg.content
        return ""

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
                try:
                    data = response.json()
                except (json.JSONDecodeError, ValueError):
                    logger.warning("HttpBot response is not JSON: %s", response.text[:200])
                    return response.text
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
        """Extract reply from response.

        Supports:
          "."           → entire response as JSON string
          "reply"       → response["reply"]
          "data.reply"  → response["data"]["reply"]
        """
        if self.reply_field == ".":
            return json.dumps(data, ensure_ascii=False, indent=2)

        parts = self.reply_field.split(".")
        current: Any = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part, "")
            else:
                return ""
        if isinstance(current, (dict, list)):
            return json.dumps(current, ensure_ascii=False, indent=2)
        return str(current) if current else ""

    @staticmethod
    def _extract_last_user_content(messages: list[Message]) -> Any:
        """Extract the last user message content, preserving type (str or dict)."""
        for msg in reversed(messages):
            if msg.role == Role.USER and msg.content:
                return msg.content
        return ""

    @staticmethod
    def _extract_last_user_message(messages: list[Message]) -> str:
        for msg in reversed(messages):
            if msg.role == Role.USER and msg.content:
                content = msg.content
                return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
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
