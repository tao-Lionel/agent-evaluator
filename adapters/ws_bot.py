from __future__ import annotations

import asyncio
import json
import time as _time
import uuid
import logging
from typing import Any, Callable

import websockets

from core.types import Role, Message
from core.base import AgentAdapter
from core.registry import registry

logger = logging.getLogger(__name__)

# Callback type: (event: str, data: dict) -> None
ProgressCallback = Callable[[str, dict], None]


@registry.adapter("ws_bot")
class WsBotAdapter(AgentAdapter):
    """WebSocket adapter for agents that communicate via WebSocket.

    Config example:

      agent:
        adapter: ws_bot
        ws_url: "ws://192.168.11.18:8501/ws/chat"
        timeout: 120        # 单条消息最大等待秒数（默认 120）
        total_timeout: 300   # 整轮对话最大总时长秒数（默认 300，0=不限）
    """

    def __init__(
        self,
        ws_url: str,
        timeout: float = 120.0,
        total_timeout: float = 300.0,
        **kwargs,
    ):
        self.ws_url = ws_url
        self.timeout = timeout
        self.total_timeout = total_timeout
        self.session_id: str | None = None
        self.on_progress: ProgressCallback | None = None

    def reset(self) -> None:
        self.session_id = f"eval_{uuid.uuid4().hex[:12]}"

    def act(self, messages: list[Message]) -> Message:
        user_text = self._extract_last_user_message(messages)
        reply = asyncio.get_event_loop().run_until_complete(
            self._ws_chat(user_text)
        ) if self._has_running_loop() else asyncio.run(
            self._ws_chat(user_text)
        )
        logger.debug("WsBot reply: %s", reply[:200] if reply else "")
        return Message(role=Role.AGENT, content=reply)

    @staticmethod
    def _has_running_loop() -> bool:
        try:
            loop = asyncio.get_running_loop()
            return loop.is_running()
        except RuntimeError:
            return False

    async def _ws_chat(self, text: str) -> str:
        async with websockets.connect(
            self.ws_url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            # Init session
            await ws.send(json.dumps({
                "type": "init",
                "session_id": self.session_id,
            }))

            # Wait for session_ready
            try:
                ready = await asyncio.wait_for(ws.recv(), timeout=10)
                data = json.loads(ready)
                logger.debug("WsBot session init: %s", data.get("type"))
            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                logger.warning("WsBot session init timeout or closed")

            # Send user message
            await ws.send(json.dumps({"message": text}))

            # Collect streamed response
            full_text = ""
            tool_calls = []
            t0 = _time.monotonic()
            self._emit("ws_connected", {"session_id": self.session_id})
            try:
                while True:
                    # Check total timeout
                    if self.total_timeout > 0:
                        elapsed = _time.monotonic() - t0
                        remaining = self.total_timeout - elapsed
                        if remaining <= 0:
                            self._emit("ws_total_timeout", {"elapsed": self.total_timeout})
                            break
                        msg_timeout = min(self.timeout, remaining)
                    else:
                        msg_timeout = self.timeout

                    raw = await asyncio.wait_for(ws.recv(), timeout=msg_timeout)
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        logger.warning("WsBot received non-JSON message: %s", raw[:200])
                        continue
                    msg_type = msg.get("type", "")
                    elapsed = _time.monotonic() - t0

                    if msg_type == "content":
                        full_text += msg.get("text", "")
                        self._emit("ws_content", {"chars": len(full_text), "elapsed": elapsed})
                    elif msg_type == "tool_start":
                        tool_name = msg.get("name", "")
                        tool_calls.append({
                            "name": tool_name,
                            "arguments": msg.get("arguments", {}),
                        })
                        self._emit("ws_tool_start", {"name": tool_name, "elapsed": elapsed})
                    elif msg_type == "tool_result":
                        name = msg.get("name", "tool")
                        status = msg.get("status", "")
                        self._emit("ws_tool_result", {"name": name, "status": status, "elapsed": elapsed})
                    elif msg_type in ("done", "end"):
                        self._emit("ws_done", {"chars": len(full_text), "tools": len(tool_calls), "elapsed": elapsed})
                        break
                    elif msg_type == "error":
                        full_text += f"\n[ERROR] {msg.get('message', 'unknown error')}"
                        self._emit("ws_error", {"message": msg.get("message", ""), "elapsed": elapsed})
                        break
            except asyncio.TimeoutError:
                elapsed = _time.monotonic() - t0
                self._emit("ws_msg_timeout", {"timeout": self.timeout, "elapsed": elapsed})
                logger.warning("WsBot message timeout after %.1fs (total elapsed: %.1fs)", self.timeout, elapsed)
            except websockets.exceptions.ConnectionClosed as e:
                self._emit("ws_closed", {"reason": str(e), "elapsed": _time.monotonic() - t0})
                logger.debug("WsBot connection closed: %s", e)

            # If agent used tools, include that info in the reply
            if tool_calls and full_text:
                tool_summary = json.dumps(tool_calls, ensure_ascii=False)
                full_text = f"[Tools used: {tool_summary}]\n\n{full_text}"
            elif not full_text and tool_calls:
                full_text = json.dumps(tool_calls, ensure_ascii=False)

            return full_text

    def _emit(self, event: str, data: dict) -> None:
        if self.on_progress:
            try:
                self.on_progress(event, data)
            except Exception:
                logger.debug("Progress callback error for event %s", event, exc_info=True)

    @staticmethod
    def _extract_last_user_message(messages: list[Message]) -> str:
        for msg in reversed(messages):
            if msg.role == Role.USER and msg.content:
                content = msg.content
                return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        return ""

    @property
    def capabilities(self) -> set[str]:
        return {"chat"}
