from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path

import pytest
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import Role, Message
from adapters.ws_bot import WsBotAdapter

# Mock WebSocket Server

MOCK_WS_PORT = 18950


async def _mock_ws_handler(ws):
    """Simulates an Agent WebSocket server."""
    async for raw in ws:
        msg = json.loads(raw)

        # Handle init
        if msg.get("type") == "init":
            await ws.send(json.dumps({
                "type": "session_ready",
                "session_id": msg.get("session_id", ""),
                "context_messages": 0,
            }))
            continue

        # Handle chat message
        text = msg.get("message", "")

        if text == "__tool_test__":
            # Simulate tool call flow
            await ws.send(json.dumps({
                "type": "tool_start",
                "name": "search_db",
                "arguments": {"query": "test"},
            }))
            await ws.send(json.dumps({
                "type": "tool_result",
                "name": "search_db",
                "status": "success",
            }))
            await ws.send(json.dumps({
                "type": "content",
                "text": "Found 3 results.",
            }))
            await ws.send(json.dumps({"type": "done"}))
        elif text == "__error_test__":
            await ws.send(json.dumps({
                "type": "error",
                "message": "Something went wrong",
            }))
        elif text == "__empty_test__":
            await ws.send(json.dumps({"type": "done"}))
        else:
            # Normal streaming reply
            for chunk in ["Hello", ", ", "world", "!"]:
                await ws.send(json.dumps({
                    "type": "content",
                    "text": chunk,
                }))
            await ws.send(json.dumps({"type": "done"}))


def _run_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _start_ws_server(port):
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=_run_event_loop, args=(loop,), daemon=True)
    thread.start()

    async def start_server():
        return await websockets.serve(_mock_ws_handler, "127.0.0.1", port)

    server = asyncio.run_coroutine_threadsafe(start_server(), loop).result(timeout=5)
    return loop, server, thread


@pytest.fixture(scope="module", autouse=True)
def ws_server():
    loop, server, thread = _start_ws_server(MOCK_WS_PORT)
    try:
        yield
    finally:
        server.close()
        asyncio.run_coroutine_threadsafe(server.wait_closed(), loop).result(timeout=5)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


# Tests

def test_ws_bot_basic():
    """Basic streaming reply."""
    adapter = WsBotAdapter(ws_url=f"ws://127.0.0.1:{MOCK_WS_PORT}")
    adapter.reset()

    messages = [
        Message(role=Role.SYSTEM, content="System prompt"),
        Message(role=Role.USER, content="Hi"),
    ]
    result = adapter.act(messages)

    assert result.role == Role.AGENT
    assert result.content == "Hello, world!"
    print("  test_ws_bot_basic: PASSED")


def test_ws_bot_tool_calls():
    """Tool call events are captured in reply."""
    adapter = WsBotAdapter(ws_url=f"ws://127.0.0.1:{MOCK_WS_PORT}")
    adapter.reset()

    messages = [Message(role=Role.USER, content="__tool_test__")]
    result = adapter.act(messages)

    assert "search_db" in result.content
    assert "Found 3 results." in result.content
    print("  test_ws_bot_tool_calls: PASSED")


def test_ws_bot_error():
    """Error message is captured."""
    adapter = WsBotAdapter(ws_url=f"ws://127.0.0.1:{MOCK_WS_PORT}")
    adapter.reset()

    messages = [Message(role=Role.USER, content="__error_test__")]
    result = adapter.act(messages)

    assert "[ERROR]" in result.content
    assert "Something went wrong" in result.content
    print("  test_ws_bot_error: PASSED")


def test_ws_bot_empty_reply():
    """Done with no content returns empty string."""
    adapter = WsBotAdapter(ws_url=f"ws://127.0.0.1:{MOCK_WS_PORT}")
    adapter.reset()

    messages = [Message(role=Role.USER, content="__empty_test__")]
    result = adapter.act(messages)

    assert result.content == ""
    print("  test_ws_bot_empty_reply: PASSED")


def test_ws_bot_reset_new_session():
    """Each reset() generates a unique session_id."""
    adapter = WsBotAdapter(ws_url="ws://localhost:9999")
    adapter.reset()
    id1 = adapter.session_id
    adapter.reset()
    id2 = adapter.session_id
    assert id1 != id2
    assert id1 is not None and id2 is not None
    print("  test_ws_bot_reset_new_session: PASSED")


def test_ws_bot_progress_callback():
    """on_progress callback is called with events."""
    events = []

    def on_progress(event, data):
        events.append(event)

    adapter = WsBotAdapter(ws_url=f"ws://127.0.0.1:{MOCK_WS_PORT}")
    adapter.on_progress = on_progress
    adapter.reset()

    messages = [Message(role=Role.USER, content="__tool_test__")]
    adapter.act(messages)

    assert "ws_connected" in events
    assert "ws_tool_start" in events
    assert "ws_tool_result" in events
    assert "ws_content" in events
    assert "ws_done" in events
    print("  test_ws_bot_progress_callback: PASSED")


def test_ws_bot_timeout():
    """Timeout returns whatever was collected."""
    adapter = WsBotAdapter(
        ws_url=f"ws://127.0.0.1:{MOCK_WS_PORT}",
        timeout=0.001,  # extremely short
    )
    adapter.reset()

    messages = [Message(role=Role.USER, content="Hi")]
    result = adapter.act(messages)
    # May get partial or empty; should not raise
    assert result.role == Role.AGENT
    print("  test_ws_bot_timeout: PASSED")


def test_ws_bot_inside_running_loop():
    """Calling act() from an existing event loop should still succeed."""
    adapter = WsBotAdapter(ws_url=f"ws://127.0.0.1:{MOCK_WS_PORT}")
    adapter.reset()

    async def run_inside_loop():
        messages = [Message(role=Role.USER, content="Hi")]
        return adapter.act(messages)

    result = asyncio.run(run_inside_loop())
    assert result.role == Role.AGENT
    assert result.content == "Hello, world!"
    print("  test_ws_bot_inside_running_loop: PASSED")


if __name__ == "__main__":
    _loop, _server, _thread = _start_ws_server(MOCK_WS_PORT)

    try:
        test_ws_bot_basic()
        test_ws_bot_tool_calls()
        test_ws_bot_error()
        test_ws_bot_empty_reply()
        test_ws_bot_reset_new_session()
        test_ws_bot_progress_callback()
        test_ws_bot_timeout()
        test_ws_bot_inside_running_loop()

        print("\nAll ws_bot tests passed!")
    finally:
        _server.close()
        asyncio.run_coroutine_threadsafe(_server.wait_closed(), _loop).result(timeout=5)
        _loop.call_soon_threadsafe(_loop.stop)
        _thread.join(timeout=5)
        _loop.close()
