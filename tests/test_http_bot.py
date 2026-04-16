from __future__ import annotations
import sys
import threading
import time
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import Role, Message
from adapters.http_bot import HttpBotAdapter


class MockBotHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        reply = f"Echo: {body.get('message', '')}"
        resp = {"reply": reply}
        # Echo back conversation_id if present
        if "conversation_id" in body:
            resp["conversation_id"] = body["conversation_id"]
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode())

    def log_message(self, format, *args):
        pass  # suppress logs


class MockHistoryHandler(BaseHTTPRequestHandler):
    """Accepts messages array, replies with message count."""
    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        messages = body.get("messages", [])
        user_msgs = [m for m in messages if m.get("role") == "user"]
        reply = f"Got {len(messages)} messages, {len(user_msgs)} from user"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"reply": reply}).encode())

    def log_message(self, format, *args):
        pass


class MockNestedReplyHandler(BaseHTTPRequestHandler):
    """Returns nested response like {"data": {"reply": "..."}}."""
    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        resp = {"data": {"reply": f"Nested: {body.get('message', '')}"}}
        self.wfile.write(json.dumps(resp).encode())

    def log_message(self, format, *args):
        pass


def _start_server(handler_class, port):
    server = HTTPServer(("127.0.0.1", port), handler_class)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.3)
    return server


def test_http_bot_adapter():
    """Basic test: last mode (default), sends only last user message."""
    server = _start_server(MockBotHandler, 18932)
    try:
        adapter = HttpBotAdapter(
            bot_url="http://127.0.0.1:18932/chat",
            message_field="message",
            reply_field="reply",
        )
        adapter.reset()

        messages = [
            Message(role=Role.SYSTEM, content="You are a bot."),
            Message(role=Role.USER, content="Hello world"),
        ]
        result = adapter.act(messages)

        assert result.role == Role.AGENT
        assert "Echo: Hello world" in result.content
        assert result.tool_calls is None
        print("  test_http_bot_adapter (last mode): PASSED")
    finally:
        server.shutdown()


def test_http_bot_session_mode():
    """Session mode: sends conversation_id with each request."""
    server = _start_server(MockBotHandler, 18933)
    try:
        adapter = HttpBotAdapter(
            bot_url="http://127.0.0.1:18933/chat",
            history_mode="session",
            session_field="conversation_id",
        )
        adapter.reset()
        assert adapter.conversation_id is not None

        messages = [
            Message(role=Role.SYSTEM, content="System."),
            Message(role=Role.USER, content="Hi"),
        ]
        result = adapter.act(messages)
        assert "Echo: Hi" in result.content
        print("  test_http_bot_session_mode: PASSED")
    finally:
        server.shutdown()


def test_http_bot_history_mode():
    """History mode: sends full conversation as messages array."""
    server = _start_server(MockHistoryHandler, 18934)
    try:
        adapter = HttpBotAdapter(
            bot_url="http://127.0.0.1:18934/chat",
            history_mode="history",
            message_field="messages",
            reply_field="reply",
        )
        adapter.reset()

        messages = [
            Message(role=Role.SYSTEM, content="System prompt."),
            Message(role=Role.USER, content="First question"),
            Message(role=Role.AGENT, content="First answer"),
            Message(role=Role.USER, content="Second question"),
        ]
        result = adapter.act(messages)

        assert "Got 4 messages" in result.content
        assert "2 from user" in result.content
        print("  test_http_bot_history_mode: PASSED")
    finally:
        server.shutdown()


def test_http_bot_nested_reply():
    """Nested reply field: supports dot-separated paths like 'data.reply'."""
    server = _start_server(MockNestedReplyHandler, 18935)
    try:
        adapter = HttpBotAdapter(
            bot_url="http://127.0.0.1:18935/chat",
            reply_field="data.reply",
        )
        adapter.reset()

        messages = [Message(role=Role.USER, content="Test")]
        result = adapter.act(messages)

        assert "Nested: Test" in result.content
        print("  test_http_bot_nested_reply: PASSED")
    finally:
        server.shutdown()


def test_http_bot_reset_new_session():
    """Each reset() generates a new conversation_id."""
    adapter = HttpBotAdapter(bot_url="http://localhost:9999", history_mode="session")
    adapter.reset()
    id1 = adapter.conversation_id
    adapter.reset()
    id2 = adapter.conversation_id
    assert id1 != id2
    assert id1 is not None and id2 is not None
    print("  test_http_bot_reset_new_session: PASSED")


def test_http_bot_extra_body():
    """extra_body fields are included in payload."""
    adapter = HttpBotAdapter(
        bot_url="http://localhost:9999",
        extra_body={"api_key": "test123", "model": "gpt-4"},
    )
    adapter.reset()
    messages = [Message(role=Role.USER, content="Hi")]
    payload = adapter._build_payload(messages)
    assert payload["api_key"] == "test123"
    assert payload["model"] == "gpt-4"
    assert payload["message"] == "Hi"
    print("  test_http_bot_extra_body: PASSED")


class MockTemplateHandler(BaseHTTPRequestHandler):
    """Echoes back the received request body as JSON."""
    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        # Return the request body as response so tests can verify it
        resp = {"received": body, "status": "ok", "slides": [{"page": 1, "layout": "cover"}]}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode())

    def log_message(self, format, *args):
        pass


def test_request_template_basic():
    """request_template substitutes ${initial_message} into configured fields."""
    server = _start_server(MockTemplateHandler, 18936)
    try:
        adapter = HttpBotAdapter(
            bot_url="http://127.0.0.1:18936/api/generate",
            request_template={
                "topic": "${initial_message}",
                "page_count": 10,
                "output_format": "web",
            },
            reply_field=".",
        )
        adapter.reset()

        messages = [
            Message(role=Role.SYSTEM, content="System prompt"),
            Message(role=Role.USER, content="AI trends in 2025"),
        ]
        result = adapter.act(messages)

        # reply_field="." should return entire response as JSON
        data = json.loads(result.content)
        assert data["received"]["topic"] == "AI trends in 2025"
        assert data["received"]["page_count"] == 10
        assert data["received"]["output_format"] == "web"
        assert data["slides"][0]["layout"] == "cover"
        print("  test_request_template_basic: PASSED")
    finally:
        server.shutdown()


def test_request_template_nested():
    """request_template supports nested dicts and lists."""
    adapter = HttpBotAdapter(
        bot_url="http://localhost:9999",
        request_template={
            "prompt": "${initial_message}",
            "config": {
                "size": "1024x1024",
                "style": "${description}",
            },
            "tags": ["art", "${initial_message}"],
        },
    )
    adapter.reset()

    messages = [
        Message(role=Role.SYSTEM, content="Generate a landscape"),
        Message(role=Role.USER, content="sunset over ocean"),
    ]
    payload = adapter._build_payload(messages)

    assert payload["prompt"] == "sunset over ocean"
    assert payload["config"]["size"] == "1024x1024"
    assert payload["config"]["style"] == "Generate a landscape"
    assert payload["tags"] == ["art", "sunset over ocean"]
    print("  test_request_template_nested: PASSED")


def test_reply_field_dot():
    """reply_field='.' serializes entire response as JSON."""
    server = _start_server(MockTemplateHandler, 18937)
    try:
        adapter = HttpBotAdapter(
            bot_url="http://127.0.0.1:18937/test",
            request_template={"msg": "${initial_message}"},
            reply_field=".",
        )
        adapter.reset()

        messages = [Message(role=Role.USER, content="test")]
        result = adapter.act(messages)

        data = json.loads(result.content)
        assert "received" in data
        assert "status" in data
        print("  test_reply_field_dot: PASSED")
    finally:
        server.shutdown()


def test_reply_field_extracts_nested_json():
    """When reply_field points to a dict/list, it's serialized as JSON."""
    server = _start_server(MockTemplateHandler, 18938)
    try:
        adapter = HttpBotAdapter(
            bot_url="http://127.0.0.1:18938/test",
            request_template={"msg": "${initial_message}"},
            reply_field="slides",
        )
        adapter.reset()

        messages = [Message(role=Role.USER, content="test")]
        result = adapter.act(messages)

        data = json.loads(result.content)
        assert isinstance(data, list)
        assert data[0]["layout"] == "cover"
        print("  test_reply_field_extracts_nested_json: PASSED")
    finally:
        server.shutdown()


def test_request_template_dict_message():
    """When user message content is a dict, use it directly as request body."""
    server = _start_server(MockTemplateHandler, 18939)
    try:
        adapter = HttpBotAdapter(
            bot_url="http://127.0.0.1:18939/api/generate",
            request_template={
                "topic": "${initial_message}",
                "page_count": 10,
            },
            reply_field=".",
        )
        adapter.reset()

        # Simulate what happens when initial_message is a dict in scenario JSON
        messages = [
            Message(role=Role.SYSTEM, content="System"),
            Message(role=Role.USER, content={"topic": "AI趋势", "page_count": 20, "theme": "dark"}),
        ]
        result = adapter.act(messages)

        data = json.loads(result.content)
        received = data["received"]
        # Dict message should be used directly, NOT template-substituted
        assert received["topic"] == "AI趋势"
        assert received["page_count"] == 20     # from dict, not template's 10
        assert received["theme"] == "dark"       # extra field from dict
        print("  test_request_template_dict_message: PASSED")
    finally:
        server.shutdown()


def test_backward_compatibility():
    """Without request_template, behaves exactly like before."""
    adapter = HttpBotAdapter(
        bot_url="http://localhost:9999",
        message_field="message",
        extra_body={"model": "gpt-4"},
    )
    adapter.reset()
    messages = [Message(role=Role.USER, content="Hello")]
    payload = adapter._build_payload(messages)

    assert payload["message"] == "Hello"
    assert payload["model"] == "gpt-4"
    assert "request_template" not in payload
    print("  test_backward_compatibility: PASSED")


def test_http_bot_retry_count_matches_max_retries():
    """max_retries means extra retries after the initial attempt."""
    adapter = HttpBotAdapter(
        bot_url="http://localhost:9999",
        max_retries=2,
        retry_delay=0.01,
    )
    request = httpx.Request("POST", adapter.bot_url)
    call_count = 0

    class FakeClient:
        def post(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ConnectError("connection refused", request=request)
            return httpx.Response(200, request=request, json={"reply": "ok"})

    adapter.client = FakeClient()
    reply = adapter._send_with_retry({"message": "hi"})

    assert reply == "ok"
    assert call_count == 3
    print("  test_http_bot_retry_count_matches_max_retries: PASSED")


if __name__ == "__main__":
    test_http_bot_adapter()
    test_http_bot_session_mode()
    test_http_bot_history_mode()
    test_http_bot_nested_reply()
    test_http_bot_reset_new_session()
    test_http_bot_extra_body()
    test_request_template_basic()
    test_request_template_nested()
    test_reply_field_dot()
    test_reply_field_extracts_nested_json()
    test_request_template_dict_message()
    test_backward_compatibility()
    print("\nAll http_bot tests passed!")
