from __future__ import annotations
import sys
import threading
import time
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import Role, Message
from adapters.http_bot import HttpBotAdapter


class MockBotHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        reply = f"Echo: {body['message']}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"reply": reply}).encode())

    def log_message(self, format, *args):
        pass  # suppress logs


def test_http_bot_adapter():
    # Start mock server
    server = HTTPServer(("127.0.0.1", 18932), MockBotHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.3)

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
        print("  HttpBotAdapter: PASSED")
    finally:
        server.shutdown()


if __name__ == "__main__":
    test_http_bot_adapter()
