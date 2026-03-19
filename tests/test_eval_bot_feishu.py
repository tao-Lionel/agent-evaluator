import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


class TestFeishuWebhook(unittest.TestCase):
    def setUp(self):
        # Patch dispatcher and runner before import
        self.patcher_dispatch = patch("eval_bot.feishu.dispatcher")
        self.patcher_runner = patch("eval_bot.feishu.runner")
        self.mock_dispatcher = self.patcher_dispatch.start()
        self.mock_runner = self.patcher_runner.start()

        from eval_bot.feishu import app
        self.client = TestClient(app)

    def tearDown(self):
        self.patcher_dispatch.stop()
        self.patcher_runner.stop()

    def test_url_verification(self):
        resp = self.client.post("/feishu/event", json={
            "challenge": "test-challenge-token",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["challenge"], "test-challenge-token")

    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["service"], "eval_bot")

    def test_non_text_message_ignored(self):
        resp = self.client.post("/feishu/event", json={
            "header": {
                "event_id": "evt-001",
                "event_type": "im.message.receive_v1",
                "token": "",
            },
            "event": {
                "message": {
                    "message_id": "msg-001",
                    "message_type": "image",
                    "content": "{}",
                },
            },
        })
        self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
