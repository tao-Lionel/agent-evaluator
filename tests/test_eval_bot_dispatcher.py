import json
import unittest
from unittest.mock import MagicMock, patch

from eval_bot.dispatcher import DISPATCHER_SYSTEM_PROMPT, Dispatcher


class TestDispatcher(unittest.TestCase):
    def _make_mock_response(self, tool_call=None, content=None):
        mock_msg = MagicMock()
        mock_msg.content = content
        if tool_call:
            mock_tc = MagicMock()
            mock_tc.function.name = tool_call["name"]
            mock_tc.function.arguments = json.dumps(tool_call["arguments"])
            mock_msg.tool_calls = [mock_tc]
        else:
            mock_msg.tool_calls = None
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        return mock_resp

    def test_system_prompt_is_readable(self):
        self.assertIn("快速评测", DISPATCHER_SYSTEM_PROMPT)
        self.assertIn("历史评测结果", DISPATCHER_SYSTEM_PROMPT)

    @patch("eval_bot.dispatcher.OpenAI")
    def test_dispatch_quick_eval(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            tool_call={"name": "quick_eval", "arguments": {"bot_url": "http://test/chat"}}
        )

        dispatcher = Dispatcher()
        intent, args = dispatcher.classify("帮我测一下 http://test/chat")
        self.assertEqual(intent, "quick_eval")
        self.assertEqual(args["bot_url"], "http://test/chat")

    @patch("eval_bot.dispatcher.OpenAI")
    def test_dispatch_chitchat(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            content="你好！有什么可以帮你的吗？"
        )

        dispatcher = Dispatcher()
        intent, args = dispatcher.classify("你好")
        self.assertEqual(intent, "chitchat")
        self.assertEqual(args["reply"], "你好！有什么可以帮你的吗？")

    @patch("eval_bot.dispatcher.OpenAI")
    def test_dispatch_chitchat_fallback_reply(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._make_mock_response(content=None)

        dispatcher = Dispatcher()
        intent, args = dispatcher.classify("你好")
        self.assertEqual(intent, "chitchat")
        self.assertEqual(args["reply"], "你好，请问有什么评测需求？")


if __name__ == "__main__":
    unittest.main()
