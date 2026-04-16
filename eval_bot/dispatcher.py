from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

DISPATCHER_SYSTEM_PROMPT = """\
你是 Agent Evaluator 的智能助手，帮助用户通过飞书进行 Bot 评测。
你可以：
1. 对 HTTP Bot 进行快速评测
2. 查询历史评测结果
3. 生成测试场景

请根据用户消息选择合适的工具。如果用户只是在闲聊，直接回复即可。"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "quick_eval",
            "description": "对指定 HTTP Bot 进行快速评测。当用户提供了 Bot 地址并希望进行测试时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "bot_url": {
                        "type": "string",
                        "description": "被测 Bot 的 HTTP 地址",
                    },
                    "domain": {
                        "type": "string",
                        "description": "Bot 的业务领域，如“客服”“电商”",
                    },
                    "eval_modes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "评测维度列表，可选 info_delivery、llm_judge",
                    },
                },
                "required": ["bot_url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_results",
            "description": "查询历史评测结果。当用户询问之前的评测情况、分数、对比时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户的查询意图",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gen_scenarios",
            "description": "生成测试场景。当用户希望为特定业务领域创建测试用例时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "业务领域，如“退款处理”“投诉处理”",
                    },
                    "count": {
                        "type": "integer",
                        "description": "生成数量，默认 5",
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": ["easy", "medium", "hard", "mixed"],
                        "description": "难度偏好，默认 mixed",
                    },
                },
                "required": ["domain"],
            },
        },
    },
]


class Dispatcher:
    """Use LLM function calling to classify user intent and extract parameters."""

    def __init__(
        self,
        model: str = "glm-4",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.getenv("ZHIPU_API_KEY", ""),
            base_url=base_url or "https://open.bigmodel.cn/api/paas/v4",
        )

    def classify(self, user_message: str) -> tuple[str, dict[str, Any]]:
        """Classify user intent. Returns (intent_name, arguments)."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": DISPATCHER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            tools=TOOLS,
            temperature=0.0,
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            tc = msg.tool_calls[0]
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            logger.info("Dispatched to %s with args: %s", name, args)
            return name, args

        reply = msg.content or "你好，请问有什么评测需求？"
        return "chitchat", {"reply": reply}
