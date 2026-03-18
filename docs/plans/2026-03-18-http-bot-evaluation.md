# HTTP Bot 黑盒评估 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 支持通过 HTTP 接口评估黑盒 Agent（测试机器人 + http_bot Adapter + LLM-as-Judge 评估器）

**Architecture:** 黑盒 Agent 没有可观测的工具调用和数据库状态，所以评估策略从"状态比对"变为"回复质量判定"。新增一个 passthrough 环境（不做任何操作），一个 http_bot 适配器（POST 到机器人接口），一个 LLM-as-Judge 评估器（用 LLM 判断回复是否满足要求）。同时创建一个简单的测试用 HTTP 机器人（FastAPI + 智谱 API）来验证整条链路。

**Tech Stack:** FastAPI, httpx, OpenAI SDK (for LLM-as-Judge)

---

### Task 1: 创建测试用 HTTP 机器人

**Files:**
- Create: `test_bot/server.py`
- Create: `test_bot/requirements.txt`

**Step 1: 写测试机器人代码**

```python
# test_bot/server.py
"""A simple test chatbot with HTTP interface, backed by Zhipu API."""

from __future__ import annotations

import os
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4",
)

SYSTEM_PROMPT = (
    "You are a helpful customer service agent for an e-commerce platform. "
    "Answer user questions about orders, shipping, refunds, etc. "
    "Be concise and helpful."
)


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    conversation_id: str | None = None


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req.message},
        ],
        temperature=0.7,
        max_tokens=1024,
    )
    return ChatResponse(
        reply=response.choices[0].message.content,
        conversation_id=req.conversation_id,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
```

```
# test_bot/requirements.txt
fastapi>=0.100.0
uvicorn>=0.20.0
openai>=1.0.0
python-dotenv>=1.0.0
```

**Step 2: 手动验证测试机器人能启动和回复**

Run: `cd test_bot && pip install -r requirements.txt && uvicorn server:app --port 8100`

另一个终端测试:
```bash
curl -X POST http://localhost:8100/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is your return policy?"}'
```

Expected: 返回 JSON `{"reply": "...", "conversation_id": null}`

**Step 3: Commit**

```bash
git add test_bot/
git commit -m "feat: add test HTTP chatbot backed by Zhipu API"
```

---

### Task 2: 创建 passthrough 环境

黑盒 Agent 不需要环境执行工具调用，但 Orchestrator 要求有 Environment。创建一个什么都不做的 passthrough 环境。

**Files:**
- Create: `environments/passthrough.py`
- Modify: `environments/__init__.py`
- Create: `tests/test_passthrough.py`

**Step 1: 写测试**

```python
# tests/test_passthrough.py
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import Task, ToolCall
from environments.passthrough import PassthroughEnvironment


def test_passthrough_basics():
    env = PassthroughEnvironment()
    task = Task(
        id="test", description="test", initial_message="hello",
        initial_state={},
    )
    obs = env.reset(task)
    assert isinstance(obs, str)

    # Tool schemas should be empty
    assert env.get_tool_schemas() == []

    # State hash should be stable
    h1 = env.get_state_hash()
    h2 = env.get_state_hash()
    assert h1 == h2

    print("  Passthrough environment: PASSED")


if __name__ == "__main__":
    test_passthrough_basics()
```

**Step 2: 运行测试确认失败**

Run: `python tests/test_passthrough.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'environments.passthrough'`

**Step 3: 写实现**

```python
# environments/passthrough.py
from __future__ import annotations

import hashlib

from core.types import Task, StepResult, ToolCall
from core.base import Environment
from core.registry import registry


@registry.environment("passthrough")
class PassthroughEnvironment(Environment):
    """No-op environment for black-box agents that handle everything internally."""

    def reset(self, task: Task) -> str:
        return "Black-box agent mode. No environment tools available."

    def step(self, tool_call: ToolCall) -> StepResult:
        return StepResult(observation="No environment available.", done=False)

    def get_state_hash(self) -> str:
        return hashlib.md5(b"passthrough").hexdigest()

    def get_tool_schemas(self) -> list[dict]:
        return []
```

更新 `environments/__init__.py`，添加一行:
```python
from environments.passthrough import PassthroughEnvironment  # noqa: F401
```

**Step 4: 运行测试确认通过**

Run: `python tests/test_passthrough.py`
Expected: PASS

**Step 5: Commit**

```bash
git add environments/passthrough.py environments/__init__.py tests/test_passthrough.py
git commit -m "feat: add passthrough environment for black-box agents"
```

---

### Task 3: 创建 http_bot Adapter

**Files:**
- Create: `adapters/http_bot.py`
- Modify: `adapters/__init__.py`
- Create: `tests/test_http_bot.py`

**Step 1: 写测试（用 mock HTTP server）**

```python
# tests/test_http_bot.py
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
```

**Step 2: 运行测试确认失败**

Run: `python tests/test_http_bot.py`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: 写实现**

```python
# adapters/http_bot.py
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
```

更新 `adapters/__init__.py`，添加一行:
```python
from adapters.http_bot import HttpBotAdapter  # noqa: F401
```

添加 `httpx` 到 `requirements.txt`:
```
httpx>=0.24.0
```

**Step 4: 运行测试确认通过**

Run: `pip install httpx && python tests/test_http_bot.py`
Expected: PASS

**Step 5: Commit**

```bash
git add adapters/http_bot.py adapters/__init__.py tests/test_http_bot.py requirements.txt
git commit -m "feat: add http_bot adapter for black-box agent evaluation"
```

---

### Task 4: 创建 LLM-as-Judge 评估器

黑盒 Agent 无法用 state_match / action_match，需要 LLM 来判断回复质量。

**Files:**
- Create: `evaluators/llm_judge.py`
- Modify: `evaluators/__init__.py`
- Create: `tests/test_llm_judge.py`

**Step 1: 写测试（用 mock 避免调 API）**

```python
# tests/test_llm_judge.py
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import Role, Message, Task
from environments.passthrough import PassthroughEnvironment
from evaluators.llm_judge import LLMJudgeEvaluator


def test_scoring_logic():
    """Test the score extraction logic without calling LLM."""
    evaluator = LLMJudgeEvaluator.__new__(LLMJudgeEvaluator)

    assert evaluator._parse_score("SCORE: 5") == 1.0
    assert evaluator._parse_score("SCORE: 4") == 0.8
    assert evaluator._parse_score("SCORE: 3") == 0.6
    assert evaluator._parse_score("SCORE: 1") == 0.2
    assert evaluator._parse_score("some random text") == 0.0
    assert evaluator._parse_score("The answer is great. SCORE: 5") == 1.0

    print("  LLMJudge score parsing: PASSED")


def test_prompt_building():
    """Test that the judge prompt is correctly built."""
    evaluator = LLMJudgeEvaluator.__new__(LLMJudgeEvaluator)

    task = Task(
        id="t1", description="Answer shipping question",
        initial_message="When will my order arrive?",
        initial_state={},
        required_info=["3-5 business days"],
    )
    trajectory = [
        Message(role=Role.USER, content="When will my order arrive?"),
        Message(role=Role.AGENT, content="Your order will arrive in 3-5 business days."),
    ]

    prompt = evaluator._build_judge_prompt(task, trajectory)
    assert "When will my order arrive?" in prompt
    assert "3-5 business days" in prompt
    assert "SCORE:" in prompt

    print("  LLMJudge prompt building: PASSED")


if __name__ == "__main__":
    test_scoring_logic()
    test_prompt_building()
```

**Step 2: 运行测试确认失败**

Run: `python tests/test_llm_judge.py`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: 写实现**

```python
# evaluators/llm_judge.py
from __future__ import annotations

import logging
import re

from openai import OpenAI

from core.types import Role, Message, Task
from core.base import Environment, Evaluator
from core.registry import registry

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator assessing the quality of a chatbot's response.
Score the response on a scale of 1-5:

5 = Perfect: Fully addresses the request, accurate, helpful, well-formatted
4 = Good: Mostly addresses the request with minor issues
3 = Acceptable: Partially addresses the request, missing some information
2 = Poor: Barely addresses the request, significant issues
1 = Fail: Does not address the request, wrong, or harmful

Respond with your reasoning first, then end with exactly "SCORE: N" where N is 1-5."""

JUDGE_USER_TEMPLATE = """\
## Task Description
{description}

## User Message
{user_message}

## Required Information
The response should include: {required_info}

## Agent Response
{agent_response}

Please evaluate the agent's response."""


@registry.evaluator("llm_judge")
class LLMJudgeEvaluator(Evaluator):
    """Use an LLM to judge the quality of agent responses.

    Designed for black-box agents where state_match / action_match
    are not applicable.
    """

    def __init__(
        self,
        model: str = "glm-4-flash",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        import os
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.getenv("ZHIPU_API_KEY", ""),
            base_url=base_url or "https://open.bigmodel.cn/api/paas/v4",
        )

    def evaluate(self, task: Task, trajectory: list[Message], env: Environment) -> float:
        prompt = self._build_judge_prompt(task, trajectory)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            judge_text = response.choices[0].message.content or ""
            score = self._parse_score(judge_text)
            logger.info("LLMJudge for %s: score=%.2f\n%s", task.id, score, judge_text)
            return score
        except Exception as e:
            logger.error("LLMJudge failed: %s", e)
            return 0.0

    def _build_judge_prompt(self, task: Task, trajectory: list[Message]) -> str:
        agent_texts = [
            msg.content for msg in trajectory
            if msg.role == Role.AGENT and msg.content
        ]
        agent_response = "\n".join(agent_texts) if agent_texts else "(No response)"

        required = ", ".join(task.required_info) if task.required_info else "N/A"

        return JUDGE_USER_TEMPLATE.format(
            description=task.description,
            user_message=task.initial_message,
            required_info=required,
            agent_response=agent_response,
        )

    def _parse_score(self, text: str) -> float:
        match = re.search(r"SCORE:\s*(\d)", text)
        if match:
            raw = int(match.group(1))
            return max(0.0, min(1.0, raw / 5.0))
        return 0.0
```

更新 `evaluators/__init__.py`，添加一行:
```python
from evaluators.llm_judge import LLMJudgeEvaluator  # noqa: F401
```

**Step 4: 运行测试确认通过**

Run: `python tests/test_llm_judge.py`
Expected: PASS

**Step 5: Commit**

```bash
git add evaluators/llm_judge.py evaluators/__init__.py tests/test_llm_judge.py
git commit -m "feat: add LLM-as-Judge evaluator for black-box agents"
```

---

### Task 5: 适配 Orchestrator 和 run.py 支持黑盒模式

当前 Orchestrator 循环假设 Agent 会调用工具。黑盒 Agent 只回复文本，需要让 Orchestrator 在收到纯文本回复时直接结束（一问一答模式）。同时 run.py 的 adapter 构建逻辑需要兼容不同参数。

**Files:**
- Modify: `core/orchestrator.py:63-69`
- Modify: `run.py:152-160`

**Step 1: 修改 Orchestrator — 黑盒 Agent 收到纯文本回复即结束**

在 `core/orchestrator.py` 的 `_is_stop_signal` 方法中，增加判断：如果环境是 passthrough 且 Agent 返回了纯文本，视为完成。

更简洁的方案：给 Task 增加一个 `single_turn` 字段。如果 `single_turn=True`，Agent 返回第一条纯文本回复就视为完成。

在 `core/types.py` 的 Task 中添加:
```python
single_turn: bool = False
```

在 `core/orchestrator.py` 的 Case 1 分支中修改:
```python
# ── Case 1: text-only reply (no tool calls) ──
if not agent_msg.tool_calls:
    if self._is_stop_signal(agent_msg) or task.single_turn:
        termination = TerminationReason.SUCCESS
        break
    continue
```

**Step 2: 修改 run.py — adapter 构建逻辑支持不同参数**

将 `run.py:152-160` 中硬编码的 adapter 参数改为通用传参:

```python
# Build agent adapter
agent_cfg = config["agent"]
AdapterClass = registry.get_adapter(agent_cfg["adapter"])
adapter_params = {k: v for k, v in agent_cfg.items() if k != "adapter"}
# Pass tool schemas if adapter accepts them
if env.get_tool_schemas():
    adapter_params.setdefault("tools", env.get_tool_schemas())
adapter = AdapterClass(**adapter_params)
```

**Step 3: 运行所有已有测试确认不破坏**

Run: `python tests/test_core.py && python tests/test_passthrough.py && python tests/test_http_bot.py && python tests/test_llm_judge.py`
Expected: All PASS

**Step 4: Commit**

```bash
git add core/types.py core/orchestrator.py run.py
git commit -m "feat: support single-turn black-box agent evaluation"
```

---

### Task 6: 创建黑盒评估配置和场景

**Files:**
- Create: `config_http_bot.yaml`
- Create: `scenarios/http_bot_tasks.json`

**Step 1: 写黑盒评估配置**

```yaml
# config_http_bot.yaml
agent:
  adapter: http_bot
  bot_url: http://localhost:8100/chat
  message_field: message
  reply_field: reply

environment:
  type: passthrough

evaluators:
  - info_delivery
  - llm_judge

scenarios: scenarios/http_bot_tasks.json

run:
  num_trials: 1
  log_level: INFO
```

**Step 2: 写黑盒测试场景**

```json
[
  {
    "id": "greeting-001",
    "description": "User greets the bot, expecting a friendly response.",
    "difficulty": "easy",
    "single_turn": true,
    "initial_message": "Hi there! What can you help me with?",
    "initial_state": {},
    "max_steps": 1,
    "required_info": []
  },
  {
    "id": "shipping-002",
    "description": "User asks about shipping time.",
    "difficulty": "easy",
    "single_turn": true,
    "initial_message": "How long does standard shipping take?",
    "initial_state": {},
    "max_steps": 1,
    "required_info": []
  },
  {
    "id": "refund-policy-003",
    "description": "User asks about refund policy, agent should mention the return window and process.",
    "difficulty": "medium",
    "single_turn": true,
    "initial_message": "What is your refund policy? Can I return an item after 30 days?",
    "initial_state": {},
    "max_steps": 1,
    "required_info": ["refund", "return"]
  },
  {
    "id": "product-recommend-004",
    "description": "User asks for a product recommendation, agent should provide helpful suggestions.",
    "difficulty": "medium",
    "single_turn": true,
    "initial_message": "I'm looking for a gift for my friend who likes cooking. Any suggestions under $50?",
    "initial_state": {},
    "max_steps": 1,
    "required_info": []
  },
  {
    "id": "complaint-005",
    "description": "User complains about a bad experience, agent should be empathetic and offer a solution.",
    "difficulty": "hard",
    "single_turn": true,
    "initial_message": "I'm very disappointed. My order arrived late and the packaging was damaged. This is the second time this has happened!",
    "initial_state": {},
    "max_steps": 1,
    "required_info": ["sorry", "apolog"]
  }
]
```

**Step 3: Commit**

```bash
git add config_http_bot.yaml scenarios/http_bot_tasks.json
git commit -m "feat: add config and scenarios for HTTP bot evaluation"
```

---

### Task 7: 端到端验证

**Step 1: 启动测试机器人**

```bash
cd test_bot && uvicorn server:app --port 8100
```

**Step 2: 另一个终端运行黑盒评估**

```bash
python run.py config_http_bot.yaml
```

Expected: 5 个任务全部运行完成，输出每个任务的:
- 用户消息
- Agent 回复
- info_delivery 和 llm_judge 两个维度的评分
- 汇总报告

**Step 3: 确认所有测试通过**

```bash
python tests/test_core.py && python tests/test_passthrough.py && python tests/test_http_bot.py && python tests/test_llm_judge.py
```
Expected: All PASS

**Step 4: Commit**

```bash
git commit -m "test: verify end-to-end HTTP bot evaluation pipeline"
```
