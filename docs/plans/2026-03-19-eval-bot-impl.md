# eval_bot 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在现有 Agent Evaluator 框架上新增 `eval_bot/` 模块，实现飞书 Bot + LLM 智能编排，支持快速评测、结果查询、场景生成三个功能。

**Architecture:** 飞书 Webhook 接收用户消息 -> dispatcher 用智谱 function calling 识别意图 -> 路由到对应 command -> command 调用现有框架执行 -> 结果推回飞书。评测任务通过线程池异步执行。

**Tech Stack:** FastAPI, OpenAI SDK (智谱兼容), httpx, 现有 core/orchestrator + report.py

---

### Task 1: 项目骨架 + 配置

**Files:**
- Create: `eval_bot/__init__.py`
- Create: `eval_bot/commands/__init__.py`
- Create: `eval_bot/config.yaml`

**Step 1: 创建 eval_bot 包**

```python
# eval_bot/__init__.py
"""eval_bot — Feishu-based evaluation agent with LLM orchestration."""
```

```python
# eval_bot/commands/__init__.py
```

**Step 2: 创建 eval_bot 配置文件**

```yaml
# eval_bot/config.yaml
llm:
  model: glm-4
  api_key: ${ZHIPU_API_KEY}
  base_url: https://open.bigmodel.cn/api/paas/v4

feishu:
  app_id: ${FEISHU_APP_ID}
  app_secret: ${FEISHU_APP_SECRET}
  verify_token: ${FEISHU_VERIFY_TOKEN}
  encrypt_key: ${FEISHU_ENCRYPT_KEY}

defaults:
  evaluators:
    - info_delivery
    - llm_judge
  scenarios: scenarios/http_bot_tasks.json
  max_workers: 2
```

**Step 3: Commit**

```bash
git add eval_bot/
git commit -m "feat(eval_bot): scaffold directory structure and config"
```

---

### Task 2: runner.py — 异步任务执行器

**Files:**
- Create: `eval_bot/runner.py`
- Create: `tests/test_eval_bot_runner.py`

**Step 1: 写测试**

```python
# tests/test_eval_bot_runner.py
import time
import unittest
from eval_bot.runner import TaskRunner


class TestTaskRunner(unittest.TestCase):
    def test_submit_and_callback(self):
        """Submit a task, verify callback is called with result."""
        results = []

        def on_done(task_id, result):
            results.append((task_id, result))

        runner = TaskRunner(max_workers=1)

        def fake_work():
            time.sleep(0.1)
            return {"score": 0.85}

        runner.submit("test-001", fake_work, on_done)
        time.sleep(0.5)
        runner.shutdown()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "test-001")
        self.assertEqual(results[0][1], {"score": 0.85})

    def test_submit_error_callback(self):
        """If task raises, callback receives the exception."""
        results = []

        def on_done(task_id, result):
            results.append((task_id, result))

        runner = TaskRunner(max_workers=1)

        def bad_work():
            raise ValueError("boom")

        runner.submit("test-err", bad_work, on_done)
        time.sleep(0.5)
        runner.shutdown()

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0][1], Exception)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_eval_bot_runner.py -v
```

Expected: ImportError

**Step 3: 实现 runner.py**

```python
# eval_bot/runner.py
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TaskRunner:
    """Async task executor using thread pool with completion callbacks."""

    def __init__(self, max_workers: int = 2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit(
        self,
        task_id: str,
        work_fn: Callable[[], Any],
        on_done: Callable[[str, Any], None],
    ) -> None:
        future = self.executor.submit(work_fn)

        def callback(fut):
            try:
                result = fut.result()
                on_done(task_id, result)
            except Exception as e:
                logger.error("Task %s failed: %s", task_id, e)
                on_done(task_id, e)

        future.add_done_callback(callback)

    def shutdown(self):
        self.executor.shutdown(wait=True)
```

**Step 4: 运行测试，确认通过**

```bash
python -m pytest tests/test_eval_bot_runner.py -v
```

Expected: 2 passed

**Step 5: Commit**

```bash
git add eval_bot/runner.py tests/test_eval_bot_runner.py
git commit -m "feat(eval_bot): add TaskRunner async executor"
```

---

### Task 3: commands/quick_eval.py — 快速评测

**Files:**
- Create: `eval_bot/commands/quick_eval.py`
- Create: `tests/test_eval_bot_quick_eval.py`

**Step 1: 写测试**

```python
# tests/test_eval_bot_quick_eval.py
import unittest
from unittest.mock import patch, MagicMock
from eval_bot.commands.quick_eval import run_quick_eval, build_eval_config


class TestBuildEvalConfig(unittest.TestCase):
    def test_default_config(self):
        config = build_eval_config("http://localhost:8000/chat")
        self.assertEqual(config["agent"]["adapter"], "http_bot")
        self.assertEqual(config["agent"]["bot_url"], "http://localhost:8000/chat")
        self.assertEqual(config["environment"]["type"], "passthrough")
        self.assertIn("info_delivery", config["evaluators"])
        self.assertIn("llm_judge", config["evaluators"])

    def test_custom_evaluators(self):
        config = build_eval_config(
            "http://localhost:8000/chat",
            eval_modes=["info_delivery"],
        )
        self.assertEqual(config["evaluators"], ["info_delivery"])


class TestRunQuickEval(unittest.TestCase):
    @patch("eval_bot.commands.quick_eval.Orchestrator")
    @patch("eval_bot.commands.quick_eval.load_tasks")
    def test_returns_summary(self, mock_load_tasks, mock_orch_cls):
        from core.types import EvalResult, TerminationReason

        mock_task = MagicMock()
        mock_task.id = "t1"
        mock_task.difficulty = "easy"
        mock_task.description = "test"
        mock_task.initial_message = "hi"
        mock_load_tasks.return_value = [mock_task]

        mock_result = MagicMock(spec=EvalResult)
        mock_result.task_id = "t1"
        mock_result.overall_score = 0.8
        mock_result.scores = {"info_delivery": 0.8}
        mock_result.terminated = TerminationReason.SUCCESS
        mock_result.steps_taken = 1
        mock_result.summary.return_value = {
            "task_id": "t1",
            "overall_score": 0.8,
            "scores": {"info_delivery": 0.8},
        }

        mock_orch = MagicMock()
        mock_orch.run.return_value = mock_result
        mock_orch_cls.return_value = mock_orch

        result = run_quick_eval("http://localhost:8000/chat")
        self.assertIn("summary_text", result)
        self.assertIn("results_file", result)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_eval_bot_quick_eval.py -v
```

**Step 3: 实现 quick_eval.py**

```python
# eval_bot/commands/quick_eval.py
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.types import Task, EvalResult
from core.registry import registry
from core.orchestrator import Orchestrator
from report import generate_html, load_results

import adapters  # noqa: F401
import environments  # noqa: F401
import evaluators  # noqa: F401
import users  # noqa: F401

logger = logging.getLogger(__name__)

DEFAULT_EVALUATORS = ["info_delivery", "llm_judge"]
DEFAULT_SCENARIOS = "scenarios/http_bot_tasks.json"


def build_eval_config(
    bot_url: str,
    eval_modes: list[str] | None = None,
    scenarios_path: str | None = None,
) -> dict[str, Any]:
    return {
        "agent": {
            "adapter": "http_bot",
            "bot_url": bot_url,
            "message_field": "message",
            "reply_field": "reply",
            "timeout": 60,
        },
        "environment": {"type": "passthrough"},
        "evaluators": eval_modes or DEFAULT_EVALUATORS,
        "scenarios": scenarios_path or DEFAULT_SCENARIOS,
    }


def load_tasks(path: str) -> list[Task]:
    full_path = path if os.path.isabs(path) else str(PROJECT_ROOT / path)
    with open(full_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Task.from_dict(item) for item in data]


def run_quick_eval(
    bot_url: str,
    eval_modes: list[str] | None = None,
    scenarios_path: str | None = None,
) -> dict[str, Any]:
    config = build_eval_config(bot_url, eval_modes, scenarios_path)

    tasks = load_tasks(config["scenarios"])

    # Build components
    env_cls = registry.get_environment(config["environment"]["type"])
    env = env_cls()

    agent_cfg = config["agent"]
    adapter_cls = registry.get_adapter(agent_cfg["adapter"])
    adapter_params = {k: v for k, v in agent_cfg.items() if k != "adapter"}
    if env.get_tool_schemas():
        adapter_params.setdefault("tools", env.get_tool_schemas())
    adapter = adapter_cls(**adapter_params)

    evaluator_map = {}
    for name in config["evaluators"]:
        eval_cls = registry.get_evaluator(name)
        evaluator_map[name] = eval_cls()

    orchestrator = Orchestrator(adapter, env, evaluator_map)

    # Run
    all_results: list[EvalResult] = []
    for task in tasks:
        result = orchestrator.run(task)
        all_results.append(result)

    # Save results
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"results_{int(time.time())}.json"

    task_map = {t.id: t for t in tasks}
    results_data = []
    for r in all_results:
        entry = r.summary()
        t = task_map.get(r.task_id)
        if t:
            entry["task"] = {
                "description": t.description,
                "difficulty": t.difficulty,
                "initial_message": t.initial_message,
            }
        results_data.append(entry)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)

    # Generate HTML report
    report_data = load_results(str(output_file))
    report_file = output_file.with_suffix(".zh.html")
    html = generate_html(report_data, str(output_file), lang="zh")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(html)

    # Build summary text
    total = len(all_results)
    passed = sum(1 for r in all_results if r.overall_score >= 1.0 - 1e-6)
    avg = sum(r.overall_score for r in all_results) / total if total else 0

    summary_lines = [
        f"评测完成 | 目标: {bot_url}",
        f"任务数: {total} | 通过: {passed} | 失败: {total - passed}",
        f"平均得分: {avg:.1%}",
        "",
    ]
    for r in all_results:
        status = "PASS" if r.overall_score >= 1.0 - 1e-6 else "FAIL"
        t = task_map.get(r.task_id)
        desc = t.description if t else r.task_id
        summary_lines.append(f"  [{status}] {desc} ({r.overall_score:.1%})")

    summary_lines.append(f"\n详细报告: {report_file}")

    return {
        "summary_text": "\n".join(summary_lines),
        "results_file": str(output_file),
        "report_file": str(report_file),
        "avg_score": avg,
        "total": total,
        "passed": passed,
    }
```

**Step 4: 运行测试，确认通过**

```bash
python -m pytest tests/test_eval_bot_quick_eval.py -v
```

**Step 5: Commit**

```bash
git add eval_bot/commands/quick_eval.py tests/test_eval_bot_quick_eval.py
git commit -m "feat(eval_bot): add quick_eval command"
```

---

### Task 4: commands/query_results.py — 结果查询

**Files:**
- Create: `eval_bot/commands/query_results.py`
- Create: `tests/test_eval_bot_query.py`

**Step 1: 写测试**

```python
# tests/test_eval_bot_query.py
import json
import os
import tempfile
import unittest
from eval_bot.commands.query_results import scan_results


class TestScanResults(unittest.TestCase):
    def test_scan_finds_json_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake results
            data = [{"task_id": "t1", "overall_score": 0.8, "scores": {"x": 0.8}}]
            path = os.path.join(tmpdir, "results_123.json")
            with open(path, "w") as f:
                json.dump(data, f)

            results = scan_results(tmpdir)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["file"], path)

    def test_scan_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results = scan_results(tmpdir)
            self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
```

**Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_eval_bot_query.py -v
```

**Step 3: 实现 query_results.py**

```python
# eval_bot/commands/query_results.py
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def scan_results(results_dir: str | None = None) -> list[dict[str, Any]]:
    """Scan results directory, return list of result summaries sorted by time (newest first)."""
    rdir = Path(results_dir) if results_dir else RESULTS_DIR
    if not rdir.exists():
        return []

    entries = []
    for f in sorted(rdir.glob("results_*.json"), reverse=True):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            total = len(data)
            avg = sum(r.get("overall_score", 0) for r in data) / total if total else 0
            passed = sum(1 for r in data if r.get("overall_score", 0) >= 1.0 - 1e-6)
            entries.append({
                "file": str(f),
                "filename": f.name,
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "avg_score": round(avg, 3),
                "tasks": [
                    {
                        "task_id": r.get("task_id"),
                        "score": r.get("overall_score"),
                        "desc": r.get("task", {}).get("description", ""),
                    }
                    for r in data
                ],
            })
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Skipping %s: %s", f, e)

    return entries


def query_results(query: str) -> str:
    """Answer a user query about historical evaluation results using LLM."""
    entries = scan_results()

    if not entries:
        return "暂无评测记录。请先运行一次评测。"

    # Limit context: only send last 5 results
    context_entries = entries[:5]
    context = json.dumps(context_entries, ensure_ascii=False, indent=2)

    client = OpenAI(
        api_key=os.getenv("ZHIPU_API_KEY", ""),
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )

    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个评测结果分析助手。根据以下历史评测数据回答用户问题。"
                    "回答要简洁明了，突出关键数据。\n\n"
                    f"评测历史数据:\n{context}"
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=0.3,
        max_tokens=1024,
    )

    return response.choices[0].message.content or "无法生成回答。"
```

**Step 4: 运行测试，确认通过**

```bash
python -m pytest tests/test_eval_bot_query.py -v
```

**Step 5: Commit**

```bash
git add eval_bot/commands/query_results.py tests/test_eval_bot_query.py
git commit -m "feat(eval_bot): add query_results command"
```

---

### Task 5: commands/gen_scenarios.py — 场景生成

**Files:**
- Create: `eval_bot/commands/gen_scenarios.py`
- Create: `tests/test_eval_bot_gen.py`

**Step 1: 写测试**

```python
# tests/test_eval_bot_gen.py
import json
import unittest
from unittest.mock import patch, MagicMock
from eval_bot.commands.gen_scenarios import build_gen_prompt, parse_scenarios


class TestGenScenarios(unittest.TestCase):
    def test_build_prompt_contains_domain(self):
        prompt = build_gen_prompt("退款处理", count=3, difficulty="medium")
        self.assertIn("退款处理", prompt)
        self.assertIn("3", prompt)

    def test_parse_valid_json(self):
        raw = '''```json
[
  {
    "id": "gen-001",
    "description": "test",
    "initial_message": "hello",
    "initial_state": {},
    "max_steps": 1,
    "single_turn": true,
    "required_info": [],
    "difficulty": "easy"
  }
]
```'''
        result = parse_scenarios(raw)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "gen-001")

    def test_parse_no_json(self):
        result = parse_scenarios("no json here")
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
```

**Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_eval_bot_gen.py -v
```

**Step 3: 实现 gen_scenarios.py**

```python
# eval_bot/commands/gen_scenarios.py
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

FEW_SHOT_PATH = PROJECT_ROOT / "scenarios" / "http_bot_tasks.json"

GEN_SYSTEM_PROMPT = """\
你是一个测试场景生成专家。你需要根据用户指定的业务领域，生成用于评测聊天 Bot 的测试场景。

每个场景必须包含以下字段:
- id: 唯一标识（格式: gen-xxx）
- description: 场景描述
- initial_message: 用户发送的消息
- initial_state: 空对象 {}
- max_steps: 1
- single_turn: true
- required_info: 回复中应包含的关键词列表（用于评分）
- difficulty: easy/medium/hard

输出格式: JSON 数组，用 ```json ``` 包裹。"""


def build_gen_prompt(domain: str, count: int = 5, difficulty: str = "mixed") -> str:
    # Load few-shot examples
    few_shot = ""
    if FEW_SHOT_PATH.exists():
        with open(FEW_SHOT_PATH, "r", encoding="utf-8") as f:
            examples = json.load(f)
        few_shot = f"\n\n参考示例:\n```json\n{json.dumps(examples[:2], ensure_ascii=False, indent=2)}\n```"

    difficulty_hint = f"难度统一为 {difficulty}" if difficulty != "mixed" else "难度混合分配（easy/medium/hard）"

    return (
        f"请为「{domain}」领域生成 {count} 个测试场景。\n"
        f"{difficulty_hint}。\n"
        f"场景要覆盖该领域的不同方面，包括正常请求和边界情况。"
        f"{few_shot}"
    )


def parse_scenarios(raw: str) -> list[dict[str, Any]]:
    """Extract JSON array from LLM response."""
    match = re.search(r"```json\s*\n(.*?)\n\s*```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try parsing the whole string as JSON
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    return []


def gen_scenarios(
    domain: str,
    count: int = 5,
    difficulty: str = "mixed",
) -> dict[str, Any]:
    """Generate test scenarios for a given domain using LLM."""
    client = OpenAI(
        api_key=os.getenv("ZHIPU_API_KEY", ""),
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )

    prompt = build_gen_prompt(domain, count, difficulty)

    response = client.chat.completions.create(
        model="glm-4",
        messages=[
            {"role": "system", "content": GEN_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=4096,
    )

    raw = response.choices[0].message.content or ""
    scenarios = parse_scenarios(raw)

    if not scenarios:
        return {
            "success": False,
            "message": "场景生成失败，LLM 返回格式异常。",
            "file": None,
        }

    # Save
    output_dir = PROJECT_ROOT / "scenarios"
    output_dir.mkdir(exist_ok=True)
    filename = f"generated_{domain}_{int(time.time())}.json"
    output_path = output_dir / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, ensure_ascii=False, indent=2)

    return {
        "success": True,
        "message": f"已生成 {len(scenarios)} 个「{domain}」领域测试场景，保存在 {output_path}",
        "file": str(output_path),
        "count": len(scenarios),
    }
```

**Step 4: 运行测试，确认通过**

```bash
python -m pytest tests/test_eval_bot_gen.py -v
```

**Step 5: Commit**

```bash
git add eval_bot/commands/gen_scenarios.py tests/test_eval_bot_gen.py
git commit -m "feat(eval_bot): add gen_scenarios command"
```

---

### Task 6: dispatcher.py — LLM 意图识别

**Files:**
- Create: `eval_bot/dispatcher.py`
- Create: `tests/test_eval_bot_dispatcher.py`

**Step 1: 写测试**

```python
# tests/test_eval_bot_dispatcher.py
import json
import unittest
from unittest.mock import patch, MagicMock
from eval_bot.dispatcher import Dispatcher


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

    @patch("eval_bot.dispatcher.OpenAI")
    def test_dispatch_quick_eval(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            tool_call={"name": "quick_eval", "arguments": {"bot_url": "http://test/chat"}}
        )

        d = Dispatcher()
        intent, args = d.classify("帮我测一下 http://test/chat")
        self.assertEqual(intent, "quick_eval")
        self.assertEqual(args["bot_url"], "http://test/chat")

    @patch("eval_bot.dispatcher.OpenAI")
    def test_dispatch_chitchat(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            content="你好！有什么可以帮你的？"
        )

        d = Dispatcher()
        intent, args = d.classify("你好")
        self.assertEqual(intent, "chitchat")
        self.assertEqual(args["reply"], "你好！有什么可以帮你的？")


if __name__ == "__main__":
    unittest.main()
```

**Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_eval_bot_dispatcher.py -v
```

**Step 3: 实现 dispatcher.py**

```python
# eval_bot/dispatcher.py
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

请根据用户消息选择合适的工具。如果用户只是闲聊，直接回复即可。"""

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
                        "description": "Bot 的业务领域，如'客服'、'电商'",
                    },
                    "eval_modes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "评测维度列表，可选: info_delivery, llm_judge",
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
                        "description": "业务领域，如'退款处理'、'投诉处理'",
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

        # No tool call = chitchat
        reply = msg.content or "你好，请问有什么评测需求？"
        return "chitchat", {"reply": reply}
```

**Step 4: 运行测试，确认通过**

```bash
python -m pytest tests/test_eval_bot_dispatcher.py -v
```

**Step 5: Commit**

```bash
git add eval_bot/dispatcher.py tests/test_eval_bot_dispatcher.py
git commit -m "feat(eval_bot): add Dispatcher with LLM function calling"
```

---

### Task 7: feishu.py — 飞书 Webhook + 消息推送

**Files:**
- Create: `eval_bot/feishu.py`
- Create: `tests/test_eval_bot_feishu.py`

**Step 1: 写测试**

```python
# tests/test_eval_bot_feishu.py
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
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
```

**Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_eval_bot_feishu.py -v
```

**Step 3: 实现 feishu.py**

```python
# eval_bot/feishu.py
"""Feishu-based evaluation agent bot.

Start:
  uvicorn eval_bot.feishu:app --port 8102
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import httpx
from fastapi import FastAPI, Request
from dotenv import load_dotenv

from eval_bot.dispatcher import Dispatcher
from eval_bot.runner import TaskRunner
from eval_bot.commands.quick_eval import run_quick_eval
from eval_bot.commands.query_results import query_results
from eval_bot.commands.gen_scenarios import gen_scenarios

load_dotenv()

logger = logging.getLogger("eval_bot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI()

# ── Globals ──
dispatcher = Dispatcher()
runner = TaskRunner(max_workers=2)

# ── Feishu config ──
APP_ID = os.getenv("FEISHU_APP_ID", "")
APP_SECRET = os.getenv("FEISHU_APP_SECRET", "")
VERIFY_TOKEN = os.getenv("FEISHU_VERIFY_TOKEN", "")

# ── Token cache ──
_tenant_access_token: str = ""
_token_expires_at: float = 0


def get_tenant_access_token() -> str:
    global _tenant_access_token, _token_expires_at
    if _tenant_access_token and time.time() < _token_expires_at - 60:
        return _tenant_access_token

    resp = httpx.post(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
        json={"app_id": APP_ID, "app_secret": APP_SECRET},
    )
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"Feishu auth failed: {data.get('msg')}")

    _tenant_access_token = data["tenant_access_token"]
    _token_expires_at = time.time() + data.get("expire", 7200)
    return _tenant_access_token


def send_reply(message_id: str, text: str) -> None:
    token = get_tenant_access_token()
    resp = httpx.post(
        f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/reply",
        headers={"Authorization": f"Bearer {token}"},
        json={"content": json.dumps({"text": text}), "msg_type": "text"},
    )
    data = resp.json()
    if data.get("code") != 0:
        logger.error("Failed to send reply: %s", data)


def send_message_to_chat(chat_id: str, text: str) -> None:
    token = get_tenant_access_token()
    resp = httpx.post(
        "https://open.feishu.cn/open-apis/im/v1/messages",
        headers={"Authorization": f"Bearer {token}"},
        params={"receive_id_type": "chat_id"},
        json={
            "receive_id": chat_id,
            "content": json.dumps({"text": text}),
            "msg_type": "text",
        },
    )
    data = resp.json()
    if data.get("code") != 0:
        logger.error("Failed to send message: %s", data)


# ── Dedup ──
_seen_event_ids: dict[str, float] = {}


def _is_duplicate(event_id: str) -> bool:
    now = time.time()
    if len(_seen_event_ids) > 500:
        cutoff = now - 300
        for k in [k for k, v in _seen_event_ids.items() if v < cutoff]:
            del _seen_event_ids[k]
    if event_id in _seen_event_ids:
        return True
    _seen_event_ids[event_id] = now
    return False


def _extract_text_and_ids(body: dict) -> tuple[str | None, str | None, str | None]:
    """Extract text, message_id, chat_id from event."""
    try:
        event = body["event"]
        msg = event["message"]
        message_id = msg["message_id"]
        chat_id = msg.get("chat_id", "")
        if msg.get("message_type") != "text":
            return None, None, None
        content = json.loads(msg["content"])
        text = content.get("text", "")
        return text, message_id, chat_id
    except (KeyError, json.JSONDecodeError):
        return None, None, None


def _handle_intent(intent: str, args: dict[str, Any], message_id: str, chat_id: str):
    """Route intent to the correct command."""

    if intent == "chitchat":
        send_reply(message_id, args["reply"])
        return

    if intent == "quick_eval":
        send_reply(message_id, f"评测已开始，目标: {args['bot_url']}\n请稍候，完成后会通知你。")

        def work():
            return run_quick_eval(
                bot_url=args["bot_url"],
                eval_modes=args.get("eval_modes"),
                scenarios_path=args.get("scenarios_path"),
            )

        def on_done(task_id, result):
            if isinstance(result, Exception):
                send_reply(message_id, f"评测失败: {result}")
            else:
                send_reply(message_id, result["summary_text"])

        runner.submit(f"eval-{message_id}", work, on_done)
        return

    if intent == "query_results":
        answer = query_results(args["query"])
        send_reply(message_id, answer)
        return

    if intent == "gen_scenarios":
        result = gen_scenarios(
            domain=args["domain"],
            count=args.get("count", 5),
            difficulty=args.get("difficulty", "mixed"),
        )
        send_reply(message_id, result["message"])
        return

    send_reply(message_id, "抱歉，我不理解你的请求。")


@app.post("/feishu/event")
async def feishu_event(request: Request):
    body = await request.json()

    # URL verification
    if "challenge" in body:
        return {"challenge": body["challenge"]}

    header = body.get("header", {})
    event_id = header.get("event_id", "")

    if VERIFY_TOKEN and header.get("token") != VERIFY_TOKEN:
        return {"code": 0}

    if event_id and _is_duplicate(event_id):
        return {"code": 0}

    if header.get("event_type") != "im.message.receive_v1":
        return {"code": 0}

    text, message_id, chat_id = _extract_text_and_ids(body)
    if not text or not message_id:
        return {"code": 0}

    logger.info("Received: %s (msg=%s)", text[:100], message_id)

    try:
        intent, args = dispatcher.classify(text)
        logger.info("Intent: %s, Args: %s", intent, args)
        _handle_intent(intent, args, message_id, chat_id or "")
    except Exception as e:
        logger.error("Error handling message: %s", e)
        send_reply(message_id, f"处理出错: {e}")

    return {"code": 0}


@app.get("/health")
def health():
    return {"status": "ok", "service": "eval_bot"}
```

**Step 4: 运行测试，确认通过**

```bash
python -m pytest tests/test_eval_bot_feishu.py -v
```

**Step 5: Commit**

```bash
git add eval_bot/feishu.py tests/test_eval_bot_feishu.py
git commit -m "feat(eval_bot): add Feishu webhook handler with intent routing"
```

---

### Task 8: 集成验证

**Step 1: 运行所有 eval_bot 测试**

```bash
python -m pytest tests/test_eval_bot_*.py -v
```

Expected: All tests pass

**Step 2: 运行全部现有测试确认无回归**

```bash
python -m pytest tests/ -v
```

**Step 3: 最终 commit**

```bash
git add -A
git commit -m "feat(eval_bot): complete Feishu evaluation agent with smart orchestration"
```

---

计划完成，保存在 `docs/plans/2026-03-19-eval-bot-impl.md`。两种执行方式：

**1. Subagent 驱动（本会话）** — 每个 Task 分派子 agent 执行，中间做 code review，快速迭代

**2. 并行会话（新会话）** — 在新会话中用 executing-plans 批量执行

选哪种？