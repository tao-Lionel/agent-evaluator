"""CLI tool to auto-generate evaluation config and scenarios for any Agent.

Usage:
  # HTTP Agent (auto-detect OpenAPI)
  python generate.py http://localhost:8000/api/generate --desc "PPT生成Agent"

  # WebSocket Agent (auto-probe via chat)
  python generate.py ws://192.168.11.18:8501/ws/chat --desc "Amazon评论分析助手"

  # HTTP Agent with example request/response
  python generate.py http://localhost:8000/chat --desc "客服Bot" \
    --request '{"message": "你好"}' --response '{"reply": "你好！"}'
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

from core.eval_generator import (
    auto_generate,
    generate_scenarios_with_llm,
    _parse_scenarios_json,
)

logger = logging.getLogger(__name__)


# ── WebSocket Agent probing ─────────────────────────────────────────────────

WS_SCENARIO_GEN_PROMPT = """\
你是一个 AI Agent 评估专家。根据以下 Agent 信息，生成 6 个多样化的评估场景。

## Agent 信息
- WebSocket URL: {url}
- Agent 自我介绍: {self_intro}
- 用户描述: {description}

## 要求

生成 6 个场景，覆盖：
1. 基础问候 (easy) — 验证 Agent 能正确介绍自身功能
2. 核心功能 (easy) — 测试 Agent 最基本的能力
3. 标准用例 (medium) — 典型使用场景
4. 引导能力 (medium) — 缺少必要信息时 Agent 应主动询问
5. 边界情况 (hard) — 超出能力范围的请求，Agent 应合理拒绝或引导
6. 复杂需求 (hard) — 专业性强或多步骤的请求

每个场景的 nl_assertions 应该检查 Agent 回复的质量和正确性。

## 输出格式

输出纯 JSON 数组，不要 markdown 代码块。每个元素格式：
{{
  "id": "场景ID",
  "description": "场景描述",
  "difficulty": "easy|medium|hard",
  "single_turn": true,
  "initial_message": "用户输入",
  "initial_state": {{}},
  "max_steps": 1,
  "expected_actions": [],
  "expected_state": {{}},
  "required_info": [],
  "nl_assertions": ["断言1", "断言2", "断言3"]
}}"""


async def probe_ws_agent(ws_url: str, timeout: float = 60.0) -> str:
    """Connect to a WebSocket agent, send a greeting, return its self-introduction."""
    import websockets

    async with websockets.connect(ws_url, ping_interval=20, close_timeout=5) as ws:
        await ws.send(json.dumps({"type": "init", "session_id": "probe_001"}))

        # Wait for session_ready
        try:
            await asyncio.wait_for(ws.recv(), timeout=10)
        except (asyncio.TimeoutError, Exception):
            pass

        await ws.send(json.dumps({"message": "你好，请介绍一下你能做什么"}))

        full_text = ""
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                msg = json.loads(raw)
                if msg.get("type") == "content":
                    full_text += msg.get("text", "")
                elif msg.get("type") in ("done", "end"):
                    break
                elif msg.get("type") == "error":
                    break
        except (asyncio.TimeoutError, Exception):
            pass

        return full_text


def generate_ws_config(ws_url: str, name: str, scenarios_path: str) -> str:
    """Generate YAML config for a WebSocket agent."""
    config = {
        "name": name,
        "agent": {
            "adapter": "ws_bot",
            "ws_url": ws_url,
            "timeout": 120,
            "total_timeout": 300,
        },
        "environment": {"type": "passthrough"},
        "evaluators": [
            {"name": "llm_judge", "model": "glm-4-flash", "max_tokens": 1024},
            {"name": "nl_assertion", "model": "glm-4-flash", "max_tokens": 1024},
            "info_delivery",
        ],
        "scenarios": scenarios_path,
        "run": {"num_trials": 1, "log_level": "INFO"},
    }
    return yaml.dump(config, allow_unicode=True, default_flow_style=False, sort_keys=False)


def generate_ws_scenarios(
    ws_url: str,
    self_intro: str,
    description: str,
    model: str = "glm-4-flash",
) -> list[dict]:
    """Use LLM to generate evaluation scenarios for a WebSocket agent."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("ZHIPU_API_KEY", ""),
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )

    prompt = WS_SCENARIO_GEN_PROMPT.format(
        url=ws_url,
        self_intro=self_intro[:1000],
        description=description or "(未提供)",
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=4096,
    )
    text = response.choices[0].message.content or ""
    return _parse_scenarios_json(text)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Auto-generate evaluation config and scenarios")
    parser.add_argument("url", help="Agent URL (http:// or ws://)")
    parser.add_argument("--desc", default="", help="Agent description")
    parser.add_argument("--name", default="", help="Agent name for report")
    parser.add_argument("--request", default=None, help="Example request JSON (HTTP only)")
    parser.add_argument("--response", default=None, help="Example response JSON (HTTP only)")
    parser.add_argument("--model", default="glm-4-flash", help="LLM model for scenario generation")
    parser.add_argument("--output", default=None, help="Output prefix (default: derived from URL)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parsed = urlparse(args.url)
    is_ws = parsed.scheme in ("ws", "wss")

    # Derive output prefix
    if args.output:
        prefix = args.output
    else:
        path = parsed.path.rstrip("/").split("/")[-1]
        prefix = path or (args.name or "agent").replace(" ", "_")

    scenarios_path = f"scenarios/{prefix}_tasks.json"

    if is_ws:
        # ── WebSocket Agent ──
        print(f"Probing WebSocket Agent at {args.url} ...")
        self_intro = asyncio.run(probe_ws_agent(args.url))

        if self_intro:
            print(f"Agent self-intro ({len(self_intro)} chars):")
            print(f"  {self_intro[:200]}{'...' if len(self_intro) > 200 else ''}\n")
        else:
            print("Warning: Agent did not respond to probe.\n")

        name = args.name or args.desc or prefix
        print("Generating scenarios with LLM ...")
        scenarios = generate_ws_scenarios(args.url, self_intro, args.desc, model=args.model)
        config_yaml = generate_ws_config(args.url, name, scenarios_path)

    else:
        # ── HTTP Agent ──
        req_example = json.loads(args.request) if args.request else None
        resp_example = json.loads(args.response) if args.response else None

        config_yaml, scenarios_json_str, scenarios = auto_generate(
            url=args.url,
            request_example=req_example,
            response_example=resp_example,
            agent_description=args.desc,
            output_prefix=prefix,
            use_llm=True,
            model=args.model,
        )

    # ── Save outputs ──
    config_file = PROJECT_ROOT / f"config_{prefix}.yaml"
    scenarios_file = PROJECT_ROOT / scenarios_path

    scenarios_file.parent.mkdir(parents=True, exist_ok=True)

    scenarios_json = json.dumps(scenarios, ensure_ascii=False, indent=2)
    with open(scenarios_file, "w", encoding="utf-8") as f:
        f.write(scenarios_json)
    print(f"Scenarios saved to {scenarios_file} ({len(scenarios)} tasks)")

    with open(config_file, "w", encoding="utf-8") as f:
        f.write(config_yaml)
    print(f"Config saved to {config_file}")

    print(f"\nRun evaluation with:")
    print(f"  python run.py {config_file.name}")


if __name__ == "__main__":
    main()
