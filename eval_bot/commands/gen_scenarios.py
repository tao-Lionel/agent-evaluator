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
