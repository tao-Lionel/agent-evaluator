"""Interactive CLI: auto-generate evaluation config and scenarios for any HTTP Agent.

Usage:
    python generate_eval.py                          # Interactive mode
    python generate_eval.py http://localhost:8000/api/generate  # With URL
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from core.eval_generator import (
    try_fetch_openapi,
    parse_openapi_endpoint,
    auto_generate,
)


def _input(prompt: str) -> str:
    """Read input, stripping whitespace."""
    sys.stdout.write(prompt)
    sys.stdout.flush()
    return input().strip()


def _read_json_block(prompt: str) -> dict:
    """Read a JSON block from stdin. Supports multi-line paste."""
    print(prompt)
    print("  (粘贴 JSON 后按 Enter，再输入空行结束)")
    print()

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "" and lines:
            break
        lines.append(line)

    text = "\n".join(lines).strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"\n  JSON 解析失败: {e}")
        print("  请重新粘贴。\n")
        return _read_json_block(prompt)


def main():
    # Fix Windows console encoding
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print()
    print("=" * 56)
    print("  Agent Evaluator — 自动生成评估配置")
    print("=" * 56)
    print()

    # ── Step 1: Get URL ──
    if len(sys.argv) > 1:
        url = sys.argv[1]
        print(f"  Agent URL: {url}")
    else:
        url = _input("? Agent 的 URL: ")
        if not url:
            print("  URL 不能为空。")
            sys.exit(1)

    print()

    # ── Step 2: Try OpenAPI auto-detection ──
    print("  正在尝试获取 OpenAPI schema...", end=" ")
    sys.stdout.flush()

    openapi = try_fetch_openapi(url)
    schema_info = None
    if openapi:
        schema_info = parse_openapi_endpoint(openapi, url)

    if schema_info:
        print("成功!")
        fields = schema_info.get("request_fields", {})
        print(f"  检测到 {len(fields)} 个请求字段: {', '.join(fields.keys())}")
        api_desc = schema_info.get("description", "")
        if api_desc:
            print(f"  接口描述: {api_desc}")
        print()
    else:
        print("未找到。")
        print()
        print("  需要你提供一个请求/响应示例。")
        print()

    # ── Step 3: Get examples (if no OpenAPI) ──
    request_example = None
    response_example = None

    if not schema_info:
        request_example = _read_json_block("? 请粘贴一个请求示例 (JSON):")
        if not request_example:
            print("  请求示例不能为空。")
            sys.exit(1)
        print(f"  收到请求示例: {len(request_example)} 个字段")
        print()

        response_example = _read_json_block("? 请粘贴对应的响应示例 (JSON):")
        if not response_example:
            print("  响应示例不能为空。")
            sys.exit(1)
        print(f"  收到响应示例: {len(response_example)} 个字段")
        print()

    # ── Step 4: Get agent description ──
    description = _input("? 这个 Agent 是做什么的？(一句话描述): ")
    print()

    # ── Step 5: Generate ──
    print("  正在生成评估配置和场景...")

    has_api_key = bool(os.getenv("ZHIPU_API_KEY"))
    if not has_api_key:
        print("  (未检测到 ZHIPU_API_KEY，将生成基础场景，不使用 LLM)")

    try:
        config_yaml, scenarios_json, scenarios = auto_generate(
            url=url,
            request_example=request_example,
            response_example=response_example,
            agent_description=description,
            use_llm=has_api_key,
        )
    except Exception as e:
        print(f"\n  生成失败: {e}")
        sys.exit(1)

    # ── Step 6: Derive file names ──
    from urllib.parse import urlparse
    path_part = urlparse(url).path.rstrip("/").split("/")[-1]
    prefix = path_part or "agent"

    config_path = PROJECT_ROOT / f"config_{prefix}.yaml"
    scenarios_path = PROJECT_ROOT / f"scenarios/{prefix}_tasks.json"

    # ── Step 7: Save files ──
    scenarios_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_yaml)

    with open(scenarios_path, "w", encoding="utf-8") as f:
        f.write(scenarios_json)

    print()
    print(f"  ✓ {config_path.name:<40s} 已生成")
    print(f"  ✓ {scenarios_path.name:<40s} 已生成 ({len(scenarios)} 个场景)")
    print()
    print("  运行评估:")
    print(f"    python run.py {config_path.name}")
    print()


if __name__ == "__main__":
    main()
