from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.types import Task, EvalResult, Role
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

ANALYSIS_SYSTEM_PROMPT = """\
你是一个评测分析专家。请根据失败任务的对话轨迹和评分，分析失败原因并给出改进建议。

要求：
1. 找出失败任务的共性模式（理解力问题？信息遗漏？回复不准确？）
2. 按严重程度排序
3. 给出具体、可操作的改进建议
4. 回复简洁，不超过 300 字"""


def analyze_failures(
    results_data: list[dict],
    all_results: list[EvalResult],
    task_map: dict[str, Task],
) -> str:
    """Use LLM to analyze failed tasks and provide improvement suggestions."""
    failed = [
        (r, rd) for r, rd in zip(all_results, results_data)
        if r.overall_score < 1.0 - 1e-6
    ]
    if not failed:
        return ""

    # Build context from failed tasks' trajectories
    context_parts = []
    for result, rdata in failed:
        task = task_map.get(result.task_id)
        desc = task.description if task else result.task_id
        scores = rdata.get("scores", {})
        trajectory = getattr(result, "trajectory", None) or []

        # Extract key messages from trajectory
        messages = []
        for msg in trajectory:
            content = getattr(msg, "content", None)
            if not content:
                continue
            role = getattr(msg, "role", None)
            content_text = str(content)[:200]
            if role == Role.USER:
                messages.append(f"用户: {content_text}")
            elif role == Role.AGENT:
                messages.append(f"Bot: {content_text}")
        conversation = "\n".join(messages[-6:])  # Last 6 messages

        context_parts.append(
            f"### 失败任务: {desc}\n"
            f"分项得分: {json.dumps(scores, ensure_ascii=False)}\n"
            f"对话摘要:\n{conversation}\n"
        )

    context = "\n".join(context_parts)

    try:
        client = OpenAI(
            api_key=os.getenv("ZHIPU_API_KEY", ""),
            base_url="https://open.bigmodel.cn/api/paas/v4",
        )
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": f"以下是本次评测中失败的任务:\n\n{context}"},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error("Failed to analyze results: %s", e)
        return ""


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

    # Analyze failures
    analysis = analyze_failures(results_data, all_results, task_map)

    # Save analysis into results JSON (single write)
    if analysis:
        results_data.append({"_analysis": analysis})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)

    # Generate HTML report
    report_data = load_results(str(output_file))
    report_file = output_file.with_suffix(".zh.html")
    html = generate_html(report_data, str(output_file), lang="zh", analysis=analysis)
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

    if analysis:
        summary_lines.append(f"\n📋 失败分析:\n{analysis}")

    summary_lines.append(f"\n详细报告: {report_file}")

    return {
        "summary_text": "\n".join(summary_lines),
        "results_file": str(output_file),
        "report_file": str(report_file),
        "avg_score": avg,
        "total": total,
        "passed": passed,
        "analysis": analysis,
    }
