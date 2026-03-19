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
