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
