"""Generate a side-by-side comparison report from two evaluation results.

Usage:
  python compare.py results/results_A.json results/results_B.json
  python compare.py results/results_A.json results/results_B.json -o compare.html
"""
from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path


def load_results(path: str) -> tuple[list[dict], dict]:
    """Load results and extract metadata."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = {}
    results = []
    for r in data:
        if any(k.startswith("_") for k in r):
            for k, v in r.items():
                meta[k] = v
        else:
            results.append(r)
    return results, meta


def _score_bar(score: float, width: int = 100) -> str:
    color = "#34d399" if score >= 0.9 else "#fbbf24" if score >= 0.5 else "#f87171"
    return (
        f'<div class="cmp-bar" style="width:{width}px">'
        f'<div class="cmp-fill" style="width:{score*100:.1f}%;background:{color}"></div>'
        f'</div>'
    )


def _diff_badge(a: float, b: float) -> str:
    diff = b - a
    if abs(diff) < 0.005:
        return '<span class="diff neutral">-</span>'
    if diff > 0:
        return f'<span class="diff up">+{diff:.2f}</span>'
    return f'<span class="diff down">{diff:.2f}</span>'


def generate_compare_html(
    results_a: list[dict], meta_a: dict, path_a: str,
    results_b: list[dict], meta_b: dict, path_b: str,
) -> str:
    name_a = meta_a.get("_agent_name", Path(path_a).stem)
    name_b = meta_b.get("_agent_name", Path(path_b).stem)

    # Index by task_id
    map_a = {r["task_id"]: r for r in results_a}
    map_b = {r["task_id"]: r for r in results_b}
    all_task_ids = list(dict.fromkeys(
        [r["task_id"] for r in results_a] + [r["task_id"] for r in results_b]
    ))

    # Collect all evaluator names
    eval_names = set()
    for r in results_a + results_b:
        eval_names.update(r.get("scores", {}).keys())
    eval_names = sorted(eval_names)

    # ── Summary stats ──
    def _stats(results):
        total = len(results)
        if total == 0:
            return {"total": 0, "passed": 0, "avg": 0, "evals": {}}
        passed = sum(1 for r in results if r.get("overall_score", 0) >= 1.0 - 1e-6)
        avg = sum(r.get("overall_score", 0) for r in results) / total
        evals = {}
        for name in eval_names:
            scores = [r.get("scores", {}).get(name, 0) for r in results if name in r.get("scores", {})]
            evals[name] = sum(scores) / len(scores) if scores else 0
        return {"total": total, "passed": passed, "avg": avg, "evals": evals}

    stats_a = _stats(results_a)
    stats_b = _stats(results_b)

    # ── Summary table ──
    summary_rows = f"""
    <tr>
      <td>总任务数</td>
      <td>{stats_a['total']}</td>
      <td>{stats_b['total']}</td>
      <td></td>
    </tr>
    <tr>
      <td>通过数</td>
      <td class="{'green' if stats_a['passed'] > 0 else ''}">{stats_a['passed']}</td>
      <td class="{'green' if stats_b['passed'] > 0 else ''}">{stats_b['passed']}</td>
      <td>{_diff_badge(stats_a['passed'], stats_b['passed'])}</td>
    </tr>
    <tr>
      <td>平均得分</td>
      <td>{stats_a['avg']:.2%}</td>
      <td>{stats_b['avg']:.2%}</td>
      <td>{_diff_badge(stats_a['avg'], stats_b['avg'])}</td>
    </tr>"""

    for name in eval_names:
        avg_a = stats_a["evals"].get(name, 0)
        avg_b = stats_b["evals"].get(name, 0)
        summary_rows += f"""
    <tr>
      <td><code>{escape(name)}</code></td>
      <td>{avg_a:.2%}</td>
      <td>{avg_b:.2%}</td>
      <td>{_diff_badge(avg_a, avg_b)}</td>
    </tr>"""

    # ── Per-task comparison ──
    task_rows = ""
    for tid in all_task_ids:
        ra = map_a.get(tid)
        rb = map_b.get(tid)
        desc = ""
        if ra:
            desc = ra.get("task", {}).get("description", "")
        elif rb:
            desc = rb.get("task", {}).get("description", "")

        score_a = ra.get("overall_score", 0) if ra else None
        score_b = rb.get("overall_score", 0) if rb else None

        sa_str = f"{score_a:.2f}" if score_a is not None else "-"
        sb_str = f"{score_b:.2f}" if score_b is not None else "-"

        diff_html = ""
        if score_a is not None and score_b is not None:
            diff_html = _diff_badge(score_a, score_b)

        # Per-evaluator scores
        eval_cells = ""
        for name in eval_names:
            ea = ra.get("scores", {}).get(name) if ra else None
            eb = rb.get("scores", {}).get(name) if rb else None
            ea_str = f"{ea:.2f}" if ea is not None else "-"
            eb_str = f"{eb:.2f}" if eb is not None else "-"
            eval_cells += f"<td>{ea_str}</td><td>{eb_str}</td>"

        task_rows += f"""
    <tr>
      <td class="task-id-cell"><code>{escape(tid)}</code><br><small>{escape(desc[:40])}</small></td>
      <td>{sa_str}</td>
      <td>{sb_str}</td>
      <td>{diff_html}</td>
      {eval_cells}
    </tr>"""

    # Eval header columns
    eval_headers = ""
    for name in eval_names:
        eval_headers += f'<th colspan="2">{escape(name)}</th>'

    eval_subheaders = ""
    for _ in eval_names:
        eval_subheaders += f"<th>A</th><th>B</th>"

    return HTML_TEMPLATE.format(
        title=f"对比报告: {escape(name_a)} vs {escape(name_b)}",
        name_a=escape(name_a),
        name_b=escape(name_b),
        path_a=escape(path_a),
        path_b=escape(path_b),
        summary_rows=summary_rows,
        task_rows=task_rows,
        eval_headers=eval_headers,
        eval_subheaders=eval_subheaders,
    )


HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {{
  --bg: #0b0f14; --surface: #141a22; --surface2: #1a2230;
  --border: #252e3b; --text: #e2e8f0; --text2: #7a8ba0; --text3: #4a5568;
  --green: #34d399; --red: #f87171; --yellow: #fbbf24; --blue: #60a5fa;
  --accent: #3b82f6; --radius: 10px;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; }}
.container {{ max-width: 1100px; margin: 0 auto; padding: 40px 24px; }}
code {{ font-family: 'JetBrains Mono', monospace; font-size: 0.82em; }}
h1 {{ font-size: 1.5rem; font-weight: 700; margin-bottom: 6px; }}
.source {{ color: var(--text3); font-size: 0.78rem; margin-bottom: 32px; }}

/* Tags */
.tag {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; }}
.tag-a {{ background: rgba(59,130,246,0.15); color: var(--blue); }}
.tag-b {{ background: rgba(168,85,247,0.15); color: #a855f7; }}

/* Tables */
.section {{ background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 24px; margin-bottom: 24px; }}
.section h2 {{ font-size: 1.05rem; font-weight: 600; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid var(--border); }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); font-size: 0.85rem; }}
th {{ color: var(--text2); font-weight: 500; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }}
td code {{ font-size: 0.82rem; }}
td small {{ color: var(--text3); }}
.task-id-cell {{ min-width: 140px; }}
.green {{ color: var(--green); }}
.red {{ color: var(--red); }}

/* Diff badges */
.diff {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; }}
.diff.up {{ background: rgba(52,211,153,0.12); color: var(--green); }}
.diff.down {{ background: rgba(248,113,113,0.12); color: var(--red); }}
.diff.neutral {{ background: rgba(122,139,160,0.08); color: var(--text3); }}

/* Score bars */
.cmp-bar {{ height: 14px; background: var(--surface2); border-radius: 3px; overflow: hidden; display: inline-block; vertical-align: middle; }}
.cmp-fill {{ height: 100%; border-radius: 3px; }}

.footer {{ color: var(--text3); font-size: 0.75rem; text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--border); }}
@media (max-width: 640px) {{ .container {{ padding: 20px 16px; }} }}
</style>
</head>
<body>
<div class="container">
<h1>{title}</h1>
<p class="source">
  <span class="tag tag-a">A</span> {name_a} — {path_a}<br>
  <span class="tag tag-b">B</span> {name_b} — {path_b}
</p>

<div class="section">
  <h2>总览对比</h2>
  <table>
    <tr><th>指标</th><th><span class="tag tag-a">A</span></th><th><span class="tag tag-b">B</span></th><th>变化</th></tr>
    {summary_rows}
  </table>
</div>

<div class="section">
  <h2>逐任务对比</h2>
  <div style="overflow-x:auto">
  <table>
    <tr>
      <th>任务</th>
      <th><span class="tag tag-a">A</span> 总分</th>
      <th><span class="tag tag-b">B</span> 总分</th>
      <th>变化</th>
      {eval_headers}
    </tr>
    <tr>
      <th></th><th></th><th></th><th></th>
      {eval_subheaders}
    </tr>
    {task_rows}
  </table>
  </div>
</div>

<p class="footer">由 Agent Evaluator 生成</p>
</div>
</body>
</html>"""


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare.py <results_A.json> <results_B.json> [-o output.html]")
        sys.exit(1)

    path_a = sys.argv[1]
    path_b = sys.argv[2]

    output = None
    if "-o" in sys.argv:
        idx = sys.argv.index("-o")
        if idx + 1 < len(sys.argv):
            output = sys.argv[idx + 1]

    if not output:
        output = "results/compare.html"

    results_a, meta_a = load_results(path_a)
    results_b, meta_b = load_results(path_b)

    html = generate_compare_html(results_a, meta_a, path_a, results_b, meta_b, path_b)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Comparison report saved to {output}")


if __name__ == "__main__":
    main()
