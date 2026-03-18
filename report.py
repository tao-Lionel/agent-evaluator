"""Generate a self-contained HTML evaluation report from results JSON."""

from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path


def load_results(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_html(results: list[dict], source_path: str) -> str:
    total = len(results)
    passed = sum(1 for r in results if r.get("overall_score", 0) >= 1.0 - 1e-6)
    failed = total - passed
    avg_score = sum(r.get("overall_score", 0) for r in results) / total if total else 0

    # Collect evaluator names
    eval_names = set()
    for r in results:
        eval_names.update(r.get("scores", {}).keys())
    eval_names = sorted(eval_names)

    # Per-evaluator averages
    eval_avgs = {}
    for name in eval_names:
        scores = [r.get("scores", {}).get(name, 0) for r in results]
        eval_avgs[name] = sum(scores) / len(scores) if scores else 0

    # Difficulty breakdown
    by_diff: dict[str, list[dict]] = {}
    for r in results:
        diff = r.get("task", {}).get("difficulty", "unknown")
        by_diff.setdefault(diff, []).append(r)

    # Build task cards
    task_cards = "\n".join(_render_task_card(r, eval_names) for r in results)

    # Build evaluator chart bars
    eval_bars = "\n".join(
        f'<div class="eval-bar-row">'
        f'<span class="eval-bar-label">{escape(name)}</span>'
        f'<div class="eval-bar-track"><div class="eval-bar-fill" style="width:{avg*100:.1f}%">'
        f'{avg:.0%}</div></div></div>'
        for name, avg in eval_avgs.items()
    )

    # Difficulty breakdown rows
    diff_rows = "\n".join(
        f'<tr><td>{escape(diff)}</td>'
        f'<td>{len(rs)}</td>'
        f'<td>{sum(r.get("overall_score",0) for r in rs)/len(rs):.0%}</td></tr>'
        for diff, rs in sorted(by_diff.items())
    )

    return HTML_TEMPLATE.format(
        source=escape(source_path),
        total=total,
        passed=passed,
        failed=failed,
        avg_score=f"{avg_score:.0%}",
        avg_score_pct=f"{avg_score*100:.1f}",
        eval_bars=eval_bars,
        diff_rows=diff_rows,
        task_cards=task_cards,
    )


def _render_task_card(result: dict, eval_names: list[str]) -> str:
    task_id = result.get("task_id", "unknown")
    task_info = result.get("task", {})
    desc = task_info.get("description", "")
    difficulty = task_info.get("difficulty", "")
    overall = result.get("overall_score", 0)
    terminated = result.get("terminated", "")
    steps = result.get("steps_taken", 0)
    is_pass = overall >= 1.0 - 1e-6

    # Score bars
    score_bars = ""
    for name in eval_names:
        score = result.get("scores", {}).get(name, 0)
        score_bars += (
            f'<div class="score-row">'
            f'<span class="score-label">{escape(name)}</span>'
            f'<div class="score-track"><div class="score-fill {_score_color(score)}" '
            f'style="width:{score*100:.1f}%"></div></div>'
            f'<span class="score-value">{score:.2f}</span></div>\n'
        )

    # Trajectory
    traj_html = _render_trajectory(result.get("trajectory", []))

    status_class = "pass" if is_pass else "fail"
    status_text = "PASS" if is_pass else "FAIL"

    return f"""
    <div class="task-card {status_class}">
      <div class="task-header">
        <span class="task-status {status_class}">{status_text}</span>
        <span class="task-id">{escape(task_id)}</span>
        <span class="task-diff">{escape(difficulty)}</span>
        <span class="task-meta">{steps} steps &middot; {escape(terminated)}</span>
        <span class="task-overall">{overall:.2f}</span>
      </div>
      <p class="task-desc">{escape(desc)}</p>
      <div class="scores-section">{score_bars}</div>
      <details class="trajectory-details">
        <summary>Conversation Trajectory</summary>
        <div class="trajectory">{traj_html}</div>
      </details>
    </div>"""


def _render_trajectory(trajectory: list[dict]) -> str:
    html = ""
    for msg in trajectory:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
        tool_results = msg.get("tool_results", [])

        if role == "system":
            html += f'<div class="msg msg-system"><span class="role">SYSTEM</span>{escape(content)}</div>\n'
        elif role == "user":
            html += f'<div class="msg msg-user"><span class="role">USER</span>{escape(content or "")}</div>\n'
        elif role == "agent":
            if content:
                text = content if len(content) <= 500 else content[:500] + "..."
                html += f'<div class="msg msg-agent"><span class="role">AGENT</span>{escape(text)}</div>\n'
            for tc in tool_calls:
                args = json.dumps(tc.get("arguments", {}), ensure_ascii=False)
                html += (
                    f'<div class="msg msg-tool-call"><span class="role">TOOL CALL</span>'
                    f'<code>{escape(tc.get("name", ""))}({escape(args)})</code></div>\n'
                )
        elif role == "env":
            for tr in tool_results:
                output = tr.get("output", "")
                if len(output) > 300:
                    output = output[:300] + "..."
                is_err = tr.get("error", False)
                cls = "msg-env-error" if is_err else "msg-env"
                html += (
                    f'<div class="msg {cls}"><span class="role">ENV</span>'
                    f'<code>{escape(output)}</code></div>\n'
                )
    return html


def _score_color(score: float) -> str:
    if score >= 0.9:
        return "green"
    elif score >= 0.5:
        return "yellow"
    return "red"


HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Evaluator Report</title>
<style>
:root {{
  --bg: #0d1117; --surface: #161b22; --border: #30363d;
  --text: #e6edf3; --text2: #8b949e; --green: #3fb950;
  --red: #f85149; --yellow: #d29922; --blue: #58a6ff;
  --accent: #1f6feb;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  background: var(--bg); color: var(--text); padding: 24px; line-height: 1.5; }}
h1 {{ font-size: 1.5rem; margin-bottom: 4px; }}
.source {{ color: var(--text2); font-size: 0.85rem; margin-bottom: 24px; }}

/* Dashboard */
.dashboard {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px; margin-bottom: 32px; }}
.stat-card {{ background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 16px; text-align: center; }}
.stat-value {{ font-size: 2rem; font-weight: 700; }}
.stat-value.green {{ color: var(--green); }}
.stat-value.red {{ color: var(--red); }}
.stat-value.blue {{ color: var(--blue); }}
.stat-label {{ color: var(--text2); font-size: 0.85rem; }}

/* Evaluator bars */
.eval-section {{ background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 20px; margin-bottom: 32px; }}
.eval-section h2 {{ font-size: 1.1rem; margin-bottom: 12px; }}
.eval-bar-row {{ display: flex; align-items: center; margin-bottom: 8px; }}
.eval-bar-label {{ width: 140px; font-size: 0.85rem; color: var(--text2); }}
.eval-bar-track {{ flex: 1; height: 24px; background: var(--border); border-radius: 4px; overflow: hidden; }}
.eval-bar-fill {{ height: 100%; background: var(--accent); border-radius: 4px;
  display: flex; align-items: center; justify-content: flex-end; padding-right: 8px;
  font-size: 0.8rem; font-weight: 600; min-width: 40px; }}
.diff-table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
.diff-table th, .diff-table td {{ padding: 6px 12px; text-align: left;
  border-bottom: 1px solid var(--border); font-size: 0.9rem; }}
.diff-table th {{ color: var(--text2); }}

/* Task cards */
.task-card {{ background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
.task-card.pass {{ border-left: 3px solid var(--green); }}
.task-card.fail {{ border-left: 3px solid var(--red); }}
.task-header {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 8px; }}
.task-status {{ font-weight: 700; font-size: 0.8rem; padding: 2px 8px; border-radius: 4px; }}
.task-status.pass {{ background: rgba(63,185,80,0.15); color: var(--green); }}
.task-status.fail {{ background: rgba(248,81,73,0.15); color: var(--red); }}
.task-id {{ font-weight: 600; }}
.task-diff {{ background: var(--border); padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }}
.task-meta {{ color: var(--text2); font-size: 0.85rem; margin-left: auto; }}
.task-overall {{ font-weight: 700; font-size: 1.1rem; }}
.task-desc {{ color: var(--text2); font-size: 0.9rem; margin-bottom: 12px; }}

/* Score bars */
.score-row {{ display: flex; align-items: center; margin-bottom: 4px; }}
.score-label {{ width: 120px; font-size: 0.8rem; color: var(--text2); }}
.score-track {{ flex: 1; height: 16px; background: var(--border); border-radius: 3px; overflow: hidden; }}
.score-fill {{ height: 100%; border-radius: 3px; transition: width 0.3s; }}
.score-fill.green {{ background: var(--green); }}
.score-fill.yellow {{ background: var(--yellow); }}
.score-fill.red {{ background: var(--red); }}
.score-value {{ width: 40px; text-align: right; font-size: 0.8rem; font-weight: 600; }}

/* Trajectory */
.trajectory-details {{ margin-top: 12px; }}
.trajectory-details summary {{ cursor: pointer; color: var(--blue); font-size: 0.9rem; }}
.trajectory {{ margin-top: 8px; max-height: 600px; overflow-y: auto;
  border: 1px solid var(--border); border-radius: 6px; padding: 12px; }}
.msg {{ padding: 8px 12px; margin-bottom: 6px; border-radius: 6px; font-size: 0.85rem;
  white-space: pre-wrap; word-break: break-word; }}
.role {{ display: inline-block; font-weight: 700; font-size: 0.75rem; margin-right: 8px;
  padding: 1px 6px; border-radius: 3px; }}
.msg-system {{ background: rgba(88,166,255,0.08); }}
.msg-system .role {{ background: rgba(88,166,255,0.2); color: var(--blue); }}
.msg-user {{ background: rgba(63,185,80,0.08); }}
.msg-user .role {{ background: rgba(63,185,80,0.2); color: var(--green); }}
.msg-agent {{ background: rgba(210,153,34,0.08); }}
.msg-agent .role {{ background: rgba(210,153,34,0.2); color: var(--yellow); }}
.msg-tool-call {{ background: rgba(139,148,158,0.08); }}
.msg-tool-call .role {{ background: rgba(139,148,158,0.2); color: var(--text2); }}
.msg-tool-call code {{ font-size: 0.82rem; }}
.msg-env {{ background: rgba(139,148,158,0.05); }}
.msg-env .role {{ background: rgba(139,148,158,0.15); color: var(--text2); }}
.msg-env code {{ font-size: 0.82rem; }}
.msg-env-error {{ background: rgba(248,81,73,0.08); }}
.msg-env-error .role {{ background: rgba(248,81,73,0.2); color: var(--red); }}
</style>
</head>
<body>
<h1>Agent Evaluator Report</h1>
<p class="source">Source: {source}</p>

<div class="dashboard">
  <div class="stat-card">
    <div class="stat-value blue">{total}</div>
    <div class="stat-label">Total Tasks</div>
  </div>
  <div class="stat-card">
    <div class="stat-value green">{passed}</div>
    <div class="stat-label">Passed</div>
  </div>
  <div class="stat-card">
    <div class="stat-value red">{failed}</div>
    <div class="stat-label">Failed</div>
  </div>
  <div class="stat-card">
    <div class="stat-value blue">{avg_score}</div>
    <div class="stat-label">Average Score</div>
  </div>
</div>

<div class="eval-section">
  <h2>Evaluator Breakdown</h2>
  {eval_bars}
  <table class="diff-table">
    <tr><th>Difficulty</th><th>Tasks</th><th>Avg Score</th></tr>
    {diff_rows}
  </table>
</div>

<h2 style="margin-bottom:12px">Task Results</h2>
{task_cards}

<p style="color:var(--text2);font-size:0.8rem;margin-top:24px;text-align:center">
  Generated by Agent Evaluator
</p>
</body>
</html>"""


def main():
    if len(sys.argv) < 2:
        print("Usage: python report.py <results.json> [output.html]")
        sys.exit(1)

    results_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else results_path.replace(".json", ".html")

    results = load_results(results_path)
    html = generate_html(results, results_path)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report generated: {output_path}")


if __name__ == "__main__":
    main()
