"""Agent Evaluator MVP — Main Entry Point"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

# ── Ensure project root is on sys.path ──
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

from core.types import Role, Task, EvalResult
from core.registry import registry
from core.orchestrator import Orchestrator
from core.metrics import compute_trial_stats, check_saturation
from report import generate_html, load_results

# Register all pluggable components
import adapters  # noqa: F401
import environments  # noqa: F401
import evaluators  # noqa: F401
import users  # noqa: F401


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Expand ${ENV_VAR} references
    for key, value in os.environ.items():
        raw = raw.replace(f"${{{key}}}", value)

    return yaml.safe_load(raw)


def load_tasks(path: str) -> list[Task]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Task.from_dict(item) for item in data]


def print_result(result: EvalResult, task: Task):
    status_icon = "PASS" if result.overall_score >= 1.0 - 1e-6 else "FAIL"
    print(f"  [{status_icon}] {result.task_id} ({task.difficulty})")
    print(f"       Task     : {task.description}")
    print(f"       User     : {task.initial_message}")
    print()

    # Agent trajectory: tool calls + replies
    step_num = 0
    for msg in result.trajectory:
        if msg.role == Role.AGENT:
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    step_num += 1
                    args_str = json.dumps(tc.arguments, ensure_ascii=False)
                    print(f"       Step {step_num:<3}  : {tc.name}({args_str})")
            if msg.content:
                text = msg.content.replace("\n", " ").strip()
                if len(text) > 120:
                    text = text[:120] + "..."
                print(f"       Reply    : {text}")
    print()

    # Scores
    print(f"       Result   : {result.terminated.value}  ({result.elapsed_seconds:.1f}s)")
    for name, score in result.scores.items():
        bar = "+" * int(score * 10) + "-" * (10 - int(score * 10))
        print(f"       {name:15s}: {score:.2f} [{bar}]")
    print(f"       Overall     : {result.overall_score:.2f}")
    print()
    print("  " + "- " * 28)
    print()


def print_summary(results: list[EvalResult]):
    total = len(results)
    passed = sum(1 for r in results if r.overall_score >= 1.0 - 1e-6)
    avg_score = sum(r.overall_score for r in results) / total if total else 0

    print("=" * 60)
    print(f"  Tasks: {total}  |  Passed: {passed}  |  Failed: {total - passed}")
    print(f"  Average Score: {avg_score:.2%}")
    print()

    # Per-evaluator breakdown
    all_evaluator_names = set()
    for r in results:
        all_evaluator_names.update(r.scores.keys())
    for name in sorted(all_evaluator_names):
        scores = [r.scores.get(name, 0) for r in results]
        avg = sum(scores) / len(scores)
        print(f"  {name:20s}: {avg:.2%}")

    # Difficulty breakdown
    by_difficulty: dict[str, list[EvalResult]] = {}
    for r in results:
        task = next((t for t in _all_tasks if t.id == r.task_id), None)
        if task:
            by_difficulty.setdefault(task.difficulty, []).append(r)
    if len(by_difficulty) > 1:
        print()
        for diff, rs in sorted(by_difficulty.items()):
            avg = sum(r.overall_score for r in rs) / len(rs)
            print(f"  [{diff:6s}] {avg:.2%}  ({len(rs)} tasks)")

    print("=" * 60)


_all_tasks: list[Task] = []


def main():
    global _all_tasks

    # Fix Windows console encoding
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)

    # Logging
    log_level = config.get("run", {}).get("log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load tasks
    scenario_path = config["scenarios"]
    if not os.path.isabs(scenario_path):
        scenario_path = str(PROJECT_ROOT / scenario_path)
    tasks = load_tasks(scenario_path)
    _all_tasks = tasks

    print(f"\nLoaded {len(tasks)} tasks from {scenario_path}\n")

    # Build environment
    env_type = config["environment"]["type"]
    EnvClass = registry.get_environment(env_type)
    env = EnvClass()

    # Build agent adapter
    agent_cfg = config["agent"]
    AdapterClass = registry.get_adapter(agent_cfg["adapter"])
    adapter_params = {k: v for k, v in agent_cfg.items() if k != "adapter"}
    # Pass tool schemas if adapter accepts them
    if env.get_tool_schemas():
        adapter_params.setdefault("tools", env.get_tool_schemas())
    adapter = AdapterClass(**adapter_params)

    # Build evaluators
    # Supports two formats:
    #   evaluators: [state_match, action_match]           # simple list
    #   evaluators:                                        # with params
    #     - name: llm_judge
    #       model: glm-5
    #     - state_match                                    # mixed OK
    evaluator_configs = config.get("evaluators", ["state_match", "action_match"])
    evaluator_map = {}
    evaluator_names = []
    for item in evaluator_configs:
        if isinstance(item, str):
            name, params = item, {}
        elif isinstance(item, dict):
            name = item["name"]
            params = {k: v for k, v in item.items() if k != "name"}
        else:
            continue
        EvalClass = registry.get_evaluator(name)
        evaluator_map[name] = EvalClass(**params)
        evaluator_names.append(name)

    # Build user simulator (optional)
    user = None
    if "user" in config:
        user_cfg = config["user"]
        UserClass = registry.get_user(user_cfg["type"])
        user_params = {k: v for k, v in user_cfg.items() if k != "type"}
        user = UserClass(**user_params)

    # Build orchestrator (on_progress callback attached after definition below)
    orchestrator = Orchestrator(adapter, env, evaluator_map, user=user)

    # Run evaluation
    num_trials = config.get("run", {}).get("num_trials", 1)
    all_results: list[EvalResult] = []

    agent_name = config.get("name", "").strip()

    # ── Live progress callback for ws_bot adapter ──
    _progress_status = ""
    def _on_ws_progress(event: str, data: dict):
        nonlocal _progress_status
        elapsed = data.get("elapsed", 0)
        if event == "ws_connected":
            _progress_status = "已连接"
        elif event == "ws_tool_start":
            _progress_status = f"调用工具 {data['name']}"
        elif event == "ws_tool_result":
            status = data.get("status", "done")
            _progress_status = f"工具 {data['name']} {status}"
        elif event == "ws_content":
            _progress_status = f"接收回复 {data['chars']}字"
        elif event == "ws_done":
            _progress_status = f"Agent 完成 ({data['chars']}字, {data['tools']}次工具)"
        elif event == "ws_error":
            _progress_status = f"Agent 错误: {data.get('message', '')[:30]}"
        elif event == "ws_msg_timeout":
            _progress_status = f"消息超时 ({data['timeout']:.0f}s)"
        elif event == "ws_total_timeout":
            _progress_status = f"总超时 ({data['elapsed']:.0f}s)"
        elif event == "ws_closed":
            _progress_status = "连接关闭"
        elif event == "eval_start":
            _progress_status = f"评估中 [{data['name']}]"

    if hasattr(adapter, "on_progress"):
        adapter.on_progress = _on_ws_progress
    orchestrator.on_progress = _on_ws_progress

    print("-" * 60)
    agent_display = agent_name or agent_cfg.get('model', agent_cfg['adapter'])
    print(f"  Agent   : {agent_display}")
    print(f"  Env     : {env_type}")
    print(f"  User    : {config.get('user', {}).get('type', 'none')}")
    print(f"  Evals   : {', '.join(evaluator_names)}")
    print(f"  Trials  : {num_trials}")
    print("-" * 60)
    print()

    start = time.time()

    for trial in range(num_trials):
        if num_trials > 1:
            print(f"── Trial {trial + 1}/{num_trials} ──\n")

        for i, task in enumerate(tasks, 1):
            total = len(tasks)
            prefix = f"  [{i}/{total}]"
            _progress_status = ""
            print(f"{prefix} {task.id} ({task.difficulty})", end="", flush=True)
            task_start = time.time()

            # Live progress thread — shows elapsed time + ws_bot status
            import threading
            _stop_timer = threading.Event()
            def _print_live():
                last_status = ""
                while not _stop_timer.wait(1):
                    elapsed = time.time() - task_start
                    status_part = f" | {_progress_status}" if _progress_status else ""
                    line = f"\r{prefix} {task.id} ({task.difficulty}) {elapsed:.0f}s{status_part}"
                    # Pad to overwrite previous longer line
                    pad = max(0, len(last_status) - len(line))
                    print(line + " " * pad, end="", flush=True)
                    last_status = line
            timer = threading.Thread(target=_print_live, daemon=True)
            timer.start()

            result = orchestrator.run(task)

            _stop_timer.set()
            elapsed = time.time() - task_start
            status = "PASS" if result.overall_score >= 1.0 - 1e-6 else f"{result.overall_score:.2f}"
            final = f"\r{prefix} {task.id} ({task.difficulty}) -> {status}  [{elapsed:.0f}s]"
            print(final + " " * 30)

            all_results.append(result)
            print_result(result, task)

    elapsed = time.time() - start

    # Summary
    print_summary(all_results)

    # Saturation warning
    all_scores = [r.overall_score for r in all_results]
    sat_warning = check_saturation(all_scores)
    if sat_warning:
        print(f"  *** {sat_warning} ***\n")

    # Trial consistency stats (when num_trials > 1)
    trial_stats = None
    if num_trials > 1:
        results_by_task: dict[str, list[float]] = {}
        for r in all_results:
            results_by_task.setdefault(r.task_id, []).append(r.overall_score)
        k_values = [k for k in [1, 3, 5] if k <= num_trials]
        trial_stats = compute_trial_stats(results_by_task, k_values=k_values)
        print("  Consistency Metrics:")
        for tid, st in trial_stats.items():
            parts = [f"pass_rate={st['pass_rate']:.0%}"]
            for k in k_values:
                parts.append(f"pass@{k}={st[f'pass@{k}']:.2f}")
                parts.append(f"pass^{k}={st[f'pass^{k}']:.2f}")
            print(f"    {tid}: {', '.join(parts)}")
        print()

    print(f"  Total time: {elapsed:.1f}s\n")

    # Save results
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    # Use agent name in filename for easy identification
    if agent_name:
        import re as _re
        safe_name = _re.sub(r'[\\/:*?"<>|\s]+', '-', agent_name).strip('-')
        output_file = output_dir / f"results_{safe_name}_{int(time.time())}.json"
    else:
        output_file = output_dir / f"results_{int(time.time())}.json"
    # Build task lookup for adding task info to results
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
    # Append metadata
    meta = {}
    if agent_name:
        meta["_agent_name"] = agent_name
    if trial_stats:
        meta["_trial_stats"] = trial_stats
    if sat_warning:
        meta["_saturation_warning"] = sat_warning
    if meta:
        results_data.append(meta)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            results_data,
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"  Results saved to {output_file}")

    # Generate HTML report (Chinese only)
    report_data = load_results(str(output_file))
    report_file = output_file.with_suffix(".html")
    html = generate_html(report_data, str(output_file), lang="zh")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Report saved to {report_file}")
    print()


if __name__ == "__main__":
    main()
