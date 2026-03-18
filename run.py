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

# Register all pluggable components
import adapters  # noqa: F401
import environments  # noqa: F401
import evaluators  # noqa: F401


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
    print(f"       Result   : {result.terminated.value}")
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
    evaluator_names = config.get("evaluators", ["state_match", "action_match"])
    evaluator_map = {}
    for name in evaluator_names:
        EvalClass = registry.get_evaluator(name)
        evaluator_map[name] = EvalClass()

    # Build orchestrator
    orchestrator = Orchestrator(adapter, env, evaluator_map)

    # Run evaluation
    num_trials = config.get("run", {}).get("num_trials", 1)
    all_results: list[EvalResult] = []

    print("-" * 60)
    print(f"  Agent   : {agent_cfg.get('model', agent_cfg['adapter'])}")
    print(f"  Env     : {env_type}")
    print(f"  Evals   : {', '.join(evaluator_names)}")
    print(f"  Trials  : {num_trials}")
    print("-" * 60)
    print()

    start = time.time()

    for trial in range(num_trials):
        if num_trials > 1:
            print(f"── Trial {trial + 1}/{num_trials} ──\n")

        for task in tasks:
            result = orchestrator.run(task)
            all_results.append(result)
            print_result(result, task)

    elapsed = time.time() - start

    # Summary
    print_summary(all_results)
    print(f"\n  Total time: {elapsed:.1f}s\n")

    # Save results
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
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
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            results_data,
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"  Results saved to {output_file}\n")


if __name__ == "__main__":
    main()
