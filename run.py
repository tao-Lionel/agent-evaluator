"""Agent Evaluator MVP — Main Entry Point"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml

# ── Ensure project root is on sys.path ──
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.types import Task, EvalResult
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


def print_result(result: EvalResult):
    status_icon = "PASS" if result.overall_score >= 1.0 - 1e-6 else "FAIL"
    print(f"  [{status_icon}] {result.task_id}")
    print(f"       Termination : {result.terminated.value}")
    print(f"       Steps       : {result.steps_taken}")
    for name, score in result.scores.items():
        bar = "+" * int(score * 10) + "-" * (10 - int(score * 10))
        print(f"       {name:15s}: {score:.2f} [{bar}]")
    print(f"       Overall     : {result.overall_score:.2f}")
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
    adapter = AdapterClass(
        model=agent_cfg["model"],
        api_key=agent_cfg["api_key"],
        base_url=agent_cfg.get("base_url"),
        tools=env.get_tool_schemas(),
        temperature=agent_cfg.get("temperature", 0.0),
    )

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
    print(f"  Agent   : {agent_cfg['model']}")
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
            print_result(result)

    elapsed = time.time() - start

    # Summary
    print_summary(all_results)
    print(f"\n  Total time: {elapsed:.1f}s\n")

    # Save results
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"results_{int(time.time())}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            [r.summary() for r in all_results],
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"  Results saved to {output_file}\n")


if __name__ == "__main__":
    main()
