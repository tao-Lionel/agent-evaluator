# tests/test_concurrent_eval.py
"""Tests for concurrent evaluator execution."""
from __future__ import annotations

import sys
import time
import threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import Role, Message, ToolCall, Task
from core.base import AgentAdapter, Evaluator, Environment
from core.orchestrator import Orchestrator
from environments.mock_db import MockDBEnvironment


class InstantAgent(AgentAdapter):
    def reset(self): pass
    def act(self, messages):
        return Message(role=Role.AGENT, content="Done. ###DONE###")


class SlowEvaluator(Evaluator):
    """Evaluator that takes a configurable delay — used to test concurrency."""
    def __init__(self, score: float, delay: float = 0.5):
        self._score = score
        self._delay = delay
        self.last_reason = ""
        self.thread_id: int | None = None

    def evaluate(self, task, trajectory, env):
        self.thread_id = threading.current_thread().ident
        time.sleep(self._delay)
        self.last_reason = f"SlowEval score={self._score}"
        return self._score


def test_evaluators_run_concurrently():
    """Three 0.5s evaluators should complete in ~0.5s (not 1.5s) if concurrent."""
    task = Task(
        id="t-concurrent", description="test", initial_message="hi",
        initial_state={}, max_steps=1, single_turn=True,
    )
    evals = {
        "slow_a": SlowEvaluator(1.0, delay=0.5),
        "slow_b": SlowEvaluator(0.8, delay=0.5),
        "slow_c": SlowEvaluator(0.6, delay=0.5),
    }
    agent = InstantAgent()
    env = MockDBEnvironment()
    orch = Orchestrator(agent, env, evals)

    start = time.time()
    result = orch.run(task)
    elapsed = time.time() - start

    assert result.scores["slow_a"] == 1.0
    assert result.scores["slow_b"] == 0.8
    assert result.scores["slow_c"] == 0.6
    assert abs(result.overall_score - 1.0 * 0.8 * 0.6) < 0.01

    # Should be concurrent: ~0.5s, not ~1.5s
    assert elapsed < 1.2, f"Expected <1.2s (concurrent), got {elapsed:.2f}s"
    print(f"  test_evaluators_run_concurrently PASSED ({elapsed:.2f}s)")


def test_evaluator_crash_isolation():
    """A crashing evaluator should not break other evaluators."""
    task = Task(
        id="t-crash", description="test", initial_message="hi",
        initial_state={}, max_steps=1, single_turn=True,
    )

    class CrashEvaluator(Evaluator):
        def __init__(self):
            self.last_reason = ""
        def evaluate(self, task, trajectory, env):
            raise RuntimeError("boom")

    evals = {
        "good": SlowEvaluator(1.0, delay=0.1),
        "crash": CrashEvaluator(),
    }
    agent = InstantAgent()
    env = MockDBEnvironment()
    result = Orchestrator(agent, env, evals).run(task)

    assert result.scores["good"] == 1.0
    assert result.scores["crash"] == 0.0
    assert result.overall_score == 1.0  # crashed evaluator skipped in overall
    print("  test_evaluator_crash_isolation PASSED")


def test_concurrent_threads_are_different():
    """Verify evaluators actually run on different threads."""
    task = Task(
        id="t-threads", description="test", initial_message="hi",
        initial_state={}, max_steps=1, single_turn=True,
    )
    evals = {
        "a": SlowEvaluator(1.0, delay=0.3),
        "b": SlowEvaluator(1.0, delay=0.3),
    }
    agent = InstantAgent()
    env = MockDBEnvironment()
    Orchestrator(agent, env, evals).run(task)

    assert evals["a"].thread_id is not None
    assert evals["b"].thread_id is not None
    assert evals["a"].thread_id != evals["b"].thread_id, "Evaluators should run on different threads"
    print("  test_concurrent_threads_are_different PASSED")


def test_profiling_data_in_result():
    """EvalResult should contain profiling data."""
    task = Task(
        id="t-profile", description="test", initial_message="hi",
        initial_state={}, max_steps=1, single_turn=True,
    )
    evals = {"quick": SlowEvaluator(1.0, delay=0.1)}
    agent = InstantAgent()
    env = MockDBEnvironment()
    result = Orchestrator(agent, env, evals).run(task)

    assert hasattr(result, "profiling"), "EvalResult should have profiling field"
    assert "agent_act" in result.profiling
    assert "eval" in result.profiling
    assert "total" in result.profiling
    print(f"  test_profiling_data_in_result PASSED (profiling={result.profiling})")


if __name__ == "__main__":
    print("\n=== Concurrent Evaluator Tests ===\n")
    test_evaluators_run_concurrently()
    test_evaluator_crash_isolation()
    test_concurrent_threads_are_different()
    test_profiling_data_in_result()
    print("\n=== All concurrent eval tests passed ===\n")
