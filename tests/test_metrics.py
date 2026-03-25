"""Unit tests for consistency metrics, saturation warning, and calibration."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.metrics import (
    compute_pass_at_k,
    compute_pass_power_k,
    compute_trial_stats,
    check_saturation,
)
from core.calibration import compute_calibration


def test_pass_at_k_all_pass():
    assert compute_pass_at_k(n=5, c=5, k=1) == 1.0
    assert compute_pass_at_k(n=5, c=5, k=3) == 1.0


def test_pass_at_k_none_pass():
    assert compute_pass_at_k(n=5, c=0, k=1) == 0.0
    assert compute_pass_at_k(n=5, c=0, k=3) == 0.0


def test_pass_at_k_partial():
    p1 = compute_pass_at_k(n=5, c=3, k=1)
    p3 = compute_pass_at_k(n=5, c=3, k=3)
    assert abs(p1 - 0.6) < 0.01
    assert p3 > p1  # pass@k increases with k


def test_pass_power_k_all_pass():
    assert compute_pass_power_k(n=5, c=5, k=3) == 1.0


def test_pass_power_k_partial():
    pk = compute_pass_power_k(n=5, c=3, k=3)
    assert abs(pk - 0.6**3) < 0.01


def test_pass_power_k_none_pass():
    assert compute_pass_power_k(n=5, c=0, k=1) == 0.0


def test_k_greater_than_n():
    p = compute_pass_at_k(n=3, c=2, k=5)
    assert 0 <= p <= 1.0


def test_edge_zero_trials():
    assert compute_pass_at_k(n=0, c=0, k=1) == 0.0
    assert compute_pass_power_k(n=0, c=0, k=1) == 0.0


def test_trial_stats():
    results = {
        "task-1": [1.0, 1.0, 0.5, 1.0, 0.0],
        "task-2": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    stats = compute_trial_stats(results, k_values=[1, 3])
    assert stats["task-1"]["n"] == 5
    assert stats["task-1"]["c"] == 3
    assert abs(stats["task-1"]["pass_rate"] - 0.6) < 0.01
    assert stats["task-1"]["pass@1"] > 0
    assert stats["task-2"]["pass@1"] == 0.0
    assert stats["task-2"]["pass^1"] == 0.0


def test_saturation_warning():
    # 90% pass rate should trigger warning
    scores = [1.0] * 9 + [0.0]
    warning = check_saturation(scores, threshold=0.85)
    assert warning is not None
    assert "Saturation" in warning

    # 50% pass rate should NOT trigger warning
    scores = [1.0] * 5 + [0.0] * 5
    assert check_saturation(scores, threshold=0.85) is None

    # Empty list
    assert check_saturation([]) is None


def test_calibration_perfect_agreement():
    pairs = [(0.8, 0.8), (1.0, 1.0), (0.6, 0.6)]
    report = compute_calibration(pairs)
    assert report["agreement_rate"] == 1.0
    assert report["mean_divergence"] == 0.0
    assert report["needs_recalibration"] is False


def test_calibration_poor_agreement():
    pairs = [(1.0, 0.2), (0.8, 0.1), (0.9, 0.3), (1.0, 0.0)]
    report = compute_calibration(pairs, threshold=0.20)
    assert report["agreement_rate"] < 0.80
    assert report["needs_recalibration"] is True
    assert report["llm_bias"] > 0  # LLM scores higher than human


def test_calibration_mixed():
    pairs = [(0.8, 0.9), (0.6, 0.5), (1.0, 0.4), (0.7, 0.7), (0.5, 0.6)]
    report = compute_calibration(pairs, threshold=0.20)
    assert report["total"] == 5
    assert 0 <= report["agreement_rate"] <= 1.0
    assert report["max_divergence"] >= report["mean_divergence"]


def test_calibration_empty():
    report = compute_calibration([])
    assert report["total"] == 0
    assert report["needs_recalibration"] is False


def test_progress_rate_tracking():
    """Verify Orchestrator tracks step_rewards and progress_rate."""
    from core.types import Role, Message, ToolCall, Task
    from core.base import AgentAdapter
    from core.orchestrator import Orchestrator
    from environments.mock_db import MockDBEnvironment
    from evaluators.state_evaluator import StateEvaluator

    class ScriptedAgent(AgentAdapter):
        def __init__(self, script):
            self._script = list(script)
            self._step = 0
        def reset(self):
            self._step = 0
        def act(self, messages):
            if self._step < len(self._script):
                msg = self._script[self._step]
                self._step += 1
                return msg
            return Message(role=Role.AGENT, content="###DONE###")

    task = Task(
        id="progress-test",
        description="Cancel order ORD-1001",
        initial_message="Cancel ORD-1001",
        initial_state={"orders": [{"id": "ORD-1001", "status": "pending", "amount": 299}]},
        max_steps=10,
        expected_actions=[
            {"name": "update", "arguments": {"table": "orders", "filters": {"id": "ORD-1001"}, "updates": {"status": "cancelled"}}},
        ],
        expected_state={"orders": [{"id": "ORD-1001", "status": "cancelled", "amount": 299}]},
    )

    script = [
        Message(role=Role.AGENT, tool_calls=[
            ToolCall(name="query", arguments={"table": "orders", "filters": {"id": "ORD-1001"}}, id="c1"),
        ]),
        Message(role=Role.AGENT, tool_calls=[
            ToolCall(name="update", arguments={"table": "orders", "filters": {"id": "ORD-1001"}, "updates": {"status": "cancelled"}}, id="c2"),
        ]),
        Message(role=Role.AGENT, tool_calls=[
            ToolCall(name="done", arguments={}, id="c3"),
        ]),
    ]

    agent = ScriptedAgent(script)
    env = MockDBEnvironment()
    evaluators = {"state_match": StateEvaluator()}
    result = Orchestrator(agent, env, evaluators).run(task)

    assert len(result.step_rewards) > 0, "Should have step rewards"
    # After query, state hasn't changed → partial reward
    # After update, state matches → reward should be 1.0
    assert result.step_rewards[-1] == 1.0, f"Final step reward should be 1.0, got {result.step_rewards[-1]}"
    assert result.progress_rate > 0, "Progress rate should be > 0"
    print(f"  step_rewards={result.step_rewards}, progress_rate={result.progress_rate:.2f}")


if __name__ == "__main__":
    print("\n=== Metrics Unit Tests ===\n")
    test_pass_at_k_all_pass()
    print("  test_pass_at_k_all_pass PASSED")
    test_pass_at_k_none_pass()
    print("  test_pass_at_k_none_pass PASSED")
    test_pass_at_k_partial()
    print("  test_pass_at_k_partial PASSED")
    test_pass_power_k_all_pass()
    print("  test_pass_power_k_all_pass PASSED")
    test_pass_power_k_partial()
    print("  test_pass_power_k_partial PASSED")
    test_pass_power_k_none_pass()
    print("  test_pass_power_k_none_pass PASSED")
    test_k_greater_than_n()
    print("  test_k_greater_than_n PASSED")
    test_edge_zero_trials()
    print("  test_edge_zero_trials PASSED")
    test_trial_stats()
    print("  test_trial_stats PASSED")
    test_saturation_warning()
    print("  test_saturation_warning PASSED")
    test_calibration_perfect_agreement()
    print("  test_calibration_perfect_agreement PASSED")
    test_calibration_poor_agreement()
    print("  test_calibration_poor_agreement PASSED")
    test_calibration_mixed()
    print("  test_calibration_mixed PASSED")
    test_calibration_empty()
    print("  test_calibration_empty PASSED")
    test_progress_rate_tracking()
    print("  test_progress_rate_tracking PASSED")
    print("\n=== All metrics tests passed ===\n")
