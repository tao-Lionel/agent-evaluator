"""Unit tests that verify the entire pipeline WITHOUT calling any LLM API.

Uses a MockAgent that follows a scripted sequence of tool calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import Role, Message, ToolCall, Task
from core.base import AgentAdapter
from core.orchestrator import Orchestrator
from environments.mock_db import MockDBEnvironment
from evaluators.state_evaluator import StateEvaluator
from evaluators.action_evaluator import ActionEvaluator
from evaluators.info_evaluator import InfoDeliveryEvaluator


class ScriptedAgent(AgentAdapter):
    """Agent that follows a pre-defined script of actions."""

    def __init__(self, script: list[Message]):
        self._script = list(script)
        self._step = 0

    def reset(self):
        self._step = 0

    def act(self, messages: list[Message]) -> Message:
        if self._step < len(self._script):
            msg = self._script[self._step]
            self._step += 1
            return msg
        return Message(role=Role.AGENT, content="###DONE###")


def make_task() -> Task:
    return Task(
        id="test-cancel",
        description="Cancel order ORD-1001",
        initial_message="Please cancel ORD-1001",
        initial_state={
            "orders": [
                {"id": "ORD-1001", "user": "Alice", "product": "Headphones", "status": "pending", "amount": 299},
            ]
        },
        max_steps=10,
        expected_actions=[
            {"name": "query", "arguments": {"table": "orders", "filters": {"id": "ORD-1001"}}, "match_args": {"table": "orders"}},
            {"name": "update", "arguments": {"table": "orders", "filters": {"id": "ORD-1001"}, "updates": {"status": "cancelled"}}, "match_args": {"table": "orders", "updates": {"status": "cancelled"}}},
        ],
        required_info=["cancelled", "ORD-1001"],
    )


def test_perfect_agent():
    """Agent that does everything correctly should score 1.0 across all evaluators."""
    task = make_task()

    script = [
        # Step 1: query the order
        Message(role=Role.AGENT, tool_calls=[
            ToolCall(name="query", arguments={"table": "orders", "filters": {"id": "ORD-1001"}}, id="call_1"),
        ]),
        # Step 2: tell user and update
        Message(role=Role.AGENT, content="I found your order ORD-1001. Let me cancel it for you.", tool_calls=[
            ToolCall(name="update", arguments={"table": "orders", "filters": {"id": "ORD-1001"}, "updates": {"status": "cancelled"}}, id="call_2"),
        ]),
        # Step 3: confirm and finish
        Message(role=Role.AGENT, content="Your order ORD-1001 has been cancelled successfully."),
        Message(role=Role.AGENT, tool_calls=[
            ToolCall(name="done", arguments={"summary": "Cancelled order ORD-1001"}, id="call_3"),
        ]),
    ]

    agent = ScriptedAgent(script)
    env = MockDBEnvironment()
    evaluators = {
        "state_match": StateEvaluator(),
        "action_match": ActionEvaluator(),
        "info_delivery": InfoDeliveryEvaluator(),
    }

    result = Orchestrator(agent, env, evaluators).run(task)

    print(f"  Perfect agent: {result.scores} overall={result.overall_score}")
    assert result.scores["state_match"] == 1.0, f"state_match should be 1.0, got {result.scores['state_match']}"
    assert result.scores["action_match"] == 1.0, f"action_match should be 1.0, got {result.scores['action_match']}"
    assert result.scores["info_delivery"] == 1.0, f"info_delivery should be 1.0, got {result.scores['info_delivery']}"
    assert result.overall_score == 1.0
    print("  PASSED")


def test_wrong_action_agent():
    """Agent that queries but doesn't update should get partial action score and 0 state score."""
    task = make_task()

    script = [
        Message(role=Role.AGENT, tool_calls=[
            ToolCall(name="query", arguments={"table": "orders", "filters": {"id": "ORD-1001"}}, id="call_1"),
        ]),
        # Skips update, goes straight to done
        Message(role=Role.AGENT, content="Your order ORD-1001 is pending. I cancelled it."),
        Message(role=Role.AGENT, tool_calls=[
            ToolCall(name="done", arguments={}, id="call_2"),
        ]),
    ]

    agent = ScriptedAgent(script)
    env = MockDBEnvironment()
    evaluators = {
        "state_match": StateEvaluator(),
        "action_match": ActionEvaluator(),
        "info_delivery": InfoDeliveryEvaluator(),
    }

    result = Orchestrator(agent, env, evaluators).run(task)

    print(f"  Wrong agent:   {result.scores} overall={result.overall_score}")
    assert result.scores["state_match"] == 0.0, "State should NOT match (no update was done)"
    assert result.scores["action_match"] == 0.5, "Should match 1/2 expected actions"
    assert result.overall_score == 0.0, "Overall should be 0 because state_match=0"
    print("  PASSED")


def test_state_subset_match():
    """Agent that produces correct state with extra fields should get partial state score."""
    task = Task(
        id="test-refund",
        description="Refund order ORD-2001",
        initial_message="I need a refund",
        initial_state={
            "orders": [
                {"id": "ORD-2001", "user": "Bob", "product": "Glass Vase", "status": "delivered", "amount": 129}
            ],
            "refunds": [],
        },
        max_steps=10,
        expected_actions=[
            # Gold actions produce exact state — but agent will differ
            {"name": "insert", "arguments": {"table": "refunds", "record": {"order_id": "ORD-2001", "amount": 129, "reason": "damaged"}}},
            {"name": "update", "arguments": {"table": "orders", "filters": {"id": "ORD-2001"}, "updates": {"status": "refunded"}}},
        ],
        expected_state={
            "orders": [
                {"id": "ORD-2001", "status": "refunded"}
            ],
            "refunds": [
                {"order_id": "ORD-2001", "amount": 129}
            ],
        },
    )

    # Agent inserts refund with extra fields and different reason text
    script = [
        Message(role=Role.AGENT, tool_calls=[
            ToolCall(name="insert", arguments={
                "table": "refunds",
                "record": {"order_id": "ORD-2001", "amount": 129, "reason": "Item was shattered", "user": "Bob"}
            }, id="call_1"),
        ]),
        Message(role=Role.AGENT, tool_calls=[
            ToolCall(name="update", arguments={
                "table": "orders", "filters": {"id": "ORD-2001"}, "updates": {"status": "refunded"}
            }, id="call_2"),
        ]),
        Message(role=Role.AGENT, content="Refund processed for ORD-2001."),
        Message(role=Role.AGENT, tool_calls=[
            ToolCall(name="done", arguments={}, id="call_3"),
        ]),
    ]

    agent = ScriptedAgent(script)
    env = MockDBEnvironment()
    evaluator = StateEvaluator()

    result = Orchestrator(agent, env, {"state_match": evaluator}).run(task)
    score = result.scores["state_match"]

    # Hash won't match (extra fields + different reason), but subset should:
    # orders: id=ORD-2001 ✓, status=refunded ✓ → 2/2
    # refunds: order_id=ORD-2001 ✓, amount=129 ✓ → 2/2
    # Total: 4/4 = 1.0
    print(f"  Subset match:  state_match={score:.2f}")
    assert score == 1.0, f"Subset match should be 1.0, got {score}"
    print("  PASSED")


def test_state_subset_partial():
    """Agent with partially correct state should get proportional score."""
    task = Task(
        id="test-partial",
        description="Cancel order",
        initial_message="Cancel ORD-1001",
        initial_state={
            "orders": [
                {"id": "ORD-1001", "status": "pending", "amount": 299}
            ],
        },
        max_steps=10,
        expected_actions=[
            {"name": "update", "arguments": {"table": "orders", "filters": {"id": "ORD-1001"}, "updates": {"status": "cancelled"}}},
        ],
        expected_state={
            "orders": [
                {"id": "ORD-1001", "status": "cancelled", "amount": 299}
            ],
        },
    )

    # Agent does nothing — status stays "pending"
    script = [
        Message(role=Role.AGENT, content="Done."),
        Message(role=Role.AGENT, tool_calls=[
            ToolCall(name="done", arguments={}, id="call_1"),
        ]),
    ]

    agent = ScriptedAgent(script)
    env = MockDBEnvironment()
    evaluator = StateEvaluator()

    result = Orchestrator(agent, env, {"state_match": evaluator}).run(task)
    score = result.scores["state_match"]

    # id=ORD-1001 ✓, status=cancelled ✗ (still pending), amount=299 ✓ → 2/3
    expected = round(2 / 3, 2)
    print(f"  Partial match: state_match={score:.2f} (expected ~{expected})")
    assert abs(score - 2 / 3) < 0.01, f"Should be ~0.67, got {score}"
    print("  PASSED")


def test_state_match_without_expected_actions_uses_expected_state():
    """When expected_actions is empty, evaluator should still use expected_state."""
    task = Task(
        id="state-only",
        description="State-only evaluation",
        initial_message="Do not mutate state",
        initial_state={
            "orders": [
                {"id": "ORD-9001", "status": "pending", "amount": 299}
            ],
        },
        max_steps=10,
        expected_actions=[],
        expected_state={
            "orders": [
                {"id": "ORD-9001", "status": "cancelled", "amount": 299}
            ],
        },
    )

    script = [
        Message(role=Role.AGENT, content="Done."),
        Message(role=Role.AGENT, tool_calls=[
            ToolCall(name="done", arguments={}, id="call_1"),
        ]),
    ]

    agent = ScriptedAgent(script)
    env = MockDBEnvironment()
    evaluator = StateEvaluator()

    result = Orchestrator(agent, env, {"state_match": evaluator}).run(task)
    score = result.scores["state_match"]

    assert abs(score - 2 / 3) < 0.01, f"Should be ~0.67, got {score}"
    print("  PASSED")


def test_max_steps():
    """Agent that never finishes should hit MAX_STEPS."""
    task = make_task()
    task.max_steps = 3

    # Agent always says something useless
    script = [
        Message(role=Role.AGENT, content="Let me think..."),
        Message(role=Role.AGENT, content="Still thinking..."),
        Message(role=Role.AGENT, content="Almost there..."),
        Message(role=Role.AGENT, content="Just a moment..."),
    ]

    agent = ScriptedAgent(script)
    env = MockDBEnvironment()

    result = Orchestrator(agent, env, {}).run(task)

    print(f"  Timeout agent: terminated={result.terminated.value} steps={result.steps_taken}")
    assert result.terminated.value == "max_steps"
    print("  PASSED")


def test_environment_crud():
    """Direct environment operations test."""
    env = MockDBEnvironment()
    task = Task(
        id="env-test", description="", initial_message="",
        initial_state={"users": [{"name": "Alice", "age": 30}]},
    )
    env.reset(task)

    # Query
    r = env.step(ToolCall(name="query", arguments={"table": "users"}))
    assert "Alice" in r.observation

    # Insert
    r = env.step(ToolCall(name="insert", arguments={"table": "users", "record": {"name": "Bob", "age": 25}}))
    assert "Inserted" in r.observation

    # Update
    r = env.step(ToolCall(name="update", arguments={"table": "users", "filters": {"name": "Alice"}, "updates": {"age": 31}}))
    assert "Updated 1" in r.observation

    # Delete
    r = env.step(ToolCall(name="delete", arguments={"table": "users", "filters": {"name": "Bob"}}))
    assert "Deleted 1" in r.observation

    # Verify state
    r = env.step(ToolCall(name="query", arguments={"table": "users"}))
    assert "31" in r.observation
    assert "Bob" not in r.observation

    # Done
    r = env.step(ToolCall(name="done", arguments={"summary": "test complete"}))
    assert r.done is True

    print("  Environment CRUD: PASSED")


if __name__ == "__main__":
    print("\n=== Agent Evaluator Unit Tests ===\n")
    test_environment_crud()
    test_perfect_agent()
    test_wrong_action_agent()
    test_state_subset_match()
    test_state_subset_partial()
    test_max_steps()
    print("\n=== All tests passed ===\n")
