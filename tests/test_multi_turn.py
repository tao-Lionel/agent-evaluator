from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import TerminationReason, Task


def test_user_stop_termination():
    assert TerminationReason.USER_STOP.value == "user_stop"
    print("  USER_STOP termination: PASSED")


def test_task_user_scenario():
    task = Task(
        id="t1", description="test", initial_message="hi",
        initial_state={},
        user_scenario={"persona": "angry user", "goal": "get refund"},
    )
    assert task.user_scenario is not None
    assert task.user_scenario["persona"] == "angry user"

    # Without user_scenario
    task2 = Task(id="t2", description="test", initial_message="hi", initial_state={})
    assert task2.user_scenario is None
    print("  Task user_scenario: PASSED")


from core.base import UserSimulator
from core.registry import registry


def test_user_registry():
    @registry.user("test_user")
    class FakeUser(UserSimulator):
        def reset(self, task):
            pass
        def respond(self, task, trajectory):
            return None

    cls = registry.get_user("test_user")
    assert cls is FakeUser
    print("  User registry: PASSED")


from core.types import Role, Message, ToolCall
from core.orchestrator import Orchestrator
from environments.mock_db import MockDBEnvironment


class ScriptedAgent:
    """Agent that follows a pre-defined script."""
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


class FakeUserSimulator(UserSimulator):
    """User that answers the first agent question, then ends on the second reply."""
    def __init__(self):
        self._turn = 0
    def reset(self, task):
        self._turn = 0
    def respond(self, task, trajectory):
        self._turn += 1
        if self._turn == 1:
            return Message(role=Role.USER, content="ORD-1001")
        return None  # satisfied


def test_orchestrator_with_user():
    task = Task(
        id="multi-turn-test",
        description="Cancel an order after asking for order ID",
        initial_message="I want to cancel an order",
        initial_state={
            "orders": [
                {"id": "ORD-1001", "user": "Alice", "product": "Headphones", "status": "pending", "amount": 299},
            ]
        },
        max_steps=10,
        user_scenario={"persona": "test", "goal": "cancel ORD-1001"},
    )

    script = [
        # Agent asks for order ID
        Message(role=Role.AGENT, content="Sure, what is your order ID?"),
        # After user says ORD-1001, agent queries and updates
        Message(role=Role.AGENT, tool_calls=[
            ToolCall(name="query", arguments={"table": "orders", "filters": {"id": "ORD-1001"}}, id="c1"),
        ]),
        Message(role=Role.AGENT, tool_calls=[
            ToolCall(name="update", arguments={"table": "orders", "filters": {"id": "ORD-1001"}, "updates": {"status": "cancelled"}}, id="c2"),
        ]),
        # Agent confirms
        Message(role=Role.AGENT, content="Your order ORD-1001 has been cancelled."),
    ]

    agent = ScriptedAgent(script)
    env = MockDBEnvironment()
    user = FakeUserSimulator()

    result = Orchestrator(agent, env, {}, user=user).run(task)

    assert result.terminated == TerminationReason.USER_STOP, f"Expected USER_STOP, got {result.terminated}"

    # Verify trajectory contains user reply
    user_msgs = [m for m in result.trajectory if m.role == Role.USER]
    assert len(user_msgs) == 2  # initial_message + "ORD-1001"
    assert user_msgs[1].content == "ORD-1001"

    print("  Orchestrator with user: PASSED")


def test_orchestrator_without_user_unchanged():
    """Existing behavior: no user simulator, single-turn stop signal."""
    task = Task(
        id="no-user-test",
        description="Simple test",
        initial_message="Cancel ORD-1001",
        initial_state={"orders": [{"id": "ORD-1001", "status": "pending"}]},
        max_steps=5,
    )

    script = [
        Message(role=Role.AGENT, content="Done. ###DONE###"),
    ]

    agent = ScriptedAgent(script)
    env = MockDBEnvironment()

    result = Orchestrator(agent, env, {}).run(task)

    assert result.terminated == TerminationReason.SUCCESS
    print("  Orchestrator without user (backward compat): PASSED")


from users.scripted_user import ScriptedUserSimulator


def test_scripted_user_keyword_match():
    user = ScriptedUserSimulator()
    task = Task(
        id="t1", description="test", initial_message="cancel order",
        initial_state={},
        user_scenario={
            "persona": "test user",
            "goal": "cancel order",
            "script": [
                {"if_contains": ["which", "订单号"], "reply": "ORD-1001"},
                {"if_contains": ["cancelled", "已取消"], "reply": None},
                {"default": True, "reply": "just help me"},
            ],
        },
    )
    user.reset(task)

    # Agent asks for order ID -> matches "order"
    trajectory = [
        Message(role=Role.USER, content="cancel order"),
        Message(role=Role.AGENT, content="Which order would you like to cancel?"),
    ]
    result = user.respond(task, trajectory)
    assert result is not None
    assert result.content == "ORD-1001"
    assert result.role == Role.USER

    # Agent confirms cancellation -> matches "cancelled", reply is None -> done
    trajectory.append(result)
    trajectory.append(Message(role=Role.AGENT, content="Order ORD-1001 has been cancelled."))
    result = user.respond(task, trajectory)
    assert result is None  # user satisfied

    print("  ScriptedUser keyword match: PASSED")


def test_scripted_user_default_branch():
    user = ScriptedUserSimulator()
    task = Task(
        id="t2", description="test", initial_message="hi",
        initial_state={},
        user_scenario={
            "persona": "test",
            "goal": "test",
            "script": [
                {"if_contains": ["xyz_no_match"], "reply": "matched"},
                {"default": True, "reply": "I need help"},
            ],
        },
    )
    user.reset(task)

    trajectory = [
        Message(role=Role.USER, content="hi"),
        Message(role=Role.AGENT, content="Hello! How can I help?"),
    ]
    result = user.respond(task, trajectory)
    assert result is not None
    assert result.content == "I need help"
    print("  ScriptedUser default branch: PASSED")


def test_scripted_user_no_script_ends():
    """Task without script -> immediate end."""
    user = ScriptedUserSimulator()
    task = Task(
        id="t3", description="test", initial_message="hi",
        initial_state={},
        user_scenario={"persona": "test", "goal": "test"},
    )
    user.reset(task)

    trajectory = [
        Message(role=Role.USER, content="hi"),
        Message(role=Role.AGENT, content="Hello!"),
    ]
    result = user.respond(task, trajectory)
    assert result is None
    print("  ScriptedUser no script ends: PASSED")


if __name__ == "__main__":
    print("\n=== Multi-Turn Tests ===\n")
    test_user_stop_termination()
    test_task_user_scenario()
    test_user_registry()
    test_orchestrator_with_user()
    test_orchestrator_without_user_unchanged()
    test_scripted_user_keyword_match()
    test_scripted_user_default_branch()
    test_scripted_user_no_script_ends()
    print("\n=== All passed ===\n")
