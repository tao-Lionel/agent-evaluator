# Multi-Turn Conversation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add UserSimulator support so evaluation can run multi-turn Agent-User-Environment conversations, with both scripted and LLM-based user simulators.

**Architecture:** Add a `UserSimulator` abstract interface alongside the existing three pluggable components. The Orchestrator gains an optional `user` parameter; when present, Agent text replies route to the UserSimulator, which either continues the conversation or signals completion. Two implementations: `ScriptedUserSimulator` (keyword/default branch matching from task JSON) and `LLMUserSimulator` (LLM role-plays a user persona). Registry gains a `user` type. Fully backward-compatible: no `user` config means existing behavior unchanged.

**Tech Stack:** Python, OpenAI SDK (for LLM user), existing registry/orchestrator patterns

---

### Task 1: Add UserSimulator abstract class and type changes

**Files:**
- Modify: `core/base.py:1-55`
- Modify: `core/types.py:71-75` (TerminationReason), `core/types.py:46-56` (Task)

**Step 1: Write test for new types**

```python
# tests/test_multi_turn.py
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


if __name__ == "__main__":
    print("\n=== Multi-Turn Tests ===\n")
    test_user_stop_termination()
    test_task_user_scenario()
    print("\n=== All passed ===\n")
```

**Step 2: Run test to verify it fails**

Run: `python tests/test_multi_turn.py`
Expected: FAIL — `AttributeError: 'TerminationReason' has no attribute 'USER_STOP'`

**Step 3: Implement type changes**

In `core/types.py`, add to `TerminationReason`:
```python
class TerminationReason(Enum):
    SUCCESS = "success"
    MAX_STEPS = "max_steps"
    AGENT_ERROR = "agent_error"
    ENV_ERROR = "env_error"
    USER_STOP = "user_stop"
```

In `core/types.py`, add to `Task` dataclass (after `single_turn`):
```python
    user_scenario: dict[str, Any] | None = None
```

In `core/base.py`, add `UserSimulator` abstract class after `Evaluator`:
```python
class UserSimulator(ABC):
    """Simulated user for multi-turn conversations."""

    @abstractmethod
    def reset(self, task: Task) -> None:
        """Reset state for a new task."""
        ...

    @abstractmethod
    def respond(self, task: Task, trajectory: list[Message]) -> Message | None:
        """Generate user reply based on conversation so far.

        Returns a Message with role=USER, or None if the user is satisfied
        and the conversation should end.
        """
        ...
```

**Step 4: Run test to verify it passes**

Run: `python tests/test_multi_turn.py`
Expected: PASS

**Step 5: Run existing tests to verify no breakage**

Run: `python tests/test_core.py`
Expected: All PASS

**Step 6: Commit**

```bash
git add core/types.py core/base.py tests/test_multi_turn.py
git commit -m "feat: add UserSimulator abstract class, USER_STOP termination, Task.user_scenario"
```

---

### Task 2: Add user registration to Registry

**Files:**
- Modify: `core/registry.py:1-59`

**Step 1: Write test**

Append to `tests/test_multi_turn.py`:
```python
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
```

Add `test_user_registry()` to the `__main__` block.

**Step 2: Run test to verify it fails**

Run: `python tests/test_multi_turn.py`
Expected: FAIL — `AttributeError: 'Registry' object has no attribute 'user'`

**Step 3: Implement registry changes**

In `core/registry.py`, add `_users` dict to `__init__`:
```python
self._users: dict[str, type] = {}
```

Add `user` decorator after `evaluator`:
```python
def user(self, name: str):
    def wrap(cls):
        self._users[name] = cls
        return cls
    return wrap
```

Add `get_user` lookup after `get_evaluator`:
```python
def get_user(self, name: str) -> type:
    if name not in self._users:
        raise KeyError(f"User '{name}' not registered. Available: {list(self._users)}")
    return self._users[name]
```

Update `list_all` to include users:
```python
def list_all(self) -> dict[str, list[str]]:
    return {
        "adapters": list(self._adapters),
        "environments": list(self._environments),
        "evaluators": list(self._evaluators),
        "users": list(self._users),
    }
```

**Step 4: Run test to verify it passes**

Run: `python tests/test_multi_turn.py`
Expected: PASS

**Step 5: Commit**

```bash
git add core/registry.py tests/test_multi_turn.py
git commit -m "feat: add user simulator registration to registry"
```

---

### Task 3: Modify Orchestrator for three-party routing

**Files:**
- Modify: `core/orchestrator.py:22-143`

**Step 1: Write test**

Append to `tests/test_multi_turn.py`:
```python
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
```

Add both to the `__main__` block.

**Step 2: Run test to verify it fails**

Run: `python tests/test_multi_turn.py`
Expected: FAIL — `TypeError: Orchestrator.__init__() got an unexpected keyword argument 'user'`

**Step 3: Implement Orchestrator changes**

In `core/orchestrator.py`:

1. Add import for `UserSimulator` (use TYPE_CHECKING to avoid circular):
```python
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.base import UserSimulator as UserSimulatorType
```

2. Modify `__init__`:
```python
def __init__(
    self,
    agent: AgentAdapter,
    env: Environment,
    evaluators: dict[str, Evaluator],
    user: UserSimulatorType | None = None,
):
    self.agent = agent
    self.env = env
    self.evaluators = evaluators
    self.user = user
```

3. In `run()`, after `self.agent.reset()` add:
```python
if self.user:
    self.user.reset(task)
```

4. Replace the Case 1 block (text-only reply, lines 62-69):
```python
# ── Case 1: text-only reply (no tool calls) ──
if not agent_msg.tool_calls:
    if self._is_stop_signal(agent_msg) or task.single_turn:
        termination = TerminationReason.SUCCESS
        break

    # Route to user simulator if available
    if self.user:
        user_msg = self.user.respond(task, trajectory)
        if user_msg is None:
            termination = TerminationReason.USER_STOP
            break
        trajectory.append(user_msg)
    continue
```

**Step 4: Run test to verify it passes**

Run: `python tests/test_multi_turn.py`
Expected: PASS

**Step 5: Run all existing tests**

Run: `python tests/test_core.py && python tests/test_passthrough.py && python tests/test_http_bot.py && python tests/test_llm_judge.py`
Expected: All PASS

**Step 6: Commit**

```bash
git add core/orchestrator.py tests/test_multi_turn.py
git commit -m "feat: add three-party routing to Orchestrator with optional UserSimulator"
```

---

### Task 4: Implement ScriptedUserSimulator

**Files:**
- Create: `users/__init__.py`
- Create: `users/scripted_user.py`

**Step 1: Write test**

Append to `tests/test_multi_turn.py`:
```python
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
                {"if_contains": ["order", "订单号"], "reply": "ORD-1001"},
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
```

Add all three to the `__main__` block.

**Step 2: Run test to verify it fails**

Run: `python tests/test_multi_turn.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'users'`

**Step 3: Implement**

```python
# users/__init__.py
from users.scripted_user import ScriptedUserSimulator  # noqa: F401
```

```python
# users/scripted_user.py
from __future__ import annotations

import logging

from core.types import Role, Message, Task
from core.base import UserSimulator
from core.registry import registry

logger = logging.getLogger(__name__)


@registry.user("scripted")
class ScriptedUserSimulator(UserSimulator):
    """Condition-branch user simulator driven by task JSON scripts.

    Each branch in the script has:
      - "if_contains": list of keywords to match in agent's last reply
      - "default": true for fallback branch
      - "reply": string to send, or null to signal task done
    """

    def reset(self, task: Task) -> None:
        pass

    def respond(self, task: Task, trajectory: list[Message]) -> Message | None:
        if not task.user_scenario:
            return None

        script = task.user_scenario.get("script", [])
        if not script:
            return None

        agent_reply = self._get_last_agent_text(trajectory)

        # Keyword match branches (first match wins)
        for branch in script:
            if "if_contains" in branch:
                keywords = branch["if_contains"]
                if any(kw.lower() in agent_reply.lower() for kw in keywords):
                    return self._make_reply(branch["reply"])

        # Default branch
        for branch in script:
            if branch.get("default"):
                return self._make_reply(branch["reply"])

        # No match -> end conversation
        return None

    @staticmethod
    def _get_last_agent_text(trajectory: list[Message]) -> str:
        for msg in reversed(trajectory):
            if msg.role == Role.AGENT and msg.content:
                return msg.content
        return ""

    @staticmethod
    def _make_reply(reply: str | None) -> Message | None:
        if reply is None:
            return None
        return Message(role=Role.USER, content=reply)
```

**Step 4: Run test to verify it passes**

Run: `python tests/test_multi_turn.py`
Expected: PASS

**Step 5: Commit**

```bash
git add users/__init__.py users/scripted_user.py tests/test_multi_turn.py
git commit -m "feat: add ScriptedUserSimulator with keyword/default branch matching"
```

---

### Task 5: Implement LLMUserSimulator

**Files:**
- Create: `users/llm_user.py`
- Modify: `users/__init__.py`

**Step 1: Write test (mock LLM call)**

Append to `tests/test_multi_turn.py`:
```python
from users.llm_user import LLMUserSimulator


def test_llm_user_prompt_building():
    """Test prompt construction without calling LLM."""
    user = LLMUserSimulator.__new__(LLMUserSimulator)

    task = Task(
        id="t1", description="Cancel order",
        initial_message="I want to cancel",
        initial_state={},
        user_scenario={
            "persona": "an impatient customer",
            "goal": "cancel order ORD-1001",
        },
    )
    trajectory = [
        Message(role=Role.USER, content="I want to cancel"),
        Message(role=Role.AGENT, content="Which order?"),
    ]

    messages = user._build_messages(task, trajectory)
    assert messages[0]["role"] == "system"
    assert "impatient customer" in messages[0]["content"]
    assert "cancel order ORD-1001" in messages[0]["content"]
    assert "[TASK_DONE]" in messages[0]["content"]
    # Should have user and agent messages
    assert any(m["role"] == "user" for m in messages[1:])
    assert any(m["role"] == "assistant" for m in messages[1:])
    print("  LLMUser prompt building: PASSED")


def test_llm_user_parse_done():
    """Test TASK_DONE detection."""
    user = LLMUserSimulator.__new__(LLMUserSimulator)
    assert user._is_done("Thanks! [TASK_DONE]") is True
    assert user._is_done("I still need help") is False
    assert user._is_done("[TASK_DONE] ok bye") is True
    print("  LLMUser parse done: PASSED")
```

Add both to the `__main__` block.

**Step 2: Run test to verify it fails**

Run: `python tests/test_multi_turn.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'users.llm_user'`

**Step 3: Implement**

```python
# users/llm_user.py
from __future__ import annotations

import logging

from openai import OpenAI

from core.types import Role, Message, Task
from core.base import UserSimulator
from core.registry import registry

logger = logging.getLogger(__name__)

USER_SYSTEM_TEMPLATE = """\
You are role-playing as a user talking to a customer service agent.

Your character: {persona}
Your goal: {goal}

Rules:
- Act naturally like a real user. Do not reveal you are a simulator.
- If the agent has fully satisfied your goal, reply with exactly [TASK_DONE] and nothing else.
- If the agent asks irrelevant questions or stalls, insist on your goal.
- Keep replies short, 1-2 sentences."""


@registry.user("llm")
class LLMUserSimulator(UserSimulator):
    """LLM-powered user simulator that role-plays a persona."""

    def __init__(
        self,
        model: str = "glm-4-flash",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        **kwargs,
    ):
        import os
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(
            api_key=api_key or os.getenv("ZHIPU_API_KEY", ""),
            base_url=base_url or "https://open.bigmodel.cn/api/paas/v4",
        )

    def reset(self, task: Task) -> None:
        pass

    def respond(self, task: Task, trajectory: list[Message]) -> Message | None:
        if not task.user_scenario:
            return None

        messages = self._build_messages(task, trajectory)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=256,
            )
            reply = response.choices[0].message.content or ""
        except Exception as e:
            logger.error("LLMUser API error: %s", e)
            reply = "[TASK_DONE]"

        logger.info("LLMUser reply: %s", reply[:100])

        if self._is_done(reply):
            return None

        return Message(role=Role.USER, content=reply)

    def _build_messages(self, task: Task, trajectory: list[Message]) -> list[dict]:
        scenario = task.user_scenario or {}
        system_prompt = USER_SYSTEM_TEMPLATE.format(
            persona=scenario.get("persona", "a regular customer"),
            goal=scenario.get("goal", task.description),
        )

        messages = [{"role": "system", "content": system_prompt}]

        for msg in trajectory:
            if msg.role == Role.USER:
                messages.append({"role": "user", "content": msg.content or ""})
            elif msg.role == Role.AGENT and msg.content:
                # From the user-simulator's perspective, agent replies are
                # the "assistant" it is talking to, so map to assistant role
                messages.append({"role": "assistant", "content": msg.content})
            # Skip ENV/SYSTEM messages — user doesn't see tool calls

        return messages

    @staticmethod
    def _is_done(text: str) -> bool:
        return "[TASK_DONE]" in text
```

Update `users/__init__.py`:
```python
from users.scripted_user import ScriptedUserSimulator  # noqa: F401
from users.llm_user import LLMUserSimulator  # noqa: F401
```

**Step 4: Run test to verify it passes**

Run: `python tests/test_multi_turn.py`
Expected: PASS

**Step 5: Commit**

```bash
git add users/llm_user.py users/__init__.py tests/test_multi_turn.py
git commit -m "feat: add LLMUserSimulator with persona-based role-playing"
```

---

### Task 6: Wire up run.py and config

**Files:**
- Modify: `run.py:26-28` (imports), `run.py:167-168` (orchestrator construction)

**Step 1: Implement run.py changes**

Add `import users` alongside the other plugin imports (after line 28):
```python
import users  # noqa: F401
```

After the evaluator construction block (line 165) and before `orchestrator = Orchestrator(...)` (line 168), add user construction:
```python
# Build user simulator (optional)
user = None
if "user" in config:
    user_cfg = config["user"]
    UserClass = registry.get_user(user_cfg["type"])
    user_params = {k: v for k, v in user_cfg.items() if k != "type"}
    user = UserClass(**user_params)
```

Update the Orchestrator construction:
```python
orchestrator = Orchestrator(adapter, env, evaluator_map, user=user)
```

Update the print block to show user info (after `Evals` line):
```python
print(f"  User    : {config.get('user', {}).get('type', 'none')}")
```

**Step 2: Run all existing tests to verify no breakage**

Run: `python tests/test_core.py && python tests/test_passthrough.py && python tests/test_http_bot.py && python tests/test_llm_judge.py && python tests/test_multi_turn.py`
Expected: All PASS

**Step 3: Commit**

```bash
git add run.py
git commit -m "feat: wire UserSimulator into run.py with config support"
```

---

### Task 7: Create multi-turn evaluation scenarios and config

**Files:**
- Create: `scenarios/multi_turn_tasks.json`
- Create: `config_multi_turn.yaml`

**Step 1: Write multi-turn task scenarios**

```json
[
  {
    "id": "cancel-multiturn-001",
    "description": "User wants to cancel an order but only provides the order ID when asked.",
    "difficulty": "medium",
    "initial_message": "I want to cancel an order",
    "initial_state": {
      "orders": [
        {"id": "ORD-1001", "user": "Alice", "product": "Bluetooth Headphones", "status": "pending", "amount": 299}
      ]
    },
    "max_steps": 15,
    "expected_actions": [
      {"name": "query", "arguments": {"table": "orders", "filters": {"id": "ORD-1001"}}, "match_args": {"table": "orders"}},
      {"name": "update", "arguments": {"table": "orders", "filters": {"id": "ORD-1001"}, "updates": {"status": "cancelled"}}, "match_args": {"table": "orders", "updates": {"status": "cancelled"}}}
    ],
    "expected_state": {
      "orders": [
        {"id": "ORD-1001", "user": "Alice", "product": "Bluetooth Headphones", "status": "cancelled", "amount": 299}
      ]
    },
    "required_info": ["ORD-1001", "cancelled"],
    "user_scenario": {
      "persona": "A busy user who doesn't give all info upfront",
      "goal": "Cancel order ORD-1001",
      "script": [
        {"if_contains": ["order", "订单", "which", "哪"], "reply": "ORD-1001"},
        {"if_contains": ["cancelled", "已取消", "success", "成功"], "reply": null},
        {"default": true, "reply": "I just want to cancel my order, please help"}
      ]
    }
  },
  {
    "id": "refund-multiturn-002",
    "description": "User wants a refund for a damaged item. Needs to provide order ID and reason.",
    "difficulty": "hard",
    "initial_message": "I need a refund",
    "initial_state": {
      "orders": [
        {"id": "ORD-2001", "user": "Bob", "product": "Glass Vase", "status": "delivered", "amount": 129}
      ],
      "refunds": []
    },
    "max_steps": 20,
    "expected_actions": [
      {"name": "query", "arguments": {"table": "orders", "filters": {"id": "ORD-2001"}}, "match_args": {"table": "orders"}},
      {"name": "insert", "arguments": {"table": "refunds", "record": {"order_id": "ORD-2001", "amount": 129, "reason": "damaged", "status": "pending"}}, "match_args": {"table": "refunds"}},
      {"name": "update", "arguments": {"table": "orders", "filters": {"id": "ORD-2001"}, "updates": {"status": "refund_pending"}}, "match_args": {"table": "orders"}}
    ],
    "expected_state": {
      "orders": [
        {"id": "ORD-2001", "user": "Bob", "product": "Glass Vase", "status": "refund_pending", "amount": 129}
      ],
      "refunds": [
        {"order_id": "ORD-2001", "amount": 129, "reason": "damaged", "status": "pending"}
      ]
    },
    "required_info": ["refund", "ORD-2001"],
    "user_scenario": {
      "persona": "A frustrated user who received a damaged product",
      "goal": "Get a full refund for damaged order ORD-2001",
      "script": [
        {"if_contains": ["order", "订单", "which", "哪"], "reply": "ORD-2001"},
        {"if_contains": ["reason", "why", "what happened", "原因", "怎么"], "reply": "The item arrived broken, the glass vase was shattered"},
        {"if_contains": ["refund", "退款", "processed", "处理"], "reply": null},
        {"default": true, "reply": "I received a damaged item and I want my money back"}
      ]
    }
  },
  {
    "id": "address-change-multiturn-003",
    "description": "User wants to change shipping address, provides details step by step.",
    "difficulty": "medium",
    "initial_message": "I need to update my shipping address",
    "initial_state": {
      "orders": [
        {"id": "ORD-3001", "user": "Carol", "product": "Desk Lamp", "status": "pending", "amount": 89, "address": "100 Old Street"}
      ]
    },
    "max_steps": 15,
    "expected_actions": [
      {"name": "query", "arguments": {"table": "orders", "filters": {"id": "ORD-3001"}}, "match_args": {"table": "orders"}},
      {"name": "update", "arguments": {"table": "orders", "filters": {"id": "ORD-3001"}, "updates": {"address": "200 New Avenue"}}, "match_args": {"table": "orders", "updates": {"address": "200 New Avenue"}}}
    ],
    "expected_state": {
      "orders": [
        {"id": "ORD-3001", "user": "Carol", "product": "Desk Lamp", "status": "pending", "amount": 89, "address": "200 New Avenue"}
      ]
    },
    "required_info": ["200 New Avenue"],
    "user_scenario": {
      "persona": "A polite user moving to a new address",
      "goal": "Change shipping address of order ORD-3001 to 200 New Avenue",
      "script": [
        {"if_contains": ["order", "订单", "which", "哪"], "reply": "ORD-3001"},
        {"if_contains": ["address", "地址", "new", "新"], "reply": "200 New Avenue"},
        {"if_contains": ["updated", "已更新", "changed", "修改"], "reply": null},
        {"default": true, "reply": "I need to change the delivery address for my order"}
      ]
    }
  }
]
```

**Step 2: Write config file for scripted user**

```yaml
# config_multi_turn.yaml
agent:
  adapter: openai_fc
  model: ${MODEL_NAME:-glm-4-flash}
  api_key: ${ZHIPU_API_KEY}
  base_url: https://open.bigmodel.cn/api/paas/v4

environment:
  type: mock_db

user:
  type: scripted

evaluators:
  - state_match
  - action_match
  - info_delivery

scenarios: scenarios/multi_turn_tasks.json

run:
  num_trials: 1
  log_level: INFO
```

**Step 3: Commit**

```bash
git add scenarios/multi_turn_tasks.json config_multi_turn.yaml
git commit -m "feat: add multi-turn conversation scenarios and config"
```

---

### Task 8: End-to-end verification

**Step 1: Run all unit tests**

Run: `python tests/test_core.py && python tests/test_passthrough.py && python tests/test_http_bot.py && python tests/test_llm_judge.py && python tests/test_multi_turn.py`
Expected: All PASS

**Step 2: Run multi-turn evaluation (requires API key)**

Run: `python run.py config_multi_turn.yaml`
Expected: 3 tasks run with multi-turn conversation, showing:
- Agent asks clarifying questions
- User (scripted) provides info per script
- Tool calls execute between conversation turns
- Evaluation scores for each task

**Step 3: Run original config to verify backward compatibility**

Run: `python run.py config.yaml`
Expected: Original 5 tasks pass exactly as before (no user simulator involved)

**Step 4: Commit if any fixes were needed**

```bash
git add -A
git commit -m "test: verify end-to-end multi-turn conversation evaluation"
```
