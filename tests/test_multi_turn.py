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
