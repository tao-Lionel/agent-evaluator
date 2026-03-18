from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import Task, ToolCall
from environments.passthrough import PassthroughEnvironment


def test_passthrough_basics():
    env = PassthroughEnvironment()
    task = Task(
        id="test", description="test", initial_message="hello",
        initial_state={},
    )
    obs = env.reset(task)
    assert isinstance(obs, str)

    # Tool schemas should be empty
    assert env.get_tool_schemas() == []

    # State hash should be stable
    h1 = env.get_state_hash()
    h2 = env.get_state_hash()
    assert h1 == h2

    print("  Passthrough environment: PASSED")


if __name__ == "__main__":
    test_passthrough_basics()
