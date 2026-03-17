from __future__ import annotations

import logging

from core.types import Role, Message, Task
from core.base import Environment, Evaluator
from core.registry import registry

logger = logging.getLogger(__name__)


@registry.evaluator("action_match")
class ActionEvaluator(Evaluator):
    """Check whether all expected actions appear in the trajectory.

    Supports partial argument matching via 'match_args'.
    """

    def evaluate(self, task: Task, trajectory: list[Message], env: Environment) -> float:
        if not task.expected_actions:
            return 1.0

        # Collect all actual tool calls from trajectory
        actual_calls: list[dict] = []
        for msg in trajectory:
            if msg.role == Role.AGENT and msg.tool_calls:
                for tc in msg.tool_calls:
                    actual_calls.append({"name": tc.name, "arguments": tc.arguments})

        matched = 0
        for expected in task.expected_actions:
            exp_name = expected["name"]
            match_args = expected.get("match_args", {})

            for actual in actual_calls:
                if actual["name"] != exp_name:
                    continue
                if all(actual["arguments"].get(k) == v for k, v in match_args.items()):
                    matched += 1
                    break

        score = matched / len(task.expected_actions)
        logger.debug("ActionEvaluator: %d/%d matched", matched, len(task.expected_actions))
        return score
