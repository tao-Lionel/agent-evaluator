from __future__ import annotations

import logging

from core.types import Message, Task, ToolCall
from core.base import Environment, Evaluator
from core.registry import registry
from environments.mock_db import MockDBEnvironment

logger = logging.getLogger(__name__)


@registry.evaluator("state_match")
class StateEvaluator(Evaluator):
    """Compare final DB state against the gold state produced by expected actions.

    Inspired by tau2-bench: replay gold actions on a fresh environment,
    hash both states, compare.
    """

    def evaluate(self, task: Task, trajectory: list[Message], env: Environment) -> float:
        if not task.expected_actions and not task.expected_state:
            return 1.0  # nothing to check

        # Build gold environment by replaying expected actions
        gold_env = MockDBEnvironment()
        gold_env.reset(task)
        for action in task.expected_actions:
            tc = ToolCall(
                name=action["name"],
                arguments=action.get("arguments", {}),
            )
            gold_env.step(tc)

        predicted_hash = env.get_state_hash()
        gold_hash = gold_env.get_state_hash()

        match = predicted_hash == gold_hash
        logger.debug(
            "StateEvaluator: gold=%s predicted=%s match=%s",
            gold_hash, predicted_hash, match,
        )
        return 1.0 if match else 0.0
