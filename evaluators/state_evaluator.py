from __future__ import annotations

import logging
from typing import Any

from core.types import Message, Task, ToolCall
from core.base import Environment, Evaluator
from core.registry import registry
from environments.mock_db import MockDBEnvironment

logger = logging.getLogger(__name__)


@registry.evaluator("state_match")
class StateEvaluator(Evaluator):
    """Compare final DB state against expected state.

    Two-phase evaluation:
    1. Fast path: MD5 hash match against gold state from replayed actions → 1.0
    2. Fallback: subset match against task.expected_state → 0.0~1.0
       (each expected field that exists in actual state counts as a match)
    """

    def evaluate(self, task: Task, trajectory: list[Message], env: Environment) -> float:
        if not task.expected_actions and not task.expected_state:
            return 1.0

        # Phase 1: exact hash match via replayed actions
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

        if predicted_hash == gold_hash:
            logger.debug("StateEvaluator: exact hash match")
            return 1.0

        # Phase 2: subset match against expected_state
        if not task.expected_state:
            logger.debug("StateEvaluator: hash mismatch, no expected_state for fallback")
            return 0.0

        actual_db = getattr(env, "db", None)
        if actual_db is None:
            logger.debug("StateEvaluator: env has no .db attribute, cannot subset match")
            return 0.0

        score = _subset_match_score(task.expected_state, actual_db)
        logger.debug(
            "StateEvaluator: hash mismatch (gold=%s predicted=%s), subset score=%.2f",
            gold_hash, predicted_hash, score,
        )
        return score


def _subset_match_score(expected: dict[str, Any], actual: dict[str, Any]) -> float:
    """Check if expected state is a subset of actual state.

    For each table, each expected row is matched against the best actual row.
    Score = total matched fields / total expected fields.
    """
    total_fields = 0
    matched_fields = 0

    for table_name, expected_rows in expected.items():
        actual_rows = actual.get(table_name)
        if actual_rows is None:
            # Table missing entirely — count all fields as mismatched
            for row in expected_rows:
                total_fields += len(row)
            continue

        for expected_row in expected_rows:
            total_fields += len(expected_row)
            best = _best_row_match(expected_row, actual_rows)
            matched_fields += best

    if total_fields == 0:
        return 1.0
    return matched_fields / total_fields


def _best_row_match(expected_row: dict, actual_rows: list[dict]) -> int:
    """Find the actual row with the most matching fields for expected_row."""
    best = 0
    for actual_row in actual_rows:
        count = sum(
            1 for k, v in expected_row.items()
            if k in actual_row and actual_row[k] == v
        )
        best = max(best, count)
    return best
