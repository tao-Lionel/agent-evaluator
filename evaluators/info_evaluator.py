from __future__ import annotations

import logging
import re

from core.types import Role, Message, Task
from core.base import Environment, Evaluator
from core.registry import registry

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation/extra whitespace for flexible matching."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # punctuation → space
    return re.sub(r"\s+", " ", text).strip()


def _fuzzy_contains(haystack: str, needle: str) -> bool:
    """Check if all words in needle appear in haystack in order."""
    norm_haystack = _normalize(haystack)
    norm_needle = _normalize(needle)

    # Try exact substring first
    if norm_needle in norm_haystack:
        return True

    # Fallback: check all words present (handles formatting differences)
    words = norm_needle.split()
    return all(w in norm_haystack for w in words)


@registry.evaluator("info_delivery")
class InfoDeliveryEvaluator(Evaluator):
    """Check whether the agent communicated all required information to the user.

    Inspired by tau2-bench's communicate evaluator.
    Uses fuzzy matching: normalizes punctuation/whitespace before comparison.
    """

    def evaluate(self, task: Task, trajectory: list[Message], env: Environment) -> float:
        if not task.required_info:
            return 1.0

        # Collect all agent text messages
        agent_text = " ".join(
            msg.content
            for msg in trajectory
            if msg.role == Role.AGENT and msg.content
        )

        matched = 0
        for info in task.required_info:
            if _fuzzy_contains(agent_text, info):
                matched += 1
                logger.debug("InfoDelivery: found '%s'", info)
            else:
                logger.debug("InfoDelivery: missing '%s'", info)

        score = matched / len(task.required_info)
        logger.debug("InfoDelivery: %d/%d matched", matched, len(task.required_info))
        return score
