from __future__ import annotations

import logging

from core.types import Role, Message, Task
from core.base import Environment, Evaluator
from core.registry import registry

logger = logging.getLogger(__name__)


@registry.evaluator("info_delivery")
class InfoDeliveryEvaluator(Evaluator):
    """Check whether the agent communicated all required information to the user.

    Inspired by tau2-bench's communicate evaluator.
    """

    def evaluate(self, task: Task, trajectory: list[Message], env: Environment) -> float:
        if not task.required_info:
            return 1.0

        # Collect all agent text messages
        agent_text = " ".join(
            msg.content
            for msg in trajectory
            if msg.role == Role.AGENT and msg.content
        ).lower()

        matched = 0
        for info in task.required_info:
            normalized = info.lower().replace(",", "").strip()
            if normalized in agent_text:
                matched += 1
                logger.debug("InfoDelivery: found '%s'", info)
            else:
                logger.debug("InfoDelivery: missing '%s'", info)

        score = matched / len(task.required_info)
        logger.debug("InfoDelivery: %d/%d matched", matched, len(task.required_info))
        return score
