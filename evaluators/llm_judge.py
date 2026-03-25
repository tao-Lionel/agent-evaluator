from __future__ import annotations

import logging
import re

from openai import OpenAI

from core.types import Role, Message, Task
from core.base import Environment, Evaluator
from core.registry import registry

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator assessing the quality of a chatbot's response.
Score the response on a scale of 1-5:

5 = Perfect: Fully addresses the request, accurate, helpful, well-formatted
4 = Good: Mostly addresses the request with minor issues
3 = Acceptable: Partially addresses the request, missing some information
2 = Poor: Barely addresses the request, significant issues
1 = Fail: Does not address the request, wrong, or harmful

If you cannot reliably judge the response (e.g., missing context, ambiguous task, \
or insufficient information to make a judgment), respond with "SCORE: INSUFFICIENT_INFO" \
instead of guessing.

Respond with your reasoning first, then end with exactly "SCORE: N" where N is 1-5, \
or "SCORE: INSUFFICIENT_INFO" if you cannot judge."""

JUDGE_USER_TEMPLATE = """\
## Task Description
{description}

## User Message
{user_message}

## Required Information
The response should include: {required_info}

## Agent Response
{agent_response}

Please evaluate the agent's response."""


@registry.evaluator("llm_judge")
class LLMJudgeEvaluator(Evaluator):
    """Use an LLM to judge the quality of agent responses.

    Designed for black-box agents where state_match / action_match
    are not applicable.
    """

    def __init__(
        self,
        model: str = "glm-4-flash",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        import os
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_reason: str = ""
        self.client = OpenAI(
            api_key=api_key or os.getenv("ZHIPU_API_KEY", ""),
            base_url=base_url or "https://open.bigmodel.cn/api/paas/v4",
        )

    def evaluate(self, task: Task, trajectory: list[Message], env: Environment) -> float:
        self.last_reason = ""
        prompt = self._build_judge_prompt(task, trajectory)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            judge_text = response.choices[0].message.content or ""
            self.last_reason = judge_text
            score = self._parse_score(judge_text)
            if score < 0:
                logger.warning("LLMJudge for %s: INSUFFICIENT_INFO — returning 0.5\n%s", task.id, judge_text)
                return 0.5
            logger.info("LLMJudge for %s: score=%.2f\n%s", task.id, score, judge_text)
            return score
        except Exception as e:
            self.last_reason = f"Error: {e}"
            logger.error("LLMJudge failed: %s", e)
            return 0.0

    def _build_judge_prompt(self, task: Task, trajectory: list[Message]) -> str:
        agent_texts = [
            msg.content for msg in trajectory
            if msg.role == Role.AGENT and msg.content
        ]
        agent_response = "\n".join(agent_texts) if agent_texts else "(No response)"

        required = ", ".join(task.required_info) if task.required_info else "N/A"

        return JUDGE_USER_TEMPLATE.format(
            description=task.description,
            user_message=task.initial_message,
            required_info=required,
            agent_response=agent_response,
        )

    def _parse_score(self, text: str) -> float:
        """Parse score from judge response. Returns 0.0-1.0, or -1.0 for INSUFFICIENT_INFO."""
        if re.search(r"SCORE:\s*INSUFFICIENT_INFO", text, re.IGNORECASE):
            return -1.0
        # Primary: look for "SCORE: N"
        match = re.search(r"SCORE:\s*(\d)", text)
        if match:
            raw = int(match.group(1))
            return max(0.0, min(1.0, raw / 5.0))
        # Fallback: model may output score in other formats (e.g. truncated output)
        fallback = re.search(r"(?:score|评分)[:\s]*(\d)\s*(?:/\s*5)?", text, re.IGNORECASE)
        if fallback:
            raw = int(fallback.group(1))
            logger.warning("LLMJudge: parsed score via fallback pattern: %d/5", raw)
            return max(0.0, min(1.0, raw / 5.0))
        logger.warning("LLMJudge: could not parse score from response (len=%d), returning 0. "
                        "Tail: ...%s", len(text), text[-100:] if text else "")
        return 0.0
