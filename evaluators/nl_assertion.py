from __future__ import annotations

import json
import logging
import re

from openai import OpenAI

from core.types import Message, Task
from core.base import Environment, Evaluator
from core.registry import registry

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT_DB = """\
You are a database state verifier. You will receive the current database state \
and a list of assertions to check. For each assertion, determine if it PASSES \
or FAILS based on the actual data.

Respond with one line per assertion in this exact format:
[PASS] assertion text - brief reason
[FAIL] assertion text - brief reason

Nothing else."""

JUDGE_SYSTEM_PROMPT_RESPONSE = """\
You are an AI response verifier. You will receive an agent's response content \
and a list of assertions to check. For each assertion, determine if it PASSES \
or FAILS based on the actual response.

Respond with one line per assertion in this exact format:
[PASS] assertion text - brief reason
[FAIL] assertion text - brief reason

Nothing else."""

JUDGE_USER_TEMPLATE_DB = """\
## Current Database State
{context}

## Assertions to Verify
{assertions}

Check each assertion against the database state above."""

JUDGE_USER_TEMPLATE_RESPONSE = """\
## Agent Response
{context}

## Assertions to Verify
{assertions}

Check each assertion against the agent's response above."""


@registry.evaluator("nl_assertion")
class NLAssertionEvaluator(Evaluator):
    """Use an LLM to verify natural-language assertions about DB state.

    Tasks define assertions in task JSON under "nl_assertions", e.g.:
      ["orders table should have ORD-2001 with status refunded or refund_pending",
       "refunds table should contain a record for ORD-2001"]

    Score = number of passed assertions / total assertions.
    """

    def __init__(
        self,
        model: str = "glm-4-flash",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        import os
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.getenv("ZHIPU_API_KEY", ""),
            base_url=base_url or "https://open.bigmodel.cn/api/paas/v4",
        )

    def evaluate(self, task: Task, trajectory: list[Message], env: Environment) -> float:
        assertions = task.nl_assertions
        if not assertions:
            return 1.0  # no assertions to check

        # Determine context: DB state or agent response
        db = getattr(env, "db", None)
        if db is not None:
            context = json.dumps(db, ensure_ascii=False, indent=2)
            system_prompt = JUDGE_SYSTEM_PROMPT_DB
            user_template = JUDGE_USER_TEMPLATE_DB
        else:
            # No DB — use agent response content for assertion checking
            from core.types import Role
            agent_texts = [
                msg.content if isinstance(msg.content, str)
                else json.dumps(msg.content, ensure_ascii=False)
                for msg in trajectory
                if msg.role == Role.AGENT and msg.content
            ]
            if not agent_texts:
                logger.warning("NLAssertion: no agent response to check")
                return 0.0
            context = "\n".join(agent_texts)
            system_prompt = JUDGE_SYSTEM_PROMPT_RESPONSE
            user_template = JUDGE_USER_TEMPLATE_RESPONSE

        assertion_text = "\n".join(f"{i+1}. {a}" for i, a in enumerate(assertions))

        prompt = user_template.format(
            context=context,
            assertions=assertion_text,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            judge_text = response.choices[0].message.content or ""
            passed, total = _parse_results(judge_text, len(assertions))
            score = passed / total if total > 0 else 0.0
            logger.info(
                "NLAssertion for %s: %d/%d passed (%.2f)\n%s",
                task.id, passed, total, score, judge_text,
            )
            return score
        except Exception as e:
            logger.error("NLAssertion failed: %s", e)
            return 0.0


def _parse_results(text: str, expected_count: int) -> tuple[int, int]:
    """Parse [PASS]/[FAIL] lines from judge response."""
    passes = len(re.findall(r"\[PASS\]", text, re.IGNORECASE))
    fails = len(re.findall(r"\[FAIL\]", text, re.IGNORECASE))
    total = passes + fails
    # If judge didn't produce expected number of results, use expected_count
    if total != expected_count:
        total = expected_count
    return passes, total
