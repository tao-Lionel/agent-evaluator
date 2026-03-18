from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import Role, Message, Task
from environments.passthrough import PassthroughEnvironment
from evaluators.llm_judge import LLMJudgeEvaluator


def test_scoring_logic():
    """Test the score extraction logic without calling LLM."""
    evaluator = LLMJudgeEvaluator.__new__(LLMJudgeEvaluator)

    assert evaluator._parse_score("SCORE: 5") == 1.0
    assert evaluator._parse_score("SCORE: 4") == 0.8
    assert evaluator._parse_score("SCORE: 3") == 0.6
    assert evaluator._parse_score("SCORE: 1") == 0.2
    assert evaluator._parse_score("some random text") == 0.0
    assert evaluator._parse_score("The answer is great. SCORE: 5") == 1.0

    print("  LLMJudge score parsing: PASSED")


def test_prompt_building():
    """Test that the judge prompt is correctly built."""
    evaluator = LLMJudgeEvaluator.__new__(LLMJudgeEvaluator)

    task = Task(
        id="t1", description="Answer shipping question",
        initial_message="When will my order arrive?",
        initial_state={},
        required_info=["3-5 business days"],
    )
    trajectory = [
        Message(role=Role.USER, content="When will my order arrive?"),
        Message(role=Role.AGENT, content="Your order will arrive in 3-5 business days."),
    ]

    prompt = evaluator._build_judge_prompt(task, trajectory)
    assert "When will my order arrive?" in prompt
    assert "3-5 business days" in prompt
    assert "Please evaluate" in prompt

    print("  LLMJudge prompt building: PASSED")


if __name__ == "__main__":
    test_scoring_logic()
    test_prompt_building()
