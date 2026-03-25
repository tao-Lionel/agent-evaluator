from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import Role, Message, Task
from environments.passthrough import PassthroughEnvironment
from evaluators.llm_judge import LLMJudgeEvaluator
from evaluators.nl_assertion import NLAssertionEvaluator, _parse_results


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


def test_insufficient_info_escape_hatch():
    """Test INSUFFICIENT_INFO escape hatch returns -1.0 sentinel."""
    evaluator = LLMJudgeEvaluator.__new__(LLMJudgeEvaluator)

    assert evaluator._parse_score("SCORE: INSUFFICIENT_INFO") == -1.0
    assert evaluator._parse_score("Cannot judge this. SCORE: INSUFFICIENT_INFO") == -1.0
    assert evaluator._parse_score("score: insufficient_info") == -1.0
    # Regular scores should still work
    assert evaluator._parse_score("SCORE: 3") == 0.6

    print("  LLMJudge INSUFFICIENT_INFO escape hatch: PASSED")


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


def test_nl_assertion_parse_results():
    """Test parsing [PASS]/[FAIL] lines from judge response."""
    text = """[PASS] orders table has ORD-2001 with status refunded - status is refunded
[PASS] refunds table contains record for ORD-2001 - record exists
[FAIL] refund amount should be 200 - actual amount is 129"""

    passed, total = _parse_results(text, 3)
    assert passed == 2, f"Expected 2 passes, got {passed}"
    assert total == 3, f"Expected 3 total, got {total}"
    print("  NLAssertion parse results: PASSED")


def test_nl_assertion_no_assertions():
    """Task with no nl_assertions should return 1.0."""
    evaluator = NLAssertionEvaluator.__new__(NLAssertionEvaluator)

    task = Task(
        id="t1", description="test", initial_message="hi",
        initial_state={}, nl_assertions=[],
    )
    score = evaluator.evaluate(task, [], PassthroughEnvironment())
    assert score == 1.0, f"Expected 1.0 for no assertions, got {score}"
    print("  NLAssertion no assertions: PASSED")


def test_nl_assertion_parse_edge_cases():
    """Test parse_results with edge cases."""
    # All pass
    passed, total = _parse_results("[PASS] a\n[PASS] b", 2)
    assert passed == 2 and total == 2

    # Empty response — judge produced nothing
    passed, total = _parse_results("", 3)
    assert passed == 0 and total == 3

    # Case insensitive
    passed, total = _parse_results("[pass] a\n[FAIL] b", 2)
    assert passed == 1 and total == 2

    print("  NLAssertion edge cases: PASSED")


if __name__ == "__main__":
    test_scoring_logic()
    test_insufficient_info_escape_hatch()
    test_prompt_building()
    test_nl_assertion_parse_results()
    test_nl_assertion_no_assertions()
    test_nl_assertion_parse_edge_cases()
