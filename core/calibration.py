"""Grader calibration: compare LLM evaluator scores against human labels.

Usage:
    pairs = [(llm_score, human_score), ...]
    report = compute_calibration(pairs)
    # report = {"agreement_rate": 0.85, "mean_divergence": 0.12, ...}
"""
from __future__ import annotations


def compute_calibration(
    pairs: list[tuple[float, float]],
    threshold: float = 0.20,
) -> dict:
    """Compute LLM vs human agreement statistics.

    Args:
        pairs: list of (llm_score, human_score), both in [0, 1]
        threshold: max absolute difference to count as "agreement"

    Returns:
        {
            "total": N,
            "agreement_rate": fraction where |llm - human| <= threshold,
            "mean_divergence": average |llm - human|,
            "max_divergence": max |llm - human|,
            "llm_bias": mean(llm - human), positive = LLM scores higher,
            "needs_recalibration": True if agreement_rate < 0.80,
        }
    """
    if not pairs:
        return {
            "total": 0,
            "agreement_rate": 0.0,
            "mean_divergence": 0.0,
            "max_divergence": 0.0,
            "llm_bias": 0.0,
            "needs_recalibration": False,
        }

    divergences = [abs(llm - human) for llm, human in pairs]
    biases = [llm - human for llm, human in pairs]
    agreements = sum(1 for d in divergences if d <= threshold)

    n = len(pairs)
    agreement_rate = agreements / n

    return {
        "total": n,
        "agreement_rate": round(agreement_rate, 4),
        "mean_divergence": round(sum(divergences) / n, 4),
        "max_divergence": round(max(divergences), 4),
        "llm_bias": round(sum(biases) / n, 4),
        "needs_recalibration": agreement_rate < 0.80,
    }
