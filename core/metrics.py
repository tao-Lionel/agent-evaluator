"""Consistency metrics: Pass@k and Pass^k, saturation warning, trial stats.

Pass@k = 1 - C(n-c, k) / C(n, k)  -- probability of at least 1 success in k tries
Pass^k = p^k                        -- probability of all k tries succeeding

Reference: Chen et al. "Evaluating Large Language Models Trained on Code" (2021)
"""
from __future__ import annotations

from math import comb


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator.

    Args:
        n: total number of trials
        c: number of correct (passing) trials
        k: number of samples (if k > n, clamp to n)
    """
    if n <= 0:
        return 0.0
    k = min(k, n)
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def compute_pass_power_k(n: int, c: int, k: int) -> float:
    """Pass^k: probability all k attempts succeed. p^k where p = c/n."""
    if n <= 0:
        return 0.0
    p = c / n
    return p ** k


def compute_trial_stats(
    results_by_task: dict[str, list[float]],
    k_values: list[int] | None = None,
) -> dict[str, dict]:
    """Compute Pass@k and Pass^k for grouped trial results.

    Args:
        results_by_task: {task_id: [overall_score_trial1, ...]}
        k_values: list of k values to compute (default [1, 3, 5])

    Returns:
        {task_id: {"n": N, "c": C, "pass_rate": ..., "pass@1": ..., "pass^1": ..., ...}}
    """
    if k_values is None:
        k_values = [1, 3, 5]

    stats = {}
    for task_id, scores in results_by_task.items():
        n = len(scores)
        c = sum(1 for s in scores if s >= 1.0 - 1e-6)
        entry: dict = {"n": n, "c": c, "pass_rate": c / n if n > 0 else 0.0}
        for k in k_values:
            entry[f"pass@{k}"] = compute_pass_at_k(n, c, k)
            entry[f"pass^{k}"] = compute_pass_power_k(n, c, k)
        stats[task_id] = entry
    return stats


def check_saturation(scores: list[float], threshold: float = 0.85) -> str | None:
    """Return warning message if pass rate exceeds threshold."""
    if not scores:
        return None
    pass_rate = sum(1 for s in scores if s >= 1.0 - 1e-6) / len(scores)
    if pass_rate > threshold:
        return (
            f"Saturation warning: pass rate {pass_rate:.0%} exceeds {threshold:.0%}. "
            f"Consider adding harder scenarios to improve evaluation discriminability."
        )
    return None
