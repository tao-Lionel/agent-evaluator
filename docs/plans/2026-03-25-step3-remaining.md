# Step 3 Remaining Items Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete all Step 3 remaining items: Pass@k/Pass^k metrics, Progress Rate, radar chart, efficiency metrics, negative test scenarios, LLM Judge escape hatch, saturation warning, and grader calibration.

**Architecture:** Extend `run.py` to collect multi-trial stats and compute Pass@k/Pass^k. Add efficiency tracking to `Orchestrator`. New metrics integrated into `report.py` HTML output. New evaluator features via minimal changes to existing evaluators. New negative scenarios added to `scenarios/`. All changes are additive — `core/base.py` interface unchanged.

**Tech Stack:** Python, OpenAI SDK (for LLM evaluators), HTML/CSS/JS (for radar chart in report)

---

### Task 1: Pass@k / Pass^k Consistency Metrics

**Files:**
- Create: `core/metrics.py`
- Modify: `run.py:181-243`
- Modify: `report.py:56-143`
- Test: `tests/test_metrics.py`

**Step 1: Write the failing test**

```python
# tests/test_metrics.py
"""Unit tests for Pass@k / Pass^k consistency metrics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.metrics import compute_pass_at_k, compute_pass_power_k


def test_pass_at_k_all_pass():
    """All trials pass → pass@k = 1.0 for any k."""
    assert compute_pass_at_k(n=5, c=5, k=1) == 1.0
    assert compute_pass_at_k(n=5, c=5, k=3) == 1.0


def test_pass_at_k_none_pass():
    """No trials pass → pass@k = 0.0 for any k."""
    assert compute_pass_at_k(n=5, c=0, k=1) == 0.0
    assert compute_pass_at_k(n=5, c=0, k=3) == 0.0


def test_pass_at_k_partial():
    """3/5 pass → pass@1 ≈ 0.6, pass@3 ≈ 0.9."""
    p1 = compute_pass_at_k(n=5, c=3, k=1)
    p3 = compute_pass_at_k(n=5, c=3, k=3)
    assert abs(p1 - 0.6) < 0.01
    assert p3 > p1  # pass@k increases with k


def test_pass_power_k_all_pass():
    """All trials pass → pass^k = 1.0."""
    assert compute_pass_power_k(n=5, c=5, k=3) == 1.0


def test_pass_power_k_partial():
    """3/5 pass → pass^k = 0.6^3 ≈ 0.216."""
    pk = compute_pass_power_k(n=5, c=3, k=3)
    assert abs(pk - 0.6**3) < 0.01


def test_pass_power_k_none_pass():
    assert compute_pass_power_k(n=5, c=0, k=1) == 0.0


def test_k_greater_than_n():
    """k > n should clamp to n."""
    p = compute_pass_at_k(n=3, c=2, k=5)
    assert 0 <= p <= 1.0


if __name__ == "__main__":
    test_pass_at_k_all_pass()
    test_pass_at_k_none_pass()
    test_pass_at_k_partial()
    test_pass_power_k_all_pass()
    test_pass_power_k_partial()
    test_pass_power_k_none_pass()
    test_k_greater_than_n()
    print("All metrics tests passed!")
```

**Step 2: Run test to verify it fails**

Run: `python tests/test_metrics.py`
Expected: FAIL with "No module named 'core.metrics'"

**Step 3: Write minimal implementation**

```python
# core/metrics.py
"""Consistency metrics: Pass@k and Pass^k.

Pass@k = 1 - C(n-c, k) / C(n, k)  — probability of at least 1 success in k tries
Pass^k = p^k                        — probability of all k tries succeeding

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
    # 1 - C(n-c, k) / C(n, k)
    return 1.0 - comb(n - c, k) / comb(n, k)


def compute_pass_power_k(n: int, c: int, k: int) -> float:
    """Pass^k: probability all k attempts succeed. p^k where p = c/n.

    Args:
        n: total number of trials
        c: number of correct trials
        k: exponent (number of consecutive successes required)
    """
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
        {task_id: {"n": N, "c": C, "pass@1": ..., "pass^1": ..., ...}}
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
```

**Step 4: Run test to verify it passes**

Run: `python tests/test_metrics.py`
Expected: PASS

**Step 5: Integrate into run.py — add trial stats output when num_trials > 1**

In `run.py`, after computing `all_results`, group by task_id, compute stats, print and save.

**Step 6: Integrate into report.py — show Pass@k / Pass^k table when available**

**Step 7: Commit**

```bash
git add core/metrics.py tests/test_metrics.py run.py report.py
git commit -m "feat: add Pass@k / Pass^k consistency metrics"
```

---

### Task 2: Efficiency Metrics (Steps, Tokens, Response Time)

**Files:**
- Modify: `core/types.py:81-118` (EvalResult add timing/token fields)
- Modify: `core/orchestrator.py:40-140` (track per-step timing)
- Modify: `run.py` (display efficiency stats)
- Modify: `report.py` (show efficiency in report)
- Test: `tests/test_metrics.py` (add efficiency tests)

**Step 1: Add timing fields to EvalResult**

Add `elapsed_seconds`, `total_tokens` (optional) fields to `EvalResult`.

**Step 2: Track per-step timing in Orchestrator**

Record start/end time for each agent.act() call.

**Step 3: Add efficiency display in run.py and report.py**

**Step 4: Commit**

```bash
git add core/types.py core/orchestrator.py run.py report.py tests/test_metrics.py
git commit -m "feat: add efficiency metrics (time, steps per task)"
```

---

### Task 3: LLM Judge Escape Hatch (INSUFFICIENT_INFO)

**Files:**
- Modify: `evaluators/llm_judge.py:14-24,100-105`
- Test: `tests/test_llm_judge.py` (add escape hatch test)

**Step 1: Update JUDGE_SYSTEM_PROMPT to allow INSUFFICIENT_INFO**

Add instruction: "If you cannot reliably judge the response, respond with 'SCORE: INSUFFICIENT_INFO'."

**Step 2: Update _parse_score to handle INSUFFICIENT_INFO → return None, fallback to 0.5**

**Step 3: Add test for INSUFFICIENT_INFO parsing**

**Step 4: Commit**

```bash
git add evaluators/llm_judge.py tests/test_llm_judge.py
git commit -m "feat: add LLM Judge escape hatch for INSUFFICIENT_INFO"
```

---

### Task 4: Saturation Warning

**Files:**
- Modify: `run.py` (add saturation check after results)
- Modify: `report.py` (add saturation warning banner)
- Test: `tests/test_metrics.py` (add saturation check test)

**Step 1: Add `check_saturation` function to core/metrics.py**

```python
def check_saturation(results: list, threshold: float = 0.85) -> str | None:
    """Return warning message if pass rate exceeds threshold."""
    if not results:
        return None
    pass_rate = sum(1 for r in results if r >= 1.0 - 1e-6) / len(results)
    if pass_rate > threshold:
        return f"Saturation warning: pass rate {pass_rate:.0%} > {threshold:.0%}. Consider adding harder scenarios."
    return None
```

**Step 2: Integrate into run.py print_summary and report.py**

**Step 3: Commit**

```bash
git add core/metrics.py run.py report.py tests/test_metrics.py
git commit -m "feat: add saturation warning when pass rate > 85%"
```

---

### Task 5: Progress Rate (per-step reward tracking)

**Files:**
- Modify: `core/orchestrator.py` (track intermediate rewards after each tool execution)
- Modify: `core/types.py` (add `step_rewards` to EvalResult)
- Modify: `report.py` (progress rate visualization)
- Test: `tests/test_metrics.py`

**Step 1: Track step rewards in Orchestrator**

After each env.step(), evaluate state_match on intermediate state and record it.

**Step 2: Compute progress_rate = area under step-reward curve / max possible area**

**Step 3: Add to report as a mini sparkline or progress bar**

**Step 4: Commit**

```bash
git add core/orchestrator.py core/types.py report.py tests/test_metrics.py
git commit -m "feat: add Progress Rate metric with per-step reward tracking"
```

---

### Task 6: Capability Radar Chart

**Files:**
- Modify: `report.py` (add SVG radar chart)
- Test: manual visual verification

**Step 1: Map evaluator scores to capability dimensions**

Default mapping:
- state_match → Task Completion
- action_match → Tool Usage
- info_delivery → Communication
- llm_judge → Response Quality
- nl_assertion → Reasoning
- (efficiency) → Efficiency

**Step 2: Add SVG radar chart generator in report.py**

Pure SVG, no external JS dependency. Embedded in the HTML report.

**Step 3: Commit**

```bash
git add report.py
git commit -m "feat: add capability radar chart to HTML report"
```

---

### Task 7: Negative Test Scenarios

**Files:**
- Create: `scenarios/negative_tasks.json`
- Create: `config_negative.yaml`
- Test: `tests/test_negative.py`

**Step 1: Create negative scenarios**

5 scenarios covering:
1. Refuse unauthorized access (user asks to access another user's data)
2. Ask for clarification when info is insufficient
3. Refuse unreasonable operation (delete all records)
4. Handle non-existent resource gracefully
5. Refuse to bypass business rules

**Step 2: Add config_negative.yaml**

Uses nl_assertion evaluator with assertions that check Agent refused correctly.

**Step 3: Add basic test**

**Step 4: Commit**

```bash
git add scenarios/negative_tasks.json config_negative.yaml tests/test_negative.py
git commit -m "feat: add negative test scenarios for adversarial evaluation"
```

---

### Task 8: Grader Calibration Framework

**Files:**
- Create: `core/calibration.py`
- Test: `tests/test_metrics.py` (add calibration test)

**Step 1: Create calibration module**

Simple framework: given a list of (llm_score, human_score) pairs, compute agreement rate and divergence stats.

**Step 2: Add calibration check function**

```python
def compute_calibration(pairs: list[tuple[float, float]], threshold: float = 0.20) -> dict:
    """Compute LLM vs human agreement stats."""
```

**Step 3: Commit**

```bash
git add core/calibration.py tests/test_metrics.py
git commit -m "feat: add grader calibration framework"
```

---

## Execution Order

Tasks 1-4 are independent and can be parallelized. Task 5 depends on Task 2 (timing infra). Task 6 depends on Tasks 1-5 (needs all metrics). Tasks 7-8 are independent.

Recommended: 1 → 2 → 3 → 4 → 5 → 7 → 8 → 6 (radar chart last, uses all data)
