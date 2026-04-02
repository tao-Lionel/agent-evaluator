# Concurrent Evaluators + LLM Retry + Performance Profiling

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make evaluation faster (concurrent evaluators), more reliable (LLM retry with backoff), and observable (per-phase timing profiler).

**Architecture:** Three independent modules — `core/retry.py` (retry utility), `core/profiler.py` (timing profiler), and modifications to `core/orchestrator.py` (concurrent eval + profiling integration). Retry is consumed by all LLM evaluators. Profiler is consumed by orchestrator and surfaced in `EvalResult`.

**Tech Stack:** Python stdlib (`concurrent.futures.ThreadPoolExecutor`, `time`, `random`), OpenAI SDK error types for retry classification.

---

### Task 1: LLM Retry Utility (`core/retry.py`)

**Files:**
- Create: `core/retry.py`
- Test: `tests/test_retry.py`

**Step 1: Write the failing test**

```python
# tests/test_retry.py
"""Tests for LLM call retry utility."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.retry import with_retry, RetryableError, PermanentError


class FakeRetryableError(RetryableError):
    pass

class FakePermanentError(PermanentError):
    pass


def test_succeeds_first_try():
    calls = []
    def fn():
        calls.append(1)
        return "ok"
    result = with_retry(fn, max_retries=3, base_delay=0.01)
    assert result == "ok"
    assert len(calls) == 1
    print("  test_succeeds_first_try PASSED")


def test_retries_on_retryable_error():
    calls = []
    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise FakeRetryableError("transient")
        return "recovered"
    result = with_retry(fn, max_retries=5, base_delay=0.01)
    assert result == "recovered"
    assert len(calls) == 3
    print("  test_retries_on_retryable_error PASSED")


def test_raises_after_max_retries():
    calls = []
    def fn():
        calls.append(1)
        raise FakeRetryableError("always fails")
    try:
        with_retry(fn, max_retries=3, base_delay=0.01)
        assert False, "Should have raised"
    except FakeRetryableError:
        pass
    assert len(calls) == 4  # 1 initial + 3 retries
    print("  test_raises_after_max_retries PASSED")


def test_no_retry_on_permanent_error():
    calls = []
    def fn():
        calls.append(1)
        raise FakePermanentError("bad auth")
    try:
        with_retry(fn, max_retries=3, base_delay=0.01)
        assert False, "Should have raised"
    except FakePermanentError:
        pass
    assert len(calls) == 1
    print("  test_no_retry_on_permanent_error PASSED")


def test_classify_openai_errors():
    """Verify OpenAI error classification without importing openai."""
    from core.retry import classify_error
    # Simulate by class name matching
    assert classify_error(ConnectionError("net")) == "retryable"
    assert classify_error(TimeoutError("timeout")) == "retryable"
    assert classify_error(ValueError("bad input")) == "permanent"
    print("  test_classify_openai_errors PASSED")


if __name__ == "__main__":
    print("\n=== Retry Utility Tests ===\n")
    test_succeeds_first_try()
    test_retries_on_retryable_error()
    test_raises_after_max_retries()
    test_no_retry_on_permanent_error()
    test_classify_openai_errors()
    print("\n=== All retry tests passed ===\n")
```

**Step 2: Run test to verify it fails**

Run: `python tests/test_retry.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.retry'`

**Step 3: Write the implementation**

```python
# core/retry.py
"""Retry utility with exponential backoff for LLM API calls.

Classifies errors as retryable (network, rate limit, timeout, server errors)
vs permanent (auth, bad request). Uses exponential backoff with jitter.
"""
from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryableError(Exception):
    """Base class for errors that should trigger a retry."""
    pass


class PermanentError(Exception):
    """Base class for errors that should NOT be retried."""
    pass


# OpenAI SDK error class names that are retryable
_RETRYABLE_NAMES = frozenset({
    "APIConnectionError",
    "RateLimitError",
    "APITimeoutError",
    "InternalServerError",
    "ConnectionError",
    "TimeoutError",
    "Timeout",
})

# OpenAI SDK error class names that are permanent
_PERMANENT_NAMES = frozenset({
    "AuthenticationError",
    "BadRequestError",
    "PermissionDeniedError",
    "NotFoundError",
    "ValueError",
})


def classify_error(exc: Exception) -> str:
    """Classify an exception as 'retryable' or 'permanent'.

    Checks the exception class name and its MRO against known patterns.
    """
    names = {cls.__name__ for cls in type(exc).__mro__}
    if names & _RETRYABLE_NAMES:
        return "retryable"
    if names & _PERMANENT_NAMES:
        return "permanent"
    if isinstance(exc, (RetryableError, ConnectionError, TimeoutError)):
        return "retryable"
    if isinstance(exc, (PermanentError, ValueError, TypeError)):
        return "permanent"
    # Unknown errors default to retryable (safer — avoids silent data loss)
    return "retryable"


def with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True,
) -> T:
    """Execute fn() with exponential backoff retry on retryable errors.

    Args:
        fn: Zero-arg callable to execute.
        max_retries: Maximum number of retries after initial attempt.
        base_delay: Initial delay in seconds between retries.
        max_delay: Maximum delay cap in seconds.
        jitter: Add random jitter to delay to avoid thundering herd.

    Returns:
        The return value of fn().

    Raises:
        The last exception if all retries are exhausted, or immediately
        for permanent errors.
    """
    last_exc: Exception | None = None
    for attempt in range(1 + max_retries):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            kind = classify_error(exc)
            if kind == "permanent":
                logger.warning("Permanent error (no retry): %s", exc)
                raise
            if attempt >= max_retries:
                logger.error(
                    "All %d retries exhausted: %s", max_retries, exc,
                )
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            if jitter:
                delay *= 0.5 + random.random()
            logger.warning(
                "Retryable error (attempt %d/%d, retry in %.1fs): %s",
                attempt + 1, 1 + max_retries, delay, exc,
            )
            time.sleep(delay)
    raise last_exc  # unreachable, but makes type checker happy
```

**Step 4: Run test to verify it passes**

Run: `python tests/test_retry.py`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add core/retry.py tests/test_retry.py
git commit -m "feat: add LLM call retry utility with error classification"
```

---

### Task 2: Integrate Retry into LLM Evaluators

**Files:**
- Modify: `evaluators/llm_judge.py:83-103` (wrap API call)
- Modify: `evaluators/nl_assertion.py:124-145` (wrap API call)
- Modify: `evaluators/safety_evaluator.py:182-209` (wrap API call)
- Modify: `users/llm_user.py:58-65` (wrap API call)

**Step 1: Write the failing test**

```python
# Add to tests/test_retry.py at the end, before __main__

def test_llm_judge_retries(monkeypatch=None):
    """LLMJudge should retry on transient API errors."""
    import os
    os.environ.setdefault("ZHIPU_API_KEY", "test-key")
    from evaluators.llm_judge import LLMJudgeEvaluator
    from core.types import Task, Message, Role

    judge = LLMJudgeEvaluator(api_key="test-key")
    calls = []

    class FakeResponse:
        class Choice:
            class Msg:
                content = "Good response. SCORE: 4"
            message = Msg()
        choices = [Choice()]

    original_create = judge.client.chat.completions.create
    def fake_create(**kwargs):
        calls.append(1)
        if len(calls) < 2:
            raise ConnectionError("transient network error")
        return FakeResponse()

    judge.client.chat.completions.create = fake_create

    task = Task(id="t1", description="test", initial_message="hi",
                initial_state={}, required_info=["hello"])
    trajectory = [
        Message(role=Role.SYSTEM, content="sys"),
        Message(role=Role.USER, content="hi"),
        Message(role=Role.AGENT, content="hello world"),
    ]
    score = judge.evaluate(task, trajectory, None)
    assert score == 0.8  # 4/5
    assert len(calls) == 2  # 1 failure + 1 success
    print("  test_llm_judge_retries PASSED")
```

**Step 2: Run test to verify it fails**

Run: `python -c "from tests.test_retry import test_llm_judge_retries; test_llm_judge_retries()"`
Expected: FAIL — LLMJudge returns 0.0 (no retry, catches the ConnectionError and returns 0.0)

**Step 3: Apply retry to each LLM evaluator**

In `evaluators/llm_judge.py`, wrap the API call in `evaluate()`:

```python
# In evaluate() method, replace the try/except block:
        from core.retry import with_retry
        try:
            response = with_retry(
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                ),
                max_retries=3,
                base_delay=1.0,
            )
```

Apply the same pattern to:
- `evaluators/nl_assertion.py` — wrap `self.client.chat.completions.create(...)` in `with_retry(lambda: ...)`
- `evaluators/safety_evaluator.py` — wrap in `_evaluate_with_llm()`
- `users/llm_user.py` — wrap in `respond()`

**Step 4: Run test to verify it passes**

Run: `python tests/test_retry.py`
Expected: All tests PASS including `test_llm_judge_retries`

**Step 5: Commit**

```bash
git add evaluators/llm_judge.py evaluators/nl_assertion.py evaluators/safety_evaluator.py users/llm_user.py
git commit -m "feat: integrate retry with backoff into all LLM evaluators and user simulator"
```

---

### Task 3: Performance Profiler (`core/profiler.py`)

**Files:**
- Create: `core/profiler.py`
- Test: `tests/test_profiler.py`

**Step 1: Write the failing test**

```python
# tests/test_profiler.py
"""Tests for performance profiler."""
from __future__ import annotations

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.profiler import Profiler


def test_basic_timing():
    p = Profiler()
    with p.phase("setup"):
        time.sleep(0.05)
    with p.phase("work"):
        time.sleep(0.05)
    summary = p.summary()
    assert "setup" in summary
    assert "work" in summary
    assert summary["setup"] >= 0.04
    assert summary["work"] >= 0.04
    assert summary["total"] >= 0.08
    print("  test_basic_timing PASSED")


def test_accumulate_same_phase():
    p = Profiler()
    with p.phase("eval"):
        time.sleep(0.03)
    with p.phase("eval"):
        time.sleep(0.03)
    summary = p.summary()
    assert summary["eval"] >= 0.05
    assert summary["eval_count"] == 2
    print("  test_accumulate_same_phase PASSED")


def test_nested_phases():
    p = Profiler()
    with p.phase("outer"):
        time.sleep(0.02)
        with p.phase("inner"):
            time.sleep(0.02)
    summary = p.summary()
    assert "outer" in summary
    assert "inner" in summary
    print("  test_nested_phases PASSED")


def test_format_report():
    p = Profiler()
    with p.phase("agent_act"):
        time.sleep(0.02)
    with p.phase("env_step"):
        time.sleep(0.01)
    report = p.format_report()
    assert "agent_act" in report
    assert "env_step" in report
    print("  test_format_report PASSED")


if __name__ == "__main__":
    print("\n=== Profiler Tests ===\n")
    test_basic_timing()
    test_accumulate_same_phase()
    test_nested_phases()
    test_format_report()
    print("\n=== All profiler tests passed ===\n")
```

**Step 2: Run test to verify it fails**

Run: `python tests/test_profiler.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.profiler'`

**Step 3: Write the implementation**

```python
# core/profiler.py
"""Lightweight performance profiler for evaluation phases.

Tracks cumulative time spent in named phases (agent_act, env_step,
evaluator:name, etc.) with context manager API.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any


class Profiler:
    """Accumulates timing for named phases."""

    def __init__(self):
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._start_time: float = time.time()

    @contextmanager
    def phase(self, name: str):
        """Time a named phase. Accumulates if called multiple times."""
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            self._totals[name] = self._totals.get(name, 0.0) + elapsed
            self._counts[name] = self._counts.get(name, 0) + 1

    def summary(self) -> dict[str, Any]:
        """Return timing summary as a dict."""
        result: dict[str, Any] = {}
        for name, total in self._totals.items():
            result[name] = round(total, 3)
            count = self._counts[name]
            if count > 1:
                result[f"{name}_count"] = count
        result["total"] = round(time.time() - self._start_time, 3)
        return result

    def format_report(self) -> str:
        """Return a human-readable timing report."""
        summary = self.summary()
        total = summary.pop("total", 0)
        lines = []
        for key, val in sorted(summary.items()):
            if key.endswith("_count"):
                continue
            count = summary.get(f"{key}_count", 1)
            pct = (val / total * 100) if total > 0 else 0
            count_str = f" x{count}" if count > 1 else ""
            lines.append(f"  {key:20s}: {val:6.2f}s ({pct:4.1f}%){count_str}")
        lines.append(f"  {'total':20s}: {total:6.2f}s")
        return "\n".join(lines)
```

**Step 4: Run test to verify it passes**

Run: `python tests/test_profiler.py`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add core/profiler.py tests/test_profiler.py
git commit -m "feat: add lightweight performance profiler for evaluation phases"
```

---

### Task 4: Concurrent Evaluator Execution in Orchestrator

**Files:**
- Modify: `core/orchestrator.py:135-155` (replace sequential eval loop)
- Test: `tests/test_concurrent_eval.py`

**Step 1: Write the failing test**

```python
# tests/test_concurrent_eval.py
"""Tests for concurrent evaluator execution."""
from __future__ import annotations

import sys
import time
import threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import Role, Message, ToolCall, Task
from core.base import AgentAdapter, Evaluator, Environment
from core.orchestrator import Orchestrator
from environments.mock_db import MockDBEnvironment


class InstantAgent(AgentAdapter):
    def reset(self): pass
    def act(self, messages):
        return Message(role=Role.AGENT, content="Done. ###DONE###")


class SlowEvaluator(Evaluator):
    """Evaluator that takes 0.5s — used to test concurrency."""
    def __init__(self, score: float, delay: float = 0.5):
        self._score = score
        self._delay = delay
        self.last_reason = ""
        self.thread_id: int | None = None

    def evaluate(self, task, trajectory, env):
        self.thread_id = threading.current_thread().ident
        time.sleep(self._delay)
        self.last_reason = f"SlowEval score={self._score}"
        return self._score


def test_evaluators_run_concurrently():
    """Three 0.5s evaluators should complete in ~0.5s (not 1.5s) if concurrent."""
    task = Task(
        id="t-concurrent", description="test", initial_message="hi",
        initial_state={}, max_steps=1, single_turn=True,
    )
    evals = {
        "slow_a": SlowEvaluator(1.0, delay=0.5),
        "slow_b": SlowEvaluator(0.8, delay=0.5),
        "slow_c": SlowEvaluator(0.6, delay=0.5),
    }
    agent = InstantAgent()
    env = MockDBEnvironment()
    orch = Orchestrator(agent, env, evals)

    start = time.time()
    result = orch.run(task)
    elapsed = time.time() - start

    assert result.scores["slow_a"] == 1.0
    assert result.scores["slow_b"] == 0.8
    assert result.scores["slow_c"] == 0.6
    assert abs(result.overall_score - 1.0 * 0.8 * 0.6) < 0.01

    # Should be concurrent: ~0.5s, not ~1.5s
    assert elapsed < 1.0, f"Expected <1.0s (concurrent), got {elapsed:.2f}s"
    print(f"  test_evaluators_run_concurrently PASSED ({elapsed:.2f}s)")


def test_evaluator_crash_isolation():
    """A crashing evaluator should not break other evaluators."""
    task = Task(
        id="t-crash", description="test", initial_message="hi",
        initial_state={}, max_steps=1, single_turn=True,
    )

    class CrashEvaluator(Evaluator):
        def __init__(self):
            self.last_reason = ""
        def evaluate(self, task, trajectory, env):
            raise RuntimeError("boom")

    evals = {
        "good": SlowEvaluator(1.0, delay=0.1),
        "crash": CrashEvaluator(),
    }
    agent = InstantAgent()
    env = MockDBEnvironment()
    result = Orchestrator(agent, env, evals).run(task)

    assert result.scores["good"] == 1.0
    assert result.scores["crash"] == 0.0
    assert result.overall_score == 1.0  # crashed evaluator skipped in overall
    print("  test_evaluator_crash_isolation PASSED")


def test_concurrent_threads_are_different():
    """Verify evaluators actually run on different threads."""
    task = Task(
        id="t-threads", description="test", initial_message="hi",
        initial_state={}, max_steps=1, single_turn=True,
    )
    evals = {
        "a": SlowEvaluator(1.0, delay=0.3),
        "b": SlowEvaluator(1.0, delay=0.3),
    }
    agent = InstantAgent()
    env = MockDBEnvironment()
    Orchestrator(agent, env, evals).run(task)

    # Both evaluators should have recorded their thread id
    assert evals["a"].thread_id is not None
    assert evals["b"].thread_id is not None
    assert evals["a"].thread_id != evals["b"].thread_id, "Evaluators should run on different threads"
    print("  test_concurrent_threads_are_different PASSED")


if __name__ == "__main__":
    print("\n=== Concurrent Evaluator Tests ===\n")
    test_evaluators_run_concurrently()
    test_evaluator_crash_isolation()
    test_concurrent_threads_are_different()
    print("\n=== All concurrent eval tests passed ===\n")
```

**Step 2: Run test to verify it fails**

Run: `python tests/test_concurrent_eval.py`
Expected: `test_evaluators_run_concurrently` FAILS with `AssertionError: Expected <1.0s (concurrent), got 1.5Xs`

**Step 3: Rewrite the evaluation section of orchestrator.py**

Replace `orchestrator.py` lines 135-155 (the sequential `# ── Evaluation ──` block) with:

```python
        # ── Evaluation (concurrent) ──
        from concurrent.futures import ThreadPoolExecutor, as_completed

        scores: dict[str, float] = {}
        score_details: dict[str, str] = {}

        def _run_evaluator(name: str, evaluator: Evaluator) -> tuple[str, float, str]:
            """Run a single evaluator in a thread, return (name, score, detail)."""
            if self.on_progress:
                try:
                    self.on_progress("eval_start", {"name": name})
                except Exception:
                    pass
            detail = ""
            try:
                score = evaluator.evaluate(task, trajectory, self.env)
            except Exception as e:
                logger.error("Evaluator '%s' crashed: %s", name, e, exc_info=True)
                score = 0.0
                detail = f"[EVALUATOR_ERROR] {type(e).__name__}: {e}"
            if not detail:
                reason = getattr(evaluator, "last_reason", "")
                if reason:
                    model = getattr(evaluator, "model", "")
                    prefix = f"[评判模型: {model}]\n" if model else ""
                    detail = prefix + reason
            return name, score, detail

        with ThreadPoolExecutor(max_workers=len(self.evaluators)) as pool:
            futures = {
                pool.submit(_run_evaluator, name, ev): name
                for name, ev in self.evaluators.items()
            }
            for future in as_completed(futures):
                name, score, detail = future.result()
                scores[name] = score
                if detail:
                    score_details[name] = detail
```

**Step 4: Run tests to verify**

Run: `python tests/test_concurrent_eval.py && python tests/test_core.py`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add core/orchestrator.py tests/test_concurrent_eval.py
git commit -m "feat: run evaluators concurrently via ThreadPoolExecutor"
```

---

### Task 5: Integrate Profiler into Orchestrator

**Files:**
- Modify: `core/orchestrator.py` (add profiler instrumentation)
- Modify: `core/types.py:86-100` (add `profiling` field to `EvalResult`)
- Modify: `run.py:80-88` (print profiling in output)

**Step 1: Write the failing test**

```python
# Add to tests/test_concurrent_eval.py before __main__

def test_profiling_data_in_result():
    """EvalResult should contain profiling data."""
    task = Task(
        id="t-profile", description="test", initial_message="hi",
        initial_state={}, max_steps=1, single_turn=True,
    )
    evals = {"quick": SlowEvaluator(1.0, delay=0.1)}
    agent = InstantAgent()
    env = MockDBEnvironment()
    result = Orchestrator(agent, env, evals).run(task)

    assert hasattr(result, "profiling"), "EvalResult should have profiling field"
    assert "agent_act" in result.profiling
    assert "eval" in result.profiling
    assert "total" in result.profiling
    print(f"  test_profiling_data_in_result PASSED (profiling={result.profiling})")
```

**Step 2: Run test to verify it fails**

Run: `python -c "from tests.test_concurrent_eval import test_profiling_data_in_result; test_profiling_data_in_result()"`
Expected: FAIL — `AttributeError: 'EvalResult' has no attribute 'profiling'`

**Step 3: Add profiling field to EvalResult**

In `core/types.py`, add to the `EvalResult` dataclass:
```python
    profiling: dict[str, Any] = field(default_factory=dict)
```

And include in `summary()`:
```python
        if self.profiling:
            result["profiling"] = self.profiling
```

**Step 4: Instrument the orchestrator**

In `core/orchestrator.py`, at the top of `run()`:
```python
from core.profiler import Profiler
profiler = Profiler()
```

Wrap agent.act:
```python
with profiler.phase("agent_act"):
    agent_msg = self.agent.act(trajectory)
```

Wrap env.step:
```python
with profiler.phase("env_step"):
    result = self.env.step(tc)
```

Wrap the entire evaluation block:
```python
with profiler.phase("eval"):
    # ... concurrent evaluation code ...
```

Add to EvalResult construction:
```python
profiling=profiler.summary(),
```

**Step 5: Print profiling in run.py**

In `run.py`, add after the scores display in `print_result()`:
```python
    # Profiling
    if result.profiling:
        print(f"       Profiling:")
        for key, val in result.profiling.items():
            if not key.endswith("_count"):
                print(f"         {key:15s}: {val:.2f}s")
        print()
```

**Step 6: Run all tests**

Run: `python tests/test_concurrent_eval.py && python tests/test_core.py && python tests/test_profiler.py`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add core/types.py core/orchestrator.py run.py
git commit -m "feat: integrate performance profiler into orchestrator and results"
```

---

### Task 6: Run Full Test Suite & Verify

**Step 1: Run all existing tests**

```bash
python tests/test_core.py
python tests/test_multi_turn.py
python tests/test_passthrough.py
python tests/test_retry.py
python tests/test_profiler.py
python tests/test_concurrent_eval.py
```

Expected: All pass.

**Step 2: Verify no regressions by running a quick evaluation (if API key available)**

```bash
python run.py config.yaml
```

Check that:
- Evaluators show timing in output
- Profiling data appears in the result JSON
- Evaluation is noticeably faster with concurrent evaluators

**Step 3: Final commit (if any fixups needed)**

```bash
git add -A
git commit -m "fix: address test suite regressions from concurrent eval changes"
```
