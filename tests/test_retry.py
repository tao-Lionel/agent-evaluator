"""Unit tests for core/retry.py — retry utility with exponential backoff."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.retry import with_retry, classify_error, RetryableError, PermanentError


def test_succeeds_first_try():
    """fn succeeds immediately, called once."""
    call_count = 0

    def fn():
        nonlocal call_count
        call_count += 1
        return "ok"

    result = with_retry(fn, base_delay=0.01)
    assert result == "ok", f"Expected 'ok', got {result}"
    assert call_count == 1, f"Expected 1 call, got {call_count}"
    print("  PASSED")


def test_retries_on_retryable_error():
    """fn fails twice with retryable error then succeeds, 3 total calls."""
    call_count = 0

    def fn():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("connection refused")
        return "recovered"

    result = with_retry(fn, max_retries=3, base_delay=0.01)
    assert result == "recovered", f"Expected 'recovered', got {result}"
    assert call_count == 3, f"Expected 3 calls, got {call_count}"
    print("  PASSED")


def test_raises_after_max_retries():
    """fn always fails with retryable error, raises after 1 + max_retries attempts."""
    call_count = 0

    def fn():
        nonlocal call_count
        call_count += 1
        raise TimeoutError("timed out")

    try:
        with_retry(fn, max_retries=2, base_delay=0.01)
        assert False, "Should have raised"
    except TimeoutError:
        pass

    expected = 3  # 1 initial + 2 retries
    assert call_count == expected, f"Expected {expected} calls, got {call_count}"
    print("  PASSED")


def test_no_retry_on_permanent_error():
    """Permanent error is raised immediately, fn called only once."""
    call_count = 0

    def fn():
        nonlocal call_count
        call_count += 1
        raise ValueError("bad input")

    try:
        with_retry(fn, max_retries=3, base_delay=0.01)
        assert False, "Should have raised"
    except ValueError:
        pass

    assert call_count == 1, f"Expected 1 call, got {call_count}"
    print("  PASSED")


def test_classify_openai_errors():
    """Verify classify_error works on ConnectionError, TimeoutError, ValueError."""
    assert classify_error(ConnectionError("conn")) == "retryable", \
        "ConnectionError should be retryable"
    assert classify_error(TimeoutError("timeout")) == "retryable", \
        "TimeoutError should be retryable"
    assert classify_error(ValueError("bad")) == "permanent", \
        "ValueError should be permanent"

    # Also test custom error classes
    assert classify_error(RetryableError("retry me")) == "retryable", \
        "RetryableError should be retryable"
    assert classify_error(PermanentError("stop")) == "permanent", \
        "PermanentError should be permanent"
    print("  PASSED")


if __name__ == "__main__":
    print("\n=== Retry Utility Tests ===\n")
    test_succeeds_first_try()
    test_retries_on_retryable_error()
    test_raises_after_max_retries()
    test_no_retry_on_permanent_error()
    test_classify_openai_errors()
    print("\n=== All tests passed ===\n")
