"""Retry utility with exponential backoff for LLM API calls."""

from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# --- Error base classes ---------------------------------------------------

class RetryableError(Exception):
    """Base class for errors that should trigger a retry."""


class PermanentError(Exception):
    """Base class for errors that should NOT be retried."""


# --- Classification -------------------------------------------------------

# Class names (checked against the exception class and its MRO) that
# indicate the error is retryable.
_RETRYABLE_NAMES = frozenset({
    "APIConnectionError",
    "RateLimitError",
    "APITimeoutError",
    "InternalServerError",
    "ConnectionError",
    "TimeoutError",
    "Timeout",
    "RetryableError",
})

# Class names that indicate the error is permanent (no retry).
_PERMANENT_NAMES = frozenset({
    "AuthenticationError",
    "BadRequestError",
    "PermissionDeniedError",
    "NotFoundError",
    "ValueError",
    "PermanentError",
})


def classify_error(exc: Exception) -> str:
    """Return ``"retryable"`` or ``"permanent"`` for the given exception.

    Classification is based on the class name and its MRO so that it works
    with OpenAI SDK subclasses without importing them.  Unknown errors
    default to ``"retryable"`` (safer — avoids silently dropping transient
    failures).
    """
    mro_names = {cls.__name__ for cls in type(exc).__mro__}

    # Check permanent first so that an explicit permanent marker wins if
    # someone creates a class that somehow appears in both sets.
    if mro_names & _PERMANENT_NAMES:
        return "permanent"
    if mro_names & _RETRYABLE_NAMES:
        return "retryable"

    # Unknown → treat as retryable (safer default).
    return "retryable"


# --- Retry wrapper ---------------------------------------------------------

def with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True,
) -> T:
    """Execute *fn()* with exponential backoff on retryable errors.

    Parameters
    ----------
    fn:
        Zero-argument callable to execute.
    max_retries:
        Maximum number of **retries** (total attempts = 1 + max_retries).
    base_delay:
        Initial delay in seconds before the first retry.
    max_delay:
        Upper bound on the computed delay.
    jitter:
        If ``True``, multiply the delay by a random factor in [0.5, 1.5).

    Returns
    -------
    T
        The return value of *fn()* on success.

    Raises
    ------
    Exception
        The last exception if all attempts are exhausted, or a permanent
        error on the first occurrence.
    """
    last_exc: Exception | None = None

    for attempt in range(1 + max_retries):
        try:
            return fn()
        except Exception as exc:
            classification = classify_error(exc)

            if classification == "permanent":
                logger.error(
                    "Permanent error on attempt %d/%d: %s",
                    attempt + 1,
                    1 + max_retries,
                    exc,
                )
                raise

            last_exc = exc

            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                if jitter:
                    delay *= random.uniform(0.5, 1.5)
                logger.warning(
                    "Retryable error on attempt %d/%d, retrying in %.2fs: %s",
                    attempt + 1,
                    1 + max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "All %d attempts exhausted. Last error: %s",
                    1 + max_retries,
                    exc,
                )

    # Should not be reachable, but just in case.
    raise last_exc  # type: ignore[misc]
