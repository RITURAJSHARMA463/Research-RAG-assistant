"""Generic retry-with-backoff for rate-limited API calls (429 / RESOURCE_EXHAUSTED).

Used by both the embeddings wrapper and the LLM call sites so free-tier quotas
degrade into "wait a bit" instead of a hard failure. Provider-agnostic: it just
looks for "429" / "RESOURCE_EXHAUSTED" in the exception string and, if present,
tries to parse the provider's own suggested wait time before falling back to
exponential backoff.
"""
from __future__ import annotations

import re
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def _parse_retry_delay(exc: Exception, attempt: int) -> float:
    # Google's error body includes a suggested wait, e.g. "retry in 28.38s".
    match = re.search(r"retry.{0,15}?([\d.]+)\s*s", str(exc), re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1.0  # small buffer
    return min(60.0, (2 ** attempt) + 1.0)


def is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def call_with_retry(fn: Callable[..., T], *args, max_retries: int = 6, label: str = "call", **kwargs) -> T:
    """Call fn(*args, **kwargs); on a rate-limit error, wait and retry.

    Non-rate-limit exceptions are raised immediately (no point retrying those).
    """
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - inspected via is_rate_limit_error
            if not is_rate_limit_error(exc) or attempt == max_retries - 1:
                raise
            wait = _parse_retry_delay(exc, attempt)
            print(f"  ! {label} rate-limited, retrying in {wait:.0f}s "
                  f"(attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
    raise RuntimeError("unreachable")  # loop always returns or raises
