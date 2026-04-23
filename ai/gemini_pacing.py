"""Centralized Gemini pacing to reduce 429 RESOURCE_EXHAUSTED (RPM/TPM bursts).

All delays are optional and tunable via environment variables (seconds, floats).
"""

from __future__ import annotations

import os
import time


def _ef(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def gemini_retry_sleep_seconds(attempt_zero_based: int) -> float:
    """Exponential backoff after a failed Gemini call (e.g. 429/503).

    ``sleep = min(max_sec, base * mult**attempt)``.
    """
    base = _ef("GEMINI_RETRY_BACKOFF_BASE_SEC", 5.0)
    mult = _ef("GEMINI_RETRY_BACKOFF_MULT", 2.0)
    cap = _ef("GEMINI_RETRY_BACKOFF_MAX_SEC", 120.0)
    n = max(0, int(attempt_zero_based))
    return min(cap, base * (mult**n))


def sleep_between_gemini_pipeline_agents() -> None:
    """Pause between Junior → Critic → Lead PM (sequential Gemini on one API key)."""
    sec = _ef("GEMINI_BETWEEN_AGENTS_SEC", 8.0)
    if sec > 0:
        time.sleep(sec)


def sleep_between_gemini_tool_presets() -> None:
    """Pause before trying the next tool preset (e.g. Search-only after Search+URL)."""
    sec = _ef("GEMINI_BETWEEN_TOOL_PRESETS_SEC", 5.0)
    if sec > 0:
        time.sleep(sec)


def optional_gemini_preflight_delay() -> None:
    """Optional pause before the first heavy Gemini call in a run (e.g. narrative stream)."""
    sec = _ef("GEMINI_PREFLIGHT_DELAY_SEC", 2.0)
    if sec > 0:
        time.sleep(sec)


def gemini_pdf_repair_stagger(extra_zero_based: int) -> None:
    """Backoff between outer LaTeX repair retries when the inner repair still looks transient."""
    base = _ef("GEMINI_PDF_REPAIR_STAGGER_BASE_SEC", 12.0)
    mx = _ef("GEMINI_PDF_REPAIR_STAGGER_MAX_SEC", 60.0)
    n = max(0, int(extra_zero_based))
    sec = min(mx, base * (n + 1))
    if sec > 0:
        time.sleep(sec)
