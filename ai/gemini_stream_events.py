"""Lightweight SSE helpers for Gemini (avoid import cycles with ``research_common`` / ``report_builder``)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def emit_gemini_request(
    emit: Callable[[dict[str, Any]], None] | None,
    *,
    call_site: str | None = None,
) -> None:
    """Notify SSE clients that an outbound Gemini API call is about to be made (for UI counters)."""
    if emit is None:
        return
    ev: dict[str, Any] = {"type": "gemini_request"}
    if call_site:
        ev["call_site"] = call_site
    emit(ev)
