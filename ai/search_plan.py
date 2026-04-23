"""LLM-planned headline search queries (SearXNG / meta-search), with safe fallbacks."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any

from google import genai
from google.genai import types

from gemini_pacing import gemini_retry_sleep_seconds
from gemini_report import is_transient_gemini_api_error
from gemini_stream_events import emit_gemini_request

_log = logging.getLogger(__name__)

_PLAN_SYSTEM = (
    "You help equity researchers plan web search queries. "
    "Reply with **only** a single JSON object, no markdown fences, no commentary."
)


def fallback_headline_queries(ticker: str, company_name: str) -> list[str]:
    """Diverse business-news angles when no model key or planning fails."""
    sym = ticker.strip().upper()
    name = (company_name or sym).strip() or sym
    return [
        f"{sym} stock company news latest",
        f"{name} earnings revenue outlook",
        f"{sym} industry competition analysts",
        f"{name} CEO strategy investors",
        f"{sym} business expansion partnership",
    ]


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines.pop()
        text = "\n".join(lines).strip()
    m = re.search(r"\{[\s\S]*\}\s*$", text)
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def plan_headline_search_queries(
    *,
    ticker: str,
    company_name: str,
    api_key: str | None,
    model: str,
    emit: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[str], str | None]:
    """Return ``(queries, planning_error)``.

    ``planning_error`` is a short diagnostic when the model could not produce
    usable queries (caller may still use ``fallback_headline_queries``).
    """
    key = (api_key or "").strip() or None
    if not key:
        return fallback_headline_queries(ticker, company_name), None

    sym = ticker.strip().upper()
    name = (company_name or sym).strip() or sym
    user = f"""Ticker: {sym}
Company name: {name}

Design **5 to 7** distinct meta-search queries for a headline aggregator covering **general business and market news** for this issuer—not U.S. SEC filing or EDGAR-specific searches.

Cover a **mix** of angles, for example: recent company developments, sector and competitive context, financial performance and guidance narrative, strategy / capital allocation / M&A where relevant, regulatory or macro only if plausibly material to this name.

Requirements:
- Each query is one line, English, **under 92 characters**, no `site:` operators.
- Do **not** center queries on SEC, 10-K, 10-Q, EDGAR, or XBRL (filings can be mentioned elsewhere in the research stack).
- Avoid near-duplicate wording across queries.

Return exactly this JSON shape (double quotes, valid JSON):
{{"queries": ["query1", "query2", ...]}}
"""

    client = genai.Client(api_key=key)
    cfg = types.GenerateContentConfig(
        system_instruction=_PLAN_SYSTEM,
        temperature=0.4,
        max_output_tokens=1024,
    )
    text = ""
    max_attempts = 6
    _log.info("search_plan: calling Gemini model=%r for ticker=%s", model, sym)
    for attempt in range(max_attempts):
        try:
            emit_gemini_request(emit, call_site="search_plan")
            response = client.models.generate_content(
                model=model,
                contents=user,
                config=cfg,
            )
            text = (response.text or "").strip()
            break
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "search_plan: Gemini error attempt=%s/%s: %s",
                attempt + 1,
                max_attempts,
                exc,
                exc_info=_log.isEnabledFor(logging.DEBUG),
            )
            if attempt == max_attempts - 1 or not is_transient_gemini_api_error(exc):
                return fallback_headline_queries(sym, name), f"search_plan_model: {exc}"
            time.sleep(gemini_retry_sleep_seconds(attempt))

    obj = _extract_json_object(text)
    raw_list = obj.get("queries") if isinstance(obj, dict) else None
    if not isinstance(raw_list, list):
        return fallback_headline_queries(sym, name), "search_plan_model: missing queries array"

    cleaned: list[str] = []
    for item in raw_list:
        if isinstance(item, str):
            q = " ".join(item.split())
            if 3 <= len(q) <= 120:
                cleaned.append(q)
        if len(cleaned) >= 8:
            break

    if len(cleaned) < 3:
        return fallback_headline_queries(sym, name), "search_plan_model: too few valid queries"

    _log.info("search_plan: ok %s queries for ticker=%s", len(cleaned), sym)
    return cleaned, None
