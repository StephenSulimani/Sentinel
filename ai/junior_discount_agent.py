"""Junior Researcher: choose illustrative discount rate after headline context, before quant."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from google import genai
from google.genai import types

from gemini_pacing import gemini_retry_sleep_seconds
from gemini_report import is_transient_gemini_api_error
from gemini_stream_events import emit_gemini_request
from search_plan import _extract_json_object

_log = logging.getLogger(__name__)

_SYSTEM = (
    "You are the Junior Researcher on an equity desk. Choose a single **annual discount rate** "
    "for a *stylized five-year illustrative FCF-to-NPV sensitivity* (teaching / bridge math—not a "
    "formal WACC or fairness opinion).\n\n"
    "Reply with **only** one JSON object (no markdown fences, no commentary) with keys:\n"
    '- `"discount_rate"`: number, the rate as a **decimal** (e.g. 0.095 for 9.5%). '
    "Must be between **0.06 and 0.20** inclusive.\n"
    '- `"basis"`: string, **2–4 sentences** explaining the choice from the supplied fundamentals, '
    "beta/return window if present, and **tone of headline risk** (litigation, macro, idiosyncratic "
    "shocks). Do not name vendor SDKs or search engines.\n\n"
    "You may align with the provided **seed** from a simple CAPM sketch or override it when the "
    "issuer profile and headline scan justify a different mechanical rate."
)


def _compact_market(snap: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(snap, dict):
        return {}
    info = snap.get("info") if isinstance(snap.get("info"), dict) else {}
    out: dict[str, Any] = {}
    for k in ("beta", "marketCap", "trailingPE", "shortName", "longName", "exchange", "currency"):
        if k in info:
            out[k] = info.get(k)
    perf = snap.get("window_return_approx")
    if isinstance(perf, (int, float)) and perf == perf:
        out["window_return_approx"] = float(perf)
    if snap.get("error"):
        out["vendor_error"] = str(snap["error"])[:200]
    return out


def _clamp_rate(raw: Any) -> float | None:
    if not isinstance(raw, (int, float)) or raw != raw:
        return None
    r = float(raw)
    if r > 1.0:
        r = r / 100.0
    if 0.06 <= r <= 0.20:
        return r
    return None


def run_junior_discount_decision(
    *,
    api_key: str | None,
    model: str,
    ticker: str,
    company_name: str,
    market_snapshot: dict[str, Any],
    headline_titles: list[str],
    planned_queries: list[str],
    seed_rate: float,
    seed_basis: str,
    emit: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[float, str, str | None, bool]:
    """Return ``(discount_rate, basis, warning_or_none, used_model)``.

    Without an API key or on failure, returns the deterministic **seed** (same sketch as before)
    and ``used_model`` False.
    """
    sym = ticker.strip().upper()
    name = (company_name or sym).strip() or sym
    key = (api_key or "").strip() or None
    if not key:
        return seed_rate, seed_basis, None, False

    payload = {
        "ticker": sym,
        "company_name": name,
        "seed": {"discount_rate": seed_rate, "rationale": seed_basis},
        "market": _compact_market(market_snapshot),
        "headline_scan": {
            "n_headlines": len(headline_titles),
            "sample_titles": headline_titles[:12],
        },
        "search_queries_sample": planned_queries[:6],
    }
    user = json.dumps(payload, indent=2, default=str)

    client = genai.Client(api_key=key)
    cfg = types.GenerateContentConfig(
        system_instruction=_SYSTEM,
        temperature=0.25,
        max_output_tokens=768,
    )
    text = ""
    max_attempts = 6
    _log.info("junior_discount: calling Gemini model=%r ticker=%s", model, sym)
    for attempt in range(max_attempts):
        try:
            emit_gemini_request(emit, call_site="junior_discount")
            response = client.models.generate_content(
                model=model,
                contents=user,
                config=cfg,
            )
            text = (response.text or "").strip()
            break
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "junior_discount: Gemini error attempt=%s/%s: %s",
                attempt + 1,
                max_attempts,
                exc,
                exc_info=_log.isEnabledFor(logging.DEBUG),
            )
            if attempt == max_attempts - 1 or not is_transient_gemini_api_error(exc):
                warn = f"junior_discount_model: {exc}"
                return (
                    seed_rate,
                    f"{seed_basis} Fallback: model error; using deterministic seed.",
                    warn,
                    False,
                )
            time.sleep(gemini_retry_sleep_seconds(attempt))

    obj = _extract_json_object(text)
    rate = _clamp_rate(obj.get("discount_rate")) if isinstance(obj, dict) else None
    basis_raw = obj.get("basis") if isinstance(obj, dict) else None
    basis = basis_raw.strip() if isinstance(basis_raw, str) else ""

    if rate is None or len(basis) < 20:
        return (
            seed_rate,
            f"{seed_basis} Fallback: invalid JSON or missing basis; using deterministic seed.",
            "junior_discount_model: unparseable or out-of-range response",
            False,
        )

    return rate, basis, None, True
