"""Junior Researcher — revises the plain-text research memo after The Critic (no LaTeX)."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from google import genai
from google.genai import types

from gemini_pacing import gemini_retry_sleep_seconds
from gemini_report import _extract_delimited, is_transient_gemini_api_error
from gemini_stream_events import emit_gemini_request

MEMO_REFINER_SYSTEM = """You are **Junior Researcher** on a **revision pass** after editorial feedback from **The Critic** at the same desk.

You receive your **prior plain-text / Markdown research memo** (no LaTeX) and The Critic's memo. Revise and **materially expand** the memo—do **not** output LaTeX, documentclass, tables as TeX, or PDF layout. The **Lead Portfolio Manager** will typeset the final PDF.

Requirements:
- **Address** The Critic's numbered priority fixes explicitly.
- **Structure:** use clear Markdown headings (`##`, `###`) that mirror a sell-side note (thesis, business model, industry, financials, valuation logic, catalysts, risks, data limitations) so the Lead PM can map sections to LaTeX `\\section` commands without guessing.
- **Length:** aim for a **long** desk memo (roughly **5k–14k words** when substance supports it). Add interpretation, scenario discussion, and explicit evidence→inference chains (**anti–black-box**); the Lead PM will **expand** further—give them dense raw material, not a short brief.
- **URLs:** keep headline URLs in prose (domain + path is fine); do not strip sourcing—the Lead PM will convert to LaTeX footnotes later.
- **Do not** invent filing facts not implied by the prior memo and reasoning context.

Output **only** one block (ASCII delimiters, no markdown fences around the whole reply):

<<<MEMO>>>
...full revised memo...
<<<END_MEMO>>>
"""

_MAX_MEMO_IN = 600_000
_MAX_CRITIC = 120_000
_MAX_REASONING = 80_000


def run_junior_refine_memo(
    *,
    api_key: str,
    model: str,
    ticker: str,
    company_name: str,
    current_memo: str,
    critic_memo: str,
    junior_reasoning: str | None,
    peer_round: int,
    emit: Callable[[dict[str, Any]], None] | None = None,
    max_attempts: int = 3,
) -> tuple[str, str]:
    """Return ``(revised_memo, error_message)``; ``error_message`` empty on success."""
    key = (api_key or "").strip()
    if not key:
        return "", "no_api_key"

    memo = (current_memo or "").strip()
    if len(memo) > _MAX_MEMO_IN:
        memo = memo[:_MAX_MEMO_IN] + "\n\n[… memo truncated …]\n"

    crit = (critic_memo or "").strip()
    if len(crit) > _MAX_CRITIC:
        crit = crit[:_MAX_CRITIC] + "\n[… critic truncated …]\n"

    reason = (junior_reasoning or "").strip()
    if len(reason) > _MAX_REASONING:
        reason = reason[:_MAX_REASONING] + "\n[… truncated …]\n"

    user = (
        f"Issuer: {company_name} ({ticker}) — peer memo revision round {peer_round}\n\n"
        f"--- Your prior internal reasoning (consistency only; do not paste verbatim into MEMO) ---\n"
        f"{reason or '(none)'}\n\n"
        f"--- The Critic's memo for this round ---\n{crit or '(empty)'}\n\n"
        f"--- Your prior research memo (revise, expand, return full MEMO only) ---\n{memo or '(empty)'}\n"
    )

    client = genai.Client(api_key=key)
    cfg = types.GenerateContentConfig(
        system_instruction=MEMO_REFINER_SYSTEM,
        temperature=0.24,
        max_output_tokens=65536,
    )
    last_err = ""
    for attempt in range(max(1, max_attempts)):
        try:
            emit_gemini_request(emit, call_site="junior_refine_memo")
            response = client.models.generate_content(
                model=model,
                contents=user,
                config=cfg,
            )
            raw = (response.text or "").strip()
            if not raw:
                return "", "empty_junior_memo_refine_response"
            body = _extract_delimited(raw, "<<<MEMO>>>", "<<<END_MEMO>>>")
            if body and len(body.strip()) >= 400:
                return body.strip(), ""
            return "", "junior_refine_memo_missing_delimiters"
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            if attempt == max_attempts - 1 or not is_transient_gemini_api_error(exc):
                return "", f"junior_refine_memo_model_error: {exc}"
            time.sleep(gemini_retry_sleep_seconds(attempt))
    return "", last_err or "junior_refine_memo_model_error"
