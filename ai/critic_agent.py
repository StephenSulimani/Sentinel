"""The Critic — reviews the Junior Researcher's draft (plain-text memo)."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from google import genai
from google.genai import types

from gemini_pacing import gemini_retry_sleep_seconds
from gemini_report import is_transient_gemini_api_error
from gemini_stream_events import emit_gemini_request

CRITIC_SYSTEM = """You are **The Critic**, a senior editorial reviewer at a top-tier asset manager. Your job is to critique an equity **research memo** produced by a junior analyst (the "Junior Researcher" pipeline).

You receive their **internal reasoning** (verbatim where provided) plus the Junior’s **plain-text research memo** (no LaTeX—another agent will typeset the PDF). Treat it like desk notes: headings, bullets, and short tables in plain text are fine. Respond in **plain English only** (no LaTeX, no JSON). Be direct, professional, and constructive.

Cover, where relevant:
- Factual discipline: overstated claims, missing caveats, alignment with stated data limitations.
- Argument structure: thesis clarity, flow, missing bull/bear balance.
- **Reasoning transparency (“black box” test):** flag passages that read like conclusions without logic chains; demand explicit **interpretation** paragraphs so a PM can follow evidence → inference (the Lead PM will mirror that structure in the final PDF).
- Risk and catalyst coverage: gaps, boilerplate, what a PM would still ask.
- Valuation and price-target narrative: whether assumptions and uncertainty are communicated honestly; whether the **rating** and **numeric target** are justified in prose (do not ask for LaTeX footnotes here—ask for **which claims need a cited URL or filing tag** so the typesetter can footnote them).

End with **three** numbered priority fixes. In **follow-up peer-review rounds** (when the user message says so), focus on what is still weak relative to an **8–15 page** institutional desk note worth of **substance** after the Junior’s revision—do not repeat praise for issues already fixed.

Keep length roughly 700–1,400 words unless the draft is so thin that a shorter memo suffices."""

# Reasoning can be long; cap only as a safety rail against pathological payloads.
_MAX_REASONING_CHARS = 500_000


def run_critic_memo(
    *,
    api_key: str,
    model: str,
    ticker: str,
    company_name: str,
    junior_note_plaintext: str,
    junior_reasoning: str | None,
    peer_round: int = 1,
    prior_memos_excerpt: str | None = None,
    emit: Callable[[dict[str, Any]], None] | None = None,
    max_attempts: int = 3,
) -> tuple[str, str]:
    """Return ``(memo_plain_text, error_message)``. ``error_message`` empty on success.

    ``junior_note_plaintext`` should be the Junior’s **research memo** (plain text / Markdown),
    not LaTeX—the Lead PM owns PDF typesetting.
    """
    key = (api_key or "").strip()
    if not key:
        return "", "no_api_key"

    note = (junior_note_plaintext or "").strip()
    if not note:
        return "", "empty_critic_note_context"

    reason = (junior_reasoning or "").strip()
    if len(reason) > _MAX_REASONING_CHARS:
        reason = reason[:_MAX_REASONING_CHARS] + "\n[… reasoning truncated (safety cap) …]\n"

    pr = max(1, int(peer_round))
    peer_ctx = ""
    if pr > 1:
        peer_ctx = (
            f"\n\n**Peer-review round {pr}.** The Junior revised the **research memo** after your prior memo. "
            "Assess the **updated** memo: what improved, what still misses depth for an **8–15 page** institutional desk note.\n"
        )
        prev = (prior_memos_excerpt or "").strip()
        if prev:
            peer_ctx += (
                "\n--- Excerpt of your earlier memos (continuity only; do not re-litigate fixes already applied) ---\n"
                f"{prev[:14_000]}\n"
            )

    user = (
        f"Issuer: {company_name} ({ticker})\n\n"
        f"--- Junior reasoning (internal) ---\n{reason or '(none provided)'}\n\n"
        f"--- Junior note (plain text from LaTeX body) ---\n{note}\n"
        f"{peer_ctx}"
    )

    client = genai.Client(api_key=key)
    cfg = types.GenerateContentConfig(
        system_instruction=CRITIC_SYSTEM,
        temperature=0.25,
        max_output_tokens=16384,
    )
    last_err = ""
    for attempt in range(max(1, max_attempts)):
        try:
            emit_gemini_request(emit, call_site="critic_memo")
            response = client.models.generate_content(
                model=model,
                contents=user,
                config=cfg,
            )
            text = (response.text or "").strip()
            if not text:
                return "", "empty_critic_response"
            return text, ""
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            if attempt == max_attempts - 1 or not is_transient_gemini_api_error(exc):
                return "", f"critic_model_error: {exc}"
            time.sleep(gemini_retry_sleep_seconds(attempt))
    return "", last_err or "critic_model_error"
