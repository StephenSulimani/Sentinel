"""Junior Researcher — peer-response revision pass (LaTeX) after The Critic's memo."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from google import genai
from google.genai import types

from gemini_pacing import gemini_retry_sleep_seconds
from gemini_report import _extract_delimited, _latex_from_fenced, is_transient_gemini_api_error
from gemini_stream_events import emit_gemini_request

REFINER_SYSTEM = """You are **Junior Researcher** on a **revision pass** after editorial feedback from **The Critic** at the same desk.

You receive your **prior pdflatex equity note** (full LaTeX) and The Critic's plain-English memo (priorities, risks, structural fixes). Your job is to **revise and materially expand** the note—not to start over from scratch.

Requirements:
- **Address** The Critic's numbered priority fixes explicitly in structure and prose (you may add subsections, tables, or risk bullets where they flagged thin coverage).
- **Target length:** aim for **8–15 pages** of substantive PDF when the data supports it—**expand** thin or telegraphic passages into explicit reasoning (evidence → inference → implication). If the workbook is sparse, say so—but still deepen analysis where honest.
- **Anti–black-box:** add or lengthen `\\paragraph{Interpretation}` / “**Why this matters**” blocks where the draft reads like a fact list; the PM must see *how* you reasoned, not only *what* you claim.
- **Preserve** all numerical facts, filing tags, and workbook-grounded figures unless the Critic flagged a clear inconsistency—then reconcile conservatively with a footnote.
- **Preserve and extend** headline-sourced narrative and each `\\footnote{...}` / `\\url{https://...}` tied to the desk headline scan. If The Critic flagged **sparse citations**, **add** substantive `\\footnote{...}` markers (with `\\url{...}` for external URLs) until web-sourced narrative sections show **frequent** superscript footnote callouts—do not strip footnotes to shorten the file.
- **Do not** invent new primary financial facts not implied by the workbook.
- Output **only** a complete pdflatex document using delimiters:

<<<LATEX>>>
\\documentclass[11pt]{article}
...
<<<END_LATEX>>>

Use `xurl` + `hyperref` (hyperref after `xurl`) when URLs appear. Professional sell-side tone. Not-investment-advice disclaimer in-document."""

_MAX_INPUT_TEX = 1_200_000
_MAX_CRITIC = 120_000
_MAX_REASONING = 80_000


def run_junior_refine_latex(
    *,
    api_key: str,
    model: str,
    ticker: str,
    company_name: str,
    current_latex: str,
    critic_memo: str,
    junior_reasoning: str | None,
    peer_round: int,
    emit: Callable[[dict[str, Any]], None] | None = None,
    max_attempts: int = 3,
) -> tuple[str, str]:
    """Return ``(revised_latex, error_message)``; ``error_message`` empty on success."""
    key = (api_key or "").strip()
    if not key:
        return "", "no_api_key"

    tex = (current_latex or "").strip()
    if len(tex) > _MAX_INPUT_TEX:
        tex = tex[:_MAX_INPUT_TEX] + "\n\n[… truncated …]\n"

    crit = (critic_memo or "").strip()
    if len(crit) > _MAX_CRITIC:
        crit = crit[:_MAX_CRITIC] + "\n[… critic truncated …]\n"

    reason = (junior_reasoning or "").strip()
    if len(reason) > _MAX_REASONING:
        reason = reason[:_MAX_REASONING] + "\n[… truncated …]\n"

    user = (
        f"Issuer: {company_name} ({ticker}) — peer revision round {peer_round}\n\n"
        f"--- Your prior internal reasoning (for consistency; do not paste verbatim into LaTeX) ---\n"
        f"{reason or '(none)'}\n\n"
        f"--- The Critic's memo for this round ---\n{crit or '(empty)'}\n\n"
        f"--- Your prior LaTeX note (revise, expand, return full document) ---\n{tex or '(empty)'}\n"
    )

    client = genai.Client(api_key=key)
    cfg = types.GenerateContentConfig(
        system_instruction=REFINER_SYSTEM,
        temperature=0.22,
        max_output_tokens=65536,
    )
    last_err = ""
    for attempt in range(max(1, max_attempts)):
        try:
            emit_gemini_request(emit, call_site="junior_refine")
            response = client.models.generate_content(
                model=model,
                contents=user,
                config=cfg,
            )
            raw = (response.text or "").strip()
            if not raw:
                return "", "empty_junior_refine_response"
            latex = _extract_delimited(raw, "<<<LATEX>>>", "<<<END_LATEX>>>")
            if latex and "\\documentclass" in latex:
                return latex.strip(), ""
            fb = _latex_from_fenced(raw)
            if fb:
                return fb.strip(), ""
            return "", "junior_refine_missing_latex_delimiters"
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            if attempt == max_attempts - 1 or not is_transient_gemini_api_error(exc):
                return "", f"junior_refine_model_error: {exc}"
            time.sleep(gemini_retry_sleep_seconds(attempt))
    return "", last_err or "junior_refine_model_error"
