"""The Lead Portfolio Manager — sole author of client LaTeX from desk research memos + workbook."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

from google import genai
from google.genai import types

from gemini_pacing import gemini_retry_sleep_seconds
from gemini_report import _extract_delimited, _latex_from_fenced, is_transient_gemini_api_error
from gemini_stream_events import emit_gemini_request

LPM_SYSTEM = r"""You are **The Lead Portfolio Manager** at an institutional asset manager. You are the **only** author of the client-facing **LaTeX / PDF** equity research note.

**Product:** This is **not** a memo, slide outline, or executive summary. It must read like a **published sell-side initiation / update note**: long-form, self-contained, and suitable for a PM who did not see the pipeline. **Expand** every idea from the Junior memos into full prose—do **not** compress, bullet-strip, or “tighten” into brevity. When the memos are thin, you still **infer structure** and write **explicit reasoning** (clearly labeled as interpretation vs. hard fact) while **never inventing** filing line items or numbers absent from the workbook JSON.

**Inputs (user message, in order):**
1) **Workbook JSON excerpt** — authoritative numbers (NPV, discount, FCF bridge, filing tags, consensus, last close, etc.). **Never contradict** these; every numeric table must tie back with cited logic.
2) **Headline annex** — canonical URLs; each material headline thread gets prose + `\footnote{\url{...}}`.
3) **Junior PRICE_TARGET / rating block** — use this **rating** and **USD** on page 1 **and** again in the valuation section with full methodology.
4) **Junior reasoning** + **The Critic** — merge into narrative; surface disagreements and how you resolved them.
5) **Original** and **final peer-reviewed research memos** — primary substance to **typeset and elaborate** (not to summarize away).

**Length (hard expectations):** Aim for **15–25+ pages** PDF when the source material supports it; **never** ship a note under **~12 pages** of substantive body (prose + tables/figures) except if the user payload is nearly empty—in that case still write **long** interpretive sections explaining data gaps and what would be needed next. Dense pages count: narrow margins (`geometry`), normal 11pt, full `\textwidth` tables.

**Mandatory document flow (LaTeX):**
1. `\documentclass[11pt]{article}` … load `geometry`, `setspace`, `booktabs`, `amsmath`, `longtable` (if needed), **xurl**, **hyperref** (after xurl). After `\begin{document}`, use `\onehalfspacing` (or `\setstretch{1.15}`) for a conventional research note layout.
2. Title block: issuer, ticker, **date**, “Equity research note” (or initiation), one-line **analyst desk** attribution (generic is fine).
3. **Page-1 framed callout** immediately after `\begin{document}` (before TOC): `\fbox{\parbox{0.97\textwidth}{...}}` containing, **in order**:
   - `\textbf{\Large RECOMMENDATION: <RATING>}` — exactly one of: Strong Buy, Buy, Weak Buy, Hold, Weak Sell, Sell, Strong Sell (from anchor when present).
   - `\textbf{\large 12-month price target: \$<usd>}` (two decimals; anchor USD).
   - Horizon line if anchor ≠ 12 months.
   - **Two to four sentences** thesis + key risk (glanceable only—**not** a substitute for the rest of the report).
4. `\tableofcontents` then `\newpage` — the body must follow **sell-side section order** below (use `\section` / `\subsection`; numbering on).

**Required body sections (each with multi-paragraph prose, not bullet lists alone):**
- **Investment thesis & variants** — bull / base / bear narrative; what would prove you wrong.
- **Company overview & business model** — how they make money; segments if inferable from memos/workbook.
- **Industry context & competition** — where the company sits; only factual claims grounded in memos or generic industry knowledge framed as such.
- **Recent developments & news flow** — chronological or thematic treatment; **for each important story**, state what the source said, **how** you weighed it, and **how** it affects the thesis (footnotes to annex URLs).
- **Financial analysis** — tables from workbook (DCF inputs, FCF path, discount, NPV bridge); **before and after each table**, `\paragraph{Interpretation}` (or equivalent) explaining **line-by-line** how numbers were read and what they imply for cash generation and risk.
- **Valuation** — tie to workbook bridge; discuss multiples only if memos/data support; otherwise focus on the provided bridge. **Sensitivities**: at least a small scenario table (WACC / growth / margin) with prose on how each lever moves value—use workbook anchors where possible, label hypothetical cells clearly.
- **Price target & methodology** — repeat **rating + numeric PT**; long-form: anchors, consensus vs. model, adjustments, confidence, **chain of reasoning from evidence → assumption → output**.
- **Catalysts & risks** — idiosyncratic and market; legal, regulatory, execution.
- **ESG / governance** — brief subsection if memos mention; else one honest “not covered in source data” paragraph (no fabrication).
- **Data sources & methodology appendix** — explain **each** input class: SEC / filings excerpt in JSON, market data fields, SearXNG headlines, internal NPV bridge—**how** each informed sections above and **limits** of each source.

**Anti-summary rules:** Do not replace sections with bullet decks. Do not use a one-page “Executive summary” as the main content—if you include `\section*{Executive overview}`, cap it at **~1 page** and **repeat** the same arguments **in depth** later in the body. Every major conclusion appears **at least twice**: once in overview, once in the dedicated section with evidence.

**Citations:** Dense `\footnote{...}` with `\url{...}` for web/headlines; workbook/filings cited in prose (“per workbook excerpt, …”). Final `\section{References}` listing **all** URLs cited.

**Output delimiters (exactly):**

<<<SYNTHESIS>>>
14–22 sentences: section-by-section map, how memos were expanded (not summarized), how Critic points were addressed, citation strategy, and residual uncertainties.
<<<END_SYNTHESIS>>>

<<<LATEX>>>
\documentclass[11pt]{article}
...
<<<END_LATEX>>>

Professional sell-side tone. Include a clear **not investment advice / for professional clients** disclaimer near the end."""

_MAX_MEMO = 450_000
_MAX_MEMO_ORIGINAL = 200_000
_MAX_CRITIC = 500_000
_MAX_REASONING = 500_000
_MAX_HEADLINE_ANNEX = 100_000
_MAX_WORKBOOK_EXCERPT = 95_000


def _build_headline_annex_plaintext(
    headlines: list[dict[str, Any]] | None,
    *,
    max_items: int = 40,
) -> str:
    """Stable plain-text block so Lead PM can cite canonical SearXNG URLs after peer review."""
    if not headlines:
        return ""
    chunks: list[str] = []
    for i, raw in enumerate(headlines[:max_items], 1):
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title") or "Untitled").strip()
        url = str(raw.get("url") or "").strip()
        content = str(raw.get("content") or "").strip().replace("\n", " ")
        if len(content) > 420:
            content = content[:417].rstrip() + "…"
        line = f"{i}. {title}"
        if url:
            line += f"\n   URL: {url}"
        if content:
            line += f"\n   Snippet: {content}"
        chunks.append(line)
    body = "\n\n".join(chunks).strip()
    if len(body) > _MAX_HEADLINE_ANNEX:
        body = body[:_MAX_HEADLINE_ANNEX] + "\n\n[… headline annex truncated …]\n"
    return body


def run_lead_pm_final(
    *,
    api_key: str,
    model: str,
    ticker: str,
    company_name: str,
    research_memo: str,
    research_memo_original: str | None,
    junior_reasoning: str | None,
    critic_memo: str,
    data_json_excerpt: str,
    price_anchor_block: str,
    headlines: list[dict[str, Any]] | None = None,
    emit: Callable[[dict[str, Any]], None] | None = None,
    max_attempts: int = 8,
) -> tuple[str, str | None, str]:
    """Returns ``(final_latex, synthesis_or_none, error_message)``.

    The Lead PM compiles **LaTeX only** from peer-reviewed **plain-text** memos plus workbook excerpt.

    Retries transient API failures (e.g. **503** / 429) with exponential backoff
    (``gemini_retry_sleep_seconds``). Override attempt count with env
    ``LEAD_PM_GEMINI_MAX_ATTEMPTS`` (clamped 3–15) or the ``max_attempts`` argument when env is unset.
    """
    key = (api_key or "").strip()
    if not key:
        return "", None, "no_api_key"

    memo = (research_memo or "").strip()
    if len(memo) > _MAX_MEMO:
        memo = memo[:_MAX_MEMO] + "\n\n[… junior memo truncated (safety cap) …]\n"

    crit = (critic_memo or "").strip()
    if len(crit) > _MAX_CRITIC:
        crit = crit[:_MAX_CRITIC] + "\n[… critic truncated …]\n"

    reason = (junior_reasoning or "").strip()
    if len(reason) > _MAX_REASONING:
        reason = reason[:_MAX_REASONING] + "\n[… truncated …]\n"

    orig = (research_memo_original or "").strip()
    if orig and len(orig) > _MAX_MEMO_ORIGINAL:
        orig = orig[:_MAX_MEMO_ORIGINAL] + "\n[… original junior memo truncated …]\n"

    orig_block = ""
    if orig:
        orig_block = (
            f"--- Junior Researcher **original** research memo (first desk pass; content anchor) ---\n{orig}\n\n"
        )

    wb = (data_json_excerpt or "").strip()
    if len(wb) > _MAX_WORKBOOK_EXCERPT:
        wb = wb[:_MAX_WORKBOOK_EXCERPT] + "\n[… workbook excerpt truncated …]\n"
    workbook_block = (
        "--- Workbook excerpt (JSON; authoritative numbers for LaTeX—do not contradict) ---\n"
        f"{wb or '(none)'}\n\n"
    )

    anchor = (price_anchor_block or "").strip()
    anchor_block = (
        "--- Junior PRICE_TARGET / rating (plain text from JSON pass—use on page 1 callout) ---\n"
        f"{anchor or '(none)'}\n\n"
    )

    annex = _build_headline_annex_plaintext(headlines)
    annex_block = (
        "--- Headline scan (factual annex; canonical URLs for footnotes; do not invent URLs) ---\n"
        f"{annex or '(no headlines returned for this run)'}\n\n"
    )

    user = (
        f"Final note issuer: {company_name} ({ticker})\n\n"
        f"{workbook_block}"
        f"{annex_block}"
        f"{anchor_block}"
        f"--- Junior internal reasoning ---\n{reason or '(none)'}\n\n"
        f"--- The Critic (all peer-review rounds; integrate holistically) ---\n{crit or '(none)'}\n\n"
        f"{orig_block}"
        f"--- Junior **latest** research memo after peer dialogue (primary narrative to typeset) ---\n"
        f"{memo or '(empty)'}\n\n"
        "--- Lead PM delivery checklist (obey in <<<LATEX>>> output) ---\n"
        "1) Follow the system prompt’s **sell-side section order** after \\tableofcontents; the page-1 callout is only a **header**, not the report.\n"
        "2) **Minimum ~12 pages** substantive PDF body in normal compilation; prefer **15–25+** when memos + workbook are rich—expand with interpretation, "
        "not padding.\n"
        "3) Every workbook table: prose **before** (what the reader is about to see) and **after** (what changed vs priors, what it implies).\n"
        "4) Include the **Data sources & methodology appendix** explaining how each input type fed conclusions.\n"
    )

    raw_attempts = (os.environ.get("LEAD_PM_GEMINI_MAX_ATTEMPTS") or "").strip()
    attempts_cap = max_attempts
    if raw_attempts:
        try:
            attempts_cap = int(raw_attempts)
        except ValueError:
            attempts_cap = max_attempts
    attempts_cap = max(3, min(int(attempts_cap), 15))

    client = genai.Client(api_key=key)
    cfg = types.GenerateContentConfig(
        system_instruction=LPM_SYSTEM,
        temperature=0.2,
        max_output_tokens=65536,
    )
    last_err = ""
    for attempt in range(max(1, attempts_cap)):
        try:
            emit_gemini_request(emit, call_site="lead_pm_final")
            response = client.models.generate_content(
                model=model,
                contents=user,
                config=cfg,
            )
            raw = (response.text or "").strip()
            if not raw:
                return "", None, "empty_lead_pm_response"
            syn = _extract_delimited(raw, "<<<SYNTHESIS>>>", "<<<END_SYNTHESIS>>>")
            syn = syn.strip() if syn else None
            latex = _extract_delimited(raw, "<<<LATEX>>>", "<<<END_LATEX>>>")
            if latex and "\\documentclass" in latex:
                return latex.strip(), syn, ""
            fb = _latex_from_fenced(raw)
            if fb:
                return fb.strip(), syn, ""
            return "", syn, "lead_pm_missing_latex_delimiters"
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            if attempt == attempts_cap - 1 or not is_transient_gemini_api_error(exc):
                return "", None, f"lead_pm_model_error: {exc}"
            time.sleep(gemini_retry_sleep_seconds(attempt))
    return "", None, last_err or "lead_pm_model_error"
