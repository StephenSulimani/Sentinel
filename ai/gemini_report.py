"""Gemini-based LaTeX + analyst reasoning (delimiter format avoids JSON/LaTeX escape bugs)."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from typing import Any

from google import genai
from google.genai import types

SYSTEM_INSTRUCTION = """You are a senior sell-side equity research analyst at a major investment bank covering US equities. Write as you would for a managing director and institutional clients: confident but careful, skeptical, second-order thinking, strictly grounded in the research workbook JSON supplied in the user message (your factual ground truth for numbers, dates, and tags).

Produce an **extended** institutional-quality note. Aim for depth comparable to a published initiation or update (longer sections, concrete bullets, explicit gaps where data is thin).

Voice and disclosure: In REASONING and in the LaTeX body, **never** name internal tools, codebases, orchestration layers, meta-search products, or programming languages. Do not use the phrase DATA_JSON or call out “the JSON bundle.” Speak naturally: e.g. company filings (SEC EDGAR where applicable), market price history, headline scan, illustrative discounted cash-flow bridge, consensus estimates when present. You may cite SEC EDGAR and standard market data conventions where a professional would.

The workbook's news section lists the **search queries that were run** plus merged **general business and market headline** snippets (a discovery pass, not a filing drill or verified newsroom). Synthesise cross-query themes, flag contradictions or thin coverage, and spell out how you would **go deeper on a real desk**—extra datasets, expert checks, local outlets, supply chain or regulatory channels, management access, and what would change your mind—without naming internal pipeline stages.

The workbook includes a single **inferred** discount assumption and a plain-language rationale for the mechanical five-year cash-flow illustration only. Paraphrase that rationale for clients (do not paste raw workbook keys). You should still state your own cost-of-equity or WACC judgment in the narrative if you would choose a different assumption, and explain how that affects relative valuation versus the illustration.

Work in three explicit parts in your reply (exact delimiters, ASCII only):

1) REASONING (plain English, no LaTeX): **at least 8 dense paragraphs** covering, where the workbook allows: (a) business model and revenue quality; (b) balance sheet, cash, and capital returns if visible in filing-derived facts; (c) recent price action versus fundamentals; (d) filing and facts highlights (cite tags and periods; no invented line items); (e) **news and narrative**—integrate headline themes across the planned search angles, headline versus fundamental risk, and what primary-source work you would do next; (f) the illustrative cash-flow valuation bridge—state clearly it is a teaching / sensitivity illustration, not a fair-value opinion, unless the workbook clearly supports more; (g) key debates, bull versus bear; (h) data limitations and what you would request next with full data access.
2) PRICE_TARGET: a single JSON object. Field 'usd' MUST be a positive float (USD per share). Prefer anchoring to price_target_seed.consensus_mean_usd and reference_last_close_usd when present; explain adjustments in 'method' (multi-sentence allowed in JSON string). Never invent filing facts.
3) LATEX: a **long** pdflatex-ready document (target **several pages** of substantive prose + tables). Use \\documentclass[11pt]{article}, geometry, hyperref, booktabs, amsmath as needed; prefer packages from texlive-latex-recommended / texlive-latex-extra.

LaTeX must include these **\\section** headings in order (omit only if truly no content, then say so explicitly):
- Executive summary
- Company snapshot and data map (what is evidenced in the workbook versus missing)
- Operating and financial context (filing-derived facts and statement data; no fabricated KPIs)
- Market context and price action
- Recent developments and narrative (headline scan; verify primary sources)
- Valuation bridge and cash-flow illustration (assumptions table; not a published fair value)
- **Price target** (must mirror PRICE_TARGET JSON: USD, horizon, methodology, downside)
- Risks, catalysts, and bear case
- Data sources, methodology, and limitations
- Disclaimer (not investment advice)

Hard rules:
- Do not invent numbers or filing facts not present in the supplied workbook JSON.
- If data is missing, say so; do not fabricate.
- LaTeX must start with \\documentclass (single backslash in the final document body).
- In client-facing prose (REASONING and LaTeX), do not mention Gemini, vendor model names, raw JSON field names, or implementation stack details—translate the workbook into normal research language.
- Use the planned headline queries only as **intellectual scaffolding** (angles explored); do not quote raw query strings verbatim unless a professional would repeat a search line in a memo.

Output format (exactly, no markdown fences, no JSON except inside PRICE_TARGET):

<<<REASONING>>>
...plain text...
<<<END_REASONING>>>
<<<PRICE_TARGET>>>
{"usd": 100.0, "horizon_months": 12, "method": "one-line methodology string"}
<<<END_PRICE_TARGET>>>
<<<LATEX>>>
\\documentclass{article}
...
<<<END_LATEX>>>

The PRICE_TARGET JSON must be valid JSON. 'horizon_months' is typically 12.

Disclaimer in LaTeX: not investment advice; cite sources in conventional sell-side terms (e.g. SEC EDGAR for filings and company facts, market data for prices, headline scan for news snippets, illustrative cash-flow model where used)."""

REPAIR_LATEX_SYSTEM = """You fix pdflatex build failures for TeX Live on Debian.

The installation includes texlive-latex-recommended, texlive-latex-extra, and lmodern (article, geometry, hyperref, amsmath, amssymb, amsfonts, graphicx, booktabs, siunitx, xurl, microtype, etc.). Prefer packages that exist there; replace or drop unsupported \\usepackage lines. Do not change facts, numbers, or narrative meaning—only preamble/macros and LaTeX syntax so the file compiles.

Output **one** of these (in order of preference):
1) Delimited block (ASCII only):
<<<LATEX>>>
\\documentclass{article}
...
<<<END_LATEX>>>
2) Or a single Markdown ```latex fenced block containing the **entire** document.
3) Or the raw full document starting with \\documentclass and ending with \\end{document} with nothing before or after.

The body must be a complete pdflatex-ready document starting with \\documentclass."""


def _extract_delimited(text: str, start: str, end: str) -> str | None:
    if start not in text or end not in text:
        return None
    try:
        body = text.split(start, 1)[1].split(end, 1)[0]
    except IndexError:
        return None
    return body.strip()


def _parse_price_target_json(raw: str | None) -> dict[str, Any] | None:
    if not raw or not raw.strip():
        return None
    try:
        obj = json.loads(raw.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def _trim_suffix_if_prefix_of_tag(s: str, tag: str) -> str:
    """Drop a trailing substring that is a strict prefix of ``tag`` (delimiter may span chunks)."""
    for k in range(len(tag) - 1, 0, -1):
        if s.endswith(tag[:k]):
            return s[:-k]
    return s


def _reasoning_stream_update(
    buffer: str,
    emitted_len: int,
    stream_reasoning: Callable[[str], None],
) -> int:
    """Emit new reasoning text since ``emitted_len``; return new emitted character count."""
    start_tag = "<<<REASONING>>>"
    end_tag = "<<<END_REASONING>>>"
    if start_tag not in buffer:
        return emitted_len
    i = buffer.index(start_tag) + len(start_tag)
    if end_tag in buffer:
        j = buffer.index(end_tag)
        segment = buffer[i:j]
    else:
        segment = _trim_suffix_if_prefix_of_tag(buffer[i:], end_tag)
    if emitted_len > len(segment):
        emitted_len = len(segment)
    if len(segment) > emitted_len:
        stream_reasoning(segment[emitted_len:])
    return len(segment)


def _latex_from_fenced(text: str) -> str | None:
    best: str | None = None
    for m in re.finditer(r"```(?:latex)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE):
        body = m.group(1).strip()
        if "\\documentclass" not in body:
            continue
        if best is None or len(body) > len(best):
            best = body
    return best


def _latex_from_model_repair_text(text: str) -> str | None:
    """Accept delimiter block, ```latex``` fence, or raw document from model output."""
    raw = text.strip()
    if not raw:
        return None
    fixed = _extract_delimited(raw, "<<<LATEX>>>", "<<<END_LATEX>>>")
    if fixed and "\\documentclass" in fixed:
        return fixed.strip()
    fb = _latex_from_fenced(raw)
    if fb:
        return fb.strip()
    # Strip a single leading/trailing markdown wrapper if fence was unclosed
    t = raw
    if t.startswith("```"):
        lines = t.split("\n")
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        t = "\n".join(lines).strip()
        if t.endswith("```"):
            t = t.rsplit("```", 1)[0].strip()
    i = t.find("\\documentclass")
    if i < 0:
        return None
    tail = t[i:].strip()
    end_mark = "\\end{document}"
    j = tail.find(end_mark)
    if j >= 0:
        tail = tail[: j + len(end_mark)]
    return tail.strip() if "\\documentclass" in tail else None


def generate_latex_from_context(
    *,
    api_key: str,
    model: str,
    data_json: str,
    stream_reasoning: Callable[[str], None] | None = None,
) -> tuple[str, str, dict[str, Any] | None, str]:
    """Returns (latex, reasoning, price_target_dict_or_none, error_message).

    When ``stream_reasoning`` is set, uses the streaming API and forwards reasoning
    body text incrementally (after ``<<<REASONING>>>``, before ``<<<END_REASONING>>>``).
    """
    client = genai.Client(api_key=api_key)
    prompt = (
        "Research workbook (JSON; factual ground truth—do not echo this header in your note):\n"
        + data_json
        + "\n\nFollow the delimiter output contract in the system instructions.\n"
    )
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.35,
        max_output_tokens=24576,
    )
    text = ""
    max_attempts = 10
    if stream_reasoning is not None:
        for attempt in range(max_attempts):
            emitted = 0
            text = ""
            try:
                stream = client.models.generate_content_stream(
                    model=model,
                    contents=prompt,
                    config=config,
                )
                for chunk in stream:
                    piece = ""
                    try:
                        piece = chunk.text or ""
                    except Exception:  # noqa: BLE001
                        piece = ""
                    if not piece:
                        continue
                    text += piece
                    emitted = _reasoning_stream_update(text, emitted, stream_reasoning)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == max_attempts - 1 or not is_transient_gemini_api_error(exc):
                    return "", "", None, f"gemini_request_error: {exc}"
                time.sleep(min(60.0, 2.0 * (2**attempt)))
    else:
        response = None
        for attempt in range(max_attempts):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == max_attempts - 1 or not is_transient_gemini_api_error(exc):
                    return "", "", None, f"gemini_request_error: {exc}"
                time.sleep(min(60.0, 2.0 * (2**attempt)))
        if response is None:
            return "", "", None, "gemini_request_error: no response after retries"

        try:
            text = (response.text or "").strip()
        except Exception as exc:  # noqa: BLE001
            return "", "", None, f"blocked_or_empty_response: {exc}"

    text = text.strip()
    if not text:
        return "", "", None, "empty Gemini response"

    reasoning = _extract_delimited(text, "<<<REASONING>>>", "<<<END_REASONING>>>") or ""
    pt_raw = _extract_delimited(text, "<<<PRICE_TARGET>>>", "<<<END_PRICE_TARGET>>>")
    price_target = _parse_price_target_json(pt_raw)
    latex = _extract_delimited(text, "<<<LATEX>>>", "<<<END_LATEX>>>")

    if latex and "\\documentclass" in latex:
        return latex.strip(), reasoning.strip(), price_target, ""

    # Fallback: fenced LaTeX block only
    fb = _latex_from_fenced(text)
    if fb:
        return (
            fb,
            reasoning.strip() or "Model returned fenced LaTeX without delimiter blocks.",
            price_target,
            "",
        )

    return (
        "",
        reasoning.strip(),
        price_target,
        "missing LATEX delimiters or \\documentclass in model output",
    )


def is_transient_gemini_error_message(message: str) -> bool:
    """Same heuristics as ``is_transient_gemini_api_error`` for plain error strings."""
    return is_transient_gemini_api_error(RuntimeError(message))


def is_transient_gemini_api_error(exc: BaseException) -> bool:
    """True when the request should be retried after a short backoff."""
    code = getattr(exc, "code", None)
    if code in (429, 503):
        return True
    status = getattr(exc, "status_code", None)
    if status in (429, 503):
        return True
    s = str(exc).lower()
    return any(
        token in s
        for token in (
            "503",
            "429",
            "unavailable",
            "resource_exhausted",
            "deadline exceeded",
            "high demand",
            "overload",
            "too many requests",
            "try again later",
            "temporarily",
            "econnreset",
            "connection reset",
            "timeout",
        )
    )


def repair_latex_for_pdflatex_log(
    *,
    api_key: str,
    model: str,
    latex_source: str,
    pdflatex_log_tail: str,
    max_api_retries: int = 10,
) -> tuple[str, str]:
    """Return ``(fixed_latex, error)``. ``error`` empty on success.

    Retries transient provider errors (503 / rate limits / overload) with backoff
    before giving up.
    """
    client = genai.Client(api_key=api_key)
    prompt = (
        "BROKEN_LATEX_SOURCE (trimmed if long):\n"
        + latex_source[:180_000]
        + "\n\nPDFLATEX_LOG_TAIL:\n"
        + (pdflatex_log_tail[-14_000:] if pdflatex_log_tail else "(no log)")
        + "\n\nApply REPAIR_LATEX_SYSTEM: emit only <<<LATEX>>>...<<<END_LATEX>>>.\n"
    )
    response = None
    last_exc: str | None = None
    attempts = max(1, min(int(max_api_retries), 20))
    for attempt in range(attempts):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=REPAIR_LATEX_SYSTEM,
                    temperature=0.15,
                    max_output_tokens=16384,
                ),
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = str(exc)
            if attempt == attempts - 1 or not is_transient_gemini_api_error(exc):
                return "", f"gemini_repair_request_error: {exc}"
            delay = min(60.0, 2.0 * (2**attempt))
            time.sleep(delay)
    if response is None:
        return "", f"gemini_repair_request_error: {last_exc or 'no response'}"

    try:
        text = (response.text or "").strip()
    except Exception as exc:  # noqa: BLE001
        return "", f"gemini_repair_empty: {exc}"

    if not text:
        return "", "empty Gemini repair response"

    fixed = _latex_from_model_repair_text(text)
    if fixed:
        return fixed, ""

    return "", "repair response missing a complete LaTeX document (\\documentclass … \\end{document})"
