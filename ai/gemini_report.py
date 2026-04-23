"""Gemini-based LaTeX + analyst reasoning (delimiter format avoids JSON/LaTeX escape bugs)."""

from __future__ import annotations

import builtins
import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any, Literal

from google import genai
from google.genai import types

from gemini_error_extras import gemini_error_log_suffix
from gemini_stream_events import emit_gemini_request
from gemini_pacing import (
    gemini_retry_sleep_seconds,
    optional_gemini_preflight_delay,
    sleep_between_gemini_tool_presets,
)

_log = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = """You are a senior sell-side equity research analyst at a major investment bank covering US equities. Write as you would for a managing director and institutional clients: confident but careful, skeptical, second-order thinking, strictly grounded in the research workbook JSON supplied in the user message (your factual ground truth for numbers, dates, and tags).

**Anti–black-box mandate:** A PM must never wonder *why* you concluded what you did. Every major section should mix **evidence** (what the workbook or a cited URL actually shows) with **explicit judgement** (how you weigh trade-offs, what you infer versus what is directly observed, and what would change your mind). Prefer longer explanatory paragraphs and short “**Logic:** …” or “**Why this matters:** …” lead-ins over telegraphic bullets. Name alternative hypotheses you rejected and why.

Produce an **extended** institutional-quality note—**verbose by design**, comparable to a published initiation or update: long sections, concrete bullets where they add clarity, explicit gaps where data is thin, and **no** compressed “AI summary” tone.

Voice and disclosure: In REASONING and in the LaTeX body, **never** name internal tools, codebases, orchestration layers, meta-search products, or programming languages. Do not use the phrase DATA_JSON or call out “the JSON bundle.” Speak naturally: e.g. company filings (SEC EDGAR where applicable), market price history, headline scan, illustrative discounted cash-flow bridge, consensus estimates when present. You may cite SEC EDGAR and standard market data conventions where a professional would.

The workbook's news section lists the **search queries that were run** (a desk headline-discovery pass via meta-search) plus merged **headline snippets** and **canonical article URLs** from that pass—not a filing drill or verified newsroom. When a **URL-reading tool** is available, use it **only on URLs already listed in the workbook** (or clearly tied to those headlines) to go deeper than snippets—verify claims and pull timely context **without** running a separate broad web search; discovery already happened upstream. **Filing-derived numbers and tags in the workbook remain authoritative**; do not replace them with different figures from the web. Web material supplements narrative, verification, and “what changed recently,” and must be described in normal client language (outlet or domain, approximate timing)—never raw tool names.

**Citations (mandatory in the LaTeX report):** Use LaTeX `\\footnote{...}` so the PDF shows **numbered superscript reference markers** in the body and the full citation text at the foot of the page—this is the house style for web-sourced or non-obvious interpretive claims (clients expect to see those markers frequently in “Recent developments,” risk, and narrative paragraphs).

Any material claim, statistic, quote, or interpretive statement that depends on the **public web** (search results or fetched article pages—not the workbook JSON itself) must carry such a **footnote at first substantive use** (repeat `\\footnotemark`/`\\footnotetext` only if you need the same source twice; otherwise a new footnote per distinct URL is fine). Each footnote must name the outlet or publisher, the piece title (short form is fine), the publication or “as listed” date when visible, and the canonical `https://...` URL inside `\\url{...}` (preamble: `\\usepackage{xurl}` and `\\usepackage[hidelinks]{hyperref}` with `hyperref` loaded **after** `xurl`).

**Density rule:** If the workbook’s headline list contains **three or more distinct article URLs**, aim for **at least twelve** substantive `\\footnote{...}` callouts in the LaTeX body tied to web/headline material (spread across “Recent developments,” risks, and narrative—do not cluster all footnotes in one paragraph). With fewer URLs, footnote **every** non-trivial web-derived sentence. Do **not** replace footnotes with a vague “see internet” sentence.

Filing-derived numbers should be tied in prose to **SEC EDGAR** (form and fiscal period when the workbook supplies them) without inventing accession numbers. Market levels should be attributed to **market data** and the workbook’s as-of context. Headline-scan themes should be labeled as such unless you verified them via a cited web primary. Do not invent URLs; only cite URLs you actually used or that appear in the workbook’s headline list.

Synthesise cross-query themes, flag contradictions or thin coverage, and spell out how you would **go deeper on a real desk**—extra datasets, expert checks, local outlets, supply chain or regulatory channels, management access, and what would change your mind—without naming internal pipeline stages.

The workbook includes a single discount assumption for the mechanical five-year cash-flow illustration—either a **Junior Researcher desk pass** (after headline scan and market context) with a short rationale, or a **deterministic seed** when no narrative model ran for that step—and a plain-language basis string. Paraphrase that rationale for clients (do not paste raw workbook keys). You should still state your own cost-of-equity or WACC judgment in the narrative if you would choose a different assumption, and explain how that affects relative valuation versus the illustration.

Work in three explicit parts in your reply (exact delimiters, ASCII only):

1) REASONING (plain English, no LaTeX): **at least 12 dense paragraphs** (more is welcome). Cover, where the workbook allows: (a) business model and revenue quality; (b) balance sheet, cash, and capital returns if visible in filing-derived facts; (c) recent price action versus fundamentals; (d) filing and facts highlights (cite tags and periods; no invented line items); (e) **news and narrative**—integrate headline themes across the planned search angles, headline versus fundamental risk, and what primary-source work you would do next—**briefly name the main source types or outlets** you relied on beyond the workbook (e.g. which domains or filings), without raw URLs unless helpful; (f) the illustrative cash-flow valuation bridge—state clearly it is a teaching / sensitivity illustration, not a fair-value opinion, unless the workbook clearly supports more; (g) key debates, bull versus bear; (h) data limitations and what you would request next with full data access. **Additionally** include: (i) an **epistemic trace**—for each major conclusion, which workbook fields or headline themes drove it; (j) what would **falsify** or weaken your view; (k) where you are **extrapolating** versus where the evidence is direct.

2) PRICE_TARGET: a single JSON object. Field 'usd' MUST be a positive float (USD per share). Prefer anchoring to price_target_seed.consensus_mean_usd and reference_last_close_usd when present; explain adjustments in 'method' using **several sentences** (verbose, transparent logic—no one-line black box). Never invent filing facts.

3) LATEX: a **long** pdflatex-ready document—target on the order of **8–15 printed pages** of substantive prose + tables when the workbook supports it (expand thin sections with interpretation, scenario discussion, and citation-backed narrative; do not pad with filler, but **err on the side of length** and explanation). Use \\documentclass[11pt]{article}, geometry, booktabs, amsmath, **xurl**, and **hyperref** (with `hyperref` after `xurl`) as needed; prefer packages from texlive-latex-recommended / texlive-latex-extra.

LaTeX must include these **\\section** headings in order (omit only if truly no content, then say so explicitly):
- Executive summary (must itself be **multi-paragraph** and name the main evidence anchors; no one-paragraph teaser)
- **How to read this note** (evidence hierarchy, footnote convention, and what is **judgement vs observation**—keep it substantive, not boilerplate)
- Company snapshot and data map (what is evidenced in the workbook versus missing)
- Operating and financial context (filing-derived facts and statement data; no fabricated KPIs)
- Market context and price action
- Recent developments and narrative (headline scan; verify primary sources; **dense `\\footnote` markers** for web-derived claims—readers should see frequent superscript numbers here)
- Valuation bridge and cash-flow illustration (assumptions table; not a published fair value; include **paragraph-level** explanation of how you map filing OCF to the illustration and why the discount choice is reasonable)
- **Price target** (must mirror PRICE_TARGET JSON: USD, horizon, methodology, downside)
- Risks, catalysts, and bear case
- Data sources, methodology, and limitations
- **References and sources cited** (complete, deduplicated list: every `\\footnote` source plus workbook-origin classes—SEC EDGAR, market data provider as described in the workbook, headline scan, illustrative model—each as a `\\item` in a `description` or `enumerate` environment; **every cited URL must appear here** with publisher and title)
- Disclaimer (not investment advice)

Hard rules:
- Do not invent numbers or filing facts not present in the supplied workbook JSON.
- If data is missing, say so; do not fabricate.
- LaTeX must start with \\documentclass (single backslash in the final document body).
- In client-facing prose (REASONING and LaTeX), do not mention Gemini, vendor model names, raw JSON field names, or implementation stack details—translate the workbook into normal research language.
- Use the planned headline queries only as **intellectual scaffolding** (angles explored); do not quote raw query strings verbatim unless a professional would repeat a search line in a memo.
- **No orphan web claims:** if the reader cannot tell where a non-workbook fact came from, add a footnote or remove the claim. The References section must reconcile with footnotes (same URLs, no extras you did not rely on).
- **Visible reasoning in the PDF body:** beyond footnotes, use `\\subsection` or `\\paragraph` headings such as “**Interpretation**” or “**Why this matters**” inside thicker sections so the logical chain is skimmable—do not bury the only explanation in REASONING while the LaTeX reads like a fact sheet.

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

Disclaimer in LaTeX: not investment advice; restate that readers should verify material facts against primary documents; acknowledge that web pages can change or disappear after the note date."""

# Junior desk pass **without LaTeX** — peer review + Lead PM produce the PDF later (token-efficient).
MEMO_JUNIOR_SYSTEM = """You are the **Junior Researcher** on a sell-side desk. The **Lead Portfolio Manager** (not you) will later turn notes into a full LaTeX PDF. Your job here is **substance only**: evidence, judgement, and a clear investment stance—**do not write LaTeX**, `\\documentclass`, or PDF layout.

Ground truth is the **research workbook JSON** in the user message (filings facts, market snapshot, headline scan URLs/snippets, valuation bridge numbers). Never invent filing facts.

**Forbidden:** LaTeX commands, preamble, `\\section`, tables as TeX, or “here is the LaTeX”.

**Required output shape** (exact delimiters, ASCII only):

1) **REASONING** (plain English, no LaTeX): **at least 10 dense paragraphs** on thesis, evidence chain, headline themes (name outlets/domains), valuation logic vs the **illustrative** bridge in the workbook, bull/bear, falsifiers, and data gaps. Be verbose—this is the main “anti–black-box” trace before the PM writes the client document.

2) **PRICE_TARGET** JSON between delimiters. Fields:
   - `usd` (float, USD per share, > 0)
   - `horizon_months` (int, typically 12)
   - `method` (string, **several sentences** of transparent logic)
   - `rating` (string, **exactly one** of: `Strong Buy`, `Buy`, `Weak Buy`, `Hold`, `Weak Sell`, `Sell`, `Strong Sell`) — must align with your thesis and PT vs spot/consensus when those appear in the workbook
   - `rating_rationale` (string, 2–5 sentences: why this label fits)

3) **MEMO** (plain English / light Markdown only): a **long** desk research memo the PM can typeset—target **roughly 5k–14k words** when the workbook is rich (shorter only if data is truly thin). Use `##` / `###` headings (not LaTeX `\\section`) in **sell-side order**: Investment thesis & variants; Company & business model; Industry & competition; Recent developments (per headline URL); Financial read-through vs workbook bridge; Valuation & PT logic (prose, numbers from workbook); Catalysts & risks; Data gaps & what would change the view; **Investment recommendation** (repeat rating + PT summary). Bullets are fine **inside** sections but each section needs **multi-paragraph** prose. **Name URLs from the headline list in prose** (Lead PM will convert to `\\footnote` later).

Voice: professional, skeptical, no internal tool names, no “DATA_JSON”. Not investment advice.

Output format (exactly):

<<<REASONING>>>
...plain text...
<<<END_REASONING>>>
<<<PRICE_TARGET>>>
{"usd": 0.0, "horizon_months": 12, "method": "...", "rating": "Buy", "rating_rationale": "..."}
<<<END_PRICE_TARGET>>>
<<<MEMO>>>
...plain text / markdown ...
<<<END_MEMO>>>
"""

_MEMO_NO_TOOLS_SUFFIX = (
    "\n\nNOTE: URL-reading tools may be off on this attempt. Still produce REASONING, PRICE_TARGET, and MEMO "
    "from the workbook and listed headline URLs/snippets only; cite URLs in prose for the Lead PM.\n"
)

REPAIR_LATEX_SYSTEM = """You fix pdflatex build failures for TeX Live on Debian.

The installation includes texlive-latex-recommended, texlive-latex-extra, and lmodern (article, geometry, hyperref, amsmath, amssymb, amsfonts, graphicx, booktabs, siunitx, xurl, microtype, etc.). Prefer packages that exist there; replace or drop unsupported \\usepackage lines. Do not change facts, numbers, or narrative meaning—only preamble/macros and LaTeX syntax so the file compiles. Preserve **all footnotes, \\url/\\href links, and the References and sources cited section** unless a syntax error forces a minimal rewrite of the same content.

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


def _gemini_research_tool_presets() -> list[list[types.Tool]]:
    """URL context only (read pages for SearXNG-supplied URLs), then **no tools** if empty/errors.

    Google Search is intentionally **not** used—headline discovery is SearXNG; Gemini deepens via URL context.
    """
    uc = getattr(types, "UrlContext", None)
    if uc is not None:
        return [
            [types.Tool(url_context=uc())],
            [],
        ]
    return [[]]


_NO_TOOLS_USER_SUFFIX = (
    "\n\nNOTE: The URL-reading tool is **not** enabled on this run. "
    "Rely exclusively on the workbook JSON and the headline snippets/URLs already produced by the "
    "desk's SearXNG headline pass above; do not claim you fetched pages you did not open. Still use "
    "the delimiter contract (REASONING, PRICE_TARGET, LATEX), the **verbose / anti–black-box** standards "
    "above, and **dense** `\\footnote{...}` markers with `\\url{...}` for every substantive web/headline "
    "claim—same footnote density rules as when URL context is on.\n"
)


def _preset_failure_try_next_preset(last_err: str, preset_index: int, num_presets: int) -> bool:
    """Empty/blocked model output on a tool preset: try the next preset instead of failing the whole graph."""
    if preset_index >= num_presets - 1:
        return False
    e = last_err.lower()
    return "blocked_or_empty" in e or "empty text" in e or "empty gemini" in e


def _tool_config_error_recoverable_without_url_context(exc: BaseException) -> bool:
    """True when retrying without UrlContext may succeed (e.g. older Gemini 2.0 builds)."""
    msg = str(exc).lower()
    return any(
        needle in msg
        for needle in (
            "url_context",
            "url context",
            "urlcontext",
            "invalid tools",
            "tool is not supported",
            "unsupported tool",
            "unknown tool",
        )
    )


def _headline_url_hints_from_workbook_json(data_json: str) -> str:
    """Bullet list of http(s) URLs from SearXNG headlines (max 20 for URL-context limits)."""
    try:
        bundle = json.loads(data_json)
    except json.JSONDecodeError:
        return ""
    searx = (bundle.get("sources") or {}).get("searxng") or {}
    headlines = searx.get("headlines") or []
    if not isinstance(headlines, list):
        return ""
    seen: set[str] = set()
    lines: list[str] = []
    for h in headlines:
        if not isinstance(h, dict):
            continue
        u = (h.get("url") or "").strip()
        if not u.startswith(("http://", "https://")):
            continue
        key = u.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {u}")
        if len(lines) >= 20:
            break
    if not lines:
        return ""
    return (
        "Headline-scan URLs from SearXNG (use the URL context tool **only** on these URLs where useful; "
        "respect the tool URL limit; skip paywalled or inaccessible pages):\n"
        + "\n".join(lines)
    )


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


def _extract_stream_chunk_text(chunk: Any) -> str:
    """Text from a stream chunk; avoids ``chunk.text`` raising on tool-only parts (google-genai)."""
    try:
        t = getattr(chunk, "text", None)
        if isinstance(t, str) and t:
            return t
    except Exception:  # noqa: BLE001
        pass
    pieces: list[str] = []
    try:
        for part in getattr(chunk, "parts", None) or []:
            txt = getattr(part, "text", None)
            if isinstance(txt, str) and txt:
                pieces.append(txt)
    except Exception:  # noqa: BLE001
        pass
    try:
        for cand in getattr(chunk, "candidates", None) or []:
            content = getattr(cand, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", None) or []:
                txt = getattr(part, "text", None)
                if isinstance(txt, str) and txt:
                    pieces.append(txt)
    except Exception:  # noqa: BLE001
        pass
    return "".join(pieces)


def _collect_generate_response_text(response: Any) -> str:
    """Best-effort full text from a ``generate_content`` response (incl. AFC history)."""
    try:
        t = (response.text or "").strip()
        if t:
            return t
    except Exception:  # noqa: BLE001
        pass
    t = _extract_stream_chunk_text(response).strip()
    if t:
        return t
    hist = getattr(response, "automatic_function_calling_history", None)
    if hist:
        parts_out: list[str] = []
        try:
            for turn in hist:
                for part in getattr(turn, "parts", None) or []:
                    txt = getattr(part, "text", None)
                    if isinstance(txt, str) and txt.strip():
                        parts_out.append(txt)
        except Exception:  # noqa: BLE001
            pass
        joined = "\n".join(parts_out).strip()
        if joined:
            return joined
    return ""


def _log_empty_gemini_response(response: Any, *, model: str) -> None:
    """Explain empty ``.text`` (blocked prompt, no candidates, etc.)."""
    try:
        pf = getattr(response, "prompt_feedback", None)
        cands = getattr(response, "candidates", None) or []
        hist = getattr(response, "automatic_function_calling_history", None)
        hlen = len(hist) if hist is not None else 0
        _log.warning(
            "gemini empty body: model=%r prompt_feedback=%r candidates=%s afc_history_turns=%s",
            model,
            pf,
            len(cands),
            hlen,
        )
        for i, c in enumerate(cands[:3]):
            _log.warning(
                "gemini empty body candidate[%s]: finish_reason=%r content=%r",
                i,
                getattr(c, "finish_reason", None),
                getattr(c, "content", None),
            )
    except Exception as exc:  # noqa: BLE001
        _log.debug("gemini empty body debug logging failed: %s", exc)


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
    emit: Callable[[dict[str, Any]], None] | None = None,
    narrative_output: Literal["latex_pdf", "research_memo"] = "latex_pdf",
) -> tuple[str, str, dict[str, Any] | None, str]:
    """Returns ``(body, reasoning, price_target_dict_or_none, error_message)``.

    When ``narrative_output`` is ``"latex_pdf"``, ``body`` is LaTeX. When it is
    ``"research_memo"``, ``body`` is the plain-text / Markdown **MEMO** (no LaTeX)—used
    so Junior + Critic avoid token-heavy TeX before the Lead PM compiles the PDF.

    When ``stream_reasoning`` is set, uses the streaming API: forwards preamble text
    before ``<<<REASONING>>>``, then text inside the reasoning delimiters. Chunks are
    parsed with ``_extract_stream_chunk_text`` because ``chunk.text`` can fail or stay
    empty when tools/AFC emit function-call parts. If the stream yields no usable text,
    falls back to one-shot ``generate_content`` (same config) and batches reasoning
    to the callback.

    Uses **URL context** when the SDK supports it (SearXNG-derived URLs in the prompt); falls back to
    **no tools** (workbook-only) if the API returns an empty body. **Google Search is not used.**
    """
    is_memo = narrative_output == "research_memo"
    client = genai.Client(api_key=api_key)
    url_block = _headline_url_hints_from_workbook_json(data_json)
    _log.info(
        "gemini narrative: model=%r mode=%s prompt_chars=%s url_hint_chars=%s stream=%s",
        model,
        narrative_output,
        len(data_json),
        len(url_block),
        stream_reasoning is not None,
    )
    tail = (
        "Produce the **RESEARCH_MEMO** delimiter contract (REASONING, PRICE_TARGET, MEMO). "
        "**No LaTeX.** The Lead Portfolio Manager will typeset the client PDF.\n"
        if is_memo
        else (
            "Follow the delimiter output contract in the system instructions. "
            "In LATEX, include footnoted citations for web-derived material and a full References section.\n"
        )
    )
    prompt_base = (
        "Research workbook (JSON; factual ground truth—do not echo this header in your note):\n"
        + data_json
        + (f"\n\n{url_block}" if url_block else "")
        + "\n\n"
        + tail
    )
    pre_stream_tags = (
        ("<<<MEMO>>>", "<<<PRICE_TARGET>>>", "<<<REASONING>>>")
        if is_memo
        else ("<<<LATEX>>>", "<<<PRICE_TARGET>>>", "<<<REASONING>>>")
    )
    prompt = prompt_base
    max_attempts = 10
    optional_gemini_preflight_delay()

    def try_stream(config: types.GenerateContentConfig) -> tuple[str, str]:
        assert stream_reasoning is not None
        for attempt in range(max_attempts):
            emitted_tag = 0
            pre_emitted = 0
            acc = ""
            chunks = 0
            nonempty = 0
            t0 = time.monotonic()
            try:
                _log.info(
                    "gemini stream open attempt=%s/%s model=%r",
                    attempt + 1,
                    max_attempts,
                    model,
                )
                emit_gemini_request(emit, call_site="narrative_stream")
                stream = client.models.generate_content_stream(
                    model=model,
                    contents=prompt,
                    config=config,
                )
                for chunk in stream:
                    chunks += 1
                    piece = _extract_stream_chunk_text(chunk)
                    if not piece:
                        continue
                    nonempty += 1
                    if nonempty == 1:
                        _log.info(
                            "gemini stream first text after %.1fs (chunk_index=%s)",
                            time.monotonic() - t0,
                            chunks,
                        )
                    if nonempty > 1 and nonempty % 200 == 0:
                        _log.debug(
                            "gemini stream progress: %s text chunks, acc_chars=%s",
                            nonempty,
                            len(acc),
                        )
                    acc += piece
                    if not any(tag in acc for tag in pre_stream_tags):
                        if len(acc) > pre_emitted:
                            stream_reasoning(acc[pre_emitted:])
                            pre_emitted = len(acc)
                    elif "<<<REASONING>>>" in acc:
                        emitted_tag = _reasoning_stream_update(
                            acc, emitted_tag, stream_reasoning
                        )
                _log.info(
                    "gemini stream done: chunks=%s nonempty_text_chunks=%s acc_chars=%s elapsed=%.1fs",
                    chunks,
                    nonempty,
                    len(acc),
                    time.monotonic() - t0,
                )
                if chunks == 0 or not acc.strip():
                    _log.warning(
                        "gemini stream produced no usable text (chunks=%s nonempty=%s acc_chars=%s); "
                        "falling back to generate_content",
                        chunks,
                        nonempty,
                        len(acc),
                    )
                    return try_generate(config)
                return acc, ""
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "gemini stream error attempt=%s after %.1fs: %s%s",
                    attempt + 1,
                    time.monotonic() - t0,
                    exc,
                    gemini_error_log_suffix(exc),
                    exc_info=_log.isEnabledFor(logging.DEBUG),
                )
                if attempt == max_attempts - 1 or not is_transient_gemini_api_error(exc):
                    return "", str(exc)
                time.sleep(gemini_retry_sleep_seconds(attempt))
        return "", "gemini_request_error: no response after retries"

    def try_generate(config: types.GenerateContentConfig) -> tuple[str, str]:
        for attempt in range(max_attempts):
            t0 = time.monotonic()
            try:
                _log.info(
                    "gemini generate_content attempt=%s/%s model=%r",
                    attempt + 1,
                    max_attempts,
                    model,
                )
                emit_gemini_request(emit, call_site="narrative_generate")
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "gemini generate_content error attempt=%s after %.1fs: %s%s",
                    attempt + 1,
                    time.monotonic() - t0,
                    exc,
                    gemini_error_log_suffix(exc),
                    exc_info=_log.isEnabledFor(logging.DEBUG),
                )
                if attempt == max_attempts - 1 or not is_transient_gemini_api_error(exc):
                    return "", str(exc)
                time.sleep(gemini_retry_sleep_seconds(attempt))
                continue
            t = _collect_generate_response_text(response)
            if not t:
                _log_empty_gemini_response(response, model=model)
                return "", "blocked_or_empty_response: empty text"
            if stream_reasoning is not None:
                r = _extract_delimited(t, "<<<REASONING>>>", "<<<END_REASONING>>>") or ""
                if r:
                    batch = 2000
                    for i in range(0, len(r), batch):
                        stream_reasoning(r[i : i + batch])
            _log.info(
                "gemini generate_content ok chars=%s elapsed=%.1fs",
                len(t),
                time.monotonic() - t0,
            )
            return t, ""
        return "", "gemini_request_error: no response after retries"

    text = ""
    last_err = ""
    tool_presets = _gemini_research_tool_presets()
    n_presets = len(tool_presets)
    for preset_i, tools in enumerate(tool_presets):
        if preset_i > 0:
            sleep_between_gemini_tool_presets()
        no_tool_suffix = (
            _MEMO_NO_TOOLS_SUFFIX if is_memo else _NO_TOOLS_USER_SUFFIX
        )
        prompt = prompt_base + (no_tool_suffix if not tools else "")
        _log.info(
            "gemini tool preset=%s builtin_tools=%s",
            preset_i,
            len(tools),
        )
        cfg_kw: dict[str, Any] = {
            "system_instruction": MEMO_JUNIOR_SYSTEM if is_memo else SYSTEM_INSTRUCTION,
            "temperature": 0.38,
            "max_output_tokens": 65536,
        }
        if tools:
            cfg_kw["tools"] = tools
        config = types.GenerateContentConfig(**cfg_kw)
        if stream_reasoning is not None:
            text, last_err = try_stream(config)
        else:
            text, last_err = try_generate(config)
        if not last_err:
            break
        _log.warning("gemini tool preset=%s failed: %s", preset_i, last_err[:500])
        if preset_i == 0 and _tool_config_error_recoverable_without_url_context(RuntimeError(last_err)):
            _log.info("gemini retrying without URL context tool")
            continue
        if _preset_failure_try_next_preset(last_err, preset_i, n_presets):
            _log.info(
                "gemini will try next preset after empty/blocked response (preset=%s/%s)",
                preset_i,
                n_presets,
            )
            continue
        return "", "", None, f"gemini_request_error: {last_err}"

    text = text.strip()
    if not text:
        return "", "", None, "empty Gemini response"

    reasoning = _extract_delimited(text, "<<<REASONING>>>", "<<<END_REASONING>>>") or ""
    pt_raw = _extract_delimited(text, "<<<PRICE_TARGET>>>", "<<<END_PRICE_TARGET>>>")
    price_target = _parse_price_target_json(pt_raw)

    if is_memo:
        memo = _extract_delimited(text, "<<<MEMO>>>", "<<<END_MEMO>>>")
        if memo and len(memo.strip()) >= 400:
            return memo.strip(), reasoning.strip(), price_target, ""
        return (
            "",
            reasoning.strip(),
            price_target,
            "missing MEMO delimiters or MEMO body too short (need substantive desk memo)",
        )

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


def _collect_related_exceptions(exc: BaseException) -> list[BaseException]:
    """Breadth-first walk of ``exc`` plus ``__cause__`` / ``__context__`` / ``ExceptionGroup`` (deduped)."""
    seen: set[int] = set()
    out: list[BaseException] = []
    stack: list[BaseException] = [exc]
    while stack:
        e = stack.pop()
        if id(e) in seen:
            continue
        seen.add(id(e))
        out.append(e)
        cause = getattr(e, "__cause__", None)
        if isinstance(cause, BaseException):
            stack.append(cause)
        ctx = getattr(e, "__context__", None)
        if isinstance(ctx, BaseException) and ctx is not cause:
            stack.append(ctx)
        eg_cls = getattr(builtins, "BaseExceptionGroup", None)
        if eg_cls is not None and isinstance(e, eg_cls):
            for sub in e.exceptions:
                if isinstance(sub, BaseException):
                    stack.append(sub)
    return out


def _intify_status_code(obj: object) -> int | None:
    """Best-effort int for HTTP / RPC numeric codes (handles some enums / string codes)."""
    if isinstance(obj, int):
        return obj
    if isinstance(obj, str) and obj.strip().isdigit():
        return int(obj.strip())
    try:
        if obj is not None:
            return int(obj)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        pass
    return None


def is_transient_gemini_api_error(exc: BaseException) -> bool:
    """True when the request should be retried after a short backoff."""
    chain = _collect_related_exceptions(exc)
    for e in chain:
        code = getattr(e, "code", None)
        ic = _intify_status_code(code)
        if ic in (429, 503):
            return True
        name = str(getattr(code, "name", "") or "").upper()
        if name in ("UNAVAILABLE", "RESOURCE_EXHAUSTED", "DEADLINE_EXCEEDED", "ABORTED"):
            return True
        status = getattr(e, "status_code", None)
        isc = _intify_status_code(status)
        if isc in (429, 503):
            return True
        resp = getattr(e, "response", None)
        if resp is not None:
            rsc = _intify_status_code(getattr(resp, "status_code", None))
            if rsc in (429, 503):
                return True
    blob = " ".join(str(e).lower() for e in chain)
    return any(
        token in blob
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
            "internal server error",
            "service unavailable",
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
            time.sleep(gemini_retry_sleep_seconds(attempt))
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
