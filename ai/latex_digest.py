"""Turn Junior LaTeX into a compact plain-text representation for downstream LLM calls (e.g. The Critic).

The Lead Portfolio Manager still receives the original ``.tex``; this module is for token-efficient
**semantic** context: same narrative, tables, and numbers, without preamble noise or most markup.
"""

from __future__ import annotations

import re
from typing import NamedTuple


class LatexDigestResult(NamedTuple):
    """Plain-text digest plus bookkeeping for logging / limits."""

    digest: str
    raw_chars: int
    body_chars: int
    digest_chars: int
    truncated: bool


def extract_document_body(tex: str) -> str:
    """Return text between ``\\begin{document}`` and ``\\end{document}``, or full ``tex`` if missing."""
    s = (tex or "").strip()
    if not s:
        return ""
    low = s.lower()
    key = r"\begin{document}"
    i = low.find(key)
    if i < 0:
        return s
    rest = s[i + len(key) :]
    low2 = rest.lower()
    j = low2.find(r"\end{document}")
    if j >= 0:
        rest = rest[:j]
    return rest.strip()


def _strip_percent_comment(line: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "%":
            j = i - 1
            bs = 0
            while j >= 0 and line[j] == "\\":
                bs += 1
                j -= 1
            if bs % 2 == 0:
                break
        out.append(ch)
        i += 1
    return "".join(out).rstrip()


def _strip_comments(s: str) -> str:
    return "\n".join(_strip_percent_comment(L) for L in s.splitlines())


def _flatten_tabular(match: re.Match[str]) -> str:
    inner = match.group(1)
    inner = re.sub(
        r"\\hline|\\toprule|\\midrule|\\bottomrule|\\cline\{[^}]+\}",
        "\n",
        inner,
        flags=re.IGNORECASE,
    )
    inner = re.sub(
        r"\\multicolumn\{[^}]+\}\{[^}]+\}\{((?:[^{}]|\{[^{}]*\})*)\}",
        r"\1",
        inner,
    )
    inner = inner.replace("&", " | ")
    inner = re.sub(r"\\\\\*?\s*", "\n", inner)
    inner = re.sub(r"[ \t\f\v]+\n", "\n", inner)
    inner = re.sub(r"\n{3,}", "\n\n", inner).strip()
    if not inner:
        return "\n[TABLE (empty)]\n"
    return f"\n[TABLE]\n{inner}\n[/TABLE]\n"


def _unwrap_simple_brace_commands(s: str, command: str) -> str:
    pat = re.compile(rf"\\{re.escape(command)}\{{([^{{}}]*)\}}")
    prev = None
    while prev != s:
        prev = s
        s = pat.sub(r"\1", s)
    return s


def latex_body_to_plaintext_digest(body: str) -> str:
    """Heuristic conversion: sections, lists, simple emphasis, tabulars; not a full TeX parser."""
    s = _strip_comments(body)

    s = re.sub(
        r"\\begin\{tabular\}\s*\{[^}]*\}([\s\S]*?)\\end\{tabular\}",
        _flatten_tabular,
        s,
        flags=re.IGNORECASE,
    )

    s = re.sub(r"\\item\s+", "\n- ", s)
    for env in ("itemize", "enumerate", "description"):
        s = re.sub(rf"\\begin\{{{env}\}}", "\n", s, flags=re.IGNORECASE)
        s = re.sub(rf"\\end\{{{env}\}}", "\n", s, flags=re.IGNORECASE)

    s = re.sub(r"\\section\*?\{([^}]*)\}", r"\n\n## \1\n", s)
    s = re.sub(r"\\subsection\*?\{([^}]*)\}", r"\n### \1\n", s)
    s = re.sub(r"\\subsubsection\*?\{([^}]*)\}", r"\n#### \1\n", s)

    for cmd in ("textbf", "textit", "emph", "textsc", "texttt", "underline"):
        s = _unwrap_simple_brace_commands(s, cmd)

    s = re.sub(r"\\href\{([^}]*)\}\{((?:[^{}]|\{[^{}]*\})*)\}", r"\2 <\1>", s)
    s = re.sub(r"\\url\{([^}]*)\}", r"<\1>", s)

    s = re.sub(
        r"\\(tiny|scriptsize|footnotesize|small|normalsize|large|Large|LARGE|huge|Huge)\b\s*",
        "",
        s,
    )
    s = re.sub(
        r"\\(centering|raggedright|flushleft|flushright|noindent|indent)\b\s*",
        "",
        s,
    )
    s = re.sub(r"\\(newline|linebreak|par)\b\s*", "\n", s)
    s = re.sub(r"\\\\\*?(\s*\[[^]]*\])?\s*", "\n", s)

    # Drop common float wrappers but keep inner caption text if simple
    s = re.sub(r"\\begin\{figure\}([\s\S]*?)\\end\{figure\}", r"\1\n", s, flags=re.IGNORECASE)
    s = re.sub(r"\\begin\{table\}([\s\S]*?)\\end\{table\}", r"\1\n", s, flags=re.IGNORECASE)
    s = re.sub(r"\\centering\b\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}", r"\n[Caption: \1]\n", s)

    def _format_math_inner(inner_raw: str) -> str:
        inner = inner_raw.strip()
        inner = re.sub(r"\s+", " ", inner)
        if len(inner) <= 220:
            return f"\n[math: {inner}]\n"
        return f"\n[math: {inner[:200]} … ({len(inner)} chars)]\n"

    # Pull display math out before the aggressive ``\cmd`` strip (otherwise ``\sum`` etc. vanish).
    math_chunks: list[str] = []

    def _pull_display(m: re.Match[str]) -> str:
        math_chunks.append(_format_math_inner(m.group(1)))
        return f"\n<<DIGEST_MATH_{len(math_chunks) - 1}>>\n"

    s = re.sub(r"\\\[([\s\S]*?)\\\]", _pull_display, s)
    s = re.sub(r"\$\$([\s\S]*?)\$\$", _pull_display, s)

    s = re.sub(r"\\[a-zA-Z@]+(\[[^]]*\])?(\{[^}]*\})?", " ", s)
    s = re.sub(r"[{}]", "", s)
    s = re.sub(r"\\[,;:!]", " ", s)

    for i, chunk in enumerate(math_chunks):
        s = s.replace(f"<<DIGEST_MATH_{i}>>", chunk.strip())

    s = re.sub(r"[ \t\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def build_critic_latex_digest(junior_latex: str, *, max_digest_chars: int = 450_000) -> LatexDigestResult:
    """Full substantive content as plain text; optional tail cap for extreme edge cases."""
    raw = junior_latex or ""
    body = extract_document_body(raw)
    digest = latex_body_to_plaintext_digest(body)
    truncated = False
    if len(digest) > max_digest_chars:
        head = max_digest_chars // 2
        tail = max_digest_chars - head - 120
        digest = (
            digest[:head]
            + "\n\n[… digest middle omitted to respect context limit; "
            "beginning and end of note preserved …]\n\n"
            + digest[-tail:]
        )
        truncated = True
    note = (
        "[Below is the Junior equity note converted from LaTeX **body** to plain text: "
        "section structure, lists, tables, and prose are preserved; preamble/macros removed "
        "to save tokens. Use this together with the internal reasoning block.]\n\n"
    )
    return LatexDigestResult(
        digest=note + digest,
        raw_chars=len(raw),
        body_chars=len(body),
        digest_chars=len(note + digest),
        truncated=truncated,
    )
