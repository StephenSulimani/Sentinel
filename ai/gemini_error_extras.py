"""Extract structured quota hints from Gemini / Generative Language API errors.

Google often returns ``google.rpc.QuotaFailure`` inside ``error.details`` with
``violations[].quotaMetric`` / ``quotaId`` (RPM vs TPM vs RPD, free-tier vs paid).
When the API only returns a generic ``Help`` link, this module returns an empty
diagnostic string—there is nothing machine-parseable in the payload.
"""

from __future__ import annotations

import ast
import json
from typing import Any


def _balanced_brace_slice(s: str) -> str | None:
    """If ``s`` starts with ``{``, return the shortest balanced ``{...}`` prefix."""
    if not s.startswith("{"):
        return None
    depth = 0
    for j, ch in enumerate(s):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[: j + 1]
    return None


def _payload_from_exception_string(exc: BaseException) -> dict[str, Any] | None:
    """Parse ``{'error': ...}`` / JSON object often appended to exception messages."""
    text = str(exc)
    for marker in ("{'error':", '{"error":'):
        i = text.find(marker)
        if i < 0:
            continue
        blob = _balanced_brace_slice(text[i:])
        if not blob:
            continue
        obj: Any = None
        try:
            obj = ast.literal_eval(blob)
        except (SyntaxError, ValueError, MemoryError):
            try:
                obj = json.loads(blob)
            except json.JSONDecodeError:
                continue
        if isinstance(obj, dict) and isinstance(obj.get("error"), dict):
            return obj
    return None


def _json_dict_maybe(raw: str) -> dict[str, Any] | None:
    try:
        j = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return j if isinstance(j, dict) else None


def _error_payload_from_exception(exc: BaseException) -> dict[str, Any] | None:
    """Return the top-level API JSON object ``{"error": {...}}`` when discoverable."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        for attr in ("body", "response_json", "_response_json"):
            v = getattr(cur, attr, None)
            if isinstance(v, dict) and isinstance(v.get("error"), dict):
                return v
            if isinstance(v, str):
                d = _json_dict_maybe(v)
                if d and isinstance(d.get("error"), dict):
                    return d

        resp = getattr(cur, "response", None)
        if resp is not None:
            try:
                j = resp.json()
                if isinstance(j, dict) and isinstance(j.get("error"), dict):
                    return j
            except Exception:
                pass
            txt = getattr(resp, "text", None)
            if isinstance(txt, str):
                d = _json_dict_maybe(txt.strip())
                if d and isinstance(d.get("error"), dict):
                    return d

        cur = cur.__cause__ or cur.__context__
    return None


def _summarize_detail(d: dict[str, Any]) -> list[str]:
    out: list[str] = []
    typ = d.get("@type") or d.get("type")
    if typ == "type.googleapis.com/google.rpc.QuotaFailure":
        viol = d.get("violations")
        if not isinstance(viol, list):
            return out
        for v in viol:
            if not isinstance(v, dict):
                continue
            metric = v.get("quotaMetric") or v.get("quota_metric")
            qid = v.get("quotaId") or v.get("quota_id")
            qval = v.get("quotaValue") or v.get("quota_value")
            dims = v.get("quotaDimensions") or v.get("quota_dimensions")
            parts = []
            if metric:
                parts.append(f"metric={metric}")
            if qid:
                parts.append(f"id={qid}")
            if qval is not None:
                parts.append(f"limit={qval}")
            if dims:
                parts.append(f"dims={dims}")
            if parts:
                out.append("QuotaFailure[" + "; ".join(parts) + "]")
        return out
    if typ == "type.googleapis.com/google.rpc.RetryInfo":
        rd = d.get("retryDelay") or d.get("retry_delay")
        if rd:
            out.append(f"RetryInfo[retryDelay={rd}]")
        return out
    if typ == "type.googleapis.com/google.rpc.ErrorInfo":
        reason = d.get("reason")
        meta = d.get("metadata")
        if reason or meta:
            out.append(f"ErrorInfo[reason={reason!r} metadata={meta!r}]")
        return out
    return out


def format_gemini_quota_diagnostics(exc: BaseException) -> str:
    """One line for WARNING logs; empty if the server did not send QuotaFailure / ErrorInfo."""
    payload = _error_payload_from_exception(exc) or _payload_from_exception_string(exc)
    if not payload:
        return ""
    err = payload.get("error")
    if not isinstance(err, dict):
        return ""
    details = err.get("details")
    if not isinstance(details, list):
        return ""
    bits: list[str] = []
    for d in details:
        if not isinstance(d, dict):
            continue
        bits.extend(_summarize_detail(d))
    if not bits:
        return ""
    return " | ".join(bits)


def gemini_error_log_suffix(exc: BaseException) -> str:
    """Suffix for WARNING lines: structured quota bits, or a short hint when Google omits them."""
    qdiag = format_gemini_quota_diagnostics(exc)
    if qdiag:
        return f" | {qdiag}"
    s = str(exc).lower()
    if "429" in s or "resource_exhausted" in s:
        return (
            " | quota_detail: (response had no QuotaFailure/RetryInfo blocks to parse; "
            "use AI Studio / ai.dev usage dashboards and billing tier — RPM vs TPM vs RPD "
            "is not named in this payload)"
        )
    return ""
