"""SEC EDGAR helpers (tickers → CIK, submissions, XBRL company facts).

SEC requires a descriptive User-Agent (company/app + contact). Set SEC_USER_AGENT.
https://www.sec.gov/os/accessing-edgar-data
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import httpx

SEC_DATA = "https://data.sec.gov"
SEC_WWW = "https://www.sec.gov"


def _headers(user_agent: str) -> dict[str, str]:
    return {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json,text/plain,*/*",
    }


@lru_cache(maxsize=1)
def _company_tickers_index(user_agent: str) -> dict[str, tuple[int, str]]:
    """Map upper ticker -> (cik_int, issuer_title)."""
    with httpx.Client(headers=_headers(user_agent), timeout=60.0) as client:
        r = client.get(f"{SEC_WWW}/files/company_tickers.json")
        r.raise_for_status()
        raw = r.json()

    rows: list[dict[str, Any]]
    if isinstance(raw, list):
        rows = [x for x in raw if isinstance(x, dict)]
    elif isinstance(raw, dict):
        rows = [x for x in raw.values() if isinstance(x, dict)]
    else:
        rows = []

    out: dict[str, tuple[int, str]] = {}
    for row in rows:
        t = str(row.get("ticker") or "").strip().upper()
        cik = row.get("cik_str")
        title = str(row.get("title") or "").strip()
        if not t or cik is None:
            continue
        try:
            cik_int = int(cik)
        except (TypeError, ValueError):
            continue
        out[t] = (cik_int, title)
    return out


def resolve_ticker(user_agent: str, ticker: str) -> tuple[int, str] | None:
    sym = ticker.strip().upper()
    idx = _company_tickers_index(user_agent)
    return idx.get(sym)


def cik_padded(cik: int) -> str:
    return str(int(cik)).zfill(10)


def fetch_submissions(user_agent: str, cik: int) -> dict[str, Any]:
    cik_s = cik_padded(cik)
    url = f"{SEC_DATA}/submissions/CIK{cik_s}.json"
    with httpx.Client(headers=_headers(user_agent), timeout=45.0) as client:
        r = client.get(url)
        r.raise_for_status()
        return r.json()


def fetch_companyfacts(user_agent: str, cik: int) -> dict[str, Any]:
    cik_s = cik_padded(cik)
    url = f"{SEC_DATA}/api/xbrl/companyfacts/CIK{cik_s}.json"
    with httpx.Client(headers=_headers(user_agent), timeout=60.0) as client:
        r = client.get(url)
        r.raise_for_status()
        return r.json()


def _latest_annual_usd(
    facts_root: dict[str, Any],
    tag: str,
    *,
    max_points: int = 6,
) -> list[dict[str, Any]]:
    """Return newest-first annual USD points for a us-gaap tag."""
    try:
        node = facts_root["facts"]["us-gaap"][tag]
    except Exception:  # noqa: BLE001
        return []
    units = node.get("units") or {}
    usd = units.get("USD") or units.get("usd")
    if not isinstance(usd, list):
        return []

    annual = [
        p
        for p in usd
        if isinstance(p, dict)
        and p.get("fp") in ("FY", "Y", None)
        and p.get("val") is not None
        and p.get("end")
    ]
    annual.sort(key=lambda p: str(p.get("end")), reverse=True)
    out: list[dict[str, Any]] = []
    for p in annual[:max_points]:
        out.append(
            {
                "end": p.get("end"),
                "fy": p.get("fy"),
                "val": float(p["val"]),
                "form": p.get("form"),
                "filed": p.get("filed"),
            }
        )
    return out


def summarize_companyfacts(facts_root: dict[str, Any]) -> dict[str, Any]:
    """Compact XBRL summary for LLM consumption (bounded size)."""
    tags = [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "OperatingIncomeLoss",
        "NetIncomeLoss",
        "Assets",
        "Liabilities",
        "StockholdersEquity",
        "NetCashProvidedByUsedInOperatingActivities",
        "CashAndCashEquivalentsAtCarryingValue",
    ]
    summary: dict[str, Any] = {}
    for tag in tags:
        pts = _latest_annual_usd(facts_root, tag, max_points=4)
        if pts:
            summary[tag] = pts
    return summary


def recent_filings(submissions: dict[str, Any], *, limit: int = 12) -> list[dict[str, str]]:
    """Recent filing metadata from SEC submissions bundle."""
    out: list[dict[str, str]] = []
    recent = submissions.get("filings", {}).get("recent", {})
    if not isinstance(recent, dict):
        return []
    forms = recent.get("form") or []
    filing_dates = recent.get("filingDate") or []
    accession = recent.get("accessionNumber") or []
    primary = recent.get("primaryDocument") or []
    n = min(len(forms), len(filing_dates), len(accession), len(primary), limit)
    for i in range(n):
        out.append(
            {
                "form": str(forms[i]),
                "filing_date": str(filing_dates[i]),
                "accession": str(accession[i]),
                "primary_document": str(primary[i]),
            }
        )
    return out
