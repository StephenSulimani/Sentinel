"""Assemble equity research LaTeX from public quote data, news search, and valuation bridge."""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import yfinance as yf
from jinja2 import Environment, FileSystemLoader, select_autoescape

from data_yfinance_market import (
    consensus_price_target_from_info,
    infer_discount_rate_for_bridge,
)
from price_target import finalize_price_target
from search_plan import plan_headline_search_queries

TICKER_RE = re.compile(r"^[A-Za-z0-9.\-]{1,12}$")
_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

_log = logging.getLogger(__name__)


def _searxng_client_identity_headers() -> dict[str, str]:
    """Headers many SearXNG botdetection builds expect for JSON API server-to-server calls."""
    ip = (os.environ.get("SEARXNG_CLIENT_IP") or "127.0.0.1").strip() or "127.0.0.1"
    return {"X-Forwarded-For": ip, "X-Real-IP": ip}


def normalize_ticker(raw: str) -> str:
    return raw.strip().upper()


def validate_ticker(raw: str) -> str | None:
    t = normalize_ticker(raw)
    if not TICKER_RE.match(t):
        return None
    return t


def latex_escape(value: str | None) -> str:
    if not value:
        return ""
    text = str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "&": r"\&",
        "#": r"\#",
        "_": r"\_",
        "%": r"\%",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def _jinja_env() -> Environment:
    template_dir = Path(__file__).resolve().parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(enabled_extensions=()),
    )
    env.filters["latex"] = latex_escape
    return env


_env = _jinja_env()


def get_template_env() -> Environment:
    return _env


@dataclass
class ReportStep:
    id: str
    label: str
    status: str  # ok | warn | error
    detail: str = ""


@dataclass
class ReportResult:
    ticker: str
    company_name: str
    latex: str
    npv: float | None
    discount_rate: float
    fcf_projection: list[float]
    steps: list[ReportStep] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    report_mode: str = "deterministic"
    data_sources: list[str] = field(default_factory=list)
    gemini_model: str | None = None
    reasoning: str | None = None
    price_target_usd: float | None = None
    price_target_horizon_months: int | None = None
    price_target_basis: str | None = None
    headlines: list[dict[str, str]] = field(default_factory=list)
    discount_rate_basis: str | None = None
    discount_rate_source: str | None = None  # junior_model | seed_no_key | seed_fallback
    # Multi-agent portfolio pipeline (Junior ↔ Critic peer rounds → Lead PM)
    junior_latex: str | None = None
    post_peer_junior_latex: str | None = None
    junior_research_memo: str | None = None
    post_peer_junior_research_memo: str | None = None
    investment_recommendation: str | None = None
    workbook_json_excerpt: str | None = None
    critic_memo: str | None = None
    lead_pm_synthesis: str | None = None


def _append_step(steps: list[ReportStep], sid: str, label: str, status: str, detail: str = "") -> None:
    steps.append(ReportStep(id=sid, label=label, status=status, detail=detail))


def _running_inside_docker() -> bool:
    """True when this process is likely inside a container (not the dev host)."""
    return Path("/.dockerenv").is_file()


def searxng_base_url_candidates(primary: str) -> list[str]:
    """Return SearXNG base URLs to try, in order.

    Compose uses ``http://searxng:8080`` on the bridge network. Loopback fallbacks
    (127.0.0.1) only make sense when the **AI Flask process runs on the host**;
    from inside another container they hit *that* container, not SearXNG.
    Optional ``SEARXNG_FALLBACK_URLS`` (comma-separated) adds extra bases, e.g.
    ``http://192.168.1.10:8080``.
    """
    primary = primary.strip().rstrip("/")
    if not primary:
        return ["http://127.0.0.1:8080"]
    out: list[str] = []
    seen: set[str] = set()

    def add(u: str) -> None:
        u = u.strip().rstrip("/")
        if u and u not in seen:
            seen.add(u)
            out.append(u)

    add(primary)
    host = (urlparse(primary).hostname or "").lower()
    for part in os.environ.get("SEARXNG_FALLBACK_URLS", "").split(","):
        add(part)
    if host == "searxng":
        if not _running_inside_docker():
            add("http://127.0.0.1:8080")
            add("http://localhost:8080")
    return out


def dedupe_headlines(items: list[dict[str, str]], *, cap: int = 18) -> list[dict[str, str]]:
    """Drop duplicate URLs (or titles when URL missing), keep order, cap length."""
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for h in items:
        url = (h.get("url") or "").strip().lower()
        title = (h.get("title") or "").strip().lower()
        key = url if url else title
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(h)
        if len(out) >= cap:
            break
    return out


def gather_headlines_multi(
    client: httpx.Client,
    base_url: str,
    queries: list[str],
    *,
    per_query_limit: int = 4,
    categories: str = "news",
) -> tuple[list[dict[str, str]], str]:
    """Run SearXNG once per query, merge, dedupe. Empty list + error if nothing usable."""
    merged: list[dict[str, str]] = []
    errors: list[str] = []
    raw_delay = (os.environ.get("SEARXNG_QUERY_DELAY_SEC") or "0.35").strip()
    try:
        query_delay = max(0.0, float(raw_delay))
    except ValueError:
        query_delay = 0.35
    for i, q in enumerate(queries[:8]):
        q = q.strip()
        if not q:
            continue
        if i > 0 and query_delay > 0:
            time.sleep(query_delay)
        _log.info(
            "searxng query %s/%s categories=%r q=%r",
            i + 1,
            min(len(queries), 8),
            categories,
            q[:120],
        )
        h, err = searxng_headlines(
            client,
            base_url,
            q,
            limit=per_query_limit,
            categories=categories,
        )
        merged.extend(h)
        if err:
            errors.append(err)
            _log.warning("searxng query error: %s", err[:500])
        else:
            _log.info("searxng query ok: %s result rows (pre-dedupe)", len(h))
    merged = dedupe_headlines(merged, cap=20)
    if merged:
        return merged, ""
    return [], (errors[0] if errors else "headline scan returned no results")


def searxng_headlines(
    client: httpx.Client,
    base_url: str,
    query: str,
    *,
    limit: int = 8,
    categories: str = "news",
) -> tuple[list[dict[str, str]], str]:
    headers = {
        "User-Agent": (
            "ProjectSentinel/1.0 (+https://localhost; internal research; "
            "contact: set SEC_USER_AGENT for outbound SEC, not SearXNG)"
        ),
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        **_searxng_client_identity_headers(),
    }
    errors: list[str] = []
    for base in searxng_base_url_candidates(base_url):
        url = f"{base}/search"
        cat_sequence = [categories] if categories else ["news"]
        if categories and categories != "general":
            cat_sequence.append("general")
        for cat in cat_sequence:
            try:
                r = client.get(
                    url,
                    params={
                        "q": query,
                        "format": "json",
                        "categories": cat,
                    },
                    headers=headers,
                    timeout=25.0,
                )
                r.raise_for_status()
                data = r.json()
            except Exception as exc:  # noqa: BLE001 — try next base / soft failure
                errors.append(f"{base}: {exc}")
                break
            results = data.get("results") or []
            out: list[dict[str, str]] = []
            for item in results[:limit]:
                title = (item.get("title") or "").strip()
                content = (item.get("content") or item.get("snippet") or "").strip()
                url_ = (item.get("url") or "").strip()
                if title or content:
                    out.append({"title": title, "content": content, "url": url_})
            if out:
                return out, ""
        continue


    joined = "; ".join(errors)
    jl = joined.lower()
    in_docker = _running_inside_docker()
    hints: list[str] = []
    if in_docker:
        if "name or service not known" in jl or "gaierror" in jl:
            hints.append(
                "Docker: `searxng` did not resolve — run `ai` and `searxng` from the same "
                "`docker compose` project/network (see docker-compose.yml). Do not rely on "
                "127.0.0.1 from inside a container to reach another service."
            )
        elif "connection refused" in jl and ("127.0.0.1" in joined or "localhost" in joined):
            hints.append(
                "Docker: connection refused on loopback — that is this container, not SearXNG. "
                "Use SEARXNG_URL=http://searxng:8080 with both services on the compose network."
            )
    else:
        if "name or service not known" in jl or "gaierror" in jl:
            hints.append(
                "Host: `searxng` is a Compose DNS name — set SEARXNG_URL=http://127.0.0.1:8080 "
                "when SearXNG publishes 8080 on the machine running Flask."
            )
        if "connection refused" in jl:
            hints.append(
                "Host: nothing is listening on the tried ports — run `docker compose up -d searxng` "
                "(or set SEARXNG_URL / SEARXNG_FALLBACK_URLS to a live instance)."
            )
    hint = (" " + " ".join(hints)) if hints else ""
    return [], joined + hint


def _pick_base_fcf(tkr: yf.Ticker) -> tuple[float, str]:
    """Use latest annual free cash flow if present; else a small positive placeholder."""
    try:
        cf = tkr.cashflow
        if cf is not None and not cf.empty and "Free Cash Flow" in cf.index:
            series = cf.loc["Free Cash Flow"].dropna()
            if len(series) > 0:
                base = float(series.iloc[0])
                if base == base and base != 0.0:  # not NaN
                    note = "Base FCF from latest yfinance annual cash flow statement."
                    return base, note
    except Exception:  # noqa: BLE001
        pass
    return 100.0, "No reliable FCF in yfinance; using illustrative base cash flow."


def project_fcf(base: float, years: int = 5, growth: float = 0.03) -> list[float]:
    return [base * ((1.0 + growth) ** i) for i in range(years)]


def npv_from_go(client: httpx.Client, go_url: str, flows: list[float], discount: float) -> tuple[float | None, str]:
    try:
        r = client.post(
            f"{go_url.rstrip('/')}/npv",
            json={"cash_flows": flows, "discount_rate": discount},
            timeout=15.0,
        )
        if not r.is_success:
            return None, r.text[:500]
        data = r.json()
        val = data.get("npv")
        if isinstance(val, (int, float)):
            return float(val), ""
        return None, "npv missing in response"
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def build_equity_report(
    *,
    ticker: str,
    go_math_url: str,
    searxng_url: str,
    httpx_client: httpx.Client | None = None,
) -> ReportResult:
    steps: list[ReportStep] = []
    warnings: list[str] = []

    sym = validate_ticker(ticker)
    if sym is None:
        raise ValueError("invalid ticker format")

    own_client = httpx_client is None
    client = httpx_client or httpx.Client()

    try:
        tkr = yf.Ticker(sym)
        try:
            info = tkr.info or {}
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"yfinance failed for {sym}: {exc}") from exc

        company = (
            str(info.get("longName") or info.get("shortName") or sym).strip() or sym
        )
        sector = str(info.get("sector") or "n/a")
        industry = str(info.get("industry") or "n/a")
        summary = str(info.get("longBusinessSummary") or "").strip()

        quote_status = "ok"
        quote_detail = "fundamentals loaded"
        if not summary:
            summary = (
                f"{company} operates in the {sector} sector ({industry}). "
                "Automated summary was unavailable from the data provider."
            )
            warnings.append("Business summary missing from yfinance; used a short fallback.")
            quote_status = "warn"
            quote_detail = "thin fundamentals payload"
        _append_step(steps, "quote", "Load market data", quote_status, quote_detail)

        discount_rate, dr_basis = infer_discount_rate_for_bridge(info)
        _append_step(
            steps,
            "discount",
            "Discount assumption",
            "ok",
            f"{discount_rate * 100:.1f}% — {dr_basis[:160]}",
        )

        base_fcf, fcf_note = _pick_base_fcf(tkr)
        if "illustrative" in fcf_note.lower():
            warnings.append(fcf_note)
            _append_step(steps, "fcf", "Derive FCF projection", "warn", fcf_note)
        else:
            _append_step(steps, "fcf", "Derive FCF projection", "ok", fcf_note)

        flows = project_fcf(base_fcf, years=5, growth=0.03)
        npv_val, npv_err = npv_from_go(client, go_math_url, flows, discount_rate)
        if npv_val is None:
            warnings.append(f"NPV engine: {npv_err}")
            _append_step(steps, "quant", "Valuation bridge", "error", npv_err[:200])
        else:
            _append_step(steps, "quant", "Valuation bridge", "ok", f"npv={npv_val:.4f}")

        gkey = (os.environ.get("GEMINI_API_KEY") or "").strip() or None
        gmodel = (os.environ.get("GEMINI_MODEL") or "").strip() or _DEFAULT_GEMINI_MODEL
        queries, plan_err = plan_headline_search_queries(
            ticker=sym, company_name=company, api_key=gkey, model=gmodel
        )
        if plan_err:
            warnings.append(plan_err)
            _append_step(steps, "search_plan", "Search plan", "warn", plan_err[:200])
        else:
            _append_step(steps, "search_plan", "Search plan", "ok", f"{len(queries)} queries")

        headlines, hx_err = gather_headlines_multi(
            client, searxng_url, queries, per_query_limit=4, categories="news"
        )
        if hx_err:
            warnings.append(f"Headline scan: {hx_err}")
            _append_step(steps, "news", "Headline scan", "warn", hx_err[:200])
        elif not headlines:
            warnings.append("Headline scan returned no usable headlines.")
            _append_step(steps, "news", "Headline scan", "warn", "no results")
        else:
            _append_step(steps, "news", "Headline scan", "ok", f"{len(headlines)} items")

        thesis = (
            f"We initiate automated coverage on {company} ({sym}). "
            f"The company participates in {industry} within {sector}. "
            f"Key context: {summary[:900]}"
        )
        if len(summary) > 900:
            thesis += " [...]"

        risks = (
            f"Sector ({sector}) and industry ({industry}) cyclicality; execution risk; "
            "valuation sensitivity to discount rate and terminal assumptions; "
            "data latency and gaps from free data providers; model is illustrative, not a client pitch."
        )

        dcf_rows: list[tuple[str, str]] = [
            (
                "Discount rate (inferred)",
                f"{discount_rate * 100:.1f}% — {dr_basis[:220]}",
            ),
            ("FCF projection (5y, +3%/yr from base)", ", ".join(f"{x:,.0f}" for x in flows)),
            ("NPV of illustrative FCF stream", f"{npv_val:,.4f}" if npv_val is not None else "n/a"),
            ("FCF basis", fcf_note[:220]),
        ]

        c_pt, c_note = consensus_price_target_from_info(info)
        pt_u, pt_h, pt_basis = finalize_price_target(None, c_pt, c_note)
        if pt_u is not None:
            dcf_rows.append(("12-month price target (USD)", f"{pt_u:,.2f}"))

        tmpl = _env.get_template("report.tex.j2")
        latex = tmpl.render(
            ticker=sym,
            report_date=date.today().isoformat(),
            company_name=company,
            thesis=thesis,
            risks=risks,
            dcf_summary_rows=dcf_rows,
            snapshot_rows=[
                ("Sector", sector),
                ("Industry", industry),
                ("Market cap", str(info.get("marketCap") or "n/a")),
                ("Website", str(info.get("website") or "n/a")),
            ],
            headlines=headlines,
            price_target_usd=pt_u,
            price_target_horizon_months=pt_h,
            price_target_basis=pt_basis,
        )

        _append_step(steps, "editor", "Render LaTeX", "ok")

        return ReportResult(
            ticker=sym,
            company_name=company,
            latex=latex,
            npv=npv_val,
            discount_rate=discount_rate,
            fcf_projection=flows,
            steps=steps,
            warnings=warnings,
            report_mode="deterministic_yfinance",
            data_sources=sorted({"yfinance", "searxng", "go_npv"}),
            gemini_model=None,
            reasoning=None,
            price_target_usd=pt_u,
            price_target_horizon_months=pt_h,
            price_target_basis=pt_basis,
            headlines=headlines,
            discount_rate_basis=dr_basis,
            discount_rate_source=None,
        )
    finally:
        if own_client:
            client.close()


def result_to_json_dict(res: ReportResult) -> dict[str, Any]:
    return {
        "ticker": res.ticker,
        "company_name": res.company_name,
        "latex": res.latex,
        "npv": res.npv,
        "discount_rate": res.discount_rate,
        "discount_rate_basis": res.discount_rate_basis,
        "discount_rate_source": res.discount_rate_source,
        "fcf_projection": res.fcf_projection,
        "warnings": res.warnings,
        "report_mode": res.report_mode,
        "data_sources": res.data_sources,
        "gemini_model": res.gemini_model,
        "reasoning": res.reasoning,
        "price_target_usd": res.price_target_usd,
        "price_target_horizon_months": res.price_target_horizon_months,
        "price_target_basis": res.price_target_basis,
        "headlines": res.headlines,
        "junior_latex": res.junior_latex,
        "post_peer_junior_latex": res.post_peer_junior_latex,
        "junior_research_memo": res.junior_research_memo,
        "post_peer_junior_research_memo": res.post_peer_junior_research_memo,
        "investment_recommendation": res.investment_recommendation,
        "workbook_json_excerpt": res.workbook_json_excerpt,
        "critic_memo": res.critic_memo,
        "lead_pm_synthesis": res.lead_pm_synthesis,
        "steps": [
            {"id": s.id, "label": s.label, "status": s.status, "detail": s.detail}
            for s in res.steps
        ],
    }
