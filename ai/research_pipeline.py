"""Orchestrate public filings, market data, headlines, valuation bridge, then LLM or deterministic LaTeX."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from typing import Any

import httpx

from data_sec import (
    fetch_companyfacts,
    fetch_submissions,
    recent_filings,
    resolve_ticker,
    summarize_companyfacts,
)
from data_yfinance_market import (
    consensus_price_target_from_snapshot,
    fetch_market_history,
    infer_discount_rate_for_bridge,
    reference_last_close_usd,
)
from gemini_report import generate_latex_from_context
from price_target import finalize_price_target
from report_builder import (
    ReportResult,
    ReportStep,
    gather_headlines_multi,
    get_template_env,
    npv_from_go,
    project_fcf,
    validate_ticker,
)
from search_plan import plan_headline_search_queries


def _append(
    steps: list[ReportStep],
    sid: str,
    label: str,
    status: str,
    detail: str = "",
) -> None:
    steps.append(ReportStep(id=sid, label=label, status=status, detail=detail))


def _emit(
    emit: Callable[[dict[str, Any]], None] | None,
    payload: dict[str, Any],
) -> None:
    if emit is not None:
        emit(payload)


def _pick_ocf_base(facts_summary: dict[str, Any]) -> tuple[float, str, bool]:
    """Returns ``(base_usd, note, used_sec_ocf)``."""
    ocf = facts_summary.get("NetCashProvidedByUsedInOperatingActivities")
    if isinstance(ocf, list) and ocf:
        val = float(ocf[0]["val"])
        if val == val and val != 0.0:
            base = abs(val)
            return (
                base,
                "Operating cash flow scale from SEC annual "
                "NetCashProvidedByUsedInOperatingActivities (USD); "
                "five-year series applies +3%/yr for the illustrative valuation bridge only.",
                True,
            )
    return (
        100.0,
        "No operating cash flow tag in extracted filing facts; using illustrative base for the valuation bridge.",
        False,
    )


def _render_deterministic_tex(
    *,
    sym: str,
    company_name: str,
    thesis: str,
    risks: str,
    dcf_rows: list[tuple[str, str]],
    snapshot_rows: list[tuple[str, str]],
    headlines: list[dict[str, str]],
    price_target_usd: float | None = None,
    price_target_horizon_months: int | None = None,
    price_target_basis: str | None = None,
) -> str:
    tmpl = get_template_env().get_template("report.tex.j2")
    return tmpl.render(
        ticker=sym,
        report_date=date.today().isoformat(),
        company_name=company_name,
        thesis=thesis,
        risks=risks,
        dcf_summary_rows=dcf_rows,
        snapshot_rows=snapshot_rows,
        headlines=headlines,
        price_target_usd=price_target_usd,
        price_target_horizon_months=price_target_horizon_months,
        price_target_basis=price_target_basis,
    )


def generate_full_report(
    *,
    ticker: str,
    go_math_url: str,
    searxng_url: str,
    httpx_client: httpx.Client,
    sec_user_agent: str,
    gemini_api_key: str | None,
    gemini_model: str,
    emit: Callable[[dict[str, Any]], None] | None = None,
) -> ReportResult:
    steps: list[ReportStep] = []
    warnings: list[str] = []
    data_sources: list[str] = []

    sym = validate_ticker(ticker)
    if sym is None:
        raise ValueError("invalid ticker format")

    gemini_api_key = (gemini_api_key or "").strip() or None

    bundle: dict[str, Any] = {
        "ticker": sym,
        "as_of": date.today().isoformat(),
        "sources": {},
    }

    company_name = sym
    snapshot_rows: list[tuple[str, str]] = [("Ticker", sym)]
    consensus_pt: float | None = None
    consensus_note: str | None = None
    reference_last_close: float | None = None

    # SEC EDGAR
    _emit(emit, {"type": "phase", "id": "sec", "status": "start", "label": "Public filings (SEC)"})
    if not sec_user_agent.strip():
        warnings.append("SEC_USER_AGENT empty; skipping EDGAR per SEC policy.")
        _append(steps, "sec", "Public filings (SEC)", "warn", "missing SEC_USER_AGENT")
        _emit(emit, {"type": "phase", "id": "sec", "status": "warn", "detail": "missing SEC_USER_AGENT"})
    else:
        try:
            resolved = resolve_ticker(sec_user_agent, sym)
            if not resolved:
                warnings.append("Ticker not found in SEC company_tickers mapping.")
                _append(steps, "sec", "Public filings (SEC)", "warn", "unknown ticker in SEC map")
                _emit(emit, {"type": "phase", "id": "sec", "status": "warn", "detail": "unknown ticker"})
            else:
                cik, title = resolved
                company_name = title or company_name
                data_sources.append("sec_edgar")
                submissions = fetch_submissions(sec_user_agent, cik)
                entity = str(submissions.get("name") or title or sym)
                company_name = entity
                filings = recent_filings(submissions, limit=12)
                facts_root = fetch_companyfacts(sec_user_agent, cik)
                facts_summary = summarize_companyfacts(facts_root)
                bundle["sources"]["sec"] = {
                    "cik": cik,
                    "entityName": entity,
                    "filings_recent": filings,
                    "facts_summary": facts_summary,
                }
                snapshot_rows.extend(
                    [
                        ("SEC CIK", str(cik)),
                        ("Issuer (SEC)", entity),
                    ]
                )
                _append(steps, "sec", "Public filings (SEC)", "ok", f"CIK {cik}")
                _emit(emit, {"type": "phase", "id": "sec", "status": "ok", "detail": f"CIK {cik}"})
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"SEC EDGAR error: {exc}")
            _append(steps, "sec", "Public filings (SEC)", "warn", str(exc)[:200])
            _emit(emit, {"type": "phase", "id": "sec", "status": "error", "detail": str(exc)[:200]})

    # Market data (yfinance — avoids Alpaca SIP subscription limits)
    _emit(emit, {"type": "phase", "id": "market", "status": "start", "label": "Market history"})
    try:
        snap = fetch_market_history(ticker=sym, period="6mo")
        bundle["sources"]["yfinance_market"] = snap
        data_sources.append("yfinance_market")
        if snap.get("error"):
            warnings.append(f"Market data: {snap['error']}")
            _append(steps, "market", "Market data", "warn", str(snap["error"])[:200])
            _emit(emit, {"type": "phase", "id": "market", "status": "warn", "detail": str(snap["error"])[:200]})
        else:
            bars = snap.get("bars_daily_tail") or []
            last = bars[-1] if bars else None
            if last:
                snapshot_rows.append(("Last close (daily)", str(last.get("close"))))
            consensus_pt, consensus_note = consensus_price_target_from_snapshot(snap)
            reference_last_close = reference_last_close_usd(snap)
            _append(steps, "market", "Market data", "ok", "daily bars")
            _emit(emit, {"type": "phase", "id": "market", "status": "ok", "detail": f"{len(bars)} bars"})
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Market data error: {exc}")
        _append(steps, "market", "Market data", "warn", str(exc)[:200])
        _emit(emit, {"type": "phase", "id": "market", "status": "error", "detail": str(exc)[:200]})

    snap_for_discount = bundle["sources"].get("yfinance_market")
    if not isinstance(snap_for_discount, dict):
        snap_for_discount = {}
    discount_rate, discount_basis = infer_discount_rate_for_bridge(
        snap_for_discount.get("info") or {}
    )
    bundle["discount_rate"] = discount_rate
    bundle["discount_rate_basis"] = discount_basis
    _append(
        steps,
        "discount",
        "Discount assumption",
        "ok",
        f"{discount_rate * 100:.1f}% — {discount_basis[:160]}",
    )
    _emit(
        emit,
        {
            "type": "phase",
            "id": "discount",
            "status": "ok",
            "label": "Discount assumption",
            "detail": f"{discount_rate * 100:.1f}%",
        },
    )

    # Headline scan: model-planned queries (diverse news angles), then SearXNG per query
    _emit(emit, {"type": "phase", "id": "search_plan", "status": "start", "label": "Search plan"})
    queries, plan_err = plan_headline_search_queries(
        ticker=sym,
        company_name=company_name,
        api_key=gemini_api_key,
        model=gemini_model,
    )
    if plan_err:
        warnings.append(plan_err)
        _append(steps, "search_plan", "Search plan", "warn", plan_err[:200])
        _emit(emit, {"type": "phase", "id": "search_plan", "status": "warn", "detail": plan_err[:200]})
    else:
        _append(steps, "search_plan", "Search plan", "ok", f"{len(queries)} queries")
        _emit(
            emit,
            {
                "type": "phase",
                "id": "search_plan",
                "status": "ok",
                "detail": "; ".join(queries[:4]) + ("…" if len(queries) > 4 else ""),
            },
        )

    _emit(emit, {"type": "phase", "id": "news", "status": "start", "label": "Headline scan"})
    headlines, hx_err = gather_headlines_multi(
        httpx_client, searxng_url, queries, per_query_limit=4, categories="news"
    )
    bundle["sources"]["searxng"] = {
        "queries": queries,
        "headlines": headlines,
        "error": hx_err or None,
        "planning_note": plan_err,
    }
    if hx_err:
        warnings.append(f"Headline scan: {hx_err}")
        _append(steps, "news", "Headline scan", "warn", hx_err[:200])
        _emit(emit, {"type": "phase", "id": "news", "status": "warn", "detail": hx_err[:200]})
    elif not headlines:
        warnings.append("Headline scan returned no headlines.")
        _append(steps, "news", "Headline scan", "warn", "no results")
        _emit(emit, {"type": "phase", "id": "news", "status": "warn", "detail": "no results"})
    else:
        data_sources.append("searxng")
        _append(steps, "news", "Headline scan", "ok", f"{len(headlines)} items")
        _emit(emit, {"type": "phase", "id": "news", "status": "ok", "detail": f"{len(headlines)} headlines"})

    facts_summary = (
        (bundle.get("sources") or {}).get("sec") or {}
    ).get("facts_summary") or {}
    _emit(emit, {"type": "phase", "id": "fcf", "status": "start", "label": "FCF basis"})
    base_fcf, fcf_note, ocf_from_sec = _pick_ocf_base(
        facts_summary if isinstance(facts_summary, dict) else {}
    )
    if not ocf_from_sec:
        warnings.append(fcf_note)
        _append(steps, "fcf", "FCF projection basis", "warn", fcf_note)
        _emit(emit, {"type": "phase", "id": "fcf", "status": "warn", "detail": fcf_note[:160]})
    else:
        data_sources.append("sec_facts")
        _append(steps, "fcf", "FCF projection basis", "ok", fcf_note)
        _emit(emit, {"type": "phase", "id": "fcf", "status": "ok"})

    flows = project_fcf(base_fcf, years=5, growth=0.03)
    _emit(emit, {"type": "phase", "id": "quant", "status": "start", "label": "Valuation bridge"})
    npv_val, npv_err = npv_from_go(httpx_client, go_math_url, flows, discount_rate)
    bundle["quant"] = {
        "fcf_projection": flows,
        "npv": npv_val,
        "npv_error": npv_err or None,
        "engine": "dcf_sensitivity_stub",
    }
    if npv_val is None:
        warnings.append(f"NPV engine: {npv_err}")
        _append(steps, "quant", "Valuation bridge", "error", (npv_err or "")[:200])
        _emit(emit, {"type": "phase", "id": "quant", "status": "error", "detail": (npv_err or "")[:200]})
    else:
        data_sources.append("go_npv")
        _append(steps, "quant", "Valuation bridge", "ok", f"npv={npv_val:.4f}")
        _emit(emit, {"type": "phase", "id": "quant", "status": "ok", "detail": f"npv={npv_val:.4f}"})

    dcf_rows: list[tuple[str, str]] = [
        (
            "Discount rate (inferred)",
            f"{discount_rate * 100:.1f}% — {discount_basis[:220]}",
        ),
        ("FCF projection (5y, +3%/yr from base)", ", ".join(f"{x:,.0f}" for x in flows)),
        ("NPV of illustrative FCF stream", f"{npv_val:,.4f}" if npv_val is not None else "n/a"),
        ("FCF basis", fcf_note[:220]),
    ]

    bundle["price_target_seed"] = {
        "consensus_mean_usd": consensus_pt,
        "consensus_note": consensus_note,
        "reference_last_close_usd": reference_last_close,
    }

    data_json = json.dumps(bundle, indent=2, default=str)
    if len(data_json) > 120_000:
        data_json = data_json[:120_000] + "\n...truncated...\n"

    gemini_model_used: str | None = None
    report_mode = "deterministic"
    latex = ""
    reasoning_text: str | None = None
    model_pt_from_llm: dict[str, Any] | None = None

    if gemini_api_key:
        _emit(emit, {"type": "phase", "id": "gemini", "status": "start", "label": "Research narrative"})
        def _stream_reason(delta: str) -> None:
            if delta:
                _emit(emit, {"type": "reasoning_delta", "text": delta})

        latex, reasoning_text, model_pt_from_llm, gerr = generate_latex_from_context(
            api_key=gemini_api_key,
            model=gemini_model,
            data_json=data_json,
            stream_reasoning=_stream_reason if emit is not None else None,
        )
        if gerr:
            warnings.append(f"Narrative model: {gerr}")
            _append(steps, "gemini", "Research narrative", "error", gerr[:240])
            _emit(emit, {"type": "phase", "id": "gemini", "status": "error", "detail": gerr[:240]})
            latex = ""
            reasoning_text = reasoning_text or None
        else:
            gemini_model_used = gemini_model
            report_mode = "gemini"
            data_sources.append("gemini")
            _append(steps, "gemini", "Research narrative", "ok", gemini_model)
            _emit(emit, {"type": "phase", "id": "gemini", "status": "ok", "detail": gemini_model})

    pt_usd, pt_hm, pt_basis = finalize_price_target(
        model_pt_from_llm, consensus_pt, consensus_note
    )
    if pt_usd is not None:
        _emit(
            emit,
            {
                "type": "price_target",
                "usd": pt_usd,
                "horizon_months": pt_hm,
                "basis": (pt_basis or "")[:500],
            },
        )

    if not latex:
        if not gemini_api_key:
            thesis = (
                f"This extended deterministic draft covers {company_name} ({sym}) from public sources only. "
                "Without an analyst-model API key, narrative depth is intentionally capped: you still receive "
                "snapshot tables, headline snippets, an illustrative discounted cash-flow bridge, and a price "
                "target when consensus estimates are available in the market feed. Configure the analyst model "
                "for a full sell-side-style write-up (multi-page PDF with executive summary, bull/bear, and "
                "methodology) on the same underlying data."
            )
        else:
            thesis = (
                f"This extended deterministic fallback covers {company_name} ({sym}) because the analyst model "
                "did not return usable LaTeX for this run. Inspect warnings in the API payload for quota, "
                "format, or connectivity issues, then retry. When the narrative path succeeds, expect a longer "
                "note with ordered sections from executive summary through data limitations."
            )
        risks = (
            "Key limitations: (1) Extracted filing facts may omit material line items; filings can lag price. "
            "(2) Market data can be delayed, rebased, or incomplete for ADRs and spin-offs. "
            "(3) Headline snippets are not vetted—headline risk and false positives. "
            "(4) The cash-flow bridge uses a stylized five-year path (+3\\%/yr) from an operating-cash-flow "
            "scale or placeholder—not a full DCF. "
            "(5) Regulatory, liquidity, and tail risks are not modeled here."
        )
        latex = _render_deterministic_tex(
            sym=sym,
            company_name=company_name,
            thesis=thesis,
            risks=risks,
            dcf_rows=dcf_rows,
            snapshot_rows=snapshot_rows,
            headlines=headlines,
            price_target_usd=pt_usd,
            price_target_horizon_months=pt_hm,
            price_target_basis=pt_basis,
        )

    attempted_gemini = bool(gemini_api_key)
    if latex and report_mode == "gemini":
        _append(steps, "editor", "Finalize LaTeX", "ok", f"narrative:{gemini_model}")
    elif latex and attempted_gemini and gemini_model_used is None:
        _append(
            steps,
            "editor",
            "Finalize LaTeX",
            "warn",
            "deterministic_jinja_fallback_after_narrative_error",
        )
    elif latex:
        _append(steps, "editor", "Finalize LaTeX", "ok", "deterministic_jinja_no_narrative_key")

    return ReportResult(
        ticker=sym,
        company_name=company_name,
        latex=latex,
        npv=npv_val,
        discount_rate=discount_rate,
        fcf_projection=flows,
        steps=steps,
        warnings=warnings,
        report_mode=report_mode,
        data_sources=sorted(set(data_sources)),
        gemini_model=gemini_model_used,
        reasoning=(reasoning_text.strip() or None) if reasoning_text else None,
        price_target_usd=pt_usd,
        price_target_horizon_months=pt_hm,
        price_target_basis=pt_basis,
        headlines=headlines,
        discount_rate_basis=discount_basis,
    )
