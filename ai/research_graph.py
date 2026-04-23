"""LangGraph orchestration for the equity research pipeline (first agent; more agents later)."""

from __future__ import annotations

import json
import logging
import operator
from collections.abc import Callable
from datetime import date
from typing import Annotated, Any, TypedDict

import httpx
from langgraph.graph import END, START, StateGraph

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
from junior_discount_agent import run_junior_discount_decision
from price_target import finalize_price_target
from report_builder import (
    ReportResult,
    ReportStep,
    gather_headlines_multi,
    npv_from_go,
    project_fcf,
    validate_ticker,
)
from research_common import append_step, emit_phase, pick_ocf_base, render_deterministic_tex
from search_plan import plan_headline_search_queries

_log = logging.getLogger(__name__)


class ResearchState(TypedDict, total=False):
    """Graph state for the equity analyst agent (extend for multi-agent workflows)."""

    # --- inputs (set by runner before invoke) ---
    sym: str
    go_math_url: str
    searxng_url: str
    httpx_client: httpx.Client
    sec_user_agent: str
    gemini_api_key: str | None
    gemini_model: str
    emit: Callable[[dict[str, Any]], None] | None

    # --- working / outputs ---
    steps: Annotated[list[ReportStep], operator.add]
    warnings: Annotated[list[str], operator.add]
    data_sources: Annotated[list[str], operator.add]
    bundle: dict[str, Any]
    company_name: str
    snapshot_rows: list[tuple[str, str]]
    consensus_pt: float | None
    consensus_note: str | None
    reference_last_close: float | None
    queries: list[str]
    plan_err: str | None
    headlines: list[dict[str, str]]
    base_fcf: float
    fcf_note: str
    ocf_from_sec: bool
    flows: list[float]
    discount_rate: float
    discount_basis: str
    discount_source: str  # junior_model | seed_no_key | seed_fallback
    npv_val: float | None
    npv_err: str | None
    data_json: str
    dcf_rows: list[tuple[str, str]]
    gemini_model_used: str | None
    report_mode: str
    latex: str
    research_memo: str
    reasoning_text: str | None
    model_pt_from_llm: dict[str, Any] | None
    pt_usd: float | None
    pt_hm: int | None
    pt_basis: str | None
    report: ReportResult


def _node_sec(state: ResearchState) -> dict[str, Any]:
    emit = state.get("emit")
    sym = state["sym"]
    sec_user_agent = state["sec_user_agent"]
    bundle = dict(state["bundle"])
    company_name = state["company_name"]
    snapshot_rows = list(state["snapshot_rows"])
    out: dict[str, Any] = {}

    emit_phase(emit, {"type": "phase", "id": "sec", "status": "start", "label": "Public filings (SEC)"})
    if not sec_user_agent.strip():
        out["warnings"] = ["SEC_USER_AGENT empty; skipping EDGAR per SEC policy."]
        out["steps"] = [
            ReportStep(id="sec", label="Public filings (SEC)", status="warn", detail="missing SEC_USER_AGENT"),
        ]
        emit_phase(emit, {"type": "phase", "id": "sec", "status": "warn", "detail": "missing SEC_USER_AGENT"})
        out["bundle"] = bundle
        return out

    try:
        resolved = resolve_ticker(sec_user_agent, sym)
        if not resolved:
            out["warnings"] = ["Ticker not found in SEC company_tickers mapping."]
            out["steps"] = [
                ReportStep(id="sec", label="Public filings (SEC)", status="warn", detail="unknown ticker in SEC map"),
            ]
            emit_phase(emit, {"type": "phase", "id": "sec", "status": "warn", "detail": "unknown ticker"})
        else:
            cik, title = resolved
            company_name = title or company_name
            submissions = fetch_submissions(sec_user_agent, cik)
            entity = str(submissions.get("name") or title or sym)
            company_name = entity
            filings = recent_filings(submissions, limit=12)
            facts_root = fetch_companyfacts(sec_user_agent, cik)
            facts_summary = summarize_companyfacts(facts_root)
            sources = dict(bundle.get("sources") or {})
            sources["sec"] = {
                "cik": cik,
                "entityName": entity,
                "filings_recent": filings,
                "facts_summary": facts_summary,
            }
            bundle["sources"] = sources
            snapshot_rows.extend([("SEC CIK", str(cik)), ("Issuer (SEC)", entity)])
            out["company_name"] = company_name
            out["snapshot_rows"] = snapshot_rows
            out["data_sources"] = ["sec_edgar"]
            out["steps"] = [ReportStep(id="sec", label="Public filings (SEC)", status="ok", detail=f"CIK {cik}")]
            emit_phase(emit, {"type": "phase", "id": "sec", "status": "ok", "detail": f"CIK {cik}"})
    except Exception as exc:  # noqa: BLE001
        out["warnings"] = [f"SEC EDGAR error: {exc}"]
        out["steps"] = [ReportStep(id="sec", label="Public filings (SEC)", status="warn", detail=str(exc)[:200])]
        emit_phase(emit, {"type": "phase", "id": "sec", "status": "error", "detail": str(exc)[:200]})

    out["bundle"] = bundle
    if "company_name" not in out:
        out["company_name"] = company_name
    if "snapshot_rows" not in out:
        out["snapshot_rows"] = snapshot_rows
    return out


def _node_market(state: ResearchState) -> dict[str, Any]:
    emit = state.get("emit")
    sym = state["sym"]
    bundle = dict(state["bundle"])
    snapshot_rows = list(state["snapshot_rows"])
    out: dict[str, Any] = {}

    emit_phase(emit, {"type": "phase", "id": "market", "status": "start", "label": "Market history"})
    try:
        snap = fetch_market_history(ticker=sym, period="6mo")
        ym = dict(bundle.get("sources") or {})
        ym["yfinance_market"] = snap
        bundle["sources"] = ym
        out["data_sources"] = ["yfinance_market"]
        if snap.get("error"):
            out["warnings"] = [f"Market data: {snap['error']}"]
            out["steps"] = [
                ReportStep(id="market", label="Market data", status="warn", detail=str(snap["error"])[:200]),
            ]
            emit_phase(emit, {"type": "phase", "id": "market", "status": "warn", "detail": str(snap["error"])[:200]})
        else:
            bars = snap.get("bars_daily_tail") or []
            last = bars[-1] if bars else None
            if last:
                snapshot_rows.append(("Last close (daily)", str(last.get("close"))))
            consensus_pt, consensus_note = consensus_price_target_from_snapshot(snap)
            reference_last_close = reference_last_close_usd(snap)
            out["consensus_pt"] = consensus_pt
            out["consensus_note"] = consensus_note
            out["reference_last_close"] = reference_last_close
            out["snapshot_rows"] = snapshot_rows
            out["steps"] = [ReportStep(id="market", label="Market data", status="ok", detail="daily bars")]
            emit_phase(emit, {"type": "phase", "id": "market", "status": "ok", "detail": f"{len(bars)} bars"})
    except Exception as exc:  # noqa: BLE001
        out["warnings"] = [f"Market data error: {exc}"]
        out["steps"] = [ReportStep(id="market", label="Market data", status="warn", detail=str(exc)[:200])]
        emit_phase(emit, {"type": "phase", "id": "market", "status": "error", "detail": str(exc)[:200]})

    out["bundle"] = bundle
    if "snapshot_rows" not in out:
        out["snapshot_rows"] = snapshot_rows
    return out


def _node_junior_discount(state: ResearchState) -> dict[str, Any]:
    """Junior Researcher chooses illustrative discount after market + headline context, before FCF/quant."""
    emit = state.get("emit")
    bundle = dict(state["bundle"])
    sym = state["sym"]
    company_name = state["company_name"]
    gemini_api_key = state.get("gemini_api_key")
    gemini_model = state["gemini_model"]

    snap = bundle.get("sources", {}).get("yfinance_market")
    if not isinstance(snap, dict):
        snap = {}
    info = snap.get("info") if isinstance(snap.get("info"), dict) else {}
    seed_rate, seed_basis = infer_discount_rate_for_bridge(info)

    headlines = state.get("headlines") or []
    titles: list[str] = []
    for h in headlines:
        if isinstance(h, dict):
            t = (h.get("title") or "").strip()
            if t:
                titles.append(t[:160])

    queries = [q for q in (state.get("queries") or []) if isinstance(q, str)]

    emit_phase(
        emit,
        {
            "type": "phase",
            "id": "junior_discount",
            "status": "start",
            "label": "Junior discount (desk pass)",
        },
    )

    discount_rate, discount_basis, jwarn, used_model = run_junior_discount_decision(
        api_key=gemini_api_key,
        model=gemini_model,
        ticker=sym,
        company_name=company_name,
        market_snapshot=snap,
        headline_titles=titles,
        planned_queries=queries,
        seed_rate=seed_rate,
        seed_basis=seed_basis,
        emit=emit,
    )

    if used_model:
        discount_source = "junior_model"
        step_status = "ok"
        detail = f"{discount_rate * 100:.1f}% — {discount_basis[:140]}"
    elif not (gemini_api_key or "").strip():
        discount_source = "seed_no_key"
        step_status = "ok"
        detail = f"{discount_rate * 100:.1f}% — deterministic seed (no analyst API key)"
    else:
        discount_source = "seed_fallback"
        step_status = "warn"
        detail = f"{discount_rate * 100:.1f}% — {discount_basis[:140]}"

    bundle["discount_choice"] = {
        "discount_rate": discount_rate,
        "source": discount_source,
        "seed_rate": seed_rate,
    }

    emit_phase(
        emit,
        {
            "type": "phase",
            "id": "junior_discount",
            "status": "warn" if step_status == "warn" else "ok",
            "label": "Junior discount (desk pass)",
            "detail": f"{discount_rate * 100:.1f}%",
        },
    )

    out: dict[str, Any] = {
        "bundle": bundle,
        "discount_rate": discount_rate,
        "discount_basis": discount_basis,
        "discount_source": discount_source,
        "steps": [
            ReportStep(
                id="junior_discount",
                label="Junior discount (desk pass)",
                status=step_status,
                detail=detail,
            ),
        ],
    }
    if jwarn:
        out["warnings"] = [jwarn]
    return out


def _node_search_plan(state: ResearchState) -> dict[str, Any]:
    emit = state.get("emit")
    sym = state["sym"]
    company_name = state["company_name"]
    gemini_api_key = state.get("gemini_api_key")
    gemini_model = state["gemini_model"]

    emit_phase(emit, {"type": "phase", "id": "search_plan", "status": "start", "label": "Search plan"})
    queries, plan_err = plan_headline_search_queries(
        ticker=sym,
        company_name=company_name,
        api_key=gemini_api_key,
        model=gemini_model,
        emit=emit,
    )
    out: dict[str, Any] = {"queries": queries, "plan_err": plan_err}
    if plan_err:
        out["warnings"] = [plan_err]
        out["steps"] = [ReportStep(id="search_plan", label="Search plan", status="warn", detail=plan_err[:200])]
        emit_phase(emit, {"type": "phase", "id": "search_plan", "status": "warn", "detail": plan_err[:200]})
    else:
        out["steps"] = [ReportStep(id="search_plan", label="Search plan", status="ok", detail=f"{len(queries)} queries")]
        emit_phase(
            emit,
            {
                "type": "phase",
                "id": "search_plan",
                "status": "ok",
                "detail": "; ".join(queries[:4]) + ("…" if len(queries) > 4 else ""),
            },
        )
    return out


def _node_news(state: ResearchState) -> dict[str, Any]:
    emit = state.get("emit")
    client = state["httpx_client"]
    searxng_url = state["searxng_url"]
    queries = state["queries"]
    plan_err = state.get("plan_err")
    bundle = dict(state["bundle"])

    emit_phase(emit, {"type": "phase", "id": "news", "status": "start", "label": "Headline scan"})
    headlines, hx_err = gather_headlines_multi(
        client, searxng_url, queries, per_query_limit=4, categories="news"
    )
    sx = dict(bundle.get("sources") or {})
    sx["searxng"] = {
        "queries": queries,
        "headlines": headlines,
        "error": hx_err or None,
        "planning_note": plan_err,
    }
    bundle["sources"] = sx
    out: dict[str, Any] = {"bundle": bundle, "headlines": headlines}
    if hx_err:
        out["warnings"] = [f"Headline scan: {hx_err}"]
        out["steps"] = [ReportStep(id="news", label="Headline scan", status="warn", detail=hx_err[:200])]
        emit_phase(emit, {"type": "phase", "id": "news", "status": "warn", "detail": hx_err[:200]})
    elif not headlines:
        out["warnings"] = ["Headline scan returned no headlines."]
        out["steps"] = [ReportStep(id="news", label="Headline scan", status="warn", detail="no results")]
        emit_phase(emit, {"type": "phase", "id": "news", "status": "warn", "detail": "no results"})
    else:
        out["data_sources"] = ["searxng"]
        out["steps"] = [ReportStep(id="news", label="Headline scan", status="ok", detail=f"{len(headlines)} items")]
        emit_phase(emit, {"type": "phase", "id": "news", "status": "ok", "detail": f"{len(headlines)} headlines"})
    return out


def _node_fcf(state: ResearchState) -> dict[str, Any]:
    emit = state.get("emit")
    bundle = state["bundle"]
    facts_summary = ((bundle.get("sources") or {}).get("sec") or {}).get("facts_summary") or {}

    emit_phase(emit, {"type": "phase", "id": "fcf", "status": "start", "label": "FCF basis"})
    base_fcf, fcf_note, ocf_from_sec = pick_ocf_base(facts_summary if isinstance(facts_summary, dict) else {})
    out: dict[str, Any] = {"base_fcf": base_fcf, "fcf_note": fcf_note, "ocf_from_sec": ocf_from_sec}
    if not ocf_from_sec:
        out["warnings"] = [fcf_note]
        out["steps"] = [ReportStep(id="fcf", label="FCF projection basis", status="warn", detail=fcf_note[:160])]
        emit_phase(emit, {"type": "phase", "id": "fcf", "status": "warn", "detail": fcf_note[:160]})
    else:
        out["data_sources"] = ["sec_facts"]
        out["steps"] = [ReportStep(id="fcf", label="FCF projection basis", status="ok", detail=fcf_note)]
        emit_phase(emit, {"type": "phase", "id": "fcf", "status": "ok"})
    return out


def _node_quant(state: ResearchState) -> dict[str, Any]:
    emit = state.get("emit")
    client = state["httpx_client"]
    go_math_url = state["go_math_url"]
    bundle = dict(state["bundle"])
    discount_rate = state["discount_rate"]
    base_fcf = state["base_fcf"]

    flows = project_fcf(base_fcf, years=5, growth=0.03)
    emit_phase(emit, {"type": "phase", "id": "quant", "status": "start", "label": "Valuation bridge"})
    npv_val, npv_err = npv_from_go(client, go_math_url, flows, discount_rate)
    bundle["quant"] = {
        "fcf_projection": flows,
        "npv": npv_val,
        "npv_error": npv_err or None,
        "engine": "dcf_sensitivity_stub",
    }
    out: dict[str, Any] = {"bundle": bundle, "flows": flows, "npv_val": npv_val, "npv_err": npv_err}
    if npv_val is None:
        out["warnings"] = [f"NPV engine: {npv_err}"]
        out["steps"] = [ReportStep(id="quant", label="Valuation bridge", status="error", detail=(npv_err or "")[:200])]
        emit_phase(emit, {"type": "phase", "id": "quant", "status": "error", "detail": (npv_err or "")[:200]})
    else:
        out["data_sources"] = ["go_npv"]
        out["steps"] = [ReportStep(id="quant", label="Valuation bridge", status="ok", detail=f"npv={npv_val:.4f}")]
        emit_phase(emit, {"type": "phase", "id": "quant", "status": "ok", "detail": f"npv={npv_val:.4f}"})
    return out


def _node_workbook(state: ResearchState) -> dict[str, Any]:
    bundle = dict(state["bundle"])
    discount_rate = state["discount_rate"]
    discount_basis = state["discount_basis"]
    discount_source = state.get("discount_source") or "seed_no_key"
    flows = state["flows"]
    npv_val = state["npv_val"]
    fcf_note = state["fcf_note"]
    consensus_pt = state.get("consensus_pt")
    consensus_note = state.get("consensus_note")
    reference_last_close = state.get("reference_last_close")

    bundle["price_target_seed"] = {
        "consensus_mean_usd": consensus_pt,
        "consensus_note": consensus_note,
        "reference_last_close_usd": reference_last_close,
    }
    dr_label = (
        "Discount rate (Junior Researcher)"
        if discount_source == "junior_model"
        else "Discount rate (deterministic seed)"
    )
    dcf_rows: list[tuple[str, str]] = [
        (dr_label, f"{discount_rate * 100:.1f}% — {discount_basis[:220]}"),
        ("FCF projection (5y, +3%/yr from base)", ", ".join(f"{x:,.0f}" for x in flows)),
        ("NPV of illustrative FCF stream", f"{npv_val:,.4f}" if npv_val is not None else "n/a"),
        ("FCF basis", fcf_note[:220]),
    ]
    data_json = json.dumps(bundle, indent=2, default=str)
    if len(data_json) > 120_000:
        data_json = data_json[:120_000] + "\n...truncated...\n"
    return {"bundle": bundle, "data_json": data_json, "dcf_rows": dcf_rows}


def _node_junior_memo(state: ResearchState) -> dict[str, Any]:
    """Junior desk: long **research memo** + PRICE_TARGET (no LaTeX)—Lead PM typesets in portfolio."""
    emit = state.get("emit")
    gemini_api_key = state.get("gemini_api_key")
    if not gemini_api_key:
        return {
            "report_mode": "deterministic",
            "latex": "",
            "research_memo": "",
            "gemini_model_used": None,
            "reasoning_text": None,
            "model_pt_from_llm": None,
        }

    sym = state["sym"]
    gemini_model = state["gemini_model"]
    data_json = state["data_json"]

    _log.info(
        "pipeline junior_memo start ticker=%s model=%s workbook_json_chars=%s",
        sym,
        gemini_model,
        len(data_json),
    )
    emit_phase(
        emit,
        {
            "type": "phase",
            "id": "junior_memo",
            "status": "start",
            "label": "Junior research memo",
            "detail": (
                f"{gemini_model}; ~{len(data_json)} chars — plain memo + rating (no LaTeX); "
                "Lead PM compiles PDF downstream."
            )[:400],
        },
    )

    def _stream_reason(delta: str) -> None:
        if delta:
            emit_phase(emit, {"type": "reasoning_delta", "text": delta})

    memo, reasoning_text, model_pt_from_llm, gerr = generate_latex_from_context(
        api_key=gemini_api_key,
        model=gemini_model,
        data_json=data_json,
        stream_reasoning=_stream_reason if emit is not None else None,
        emit=emit,
        narrative_output="research_memo",
    )
    _log.info(
        "pipeline junior_memo end ticker=%s ok=%s memo_chars=%s reasoning_chars=%s err=%r",
        sym,
        not bool(gerr),
        len(memo or ""),
        len(reasoning_text or ""),
        (gerr or "")[:200],
    )
    out: dict[str, Any] = {
        "reasoning_text": reasoning_text,
        "model_pt_from_llm": model_pt_from_llm,
        "research_memo": memo or "",
        "latex": "",
    }
    if gerr:
        out["warnings"] = [f"Junior memo model: {gerr}"]
        out["steps"] = [
            ReportStep(id="junior_memo", label="Junior research memo", status="error", detail=gerr[:240]),
        ]
        emit_phase(emit, {"type": "phase", "id": "junior_memo", "status": "error", "detail": gerr[:240]})
        out["gemini_model_used"] = None
        out["report_mode"] = "deterministic"
    else:
        out["gemini_model_used"] = gemini_model
        out["report_mode"] = "gemini_memo"
        out["data_sources"] = ["gemini"]
        out["steps"] = [
            ReportStep(
                id="junior_memo",
                label="Junior research memo",
                status="ok",
                detail=f"{len(memo or '')} chars memo",
            ),
        ]
        emit_phase(
            emit,
            {
                "type": "phase",
                "id": "junior_memo",
                "status": "ok",
                "detail": f"{len(memo or '')} chars",
            },
        )
    return out


def _node_finalize(state: ResearchState) -> dict[str, Any]:
    emit = state.get("emit")
    sym = state["sym"]
    company_name = state["company_name"]
    gemini_api_key = state.get("gemini_api_key")
    gemini_model = state["gemini_model"]
    latex = state.get("latex") or ""
    report_mode = state.get("report_mode") or "deterministic"
    gemini_model_used = state.get("gemini_model_used")
    reasoning_text = state.get("reasoning_text")
    model_pt_from_llm = state.get("model_pt_from_llm")
    consensus_pt = state.get("consensus_pt")
    consensus_note = state.get("consensus_note")
    discount_rate = state["discount_rate"]
    discount_basis = state["discount_basis"]
    discount_source = state.get("discount_source") or "seed_no_key"
    flows = state["flows"]
    npv_val = state["npv_val"]
    dcf_rows = state["dcf_rows"]
    snapshot_rows = state["snapshot_rows"]
    headlines = state["headlines"]
    steps = list(state["steps"])
    warnings = list(state["warnings"])
    data_sources = list(state["data_sources"])

    pt_usd, pt_hm, pt_basis = finalize_price_target(model_pt_from_llm, consensus_pt, consensus_note)
    if pt_usd is not None:
        emit_phase(
            emit,
            {
                "type": "price_target",
                "usd": pt_usd,
                "horizon_months": pt_hm,
                "basis": (pt_basis or "")[:500],
            },
        )

    research_memo = (state.get("research_memo") or "").strip()

    if not latex.strip() and research_memo and report_mode == "gemini_memo":
        latex = ""
    elif not latex.strip():
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
                "did not return a usable research memo or LaTeX for this run. Inspect warnings in the API "
                "payload for quota, format, or connectivity issues, then retry."
            )
        risks = (
            "Key limitations: (1) Extracted filing facts may omit material line items; filings can lag price. "
            "(2) Market data can be delayed, rebased, or incomplete for ADRs and spin-offs. "
            "(3) Headline snippets are not vetted—headline risk and false positives. "
            "(4) The cash-flow bridge uses a stylized five-year path (+3\\%/yr) from an operating-cash-flow "
            "scale or placeholder—not a full DCF. "
            "(5) Regulatory, liquidity, and tail risks are not modeled here."
        )
        latex = render_deterministic_tex(
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
        append_step(steps, "editor", "Finalize LaTeX", "ok", f"narrative:{gemini_model}")
    elif report_mode == "gemini_memo" and research_memo:
        append_step(
            steps,
            "editor",
            "Junior memo ready for Lead PM",
            "ok",
            (gemini_model_used or "")[:120],
        )
    elif latex and attempted_gemini and gemini_model_used is None:
        append_step(
            steps,
            "editor",
            "Finalize LaTeX",
            "warn",
            "deterministic_jinja_fallback_after_narrative_error",
        )
    elif latex:
        append_step(steps, "editor", "Finalize LaTeX", "ok", "deterministic_jinja_no_narrative_key")

    inv_rec: str | None = None
    if isinstance(model_pt_from_llm, dict):
        r = model_pt_from_llm.get("rating")
        if isinstance(r, str) and r.strip():
            inv_rec = r.strip()

    report = ReportResult(
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
        discount_rate_source=discount_source,
        junior_research_memo=research_memo or None,
        investment_recommendation=inv_rec,
        workbook_json_excerpt=(state.get("data_json") or "")[:80_000] or None,
    )
    return {"report": report, "latex": latex}


def build_equity_research_graph() -> StateGraph:
    """Compile-time graph for the first analyst agent (sequential; parallel subgraphs later)."""
    g = StateGraph(ResearchState)
    g.add_node("sec", _node_sec)
    g.add_node("market", _node_market)
    g.add_node("search_plan", _node_search_plan)
    g.add_node("news", _node_news)
    g.add_node("junior_discount", _node_junior_discount)
    g.add_node("fcf", _node_fcf)
    g.add_node("quant", _node_quant)
    g.add_node("workbook", _node_workbook)
    g.add_node("junior_memo", _node_junior_memo)
    g.add_node("finalize", _node_finalize)

    g.add_edge(START, "sec")
    g.add_edge("sec", "market")
    g.add_edge("market", "search_plan")
    g.add_edge("search_plan", "news")
    g.add_edge("news", "junior_discount")
    g.add_edge("junior_discount", "fcf")
    g.add_edge("fcf", "quant")
    g.add_edge("quant", "workbook")
    g.add_edge("workbook", "junior_memo")
    g.add_edge("junior_memo", "finalize")
    g.add_edge("finalize", END)
    return g


_compiled_graph = None


def _compiled():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_equity_research_graph().compile()
    return _compiled_graph


def run_equity_research_graph(
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
    """Run the LangGraph equity pipeline (same contract as ``generate_full_report``)."""
    sym = validate_ticker(ticker)
    if sym is None:
        raise ValueError("invalid ticker format")

    gemini_api_key = (gemini_api_key or "").strip() or None

    initial: ResearchState = {
        "sym": sym,
        "go_math_url": go_math_url,
        "searxng_url": searxng_url,
        "httpx_client": httpx_client,
        "sec_user_agent": sec_user_agent,
        "gemini_api_key": gemini_api_key,
        "gemini_model": gemini_model,
        "emit": emit,
        "steps": [],
        "warnings": [],
        "data_sources": [],
        "bundle": {"ticker": sym, "as_of": date.today().isoformat(), "sources": {}},
        "company_name": sym,
        "snapshot_rows": [("Ticker", sym)],
        "latex": "",
        "queries": [],
        "headlines": [],
        "flows": [],
        "dcf_rows": [],
    }
    final = _compiled().invoke(initial)
    rep = final.get("report")
    if not isinstance(rep, ReportResult):
        raise RuntimeError("equity research graph did not produce a ReportResult")
    return rep
