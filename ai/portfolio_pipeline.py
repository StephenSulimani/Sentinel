"""Three-agent workflow: Junior Researcher ↔ The Critic (peer rounds) → Lead Portfolio Manager."""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Any

import httpx

from agents import (
    AGENT_CRITIC,
    AGENT_JUNIOR_RESEARCHER,
    AGENT_LABELS,
    AGENT_LEAD_PORTFOLIO_MANAGER,
    tag_emit,
)
from critic_agent import run_critic_memo
from gemini_pacing import sleep_between_gemini_pipeline_agents
from junior_refiner_memo import run_junior_refine_memo
from lead_pm_agent import run_lead_pm_final
from report_builder import ReportResult, ReportStep
from research_common import emit_phase
from research_graph import run_equity_research_graph


def _emit_agent_status(
    emit: Callable[[dict[str, Any]], None] | None,
    agent: str,
    status: str,
    detail: str = "",
) -> None:
    if emit is None:
        return
    ev: dict[str, Any] = {"type": "agent_status", "agent": agent, "status": status}
    if detail:
        ev["detail"] = detail
    emit(ev)


def _peer_review_rounds() -> int:
    raw = (os.environ.get("PEER_REVIEW_ROUNDS") or "2").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 2
    return max(1, min(n, 4))


def _price_anchor_block(junior: ReportResult) -> str:
    """Plain-text block so Lead PM can render page-1 rating + PT callout."""
    lines: list[str] = []
    if junior.investment_recommendation:
        lines.append(f"rating: {junior.investment_recommendation}")
    pt = junior.price_target_usd
    if pt is not None and isinstance(pt, (int, float)) and math.isfinite(float(pt)):
        lines.append(f"usd: {float(pt):.2f}")
    if junior.price_target_horizon_months is not None:
        lines.append(f"horizon_months: {int(junior.price_target_horizon_months)}")
    if junior.price_target_basis:
        lines.append(f"basis: {junior.price_target_basis.strip()}")
    return "\n".join(lines) if lines else "(no structured price target from Junior JSON)"


def run_portfolio_research_pipeline(
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
    """Run equity research (Junior memo), peer-review on **content**, then Lead PM LaTeX."""
    junior_emit = tag_emit(emit, AGENT_JUNIOR_RESEARCHER)
    critic_emit = tag_emit(emit, AGENT_CRITIC)
    jr_peer_emit = tag_emit(emit, AGENT_JUNIOR_RESEARCHER)
    lpm_emit = tag_emit(emit, AGENT_LEAD_PORTFOLIO_MANAGER)

    _emit_agent_status(emit, AGENT_JUNIOR_RESEARCHER, "running")
    junior = run_equity_research_graph(
        ticker=ticker,
        go_math_url=go_math_url,
        searxng_url=searxng_url,
        httpx_client=httpx_client,
        sec_user_agent=sec_user_agent,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
        emit=junior_emit,
    )
    _emit_agent_status(emit, AGENT_JUNIOR_RESEARCHER, "done")

    sleep_between_gemini_pipeline_agents()

    junior_memo_original = (junior.junior_research_memo or "").strip()
    junior_reasoning = junior.reasoning
    steps = list(junior.steps)
    warnings = list(junior.warnings)

    memo_work = junior_memo_original
    critic_memos: list[str] = []
    rounds = _peer_review_rounds()

    for r in range(rounds):
        rnd = r + 1
        emit_phase(
            critic_emit,
            {
                "type": "phase",
                "id": f"peer_critic_{rnd}",
                "status": "start",
                "label": f"{AGENT_LABELS[AGENT_CRITIC]} — round {rnd}/{rounds}",
                "detail": f"Peer review on research memo ({len(memo_work)} chars)",
            },
        )
        _emit_agent_status(emit, AGENT_CRITIC, "running")

        critic_note = memo_work[:450_000]

        prior_excerpt = ""
        if r > 0 and critic_memos:
            prior_excerpt = "\n\n---\n\n".join(m for m in critic_memos if m.strip())[:14_000]

        memo, crit_err = run_critic_memo(
            api_key=gemini_api_key or "",
            model=gemini_model,
            ticker=junior.ticker,
            company_name=junior.company_name,
            junior_note_plaintext=critic_note,
            junior_reasoning=junior_reasoning,
            peer_round=rnd,
            prior_memos_excerpt=prior_excerpt or None,
            emit=critic_emit,
        )

        if crit_err:
            warnings.append(f"The Critic (round {rnd}): {crit_err}")
            critic_memos.append("")
            steps.append(
                ReportStep(
                    id=f"peer_critic_{rnd}",
                    label=f"The Critic — round {rnd}",
                    status="warn",
                    detail=crit_err[:400],
                )
            )
            emit_phase(
                critic_emit,
                {
                    "type": "phase",
                    "id": f"peer_critic_{rnd}",
                    "status": "warn",
                    "label": AGENT_LABELS[AGENT_CRITIC],
                    "detail": crit_err[:500],
                },
            )
            _emit_agent_status(emit, AGENT_CRITIC, "error", crit_err[:240])
        else:
            m = (memo or "").strip()
            critic_memos.append(m)
            steps.append(
                ReportStep(
                    id=f"peer_critic_{rnd}",
                    label=f"The Critic — round {rnd}",
                    status="ok",
                    detail="",
                )
            )
            emit_phase(
                critic_emit,
                {
                    "type": "phase",
                    "id": f"peer_critic_{rnd}",
                    "status": "ok",
                    "label": f"{AGENT_LABELS[AGENT_CRITIC]} — memo round {rnd}",
                },
            )
            _emit_agent_status(emit, AGENT_CRITIC, "done")

        sleep_between_gemini_pipeline_agents()

        memo_for_refine = critic_memos[-1] if critic_memos else ""
        if not memo_for_refine.strip() and crit_err:
            memo_for_refine = (
                f"(The Critic did not return a memo this round: {crit_err}). "
                "Still expand the research memo toward an 8–15 page desk standard without inventing facts."
            )
        elif not memo_for_refine.strip():
            memo_for_refine = (
                "The Critic returned an empty memo—still expand and deepen the research memo per desk standards; "
                "do not invent filing facts."
            )

        emit_phase(
            jr_peer_emit,
            {
                "type": "phase",
                "id": f"junior_refine_{rnd}",
                "status": "start",
                "label": f"{AGENT_LABELS[AGENT_JUNIOR_RESEARCHER]} — memo revision {rnd}/{rounds}",
                "detail": "Plain-text memo pass after The Critic",
            },
        )
        _emit_agent_status(emit, AGENT_JUNIOR_RESEARCHER, "running")

        new_memo, ref_err = run_junior_refine_memo(
            api_key=gemini_api_key or "",
            model=gemini_model,
            ticker=junior.ticker,
            company_name=junior.company_name,
            current_memo=memo_work,
            critic_memo=memo_for_refine,
            junior_reasoning=junior_reasoning,
            peer_round=rnd,
            emit=jr_peer_emit,
        )

        if ref_err:
            warnings.append(f"Junior memo revision (round {rnd}): {ref_err}")
            steps.append(
                ReportStep(
                    id=f"junior_refine_{rnd}",
                    label=f"Junior Researcher — memo revision {rnd}",
                    status="warn",
                    detail=ref_err[:400],
                )
            )
            emit_phase(
                jr_peer_emit,
                {
                    "type": "phase",
                    "id": f"junior_refine_{rnd}",
                    "status": "warn",
                    "label": AGENT_LABELS[AGENT_JUNIOR_RESEARCHER],
                    "detail": ref_err[:500],
                },
            )
            _emit_agent_status(emit, AGENT_JUNIOR_RESEARCHER, "error", ref_err[:240])
        else:
            memo_work = new_memo.strip() or memo_work
            steps.append(
                ReportStep(
                    id=f"junior_refine_{rnd}",
                    label=f"Junior Researcher — memo revision {rnd}",
                    status="ok",
                    detail=f"{len(memo_work)} chars",
                )
            )
            emit_phase(
                jr_peer_emit,
                {
                    "type": "phase",
                    "id": f"junior_refine_{rnd}",
                    "status": "ok",
                    "label": f"{AGENT_LABELS[AGENT_JUNIOR_RESEARCHER]} — revised memo ({rnd})",
                    "detail": f"{len(memo_work)} chars",
                },
            )
            _emit_agent_status(emit, AGENT_JUNIOR_RESEARCHER, "done")

        sleep_between_gemini_pipeline_agents()

    combined_critic = None
    non_empty = [m for m in critic_memos if m.strip()]
    if non_empty:
        combined_critic = "\n\n---\n\n".join(
            f"### Critic round {i + 1}\n{m}" for i, m in enumerate(critic_memos) if m.strip()
        )

    critic_for_lpm = combined_critic or ""
    if not critic_for_lpm.strip():
        critic_for_lpm = (
            "(No critic memos were produced in the peer-review loop.) "
            "As Lead PM, still produce a full 8–15 page institutional LaTeX note from the Junior memos and workbook."
        )

    if emit is not None:
        emit(
            {
                "type": "phase",
                "id": "lead_pm",
                "status": "start",
                "label": f"{AGENT_LABELS[AGENT_LEAD_PORTFOLIO_MANAGER]} — compile client LaTeX",
                "agent": AGENT_LEAD_PORTFOLIO_MANAGER,
            }
        )
    _emit_agent_status(emit, AGENT_LEAD_PORTFOLIO_MANAGER, "running")

    final_tex, synthesis, lpm_err = run_lead_pm_final(
        api_key=gemini_api_key or "",
        model=gemini_model,
        ticker=junior.ticker,
        company_name=junior.company_name,
        research_memo=memo_work,
        research_memo_original=junior_memo_original or None,
        junior_reasoning=junior_reasoning,
        critic_memo=critic_for_lpm,
        data_json_excerpt=junior.workbook_json_excerpt or "",
        price_anchor_block=_price_anchor_block(junior),
        headlines=junior.headlines,
        emit=lpm_emit,
    )

    lead_syn = synthesis.strip() if synthesis and synthesis.strip() else None

    if lpm_err or not (final_tex or "").strip():
        err_t = lpm_err or "empty_lead_pm_response"
        warnings.append(f"The Lead Portfolio Manager: {err_t}")
        steps.append(
            ReportStep(
                id="lead_pm",
                label="The Lead Portfolio Manager — final note",
                status="warn",
                detail=err_t[:400],
            )
        )
        if emit is not None:
            emit(
                {
                    "type": "phase",
                    "id": "lead_pm",
                    "status": "warn",
                    "label": AGENT_LABELS[AGENT_LEAD_PORTFOLIO_MANAGER],
                    "detail": err_t[:500],
                    "agent": AGENT_LEAD_PORTFOLIO_MANAGER,
                }
            )
        _emit_agent_status(
            emit,
            AGENT_LEAD_PORTFOLIO_MANAGER,
            "done",
            f"No final LaTeX ({err_t[:180]}). See junior_research_memo fields for desk content.",
        )
        final_latex = (
            "\\documentclass[11pt]{article}\\usepackage{geometry}\\begin{document}\n"
            "\\section*{Lead PM did not return LaTeX}\n"
            "See API \\texttt{warnings} and the JSON fields "
            "\\texttt{post\\_peer\\_junior\\_research\\_memo} / \\texttt{junior\\_research\\_memo} "
            "for the latest desk research memo.\n"
            "\\end{document}\n"
        )
    else:
        steps.append(
            ReportStep(
                id="lead_pm",
                label="The Lead Portfolio Manager — final note",
                status="ok",
                detail="",
            )
        )
        if emit is not None:
            emit(
                {
                    "type": "phase",
                    "id": "lead_pm",
                    "status": "ok",
                    "label": f"{AGENT_LABELS[AGENT_LEAD_PORTFOLIO_MANAGER]} — final LaTeX ready",
                    "agent": AGENT_LEAD_PORTFOLIO_MANAGER,
                }
            )
        _emit_agent_status(emit, AGENT_LEAD_PORTFOLIO_MANAGER, "done")
        final_latex = final_tex.strip()

    return ReportResult(
        ticker=junior.ticker,
        company_name=junior.company_name,
        latex=final_latex,
        npv=junior.npv,
        discount_rate=junior.discount_rate,
        fcf_projection=junior.fcf_projection,
        steps=steps,
        warnings=warnings,
        report_mode="gemini_portfolio" if not lpm_err and (final_tex or "").strip() else junior.report_mode,
        data_sources=junior.data_sources,
        gemini_model=junior.gemini_model,
        reasoning=junior.reasoning,
        price_target_usd=junior.price_target_usd,
        price_target_horizon_months=junior.price_target_horizon_months,
        price_target_basis=junior.price_target_basis,
        headlines=junior.headlines,
        discount_rate_basis=junior.discount_rate_basis,
        discount_rate_source=junior.discount_rate_source,
        junior_latex=None,
        post_peer_junior_latex=None,
        junior_research_memo=junior_memo_original or None,
        post_peer_junior_research_memo=memo_work if rounds >= 1 else None,
        investment_recommendation=junior.investment_recommendation,
        workbook_json_excerpt=junior.workbook_json_excerpt,
        critic_memo=combined_critic,
        lead_pm_synthesis=lead_syn,
    )
