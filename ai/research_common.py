"""Shared helpers for equity research orchestration (LangGraph nodes + legacy callers)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

from jinja2 import Environment

from report_builder import ReportStep, get_template_env


def append_step(
    steps: list[ReportStep],
    sid: str,
    label: str,
    status: str,
    detail: str = "",
) -> None:
    steps.append(ReportStep(id=sid, label=label, status=status, detail=detail))


def emit_phase(
    emit: Callable[[dict[str, Any]], None] | None,
    payload: dict[str, Any],
) -> None:
    if emit is not None:
        emit(payload)


def pick_ocf_base(facts_summary: dict[str, Any]) -> tuple[float, str, bool]:
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


def render_deterministic_tex(
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
    env: Environment | None = None,
) -> str:
    tmpl = (env or get_template_env()).get_template("report.tex.j2")
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
