"""Equity research pipeline — Junior → Critic → Lead PM (see ``research_graph`` for Junior-only graph)."""

from __future__ import annotations

from portfolio_pipeline import run_portfolio_research_pipeline
from research_graph import run_equity_research_graph

# Stable import path for Flask and tests (full multi-agent stream).
generate_full_report = run_portfolio_research_pipeline

__all__ = ["generate_full_report", "run_equity_research_graph", "run_portfolio_research_pipeline"]
