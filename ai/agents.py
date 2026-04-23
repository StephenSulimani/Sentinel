"""Multi-agent identifiers (SSE + UI)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Stable machine ids for stream payloads and state
AGENT_JUNIOR_RESEARCHER = "junior_researcher"
AGENT_CRITIC = "critic"
AGENT_LEAD_PORTFOLIO_MANAGER = "lead_portfolio_manager"

AGENT_LABELS: dict[str, str] = {
    AGENT_JUNIOR_RESEARCHER: "Junior Researcher",
    AGENT_CRITIC: "The Critic",
    AGENT_LEAD_PORTFOLIO_MANAGER: "The Lead Portfolio Manager",
}


def agent_label(agent_id: str | None) -> str:
    if not agent_id:
        return "Agent"
    return AGENT_LABELS.get(agent_id, agent_id.replace("_", " ").title())


def tag_emit(
    emit: Callable[[dict[str, Any]], None] | None,
    agent_id: str,
) -> Callable[[dict[str, Any]], None] | None:
    """Tag stream payloads so the UI can show which agent is active."""

    if emit is None:
        return None

    def _wrapped(ev: dict[str, Any]) -> None:
        d = dict(ev)
        t = d.get("type")
        if t in ("phase", "reasoning", "reasoning_delta", "price_target", "gemini_request"):
            d.setdefault("agent", agent_id)
        emit(d)

    return _wrapped
