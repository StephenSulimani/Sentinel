"""Resolve a single headline price target from model JSON vs Yahoo consensus."""

from __future__ import annotations

from typing import Any


def finalize_price_target(
    model: dict[str, Any] | None,
    consensus_usd: float | None,
    consensus_note: str | None,
) -> tuple[float | None, int | None, str | None]:
    """Prefer a validated model ``PRICE_TARGET`` block; else vendor consensus mean."""
    if model:
        u = model.get("usd")
        if isinstance(u, (int, float)) and u == u and u > 0:
            hm_raw = model.get("horizon_months", 12)
            if isinstance(hm_raw, float) and hm_raw == hm_raw:
                hm = int(hm_raw)
            elif isinstance(hm_raw, int):
                hm = hm_raw
            else:
                hm = 12
            hm = max(1, min(60, hm))
            method = str(model.get("method") or "").strip()
            if not method:
                method = (
                    "Analyst synthesis from public filings, market data, and headlines; "
                    "not investment advice."
                )
            if len(method) > 520:
                method = method[:517] + "..."
            return float(u), hm, method

    if consensus_usd is not None and consensus_usd == consensus_usd and consensus_usd > 0:
        note = (consensus_note or "").strip() or (
            "Street consensus mean price target (vendor feed)."
        )
        return float(consensus_usd), 12, note

    return None, None, None
