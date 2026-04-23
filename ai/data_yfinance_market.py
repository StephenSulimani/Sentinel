"""Delayed OHLCV context via yfinance (no Alpaca / SIP subscription required)."""

from __future__ import annotations

from typing import Any

import yfinance as yf


def fetch_market_history(*, ticker: str, period: str = "6mo") -> dict[str, Any]:
    """Daily bars (last ~6 months) + last close for LLM / snapshot tables."""
    sym = ticker.strip().upper()
    out: dict[str, Any] = {"symbol": sym, "source": "yfinance", "period": period}
    try:
        tkr = yf.Ticker(sym)
        hist = tkr.history(period=period, interval="1d", auto_adjust=True)
    except Exception as exc:  # noqa: BLE001
        return {**out, "error": str(exc)}

    if hist is None or hist.empty:
        return {**out, "error": "no price history returned"}

    tail = hist.tail(130)
    bars: list[dict[str, Any]] = []
    for ts, row in tail.iterrows():
        bars.append(
            {
                "date": str(ts.date()),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"]),
            }
        )

    first = bars[0] if bars else None
    last = bars[-1] if bars else None
    perf = None
    if first and last and first.get("close") and last.get("close"):
        try:
            perf = (float(last["close"]) / float(first["close"])) - 1.0
        except Exception:  # noqa: BLE001
            perf = None

    info: dict[str, Any] = {}
    try:
        raw = tkr.info or {}
        for k in (
            "shortName",
            "longName",
            "exchange",
            "currency",
            "marketCap",
            "trailingPE",
            "fiftyTwoWeekHigh",
            "fiftyTwoWeekLow",
            "currentPrice",
            "regularMarketPrice",
            "targetMeanPrice",
            "targetMedianPrice",
            "targetHighPrice",
            "targetLowPrice",
            "numberOfAnalystOpinions",
            "beta",
        ):
            if k in raw:
                info[k] = raw.get(k)
    except Exception:  # noqa: BLE001
        pass

    return {
        **out,
        "info": info,
        "bars_daily_tail": bars[-60:] if len(bars) > 60 else bars,
        "window_return_approx": perf,
        "window_note": "Return approximated over returned daily bars (yfinance delayed data).",
    }


def infer_discount_rate_for_bridge(info: dict[str, Any]) -> tuple[float, str]:
    """Single discount rate for the mechanical 5y NPV illustration (not a formal WACC).

    Uses a simple CAPM-style sketch when vendor beta is usable; otherwise 10%.
    The narrative analyst may adopt a different rate in prose and price-target
    methodology.
    """
    rf = 0.045
    erp = 0.055
    baseline = 0.10
    beta_raw = info.get("beta")
    if isinstance(beta_raw, (int, float)) and beta_raw == beta_raw and 0.25 <= float(beta_raw) <= 2.5:
        b = float(beta_raw)
        raw = rf + b * erp
        rate = min(0.16, max(0.08, raw))
        return rate, (
            f"Inferred ~{rate * 100:.1f}% from CAPM sketch (Rf {rf * 100:.1f}%, ERP {erp * 100:.1f}%, "
            f"beta {b:.2f}); mechanical sensitivity only."
        )
    return baseline, (
        "Inferred 10.0% baseline (beta missing or out of range); mechanical sensitivity only."
    )


def reference_last_close_usd(snapshot: dict[str, Any]) -> float | None:
    """Best-effort spot / last daily close in USD (or listing currency) from a ``fetch_market_history`` payload."""
    if snapshot.get("error"):
        return None
    bars = snapshot.get("bars_daily_tail") or []
    if bars:
        c = bars[-1].get("close")
        if isinstance(c, (int, float)) and c == c and c > 0:
            return float(c)
    info = snapshot.get("info") or {}
    for k in ("regularMarketPrice", "currentPrice"):
        v = info.get(k)
        if isinstance(v, (int, float)) and v == v and v > 0:
            return float(v)
    return None


def consensus_price_target_from_info(info: dict[str, Any]) -> tuple[float | None, str | None]:
    """Yahoo Finance mean analyst target when exposed by yfinance (not an independent valuation)."""
    mean = info.get("targetMeanPrice")
    if not isinstance(mean, (int, float)) or mean != mean or mean <= 0:
        return None, None
    n = info.get("numberOfAnalystOpinions")
    suffix = ""
    if isinstance(n, int) and n > 0:
        suffix = f" (n={n} estimates in vendor consensus feed)"
    return float(mean), (
        "Street consensus mean price target from vendor feed"
        f"{suffix}; not an independent valuation—verify with primary sources."
    )


def consensus_price_target_from_snapshot(snapshot: dict[str, Any]) -> tuple[float | None, str | None]:
    if snapshot.get("error"):
        return None, None
    return consensus_price_target_from_info(snapshot.get("info") or {})
