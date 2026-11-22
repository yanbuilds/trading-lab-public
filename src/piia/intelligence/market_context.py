"""Deterministic multi-timeframe context for research, never a trade signal."""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _ema_relation(part: dict[str, Any]) -> str:
    if not part.get("ok"):
        return "unavailable"
    fast, slow = _finite(part.get("ema20")), _finite(part.get("ema50"))
    if fast is None or slow is None or fast <= 0 or slow <= 0:
        return "unavailable"
    return "up" if fast > slow else "down" if fast < slow else "equal"


def _daily_range_position(hourly: dict, daily: dict) -> dict:
    """Compare a closed hour with the preceding UTC-day range, without a signal."""
    unavailable = {"status": "unavailable", "note": "Requires valid and time-consistent closed hourly and daily data."}
    if not hourly.get("ok") or not daily.get("ok"):
        return unavailable
    last = _finite(hourly.get("last_close"))
    high = _finite(daily.get("last_20_closed_daily_high"))
    low = _finite(daily.get("last_20_closed_daily_low"))
    if last is None or low is None or high is None or min(last, low) <= 0 or high < low:
        return unavailable
    try:
        hour_close = datetime.fromisoformat(hourly["last_closed_bar_close_at_utc"])
        day_close = datetime.fromisoformat(daily["last_closed_at_utc"])
        if hour_close.tzinfo is None or day_close.tzinfo is None or not timedelta(0) <= hour_close - day_close < timedelta(days=1):
            return unavailable
    except (KeyError, TypeError, ValueError, OverflowError):
        return unavailable
    status = "above" if last > high else "below" if last < low else "within_or_at_boundary"
    return {"status": status, "hour_close": last, "range_high": high, "range_low": low,
            "range_start_utc": daily.get("last_20_closed_daily_start_utc"),
            "range_end_utc": daily["last_closed_at_utc"], "hour_close_utc": hourly["last_closed_bar_close_at_utc"],
            "hour_included_in_range": hour_close == day_close,
            "note": "Historical closing-price relationship, not the high/low breakout entry rule, forecast, or order; at UTC midnight the hour belongs to the range."}


def describe_market(indicators: dict[str, Any] | None, daily: dict[str, Any] | None) -> dict[str, Any]:
    """Describe observed closed-bar relationships without forecasting or orders."""
    hourly = indicators or {}
    four = hourly.get("four_hour_trend") or {}
    day = daily or {}
    relations = {"1h": _ema_relation(hourly), "4h": _ema_relation(four), "1d": _ema_relation(day)}
    values = list(relations.values())
    if "unavailable" in values:
        alignment = "incomplete"
    elif all(v == "up" for v in values):
        alignment = "aligned_up"
    elif all(v == "down" for v in values):
        alignment = "aligned_down"
    else:
        alignment = "mixed_or_equal"
    valid_hourly = bool(hourly.get("ok"))
    rsi = _finite(hourly.get("rsi14_wilder")) if valid_hourly else None
    atr_pct = _finite(hourly.get("atr14_pct_of_close")) if valid_hourly else None
    high = _finite(hourly.get("range_24h_high")) if valid_hourly else None
    low = _finite(hourly.get("range_24h_low")) if valid_hourly else None
    if rsi is not None and not 0 <= rsi <= 100:
        rsi = None
    if atr_pct is not None and atr_pct < 0:
        atr_pct = None
    if high is None or low is None or low <= 0 or high < low:
        high = low = None
    return {
        "kind": "deterministic_closed_bar_context_not_signal",
        "ema_relationships": relations, "timeframe_alignment": alignment,
        "as_of_utc": {"1h": hourly.get("last_closed_bar_close_at_utc") if valid_hourly else None,
                      "4h": four.get("last_closed_at_utc") if four.get("ok") else None,
                      "1d": day.get("last_closed_at_utc") if day.get("ok") else None},
        "rsi14": rsi, "atr14_pct": atr_pct,
        "observed_24h_low": low, "observed_24h_high": high,
        "closed_hour_vs_20_completed_days": _daily_range_position(hourly, day),
        "limitations": ["EMA relationships describe past closed bars, not predicted direction or entry permission.",
                        "The 24h range is historical, not a forecast or calibrated price target.",
                        "Funding, open interest, macro and data freshness require separate review."],
        "order_permissions": False,
    }


def refresh_research_context(item: dict[str, Any], now: datetime) -> tuple[dict[str, Any], dict[str, Any]]:
    """Revalidate frozen daily candles before sending multitimeframe context to a model."""
    from piia.intelligence.research_packet import _daily_context
    old = item.get("daily_context") or {}
    if not isinstance(old, dict) or not old.get("ok"):
        daily = {"ok": False, "reason": "daily_context_unavailable_in_snapshot"}
    else:
        daily = _daily_context(item.get("all_closed_daily_candles") or [], [], now)
    return daily, describe_market(item.get("indicators_prefix_only"), daily)
