"""Deterministic labels from a current ALFRED snapshot. Not a historical vintage replay.

The existing fetch uses a conservative date vintage at least two UTC days old and
does not prove intraday release times. Today's snapshot must not be backfilled
into past decisions. Missing historical vintages block macro-enhanced alpha claims.
"""
from __future__ import annotations

import math
from typing import Any


def classify_macro_snapshot(context: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "kind": "deterministic_macro_snapshot_not_agent",
        "status": "missing",
        "curve_10y_minus_2y": None,
        "curve_label": "unavailable",
        "real_rate_10y": None,
        "real_rate_label": "unavailable",
        "fed_funds": None,
        "missing_series": [],
        "stale_series": [],
        "historical_replay_supported": False,
        "macro_enhanced_historical_alpha": "blocked",
        "block_reason": "Current ALFRED adapter is a conservative date-vintage snapshot, not a point-in-time history.",
        "order_permissions": False,
    }
    if not isinstance(context, dict) or not context.get("series"):
        out["missing_series"] = ["DGS10", "DGS2", "DFII10", "DFF"]
        out["reason"] = "macro_context_missing"
        return out
    if context.get("historical_replay_supported") is True:
        # Refuse to silently flip this; the official adapter must stay honest.
        out["reason"] = "adapter_must_not_claim_historical_replay"
        return out
    series = context.get("series") or {}
    errors = context.get("errors") or {}
    for name in ("DGS10", "DGS2", "DFII10", "DFF"):
        item = series.get(name)
        if not isinstance(item, dict) or item.get("value") is None:
            out["missing_series"].append(name)
            continue
        try:
            value = float(item["value"])
        except (TypeError, ValueError):
            out["missing_series"].append(name)
            continue
        if not math.isfinite(value):
            out["missing_series"].append(name)
            continue
        if item.get("stale"):
            out["stale_series"].append(name)
        if name == "DFII10":
            out["real_rate_10y"] = value
            out["real_rate_label"] = "positive" if value > 0 else "negative" if value < 0 else "zero"
        if name == "DFF":
            out["fed_funds"] = value
    if "DGS10" not in out["missing_series"] and "DGS2" not in out["missing_series"]:
        spread = float(series["DGS10"]["value"]) - float(series["DGS2"]["value"])
        out["curve_10y_minus_2y"] = spread
        out["curve_label"] = "upward" if spread > 0 else "inverted" if spread < 0 else "flat"
    if out["missing_series"]:
        out["status"] = "incomplete"
    elif out["stale_series"]:
        out["status"] = "stale"
    else:
        out["status"] = "available"
    out["retrieved_at_utc"] = context.get("retrieved_at_utc")
    out["vintage_date"] = context.get("vintage_date")
    out["intraday_release_verified"] = bool(context.get("intraday_release_verified"))
    return out
