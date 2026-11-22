"""Deterministic liquidity and positioning research states.

This is a research component, not an independent LLM forecast and not a trading
agent. Official states require mandatory joint conditions. Missing critical
evidence yields unknown, never normal. Suspected labels are distinct from
confirmed states. Thresholds are frozen, uncalibrated heuristics.

States: fragility_building, deleveraging, deleveraging_easing, normal, unknown.
Observational labels only: suspected_deleveraging, suspected_deleveraging_easing.

XAUUSDT is the Bitget USDT-M linear perpetual and is not XAUUSD. DTWEXBGS is the
Fed broad dollar index and is not ICE DXY. Single-venue Bitget coverage is never
described as market-wide. published_at / available_at stay unknown unless a
documented bound exists; retrieved_at is never copied into those fields.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import hashlib
import math

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = ROOT / "config" / "liquidity_positioning_v1.yaml"
OFFICIAL_STATES = {
    "fragility_building",
    "deleveraging",
    "deleveraging_easing",
    "normal",
    "unknown",
}
SUSPECTED_LABELS = {"suspected_deleveraging", "suspected_deleveraging_easing"}
RISK_ASSETS = ("BTCUSDT", "NDX100USDT")
GOLD = "XAUUSDT"
BTC = "BTCUSDT"


def _aware(value: datetime | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timezone-aware time required")
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def load_config(path: Path | None = None) -> dict[str, Any]:
    target = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = target.read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, dict) or data.get("version") != "liquidity_positioning_v1":
        raise ValueError("unexpected liquidity positioning config")
    if data.get("calibrated") is True:
        raise ValueError("this module refuses a config marked calibrated")
    data["config_path"] = str(target)
    data["config_sha256"] = hashlib.sha256(raw).hexdigest()
    return data


def condition(name: str, status: str, **detail: Any) -> dict[str, Any]:
    if status not in {"met", "unmet", "unknown", "supporting", "counterevidence"}:
        raise ValueError("invalid condition status")
    return {"name": name, "status": status, **detail}


def conservative_alfred_available_at(vintage_date: str) -> datetime:
    """Known conservative bound: end of the vintage UTC date, not the exact release tick."""
    day = datetime.strptime(vintage_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return day + timedelta(days=1)


def usable_for_replay(record: dict[str, Any] | None, cutoff: datetime) -> bool:
    """Historical replay may use a record only when available_at is known and not after cutoff."""
    if not isinstance(record, dict):
        return False
    raw = record.get("available_at_utc")
    if not raw or not record.get("available_at_known"):
        return False
    try:
        available = _aware(raw)
    except ValueError:
        return False
    return available <= cutoff


def reject_future(stamp: datetime | str | None, cutoff: datetime, label: str) -> None:
    if stamp is None:
        return
    if _aware(stamp) > cutoff:
        raise ValueError(f"future_data_rejected:{label}")


def _close_return(candles: list[dict[str, Any]], *, hours: int, cutoff: datetime) -> dict[str, Any]:
    out = {"status": "unknown", "return_pct": None, "reason": "insufficient_aligned_closed_candles"}
    if not candles:
        return out
    closed = []
    for candle in candles:
        close_at = _aware(candle["close_at_utc"])
        reject_future(close_at, cutoff, "candle")
        if close_at <= cutoff:
            closed.append((close_at, float(candle["close"]), candle.get("volume"), candle.get("open")))
    closed.sort(key=lambda item: item[0])
    if len(closed) < hours + 1:
        return out
    latest = closed[-1]
    target = latest[0] - timedelta(hours=hours)
    prior = next((item for item in reversed(closed[:-1]) if item[0] == target), None)
    if prior is None or prior[1] <= 0 or latest[1] <= 0:
        out["reason"] = "missing_joint_timestamp_or_non_positive_close"
        return out
    ret = (latest[1] / prior[1] - 1) * 100
    if not math.isfinite(ret):
        return out
    out.update(status="ok", return_pct=ret, from_close_at_utc=_iso(prior[0]),
               to_close_at_utc=_iso(latest[0]), last_close=latest[1], reason=None)
    return out


def _volume_demand(candles: list[dict[str, Any]], *, hours: int, cutoff: datetime) -> dict[str, Any]:
    out = {"status": "unknown", "up_volume": None, "down_volume": None, "ratio": None,
           "reason": "insufficient_aligned_volume", "not_orderbook": True}
    closed = []
    for candle in candles or []:
        close_at = _aware(candle["close_at_utc"])
        reject_future(close_at, cutoff, "candle")
        if close_at <= cutoff:
            volume = candle.get("volume")
            if volume is None or isinstance(volume, bool) or not math.isfinite(float(volume)):
                continue
            closed.append((close_at, float(candle["close"]), float(candle.get("open") or candle["close"]), float(volume)))
    if not closed:
        return out
    latest = closed[-1][0]
    window = [item for item in closed if latest - item[0] <= timedelta(hours=hours)]
    if len(window) < max(6, hours // 4):
        return out
    up = sum(v for _t, close, open_, v in window if close > open_)
    down = sum(v for _t, close, open_, v in window if close < open_)
    ratio = None if down <= 0 else up / down
    out.update(status="ok" if ratio is not None else "unknown", up_volume=up, down_volume=down,
               ratio=ratio, reason=None if ratio is not None else "no_down_bar_volume")
    return out


def _new_low_held(candles: list[dict[str, Any]], *, hours: int, cutoff: datetime) -> dict[str, Any]:
    """Price holds if the latest close is not below the lookback low including prior bars."""
    out = {"status": "unknown", "holds": None, "reason": "insufficient_aligned_closed_candles"}
    closed = []
    for candle in candles or []:
        close_at = _aware(candle["close_at_utc"])
        reject_future(close_at, cutoff, "candle")
        if close_at <= cutoff:
            closed.append((close_at, float(candle["low"]), float(candle["close"])))
    if len(closed) < hours:
        return out
    latest = closed[-1]
    prior = [item for item in closed[:-1] if latest[0] - item[0] <= timedelta(hours=hours)]
    if not prior:
        return out
    prior_low = min(item[1] for item in prior)
    holds = latest[2] > prior_low
    out.update(status="ok", holds=holds, prior_low=prior_low, last_close=latest[2], reason=None)
    return out


def _series_change(points: list[dict[str, Any]], *, date_key: str, value_key: str,
                   steps: int, cutoff: datetime, available_bound: datetime | None,
                   replay: bool) -> dict[str, Any]:
    out = {"status": "unknown", "change": None, "reason": "insufficient_joint_observations"}
    usable = []
    for point in points:
        raw = point.get(date_key)
        if raw is None:
            continue
        if "T" in str(raw):
            stamp = _aware(raw)
        else:
            stamp = datetime.strptime(str(raw), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        reject_future(stamp, cutoff, date_key)
        record = {**point, "available_at_utc": point.get("available_at_utc") or ( _iso(available_bound) if available_bound else None),
                  "available_at_known": bool(point.get("available_at_known") or available_bound)}
        if replay and not usable_for_replay(record, cutoff):
            continue
        if stamp <= cutoff:
            value = point.get(value_key)
            if isinstance(value, bool) or value is None or not math.isfinite(float(value)):
                continue
            usable.append((stamp, float(value)))
    usable.sort(key=lambda item: item[0])
    if len(usable) < steps + 1:
        if replay and points and out["reason"] == "insufficient_joint_observations":
            out["reason"] = "insufficient_known_available_at_or_aligned_points"
        return out
    first, last = usable[-(steps + 1)], usable[-1]
    change = last[1] - first[1]
    out.update(status="ok", change=change, from_at=_iso(first[0]), to_at=_iso(last[0]),
               from_value=first[1], to_value=last[1], reason=None)
    return out


def real_yield_change(history: dict[str, Any] | None, *, cutoff: datetime, replay: bool) -> dict[str, Any]:
    series = ((history or {}).get("series") or {}).get("DFII10") or {}
    vintage = series.get("vintage_date") or (history or {}).get("vintage_date")
    bound = conservative_alfred_available_at(vintage) if vintage else None
    points = []
    for item in series.get("observations") or []:
        rec = dict(item)
        if bound is not None:
            rec["available_at_utc"] = _iso(bound)
            rec["available_at_known"] = True
            rec["available_at_exact"] = False
            rec["published_at_utc"] = None
            rec["published_at_known"] = False
        points.append(rec)
    out = _series_change(points, date_key="observation_date", value_key="value", steps=5,
                         cutoff=cutoff, available_bound=bound, replay=replay)
    if out["status"] == "ok":
        out["change_bps"] = out["change"] * 100.0
        out["unit"] = "percentage_points_times_100_equals_bps"
        out["series_id"] = "DFII10"
    out["dollar_note"] = None
    return out


def broad_dollar_change(history: dict[str, Any] | None, *, cutoff: datetime, replay: bool) -> dict[str, Any]:
    series = ((history or {}).get("series") or {}).get("DTWEXBGS") or {}
    vintage = series.get("vintage_date") or (history or {}).get("vintage_date")
    bound = conservative_alfred_available_at(vintage) if vintage else None
    points = []
    for item in series.get("observations") or []:
        rec = dict(item)
        if bound is not None:
            rec["available_at_utc"] = _iso(bound)
            rec["available_at_known"] = True
            rec["available_at_exact"] = False
            rec["published_at_utc"] = None
            rec["published_at_known"] = False
        points.append(rec)
    out = _series_change(points, date_key="observation_date", value_key="value", steps=5,
                         cutoff=cutoff, available_bound=bound, replay=replay)
    if out["status"] == "ok" and out["from_value"]:
        out["change_pct"] = (out["to_value"] / out["from_value"] - 1) * 100
    out["series_id"] = "DTWEXBGS"
    out["is_not"] = "ICE_DXY"
    out["note"] = "Fed broad dollar index DTWEXBGS; this is not ICE DXY."
    return out


def etf_flow_weakening(etf: dict[str, Any] | None, *, cutoff: datetime, days: int, replay: bool) -> dict[str, Any]:
    out = {"status": "unknown", "weakening": None, "reason": "insufficient_complete_etf_days",
           "dashes_not_zero": True, "applies_to": [BTC]}
    if not etf:
        return out
    rows = []
    for day in etf.get("complete_daily_flows") or []:
        if not day.get("complete"):
            continue
        stamp = datetime.strptime(day["flow_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        reject_future(stamp, cutoff, "etf_flow_date")
        record = {
            **day,
            "available_at_utc": day.get("available_at_utc"),
            "available_at_known": bool(day.get("available_at_known")),
            "published_at_utc": day.get("published_at_utc"),
        }
        if replay and not usable_for_replay(record, cutoff):
            continue
        if stamp <= cutoff:
            value = day.get("sum_of_reported_funds_millions_usd")
            if value is None or isinstance(value, bool) or not math.isfinite(float(value)):
                continue
            rows.append((stamp, float(value)))
    rows.sort(key=lambda item: item[0])
    if replay and etf.get("complete_daily_flows") and len(rows) < 2 * days:
        out["reason"] = "etf_available_at_unknown_or_incomplete_for_replay"
        return out
    if len(rows) < 2 * days:
        return out
    recent = rows[-days:]
    previous = rows[-2 * days:-days]
    recent_sum = sum(v for _t, v in recent)
    previous_sum = sum(v for _t, v in previous)
    out.update(status="ok", weakening=recent_sum < previous_sum, recent_sum=recent_sum,
               previous_sum=previous_sum, recent_dates=[_iso(t) for t, _v in recent],
               previous_dates=[_iso(t) for t, _v in previous], reason=None)
    return out


def gold_real_rate_sensitivity(gold_candles: list[dict[str, Any]], yield_history: dict[str, Any] | None,
                               *, cutoff: datetime, window: int, min_pairs: int, replay: bool) -> dict[str, Any]:
    """Lagged rolling association; correlation is not causation. XAUUSDT only, not XAUUSD."""
    out = {
        "status": "unknown", "beta": None, "pairs": 0, "reason": "insufficient_joint_timestamps",
        "instrument": GOLD, "instrument_is_not": "XAUUSD",
        "instrument_type": "usdt_linear_perpetual",
        "not_causation": True,
        "note": "Bitget XAUUSDT USDT-M perpetual returns are not interchangeable with XAUUSD spot or CFD specs.",
    }
    by_day: dict[datetime, tuple[datetime, float]] = {}
    for candle in gold_candles or []:
        close_at = _aware(candle["close_at_utc"])
        reject_future(close_at, cutoff, "gold_candle")
        if close_at > cutoff or float(candle["close"]) <= 0:
            continue
        day = close_at.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        current = by_day.get(day)
        if current is None or close_at >= current[0]:
            by_day[day] = (close_at, float(candle["close"]))
    daily = {day: price for day, (_t, price) in by_day.items()}
    series = ((yield_history or {}).get("series") or {}).get("DFII10") or {}
    vintage = series.get("vintage_date") or (yield_history or {}).get("vintage_date")
    bound = conservative_alfred_available_at(vintage) if vintage else None
    yields: dict[datetime, float] = {}
    for item in series.get("observations") or []:
        day = datetime.strptime(item["observation_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        record = {"available_at_utc": _iso(bound) if bound else None, "available_at_known": bound is not None}
        if replay and not usable_for_replay(record, cutoff):
            continue
        if day <= cutoff:
            yields[day] = float(item["value"])
    # Lag yields by one observation date: gold return on day t vs yield change ending t-1,
    # because exact publication time is unverified.
    gold_days = sorted(daily)
    y_days = sorted(yields)
    pairs: list[tuple[float, float]] = []
    for i in range(1, len(gold_days)):
        d1, d0 = gold_days[i], gold_days[i - 1]
        if d1 - d0 != timedelta(days=1) and (d1 - d0).days != 1:
            # allow weekend gaps in gold if next yield dates still lag
            pass
        if daily[d0] <= 0:
            continue
        # Find yield observations at or before d0 and the prior yield observation.
        lagged = [d for d in y_days if d <= d0 - timedelta(days=1)]
        if len(lagged) < 2:
            continue
        y1, y0 = lagged[-1], lagged[-2]
        dy = yields[y1] - yields[y0]
        gret = daily[d1] / daily[d0] - 1
        pairs.append((dy, gret))
    pairs = pairs[-window:]
    out["pairs"] = len(pairs)
    if len(pairs) < min_pairs:
        if replay:
            out["reason"] = "insufficient_joint_timestamps_or_unknown_available_at"
        return out
    mean_x = sum(x for x, _y in pairs) / len(pairs)
    mean_y = sum(y for _x, y in pairs) / len(pairs)
    var_x = sum((x - mean_x) ** 2 for x, _y in pairs)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    if var_x <= 0:
        out["reason"] = "zero_variance_in_real_yield_changes"
        return out
    beta = cov / var_x
    if not math.isfinite(beta):
        return out
    out.update(status="ok", beta=beta, reason=None, window=len(pairs),
               interpretation="negative_beta_means_XAUUSDT_tended_to_fall_when_lagged_real_yields_rose")
    return out


def _oi_feature(history: dict[str, Any] | None, snapshot_oi: dict[str, Any] | None,
                *, cutoff: datetime, replay: bool) -> dict[str, Any]:
    out = {"status": "unknown", "base_change_24h_pct": None, "value_change_24h_pct": None,
           "price_contaminated": False, "reason": "missing_complete_base_unit_history",
           "coverage": "single_venue_bitget_usdt_linear", "marketwide_coverage": False,
           "unit": "base_currency"}
    source = history or {}
    if replay:
        rows = [row for row in source.get("observations") or [] if usable_for_replay(row, cutoff)]
        if len(rows) < 25:
            out["reason"] = "oi_available_at_unknown_or_incomplete_for_replay"
            return out
        # cannot use live snapshot change in replay if availability unknown
    change = None
    value_change = None
    contaminated = False
    if source.get("history_complete_24h") and source.get("change_24h_base_pct") is not None and not replay:
        reject_future(source.get("latest_period_end_utc"), cutoff, "oi")
        if source.get("latest_period_end_utc") and _aware(source["latest_period_end_utc"]) <= cutoff:
            change = float(source["change_24h_base_pct"])
            if source.get("change_24h_value_pct") is not None:
                value_change = float(source["change_24h_value_pct"])
            contaminated = bool(source.get("price_contaminated_value_change"))
    elif snapshot_oi and snapshot_oi.get("history_complete_24h") and snapshot_oi.get("change_24h_pct") is not None and not replay:
        reject_future(snapshot_oi.get("period_end_utc"), cutoff, "oi")
        if snapshot_oi.get("period_end_utc") and _aware(snapshot_oi["period_end_utc"]) <= cutoff:
            change = float(snapshot_oi["change_24h_pct"])
            out["unit_source"] = snapshot_oi.get("unit_and_scope") or "base_currency"
    if change is None or not math.isfinite(change):
        return out
    out.update(status="ok", base_change_24h_pct=change, value_change_24h_pct=value_change,
               price_contaminated=contaminated, reason=None)
    return out


def _funding_feature(current: dict[str, Any] | None, history: dict[str, Any] | None,
                     *, cutoff: datetime, config: dict[str, Any], replay: bool) -> dict[str, Any]:
    thr = config["thresholds"]
    out = {"status": "unknown", "interval_rate": None, "interval_hours": None,
           "persistent_support": False, "reason": "funding_interval_or_history_unknown",
           "not_annualized": True, "not_crowding_alone": True}
    interval = None
    rate = None
    if current and not replay:
        if current.get("retrieved_at_utc"):
            reject_future(current["retrieved_at_utc"], cutoff + timedelta(minutes=5), "funding_retrieved")
        interval = current.get("funding_interval_hours")
        if current.get("funding_rate_per_interval") is not None:
            rate = float(current["funding_rate_per_interval"])
        elif current.get("funding_rate") is not None:
            # Legacy v2 snapshot: interval unknown unless supplied.
            rate = float(current["funding_rate"])
    rows = []
    for item in (history or {}).get("observations") or []:
        reject_future(item.get("settlement_at_utc") or item.get("observed_at_utc"), cutoff, "funding")
        if replay and not usable_for_replay(item, cutoff):
            continue
        stamp = _aware(item.get("settlement_at_utc") or item.get("observed_at_utc"))
        if stamp <= cutoff:
            rows.append((stamp, float(item["funding_rate_per_interval"])))
    rows.sort(key=lambda item: item[0])
    if interval is None:
        interval = (history or {}).get("inferred_interval_hours")
    if rate is None and rows:
        rate = rows[-1][1]
    needed = int(thr["funding_support_consecutive_settlements"])
    persistent = False
    if rows and interval and len(rows) >= needed:
        tail = rows[-needed:]
        persistent = all(abs(v) >= float(thr["funding_support_interval_rate_abs_min"]) for _t, v in tail)
        signs = [math.copysign(1, v) if v else 0 for _t, v in tail]
        persistent = persistent and len(set(signs)) == 1 and signs[0] != 0
    if rate is None or not interval:
        if replay:
            out["reason"] = "funding_available_at_unknown_or_interval_missing"
        return out
    out.update(status="ok", interval_rate=rate, interval_hours=interval,
               persistent_support=persistent, reason=None,
               support_threshold_per_interval=float(thr["funding_support_interval_rate_abs_min"]))
    return out


def _liquidation_feature(payload: dict[str, Any] | None, *, cutoff: datetime,
                         config: dict[str, Any], replay: bool) -> dict[str, Any]:
    out = {"status": "unknown", "rising": None, "certified_forced": False,
           "reason": "liquidations_missing_or_window_too_short",
           "coverage": "single_venue_bitget_usdt_linear", "marketwide_coverage": False,
           "docs_window": "about_last_3_days"}
    if not payload:
        return out
    if replay:
        events = [e for e in payload.get("events") or [] if usable_for_replay(e, cutoff)]
        if not events:
            out["reason"] = "liquidation_available_at_unknown_cannot_replay"
            return out
    latest = payload.get("amount_sum_latest_24h_base")
    previous = payload.get("amount_sum_previous_24h_base")
    if latest is None or previous is None:
        return out
    if previous <= 0 and latest <= 0:
        out.update(status="ok", rising=False, reason="no_reported_liquidations_in_48h",
                   latest_24h=latest, previous_24h=previous)
        return out
    if previous <= 0:
        rising = latest > 0
        ratio = None
    else:
        ratio = latest / previous
        rising = ratio >= float(config["thresholds"]["liquidation_rise_ratio_min"])
    out.update(status="ok", rising=rising, ratio=ratio, latest_24h=latest, previous_24h=previous,
               reason=None, certified_forced=False,
               unit_conflict=True)
    return out


def _book_feature(books: dict[str, Any] | None, symbol: str) -> dict[str, Any]:
    inst = ((books or {}).get("instruments") or {}).get(symbol) or {}
    book = inst.get("orderbook") or inst
    out = {"status": "unknown", "momentary_only": True, "not_realized_demand": True,
           "reason": "no_current_book_snapshot", "symbol": symbol}
    if not book or not book.get("ok"):
        out["reason"] = book.get("reason") if isinstance(book, dict) else out["reason"]
        return out
    out.update(status="ok", reason=None, spread=book.get("spread"),
               visible_bid_size_sum=book.get("visible_bid_size_sum"),
               visible_ask_size_sum=book.get("visible_ask_size_sum"),
               best_bid=book.get("best_bid"), best_ask=book.get("best_ask"),
               note="Visible depth is a momentary snapshot and is not realized buy demand.")
    return out


def _polymarket_feature(payload: dict[str, Any] | None, *, cutoff: datetime,
                        config: dict[str, Any], replay: bool) -> dict[str, Any]:
    thr = config["thresholds"]
    out = {"status": "unknown", "reliable_macro_inference": False, "changes": [],
           "reason": "no_usable_prediction_market_history"}
    events = (payload or {}).get("events") or []
    changes = []
    thin = 0
    used = 0
    for event in events:
        end = event.get("end_date") or event.get("endDate") or event.get("closes_at_utc")
        if end:
            try:
                end_at = _aware(end) if "T" in str(end) else datetime.strptime(str(end)[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                horizon_ok = end_at >= cutoff
            except ValueError:
                horizon_ok = False
        else:
            horizon_ok = False
        for market in event.get("markets") or [event]:
            for outcome in market.get("outcomes") or []:
                change = outcome.get("change_24h") or outcome.get("price_change") or {}
                if change.get("status") != "ok" or change.get("change_24h_pp") is None:
                    continue
                record = {"available_at_utc": change.get("to_timestamp_utc") or change.get("available_at_utc"),
                          "available_at_known": bool(change.get("available_at_known"))}
                # History point time is not a verified publication clock; do not invent availability.
                if replay and not usable_for_replay({**change, **record}, cutoff):
                    continue
                used += 1
                spread = outcome.get("spread") or (outcome.get("book") or {}).get("spread")
                size = outcome.get("visible_size") or outcome.get("bid_size") or 0
                is_thin = False
                try:
                    if spread is not None and float(spread) > float(thr["polymarket_thin_spread_max"]):
                        is_thin = True
                    if size and float(size) < float(thr["polymarket_min_visible_size"]):
                        is_thin = True
                except (TypeError, ValueError):
                    is_thin = True
                if is_thin or not horizon_ok:
                    thin += 1
                    continue
                changes.append({
                    "event": event.get("title") or event.get("event_id"),
                    "change_24h_pp": change["change_24h_pp"],
                    "unit": "percentage_points",
                    "horizon_aligned": horizon_ok,
                })
    if not used and not changes:
        if replay:
            out["reason"] = "prediction_market_available_at_unknown_or_no_history"
        return out
    out.update(status="ok", changes=changes, thin_or_misaligned=thin,
               reliable_macro_inference=bool(changes), reason=None,
               note="Thin books or misaligned horizons are not used to infer macro conditions.")
    return out


def _prior_active(prior: dict[str, Any] | None, *, cutoff: datetime, max_age_hours: float) -> dict[str, Any]:
    out = {"present": False, "official_deleveraging": False, "suspected_deleveraging": False,
           "expired": False, "reason": "no_prior_state"}
    if not prior or not prior.get("state"):
        return out
    as_of = prior.get("as_of_utc") or prior.get("cutoff_at_utc")
    if not as_of:
        out["reason"] = "prior_state_missing_as_of"
        return out
    stamp = _aware(as_of)
    if stamp > cutoff:
        raise ValueError("future_data_rejected:prior_state")
    expired = cutoff - stamp > timedelta(hours=max_age_hours)
    state = prior.get("official_state") or prior.get("state")
    suspected = prior.get("observational_label")
    out.update(present=True, expired=expired, as_of_utc=_iso(stamp), state=state,
               observational_label=suspected, official_deleveraging=(state == "deleveraging" and not expired),
               suspected_deleveraging=(suspected == "suspected_deleveraging" and not expired),
               reason="expired" if expired else None)
    return out


def classify_conditions(features: dict[str, Any], config: dict[str, Any], prior: dict[str, Any]) -> dict[str, Any]:
    """Apply mandatory joint conditions. No summed score and no single-indicator states."""
    thr = config["thresholds"]
    conditions: list[dict[str, Any]] = []
    counter: list[str] = []

    def _add(*items: dict[str, Any]) -> None:
        conditions.extend(items)

    # Risk assets still strong: BTC required; NDX is supporting only.
    btc_ret = features["returns"].get(BTC) or {}
    ndx_ret = features["returns"].get("NDX100USDT") or {}
    if btc_ret.get("status") != "ok":
        _add(condition("risk_assets_still_strong", "unknown", detail=btc_ret.get("reason")))
        risk_strong = "unknown"
    else:
        risk_strong = btc_ret["return_pct"] >= float(thr["risk_asset_strong_return_24h_min"])
        _add(condition("risk_assets_still_strong", "met" if risk_strong else "unmet",
                       btc_return_24h_pct=btc_ret["return_pct"],
                       ndx_return_24h_pct=ndx_ret.get("return_pct"),
                       ndx_role="supporting_only"))
        risk_strong = "met" if risk_strong else "unmet"

    oi = features["open_interest"].get(BTC) or {}
    if oi.get("status") != "ok":
        _add(condition("leverage_crowded_base_oi", "unknown", detail=oi.get("reason")))
        crowded = "unknown"
    else:
        crowded_met = oi["base_change_24h_pct"] >= float(thr["oi_base_crowding_change_24h_pct_min"])
        crowded = "met" if crowded_met else "unmet"
        _add(condition("leverage_crowded_base_oi", crowded,
                       base_change_24h_pct=oi["base_change_24h_pct"],
                       value_change_24h_pct=oi.get("value_change_24h_pct"),
                       price_contaminated=oi.get("price_contaminated"),
                       note="USD/notional open-interest changes are not a substitute for base-unit crowding."))

    funding = features.get("funding", {}).get(BTC) or {}
    if funding.get("status") == "ok" and funding.get("persistent_support"):
        _add(condition("funding_persistent_extreme", "supporting",
                       interval_rate=funding["interval_rate"], interval_hours=funding["interval_hours"],
                       note="Funding alone is not crowding."))
    elif funding.get("status") == "ok":
        _add(condition("funding_persistent_extreme", "unmet",
                       interval_rate=funding["interval_rate"], interval_hours=funding["interval_hours"],
                       note="Funding alone is not crowding."))
    else:
        _add(condition("funding_persistent_extreme", "unknown", detail=funding.get("reason"),
                       note="Funding alone is not crowding."))

    real = features.get("real_yield") or {}
    if real.get("status") != "ok":
        _add(condition("real_yields_rising", "unknown", detail=real.get("reason")))
        yields_rising = "unknown"
    else:
        yields_rising = "met" if real["change_bps"] >= float(thr["real_yield_rise_5_obs_bps_min"]) else "unmet"
        _add(condition("real_yields_rising", yields_rising, change_bps=real["change_bps"], series="DFII10"))

    dollar = features.get("broad_dollar") or {}
    if dollar.get("status") == "ok":
        dollar_up = (dollar.get("change_pct") or 0) >= float(thr["broad_dollar_rise_5_obs_pct_min"])
        _add(condition("broad_dollar_rising_supporting", "supporting" if dollar_up else "unmet",
                       change_pct=dollar.get("change_pct"), series="DTWEXBGS", is_not="ICE_DXY"))
    else:
        _add(condition("broad_dollar_rising_supporting", "unknown", detail=dollar.get("reason"), is_not="ICE_DXY"))

    flows = features.get("etf_flows") or {}
    if flows.get("status") != "ok":
        _add(condition("etf_flows_weakening", "unknown", detail=flows.get("reason")))
        flows_weak = "unknown"
    else:
        flows_weak = "met" if flows.get("weakening") else "unmet"
        _add(condition("etf_flows_weakening", flows_weak, recent_sum=flows.get("recent_sum"),
                       previous_sum=flows.get("previous_sum"), dashes_not_zero=True))

    price = features["returns"].get(BTC) or {}
    if price.get("status") != "ok":
        _add(condition("price_falling", "unknown", detail=price.get("reason")))
        price_fall = "unknown"
    else:
        price_fall = "met" if price["return_pct"] <= float(thr["price_fall_24h_pct_max"]) else "unmet"
        _add(condition("price_falling", price_fall, btc_return_24h_pct=price["return_pct"]))

    oi_fall = "unknown"
    if oi.get("status") != "ok":
        _add(condition("base_oi_falling", "unknown", detail=oi.get("reason")))
    else:
        # Falling must be in base units; a value-only drop can be price inflation.
        falling = oi["base_change_24h_pct"] < 0
        oi_fall = "met" if falling else "unmet"
        _add(condition("base_oi_falling", oi_fall, base_change_24h_pct=oi["base_change_24h_pct"],
                       note="An open-interest drop is not by itself forced liquidation."))

    liq = features.get("liquidations", {}).get(BTC) or {}
    if liq.get("status") != "ok":
        _add(condition("liquidations_rising", "unknown", detail=liq.get("reason")))
        liq_rise = "unknown"
    else:
        liq_rise = "met" if liq.get("rising") else "unmet"
        _add(condition("liquidations_rising", liq_rise, ratio=liq.get("ratio"), certified_forced=False))

    holds = features.get("price_holds", {}).get(BTC) or {}
    if holds.get("status") != "ok":
        _add(condition("price_holds_vs_prior_low", "unknown", detail=holds.get("reason")))
        price_holds = "unknown"
    else:
        price_holds = "met" if holds.get("holds") else "unmet"
        _add(condition("price_holds_vs_prior_low", price_holds, prior_low=holds.get("prior_low"),
                       last_close=holds.get("last_close"),
                       note="Stabilization is not proof of actual buy demand."))

    demand = features.get("realized_demand", {}).get(BTC) or {}
    if demand.get("status") != "ok" or demand.get("ratio") is None:
        _add(condition("actual_buy_demand_recovering", "unknown", detail=demand.get("reason"),
                       not_orderbook=True))
        demand_ok = "unknown"
    else:
        demand_ok = "met" if demand["ratio"] >= float(thr["realized_demand_up_volume_ratio_min"]) else "unmet"
        _add(condition("actual_buy_demand_recovering", demand_ok, up_down_volume_ratio=demand["ratio"],
                       not_orderbook=True,
                       note="Closed-bar up-volume vs down-volume; the order book is momentary and is not used here."))

    macro_improve = "unknown"
    if real.get("status") != "ok":
        _add(condition("macro_conditions_improving", "unknown", detail=real.get("reason")))
    else:
        improving = real["change_bps"] <= -float(thr["real_yield_rise_5_obs_bps_min"])
        if dollar.get("status") == "ok" and dollar.get("change_pct") is not None:
            improving = improving or dollar["change_pct"] <= -float(thr["broad_dollar_rise_5_obs_pct_min"])
        macro_improve = "met" if improving else "unmet"
        _add(condition("macro_conditions_improving", macro_improve, real_yield_change_bps=real.get("change_bps"),
                       broad_dollar_change_pct=dollar.get("change_pct")))

    prior_official = bool(prior.get("official_deleveraging"))
    prior_suspected = bool(prior.get("suspected_deleveraging"))
    if prior.get("expired"):
        _add(condition("prior_active_deleveraging", "unmet", reason="prior_state_expired"))
    elif prior_official:
        _add(condition("prior_active_deleveraging", "met", kind="official"))
    elif prior_suspected:
        _add(condition("prior_active_deleveraging", "supporting", kind="suspected_only",
                       note="A suspected prior label cannot authorize official easing."))
    else:
        _add(condition("prior_active_deleveraging", "unmet" if prior.get("present") else "unknown",
                       reason=prior.get("reason")))

    def _all_met(names: tuple[str, ...]) -> str:
        statuses = {item["name"]: item["status"] for item in conditions if item["name"] in names}
        if any(statuses.get(name) == "unknown" for name in names):
            return "unknown"
        if all(statuses.get(name) == "met" for name in names):
            return "met"
        return "unmet"

    fragility = _all_met(("risk_assets_still_strong", "leverage_crowded_base_oi",
                          "real_yields_rising", "etf_flows_weakening"))
    # Official deleveraging requires price down, base OI down, and rising liquidations.
    official_delev = _all_met(("price_falling", "base_oi_falling", "liquidations_rising"))
    # Suspected only when liquidations are unknown/missing but price+OI joint holds.
    price_oi = _all_met(("price_falling", "base_oi_falling"))
    suspected_delev = price_oi == "met" and official_delev != "met" and liq_rise == "unknown"

    official_easing = "unmet"
    easing_required = ("prior_active_deleveraging", "liquidations_rising", "price_holds_vs_prior_low",
                       "actual_buy_demand_recovering", "macro_conditions_improving")
    # For official easing: prior official, liquidations NOT rising (pressure subsides),
    # price holds, actual demand, macro improves. liquidations_rising unmet means pressure subsided
    # only when liquidation status is known.
    if prior_official and liq_rise == "unmet" and price_holds == "met" and demand_ok == "met" and macro_improve == "met":
        official_easing = "met"
    elif prior_official and any(x == "unknown" for x in (liq_rise, price_holds, demand_ok, macro_improve)):
        official_easing = "unknown"
    suspected_easing = False
    if prior_suspected and not prior_official and liq_rise != "met" and price_holds == "met" and demand_ok == "met" and macro_improve == "met":
        suspected_easing = True

    if fragility == "met" and official_delev == "met":
        counter.append("fragility_and_deleveraging_conditions_both_met")
    if official_easing == "met" and official_delev == "met":
        counter.append("easing_and_active_deleveraging_both_met")
    if risk_strong == "met" and price_fall == "met":
        counter.append("risk_assets_strong_vs_price_falling")

    official_state = "unknown"
    observational_label = None
    insufficiency = []
    if counter:
        official_state = "unknown"
        insufficiency.append("contradictory_joint_conditions")
    elif official_easing == "met":
        official_state = "deleveraging_easing"
    elif official_delev == "met":
        official_state = "deleveraging"
    elif fragility == "met":
        official_state = "fragility_building"
    elif official_delev == "unknown" or fragility == "unknown" or official_easing == "unknown":
        # If a confirmed-state path is blocked only by missing evidence, stay unknown unless
        # all official paths are clearly unmet and critical current evidence is present.
        critical_unknown = any(
            item["status"] == "unknown" and item["name"] in {
                "risk_assets_still_strong", "leverage_crowded_base_oi", "real_yields_rising",
                "etf_flows_weakening", "price_falling", "base_oi_falling",
            }
            for item in conditions
        )
        if critical_unknown:
            official_state = "unknown"
            insufficiency.append("critical_evidence_unknown")
        elif official_delev == "unmet" and fragility == "unmet" and official_easing != "met":
            # Normal only when critical legs are known and unmet, not when merely guessed.
            official_state = "normal"
        else:
            official_state = "unknown"
            insufficiency.append("incomplete_or_unresolved_joint_conditions")
    elif official_delev == "unmet" and fragility == "unmet" and official_easing != "met":
        official_state = "normal"
    else:
        official_state = "unknown"

    if official_state != "deleveraging" and suspected_delev and not counter:
        observational_label = "suspected_deleveraging"
    if official_state != "deleveraging_easing" and suspected_easing and not counter:
        observational_label = "suspected_deleveraging_easing"

    if official_state == "deleveraging_easing":
        notes = ["Easing is not a bottom certification and is not an entry signal."]
    elif official_state == "deleveraging":
        notes = ["A large price decline is not by itself a bottom. Forced liquidations are not certified."]
    elif official_state == "fragility_building":
        notes = ["Risk assets can still look strong while vulnerability accumulates."]
    elif official_state == "normal":
        notes = ["Normal means the joint official states are not met; it is not a safety guarantee."]
    else:
        notes = ["Insufficient or contradictory evidence remains unknown, not normal."]

    advisory = {
        "not_an_order_or_stop_update": True,
        "does_not_change_frozen_policy": True,
        "leverage": "no_change_from_this_module",
        "concentration": "no_change_from_this_module",
        "liquidity": "no_change_from_this_module",
        "downside": "no_change_from_this_module",
        "notes": ["This module may only issue research advisories. It cannot place, cancel, or resize orders."],
    }
    if official_state == "fragility_building" or observational_label == "suspected_deleveraging":
        advisory["leverage"] = "research_downgrade_consideration"
        advisory["concentration"] = "research_downgrade_consideration"
        advisory["downside"] = "elevated_vulnerability_consideration"
        advisory["notes"].append("Consider reviewing leverage and concentration; this is not an automatic change.")
    if official_state == "deleveraging" or observational_label == "suspected_deleveraging":
        advisory["downside"] = "elevated_deleveraging_consideration"
        advisory["liquidity"] = "research_downgrade_consideration"
        advisory["notes"].append("Do not treat the decline as a bottom from this state alone.")
    if official_state == "deleveraging_easing":
        advisory["notes"].append("Easing is not a buy or bottom signal.")
    if (features.get("books") or {}).get(BTC, {}).get("status") != "ok":
        advisory["liquidity"] = "unknown_visible_depth"
    elif official_state in {"deleveraging", "fragility_building"}:
        advisory["liquidity"] = "research_downgrade_consideration"

    return {
        "official_state": official_state,
        "observational_label": observational_label,
        "easing_is_not_a_bottom": True,
        "forced_liquidations_certified": False,
        "conditions": conditions,
        "joint_results": {
            "fragility_building": fragility,
            "deleveraging": official_delev,
            "deleveraging_easing": official_easing,
            "suspected_deleveraging": suspected_delev,
            "suspected_deleveraging_easing": suspected_easing,
        },
        "counterevidence": counter,
        "insufficiency": insufficiency,
        "research_advisory": advisory,
        "notes": notes,
    }


def extract_features(snapshot: dict[str, Any], *, cutoff: datetime, config: dict[str, Any],
                     replay: bool) -> dict[str, Any]:
    instruments = snapshot.get("instruments") or {}
    positioning = snapshot.get("positioning_feeds") or {}
    pos_inst = positioning.get("instruments") or {}
    macro_history = snapshot.get("macro_history") or snapshot.get("alfred_history")
    features: dict[str, Any] = {
        "returns": {},
        "price_holds": {},
        "realized_demand": {},
        "open_interest": {},
        "funding": {},
        "liquidations": {},
        "books": {},
        "gold_instrument_note": {
            "symbol": GOLD,
            "instrument_type": "usdt_linear_perpetual",
            "settlement": "USDT",
            "is_not": "XAUUSD",
            "note": "Do not reuse XAUUSD spot or CFD contract specs, tick values, or valuation for XAUUSDT.",
        },
    }
    lookback = int(config["thresholds"]["realized_demand_lookback_hours"])
    for symbol, inst in instruments.items():
        candles = inst.get("candles_closed_1h") or inst.get("all_closed_candles") or []
        features["returns"][symbol] = _close_return(candles, hours=24, cutoff=cutoff)
        features["price_holds"][symbol] = _new_low_held(candles, hours=lookback, cutoff=cutoff)
        features["realized_demand"][symbol] = _volume_demand(candles, hours=lookback, cutoff=cutoff)
        pos = pos_inst.get(symbol) or {}
        features["open_interest"][symbol] = _oi_feature(
            pos.get("open_interest_history"), inst.get("open_interest"), cutoff=cutoff, replay=replay)
        features["funding"][symbol] = _funding_feature(
            pos.get("current_funding") or inst.get("funding"), pos.get("funding_history"),
            cutoff=cutoff, config=config, replay=replay)
        features["liquidations"][symbol] = _liquidation_feature(
            pos.get("liquidations") or inst.get("liquidations"), cutoff=cutoff, config=config, replay=replay)
        features["books"][symbol] = ({"status": "unknown", "momentary_only": True, "not_realized_demand": True,
                                      "reason": "historical_book_availability_not_verified", "symbol": symbol}
                                     if replay else _book_feature(snapshot.get("execution_books") or snapshot.get("orderbook_context"), symbol))
    features["real_yield"] = real_yield_change(macro_history, cutoff=cutoff, replay=replay)
    features["broad_dollar"] = broad_dollar_change(macro_history, cutoff=cutoff, replay=replay)
    features["etf_flows"] = etf_flow_weakening(
        snapshot.get("etf_flow_context"), cutoff=cutoff,
        days=int(config["thresholds"]["etf_flow_weakening_days"]), replay=replay)
    gold_inst = instruments.get(GOLD) or {}
    features["gold_real_rate_sensitivity"] = gold_real_rate_sensitivity(
        gold_inst.get("candles_closed_1h") or gold_inst.get("all_closed_candles") or gold_inst.get("candles_closed_1d") or [],
        macro_history, cutoff=cutoff,
        window=int(config["thresholds"]["gold_sensitivity_window"]),
        min_pairs=int(config["thresholds"]["gold_sensitivity_min_pairs"]),
        replay=replay,
    )
    features["polymarket"] = _polymarket_feature(
        snapshot.get("prediction_market_context"), cutoff=cutoff, config=config, replay=replay)
    return features


def assess_liquidity_positioning(snapshot: dict[str, Any], *,
                                 cutoff: datetime | str | None = None,
                                 prior_state: dict[str, Any] | None = None,
                                 config: dict[str, Any] | None = None,
                                 mode: str = "current_snapshot") -> dict[str, Any]:
    """Build a structured, non-trading liquidity and positioning assessment."""
    if mode not in {"current_snapshot", "historical_replay"}:
        raise ValueError("mode must be current_snapshot or historical_replay")
    cfg = config or load_config()
    if cutoff is None:
        cutoff = snapshot.get("cutoff_at_utc") or snapshot.get("retrieved_at_utc")
    cutoff_dt = _aware(cutoff)
    replay = mode == "historical_replay"
    features = extract_features(snapshot, cutoff=cutoff_dt, config=cfg, replay=replay)
    prior = _prior_active(prior_state, cutoff=cutoff_dt,
                          max_age_hours=float(cfg["thresholds"]["prior_deleveraging_max_age_hours"]))
    classified = classify_conditions(features, cfg, prior)
    source_quality = {
        "venue_coverage": "single_venue_bitget_usdt_linear",
        "marketwide_coverage": False,
        "coin_margin_mixed": False,
        "dollar_series": "DTWEXBGS",
        "dollar_series_is_not": "ICE_DXY",
        "gold_instrument": GOLD,
        "gold_instrument_is_not": "XAUUSD",
        "etf_dashes_are_not_zero": True,
        "funding_not_annualized": True,
        "liquidations_docs_window": "about_last_3_days_may_be_delayed",
        "forced_liquidations_certified": False,
        "macro_intraday_release_verified": False,
        "historical_replay_supported": False,
        "revision_vintage": {
            "alfred_vintage_date": ((snapshot.get("macro_history") or snapshot.get("alfred_history") or {})
                                    .get("vintage_date") or (snapshot.get("macro_context") or {}).get("vintage_date")),
            "note": "ALFRED uses a conservative date vintage; exact published_at is unknown.",
        },
        "timestamp_policy": cfg.get("timestamp_policy"),
    }
    return {
        "kind": "liquidity_positioning_research_assessment_v1",
        "version": cfg["version"],
        "config_sha256": cfg.get("config_sha256"),
        "calibrated": False,
        "historical_states_validated": False,
        "not_a_trading_signal": True,
        "does_not_change_frozen_policy": True,
        "order_permissions": False,
        "mode": mode,
        "cutoff_at_utc": _iso(cutoff_dt),
        "retrieved_at_utc": snapshot.get("retrieved_at_utc"),
        "official_state": classified["official_state"],
        "observational_label": classified["observational_label"],
        "easing_is_not_a_bottom": True,
        "forced_liquidations_certified": False,
        "conditions": classified["conditions"],
        "joint_results": classified["joint_results"],
        "counterevidence": classified["counterevidence"],
        "insufficiency": classified["insufficiency"],
        "features": features,
        "prior_state": prior,
        "research_advisory": classified["research_advisory"],
        "source_quality": source_quality,
        "notes": classified["notes"],
        "preregistered_validation": cfg.get("preregistered_validation"),
        "limitations": [
            "Thresholds are frozen uncalibrated heuristics, not validated warning levels.",
            "Bitget USDT-M coverage is not market-wide and is not mixed with coin-margined contracts.",
            "XAUUSDT is not XAUUSD.",
            "DTWEXBGS is not ICE DXY.",
            "If published_at or available_at is unknown, it is left unknown; retrieved_at is never substituted.",
            "Historical replay drops records without a known available_at at or before the cutoff.",
            "Official public liquidations cover about three days and may be delayed.",
            "This assessment does not change mechanical policy, risk gates, orders, or stops.",
        ],
    }


def build_liquidity_context(snapshot: dict[str, Any], *, prior_state: dict[str, Any] | None = None,
                            positioning_feeds: dict[str, Any] | None = None,
                            macro_history: dict[str, Any] | None = None,
                            execution_books: dict[str, Any] | None = None,
                            mode: str = "current_snapshot",
                            now: datetime | None = None) -> dict[str, Any]:
    packed = dict(snapshot)
    if positioning_feeds is not None:
        packed["positioning_feeds"] = positioning_feeds
    if macro_history is not None:
        packed["macro_history"] = macro_history
    if execution_books is not None:
        packed["execution_books"] = execution_books
    cutoff = packed.get("cutoff_at_utc") or packed.get("retrieved_at_utc") or (_iso(now) if now else None)
    if now is not None and mode == "current_snapshot":
        cutoff = _iso(max(_aware(cutoff), _aware(now)))
    assessment = assess_liquidity_positioning(packed, cutoff=cutoff, prior_state=prior_state, mode=mode)
    assessment["retrieved_at_utc"] = packed.get("retrieved_at_utc") or assessment["cutoff_at_utc"]
    if now is not None and mode == "current_snapshot":
        assessment["inputs_collected_at_utc"] = _iso(_aware(now))
    return assessment


def fetch_and_assess(*, timeout: float = 15.0, prior_state: dict[str, Any] | None = None,
                     snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    """Optional live current-snapshot assessment. Honest unknown is acceptable."""
    from piia.intelligence.macro_data import fetch_alfred_history
    from piia.intelligence.public_data import fetch_public_snapshot
    from piia.intelligence.public_positioning import fetch_positioning_feeds
    from piia.intelligence.public_orderbook import fetch_execution_books
    from piia.intelligence.etf_flows import fetch_us_spot_btc_etf_flows
    from piia.intelligence.polymarket import fetch_prediction_markets

    live = snapshot or fetch_public_snapshot()
    try:
        live["macro_history"] = fetch_alfred_history(["DFII10", "DTWEXBGS"], timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - live smoke must stay honest
        live["macro_history"] = {"kind": "alfred_conservative_date_vintage_history", "errors": {"fetch": type(exc).__name__}}
    if live.get("etf_flow_context") is None:
        try:
            live["etf_flow_context"] = fetch_us_spot_btc_etf_flows(timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            live["etf_flow_context"] = {"errors": {"fetch": type(exc).__name__}, "complete_daily_flows": []}
    if live.get("prediction_market_context") is None:
        try:
            live["prediction_market_context"] = fetch_prediction_markets()
        except Exception as exc:  # noqa: BLE001
            live["prediction_market_context"] = {"errors": {"fetch": type(exc).__name__}}
    try:
        feeds = fetch_positioning_feeds(list((live.get("instruments") or {"BTCUSDT": None}).keys()), timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        feeds = {"errors": {"fetch": type(exc).__name__}, "instruments": {}}
    try:
        books = fetch_execution_books(timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        books = {"ok": False, "errors": {"fetch": type(exc).__name__}, "instruments": {}}
    return build_liquidity_context(live, prior_state=prior_state, positioning_feeds=feeds,
                                   execution_books=books, mode="current_snapshot", now=datetime.now(timezone.utc))
