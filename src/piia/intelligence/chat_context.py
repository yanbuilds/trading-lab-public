"""Small, labeled projections of captured public research contexts for chat."""
from __future__ import annotations
import json
from typing import Any


def _pick(data, keys):
    return {k: data[k] for k in keys if k in data}


def compact_instrument(*, ticker, funding, open_interest, indicators, daily, market_context, candles):
    """Retain dated indicator values and focus candles without duplicating long notes."""
    ind = _pick(indicators or {}, ("ok", "reason", "last_closed_bar_close_at_utc", "last_close", "ema20", "ema50",
                                   "atr14", "atr14_pct_of_close", "rsi14_wilder", "return_24h_pct",
                                   "range_24h_high", "range_24h_low", "ema_cross_on_last_closed_bar"))
    four = (indicators or {}).get("four_hour_trend")
    if isinstance(four, dict):
        ind["four_hour_trend"] = _pick(four, ("ok", "last_closed_at_utc", "ema20", "ema50", "alignment"))
    return {"ticker": ticker, "funding": funding, "open_interest": open_interest, "indicators": ind,
            "daily_context": _pick(daily or {}, ("ok", "reason", "last_closed_at_utc", "last_close", "ema20", "ema50",
                                                "sma200", "close_vs_sma200", "alignment", "last_20_closed_daily_high", "last_20_closed_daily_low")),
            "market_context": _pick(market_context or {}, ("kind", "ema_relationships", "timeframe_alignment", "as_of_utc")),
            "last_candles": candles, "note": "Closed-bar indicators are historical context, not a validated signal."}


def compact_contexts(request: dict[str, Any]) -> list[dict[str, str]]:
    """Project existing snapshots; never fetch or fabricate a missing observation."""
    result = []
    def add(name, body):
        result.append({"id": name, "raw": json.dumps(body, ensure_ascii=False, default=str)})

    for key, label, fields in (
        ("etf_flow_context", "BTC_ETF_FLOWS", ("kind", "source_url", "attribution", "retrieved_at_utc", "completed_at_utc", "asof_complete_flow_date", "latest_complete_row", "latest_row_is_partial", "latest_complete_age_calendar_days", "latest_exchange_session_verified", "session_check", "supplemental_partial", "future_or_unclosed_session", "session_close_note", "stale", "errors", "primary_source_failure", "note")),
        ("gold_positioning_context", "GOLD_CFTC", ("kind", "source_url", "retrieved_at_utc", "asof_date", "previous_asof_date", "commitments_contracts", "open_interest_contracts", "stale", "errors", "coverage_note")),
        ("fear_greed_context", "BTC_FEAR_GREED", ("kind", "source_url", "attribution", "retrieved_at_utc", "observations", "stale", "error", "note")),
    ):
        context = request.get(key)
        if isinstance(context, dict):
            add(label, _pick(context, fields))

    onchain = request.get("onchain_context")
    if isinstance(onchain, dict):
        metrics = {}
        for name, item in (onchain.get("metrics") or {}).items():
            if isinstance(item, dict):
                metrics[name] = _pick(item, ("observation_at_utc", "source_url", "value", "height", "age_seconds",
                                            "count", "vsize", "fastestFee", "current_hashrate_exahash_per_second_derived",
                                            "difficulty_change_percent", "progress_percent"))
        add("BTC_ONCHAIN", {**_pick(onchain, ("kind", "retrieved_at_utc", "source_docs", "stale", "complete", "errors", "coverage_note")),
                            "metrics": metrics})

    calendar = request.get("event_calendar_context")
    if isinstance(calendar, dict):
        sources = {}
        for name, item in (calendar.get("sources") or {}).items():
            if not isinstance(item, dict):
                continue
            events = item.get("window_events") or item.get("upcoming_events") or []
            sources[name] = {**_pick(item, ("ok", "source_url", "note")),
                             "events_sample": [_pick(e, ("title", "scheduled_at_utc", "scheduled_at_iso", "source_url"))
                                               for e in events[:3] if isinstance(e, dict)],
                             "events_omitted": max(0, len(events) - 3)}
        add("OFFICIAL_EVENT_CALENDAR", {**_pick(calendar, ("kind", "retrieved_at_utc", "errors", "coverage_note")),
                                        "sources": sources, "note": "Scheduled times, not confirmed releases; sample only."})

    market = request.get("prediction_market_context")
    if isinstance(market, dict):
        events = []
        for event in (market.get("events") or [])[:6]:
            if not isinstance(event, dict):
                continue
            markets = []
            for m in (event.get("markets") or [])[:1]:
                if not isinstance(m, dict):
                    continue
                outcomes = []
                for o in (m.get("outcomes") or [])[:2]:
                    if not isinstance(o, dict):
                        continue
                    book = o.get("book") or {}
                    outcomes.append({**_pick(o, ("name", "platform_quoted_price")),
                                     "book": _pick(book, ("asof_utc", "retrieved_at_utc", "book_timestamp_status", "book_age_seconds",
                                                          "token_id_matches", "usable_for_implied_probability", "best_bid", "best_ask", "illiquid", "outlier_or_crossed"))})
                markets.append({**_pick(m, ("question", "group_item_title", "deadline_utc", "source_url")), "outcomes": outcomes})
            events.append({**_pick(event, ("title", "question", "category", "url", "deadline_utc", "asof_utc", "coverage_note")),
                           "markets_sample": markets, "markets_omitted_by_chat": max(0, len(event.get("markets") or []) - 1)})
        add("POLYMARKET_SAMPLE", {**_pick(market, ("kind", "collection_started_at_utc", "retrieved_at_utc", "asof_utc", "errors", "coverage_note")),
                                  "events_sample": events, "events_omitted_by_chat": max(0, len(market.get("events") or []) - 6),
                                  "note": "Sampled quoted prices, not facts, calibrated probabilities or direct trading signals. Full resolution rules are omitted; consult the event link."})
    return result


def compact_macro(context: dict[str, Any]) -> dict[str, str]:
    series = {}
    for name, item in (context.get("series") or {}).items():
        if isinstance(item, dict):
            series[name] = _pick(item, ("series_id", "label", "unit", "value", "observation_date", "vintage_date",
                                        "stale", "source_page", "series_note"))
    body = {**_pick(context, ("kind", "retrieved_at_utc", "vintage_date", "intraday_release_verified", "errors", "error_details", "coverage_note")),
            "series": series}
    return {"id": "ALFRED_MACRO", "raw": json.dumps(body, ensure_ascii=False, default=str)}


def compact_forecast(request: dict[str, Any], state: dict[str, Any], forecast: dict[str, Any],
                     archived_event: dict[str, Any] | None) -> dict[str, str] | None:
    """Only expose forecasts bound to this newly captured research run."""
    from piia.intelligence.evidence_gates import _valid_forecast
    receipt = state.get("exploratory_evidence_archive") or {}
    if (not archived_event or archived_event.get("request") != request or archived_event.get("forecast") != forecast
            or archived_event.get("created_at_utc") != receipt.get("recorded_at_utc")
            or state.get("request_hash") != request.get("request_hash") or not receipt.get("recorded_at_utc")
            or state.get("forecast_weights_ran") is not True or forecast.get("weights_ran") is not True):
        return None
    instruments = {}
    for symbol, item in (forecast.get("forecasts") or {}).items():
        if not _valid_forecast(item, (request.get("instruments") or {}).get(symbol)):
            continue
        instruments[symbol] = {"last_close": item["last_close"], "last_close_at_utc": item["last_close_at_utc"],
                               "horizon_hours": item["prediction_length"],
                               "end_q10": item["quantile_10"][-1], "end_q50": item["quantile_50"][-1],
                               "end_q90": item["quantile_90"][-1]}
    if not instruments:
        return None
    body = {"model_id": forecast.get("model_id"), "revision": forecast.get("revision"),
            "available_no_earlier_than_utc": receipt["recorded_at_utc"], "instruments": instruments,
            "training_leakage": forecast.get("training_leakage", "unknown"), "calibration_verified": False,
            "note": "Research forecast only; model quantiles are not calibrated probabilities or trade signals."}
    return {"id": "CHRONOS_RESEARCH_FORECAST", "raw": json.dumps(body, ensure_ascii=False)}
