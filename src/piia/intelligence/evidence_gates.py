"""Instrument-specific hard evidence requirements for an executable recommendation.

Bitcoin on-chain activity and US spot BTC ETF fund flows apply to BTC only.
They must not automatically fail gold or Nasdaq-100. Single-stock fundamentals
are not a universal hard gate for an index perpetual or gold. These gates do
not exist to force a trade: a dataset is complete only when its required set
is present, timely, and correctly unit-labeled.
"""
from __future__ import annotations

from typing import Any
import math

SYMBOLS = ("BTCUSDT", "XAUUSDT", "NDX100USDT")

# Hard requirements that can block a trading recommendation for that instrument.
HARD_REQUIREMENTS: dict[str, dict[str, str]] = {
    "BTCUSDT": {
        "onchain": (
            "Current Bitcoin network evidence from a documented public explorer, "
            "including chain tip, latest confirmed block time, mempool, recommended "
            "fees, difficulty adjustment, and hashrate/difficulty. One field is not enough."
        ),
        "btc_etf_flows": (
            "US spot Bitcoin ETF net fund flows with fund-level coverage and a complete "
            "reported day inside the freshness window. Flows are distinct from AUM and price."
        ),
    },
    "XAUUSDT": {},
    "NDX100USDT": {},
}

# Remaining research gaps that must stay labeled, but are not hard recommendation
# blocks until a real official/free source is wired.
RESEARCH_GAPS = {
    "onchain_exchange_flows_utxo_realized_cap": "BTC-only deeper on-chain analytics are still unavailable without a paid or unconnected provider.",
    "btc_etf_issuer_official_aum_and_sec_daily_filings": "Farside is a compiler of issuer disclosures, not a primary SEC daily filing or official issuer AUM feed.",
    "gold_etf_and_physical_demand_flows": "No official free gold ETF holdings or physical-demand feed was wired without signup or a key.",
    "nasdaq100_official_aggregate_valuation_and_breadth": "No official free Nasdaq-100 aggregate valuation or breadth feed was available without signup.",
    "event_calendar": "A contemporaneous official event calendar is not wired.",
    "event_calendar_non_us_earnings_and_intraday_release_confirmation": "Official U.S. Fed/BLS/BEA published calendars do not cover non-U.S. central banks, earnings, or verified exact release ticks.",
    "prediction_markets": "A contemporaneous official public prediction-market sample is not available.",
    "prediction_market_sample_is_not_exhaustive_coverage": "The Polymarket adapter only samples documented public-search hits for Fed/rate, inflation, recession, Bitcoin, gold, and Nasdaq-related events. Missing a topic in the sample is not proof the venue has no such market.",
    "prediction_market_odds_are_not_calibrated_probabilities": "Quoted prediction-market prices are not calibrated probabilities, not facts, and not a forced trade signal.",
}

NOT_APPLICABLE = {
    "BTCUSDT": ("equities_fundamentals",),
    "XAUUSDT": ("onchain", "btc_etf_flows", "equities_fundamentals"),
    "NDX100USDT": ("onchain", "btc_etf_flows", "equities_fundamentals"),
}


def apply_external_missing(
    missing: list[str],
    onchain: dict[str, Any] | None,
    etf: dict[str, Any] | None,
    calendar: dict[str, Any] | None = None,
    gold_positioning: dict[str, Any] | None = None,
    prediction_markets: dict[str, Any] | None = None,
) -> list[str]:
    """Update the packet missing list after current public research fetches."""
    out = list(missing)
    replacements: list[str] = []
    if onchain and onchain.get("usable_for_btc_research"):
        if "onchain" in out:
            out.remove("onchain")
        replacements.append("onchain_exchange_flows_utxo_realized_cap")
    if etf and etf.get("usable_for_btc_research"):
        if "btc_etf_flows" in out:
            out.remove("btc_etf_flows")
        replacements.append("btc_etf_issuer_official_aum_and_sec_daily_filings")
    if calendar and calendar.get("usable_for_research"):
        if "event_calendar" in out:
            out.remove("event_calendar")
        replacements.append("event_calendar_non_us_earnings_and_intraday_release_confirmation")
    # Gold COT is partial futures positioning and must not clear ETF/physical demand.
    if "equities_fundamentals" in out:
        out.remove("equities_fundamentals")
    # Standing research gaps that are not universal hard gates.
    replacements.extend([
        "nasdaq100_official_aggregate_valuation_and_breadth",
        "gold_etf_and_physical_demand_flows",
    ])
    if not (calendar and calendar.get("usable_for_research")):
        replacements.append("event_calendar")
    _ = gold_positioning  # present context does not complete physical/ETF demand
    if prediction_markets and prediction_markets.get("usable_for_research"):
        if "prediction_markets" in out:
            out.remove("prediction_markets")
        replacements.extend([
            "prediction_market_sample_is_not_exhaustive_coverage",
            "prediction_market_odds_are_not_calibrated_probabilities",
        ])
    else:
        replacements.append("prediction_markets")
    for item in replacements:
        if item not in out:
            out.append(item)
    return out


def _usable(payload: dict[str, Any] | None) -> bool:
    return bool(payload) and bool(payload.get("usable_for_btc_research"))


def _valid_forecast(item: Any, instrument: Any) -> bool:
    """A successful run for one symbol does not prove all outputs exist."""
    if not isinstance(item, dict) or item.get("ok") is not True or item.get("weights_ran") is not True:
        return False
    if not isinstance(instrument, dict) or item.get("timeframe") != "1H_closed":
        return False
    expected_at = instrument.get("last_closed_close_at_utc")
    expected_close = (instrument.get("indicators_prefix_only") or {}).get("last_close")
    if not expected_at or item.get("last_close_at_utc") != expected_at:
        return False
    actual_close = item.get("last_close")
    if (isinstance(expected_close, bool) or isinstance(actual_close, bool)
            or not isinstance(expected_close, (int, float)) or not isinstance(actual_close, (int, float))
            or not math.isfinite(expected_close) or not math.isfinite(actual_close)
            or expected_close <= 0 or not math.isclose(expected_close, actual_close, rel_tol=1e-9)):
        return False
    horizon = item.get("prediction_length")
    if type(horizon) is not int or horizon < 1 or horizon > 512:
        return False
    quantiles = [item.get(k) for k in ("quantile_10", "quantile_50", "quantile_90")]
    if any(not isinstance(q, list) or len(q) != horizon for q in quantiles):
        return False
    for values in zip(*quantiles):
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values):
            return False
        if not values[0] <= values[1] <= values[2]:
            return False
    return True


def recommendation_blockers(
    request: dict[str, Any],
    forecast: dict[str, Any],
    macro: dict[str, Any] | None,
    onchain: dict[str, Any] | None = None,
    etf: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return packet-level and instrument-specific recommendation blockers."""
    packet: list[str] = []
    if not request.get("snapshot_ok"):
        packet.append("snapshot_incomplete")
    if not forecast.get("weights_ran"):
        packet.append("neural_forecast_unavailable:" + str(forecast.get("reason") or forecast.get("blocker") or "not_run"))
    if not macro or not (macro.get("series") or {}):
        packet.append("macro_snapshot_unavailable")
    else:
        for required in ("DFII10", "DTWEXBGS", "RRPONTSYD", "WDTGAL"):
            item = (macro.get("series") or {}).get(required)
            if not item:
                packet.append(f"macro_series_missing:{required}")
            elif item.get("stale"):
                packet.append(f"macro_series_stale:{required}")
    packet = list(dict.fromkeys(packet))

    missing = [str(item) for item in (request.get("missing_datasets") or [])]
    instruments: dict[str, list[str]] = {symbol: [] for symbol in SYMBOLS}
    for symbol in (request.get("instruments") or {}):
        instruments.setdefault(symbol, [])

    onchain = onchain if onchain is not None else request.get("onchain_context")
    etf = etf if etf is not None else request.get("etf_flow_context")

    def _listed(name: str) -> bool:
        return any(item == name or item.startswith(name + ":") or item.startswith(name + "_") for item in missing)

    for symbol, required in HARD_REQUIREMENTS.items():
        reasons = instruments.setdefault(symbol, [])
        if forecast.get("weights_ran") and not _valid_forecast((forecast.get("forecasts") or {}).get(symbol),
                                                                (request.get("instruments") or {}).get(symbol)):
            reasons.append(f"neural_forecast_missing_or_invalid:{symbol}")
        if "onchain" in required and (not _usable(onchain) or _listed("onchain")):
            if not _usable(onchain):
                reasons.append("missing_critical_evidence:BTCUSDT:onchain")
        if "btc_etf_flows" in required and (not _usable(etf) or _listed("btc_etf_flows")):
            if not _usable(etf):
                reasons.append("missing_critical_evidence:BTCUSDT:btc_etf_flows")
        reasons[:] = list(dict.fromkeys(reasons))

    # Safety: never apply BTC-only prefixes to gold or the index.
    for symbol, banned in NOT_APPLICABLE.items():
        kept = []
        for reason in instruments.get(symbol, []):
            if any(tag in reason for tag in banned) and symbol != "BTCUSDT":
                continue
            if symbol != "BTCUSDT" and "BTCUSDT" in reason:
                continue
            kept.append(reason)
        instruments[symbol] = kept

    flat: list[str] = list(packet)
    for symbol, reasons in instruments.items():
        for reason in reasons:
            if reason not in flat:
                flat.append(reason)
    any_instrument_blocked = any(instruments.get(symbol) for symbol in instruments)
    return {
        "packet": packet,
        "instruments": instruments,
        "all": flat,
        "trading_recommendation_blocked": bool(packet) or any_instrument_blocked,
        "hard_requirements": HARD_REQUIREMENTS,
        "research_gaps": RESEARCH_GAPS,
        "not_applicable": {k: list(v) for k, v in NOT_APPLICABLE.items()},
    }
