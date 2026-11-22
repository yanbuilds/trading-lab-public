"""Allowlisted public Bitget positioning feeds for research only.

Covers USDT-M linear perpetuals on Bitget. These snapshots are not market-wide,
not coin-margined, and not a historical replay archive. No API keys and no orders.

Official docs (as of 2026-11-17):
- Open interest history: https://www.bitget.com/docs/catalog/market/derivatives
  GET /api/v3/market/open-interest-history
  openInterest is base currency; openInterestValue = openInterest * markPrice.
- Funding history: GET /api/v3/market/history-fund-rate (about last 90 days).
  fundingRate is the settlement-interval rate, not an annualized yield.
- Current funding: GET /api/v3/market/current-fund-rate includes fundingRateInterval hours.
- Liquidations: GET /api/v3/market/liquidations (about last 3 days, may be delayed).
  REST docs say amount is in base currency. The public WebSocket docs describe
  amount as quote coin; this module follows the REST page for this endpoint and
  flags the documentation conflict rather than guessing.

Publication timestamps are usually absent from these payloads. published_at and
available_at stay unknown unless a documented bound exists. retrieved_at is never
copied into those fields.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
import math

import httpx

ALLOWED_HOST = "api.bitget.com"
DOCS = "https://www.bitget.com/docs/catalog/market/derivatives"
CATEGORY = "USDT-FUTURES"
OI_HISTORY = "https://api.bitget.com/api/v3/market/open-interest-history?symbol={symbol}&period=1h"
CURRENT_FUNDING = "https://api.bitget.com/api/v3/market/current-fund-rate?category=USDT-FUTURES&symbol={symbol}"
FUNDING_HISTORY = "https://api.bitget.com/api/v3/market/history-fund-rate?category=USDT-FUTURES&symbol={symbol}&limit={limit}"
LIQUIDATIONS = "https://api.bitget.com/api/v3/market/liquidations?category=USDT-FUTURES&symbol={symbol}&limit=100"
MAX_LIQUIDATION_PAGES = 8
LIQUIDATION_WINDOW = timedelta(days=3)


def _aware(now: datetime) -> datetime:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("timezone-aware time required")
    return now.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _allowlisted(url: str) -> None:
    if ALLOWED_HOST not in url or not url.startswith("https://api.bitget.com/"):
        raise ValueError("refusing non-allowlisted host")


def _get_json(url: str, timeout: float) -> dict:
    _allowlisted(url)
    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        response = client.get(url)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("invalid provider JSON")
    return payload


def _finite(value: Any, *, allow_zero: bool = True, allow_negative: bool = False) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError("invalid numeric field")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite numeric field")
    if number < 0 and not allow_negative:
        raise ValueError("numeric field out of range")
    if number == 0 and not allow_zero:
        raise ValueError("numeric field out of range")
    return number


def _ms(value: Any) -> datetime:
    if isinstance(value, bool) or value is None or not str(value).isdigit():
        raise ValueError("invalid timestamp")
    stamp = datetime.fromtimestamp(int(value) / 1000, timezone.utc)
    return stamp


def empty_timestamps(*, observed: datetime | None, retrieved: datetime) -> dict[str, Any]:
    return {
        "observed_at_utc": None if observed is None else _iso(observed),
        "published_at_utc": None,
        "published_at_known": False,
        "available_at_utc": None,
        "available_at_known": False,
        "available_at_exact": False,
        "retrieved_at_utc": _iso(retrieved),
        "timestamp_note": (
            "published_at and available_at are unknown; exchange payloads usually omit "
            "publication time, and retrieved_at is not a substitute."
        ),
    }


def parse_open_interest_history(payload: Any, *, symbol: str, now: datetime, source_url: str) -> dict[str, Any]:
    now = _aware(now)
    if not isinstance(payload, dict) or payload.get("code") != "00000":
        raise ValueError("invalid open interest response")
    data = payload.get("data")
    rows = data.get("list") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("missing open interest history")
    seen: set[datetime] = set()
    observations: list[dict[str, Any]] = []
    future = 0
    for row in rows:
        if not isinstance(row, dict) or row.get("symbol") != symbol:
            raise ValueError("wrong open interest symbol")
        stamp = _ms(row.get("timestamp"))
        if stamp.minute or stamp.second or stamp.microsecond:
            raise ValueError("open interest period is not hour-aligned")
        if stamp in seen:
            raise ValueError("duplicate open interest timestamp")
        seen.add(stamp)
        if stamp > now:
            future += 1
            continue
        base = _finite(row.get("openInterest"))
        value = None
        raw_value = row.get("openInterestValue")
        if raw_value not in (None, ""):
            value = _finite(raw_value)
        item = {
            "period_end_utc": _iso(stamp),
            "open_interest_base_units": base,
            "open_interest_value_quote": value,
            "value_unit": "quote_notional_openInterest_times_mark" if value is not None else None,
            "unit_note": "openInterest is base currency; value is price-dependent and is not used as a crowding substitute",
            **empty_timestamps(observed=stamp, retrieved=now),
        }
        observations.append(item)
    observations.sort(key=lambda item: item["period_end_utc"])
    if not observations or now - datetime.fromisoformat(observations[-1]["period_end_utc"]) > timedelta(hours=2):
        raise ValueError("open interest is missing or stale")
    tail = observations[-25:]
    continuous = len(tail) == 25 and all(
        datetime.fromisoformat(b["period_end_utc"]) - datetime.fromisoformat(a["period_end_utc"]) == timedelta(hours=1)
        for a, b in zip(tail, tail[1:])
    )
    latest = observations[-1]
    baseline = tail[0] if continuous else None
    base_change = None
    value_change = None
    if baseline is not None and baseline["open_interest_base_units"] > 0:
        base_change = (latest["open_interest_base_units"] / baseline["open_interest_base_units"] - 1) * 100
    if (baseline is not None and latest["open_interest_value_quote"] is not None
            and baseline["open_interest_value_quote"] not in (None, 0)):
        value_change = (latest["open_interest_value_quote"] / baseline["open_interest_value_quote"] - 1) * 100
    price_contaminated = (
        base_change is not None and value_change is not None and math.copysign(1, base_change or 0) != math.copysign(1, value_change or 0)
    )
    return {
        "symbol": symbol,
        "venue": "bitget",
        "category": CATEGORY,
        "instrument_type": "usdt_linear_perpetual",
        "coverage": "single_venue_bitget_usdt_linear",
        "marketwide_coverage": False,
        "period": "1h",
        "observations": observations,
        "latest_period_end_utc": latest["period_end_utc"],
        "latest_open_interest_base_units": latest["open_interest_base_units"],
        "latest_open_interest_value_quote": latest["open_interest_value_quote"],
        "change_24h_base_pct": base_change if continuous else None,
        "change_24h_value_pct": value_change if continuous else None,
        "change_24h_base_units": (
            latest["open_interest_base_units"] - baseline["open_interest_base_units"] if baseline is not None else None
        ),
        "history_complete_24h": continuous,
        "price_contaminated_value_change": bool(price_contaminated),
        "future_rows_excluded": future,
        "source_url": source_url,
        "docs": DOCS,
        "retrieved_at_utc": _iso(now),
        "historical_replay_supported": False,
    }


def parse_current_funding(payload: Any, *, symbol: str, now: datetime, source_url: str) -> dict[str, Any]:
    now = _aware(now)
    if not isinstance(payload, dict) or payload.get("code") != "00000":
        raise ValueError("invalid current funding response")
    data = payload.get("data")
    row = data[0] if isinstance(data, list) and data else data
    if not isinstance(row, dict) or row.get("symbol") not in (None, symbol):
        raise ValueError("wrong or missing funding symbol")
    rate = _finite(row.get("fundingRate"), allow_zero=True, allow_negative=True)
    interval = row.get("fundingRateInterval")
    interval_hours = None
    if interval not in (None, ""):
        interval_hours = int(float(interval))
        if interval_hours < 1:
            raise ValueError("invalid funding interval")
    next_update = None
    if row.get("nextUpdate") not in (None, ""):
        next_update = _iso(_ms(row.get("nextUpdate")))
    stamps = empty_timestamps(observed=None, retrieved=now)
    stamps["timestamp_note"] = (
        "Current funding is a provider snapshot at retrieval time. Settlement time, "
        "published_at, and available_at are not verified from this payload. The rate "
        "is per funding interval, not an annualized yield."
    )
    return {
        "symbol": symbol,
        "venue": "bitget",
        "category": CATEGORY,
        "instrument_type": "usdt_linear_perpetual",
        "funding_rate_per_interval": rate,
        "funding_rate_unit": "decimal_per_settlement_interval",
        "not_annualized": True,
        "funding_interval_hours": interval_hours,
        "next_update_utc": next_update,
        "last_funding_rate_per_interval": (
            _finite(row["lastFundingRate"], allow_zero=True, allow_negative=True) if row.get("lastFundingRate") not in (None, "") else None
        ),
        "source_url": source_url,
        "docs": DOCS,
        "marketwide_coverage": False,
        **stamps,
    }


def parse_funding_history(payload: Any, *, symbol: str, now: datetime, source_url: str) -> dict[str, Any]:
    now = _aware(now)
    if not isinstance(payload, dict) or payload.get("code") != "00000":
        raise ValueError("invalid funding history response")
    data = payload.get("data")
    rows = data.get("resultList") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise ValueError("missing funding history")
    seen: set[datetime] = set()
    observations: list[dict[str, Any]] = []
    future = 0
    for row in rows:
        if not isinstance(row, dict) or row.get("symbol") not in (None, symbol):
            raise ValueError("wrong funding history symbol")
        stamp = _ms(row.get("fundingRateTimestamp"))
        if stamp in seen:
            raise ValueError("duplicate funding timestamp")
        seen.add(stamp)
        if stamp > now:
            future += 1
            continue
        item = {
            "settlement_at_utc": _iso(stamp),
            "funding_rate_per_interval": _finite(row.get("fundingRate"), allow_zero=True, allow_negative=True),
            "funding_rate_unit": "decimal_per_settlement_interval",
            "not_annualized": True,
            **empty_timestamps(observed=stamp, retrieved=now),
        }
        observations.append(item)
    observations.sort(key=lambda item: item["settlement_at_utc"])
    interval_hours = None
    if len(observations) >= 2:
        delta = datetime.fromisoformat(observations[-1]["settlement_at_utc"]) - datetime.fromisoformat(observations[-2]["settlement_at_utc"])
        hours = delta.total_seconds() / 3600
        if hours in {1, 2, 4, 8}:
            interval_hours = int(hours)
    return {
        "symbol": symbol,
        "venue": "bitget",
        "category": CATEGORY,
        "instrument_type": "usdt_linear_perpetual",
        "coverage": "single_venue_bitget_usdt_linear",
        "marketwide_coverage": False,
        "docs_window": "about_last_90_days",
        "observations": observations,
        "inferred_interval_hours": interval_hours,
        "interval_inference_note": "Cadence is inferred from adjacent settlements only when the gap is a documented interval; not an official schedule archive.",
        "future_rows_excluded": future,
        "source_url": source_url,
        "docs": DOCS,
        "retrieved_at_utc": _iso(now),
        "historical_replay_supported": False,
        "not_annualized": True,
    }


def parse_liquidations(pages: list[dict[str, Any]], *, symbol: str, now: datetime, source_urls: list[str]) -> dict[str, Any]:
    """Parse official public liquidation pages. Window is about three days and may be delayed."""
    now = _aware(now)
    events: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    future = 0
    for payload in pages:
        if not isinstance(payload, dict) or payload.get("code") != "00000":
            raise ValueError("invalid liquidations response")
        data = payload.get("data")
        rows = data.get("list") if isinstance(data, dict) else payload.get("list")
        if rows is None:
            rows = []
        if not isinstance(rows, list):
            raise ValueError("invalid liquidations list")
        for row in rows:
            if not isinstance(row, dict) or row.get("symbol") != symbol:
                raise ValueError("wrong liquidation symbol")
            stamp = _ms(row.get("ts"))
            if stamp > now:
                future += 1
                continue
            amount = _finite(row.get("amount"), allow_zero=True)
            price = _finite(row.get("price"), allow_zero=False)
            side = row.get("side")
            if side not in {"buy", "sell"}:
                raise ValueError("invalid liquidation side")
            key = (row.get("symbol"), str(row.get("ts")), str(side), str(row.get("amount")))
            if key in seen:
                continue
            seen.add(key)
            item = {
                "side": side,
                "side_note": "REST: buy=long liquidation, sell=short liquidation",
                "price": price,
                "amount": amount,
                "amount_unit": "base_currency_per_official_rest_docs",
                "unit_documentation_conflict": (
                    "Official REST docs for /api/v3/market/liquidations say amount is base currency; "
                    "the public WebSocket liquidation channel describes amount as quote coin. "
                    "This parser does not convert units and will not certify forced liquidations "
                    "if the unit cannot be trusted."
                ),
                **empty_timestamps(observed=stamp, retrieved=now),
            }
            # Delay is officially acknowledged, so availability is unknown even though ts exists.
            item["timestamp_note"] = (
                "ts is the liquidation event time on the payload. Official docs say the series may be delayed, "
                "so published_at and available_at remain unknown. retrieved_at is not used as availability."
            )
            events.append(item)
    events.sort(key=lambda item: item["observed_at_utc"])
    cutoff_3d = now - LIQUIDATION_WINDOW
    in_window = [e for e in events if datetime.fromisoformat(e["observed_at_utc"]) >= cutoff_3d]
    latest_24h = [e for e in in_window if now - datetime.fromisoformat(e["observed_at_utc"]) <= timedelta(hours=24)]
    prev_24h = [e for e in in_window if timedelta(hours=24) < now - datetime.fromisoformat(e["observed_at_utc"]) <= timedelta(hours=48)]

    def _sum(rows: list[dict[str, Any]]) -> float:
        return float(sum(item["amount"] for item in rows))

    return {
        "symbol": symbol,
        "venue": "bitget",
        "category": CATEGORY,
        "instrument_type": "usdt_linear_perpetual",
        "coverage": "single_venue_bitget_usdt_linear",
        "marketwide_coverage": False,
        "docs_window": "about_last_3_days_may_be_delayed",
        "historical_replay_supported": False,
        "forced_liquidations_certified": False,
        "events": in_window,
        "event_count_window": len(in_window),
        "amount_sum_latest_24h_base": _sum(latest_24h) if latest_24h or prev_24h else None,
        "amount_sum_previous_24h_base": _sum(prev_24h) if latest_24h or prev_24h else None,
        "latest_24h_event_count": len(latest_24h),
        "previous_24h_event_count": len(prev_24h),
        "future_rows_excluded": future,
        "source_urls": source_urls,
        "docs": DOCS,
        "retrieved_at_utc": _iso(now),
        "limitations": [
            "Official public history is about three days only.",
            "Data may be delayed; available_at is unknown.",
            "Single Bitget USDT-M venue; not market-wide.",
            "REST vs WebSocket amount-unit documentation conflict is unresolved.",
            "Cannot certify that an open-interest drop was forced liquidation.",
        ],
    }


def fetch_positioning_feeds(symbols: list[str], *, timeout: float = 12.0, now: datetime | None = None) -> dict[str, Any]:
    """Live public Bitget positioning snapshot. Missing parts stay missing."""
    now = _aware(now or datetime.now(timezone.utc))
    out: dict[str, Any] = {
        "kind": "bitget_public_positioning_feeds",
        "retrieved_at_utc": _iso(now),
        "venue": "bitget",
        "category": CATEGORY,
        "marketwide_coverage": False,
        "historical_replay_supported": False,
        "docs": DOCS,
        "instruments": {},
        "errors": {},
    }
    for symbol in symbols:
        inst: dict[str, Any] = {"symbol": symbol, "venue": "bitget", "category": CATEGORY}
        try:
            url = OI_HISTORY.format(symbol=symbol)
            inst["open_interest_history"] = parse_open_interest_history(_get_json(url, timeout), symbol=symbol, now=now, source_url=url)
        except (httpx.HTTPError, ValueError, TypeError, KeyError, OSError) as exc:
            inst["open_interest_history"] = None
            out["errors"].setdefault(symbol, {})["open_interest_history"] = type(exc).__name__
        try:
            url = CURRENT_FUNDING.format(symbol=symbol)
            inst["current_funding"] = parse_current_funding(_get_json(url, timeout), symbol=symbol, now=now, source_url=url)
        except (httpx.HTTPError, ValueError, TypeError, KeyError, OSError) as exc:
            inst["current_funding"] = None
            out["errors"].setdefault(symbol, {})["current_funding"] = type(exc).__name__
        try:
            url = FUNDING_HISTORY.format(symbol=symbol, limit=100)
            inst["funding_history"] = parse_funding_history(_get_json(url, timeout), symbol=symbol, now=now, source_url=url)
        except (httpx.HTTPError, ValueError, TypeError, KeyError, OSError) as exc:
            inst["funding_history"] = None
            out["errors"].setdefault(symbol, {})["funding_history"] = type(exc).__name__
        try:
            pages = []
            urls = []
            cursor = None
            for _ in range(MAX_LIQUIDATION_PAGES):
                url = LIQUIDATIONS.format(symbol=symbol)
                if cursor:
                    url = url + f"&cursor={cursor}"
                payload = _get_json(url, timeout)
                pages.append(payload)
                urls.append(url)
                data = payload.get("data") if isinstance(payload, dict) else None
                nxt = data.get("cursor") if isinstance(data, dict) else None
                if not nxt or nxt == cursor:
                    break
                cursor = nxt
            inst["liquidations"] = parse_liquidations(pages, symbol=symbol, now=now, source_urls=urls)
        except (httpx.HTTPError, ValueError, TypeError, KeyError, OSError) as exc:
            inst["liquidations"] = None
            out["errors"].setdefault(symbol, {})["liquidations"] = type(exc).__name__
        out["instruments"][symbol] = inst
    out["retrieved_at_utc"] = _iso(now)
    return out
