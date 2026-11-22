"""Allowlisted public Bitget market reads. No API keys, no general-purpose fetcher."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import math
import httpx
from piia.intelligence.paper.specs import CONTRACTS, PROVENANCE

ALLOWED_HOST = "api.bitget.com"
TICKER = "https://api.bitget.com/api/v2/mix/market/ticker?productType=USDT-FUTURES&symbol={symbol}"


def _parse_ticker(row: dict, now: datetime, url: str) -> dict:
    """Reject invalid or stale ticker data rather than treating it as a live mark."""
    last, mark = float(row["lastPr"]), float(row["markPrice"])
    raw_index = row.get("indexPrice")
    index = float(raw_index) if raw_index not in (None, "") else None
    if not all(math.isfinite(x) and x > 0 for x in (last, mark)):
        raise ValueError("invalid ticker price")
    if index is not None and (not math.isfinite(index) or index <= 0):
        raise ValueError("invalid index price")
    if not row.get("ts"):
        raise ValueError("ticker timestamp missing")
    observed = datetime.fromtimestamp(int(row["ts"]) / 1000, timezone.utc)
    age = now - observed
    if not timedelta(seconds=-5) <= age <= timedelta(minutes=5):
        raise ValueError("ticker timestamp stale or in the future")
    return {"last": last, "mark": mark, "index": index,
            "funding_rate": row.get("fundingRate"), "observed_at_utc": observed.isoformat(),
            "provenance": "public_bitget_ticker", "source_url": url}


def _parse_current_funding(row: dict, url: str, *, retrieved_at: datetime | None = None, symbol: str | None = None) -> dict:
    if symbol is not None and row.get("symbol") not in (None, symbol):
        raise ValueError("wrong funding symbol")
    retrieved_at = retrieved_at or datetime.now(timezone.utc)
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise ValueError("funding retrieval time must have timezone")
    rate = row.get("fundingRate")
    if rate is None:
        rate = row.get("funding_rate")
    if isinstance(rate, bool) or rate is None or not math.isfinite(float(rate)):
        raise ValueError("current funding rate missing or invalid")
    return {"funding_rate": str(rate),
            "next_update": row.get("nextUpdate") or row.get("nextFundingTime"),
            "retrieved_at_utc": retrieved_at.astimezone(timezone.utc).isoformat(),
            "source_url": url, "provenance": "public_bitget_current_funding",
            "limitations": "Current provider snapshot at local retrieval time; not a verified settlement, historical funding record, or forecast of future funding."}


def current_funding_for_research(funding: dict | None, now: datetime) -> dict | None:
    """Use a recent, explicitly timed public snapshot; never invent a rate."""
    if not isinstance(funding, dict) or now.tzinfo is None or now.utcoffset() is None:
        return None
    try:
        observed = datetime.fromisoformat(funding["retrieved_at_utc"])
        rate = funding["funding_rate"]
        if (observed.tzinfo is None or observed.utcoffset() is None or isinstance(rate, bool)
                or not math.isfinite(float(rate)) or not timedelta(0) <= now - observed <= timedelta(minutes=5)
                or funding.get("provenance") != "public_bitget_current_funding"):
            return None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return funding


def fetch_public_marks(symbols: list[str] | None = None, timeout: float = 12.0) -> dict:
    symbols = symbols or list(CONTRACTS)
    out = {"ok": True, "source": TICKER, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "marks": {}, "errors": []}
    for sym in symbols:
        url = TICKER.format(symbol=sym)
        if "api.bitget.com" not in url:
            raise ValueError("refusing non-allowlisted host")
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.get(url)
            r.raise_for_status()
            data = r.json()
            row = (data.get("data") or [None])[0]
            if not row:
                out["errors"].append({"symbol": sym, "error": "empty"})
                out["ok"] = False
                continue
            out["marks"][sym] = _parse_ticker(row, datetime.now(timezone.utc), url)
        except Exception as exc:
            out["ok"] = False
            out["errors"].append({"symbol": sym, "error": type(exc).__name__})
    out["exact_liquidation_verified"] = False
    out["contracts_provenance"] = PROVENANCE
    return out


CANDLES = "https://api.bitget.com/api/v2/mix/market/candles?productType=USDT-FUTURES&symbol={symbol}&granularity=1H&limit={limit}"
DAILY_CANDLES = "https://api.bitget.com/api/v2/mix/market/candles?productType=USDT-FUTURES&symbol={symbol}&granularity=1Dutc&limit=250"
DAILY_HISTORY = "https://api.bitget.com/api/v2/mix/market/history-candles?productType=USDT-FUTURES&symbol={symbol}&granularity=1Dutc&limit=100&startTime={start}&endTime={end}"
FUNDING = "https://api.bitget.com/api/v2/mix/market/current-fund-rate?productType=USDT-FUTURES&symbol={symbol}"
OI_HISTORY = "https://api.bitget.com/api/v3/market/open-interest-history?symbol={symbol}&period=1h"


def _parse_open_interest_history(data: dict, symbol: str, now: datetime, url: str) -> dict:
    """Summarize completed hourly observations; never use as a backtest feed."""
    if now.tzinfo is None or data.get("code") != "00000" or not isinstance(data.get("data"), dict):
        raise ValueError("invalid open interest response")
    rows = data["data"].get("list")
    if not isinstance(rows, list) or not rows:
        raise ValueError("missing open interest history")
    seen, observations, future = set(), [], 0
    for row in rows:
        if not isinstance(row, dict) or row.get("symbol") != symbol:
            raise ValueError("wrong open interest symbol")
        raw_time, raw_value = row.get("timestamp"), row.get("openInterest")
        if isinstance(raw_time, bool) or not str(raw_time).isdigit() or isinstance(raw_value, bool) or raw_value is None:
            raise ValueError("invalid open interest observation")
        try:
            stamp = datetime.fromtimestamp(int(raw_time) / 1000, timezone.utc)
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError, OSError) as exc:
            raise ValueError("invalid open interest observation") from exc
        if not math.isfinite(value) or value < 0 or stamp.minute or stamp.second or stamp.microsecond:
            raise ValueError("invalid open interest observation")
        if stamp in seen:
            raise ValueError("duplicate open interest timestamp")
        seen.add(stamp)
        if stamp > now:
            future += 1
        else:
            observations.append((stamp, value))
    observations.sort()
    if not observations or now - observations[-1][0] > timedelta(hours=2):
        raise ValueError("open interest is missing or stale")
    latest_time, latest = observations[-1]
    tail = observations[-25:]
    continuous = len(tail) == 25 and all(b[0] - a[0] == timedelta(hours=1) for a, b in zip(tail, tail[1:]))
    baseline = tail[0][1] if continuous else None
    change = (latest / baseline - 1) * 100 if baseline is not None and baseline > 0 else None
    if change is not None and not math.isfinite(change):
        change = None
    return {"symbol": symbol, "total_open_interest_base_units": latest,
            "period_end_utc": latest_time.isoformat(), "retrieved_at_utc": now.isoformat(),
            "change_24h_pct": change, "change_24h_base_units": latest - baseline if baseline is not None else None,
            "change_baseline_utc": tail[0][0].isoformat() if baseline is not None else None,
            "history_complete_24h": continuous, "future_rows_excluded": future,
            "source_url": url, "unit_and_scope": "base currency; Bitget long plus short positions",
            "limitations": "current research context only; not directional or a validated historical replay feed"}


def _allowlisted(url: str) -> None:
    if ALLOWED_HOST not in url or not url.startswith("https://api.bitget.com/"):
        raise ValueError("refusing non-allowlisted host")


def _get_json(url: str, timeout: float) -> dict:
    _allowlisted(url)
    with httpx.Client(timeout=timeout) as client:
        r = client.get(url)
    r.raise_for_status()
    return r.json()


def _closed_candles(rows: list, now: datetime, duration: timedelta = timedelta(hours=1)) -> list[dict]:
    """Keep fully closed bars only. Bitget v2 rows are [ts, o, h, l, c, vol, quoteVol]."""
    out = []
    for row in rows or []:
        ts = int(row[0])
        open_at = datetime.fromtimestamp(ts / 1000, timezone.utc)
        close_at = open_at + duration
        if close_at > now:
            continue
        out.append({
            "open_at_utc": open_at.isoformat(),
            "close_at_utc": close_at.isoformat(),
            "open": float(row[1]), "high": float(row[2]), "low": float(row[3]),
            "close": float(row[4]), "volume": float(row[5]) if len(row) > 5 else None,
        })
    out.sort(key=lambda x: x["open_at_utc"])
    return out


def candle_quality(candles: list[dict], now: datetime, duration: timedelta = timedelta(hours=1)) -> list[str]:
    """Check a closed-bar snapshot without filling or hiding gaps."""
    if len(candles) < 50:
        return ["insufficient_closed_candles"]
    errors = []
    stamps = [datetime.fromisoformat(c["open_at_utc"]) for c in candles]
    closes = [datetime.fromisoformat(c["close_at_utc"]) for c in candles]
    if now.tzinfo is None or any(t.tzinfo is None or t.utcoffset() is None for t in stamps + closes):
        return ["naive_candle_timestamp"]
    utc_stamps = [t.astimezone(timezone.utc) for t in stamps]
    if duration == timedelta(hours=1) and any(t.minute or t.second or t.microsecond for t in utc_stamps):
        errors.append("hourly_not_utc_aligned")
    if duration == timedelta(days=1) and any(t.hour or t.minute or t.second or t.microsecond for t in utc_stamps):
        errors.append("daily_not_utc_aligned")
    if any(b - a != duration for a, b in zip(stamps, stamps[1:])):
        errors.append("duplicate_or_missing_candles")
    if any(close != t + duration for close, t in zip(closes, stamps)):
        errors.append("invalid_candle_duration")
    age = now - closes[-1]
    if not timedelta(0) <= age < duration:
        errors.append("last_closed_candle_missing_or_stale")
    for c in candles:
        o, h, l, close = (c[k] for k in ("open", "high", "low", "close"))
        if (not all(math.isfinite(v) and v > 0 for v in (o, h, l, close))
                or h < max(o, l, close) or l > min(o, h, close)):
            errors.append("invalid_ohlc")
            break
    return errors


def _daily_with_history(symbol: str, now: datetime, timeout: float) -> tuple[list[dict], list[str]]:
    """Fetch closed UTC daily bars with overlapping 90-day boundary checks."""
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    urls = [DAILY_CANDLES.format(symbol=symbol)]
    rows = []
    for page in range(3):
        if page:
            end = today - timedelta(days=89 * page) + timedelta(days=1)
            start = end - timedelta(days=90)
            urls.append(DAILY_HISTORY.format(symbol=symbol, start=int(start.timestamp() * 1000),
                                              end=int(end.timestamp() * 1000)))
        response = _get_json(urls[-1], timeout)
        if response.get("code") != "00000" or not isinstance(response.get("data"), list):
            raise ValueError("daily provider response unsuccessful")
        rows.extend(response["data"])
    candles = _closed_candles(rows, now, timedelta(days=1))
    by_time = {}
    for candle in candles:
        key = candle["open_at_utc"]
        if key in by_time and by_time[key] != candle:
            raise ValueError("conflicting daily candles across pages")
        by_time[key] = candle
    ordered = sorted(by_time.values(), key=lambda c: c["open_at_utc"])
    return ordered[-250:], urls


def fetch_public_snapshot(symbols: list[str] | None = None, candle_limit: int = 300, timeout: float = 12.0) -> dict:
    """Allowlisted public snapshot for a chat-assisted research request. No API keys."""
    symbols = symbols or list(CONTRACTS)
    now = datetime.now(timezone.utc)
    out = {
        "ok": True,
        "request_started_at_utc": now.isoformat(),
        "retrieved_at_utc": now.isoformat(),
        "cutoff_at_utc": now.isoformat(),
        "source_host": ALLOWED_HOST,
        "timeframe": "1H",
        "closed_bars_only": True,
        "exact_liquidation_verified": False,
        "contracts_provenance": PROVENANCE,
        "instruments": {},
        "errors": [],
        "missing_datasets": [
            "macro", "fed_policy", "real_rates", "dxy", "tga", "rrp",
            "btc_etf_flows", "onchain", "equities_fundamentals", "sentiment_indices",
        ],
    }
    for sym in symbols:
        inst = {"symbol": sym, "instrument_id": CONTRACTS[sym]["instrument_id"], "venue": "bitget"}
        try:
            ticker_url = TICKER.format(symbol=sym)
            data = _get_json(ticker_url, timeout)
            row = (data.get("data") or [None])[0]
            if not row:
                raise ValueError("empty ticker")
            inst["ticker"] = _parse_ticker(row, datetime.now(timezone.utc), ticker_url)
        except Exception as exc:
            out["ok"] = False
            out["errors"].append({"symbol": sym, "part": "ticker", "error": type(exc).__name__})
            inst["ticker"] = None
        try:
            curl = CANDLES.format(symbol=sym, limit=int(candle_limit))
            data = _get_json(curl, timeout)
            inst["candles_closed_1h"] = _closed_candles(data.get("data") or [], now)
            inst["candles_source_url"] = curl
            inst["candles_count"] = len(inst["candles_closed_1h"])
            for problem in candle_quality(inst["candles_closed_1h"], now):
                out["ok"] = False
                out["errors"].append({"symbol": sym, "part": "candles", "error": problem})
        except Exception as exc:
            out["ok"] = False
            out["errors"].append({"symbol": sym, "part": "candles", "error": type(exc).__name__})
            inst["candles_closed_1h"] = []
        try:
            daily, daily_urls = _daily_with_history(sym, now, timeout)
            issues = candle_quality(daily, now, timedelta(days=1))
            inst["candles_closed_1d"] = daily
            inst["daily_source_url"] = daily_urls[0]
            inst["daily_history_source_urls"] = daily_urls[1:]
            inst["daily_quality_errors"] = issues
            for problem in issues:
                out["errors"].append({"symbol": sym, "part": "daily_context", "error": problem})
        except Exception as exc:
            out["errors"].append({"symbol": sym, "part": "daily_context", "error": type(exc).__name__})
            inst["candles_closed_1d"] = []
            inst["daily_quality_errors"] = ["daily_fetch_failed"]
        try:
            furl = FUNDING.format(symbol=sym)
            data = _get_json(furl, timeout)
            if data.get("code") != "00000":
                raise ValueError("funding provider response unsuccessful")
            frow = data.get("data") or {}
            if isinstance(frow, list):
                frow = frow[0] if frow else {}
            inst["funding"] = _parse_current_funding(frow, furl, retrieved_at=datetime.now(timezone.utc), symbol=sym)
        except Exception as exc:
            out["errors"].append({"symbol": sym, "part": "funding", "error": type(exc).__name__})
            inst["funding"] = None
        try:
            oi_url = OI_HISTORY.format(symbol=sym)
            oi_data = _get_json(oi_url, timeout)
            inst["open_interest"] = _parse_open_interest_history(oi_data, sym, datetime.now(timezone.utc), oi_url)
            if inst["open_interest"]["change_24h_pct"] is None:
                out["missing_datasets"].append(f"open_interest_change_24h:{sym}")
        except Exception as exc:
            inst["open_interest"] = None
            out["missing_datasets"].append(f"open_interest_history:{sym}")
            out["errors"].append({"symbol": sym, "part": "open_interest", "error": type(exc).__name__})
        out["instruments"][sym] = inst
    completed = datetime.now(timezone.utc)
    out["retrieved_at_utc"] = completed.isoformat()
    for sym, inst in out["instruments"].items():
        ticker = inst.get("ticker")
        if ticker:
            age = completed - datetime.fromisoformat(ticker["observed_at_utc"])
            if not timedelta(seconds=-5) <= age <= timedelta(minutes=5):
                out["ok"] = False
                out["errors"].append({"symbol": sym, "part": "ticker", "error": "stale_at_completion"})
        try:
            for problem in candle_quality(inst.get("candles_closed_1h") or [], completed):
                if not any(e["symbol"] == sym and e["part"] == "candles" and e["error"] == problem for e in out["errors"]):
                    out["ok"] = False
                    out["errors"].append({"symbol": sym, "part": "candles", "error": problem})
        except (KeyError, ValueError, TypeError):
            out["ok"] = False
            out["errors"].append({"symbol": sym, "part": "candles", "error": "invalid_at_completion"})
    return out
