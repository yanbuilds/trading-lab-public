"""Read-only public Bitget USDT-futures books, tickers, and contract specs.

Official public host already used by this project: https://api.bitget.com
Official v2 mix path from Bitget's published SDK (`MarketApi.orderbook`):
GET /api/v2/mix/market/orderbook
GET /api/v2/mix/market/ticker
GET /api/v2/mix/market/contracts

This module never signs requests, places orders, or uses private endpoints.
`requestTime` is the response envelope time; `data.ts` is the book generation
time; both are stored separately from local retrieval time. Mark price is
recorded and is not an executable fill.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from typing import Any
from urllib.parse import urlparse

import httpx

ALLOWED_HOSTS = {"api.bitget.com"}
BASE = "https://api.bitget.com"
ORDERBOOK_PATH = "/api/v2/mix/market/orderbook"
TICKER_PATH = "/api/v2/mix/market/ticker"
CONTRACTS_PATH = "/api/v2/mix/market/contracts"
PRODUCT = "USDT-FUTURES"
DOCS = {
    "sdk_orderbook": "https://github.com/BitgetLimited/V3-bitget-api-sdk/blob/master/bitget-python-sdk-api/bitget/v2/mix/market_api.py",
    "contracts": f"{BASE}{CONTRACTS_PATH}?productType={PRODUCT}",
    "ticker": f"{BASE}{TICKER_PATH}?productType={PRODUCT}&symbol={{symbol}}",
    "orderbook": f"{BASE}{ORDERBOOK_PATH}?productType={PRODUCT}&symbol={{symbol}}&limit={{limit}}",
}
USER_AGENT = "PIIA-research-lab/0.1 (read-only public research; no trading)"
DEFAULT_LIMIT = 15
MAX_LIMIT = 100
QUOTE_MAX_AGE = timedelta(minutes=5)
FUTURE_SKEW = timedelta(seconds=5)


def _aware(now: datetime | None) -> datetime:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("timezone-aware time required")
    return now.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _unix_ms(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        ts = int(str(value))
    except (TypeError, ValueError):
        return None
    if ts > 10**15:
        return None
    if ts < 10**11:
        ts *= 1000
    try:
        return datetime.fromtimestamp(ts / 1000.0, timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _dec(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    if not number.is_finite():
        return None
    return number


def allowed_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.hostname in ALLOWED_HOSTS and not parsed.username and not parsed.password


def _get(client: httpx.Client, url: str) -> httpx.Response:
    if not allowed_url(url):
        raise ValueError(f"refusing non-allowlisted url {url}")
    response = client.get(url, follow_redirects=False)
    if not allowed_url(str(response.url)):
        raise ValueError(f"redirected off allowlisted host {response.url}")
    response.raise_for_status()
    return response


def parse_levels(raw: Any, *, descending: bool) -> list[tuple[Decimal, Decimal]]:
    """Parse [price, quantity] rows. Bids must descend; asks must ascend. Best is first."""
    if not isinstance(raw, list):
        raise ValueError("invalid_book_levels")
    out: list[tuple[Decimal, Decimal]] = []
    for row in raw:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            raise ValueError("invalid_book_level")
        price, size = _dec(row[0]), _dec(row[1])
        if price is None or size is None or price <= 0 or size < 0:
            raise ValueError("invalid_book_level")
        out.append((price, size))
    for prev, curr in zip(out, out[1:]):
        if descending and curr[0] > prev[0]:
            raise ValueError("bids_not_descending")
        if not descending and curr[0] < prev[0]:
            raise ValueError("asks_not_ascending")
    return out


def parse_orderbook(payload: Any, *, symbol: str, source_url: str, retrieved: datetime, raw: bytes | None = None) -> dict[str, Any]:
    retrieved = _aware(retrieved)
    out: dict[str, Any] = {
        "ok": False,
        "symbol": symbol,
        "source_url": source_url,
        "retrieved_at_utc": _iso(retrieved),
        "endpoint_request_time_utc": None,
        "book_generated_at_utc": None,
        "mark_is_not_executable": True,
        "size_unit": "contracts_matching_minTradeNum",
        "raw_sha256": hashlib.sha256(raw).hexdigest() if raw else None,
    }
    if not isinstance(payload, dict) or payload.get("code") != "00000":
        out["reason"] = "invalid_or_unsuccessful_orderbook"
        return out
    envelope_ts = _unix_ms(payload.get("requestTime"))
    data = payload.get("data")
    if not isinstance(data, dict):
        out["reason"] = "missing_book_data"
        out["endpoint_request_time_utc"] = _iso(envelope_ts) if envelope_ts else None
        return out
    book_ts = _unix_ms(data.get("ts") or data.get("timestamp"))
    out["endpoint_request_time_utc"] = _iso(envelope_ts) if envelope_ts else None
    out["book_generated_at_utc"] = _iso(book_ts) if book_ts else None
    try:
        bids = parse_levels(data.get("bids"), descending=True)
        asks = parse_levels(data.get("asks"), descending=False)
    except ValueError as exc:
        out["reason"] = str(exc)
        return out
    best_bid = bids[0][0] if bids else None
    best_ask = asks[0][0] if asks else None
    crossed = best_bid is not None and best_ask is not None and best_bid >= best_ask
    empty = not bids or not asks
    stale = False
    future = False
    clock_ref = book_ts or envelope_ts
    if clock_ref is None:
        out["reason"] = "missing_book_or_endpoint_time"
        return out
    if clock_ref - retrieved > FUTURE_SKEW:
        future = True
    if retrieved - clock_ref > QUOTE_MAX_AGE:
        stale = True
    mid = None
    spread = None
    if best_bid is not None and best_ask is not None:
        mid = (best_bid + best_ask) / Decimal("2")
        spread = best_ask - best_bid
    reason = None
    if empty:
        reason = "empty_book"
    elif crossed:
        reason = "crossed_book"
    elif future:
        reason = "future_book_timestamp"
    elif stale:
        reason = "stale_book"
    out.update({
        "ok": reason is None,
        "reason": reason,
        "bids": [{"price": str(p), "size": str(s)} for p, s in bids],
        "asks": [{"price": str(p), "size": str(s)} for p, s in asks],
        "best_bid": None if best_bid is None else str(best_bid),
        "best_bid_size": None if not bids else str(bids[0][1]),
        "best_ask": None if best_ask is None else str(best_ask),
        "best_ask_size": None if not asks else str(asks[0][1]),
        "mid": None if mid is None else str(mid),
        "spread": None if spread is None else str(spread),
        "bid_levels": len(bids),
        "ask_levels": len(asks),
        "visible_bid_size_sum": str(sum((s for _p, s in bids), Decimal("0"))),
        "visible_ask_size_sum": str(sum((s for _p, s in asks), Decimal("0"))),
        "levels_note": (
            "Official v2 mix book rows are [price, quantity]. "
            "Bids are price-descending and asks price-ascending; the first row is the best level. "
            "Visible size is not a guarantee of a future fill."
        ),
    })
    return out


def parse_ticker(payload: Any, *, symbol: str, source_url: str, retrieved: datetime, raw: bytes | None = None) -> dict[str, Any]:
    retrieved = _aware(retrieved)
    out: dict[str, Any] = {
        "ok": False,
        "symbol": symbol,
        "source_url": source_url,
        "retrieved_at_utc": _iso(retrieved),
        "endpoint_request_time_utc": None,
        "ticker_observed_at_utc": None,
        "mark_is_not_executable": True,
        "raw_sha256": hashlib.sha256(raw).hexdigest() if raw else None,
    }
    if not isinstance(payload, dict) or payload.get("code") != "00000":
        out["reason"] = "invalid_or_unsuccessful_ticker"
        return out
    envelope_ts = _unix_ms(payload.get("requestTime"))
    out["endpoint_request_time_utc"] = _iso(envelope_ts) if envelope_ts else None
    data = payload.get("data")
    row = data[0] if isinstance(data, list) and data else data
    if not isinstance(row, dict):
        out["reason"] = "missing_ticker_data"
        return out
    if row.get("symbol") not in (None, symbol):
        out["reason"] = "wrong_ticker_symbol"
        return out
    last, mark = _dec(row.get("lastPr") or row.get("last")), _dec(row.get("markPrice") or row.get("mark"))
    index = _dec(row.get("indexPrice"))
    observed = _unix_ms(row.get("ts") or row.get("timestamp")) or envelope_ts
    out["ticker_observed_at_utc"] = _iso(observed) if observed else None
    if last is None or last <= 0 or mark is None or mark <= 0:
        out["reason"] = "invalid_ticker_price"
        return out
    if observed is None:
        out["reason"] = "missing_ticker_timestamp"
        return out
    if observed - retrieved > FUTURE_SKEW:
        out["reason"] = "future_ticker_timestamp"
        return out
    if retrieved - observed > QUOTE_MAX_AGE:
        out["reason"] = "stale_ticker"
        return out
    out.update({
        "ok": True,
        "last": str(last),
        "mark": str(mark),
        "index": None if index is None else str(index),
        "best_bid": None if _dec(row.get("bidPr") or row.get("bestBid")) is None else str(_dec(row.get("bidPr") or row.get("bestBid"))),
        "best_ask": None if _dec(row.get("askPr") or row.get("bestAsk")) is None else str(_dec(row.get("askPr") or row.get("bestAsk"))),
        "funding_rate": None if row.get("fundingRate") in (None, "") else str(row.get("fundingRate")),
        "reason": None,
    })
    return out


def parse_contracts(payload: Any, symbols: tuple[str, ...], *, source_url: str, retrieved: datetime, raw: bytes | None = None) -> dict[str, Any]:
    retrieved = _aware(retrieved)
    out: dict[str, Any] = {
        "ok": False,
        "source_url": source_url,
        "retrieved_at_utc": _iso(retrieved),
        "endpoint_request_time_utc": None,
        "raw_sha256": hashlib.sha256(raw).hexdigest() if raw else None,
        "contracts": {},
        "exact_liquidation_verified": False,
        "margin_tiers_supported": False,
    }
    if not isinstance(payload, dict) or payload.get("code") != "00000" or not isinstance(payload.get("data"), list):
        out["reason"] = "invalid_or_unsuccessful_contracts"
        return out
    envelope_ts = _unix_ms(payload.get("requestTime"))
    out["endpoint_request_time_utc"] = _iso(envelope_ts) if envelope_ts else None
    wanted = set(symbols)
    found = {}
    for row in payload["data"]:
        if not isinstance(row, dict) or row.get("symbol") not in wanted:
            continue
        symbol = str(row["symbol"])
        min_qty, step, min_notional = _dec(row.get("minTradeNum")), _dec(row.get("sizeMultiplier")), _dec(row.get("minTradeUSDT"))
        taker = _dec(row.get("takerFeeRate"))
        if min_qty is None or step is None or min_notional is None or taker is None:
            continue
        found[symbol] = {
            "symbol": symbol,
            "status": row.get("symbolStatus"),
            "instrument_type": row.get("symbolType"),
            "min_qty": str(min_qty),
            "qty_step": str(step),
            "min_notional": str(min_notional),
            "taker": str(taker),
            "maker": None if _dec(row.get("makerFeeRate")) is None else str(_dec(row.get("makerFeeRate"))),
            "contract_size": "1",
            "price_place": row.get("pricePlace"),
            "volume_place": row.get("volumePlace"),
            "min_leverage": row.get("minLever"),
            "max_leverage_venue": row.get("maxLever"),
            "max_market_order_qty": None if _dec(row.get("maxMarketOrderQty")) is None else str(_dec(row.get("maxMarketOrderQty"))),
            "max_order_qty": None if _dec(row.get("maxOrderQty")) is None else str(_dec(row.get("maxOrderQty"))),
            "quantity_unit": "contracts_matching_minTradeNum",
            "maintenance_tiers_in_this_payload": False,
        }
    missing = [s for s in symbols if s not in found]
    out["contracts"] = found
    out["ok"] = not missing
    out["reason"] = None if not missing else "missing_contract_specs:" + ",".join(missing)
    return out


def floor_qty(qty: Decimal, step: Decimal) -> Decimal:
    if qty < 0 or step <= 0:
        raise ValueError("invalid quantity/step")
    return (qty / step).to_integral_value(rounding=ROUND_DOWN) * step


def walk_book(levels: list[dict[str, str]] | list[tuple[Decimal, Decimal]], qty: Decimal) -> dict[str, Any]:
    """Fill using only visible levels, best first. Never invent an unseen tail."""
    if qty <= 0:
        return {"ok": False, "reason": "non_positive_qty"}
    remaining = qty
    filled = Decimal("0")
    notional = Decimal("0")
    fills = []
    parsed: list[tuple[Decimal, Decimal]] = []
    for row in levels:
        if isinstance(row, dict):
            price, size = _dec(row.get("price")), _dec(row.get("size"))
        else:
            price, size = row
        if price is None or size is None or price <= 0 or size < 0:
            return {"ok": False, "reason": "invalid_book_level"}
        parsed.append((price, size))
    for price, size in parsed:
        if remaining <= 0:
            break
        take = remaining if remaining <= size else size
        if take <= 0:
            continue
        fills.append({"price": str(price), "size": str(take)})
        filled += take
        notional += price * take
        remaining -= take
    if remaining > 0:
        return {
            "ok": False,
            "reason": "insufficient_visible_depth",
            "requested_qty": str(qty),
            "filled_qty": str(filled),
            "unfilled_qty": str(remaining),
            "fills": fills,
            "did_not_assume_unseen_tail": True,
        }
    vwap = notional / qty
    return {
        "ok": True,
        "reason": None,
        "requested_qty": str(qty),
        "filled_qty": str(filled),
        "unfilled_qty": "0",
        "vwap": str(vwap),
        "notional": str(notional),
        "fills": fills,
        "did_not_assume_unseen_tail": True,
        "future_fill_not_guaranteed": True,
    }


def fetch_execution_books(
    symbols: tuple[str, ...] = ("BTCUSDT", "XAUUSDT", "NDX100USDT"),
    *,
    timeout: float = 15.0,
    now: datetime | None = None,
    client: httpx.Client | None = None,
    limit: int = DEFAULT_LIMIT,
    snapshot_dir: Any = None,
) -> dict[str, Any]:
    """Fetch a current public execution snapshot: contracts, tickers, and books."""
    batch_started = _aware(now)
    if limit < 1 or limit > MAX_LIMIT:
        raise ValueError("orderbook limit out of documented public range")
    out: dict[str, Any] = {
        "kind": "bitget_public_execution_books",
        "retrieved_at_utc": _iso(batch_started),
        "batch_started_at_utc": _iso(batch_started),
        "historical_replay_supported": False,
        "exact_liquidation_verified": False,
        "margin_tiers_supported": False,
        "mark_is_not_executable": True,
        "future_fill_not_guaranteed": True,
        "docs": DOCS,
        "errors": {},
        "ok": False,
        "instruments": {},
    }
    close_client = False
    if client is None:
        client = httpx.Client(timeout=timeout, follow_redirects=False, headers={"User-Agent": USER_AGENT})
        close_client = True
    raw_store: dict[str, bytes] = {}
    try:
        contracts_url = f"{BASE}{CONTRACTS_PATH}?productType={PRODUCT}"
        contracts_resp = _get(client, contracts_url)
        raw_store["contracts.json"] = contracts_resp.content
        contracts = parse_contracts(contracts_resp.json(), symbols, source_url=str(contracts_resp.url), retrieved=_aware(now), raw=contracts_resp.content)
        out["contracts"] = contracts
        for symbol in symbols:
            inst: dict[str, Any] = {"symbol": symbol}
            try:
                ticker_url = f"{BASE}{TICKER_PATH}?productType={PRODUCT}&symbol={symbol}"
                ticker_resp = _get(client, ticker_url)
                raw_store[f"ticker_{symbol}.json"] = ticker_resp.content
                ticker_retrieved = _aware(now)
                inst["ticker"] = parse_ticker(ticker_resp.json(), symbol=symbol, source_url=str(ticker_resp.url), retrieved=ticker_retrieved, raw=ticker_resp.content)
            except (httpx.HTTPError, ValueError) as exc:
                inst["ticker"] = {"ok": False, "reason": f"{type(exc).__name__}: {exc}", "symbol": symbol}
                out["errors"][f"ticker:{symbol}"] = inst["ticker"]["reason"]
            try:
                book_url = f"{BASE}{ORDERBOOK_PATH}?productType={PRODUCT}&symbol={symbol}&limit={limit}"
                book_resp = _get(client, book_url)
                raw_store[f"orderbook_{symbol}.json"] = book_resp.content
                book_retrieved = _aware(now)
                inst["orderbook"] = parse_orderbook(book_resp.json(), symbol=symbol, source_url=str(book_resp.url), retrieved=book_retrieved, raw=book_resp.content)
            except (httpx.HTTPError, ValueError) as exc:
                inst["orderbook"] = {"ok": False, "reason": f"{type(exc).__name__}: {exc}", "symbol": symbol}
                out["errors"][f"orderbook:{symbol}"] = inst["orderbook"]["reason"]
            inst["contract"] = (contracts.get("contracts") or {}).get(symbol)
            inst["ok"] = bool(inst.get("ticker", {}).get("ok") and inst.get("orderbook", {}).get("ok") and inst.get("contract"))
            if inst["contract"] and inst["contract"].get("status") not in (None, "normal"):
                inst["ok"] = False
                inst["reason"] = f"contract_status_{inst['contract'].get('status')}"
            out["instruments"][symbol] = inst
        out["ok"] = all(item.get("ok") for item in out["instruments"].values()) and contracts.get("ok")
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        out["errors"]["fetch"] = f"{type(exc).__name__}: {exc}"
        out["ok"] = False
    finally:
        if close_client:
            client.close()
    if snapshot_dir is not None:
        from pathlib import Path
        import json
        dest = Path(snapshot_dir)
        dest.mkdir(parents=True, exist_ok=True)
        for name, content in raw_store.items():
            (dest / name).write_bytes(content)
        (dest / "snapshot.json").write_text(json.dumps(out, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return out
