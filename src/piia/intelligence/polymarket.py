"""Read-only Polymarket prediction-market evidence from official public APIs.

Public documented hosts only:
- Gamma discovery: https://gamma-api.polymarket.com
- CLOB books/prices: https://clob.polymarket.com
- Price history: https://data-api.polymarket.com

This adapter never places bets, signs wallets, creates API credentials, or
treats quoted odds as facts or calibrated probabilities. It collects a current
sample of Fed/rate, inflation, recession, political-macro, Bitcoin, gold, and
Nasdaq-related events when those markets exist. Absence in the sample is not
global coverage. Today's odds are not backfilled into historical decisions.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode, urlparse

import httpx

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
DATA_BASE = "https://data-api.polymarket.com"
DOCS_API = "https://docs.polymarket.com/getting-started/api"
DOCS_DISCOVER = "https://docs.polymarket.com/market-data/discover-markets"
DOCS_BOOKS = "https://docs.polymarket.com/market-data/prices-order-books"
EVENT_PAGE = "https://polymarket.com/event/{slug}"
MARKET_PAGE = "https://polymarket.com/market/{slug}"
ALLOWED_HOSTS = {
    "gamma-api.polymarket.com",
    "clob.polymarket.com",
    "data-api.polymarket.com",
}
USER_AGENT = "PIIA-research-lab/0.1 (read-only public research; no trading)"
SEARCH_QUERIES = ("fed", "inflation", "recession", "bitcoin", "gold", "nasdaq")
MAX_EVENTS = 6
MAX_MARKETS_PER_EVENT = 8
MAX_CLOB_MARKETS = 12
MAX_DESCRIPTION = 2000
THIN_SPREAD = 0.05
CHANGE_TOLERANCE = timedelta(hours=1)
BOOK_MAX_AGE = timedelta(minutes=5)

# Conservative topical filter. Short-horizon up/down contracts are excluded.
EXCLUDE_RE = re.compile(
    r"alchemy|marathon fusion|\bup or down\b|\b\d+\s*(?:s|sec|secs|m|min|mins|h|hr|hrs)\b",
    re.I,
)
FED_RE = re.compile(r"\b(?:fed|fomc|federal reserve|interest rates?|rate cuts?|rate hikes?|bps)\b", re.I)
INFLATION_RE = re.compile(r"\b(?:inflation|cpi|pce)\b", re.I)
RECESSION_RE = re.compile(r"\brecession\b", re.I)
BTC_RE = re.compile(r"\b(?:bitcoin|btc)\b", re.I)
GOLD_RE = re.compile(r"\b(?:gold|xauusd|xau)\b", re.I)
NDX_RE = re.compile(r"\b(?:nasdaq(?:-|\s+)?100|\bndx\b|\bqqq\b)\b", re.I)
POLITICAL_RE = re.compile(
    r"\b(?:election|president|congress|senate|white house|debt ceiling|government shutdown|tariff|inauguration)\b",
    re.I,
)
NESTED_RE = re.compile(r"\b(?:hit|reach|above|below|high|low)\b", re.I)
BUCKET_RE = re.compile(r"(?:[≤≥<>]=?\s*)?\d+(?:\.\d+)?\s*%|(?:\d+\s*(?:bps|bp))", re.I)


def _aware(now: datetime | None) -> datetime:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("timezone-aware time required")
    return now.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def decode_maybe_json(value: Any) -> Any:
    """Gamma often JSON-encodes arrays such as outcomes and clobTokenIds."""
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in "[{":
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _unix_utc(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        ts = float(value)
    except (TypeError, ValueError):
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        return None
    if ts > 10**12:
        ts = ts / 1000.0
    try:
        return datetime.fromtimestamp(ts, timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _text(*parts: Any) -> str:
    return " ".join(str(part or "") for part in parts)


def classify_topic(text: str) -> str | None:
    blob = text or ""
    if EXCLUDE_RE.search(blob):
        return None
    if FED_RE.search(blob):
        return "fed_rate"
    if INFLATION_RE.search(blob):
        return "inflation"
    if RECESSION_RE.search(blob):
        return "recession"
    if BTC_RE.search(blob):
        return "bitcoin"
    if GOLD_RE.search(blob):
        return "gold"
    if NDX_RE.search(blob):
        return "nasdaq"
    if POLITICAL_RE.search(blob):
        return "political_macro"
    return None


def asset_mapping(category: str, title: str) -> dict[str, Any]:
    note = (
        "Odds are a contemporaneous quoted token price, not a fact, not a "
        "calibrated probability, and not a forced directional signal."
    )
    if category == "bitcoin":
        return {"relevant_instruments": ["BTCUSDT"], "mapping_kind": "topic_overlap", "note": note}
    if category == "gold":
        return {"relevant_instruments": ["XAUUSDT"], "mapping_kind": "topic_overlap", "note": note}
    if category == "nasdaq":
        if re.search(r"round-the-clock|24\s*hour|trading hours", title or "", re.I):
            return {
                "relevant_instruments": ["NDX100USDT"],
                "mapping_kind": "exchange_structure_not_index_price",
                "note": "This is not a Nasdaq-100 price event. " + note,
            }
        return {"relevant_instruments": ["NDX100USDT"], "mapping_kind": "topic_overlap", "note": note}
    return {
        "relevant_instruments": ["BTCUSDT", "XAUUSDT", "NDX100USDT"],
        "mapping_kind": "macro_context_only",
        "note": "Macro/political event context only; no automatic asset stance. " + note,
    }


def dependence_kind(markets: list[dict[str, Any]]) -> dict[str, Any]:
    open_markets = [m for m in markets if not m.get("closed")]
    titles = [str(m.get("group_item_title") or m.get("question") or "") for m in open_markets]
    nested = sum(1 for title in titles if NESTED_RE.search(title)) >= 2
    buckets = sum(1 for title in titles if BUCKET_RE.search(title)) >= 2
    if len(open_markets) <= 1:
        kind = "single_binary"
    elif nested and buckets:
        kind = "mixed"
    elif nested:
        kind = "nested_thresholds"
    elif buckets or len(open_markets) > 1:
        kind = "mutually_exclusive_or_grouped"
    else:
        kind = "grouped"
    seen: dict[str, int] = {}
    duplicates = []
    for title in titles:
        key = re.sub(r"\s+", " ", title.strip().lower())
        if not key:
            continue
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 2:
            duplicates.append(title)
    return {
        "kind": kind,
        "do_not_sum_dependent_probabilities": True,
        "not_a_fact": True,
        "duplicate_or_relisted_titles": duplicates,
        "note": (
            "Markets under one event can be mutually exclusive, nested, or overlapping. "
            "Do not add their implied probabilities together or treat a 70% quote as a fact."
        ),
    }


def event_url(slug: str | None) -> str | None:
    if not slug or not re.fullmatch(r"[a-zA-Z0-9_-]+", str(slug)):
        return None
    return EVENT_PAGE.format(slug=slug)


def market_url(slug: str | None) -> str | None:
    if not slug or not re.fullmatch(r"[a-zA-Z0-9_-]+", str(slug)):
        return None
    return MARKET_PAGE.format(slug=slug)


def allowed_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.hostname in ALLOWED_HOSTS and not parsed.username and not parsed.password


def _get(client: httpx.Client, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
    if not allowed_url(url):
        raise ValueError(f"refusing non-allowlisted url {url}")
    response = client.get(url, params=params, follow_redirects=False)
    if not allowed_url(str(response.url)):
        raise ValueError(f"redirected off allowlisted host {response.url}")
    response.raise_for_status()
    return response


def _raw_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def best_book_level(levels: Any, *, side: str) -> tuple[float | None, float | None]:
    """Official docs: bids ascend, asks descend, so the best level is the last entry."""
    if not isinstance(levels, list) or not levels:
        return None, None
    last = levels[-1]
    if not isinstance(last, dict):
        return None, None
    return _finite(last.get("price")), _finite(last.get("size"))


def sum_sizes(levels: Any) -> float | None:
    if not isinstance(levels, list):
        return None
    total = 0.0
    found = False
    for level in levels:
        if not isinstance(level, dict):
            continue
        size = _finite(level.get("size"))
        if size is None:
            continue
        total += size
        found = True
    return total if found else None


def parse_book(payload: Any, *, token_id: str, retrieved: datetime) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "unavailable", "reason": "invalid_book_payload", "token_id": token_id}
    bids = payload.get("bids") or []
    asks = payload.get("asks") or []
    best_bid, best_bid_size = best_book_level(bids, side="bid")
    best_ask, best_ask_size = best_book_level(asks, side="ask")
    retrieved = _aware(retrieved)
    asof = _unix_utc(payload.get("timestamp"))
    asset_id = payload.get("asset_id") or payload.get("assetId")
    age = (retrieved - asof).total_seconds() if asof is not None else None
    timestamp_status = ("missing" if age is None else "future" if age < 0 else
                        "stale" if age > BOOK_MAX_AGE.total_seconds() else "current")
    token_matches = asset_id is not None and str(asset_id) == token_id
    mid = None
    spread = None
    if best_bid is not None and best_ask is not None:
        mid = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
    min_size = _finite(payload.get("min_order_size") or payload.get("minOrderSize"))
    illiquid = best_bid is None or best_ask is None or (best_bid_size or 0) <= 0 or (best_ask_size or 0) <= 0
    crossed = best_bid is not None and best_ask is not None and best_bid > best_ask
    thin = illiquid or (spread is not None and spread >= THIN_SPREAD)
    if min_size is not None and not illiquid:
        if (best_bid_size or 0) < min_size and (best_ask_size or 0) < min_size:
            thin = True
    outlier = False
    for price in (best_bid, best_ask, mid, _finite(payload.get("last_trade_price") or payload.get("lastTradePrice"))):
        if price is not None and (price < 0 or price > 1):
            outlier = True
    if crossed:
        outlier = True
    return {
        "status": "ok",
        "source": "clob_public_book",
        "token_id": token_id,
        "condition_id": payload.get("market") or payload.get("conditionId") or payload.get("condition_id"),
        "asof_utc": _iso(asof) if asof is not None else None,
        "retrieved_at_utc": _iso(retrieved),
        "book_timestamp_status": timestamp_status,
        "book_age_seconds": age,
        "token_id_matches": token_matches,
        "usable_for_implied_probability": timestamp_status == "current" and token_matches and not outlier and not illiquid,
        "book_hash": payload.get("hash"),
        "tick_size": payload.get("tick_size") or payload.get("tickSize"),
        "min_order_size": payload.get("min_order_size") or payload.get("minOrderSize"),
        "neg_risk": payload.get("neg_risk") if "neg_risk" in payload else payload.get("negRisk"),
        "best_bid": best_bid,
        "best_bid_size": best_bid_size,
        "best_ask": best_ask,
        "best_ask_size": best_ask_size,
        "spread_from_book": spread,
        "midpoint_from_book": mid,
        "last_trade_price_in_book": _finite(payload.get("last_trade_price") or payload.get("lastTradePrice")),
        "visible_bid_size_sum": sum_sizes(bids),
        "visible_ask_size_sum": sum_sizes(asks),
        "bid_levels": len(bids) if isinstance(bids, list) else 0,
        "ask_levels": len(asks) if isinstance(asks, list) else 0,
        "size_unit": "outcome_tokens_resting_on_public_book",
        "size_unit_note": (
            "Resting CLOB size is in outcome tokens visible at read time. "
            "It is not guaranteed executable depth, not a Gamma liquidity_num, "
            "and not USD notional."
        ),
        "illiquid": illiquid,
        "thin_book": thin,
        "outlier_or_crossed": outlier,
    }


def parse_price_history(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        rows = payload.get("data")
    else:
        rows = payload
    points: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return points
    for row in rows:
        if not isinstance(row, dict):
            continue
        stamp = _unix_utc(row.get("timestamp") or row.get("t"))
        price = _finite(row.get("price"))
        if stamp is None or price is None:
            continue
        points.append({
            "timestamp_utc": _iso(stamp),
            "price": price,
            "resolution_seconds": row.get("resolution_seconds") if "resolution_seconds" in row else row.get("resolutionSeconds"),
        })
    points.sort(key=lambda item: item["timestamp_utc"])
    return points


def change_24h_pp(points: list[dict[str, Any]]) -> dict[str, Any]:
    """24h change in percentage points from documented history only, never a return."""
    out = {
        "status": "not_available",
        "change_24h_pp": None,
        "reason": "insufficient_documented_history",
        "unit": "percentage_points",
        "not_a_return": True,
    }
    if len(points) < 2:
        return out
    latest = points[-1]
    latest_ts = _unix_utc(latest["timestamp_utc"])
    if latest_ts is None:
        return out
    target = latest_ts - timedelta(hours=24)
    chosen = None
    best_delta = None
    for point in points[:-1]:
        stamp = _unix_utc(point["timestamp_utc"])
        if stamp is None:
            continue
        delta = abs(stamp - target)
        if delta <= CHANGE_TOLERANCE and (best_delta is None or delta < best_delta):
            chosen = point
            best_delta = delta
    if chosen is None:
        out["reason"] = "no_documented_point_within_1h_of_24h"
        return out
    out.update({
        "status": "ok",
        "change_24h_pp": (latest["price"] - chosen["price"]) * 100.0,
        "reason": None,
        "from_price": chosen["price"],
        "to_price": latest["price"],
        "from_timestamp_utc": chosen["timestamp_utc"],
        "to_timestamp_utc": latest["timestamp_utc"],
    })
    return out


def _outcome_ids(market: dict[str, Any]) -> tuple[list[str], list[str], str]:
    version = str(market.get("version") or "v1").lower()
    names = decode_maybe_json(market.get("outcomes"))
    if not isinstance(names, list):
        names = []
    names = [str(name) for name in names]
    if version == "v2":
        ids = decode_maybe_json(market.get("positionIds") or market.get("position_ids"))
        kind = "position_id"
    else:
        ids = decode_maybe_json(market.get("clobTokenIds") or market.get("clob_token_ids"))
        kind = "token_id"
    if not isinstance(ids, list):
        ids = []
    ids = [str(item) for item in ids if item not in (None, "")]
    return names, ids, kind


def _implied(price: float | None, basis: str) -> dict[str, Any]:
    return {
        "value": price,
        "basis": basis if price is not None else None,
        "is_calibrated_probability": False,
        "is_fact": False,
        "assumption": "standard_one_dollar_binary_payoff",
        "note": (
            "A quoted CLOB or Gamma price on a standard binary token is often "
            "read as an implied probability. It is not a verified probability, "
            "not a model forecast, and not a reason to force a trade."
        ),
    }


def parse_market(market: dict[str, Any], *, event_slug: str | None = None) -> dict[str, Any]:
    names, ids, id_kind = _outcome_ids(market)
    prices = decode_maybe_json(market.get("outcomePrices") or market.get("outcome_prices"))
    if not isinstance(prices, list):
        prices = []
    outcomes = []
    for index, name in enumerate(names):
        token = ids[index] if index < len(ids) else None
        quoted = _finite(prices[index]) if index < len(prices) else None
        outcomes.append({
            "name": name,
            "asset_id_kind": id_kind,
            "asset_id": token if token and re.fullmatch(r"[0-9]+", token) else None,
            "asset_id_invalid": bool(token) and not re.fullmatch(r"[0-9]+", token),
            "platform_quoted_price": quoted,
            "platform_quoted_price_note": (
                "Gamma outcomePrices quoted field, distinct from the CLOB midpoint, "
                "bid/ask, and last trade."
            ),
            "book": {"status": "not_fetched"},
        })
    missing_outcomes = (not names) or (len(ids) != len(names))
    description = str(market.get("description") or "")[:MAX_DESCRIPTION]
    slug = market.get("slug")
    return {
        "market_id": str(market.get("id") or ""),
        "condition_id": market.get("conditionId") or market.get("condition_id"),
        "question": market.get("question"),
        "group_item_title": market.get("groupItemTitle") or market.get("group_item_title"),
        "slug": slug,
        "url": market_url(slug) or event_url(event_slug),
        "version": market.get("version"),
        "closed": bool(market.get("closed")),
        "active": market.get("active"),
        "accepting_orders": market.get("acceptingOrders"),
        "uma_resolution_status": market.get("umaResolutionStatus") or market.get("uma_resolution_status"),
        "deadline_utc": market.get("endDate") or market.get("end_date"),
        "resolve_criteria": description or None,
        "resolution_source": market.get("resolutionSource") or market.get("resolution_source") or None,
        "missing_outcomes": missing_outcomes,
        "outcomes": outcomes,
        "platform_metrics": {
            "liquidity_num": _finite(market.get("liquidityNum") if market.get("liquidityNum") is not None else market.get("liquidity")),
            "liquidity_unit_note": (
                "Gamma/platform liquidity_num is a platform-reported figure, "
                "not executable CLOB depth."
            ),
            "volume": _finite(market.get("volume")),
            "volume_24hr": _finite(market.get("volume24hr") or market.get("volume_24hr")),
            "gamma_best_bid": _finite(market.get("bestBid") or market.get("best_bid")),
            "gamma_best_ask": _finite(market.get("bestAsk") or market.get("best_ask")),
            "gamma_last_trade_price": _finite(market.get("lastTradePrice") or market.get("last_trade_price")),
            "gamma_spread": _finite(market.get("spread")),
        },
    }


def parse_event(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or not payload.get("id"):
        return None
    raw_markets = payload.get("markets") or []
    if not isinstance(raw_markets, list):
        raw_markets = []
    slug = payload.get("slug")
    markets = [parse_market(m, event_slug=slug) for m in raw_markets if isinstance(m, dict)]
    title = str(payload.get("title") or "")
    description = str(payload.get("description") or "")[:MAX_DESCRIPTION]
    text = _text(title, description, *(m.get("question") for m in markets))
    category = classify_topic(text)
    if category is None:
        return None
    closed = [m for m in markets if m.get("closed")]
    open_markets = [m for m in markets if not m.get("closed")]
    ranked = sorted(
        open_markets,
        key=lambda m: ((m.get("platform_metrics") or {}).get("volume_24hr") or 0),
        reverse=True,
    )
    included = ranked[:MAX_MARKETS_PER_EVENT]
    omitted = len(markets) - len(included)
    return {
        "event_id": str(payload.get("id")),
        "slug": slug,
        "title": title,
        "question": title,
        "resolve_criteria": description or None,
        "resolution_source": payload.get("resolutionSource") or payload.get("resolution_source") or None,
        "source": "polymarket_gamma",
        "deadline_utc": payload.get("endDate") or payload.get("end_date"),
        "status": {
            "closed": bool(payload.get("closed")),
            "active": payload.get("active"),
            "archived": payload.get("archived"),
        },
        "url": event_url(slug),
        "category": category,
        "asset_mapping": asset_mapping(category, title),
        "dependence": dependence_kind(markets),
        "condition_ids": [m.get("condition_id") for m in included if m.get("condition_id")],
        "markets_total": len(markets),
        "markets_open": len(open_markets),
        "markets_closed": len(closed),
        "markets_included": len(included),
        "markets_omitted": omitted,
        "markets": included,
        "coverage_note": (
            "This event is part of a topical sample. "
            f"{omitted} market(s) were omitted from the CLOB/detail cap."
            if omitted else "All parsed markets in this event were retained under the cap."
        ),
    }


def _scalar_payload(payload: Any, key: str) -> float | None:
    if isinstance(payload, dict):
        return _finite(payload.get(key) if key in payload else payload.get("price") or payload.get("mid") or payload.get("spread"))
    return _finite(payload)


def attach_clob(outcome: dict[str, Any], *, book: dict[str, Any], midpoint: Any, spread: Any, last: Any, history: Any, source_urls: dict[str, str], history_received: datetime | None = None) -> None:
    parsed = dict(book)
    parsed["source_urls"] = source_urls
    official_mid = _scalar_payload(midpoint, "mid")
    official_spread = _scalar_payload(spread, "spread")
    last_price = None
    last_side = None
    if isinstance(last, dict):
        last_price = _finite(last.get("price"))
        last_side = last.get("side")
    parsed["official_midpoint"] = official_mid
    parsed["official_spread"] = official_spread
    parsed["last_trade_price"] = last_price
    parsed["last_trade_side"] = last_side
    parsed["quoted_price_fields"] = {
        "best_bid": parsed.get("best_bid"),
        "best_ask": parsed.get("best_ask"),
        "official_midpoint": official_mid,
        "midpoint_from_book": parsed.get("midpoint_from_book"),
        "last_trade_price": last_price,
        "platform_quoted_price": outcome.get("platform_quoted_price"),
        "note": (
            "These are distinct quoted prices. official_midpoint comes from "
            "GET /midpoint; bid/ask come from the last book levels; last trade "
            "is GET /last-trade-price; platform_quoted_price is Gamma outcomePrices."
        ),
    }
    basis_price = (official_mid if official_mid is not None else parsed.get("midpoint_from_book")) if parsed.get("usable_for_implied_probability") else None
    if basis_price is not None and not 0 <= basis_price <= 1:
        basis_price = None
    parsed["implied_probability"] = _implied(basis_price, "official_or_book_midpoint_assuming_one_dollar_binary_payoff")
    points = parse_price_history(history)
    cutoff = _aware(history_received) if history_received is not None else _unix_utc(parsed.get("retrieved_at_utc"))
    if cutoff is None:
        points = []
    else:
        future_count = sum(_unix_utc(point["timestamp_utc"]) > cutoff for point in points)
        points = [point for point in points if _unix_utc(point["timestamp_utc"]) <= cutoff]
        parsed["history_future_points_excluded"] = future_count
        parsed["history_received_at_utc"] = _iso(cutoff)
    parsed["history_points_used"] = len(points)
    parsed["change_24h"] = change_24h_pp(points)
    outcome["book"] = parsed


def select_search_hits(payload: Any) -> list[dict[str, Any]]:
    events = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        return []
    selected = []
    for ev in events:
        if not isinstance(ev, dict) or not ev.get("id"):
            continue
        if ev.get("closed") is True:
            # Still allow a closed event through classify; later filter prefers open.
            pass
        text = _text(ev.get("title"), ev.get("slug"), ev.get("description"), ev.get("ticker"))
        category = classify_topic(text)
        if category is None:
            continue
        selected.append({"event_id": str(ev.get("id")), "slug": ev.get("slug"), "title": ev.get("title"), "closed": ev.get("closed"), "category": category})
    return selected


def _relevance_score(hit: dict[str, Any]) -> tuple[int, str]:
    """Prefer open U.S./Fed events over other-country same-topic hits."""
    title = f"{hit.get('title') or ''} {hit.get('slug') or ''}"
    score = 0 if hit.get("closed") else 10
    if hit.get("category") in {"fed_rate", "inflation", "recession", "political_macro"}:
        if re.search(r"\b(?:us|u\.s\.|united-states|united states|fed|fomc)\b", title, re.I):
            score += 8
        elif re.search(r"\b(?:japan|canada|brazil|south-africa|south africa|uk|britain|eu|china|india)\b", title, re.I):
            score -= 6
    return (-score, hit.get("event_id") or "")


def _prefer_open(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for hit in hits:
        unique.setdefault(hit["event_id"], hit)
    open_hits = [h for h in unique.values() if not h.get("closed")]
    closed_hits = [h for h in unique.values() if h.get("closed")]
    open_hits.sort(key=_relevance_score)
    # Keep topical diversity: best open hit per category, then fill.
    chosen: list[dict[str, Any]] = []
    seen_cat: set[str] = set()
    for hit in open_hits:
        if hit["category"] not in seen_cat:
            chosen.append(hit)
            seen_cat.add(hit["category"])
    for hit in open_hits:
        if hit not in chosen:
            chosen.append(hit)
        if len(chosen) >= MAX_EVENTS:
            return chosen[:MAX_EVENTS]
    if not chosen:
        return closed_hits[:MAX_EVENTS]
    return chosen[:MAX_EVENTS]


def fetch_prediction_markets(
    timeout: float = 20.0,
    *,
    now: datetime | None = None,
    client: httpx.Client | None = None,
    snapshot_dir: Any = None,
    fetch_books: bool = True,
) -> dict[str, Any]:
    """Fetch a current, sampled, read-only Polymarket research snapshot."""
    # A supplied clock is only a fixture aid with an injected client. Real
    # HTTP responses must be stamped when they are received, not backdated.
    fixed_time = now is not None and client is not None
    clock = lambda: _aware(now) if fixed_time else datetime.now(timezone.utc)
    retrieved = clock()
    out: dict[str, Any] = {
        "kind": "polymarket_public_prediction_markets",
        "retrieved_at_utc": _iso(retrieved),
        "collection_started_at_utc": _iso(retrieved),
        "asof_utc": _iso(retrieved),
        "historical_replay_supported": False,
        "historical_backfill": False,
        "alpha_claim": False,
        "odds_are_not_facts": True,
        "exhaustive_coverage": False,
        "complete": False,
        "usable_for_research": False,
        "source": "polymarket_official_public_http",
        "docs_urls": [DOCS_API, DOCS_DISCOVER, DOCS_BOOKS],
        "hosts": sorted(ALLOWED_HOSTS),
        "search_queries": list(SEARCH_QUERIES),
        "errors": {},
        "events": [],
        "raw_hashes": {},
        "cross_event_overlaps": [],
        "methodology": {
            "resolution_and_horizon": (
                "Each event keeps its published question, resolution text, "
                "resolution source, and deadline. The adapter does not override "
                "resolution rules or convert an event into a price forecast."
            ),
            "event_to_asset_mapping": (
                "BTC/gold/Nasdaq topical markets are tagged to the matching "
                "instrument only. Fed, inflation, recession, and political-macro "
                "events are macro context for all three and do not force a stance."
            ),
            "probability_policy": (
                "Quoted CLOB/Gamma prices are stored separately from implied "
                "probability. Implied probability assumes a $1 binary payoff and "
                "is not calibrated. 24h changes are percentage points from "
                "documented token history, otherwise not_available. Dependent "
                "outcomes are never summed."
            ),
        },
        "data_freshness": {
            "retrieved_at_utc": _iso(retrieved),
            "asof_utc": _iso(retrieved),
            "historical_backfill": False,
        },
    }
    close_client = False
    if client is None:
        client = httpx.Client(timeout=timeout, follow_redirects=False, headers={"User-Agent": USER_AGENT})
        close_client = True
    raw_store: dict[str, bytes] = {}
    try:
        hits: list[dict[str, Any]] = []
        for query in SEARCH_QUERIES:
            url = f"{GAMMA_BASE}/public-search"
            try:
                response = _get(client, url, params={"q": query})
            except (httpx.HTTPError, ValueError) as exc:
                out["errors"][f"search:{query}"] = f"{type(exc).__name__}: {exc}"
                continue
            raw_store[f"search_{query}.json"] = response.content
            out["raw_hashes"][f"search_{query}"] = _raw_hash(response.content)
            try:
                payload = response.json()
            except ValueError as exc:
                out["errors"][f"search:{query}"] = f"JSONDecodeError: {exc}"
                continue
            hits.extend(select_search_hits(payload))
        selected = _prefer_open(hits)
        out["search_hits_relevant"] = len({h["event_id"] for h in hits})
        out["events_selected"] = [h["event_id"] for h in selected]
        if not selected:
            out["errors"]["events"] = "no_relevant_open_or_closed_events_in_sample"
        booked = 0
        events: list[dict[str, Any]] = []
        for hit in selected:
            event_id = hit["event_id"]
            if not re.fullmatch(r"[0-9]+", event_id):
                out["errors"][f"event:{event_id}"] = "invalid_event_id"
                continue
            url = f"{GAMMA_BASE}/events/{event_id}"
            try:
                response = _get(client, url)
            except (httpx.HTTPError, ValueError) as exc:
                out["errors"][f"event:{event_id}"] = f"{type(exc).__name__}: {exc}"
                continue
            raw_store[f"event_{event_id}.json"] = response.content
            out["raw_hashes"][f"event_{event_id}"] = _raw_hash(response.content)
            try:
                parsed = parse_event(response.json())
            except ValueError as exc:
                out["errors"][f"event:{event_id}"] = f"{type(exc).__name__}: {exc}"
                continue
            if parsed is None:
                out["errors"][f"event:{event_id}"] = "not_relevant_after_detail_parse"
                continue
            parsed["asof_utc"] = _iso(clock())
            parsed["source_url"] = url
            if fetch_books:
                for market in parsed["markets"]:
                    if booked >= MAX_CLOB_MARKETS:
                        for outcome in market["outcomes"]:
                            if outcome.get("book", {}).get("status") == "not_fetched":
                                outcome["book"] = {"status": "not_fetched", "reason": "clob_market_cap_reached"}
                        continue
                    if market.get("missing_outcomes"):
                        continue
                    # Fetch the first valid outcome (typically Yes) as the representative book.
                    target = next((o for o in market["outcomes"] if o.get("asset_id") and not o.get("asset_id_invalid")), None)
                    if target is None:
                        for outcome in market["outcomes"]:
                            outcome["book"] = {"status": "unavailable", "reason": "missing_or_invalid_token_id"}
                        continue
                    token = target["asset_id"]
                    try:
                        book_resp = _get(client, f"{CLOB_BASE}/book", params={"token_id": token})
                        book_received = clock()
                        mid_resp = _get(client, f"{CLOB_BASE}/midpoint", params={"token_id": token})
                        spr_resp = _get(client, f"{CLOB_BASE}/spread", params={"token_id": token})
                        last_resp = _get(client, f"{CLOB_BASE}/last-trade-price", params={"token_id": token})
                        hist_resp = _get(
                            client,
                            f"{DATA_BASE}/v2/prices-history",
                            params={"token_id": token, "interval": "1d", "bucket_seconds": "3600"},
                        )
                        history_received = clock()
                    except (httpx.HTTPError, ValueError) as exc:
                        target["book"] = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
                        booked += 1
                        continue
                    raw_store[f"book_{token}.json"] = book_resp.content
                    out["raw_hashes"][f"book_{token}"] = _raw_hash(book_resp.content)
                    out["raw_hashes"][f"history_{token}"] = _raw_hash(hist_resp.content)
                    try:
                        book_payload = book_resp.json()
                        mid_payload = mid_resp.json()
                        spr_payload = spr_resp.json()
                        last_payload = last_resp.json()
                        hist_payload = hist_resp.json()
                    except ValueError as exc:
                        target["book"] = {"status": "unavailable", "reason": f"JSONDecodeError: {exc}"}
                        booked += 1
                        continue
                    book = parse_book(book_payload, token_id=token, retrieved=book_received)
                    attach_clob(
                        target,
                        book=book,
                        midpoint=mid_payload,
                        spread=spr_payload,
                        last=last_payload,
                        history=hist_payload,
                        history_received=history_received,
                        source_urls={
                            "book": str(book_resp.url),
                            "midpoint": str(mid_resp.url),
                            "spread": str(spr_resp.url),
                            "last_trade_price": str(last_resp.url),
                            "prices_history": str(hist_resp.url),
                        },
                    )
                    booked += 1
            events.append(parsed)
        out["events"] = events
        # Cross-event informational overlaps by category, not summed.
        by_cat: dict[str, list[str]] = {}
        for ev in events:
            by_cat.setdefault(ev["category"], []).append(ev["event_id"])
        out["cross_event_overlaps"] = [
            {
                "category": category,
                "event_ids": ids,
                "note": "These events share a topic. Their odds are not independent facts and must not be added together.",
            }
            for category, ids in by_cat.items()
            if len(ids) > 1
        ]
        completed = clock()
        out["retrieved_at_utc"] = _iso(completed)
        out["data_freshness"]["retrieved_at_utc"] = _iso(completed)
        out["usable_for_research"] = bool(events)
        out["complete"] = bool(events) and not any(k.startswith("event:") or k.startswith("search:") for k in out["errors"])
        asofs = [_unix_utc(ev.get("asof_utc")) for ev in events]
        book_asofs = []
        for ev in events:
            for market in ev.get("markets") or []:
                for outcome in market.get("outcomes") or []:
                    book = outcome.get("book") or {}
                    stamp = _unix_utc(book.get("asof_utc")) if book.get("book_timestamp_status") == "current" and book.get("token_id_matches") else None
                    if stamp:
                        book_asofs.append(stamp)
        stamps = [s for s in asofs + book_asofs if s]
        if stamps:
            out["asof_utc"] = _iso(max(stamps))
            out["data_freshness"]["asof_utc"] = out["asof_utc"]
        out["coverage_note"] = (
            "Sampled official public-search hits for "
            + ", ".join(SEARCH_QUERIES)
            + ". This is not an exhaustive scan of all Polymarket events."
        )
    except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
        out["errors"]["fetch"] = f"{type(exc).__name__}: {exc}"
        out["usable_for_research"] = False
        out["complete"] = False
    finally:
        if close_client:
            client.close()
    out["completed_at_utc"] = _iso(clock())
    if snapshot_dir is not None:
        from pathlib import Path
        dest = Path(snapshot_dir)
        dest.mkdir(parents=True, exist_ok=True)
        for name, content in raw_store.items():
            (dest / name).write_bytes(content)
        (dest / "context.json").write_text(json.dumps(out, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return out
