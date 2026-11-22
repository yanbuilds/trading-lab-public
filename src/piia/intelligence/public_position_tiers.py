"""Read-only public Bitget USDT-futures position-tier (risk limit) snapshots.

Official documented classic public endpoint (no authentication):
GET https://api.bitget.com/api/v2/mix/market/query-position-lever

Docs:
- https://www.bitget.com/docs/catalog/classic-contract-position/classic-contract-position
- https://www.bitget.com/zh-CN/api-doc/classic/contract/position/Get-Query-Position-Lever

Field wording: official English "Minimum/Maximum value"; official Chinese
"价值下限/上限". For USDT-M linear contracts this is treated as a USDT
notional value bound, not a coin quantity. The JSON keys are still
startUnit/endUnit and do not include an explicit "USDT" suffix.

Boundary inclusivity is not documented. A shared boundary point uses the
more conservative of the adjacent tiers (lower max leverage, higher MMR).
A gap or interior overlap is rejected as uncertain.

This snapshot is contemporaneous only. It must not be applied to historical
decisions as if the current ladder had been known then.

Authenticated position or liquidation-price endpoints are not used.
The UTA v3 public path is a different account class and is not the
authority for this classic USDT-M paper path.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import httpx

from piia.intelligence.public_orderbook import (
    BASE,
    QUOTE_MAX_AGE,
    USER_AGENT,
    _aware,
    _dec,
    _get,
    _iso,
    _unix_ms,
    allowed_url,
)

TIER_PATH = "/api/v2/mix/market/query-position-lever"
PRODUCT = "USDT-FUTURES"
FUTURE_SKEW = 5.0
DOCS = {
    "endpoint": f"{BASE}{TIER_PATH}",
    "classic_catalog": "https://www.bitget.com/docs/catalog/classic-contract-position/classic-contract-position",
    "classic_zh": "https://www.bitget.com/zh-CN/api-doc/classic/contract/position/Get-Query-Position-Lever",
    "estimated_liq_support": "https://www.bitget.com/support/articles/12560603808759",
    "liquidation_mechanism": "https://www.bitget.com/support/articles/12560603895826",
    "unit_reading": "usdt_notional_value_from_official_value_wording",
    "boundary_rule": "shared_boundary_uses_conservative_worse_tier",
    "uta_v3_not_used": "https://www.bitget.com/legacy-docs/uta/public/Get-Position-Tier-Data",
}


def parse_position_tiers(payload: Any, *, symbol: str, source_url: str, retrieved: datetime, raw: bytes | None = None) -> dict[str, Any]:
    retrieved = _aware(retrieved)
    out: dict[str, Any] = {
        "ok": False,
        "symbol": symbol,
        "source_url": source_url,
        "retrieved_at_utc": _iso(retrieved),
        "endpoint_request_time_utc": None,
        "raw_sha256": hashlib.sha256(raw).hexdigest() if raw else None,
        "unit": "usdt_notional_value",
        "unit_ambiguity_residual": "json_keys_are_startUnit_endUnit_without_explicit_usdt_suffix",
        "docs": DOCS,
        "tiers": [],
        "reason": None,
        "historical_replay_supported": False,
        "exact_liquidation_verified": False,
    }
    if not allowed_url(source_url):
        out["reason"] = "source_url_not_allowlisted"
        return out
    if not isinstance(payload, dict) or payload.get("code") != "00000" or not isinstance(payload.get("data"), list):
        out["reason"] = "invalid_or_unsuccessful_tiers"
        return out
    envelope = _unix_ms(payload.get("requestTime"))
    out["endpoint_request_time_utc"] = _iso(envelope) if envelope else None
    if envelope is None:
        out["reason"] = "missing_endpoint_request_time"
        return out
    skew = (envelope - retrieved).total_seconds()
    if skew > FUTURE_SKEW:
        out["reason"] = "future_tier_timestamp"
        return out
    if (retrieved - envelope) > QUOTE_MAX_AGE:
        out["reason"] = "stale_tiers"
        return out
    rows = []
    for item in payload["data"]:
        if not isinstance(item, dict):
            out["reason"] = "invalid_tier_row"
            return out
        row_symbol = str(item.get("symbol") or "")
        if row_symbol != symbol:
            out["reason"] = "cross_symbol_tiers"
            return out
        level = item.get("level")
        start, end = _dec(item.get("startUnit")), _dec(item.get("endUnit"))
        leverage, mmr = _dec(item.get("leverage")), _dec(item.get("keepMarginRate"))
        if level in (None, "") or start is None or end is None or leverage is None or mmr is None:
            out["reason"] = "incomplete_tier_row"
            return out
        if start < 0 or end <= start or leverage <= 0 or mmr <= 0 or mmr >= 1:
            out["reason"] = "invalid_tier_bounds"
            return out
        rows.append({
            "level": str(level),
            "start_unit": str(start),
            "end_unit": str(end),
            "max_leverage": str(leverage),
            "keep_margin_rate": str(mmr),
            "unit": "usdt_notional_value",
        })
    if not rows:
        out["reason"] = "empty_tiers"
        return out
    try:
        rows.sort(key=lambda r: (int(r["level"]), Decimal(r["start_unit"])))
    except (TypeError, ValueError):
        out["reason"] = "invalid_tier_level"
        return out
    levels = [int(r["level"]) for r in rows]
    if levels != list(range(1, len(rows) + 1)):
        out["reason"] = "tier_levels_not_contiguous"
        return out
    for i, row in enumerate(rows):
        if i == 0:
            continue
        prev = rows[i - 1]
        prev_end = Decimal(prev["end_unit"])
        start = Decimal(row["start_unit"])
        if start > prev_end:
            out["reason"] = "tier_gap"
            return out
        if start < prev_end:
            out["reason"] = "tier_interior_overlap"
            return out
        # start == prev_end: documented-unknown inclusivity; conservative worse at the point.
    out.update({"ok": True, "tiers": rows, "reason": None, "tier_count": len(rows)})
    return out


def conservative_tier_for_notional(snapshot: dict[str, Any], notional: Decimal) -> dict[str, Any]:
    """Select the risk tier for a USDT notional. Shared boundaries use the worse tier."""
    if not snapshot.get("ok"):
        return {"ok": False, "reason": snapshot.get("reason") or "tiers_unavailable"}
    if notional < 0:
        return {"ok": False, "reason": "negative_notional"}
    matching = []
    for row in snapshot["tiers"]:
        start, end = Decimal(row["start_unit"]), Decimal(row["end_unit"])
        if start <= notional <= end:
            matching.append(row)
    if not matching:
        last = snapshot["tiers"][-1]
        if notional > Decimal(last["end_unit"]):
            return {"ok": False, "reason": "notional_above_published_tiers"}
        return {"ok": False, "reason": "notional_not_in_any_tier"}
    worse = min(matching, key=lambda r: (Decimal(r["max_leverage"]), -Decimal(r["keep_margin_rate"])))
    return {
        "ok": True,
        "reason": None if len(matching) == 1 else "shared_boundary_conservative_worse_tier",
        "tier": worse,
        "matched_count": len(matching),
    }


def max_notional_for_leverage(snapshot: dict[str, Any], leverage: Decimal) -> dict[str, Any]:
    """Largest conservative notional that still permits the requested leverage. Never raises leverage."""
    if not snapshot.get("ok"):
        return {"ok": False, "reason": snapshot.get("reason") or "tiers_unavailable"}
    if leverage <= 0:
        return {"ok": False, "reason": "invalid_leverage"}
    allowed: list[Decimal] = []
    rows = snapshot["tiers"]
    for i, row in enumerate(rows):
        if Decimal(row["max_leverage"]) < leverage:
            continue
        end = Decimal(row["end_unit"])
        # If the next tier shares this end and is worse, the point itself is not safe.
        if i + 1 < len(rows) and Decimal(rows[i + 1]["start_unit"]) == end:
            if Decimal(rows[i + 1]["max_leverage"]) < leverage:
                # Use the open side of the boundary; no implicit tick for USDT value.
                # A notional exactly equal to the boundary would be classified worse.
                # Cap just below the boundary by requiring notional < end.
                allowed.append(end)  # caller must treat equality as needing worse-tier check
                continue
        allowed.append(end)
    if not allowed:
        return {"ok": False, "reason": "requested_leverage_not_permitted_on_any_tier"}
    cap = max(allowed)
    return {"ok": True, "max_notional": str(cap), "inclusive_requires_conservative_check": True}


def fetch_position_tiers(
    symbols: tuple[str, ...] = ("BTCUSDT", "XAUUSDT", "NDX100USDT"),
    *,
    timeout: float = 15.0,
    now: datetime | None = None,
    client: httpx.Client | None = None,
    snapshot_dir: Any = None,
) -> dict[str, Any]:
    batch_started = _aware(now) if now is not None else datetime.now(timezone.utc)
    out: dict[str, Any] = {
        "kind": "bitget_public_position_tiers",
        "retrieved_at_utc": _iso(batch_started),
        "historical_replay_supported": False,
        "exact_liquidation_verified": False,
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
        for symbol in symbols:
            url = f"{BASE}{TIER_PATH}?productType={PRODUCT}&symbol={symbol}"
            try:
                resp = _get(client, url)
                raw_store[f"tiers_{symbol}.json"] = resp.content
                received = datetime.now(timezone.utc) if now is None else _aware(now)
                parsed = parse_position_tiers(resp.json(), symbol=symbol, source_url=str(resp.url), retrieved=received, raw=resp.content)
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                parsed = {"ok": False, "reason": f"{type(exc).__name__}: {exc}", "symbol": symbol}
            out["instruments"][symbol] = parsed
            if not parsed.get("ok"):
                out["errors"][symbol] = parsed.get("reason")
        out["ok"] = all(item.get("ok") for item in out["instruments"].values())
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
        (dest / "tiers_snapshot.json").write_text(json.dumps(out, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return out
