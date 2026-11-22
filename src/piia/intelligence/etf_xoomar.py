"""Partial third-party BTC ETF holdings-derived estimates, never aggregate flows."""
from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
from pathlib import Path

import httpx

URL = "https://xoomar.com/api/markets/etf-flows"
DOCS = "https://xoomar.com/markets/api/etf-flows"
FUNDS = {"IBIT", "BITB", "ARKB"}


def fetch_partial_etf(*, client=None, now=None, snapshot_dir=None):
    """Latest per-fund observations only; no total and no trading authorization."""
    fixed = now is not None
    started = now or datetime.now(timezone.utc)
    if started.tzinfo is None or started.utcoffset() is None:
        raise ValueError("aware time required")
    started = started.astimezone(timezone.utc)
    out = {"kind": "xoomar_partial_btc_etf_holdings_derived_estimates", "source_url": URL,
           "source_docs": DOCS, "attribution": "XOOMAR, https://xoomar.com/markets/api/etf-flows",
           "retrieved_at_utc": started.isoformat(), "available": False, "usable_for_btc_research": False,
           "complete": False, "historical_replay_supported": False, "order_permissions": False,
           "note": "Only IBIT, BITB and ARKB; third-party USD estimates from holdings changes, not reported issuer fund flows or market totals."}
    own = client is None
    client = client or httpx.Client(timeout=12, follow_redirects=False)
    raw = None
    try:
        response = client.get(URL, params={"asset": "btc", "days": 10})
        response.raise_for_status()
        received = started if fixed else datetime.now(timezone.utc)
        raw = response.content
        if len(raw) > 1_000_000:
            raw = None
            raise ValueError("response too large")
        payload = response.json()
        updated = datetime.fromisoformat(payload["updatedAt"].replace("Z", "+00:00"))
        if updated.tzinfo is None or updated.utcoffset() is None or updated > received:
            raise ValueError("invalid provider update time")
        rows = payload["data"]
        if not isinstance(rows, list) or len(rows) > 1000:
            raise ValueError("invalid rows")
        seen, latest = set(), {}
        for row in rows:
            ticker = row["ticker"]
            if ticker not in FUNDS or row["asset"] != "btc":
                raise ValueError("unexpected fund or asset")
            day = date.fromisoformat(row["date"])
            if day.isoformat() != row["date"] or day > started.date() or (ticker, day) in seen:
                raise ValueError("future, invalid or duplicate row")
            seen.add((ticker, day))
            flow = row.get("flowUsd")
            if flow is not None:
                if isinstance(flow, bool) or not isinstance(flow, str):
                    raise ValueError("invalid estimate")
                amount = Decimal(flow)
                if not amount.is_finite():
                    raise ValueError("nonfinite estimate")
            if ticker not in latest or day.isoformat() > latest[ticker]["date"]:
                latest[ticker] = {"date": day.isoformat(), "issuer": str(row["issuer"]),
                                  "estimated_flow_usd": flow}
        out.update(available=bool(latest), latest_by_fund=latest, covered_funds=sorted(latest),
                   missing_expected_funds=sorted(FUNDS - set(latest)), provider_updated_at_utc=updated.isoformat(),
                   raw_sha256=hashlib.sha256(raw).hexdigest())
    except (httpx.HTTPError, OSError, ValueError, KeyError, TypeError, OverflowError, InvalidOperation) as exc:
        out["error"] = type(exc).__name__
    finally:
        if own:
            client.close()
    out["completed_at_utc"] = (started if fixed else datetime.now(timezone.utc)).isoformat()
    if snapshot_dir is not None and raw is not None:
        directory = Path(snapshot_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "xoomar_partial.json").write_bytes(raw)
        out["raw_snapshot_file"] = "xoomar_partial.json"
    return out


def verify_partial_etf(context: dict, snapshot_dir: Path) -> None:
    """Reparse the saved bytes locally; this is not independent source authentication."""
    if (context.get("raw_snapshot_file") != "xoomar_partial.json" or context.get("available") is not True
            or context.get("usable_for_btc_research") is not False or context.get("complete") is not False
            or context.get("order_permissions") is not False):
        raise ValueError("invalid partial ETF context")
    path = Path(snapshot_dir) / "xoomar_partial.json"
    if path.stat().st_size > 1_000_000:
        raise ValueError("saved ETF response too large")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != context.get("raw_sha256"):
        raise ValueError("partial ETF snapshot hash mismatch")
    completed = datetime.fromisoformat(context["completed_at_utc"].replace("Z", "+00:00"))
    transport = httpx.MockTransport(lambda req: httpx.Response(200, content=raw, request=req))
    with httpx.Client(transport=transport) as client:
        replay = fetch_partial_etf(client=client, now=completed)
    fields = ("available", "latest_by_fund", "covered_funds", "missing_expected_funds", "provider_updated_at_utc", "raw_sha256")
    if any(replay.get(field) != context.get(field) for field in fields):
        raise ValueError("partial ETF snapshot does not reproduce context")
