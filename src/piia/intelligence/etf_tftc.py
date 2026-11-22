"""Fallback current-session compiler; not a replacement for historical point-in-time data."""
import math
import hashlib
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import httpx
from piia.intelligence.etf_flows import EXPECTED_FUNDS, COMPLETE_MAX_AGE_DAYS

URL = "https://www.tftc.io/bitcoin-etf-flows/latest.json"


def fetch_tftc_etf(*, client=None, now=None, snapshot_dir=None):
    fixed_time = now is not None
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("timezone-aware time required")
    now = now.astimezone(timezone.utc)
    out = {"kind": "tftc_us_spot_btc_etf_net_fund_flows", "source_url": URL,
           "attribution": "The Commoner Bitcoin ETF Tracker by TFTC, https://www.tftc.io/bitcoin-etf-flows",
           "license": "https://creativecommons.org/licenses/by/4.0/", "retrieved_at_utc": now.isoformat(),
           "units": "millions_of_usd", "usable_for_btc_research": False, "complete": False,
           "partial": True, "stale": True, "historical_replay_supported": False,
           "note": "Third-party compiler; row completeness does not establish latest exchange session, release timing or issuer data accuracy.",
           "latest_exchange_session_verified": False}
    own = client is None
    client = client or httpx.Client(timeout=15, follow_redirects=False)
    raw = None
    try:
        response = client.get(URL)
        response.raise_for_status()
        raw = response.content
        if len(raw) > 2_000_000:
            raw = None
            raise ValueError("response too large")
        out["raw_sha256"] = hashlib.sha256(raw).hexdigest()
        data = response.json()
        row = data["latestSession"]
        asof = datetime.fromisoformat(row["date"]).date()
        if data["updatedThrough"] != row["date"] or not 0 <= (now.date()-asof).days <= COMPLETE_MAX_AGE_DAYS:
            raise ValueError("future, stale, or inconsistent session")
        modified = datetime.fromisoformat(data["dateModified"].replace("Z", "+00:00"))
        nominal_close = datetime.combine(asof, time(16), tzinfo=ZoneInfo("America/New_York"))
        if modified.tzinfo is None or modified.utcoffset() is None or modified > now or modified < nominal_close or now < nominal_close:
            raise ValueError("invalid publication timestamp")
        funds = row["perEtfUsd"]
        if not isinstance(funds, dict) or not set(EXPECTED_FUNDS).issubset(funds):
            raise ValueError("missing expected fund")
        def number(value):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("missing or invalid flow")
            return float(value)
        amounts = {k: number(v) for k, v in funds.items()}
        total = number(row["netFlowUsd"])
        if abs(sum(amounts.values()) - total) > 1:
            raise ValueError("fund sum does not reconcile")
        latest = {"flow_date": row["date"], "fund_flows_millions_usd": {k: v/1e6 for k, v in amounts.items()},
                  "table_total_millions_usd": total/1e6, "complete": True, "partial": False}
        out.update(complete=True, partial=False, stale=False, usable_for_btc_research=True,
                   asof_complete_flow_date=row["date"], latest_complete_flow_date=row["date"],
                   latest_complete_row=latest, sources=data.get("sources"), date_modified=data["dateModified"],
                   latest_row_is_partial=None, latest_complete_age_calendar_days=(now.date() - asof).days,
                   session_close_note="16:00 New York is a conservative cutoff, not a verified ETF publication time; early closes not assumed.")
    except (httpx.HTTPError, ValueError, KeyError, TypeError, OverflowError) as exc:
        out["errors"] = {"fetch": type(exc).__name__}
    finally:
        if own:
            client.close()
    out["completed_at_utc"] = (now if fixed_time else datetime.now(timezone.utc)).isoformat()
    if snapshot_dir is not None and raw is not None:
        directory = Path(snapshot_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tftc_latest.json").write_bytes(raw)
        out["raw_snapshot_file"] = "tftc_latest.json"
    return out
