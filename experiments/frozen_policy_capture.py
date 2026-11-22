"""On-demand prospective observation of frozen policy v1; never places orders.

No timer, account access or retrospective import. Hourly closes and model state
are saved privately; the next-open backtest fill has already passed by capture.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import httpx
import pandas as pd

from trading_lab.perps.mean_reversion import attach_range_bollinger
from trading_lab.perps.regime import load_manifest
from trading_lab.perps.research import validate_bars

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data/piia/frozen_policy_v1.sqlite"
API = "https://api.bitget.com/api/v3/market/candles"
CONTRACTS_API = "https://api.bitget.com/api/v2/mix/market/contracts?productType=USDT-FUTURES"
SYMBOLS = ("BTCUSDT", "XAUUSDT", "NDX100USDT")
START = datetime(2026, 11, 18, tzinfo=timezone.utc)
APP_ID = 0x46505631
PINNED = {
    "FROZEN_POLICY_V1.md": "4aed3fa2c2fe6f968582e383b265cac1a46f65f11cc809b9919480b8d5d614bd",
    "config/piia_regime_mean_reversion_v1.yaml": "acca5bd94faa9f07ce826d4f3d01e2eee50808c103db4494908ef11b351fe148",
    "src/trading_lab/perps/regime.py": "c67520a557c50402275f46b40a9f53182b659328c9b5b1c4732eeebbde8c85ed",
    "src/trading_lab/perps/mean_reversion.py": "f533b5a6b51d249771174881c25c61981560e2cfd97f93842ae598051b23b859",
    "src/trading_lab/perps/compare.py": "5533868bc9cb822be1ceea18c597e6489def72b1b3c7cc0fa33bbdfe83d83329",
    "src/trading_lab/perps/risk.py": "c49e1fbedd92d2ae42d1624c9f2fcf88ef0bd7be31860c1c5218c42280c8a6bd",
}


def now_utc():
    return datetime.now(timezone.utc)


def verify_policy():
    for name, expected in PINNED.items():
        raw = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"frozen policy source changed: {name}")
    load_manifest(ROOT / "config/piia_regime_mean_reversion_v1.yaml")


def parse_closed(payload: dict, cutoff: datetime, received: datetime) -> pd.DataFrame:
    if cutoff.tzinfo is None or received.tzinfo is None or cutoff < START or cutoff != cutoff.replace(minute=0, second=0, microsecond=0):
        raise ValueError("invalid or pre-policy cutoff")
    if not timedelta(0) <= received - cutoff <= timedelta(minutes=50):
        raise ValueError("capture not current")
    if payload.get("code") != "00000" or not isinstance(payload.get("data"), list):
        raise ValueError("provider response unavailable")
    server = datetime.fromtimestamp(int(payload["requestTime"]) / 1000, timezone.utc)
    if abs((received - server).total_seconds()) > 120 or server < cutoff:
        raise ValueError("provider clock disagreement")
    bars = {}
    for row in payload["data"]:
        if not isinstance(row, list) or len(row) < 6:
            raise ValueError("malformed candle")
        stamp_ms = int(row[0])
        stamp = datetime.fromtimestamp(stamp_ms / 1000, timezone.utc)
        values = [float(x) for x in row[1:6]]
        if (stamp_ms % 3_600_000 or stamp > cutoff or stamp in bars
                or not all(math.isfinite(v) for v in values)):
            raise ValueError("invalid or duplicate candle")
        o, high, low, close, volume = values
        if min(o, high, low, close) <= 0 or volume < 0 or high < max(o, low, close) or low > min(o, high, close):
            raise ValueError("invalid candle OHLCV")
        bars[stamp] = values
    expected = [cutoff - timedelta(hours=i) for i in range(300, 0, -1)]
    if any(stamp not in bars for stamp in expected):
        raise ValueError("missing closed hourly candle")
    frame = pd.DataFrame([bars[t] for t in expected], index=pd.DatetimeIndex(expected),
                         columns=["open", "high", "low", "close", "volume"])
    validate_bars(frame)
    return frame


def parse_current_specs(payload: dict, cutoff: datetime, received: datetime, *, raw: bytes | None = None) -> dict:
    """Check public contract specifications as actually received for this cohort."""
    from piia.intelligence.public_orderbook import parse_contracts
    if not timedelta(0) <= received - cutoff <= timedelta(minutes=50):
        raise ValueError("contract specification snapshot not current")
    server = datetime.fromtimestamp(int(payload["requestTime"]) / 1000, timezone.utc)
    if server < cutoff or abs((received - server).total_seconds()) > 120:
        raise ValueError("contract specification provider clock disagreement")
    parsed = parse_contracts(payload, SYMBOLS, source_url=CONTRACTS_API, retrieved=received, raw=raw)
    if not parsed.get("ok"):
        raise ValueError("contract specifications unavailable")
    for symbol, item in parsed["contracts"].items():
        values = [float(item[k]) for k in ("min_qty", "qty_step", "min_notional", "taker", "contract_size")]
        if (item.get("symbol") != symbol or item.get("status") != "normal" or item.get("instrument_type") != "perpetual"
                or not all(math.isfinite(x) for x in values) or min(values[0], values[1], values[4]) <= 0
                or values[2] < 0 or not 0 <= values[3] < 1):
            raise ValueError("invalid or unavailable contract specification")
    return parsed


def save_observation(path: Path, cutoff: datetime, created: datetime, payload: dict):
    if cutoff < START or not timedelta(0) <= created - cutoff <= timedelta(minutes=50) or abs((now_utc() - created).total_seconds()) > 120:
        raise ValueError("observation must be current and after policy start")
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    with sqlite3.connect(path) as db:
        app = db.execute("PRAGMA application_id").fetchone()[0]
        if existed and app != APP_ID:
            raise ValueError("unrelated database")
        if not existed:
            db.execute(f"PRAGMA application_id={APP_ID}")
        db.execute("CREATE TABLE IF NOT EXISTS observations (id INTEGER PRIMARY KEY, cutoff TEXT NOT NULL UNIQUE, "
                   "created_at TEXT NOT NULL, payload_json TEXT NOT NULL)")
        for action in ("UPDATE", "DELETE"):
            db.execute(f"CREATE TRIGGER IF NOT EXISTS no_observation_{action.lower()} BEFORE {action} ON observations "
                       "BEGIN SELECT RAISE(ABORT, 'append-only observations'); END")
        db.execute("INSERT INTO observations(cutoff, created_at, payload_json) VALUES (?, ?, ?)",
                   (cutoff.isoformat(), created.isoformat(), json.dumps(payload, sort_keys=True, allow_nan=False)))


def main():
    verify_policy()
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    started = now_utc()
    cutoff = started.replace(minute=0, second=0, microsecond=0)
    if cutoff < START or started - cutoff > timedelta(minutes=50):
        raise ValueError("policy capture is not yet eligible or this hour is too old")
    evidence, states = {}, {}
    with httpx.Client(timeout=20., follow_redirects=False) as client:
        for symbol in SYMBOLS:
            response = client.get(API, params={"category": "USDT-FUTURES", "symbol": symbol,
                                              "interval": "1H", "type": "market", "limit": "310"})
            response.raise_for_status()
            received, raw = now_utc(), response.json()
            frame = parse_closed(raw, cutoff, received)
            prepared = attach_range_bollinger(frame)
            last = prepared.iloc[-1]
            evidence[symbol] = {"received_at_utc": received.isoformat(), "provider_time_ms": raw["requestTime"],
                                "bars": [{"open_at_utc": t.isoformat(), **row} for t, row in
                                         zip(frame.index, frame.to_dict("records"), strict=True)]}
            states[symbol] = {"regime": str(last["regime"]), "candidate_signal": int(last["signal"]),
                              "kind": "research_observation_not_order"}
        response = client.get(CONTRACTS_API)
        response.raise_for_status()
        specs_received = now_utc()
        specs_snapshot = parse_current_specs(response.json(), cutoff, specs_received, raw=response.content)
    created = now_utc()
    verify_policy()
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != script_sha:
        raise ValueError("capture script changed during run")
    payload = {"cutoff_utc": cutoff.isoformat(), "created_at_utc": created.isoformat(), "policy_commit": "69fd853",
               "capture_script_sha256": script_sha, "source": API, "symbols": list(SYMBOLS),
               "states": states, "evidence": evidence, "specs_snapshot": specs_snapshot, "order_permissions": False,
               "next_open_fill_available": False, "funding": "unknown", "status": "local_unverified_research_observation"}
    save_observation(LEDGER, cutoff, created, payload)
    print(json.dumps({"status": "captured_research_only", "cutoff_utc": cutoff.isoformat(),
                      "created_at_utc": created.isoformat(), "ledger": str(LEDGER),
                      "order_permissions": False, "next_open_fill_available": False}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OverflowError, OSError, sqlite3.Error, httpx.HTTPError) as exc:
        raise SystemExit(f"Capture refused: {type(exc).__name__}: {exc}") from exc
