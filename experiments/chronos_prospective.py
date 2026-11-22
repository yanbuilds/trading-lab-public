"""Capture one prospective, private three-contract forecast cohort on demand.

No order routes, keys, scheduler, backdated input or live-trading support.
Forecasts are uncalibrated research data and are not printed as price targets.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "data" / "piia" / "chronos_prospective.sqlite"
API = "https://api.bitget.com/api/v3/market/candles"
SYMBOLS = ("BTCUSDT", "XAUUSDT", "NDX100USDT")
WEIGHTS_HASH = "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42"
CONFIG_HASH = "ef1143bfdc9c0376d9a056eefca46cb4b1ec3d0ffacd541ff56feb40fb708031"
MODEL_REVISION = "95a9710e2596287d08352589f42634fa5abdf0a7"
APP_ID = 0x50494941


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def digest(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def parse_snapshot(payload: dict, *, symbol: str, cutoff: datetime, received: datetime) -> dict:
    if symbol not in SYMBOLS or cutoff.tzinfo is None or received.tzinfo is None:
        raise ValueError("invalid symbol or timestamp")
    cutoff = cutoff.astimezone(timezone.utc)
    received = received.astimezone(timezone.utc)
    if cutoff.minute or cutoff.second or cutoff.microsecond or not timedelta(0) <= received - cutoff <= timedelta(minutes=75):
        raise ValueError("stale or non-hour-aligned cutoff")
    if payload.get("code") != "00000" or not isinstance(payload.get("data"), list):
        raise ValueError("provider did not return candles")
    try:
        server = datetime.fromtimestamp(int(payload["requestTime"]) / 1000, timezone.utc)
    except (KeyError, ValueError, TypeError, OverflowError, OSError) as exc:
        raise ValueError("invalid provider clock") from exc
    if abs((received - server).total_seconds()) > 120 or server < cutoff:
        raise ValueError("provider and local clocks disagree")
    expected = pd.date_range(pd.Timestamp(cutoff) - pd.Timedelta(hours=512), periods=512, freq="h")
    candles: dict[pd.Timestamp, list[float]] = {}
    for row in payload["data"]:
        if not isinstance(row, list) or len(row) < 6:
            raise ValueError("malformed candle")
        try:
            raw_ts = int(row[0])
            stamp = pd.Timestamp(raw_ts, unit="ms", tz="UTC")
            o, h, l, c, volume = [float(x) for x in row[1:6]]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("invalid candle values") from exc
        if raw_ts % 3_600_000 or not all(math.isfinite(x) for x in (o, h, l, c, volume)):
            raise ValueError("invalid candle timestamp or values")
        if min(o, h, l, c) <= 0 or volume < 0 or h < max(o, l, c) or l > min(o, h, c):
            raise ValueError("invalid candle OHLCV")
        if stamp in candles:
            raise ValueError("duplicate candle")
        candles[stamp] = [o, h, l, c, volume]
    if any(t not in candles for t in expected):
        raise ValueError(f"missing closed-bar history for {symbol}")
    close = [candles[t][3] for t in expected]
    evidence = [{"open_at_utc": t.isoformat(), "close": candles[t][3]} for t in expected]
    raw = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    return {"symbol": symbol, "cutoff_at_utc": cutoff.isoformat(),
            "server_time_utc": server.isoformat(), "received_at_utc": received.isoformat(),
            "evidence_sha256": hashlib.sha256(raw).hexdigest(), "evidence": evidence, "closes": close}


def append_cohort(path: Path, records: list[dict]) -> None:
    if {r.get("symbol") for r in records} != set(SYMBOLS) or len(records) != len(SYMBOLS):
        raise ValueError("complete three-symbol cohort required")
    if len({r.get("cutoff_at_utc") for r in records}) != 1:
        raise ValueError("all symbols must share one cutoff")
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    with sqlite3.connect(path) as db:
        existing_id = db.execute("PRAGMA application_id").fetchone()[0]
        if existed and existing_id != APP_ID:
            raise ValueError("refusing to use an unrelated database")
        if not existed:
            db.execute(f"PRAGMA application_id={APP_ID}")
        db.execute("CREATE TABLE IF NOT EXISTS forecast_events ("
                   "id INTEGER PRIMARY KEY, symbol TEXT NOT NULL, cutoff TEXT NOT NULL, "
                   "created_at TEXT NOT NULL, target_at TEXT NOT NULL, model_sha256 TEXT NOT NULL, "
                   "payload_json TEXT NOT NULL, UNIQUE(symbol, cutoff, model_sha256))")
        for action in ("UPDATE", "DELETE"):
            db.execute(f"CREATE TRIGGER IF NOT EXISTS no_forecast_{action.lower()} BEFORE {action} ON forecast_events "
                       "BEGIN SELECT RAISE(ABORT, 'append-only forecast events'); END")
        for r in records:
            created = datetime.fromisoformat(r["created_at_utc"])
            cutoff = datetime.fromisoformat(r["cutoff_at_utc"])
            target = datetime.fromisoformat(r["target_at_utc"])
            if created.tzinfo is None or cutoff.tzinfo is None or target.tzinfo is None:
                raise ValueError("aware timestamps required")
            if (not timedelta(0) <= created - cutoff <= timedelta(minutes=75)
                    or abs((utc_now() - created).total_seconds()) > 120
                    or target != cutoff + timedelta(hours=24)):
                raise ValueError("forecast timing is not prospective")
            if (r.get("model_sha256") != WEIGHTS_HASH or r.get("config_sha256") != CONFIG_HASH
                    or r.get("model_revision") != MODEL_REVISION or r.get("calibration") != "not_calibrated"
                    or r.get("source") != API or r.get("order_permissions") is not False):
                raise ValueError("unexpected model or calibration label")
            received = datetime.fromisoformat(r["received_at_utc"])
            server = datetime.fromisoformat(r["server_time_utc"])
            if (received.tzinfo is None or server.tzinfo is None or not cutoff <= received <= created
                    or server < cutoff
                    or abs((received - server).total_seconds()) > 120):
                raise ValueError("invalid source timing")
            evidence = r.get("evidence")
            if not isinstance(evidence, list) or len(evidence) != 512:
                raise ValueError("complete input evidence required")
            expected = pd.date_range(pd.Timestamp(cutoff) - pd.Timedelta(hours=512), periods=512, freq="h")
            for item, timestamp in zip(evidence, expected, strict=True):
                value = float(item["close"])
                if pd.Timestamp(item["open_at_utc"]) != timestamp or not math.isfinite(value) or value <= 0:
                    raise ValueError("invalid frozen input evidence")
            raw = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
            if hashlib.sha256(raw).hexdigest() != r.get("evidence_sha256"):
                raise ValueError("frozen input evidence hash mismatch")
            forecast = r.get("forecast_research_only") or {}
            q = [float(forecast[k]) for k in ("q10", "median", "q90")]
            if not all(math.isfinite(x) and x > 0 for x in q) or not q[0] <= q[1] <= q[2]:
                raise ValueError("invalid research forecast")
            db.execute("INSERT INTO forecast_events(symbol, cutoff, created_at, target_at, model_sha256, payload_json) "
                       "VALUES (?, ?, ?, ?, ?, ?)", (r["symbol"], r["cutoff_at_utc"], r["created_at_utc"],
                       r["target_at_utc"], r["model_sha256"], json.dumps(r, sort_keys=True, allow_nan=False)))


def main() -> None:
    import torch
    from chronos import BaseChronosPipeline

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()
    weights, config = args.model / "model.safetensors", args.model / "config.json"
    if digest(weights) != WEIGHTS_HASH or digest(config) != CONFIG_HASH:
        raise ValueError("model files do not match pinned hashes")
    script_hash = digest(Path(__file__))
    started = utc_now()
    cutoff = started.replace(minute=0, second=0, microsecond=0)
    snapshots = []
    with httpx.Client(timeout=20., follow_redirects=False) as client:
        for symbol in SYMBOLS:
            response = client.get(API, params={"category": "USDT-FUTURES", "symbol": symbol,
                                              "interval": "1H", "type": "market", "limit": "600"})
            response.raise_for_status()
            snapshots.append(parse_snapshot(response.json(), symbol=symbol, cutoff=cutoff, received=utc_now()))
    torch.set_num_threads(4)
    pipeline = BaseChronosPipeline.from_pretrained(str(args.model.resolve()), device_map="cpu", local_files_only=True)
    quantiles, _ = pipeline.predict_quantiles([np.array(s["closes"], dtype=np.float32) for s in snapshots],
                                              prediction_length=24, quantile_levels=[.1, .5, .9], batch_size=3)
    created = utc_now()
    if digest(weights) != WEIGHTS_HASH or digest(config) != CONFIG_HASH or digest(Path(__file__)) != script_hash:
        raise ValueError("model or script changed during inference")
    rows = []
    for snapshot, tensor in zip(snapshots, quantiles, strict=True):
        result = tensor.detach().cpu().numpy().squeeze()
        if result.shape != (24, 3) or not np.isfinite(result).all() or not (result > 0).all():
            raise ValueError("invalid forecast output")
        if not ((result[:, 0] <= result[:, 1]).all() and (result[:, 1] <= result[:, 2]).all()):
            raise ValueError("unordered forecast quantiles")
        rows.append({k: v for k, v in snapshot.items() if k != "closes"} | {
            "created_at_utc": created.isoformat(), "target_at_utc": (cutoff + timedelta(hours=24)).isoformat(),
            "model_sha256": WEIGHTS_HASH, "config_sha256": CONFIG_HASH, "model_revision": MODEL_REVISION,
            "script_sha256": script_hash, "horizon_hours": 24, "source": API,
            "calibration": "not_calibrated", "order_permissions": False,
            "forecast_research_only": {"q10": float(result[-1, 0]), "median": float(result[-1, 1]), "q90": float(result[-1, 2])}})
    append_cohort(DEFAULT_LEDGER, rows)
    print(json.dumps({"status": "captured_prospectively", "symbols": list(SYMBOLS),
                      "cutoff_at_utc": cutoff.isoformat(), "created_at_utc": created.isoformat(),
                      "target_at_utc": (cutoff + timedelta(hours=24)).isoformat(),
                      "ledger": str(DEFAULT_LEDGER), "calibration": "not_calibrated",
                      "orders": False, "note": "Raw forecasts are private research records, not displayed as trade targets."}, indent=2))


if __name__ == "__main__":
    main()
