"""Audit a pre-settlement ticker quote against an optional official funding record.

This is a third-party proxy, never an official settlement and never usable as
complete net performance by itself. Missing/stale/malformed files fail closed.
"""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path

import pandas as pd
import requests
import zstandard

ENDPOINT = "https://api.cryptohftdata.com/download"
MAX_FILE_BYTES = 20_000_000
MAX_DECOMPRESSED_BYTES = 100_000_000
MAX_LAG_MS = 1000.0


def last_pre_settlement_quote(frame: pd.DataFrame, settlement: pd.Timestamp) -> dict:
    t = pd.Timestamp(settlement)
    if t.tzinfo is None or t != t.floor("h"):
        raise ValueError("timezone-aware hour-aligned settlement required")
    t = t.tz_convert("UTC")
    t_ms, t_ns = int(t.timestamp() * 1000), t.value
    required = {"received_time", "funding_rate", "next_funding_time"}
    if not required.issubset(frame.columns):
        raise ValueError("archive columns missing")
    f = frame.copy()
    for col in required:
        f[col] = pd.to_numeric(f[col], errors="coerce")
    f = f[(f.received_time < t_ns) & (f.next_funding_time == t_ms) & f.funding_rate.notna()]
    if f.empty:
        raise ValueError("no quote received before this settlement")
    last = f.sort_values("received_time").iloc[-1]
    lag_ms = (t_ns - int(last.received_time)) / 1_000_000
    rate = float(last.funding_rate)
    if not math.isfinite(rate) or not 0 < lag_ms <= MAX_LAG_MS:
        raise ValueError("last quote is invalid or too old")
    return {"settlement_utc": t.isoformat(), "proxy_rate": rate, "lag_ms": lag_ms,
            "provenance": "third_party_pre_settlement_quote_not_official_settlement"}


def decode_archive(data: bytes) -> pd.DataFrame:
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("archive file exceeds size limit")
    if data[:4] == b"\x28\xb5\x2f\xfd":
        with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data)) as reader:
            data = reader.read(MAX_DECOMPRESSED_BYTES + 1)
        if len(data) > MAX_DECOMPRESSED_BYTES:
            raise ValueError("decompressed archive exceeds size limit")
    if data[:4] != b"PAR1" or data[-4:] != b"PAR1":
        raise ValueError("archive is not complete parquet")
    return pd.read_parquet(io.BytesIO(data), columns=["received_time", "funding_rate", "next_funding_time"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settlement", required=True, help="hour-aligned UTC settlement timestamp")
    ap.add_argument("--symbol", choices=["BTCUSDT", "XAUUSDT", "NDX100USDT"], default="BTCUSDT")
    ap.add_argument("--official-funding", type=Path, help="optional local Bitget timestamp/rate CSV for cross-check")
    ap.add_argument("--out", type=Path, default=Path("artifacts/funding_archive_audit.json"))
    args = ap.parse_args()
    t = pd.Timestamp(args.settlement)
    if t.tzinfo is None or t != t.floor("h"):
        ap.error("settlement must be timezone-aware and hour-aligned")
    t = t.tz_convert("UTC")
    hour = t - pd.Timedelta(hours=1)
    key = f"bitget_futures/{hour:%Y-%m-%d}/{hour:%H}/{args.symbol}_mark_price.parquet"
    response = requests.get(ENDPOINT, params={"file": key}, timeout=30)
    response.raise_for_status()
    report = last_pre_settlement_quote(decode_archive(response.content), t)
    report.update(symbol=args.symbol, archive_file=key, source=ENDPOINT,
                  proxy_only=True, usable_as_verified_net_performance=False)
    if args.official_funding:
        official = pd.read_csv(args.official_funding)
        rows = official[official.timestamp == int(t.timestamp() * 1000)]
        if len(rows) != 1:
            raise ValueError("expected exactly one official record")
        official_rate = float(rows.iloc[0].rate)
        if not math.isfinite(official_rate):
            raise ValueError("invalid official funding rate")
        report["official_rate"] = official_rate
        report["matches_official"] = math.isclose(report["proxy_rate"], official_rate, rel_tol=0, abs_tol=1e-12)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
