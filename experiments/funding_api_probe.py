"""Inspect public Bitget UTA funding-history page availability; no credentials.

This is an availability probe, not a funding dataset, trading engine, or proof
that all settlement records or historical funding schedules are complete.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import httpx

URL = "https://api.bitget.com/api/v3/market/history-fund-rate"
SYMBOLS = ("BTCUSDT", "XAUUSDT", "NDX100USDT")


def inspect_page(payload: dict, symbol: str) -> dict:
    if payload.get("code") != "00000" or not isinstance(payload.get("data"), dict):
        raise ValueError("provider returned an unsuccessful or malformed response")
    rows = payload["data"].get("resultList")
    if not isinstance(rows, list):
        raise ValueError("missing funding result list")
    timestamps = []
    for row in rows:
        if row.get("symbol") != symbol:
            raise ValueError("wrong symbol in funding response")
        raw_time, raw_rate = row.get("fundingRateTimestamp"), row.get("fundingRate")
        if isinstance(raw_time, bool) or isinstance(raw_rate, bool):
            raise ValueError("invalid funding record")
        try:
            if not str(raw_time).isdigit() or not math.isfinite(float(raw_rate)):
                raise ValueError("invalid funding record")
            timestamp = datetime.fromtimestamp(int(raw_time) / 1000, timezone.utc)
        except (TypeError, ValueError, OverflowError, OSError) as exc:
            raise ValueError("invalid funding record") from exc
        timestamps.append(timestamp)
    if len(set(timestamps)) != len(timestamps):
        raise ValueError("duplicate funding timestamps")
    return {"record_count": len(timestamps),
            "oldest_utc": min(timestamps).isoformat() if timestamps else None,
            "newest_utc": max(timestamps).isoformat() if timestamps else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("artifacts/funding_api_probe.json"))
    args = ap.parse_args()
    out = {"source": URL, "product_type": "USDT-FUTURES", "requested_pages": [1, 3, 5],
           "page_size": 100, "started_at_utc": datetime.now(timezone.utc).isoformat(),
           "label": "availability sample only; does not establish continuous coverage or historical schedules",
           "symbols": {}}
    with httpx.Client(timeout=12, follow_redirects=False) as client:
        for symbol in SYMBOLS:
            pages = {}
            for page in out["requested_pages"]:
                try:
                    response = client.get(URL, params={"category": "USDT-FUTURES", "symbol": symbol,
                                                        "limit": "100", "cursor": str(page)})
                    response.raise_for_status()
                    pages[str(page)] = inspect_page(response.json(), symbol)
                except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
                    pages[str(page)] = {"error": type(exc).__name__}
            out["symbols"][symbol] = pages
    out["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
