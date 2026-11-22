"""Audit local hourly contract data availability; never backtest or download data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading_lab.perps.compare import funding_complete_between

PAIRS = {"BTCUSDT": "BTC/USDT:USDT", "XAUUSDT": "XAU/USDT:USDT", "NDX100USDT": "NDX100/USDT:USDT"}
NDX_LISTING_DAY_EARLIEST_UTC = pd.Timestamp("2026-05-21T16:00:00Z")
NDX_LISTING_SOURCE = "https://www.bitget.com/support/articles/12560603884305"


def audit(directory: Path, start: str, end: str, *, warmup_hours: int = 300,
          btc_bars: Path | None = None) -> dict:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if (start.tzinfo is None or end.tzinfo is None or start >= end or start != start.floor("h")
            or end != end.floor("h") or type(warmup_hours) is not int or warmup_hours < 0):
        raise ValueError("ordered timezone-aware hourly window and nonnegative warmup required")
    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    directory = Path(directory)
    provenance_path = directory / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    report = {"start_inclusive_utc": start.isoformat(), "end_exclusive_utc": end.isoformat(),
              "warmup_hours": warmup_hours, "symbols": {}, "order_permissions": False,
              "provenance_sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
              "limitations": ["File coverage does not prove provider truth, point-in-time availability or held-out status.",
                              "Funding cadence is assumed from recorded metadata, not verified historical settlement.",
                              "Historical contract specifications, liquidity and liquidation are not verified.",
                              "Model, news, macro and prediction-market point-in-time coverage is not assessed."]}
    expected = pd.date_range(start - pd.Timedelta(hours=warmup_hours), end, freq="h", inclusive="left")
    for symbol, pair in PAIRS.items():
        output = {"bars_complete_with_warmup": False, "funding_complete_under_recorded_interval_assumption": False}
        report["symbols"][symbol] = output
        if symbol == "NDX100USDT":
            output["listing_announcement"] = {"local_listing_day": "2026-05-22", "local_timezone": "UTC+08:00",
                "earliest_possible_utc": NDX_LISTING_DAY_EARLIEST_UTC.isoformat(), "exact_listing_time_verified": False,
                "source_url": NDX_LISTING_SOURCE,
                "requested_start_definitely_before_listing_day": start < NDX_LISTING_DAY_EARLIEST_UTC}
        try:
            path = Path(btc_bars) if symbol == "BTCUSDT" and btc_bars is not None else directory / f"{symbol}.csv"
            output["bars_file_name"] = path.name
            output["using_cache_default_filename"] = not (symbol == "BTCUSDT" and btc_bars is not None)
            if symbol == "BTCUSDT" and btc_bars is not None:
                output["bars_provenance_note"] = "Alternate file: cache metadata does not establish this file's provenance."
            output["bars_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            frame = pd.read_csv(path)
            required = {"date", "open", "high", "low", "close", "volume"}
            if not required.issubset(frame.columns):
                raise ValueError("required candle columns missing")
            raw_dates = frame["date"].astype(str)
            if not raw_dates.str.contains(r"(?:Z|[+-]\d\d:\d\d)$", regex=True).all():
                raise ValueError("timestamps must include timezone")
            stamps = pd.DatetimeIndex(pd.to_datetime(raw_dates, utc=True, errors="raise"))
            values = frame[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="raise")
            o, h, l, c, volume = (values[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "volume"))
            valid = (np.isfinite(values.to_numpy(dtype=float)).all(axis=1) & (o > 0) & (h > 0) & (l > 0) & (c > 0)
                     & (volume >= 0) & (h >= np.maximum.reduce([o, l, c])) & (l <= np.minimum.reduce([o, h, c])))
            if (not valid.all() or stamps.has_duplicates or not stamps.is_monotonic_increasing
                    or not (stamps == stamps.floor("h")).all()):
                raise ValueError("invalid, duplicate, unsorted or non-hourly candles")
            missing = expected.difference(stamps)
            output.update(bars_first_utc=stamps.min().isoformat() if len(stamps) else None,
                          bars_last_utc=stamps.max().isoformat() if len(stamps) else None,
                          missing_bars_with_warmup=int(len(missing)),
                          first_missing_bar_utc=missing[0].isoformat() if len(missing) else None,
                          bars_complete_with_warmup=len(missing) == 0)
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
            output["bars_error"] = type(exc).__name__ + ": " + str(exc)
        try:
            path = directory / f"{symbol}-funding.csv"
            output["funding_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows = pd.read_csv(path).to_dict("records")
            interval = provenance["symbols"][pair]["funding_interval_hours"]
            coverage = funding_complete_between({pair: rows}, {pair: interval}, start, end - pd.Timedelta(hours=1))
            output["recorded_funding_interval_hours"] = interval
            output["funding_coverage"] = coverage
            output["funding_complete_under_recorded_interval_assumption"] = coverage["complete"]
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
            output["funding_error"] = type(exc).__name__ + ": " + str(exc)
    report["all_symbols_locally_complete_under_assumptions"] = all(
        x["bars_complete_with_warmup"] and x["funding_complete_under_recorded_interval_assumption"]
        and not x.get("listing_announcement", {}).get("requested_start_definitely_before_listing_day", False)
        for x in report["symbols"].values())
    report["verified_net_performance_available"] = False
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--warmup-hours", type=int, default=300)
    parser.add_argument("--btc-bars", type=Path, help="optional separate BTC CSV; provenance is not established by cache metadata")
    args = parser.parse_args()
    print(json.dumps(audit(args.data_dir, args.start, args.end, warmup_hours=args.warmup_hours,
                           btc_bars=args.btc_bars), indent=2))


if __name__ == "__main__":
    main()
