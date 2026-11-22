"""Retrospective direction diagnostic for saved 24h forecast experiments.

This is not a trade simulation, probability calibration, or investment signal.
The 24h momentum reference is fixed; no parameters are selected from results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def evaluate(report: dict, bars: pd.DataFrame) -> dict:
    if report.get("sample_schedule") != "fixed_mondays_2026_06_15_to_2026_11_09":
        raise ValueError("only the fixed common-window forecast report is supported")
    rows = report.get("forecasts") or []
    expected = pd.date_range("2026-06-15", "2026-11-09", freq="7D", tz="UTC")
    if len(rows) != len(expected) or report.get("samples") != len(expected):
        raise ValueError("incomplete forecast schedule")
    idx = pd.DatetimeIndex(bars.index)
    if (idx.tz is None or idx.has_duplicates or not idx.is_monotonic_increasing
            or not (idx.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()):
        raise ValueError("bar timeline must be aware, hourly, ordered and continuous")
    close = pd.to_numeric(bars["close"], errors="raise")
    if not np.isfinite(close).all() or not (close > 0).all():
        raise ValueError("invalid closing prices")
    checks = {"model": {"correct": 0, "incorrect": 0, "neutral": 0},
              "prior_24h_momentum": {"correct": 0, "incorrect": 0, "neutral": 0}}
    flat = 0
    for row, decision in zip(rows, expected, strict=True):
        if pd.Timestamp(row["decision_utc"]) != decision:
            raise ValueError("forecast timestamps do not match the fixed schedule")
        try:
            current = float(close.loc[decision - pd.Timedelta(hours=1)])
            prior = float(close.loc[decision - pd.Timedelta(hours=25)])
            actual = float(close.loc[decision + pd.Timedelta(hours=23)])
            forecast = float(row["median"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("missing input, target or model forecast") from exc
        if not math.isfinite(forecast):
            raise ValueError("invalid model forecast")
        # Chronos used float32 for its inputs; compare to the exact representation.
        if float(np.float32(current)) != float(row["last_close"]) or actual != float(row["actual_24h"]):
            raise ValueError("forecast report does not match source bars")
        observed_sign = int(np.sign(actual - current))
        if observed_sign == 0:
            flat += 1
            continue
        for name, predicted_sign in (("model", int(np.sign(forecast - float(row["last_close"])))),
                                     ("prior_24h_momentum", int(np.sign(current - prior)))):
            field = "neutral" if predicted_sign == 0 else "correct" if predicted_sign == observed_sign else "incorrect"
            checks[name][field] += 1
    return {"symbol_label": report["symbol_label"], "samples": len(rows), "flat_outcomes_excluded": flat,
            "counts": checks, "forecast_horizon_hours": 24,
            "status": "retrospective_exploratory_not_calibrated_not_a_trading_backtest",
            "limitations": ["same previously viewed historical data", "model training overlap unknown",
                            "neutral forecasts are separate from correct and incorrect counts",
                            "no fees, funding, slippage, stops, fills or liquidation", "not a live signal"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--bars", type=Path, required=True)
    args = ap.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if hashlib.sha256(args.bars.read_bytes()).hexdigest() != report.get("input_sha256"):
        raise ValueError("bar file hash differs from forecast input")
    bars = pd.read_csv(args.bars, parse_dates=["date"]).set_index("date")
    bars.index = pd.to_datetime(bars.index, utc=True)
    print(json.dumps(evaluate(report, bars), indent=2))


if __name__ == "__main__":
    main()
