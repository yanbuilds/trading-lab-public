"""Fixed exploratory BTC 24-hour forecast probe; never produces orders.

Run in a separate Chronos environment. The checkpoints and 2026 observations
were selected after the fact, so this is not a prospective or trading test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline

EXPECTED_SHA256 = "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42"
REVISION = "95a9710e2596287d08352589f42634fa5abdf0a7"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bars", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/chronos_probe.json"))
    parser.add_argument("--symbol", choices=["BTCUSDT", "XAUUSDT", "NDX100USDT"], default="BTCUSDT")
    schedule = parser.add_mutually_exclusive_group()
    schedule.add_argument("--weekly-fixed", action="store_true",
                          help="fixed Monday 2026-01-05 through 2026-11-09, exploratory not held out")
    schedule.add_argument("--weekly-common", action="store_true",
                          help="fixed common Monday windows 2026-06-15 through 2026-11-09")
    args = parser.parse_args()
    weights = args.model / "model.safetensors"
    with weights.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("model weights do not match the pinned upstream SHA-256")
    bars_digest = hashlib.sha256(args.bars.read_bytes()).hexdigest()
    script_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    bars = pd.read_csv(args.bars, parse_dates=["date"]).set_index("date")
    bars.index = pd.to_datetime(bars.index, utc=True)
    if bars.index.has_duplicates or not bars.index.is_monotonic_increasing or not (bars.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("duplicate, unordered or missing hourly bars")
    if not np.isfinite(bars.close).all() or not (bars.close > 0).all():
        raise ValueError("invalid close price")

    # Fixed decision times, 512 already closed hourly bars, forecast 24 hours
    # later. The alternative schedule is an exploratory diagnostic, not a holdout.
    windows = []
    if args.weekly_common:
        decisions = pd.date_range("2026-06-15", "2026-11-09", freq="7D", tz="UTC")
    elif args.weekly_fixed:
        decisions = pd.date_range("2026-01-05", "2026-11-09", freq="7D", tz="UTC")
    else:
        decisions = pd.date_range("2026-01-01", "2026-11-01", freq="MS", tz="UTC")
    for decision in decisions:
        past = bars.loc[bars.index < decision, "close"].tail(512)
        target = decision + pd.Timedelta(hours=23)
        if len(past) != 512 or target not in bars.index or past.index[-1] != decision - pd.Timedelta(hours=1):
            raise ValueError(f"missing input or target at {decision}")
        windows.append((decision, past.to_numpy(dtype=np.float32), float(bars.loc[target, "close"])))

    torch.set_num_threads(4)
    pipe = BaseChronosPipeline.from_pretrained(str(args.model), device_map="cpu", local_files_only=True)
    quantiles, _ = pipe.predict_quantiles([x[1] for x in windows], prediction_length=24,
                                          quantile_levels=[0.1, 0.5, 0.9], batch_size=4)
    rows = []
    for (decision, past, actual), forecast in zip(windows, quantiles, strict=True):
        q = forecast.detach().cpu().numpy().squeeze()
        if q.shape != (24, 3) or not np.isfinite(q).all():
            raise ValueError(f"unexpected forecast shape or non-finite output: {q.shape}")
        if not (q[:, 0] <= q[:, 1]).all() or not (q[:, 1] <= q[:, 2]).all():
            raise ValueError("forecast quantiles are out of order")
        low, median, high = (float(x) for x in q[-1])
        rows.append({"decision_utc": str(decision), "last_close": float(past[-1]), "actual_24h": actual,
                     "q10": low, "median": median, "q90": high,
                     "model_abs_error": abs(median-actual), "naive_abs_error": abs(float(past[-1])-actual)})
    report = {"status": "exploratory; not independently held out; no trading inference",
              "symbol_label": args.symbol, "venue": "user-supplied data; source not verified by this script",
              "model": "amazon/chronos-2", "revision": REVISION, "weights_sha256": digest,
              "training_data_overlap": "unknown", "context_hours": 512, "horizon_hours": 24,
              "sample_schedule": ("fixed_mondays_2026_06_15_to_2026_11_09" if args.weekly_common else
                                  "fixed_mondays_2026_01_05_to_2026_11_09" if args.weekly_fixed else "fixed_first_of_month"),
              "samples": len(rows), "forecast_mae": float(np.mean([r["model_abs_error"] for r in rows])),
              "last_close_baseline_mae": float(np.mean([r["naive_abs_error"] for r in rows])),
              "forecast_mean_absolute_pct_error": float(np.mean([100 * r["model_abs_error"] / r["actual_24h"] for r in rows])),
              "last_close_mean_absolute_pct_error": float(np.mean([100 * r["naive_abs_error"] / r["actual_24h"] for r in rows])),
              "nominal_80pct_interval_coverage": float(np.mean([r["q10"] <= r["actual_24h"] <= r["q90"] for r in rows])),
              "model_lower_error_count": sum(r["model_abs_error"] < r["naive_abs_error"] for r in rows),
              "input_sha256": bars_digest,
              "script_sha256": script_digest,
              "notes": "Reused historical data, unknown training overlap; no transaction cost, funding or order simulation; not a profitability test or calibration.",
              "forecasts": rows}
    if (hashlib.sha256(args.bars.read_bytes()).hexdigest() != bars_digest
            or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != script_digest):
        raise ValueError("input or experiment code changed during run; report withheld")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "forecasts"}, indent=2))


if __name__ == "__main__":
    main()
