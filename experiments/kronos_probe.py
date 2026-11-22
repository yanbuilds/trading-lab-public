"""Fixed exploratory 24h BTC OHLCV forecast probe. Never places orders.

Requires separately checked out upstream Kronos, pinned weights, and its
dependencies. All 2026 observations are already seen; this is not a holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

REFERENCE_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
MODEL_HASHES = {
    "mini/model.safetensors": "a7d5f37e2e9fbd9891f7d7d4f72574512dd1f704fee14223e0a8cd0fbf54197c",
    "mini/config.json": "70daca2cb11e3a979dd6b8ac12ee08e2aace877acf28f5b8dfb4fe5609736201",
    "tokenizer/model.safetensors": "b97ec46b3b72160509e289183eaf7bdf5f0dac5bb9b49522f6d46638a99a8717",
    "tokenizer/config.json": "0b30a443affb03e05a876a083857de9164f899feb7b4d261da02c485c9a3e3b6",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--bars", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/kronos_probe_2026.json"))
    args = ap.parse_args()
    reference = args.reference.resolve()
    commit = subprocess.run(["git", "-C", str(reference), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    if commit != REFERENCE_COMMIT:
        raise ValueError("upstream reference revision differs from the fixed experiment")
    dirty = subprocess.run(["git", "-C", str(reference), "status", "--porcelain"], check=True,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise ValueError("upstream reference checkout is modified")
    for relative, expected in MODEL_HASHES.items():
        with (args.model_dir / relative).open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != expected:
                raise ValueError(f"unexpected model file hash: {relative}")
    bars = pd.read_csv(args.bars, parse_dates=["date"]).set_index("date")
    bars.index = pd.to_datetime(bars.index, utc=True)
    if bars.index.has_duplicates or not bars.index.is_monotonic_increasing or not (bars.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("duplicate, unordered, or missing hourly bars")
    cols = ["open", "high", "low", "close", "volume"]
    if not np.isfinite(bars[cols]).all().all() or not (bars[cols[:4]] > 0).all().all() or not (bars.volume >= 0).all():
        raise ValueError("invalid OHLCV")
    if not ((bars.high >= bars[["open", "low", "close"]].max(axis=1)).all()
            and (bars.low <= bars[["open", "high", "close"]].min(axis=1)).all()):
        raise ValueError("invalid OHLC range")
    sys.path.insert(0, str(reference))
    from model import Kronos, KronosPredictor, KronosTokenizer

    torch.set_num_threads(4)
    tokenizer = KronosTokenizer.from_pretrained(str((args.model_dir / "tokenizer").resolve()), local_files_only=True).eval()
    model = Kronos.from_pretrained(str((args.model_dir / "mini").resolve()), local_files_only=True).eval()
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
    results = []
    for decision in pd.date_range("2026-01-01", "2026-11-01", freq="MS", tz="UTC"):
        past = bars.loc[bars.index < decision, cols].tail(512)
        target = decision + pd.Timedelta(hours=23)
        if len(past) != 512 or past.index[-1] != decision - pd.Timedelta(hours=1) or target not in bars.index:
            raise ValueError(f"missing input or target at {decision}")
        torch.manual_seed(0)
        future_times = pd.Series(pd.date_range(decision, periods=24, freq="h"))
        # The upstream predictor estimates missing quote turnover from OHLC*volume.
        prediction = predictor.predict(past.reset_index(drop=True), pd.Series(past.index), future_times,
                                       pred_len=24, T=1.0, top_k=0, top_p=0.9,
                                       sample_count=1, verbose=False)
        forecast = float(prediction["close"].iloc[-1])
        actual = float(bars.loc[target, "close"])
        previous = float(past.close.iloc[-1])
        if not np.isfinite(forecast):
            raise ValueError("non-finite model forecast")
        results.append({"decision_utc": str(decision), "last_close": previous, "actual_24h": actual,
                        "forecast_24h": forecast, "model_abs_error": abs(forecast - actual),
                        "naive_abs_error": abs(previous - actual)})
        print("evaluated", str(decision), flush=True)
    out = {"status": "exploratory, reused data, not a trading or prospective test",
           "model": "NeoQuasar/Kronos-mini", "model_revision": "7fdcc628d87f325ccdbcae0a372622ca7e6813aa",
           "tokenizer_revision": "b22fb9cb30a2de2f77e8b617169cd756ba964a08", "reference_commit": commit,
           "training_overlap": "unknown", "context_hours": 512, "horizon_hours": 24,
           "sampling": {"seed_per_window": 0, "temperature": 1.0, "top_p": 0.9, "sample_count": 1},
           "turnover": "upstream OHLC-mean times volume proxy, not actual turnover",
           "observations": len(results), "forecast_mae": float(np.mean([x["model_abs_error"] for x in results])),
           "last_close_baseline_mae": float(np.mean([x["naive_abs_error"] for x in results])),
           "limitations": "11 selected historical observations, stochastic single path, no costs, no trades, no profitability claim",
           "forecasts": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "forecasts"}, indent=2))


if __name__ == "__main__":
    main()
