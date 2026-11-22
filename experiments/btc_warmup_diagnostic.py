"""Exploratory weekly comparison of full-history and rolling-300 BTC signals.

Previously inspected prices; this does not calculate trading performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments import btc_frozen_policy_history_check as base, frozen_policy_capture
from trading_lab.perps.mean_reversion import attach_range_bollinger
from trading_lab.perps.research import validate_bars


def compare_window(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("ordered timezone-aware window required")
    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    validate_bars(frame)
    if frame.index.tz is None or frame.index.min() > start - pd.Timedelta(hours=300) or frame.index.max() < end - pd.Timedelta(hours=1):
        raise ValueError("insufficient coverage")
    full = attach_range_bollinger(frame.loc[frame.index < end])
    observations = []
    n = 0
    while True:
        # Cycle all four 4h boundary alignments; predetermined, not optimized.
        stamp = start + pd.Timedelta(days=7 * n, hours=n % 4)
        if stamp >= end:
            break
        rolling = frame.loc[stamp - pd.Timedelta(hours=299):stamp]
        if len(rolling) != 300 or rolling.index[-1] != stamp:
            raise ValueError("missing rolling observation")
        recent = attach_range_bollinger(rolling).iloc[-1]
        long = full.loc[stamp]
        observations.append({"bar_open_utc": stamp.isoformat(),
                             "full_regime": str(long.regime), "rolling_regime": str(recent.regime),
                             "full_signal": int(long.signal), "rolling_signal": int(recent.signal)})
        n += 1
    return {"sample_count": len(observations),
            "regime_mismatches": sum(x["full_regime"] != x["rolling_regime"] for x in observations),
            "signal_mismatches": sum(x["full_signal"] != x["rolling_signal"] for x in observations),
            "full_nonzero": sum(x["full_signal"] != 0 for x in observations),
            "rolling_nonzero": sum(x["rolling_signal"] != 0 for x in observations),
            "mismatches": [x for x in observations if x["full_regime"] != x["rolling_regime"] or x["full_signal"] != x["rolling_signal"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("artifacts/btc_warmup_diagnostic.json"))
    args = parser.parse_args()
    frozen_policy_capture.verify_policy()
    files = {name: args.data_dir / name for _, name, _, _ in base.WINDOWS}
    hashes = {name: base.sha(path) for name, path in files.items()}
    source_hash = base.sha(Path(__file__))
    result = {"status": "exploratory_seen_data_signal_sampling_not_performance",
              "sampling": "one bar weekly; UTC hour cycles 00, 01, 02, 03",
              "limitations": ["Previously viewed BTC data only; not held out.",
                              "Sampled bars only; cannot infer full-hour disagreement or trading returns.",
                              "Full history begins at each input file's first bar; rolling uses last 300 bars."],
              "input_sha256": hashes, "script_sha256": source_hash, "windows": {}}
    for label, name, start, end in base.WINDOWS:
        frame = pd.read_csv(files[name], parse_dates=["date"]).set_index("date")
        frame.index = pd.to_datetime(frame.index, utc=True)
        result["windows"][label] = compare_window(frame, pd.Timestamp(start), pd.Timestamp(end))
    if {name: base.sha(path) for name, path in files.items()} != hashes or base.sha(Path(__file__)) != source_hash:
        raise ValueError("inputs or experiment changed during run")
    frozen_policy_capture.verify_policy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
