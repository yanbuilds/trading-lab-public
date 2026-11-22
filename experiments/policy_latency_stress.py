"""Offline v1 execution-latency sensitivity; not a new policy or live replay.

One case uses the existing zero-delay next-open assumption. The other waits one
extra complete hour before acting on entry and middle-band exit signals. Stops,
time exits, and the backtest's open-price fill assumptions remain simplified.
Funding is excluded and unknown; no hyperparameter search or order access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from experiments.frozen_policy_capture import verify_policy
from experiments.regime_mean_reversion import EXPECTED, complete_manifest
from trading_lab.perps.compare import load_cache, simulate_book, summarize
from trading_lab.perps.mean_reversion import attach_range_bollinger


def lag_closed_bar_decisions(prepared: pd.DataFrame) -> pd.DataFrame:
    """Delay signal and its ATR, and middle-band exits, by exactly one bar."""
    out = prepared.copy()
    out["signal"] = prepared["signal"].shift(1).fillna(0).astype(int)
    out["atr"] = prepared["atr"].shift(1)
    for column in ("exit_long", "exit_short"):
        out[column] = prepared[column].shift(1, fill_value=False).astype(bool)
    return out


def run(frames, specs, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    if (start.tzinfo is None or end.tzinfo is None or start >= end or
            start != start.floor("h") or end != end.floor("h")):
        raise ValueError("ordered timezone-aware hourly boundaries required")
    if set(frames) != EXPECTED or set(specs) != EXPECTED:
        raise ValueError("three contract markets and matching specs required")
    prepared = {}
    for symbol, bars in frames.items():
        if specs[symbol].symbol != symbol:
            raise ValueError("contract specification mismatch")
        if bars.index.min() > start - pd.Timedelta(hours=300) or bars.index.max() < end - pd.Timedelta(hours=1):
            raise ValueError(f"missing warmup or price coverage for {symbol}")
        prepared[symbol] = attach_range_bollinger(bars.loc[bars.index < end])
    report = {"label": "retrospective sensitivity on viewed data; funding unknown; not net performance",
              "start_utc": str(start), "end_exclusive_utc": str(end), "initial_usdt": 100,
              "limitations": ["delay scenario is a sensitivity check, not policy v1 and not a prospective result",
                              "trade paths and exposures differ; difference is not a pure cost estimate",
                              "stops, timed exits, open fills and liquidation remain simplified",
                              "funding excluded and unknown; open terminal positions omit hypothetical exit costs",
                              "historical contract specs unverified; no market-depth or execution guarantee"],
              "cases": {}}
    for name, data in (("next_open_zero_delay", prepared),
                       ("extra_hour_signal_delay", {s: lag_closed_bar_decisions(x) for s, x in prepared.items()})):
        result = simulate_book(data, specs, funding=None, capital=100, start=start,
                               end=end - pd.Timedelta(hours=1), prepared=True, max_hold_hours=24)
        trades = result.pop("trades")
        result.pop("nav", None)
        report["cases"][name] = summarize({**result, "trades": trades})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/perps-extended"))
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True, help="exclusive UTC hour")
    parser.add_argument("--out", type=Path, default=Path("artifacts/policy_latency_stress.json"))
    args = parser.parse_args()
    verify_policy()
    inputs = complete_manifest(args.data_dir)
    own_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    frames, specs, _, _ = load_cache(args.data_dir)
    report = run(frames, specs, pd.Timestamp(args.start), pd.Timestamp(args.end))
    verify_policy()
    if inputs != complete_manifest(args.data_dir) or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != own_hash:
        raise ValueError("source or input changed during run")
    report["input_manifest"], report["script_sha256"] = inputs, own_hash
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"label": report["label"], "cases": report["cases"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
