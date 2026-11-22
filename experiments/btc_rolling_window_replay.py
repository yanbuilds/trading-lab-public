"""Seen-data BTC diagnostic: replay the frozen candidate with a 300-bar signal window.

Next-open fills are hypothetical and unavailable to the real prospective capture.
Funding is unknown; this is not an out-of-sample or net-performance result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments import btc_frozen_policy_history_check as base, frozen_policy_capture
from piia.intelligence.paper.specs import CONTRACTS
from trading_lab.perps.compare import simulate_book, summarize
from trading_lab.perps.mean_reversion import attach_range_bollinger
from trading_lab.perps.research import validate_bars
from trading_lab.perps.risk import Spec

LABEL, FILENAME, START, END = base.WINDOWS[2]
FIELDS = ("signal", "atr", "exit_long", "exit_short", "trail_long", "trail_short")


def rolling_prepared(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """Each decision uses only 300 bars ending at the preceding closed bar."""
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("invalid window")
    validate_bars(frame)
    if frame.index.tz is None or frame.index.min() > start - pd.Timedelta(hours=300) or frame.index.max() < end - pd.Timedelta(hours=1):
        raise ValueError("insufficient coverage")
    full = attach_range_bollinger(frame.loc[frame.index < end])
    result = full.copy()
    stamps = pd.date_range(start - pd.Timedelta(hours=1), end - pd.Timedelta(hours=2), freq="h")
    rows, regime_differences = [], 0
    for stamp in stamps:
        recent = frame.loc[stamp - pd.Timedelta(hours=299):stamp]
        if len(recent) != 300 or recent.index[-1] != stamp:
            raise ValueError("missing rolling candles")
        last = attach_range_bollinger(recent).iloc[-1]
        rows.append([last[field] for field in FIELDS])
        regime_differences += last["regime"] != full.loc[stamp, "regime"]
    values = pd.DataFrame(rows, index=stamps, columns=FIELDS)
    signal_differences = int((values["signal"] != full.loc[stamps, "signal"]).sum())
    for field in FIELDS:
        result.loc[stamps, field] = values[field]
    return result, {"decision_hours": len(stamps), "regime_differences": int(regime_differences),
                    "signal_differences": signal_differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=Path("artifacts/btc_frozen_policy_history_check.json"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/btc_rolling_window_replay.json"))
    args = parser.parse_args()
    frozen_policy_capture.verify_policy()
    data_path = args.data_dir / FILENAME
    before = {"input": base.sha(data_path), "reference": base.sha(args.reference), "script": base.sha(Path(__file__))}
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    if reference["input_sha256"][FILENAME] != before["input"] or reference["script_sha256"] != base.sha(Path(base.__file__)):
        raise ValueError("historical reference does not match current data and experiment")
    frame = pd.read_csv(data_path, parse_dates=["date"]).set_index("date")
    frame.index = pd.to_datetime(frame.index, utc=True)
    start, end = pd.Timestamp(START), pd.Timestamp(END)
    terms = CONTRACTS["BTCUSDT"]
    spec = Spec(base.SYMBOL, *(terms[k] for k in ("min_qty", "qty_step", "min_notional", "taker", "contract_size")))
    full = base.evaluate_window(frame, spec, start, end)["accounts"]["frozen_range_candidate"]
    if full != reference["windows"][LABEL]["accounts"]["frozen_range_candidate"]:
        raise ValueError("reference summary does not reproduce")
    prepared, differences = rolling_prepared(frame, start, end)
    raw = simulate_book({base.SYMBOL: prepared}, {base.SYMBOL: spec}, capital=100, leverage=2,
                        start=start, end=end - pd.Timedelta(hours=1), slippage=.0005,
                        prepared=True, max_hold_hours=24)
    trades = raw.pop("trades")
    raw.pop("nav", None)
    rolling = summarize({**raw, "trades": trades})
    if rolling.get("funding_net_paid") is not None or rolling.get("funding_coverage_checked") is not False:
        raise ValueError("unexpected funding status")
    if before != {"input": base.sha(data_path), "reference": base.sha(args.reference), "script": base.sha(Path(__file__))}:
        raise ValueError("inputs changed during run")
    frozen_policy_capture.verify_policy()
    report = {"status": "exploratory_seen_data_price_only_not_net_or_held_out", "order_permissions": False,
              "source_sha256": before, "window": LABEL, "funding": "unknown_excluded",
              "limitations": ["Previously seen BTC-only data, not held out or a three-contract portfolio.",
                              "Next-open fills are hypothetical; prospective capture occurs later.",
                              "Funding unknown; current assumed specifications and fills not historically verified.",
                              "No exact liquidation or terminal exit costs; a diagnostic, not investable performance."],
              "differences": differences, "full_history_candidate": full, "rolling_300_candidate": rolling}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
