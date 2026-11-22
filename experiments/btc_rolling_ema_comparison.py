"""Previously viewed BTC diagnostic: rolling-300 EMA versus frozen range candidate.

The two rules have different exits and market exposure. Funding is unknown.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments import btc_frozen_policy_history_check as base, btc_rolling_window_replay as candidate, frozen_policy_capture
from piia.intelligence.paper.specs import CONTRACTS
from trading_lab.perps.compare import attach_ema, simulate_book, summarize
from trading_lab.perps.research import validate_bars
from trading_lab.perps.risk import Spec


def rolling_ema(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, int]:
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("invalid window")
    validate_bars(frame)
    if frame.index.tz is None or frame.index.min() > start - pd.Timedelta(hours=300) or frame.index.max() < end - pd.Timedelta(hours=1):
        raise ValueError("insufficient coverage")
    result = attach_ema(frame.loc[frame.index < end])
    stamps = pd.date_range(start - pd.Timedelta(hours=1), end - pd.Timedelta(hours=2), freq="h")
    rows = []
    for stamp in stamps:
        recent = frame.loc[stamp - pd.Timedelta(hours=299):stamp]
        if len(recent) != 300 or recent.index[-1] != stamp:
            raise ValueError("missing rolling candles")
        last = attach_ema(recent).iloc[-1]
        rows.append([last[field] for field in candidate.FIELDS])
    values = pd.DataFrame(rows, index=stamps, columns=candidate.FIELDS)
    disagreements = int((values["signal"] != result.loc[stamps, "signal"]).sum())
    for field in candidate.FIELDS:
        result.loc[stamps, field] = values[field]
    return result, disagreements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--candidate-reference", type=Path, default=Path("artifacts/btc_rolling_window_replay.json"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/btc_rolling_ema_comparison.json"))
    args = parser.parse_args()
    frozen_policy_capture.verify_policy()
    path = args.data_dir / candidate.FILENAME
    hashes = {"input": base.sha(path), "candidate_reference": base.sha(args.candidate_reference),
              "script": base.sha(Path(__file__)), "candidate_script": base.sha(Path(candidate.__file__))}
    reference = json.loads(args.candidate_reference.read_text(encoding="utf-8"))
    if (reference["source_sha256"]["input"] != hashes["input"]
            or reference["source_sha256"]["script"] != hashes["candidate_script"]
            or reference["status"] != "exploratory_seen_data_price_only_not_net_or_held_out"
            or reference["order_permissions"] is not False or reference["window"] != candidate.LABEL):
        raise ValueError("candidate reference binding mismatch")
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    frame.index = pd.to_datetime(frame.index, utc=True)
    start, end = pd.Timestamp(candidate.START), pd.Timestamp(candidate.END)
    prepared, disagreement = rolling_ema(frame, start, end)
    terms = CONTRACTS["BTCUSDT"]
    spec = Spec(base.SYMBOL, *(terms[k] for k in ("min_qty", "qty_step", "min_notional", "taker", "contract_size")))
    raw = simulate_book({base.SYMBOL: prepared}, {base.SYMBOL: spec}, capital=100, leverage=2,
                        start=start, end=end - pd.Timedelta(hours=1), slippage=.0005, prepared=True)
    trades = raw.pop("trades")
    raw.pop("nav", None)
    ema = summarize({**raw, "trades": trades})
    cand = reference["rolling_300_candidate"]
    if any(x.get("funding_net_paid") is not None or x.get("funding_coverage_checked") is not False for x in (ema, cand)):
        raise ValueError("unexpected funding status")
    if (base.sha(path) != hashes["input"] or base.sha(args.candidate_reference) != hashes["candidate_reference"]
            or base.sha(Path(__file__)) != hashes["script"] or base.sha(Path(candidate.__file__)) != hashes["candidate_script"]):
        raise ValueError("inputs changed during run")
    frozen_policy_capture.verify_policy()
    report = {"status": "exploratory_seen_data_price_only_not_net_or_held_out", "order_permissions": False,
              "source_sha256": hashes, "funding": "unknown_excluded", "window": candidate.LABEL,
              "limitations": ["Previously viewed BTC only, not held out or a portfolio.",
                              "Both use 300 closed bars and shared costs, sizing and halts; exposures differ.",
                              "EMA has no time exit; candidate has a 24-hour time exit, per their frozen rules.",
                              "Next-open fills hypothetical; funding unknown; historic specifications and liquidation unverified."],
              "ema_full_vs_rolling_signal_differences": disagreement, "rolling_ema": ema,
              "rolling_frozen_candidate": cand, "cash_reference_usdt": 100}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
