"""Fixed BTC-only exploratory comparison; historical funding is unknown."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from experiments import frozen_policy_capture
from piia.intelligence.paper.specs import CONTRACTS
from trading_lab.perps.compare import attach_ema, simulate_book, summarize
from trading_lab.perps.mean_reversion import attach_range_bollinger
from trading_lab.perps.risk import Spec

SYMBOL = "BTC/USDT:USDT"
# Fixed, previously inspected windows. They are not held out.
WINDOWS = (
    ("2023", "BTCUSDT-2023.csv", "2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z"),
    ("2024", "BTCUSDT-2024.csv", "2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"),
    ("2026_partial", "BTCUSDT-year.csv", "2026-01-01T00:00:00Z", "2026-11-16T06:00:00Z"),
)


def sha(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def evaluate_window(frame: pd.DataFrame, spec: Spec, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    if (start.tzinfo is None or end.tzinfo is None or start >= end or start != start.floor("h")
            or end != end.floor("h") or frame.index.tz is None):
        raise ValueError("ordered timezone-aware hourly window required")
    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    if frame.index.min() > start - pd.Timedelta(days=25) or frame.index.max() < end - pd.Timedelta(hours=1):
        raise ValueError("insufficient warmup or coverage")
    frame = frame.loc[frame.index < end]
    result = {}
    for name, preparer, hold in (("frozen_range_candidate", attach_range_bollinger, 24),
                                 ("ema_baseline", attach_ema, None)):
        prepared = preparer(frame)
        raw = simulate_book({SYMBOL: prepared}, {SYMBOL: spec}, capital=100, leverage=2,
                            start=start, end=end - pd.Timedelta(hours=1), slippage=.0005,
                            prepared=True, max_hold_hours=hold)
        trades = raw.pop("trades")
        raw.pop("nav", None)
        summary = summarize({**raw, "trades": trades})
        if summary.get("funding_net_paid") is not None or summary.get("funding_coverage_checked") is not False:
            raise ValueError("unexpected funding status")
        result[name] = summary
    return {"start_utc": start.isoformat(), "end_exclusive_utc": end.isoformat(), "accounts": result,
            "cash_reference": {"initial_usdt": 100., "final_usdt": 100., "exposure": "none"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("artifacts/btc_frozen_policy_history_check.json"))
    args = parser.parse_args()
    frozen_policy_capture.verify_policy()
    files = {name: args.data_dir / name for _, name, _, _ in WINDOWS}
    before = {name: sha(path) for name, path in files.items()}
    source_hash = sha(Path(__file__))
    terms = CONTRACTS["BTCUSDT"]
    spec = Spec(SYMBOL, *(terms[k] for k in ("min_qty", "qty_step", "min_notional", "taker", "contract_size")))
    result = {"status": "exploratory_seen_data_price_only_not_net_or_held_out", "order_permissions": False,
              "funding": "unknown_excluded", "capital_per_window_usdt": 100, "symbol": SYMBOL,
              "input_sha256": before, "script_sha256": source_hash, "windows": {},
              "limitations": ["Previously viewed BTC windows; not held out, not a three-contract portfolio.",
                              "Historical funding excluded and unknown; returns are not verified net performance.",
                              "Current local specifications used as assumptions; historical terms, fills and liquidation unverified.",
                              "Capital resets per window; terminal open positions lack hypothetical exit costs.",
                              "Cash and strategies have different exposure; not a risk-matched ranking.",
                              "Full available warmup may differ from the 300-bar prospective observation."]}
    for label, name, start, end in WINDOWS:
        frame = pd.read_csv(files[name], parse_dates=["date"]).set_index("date")
        frame.index = pd.to_datetime(frame.index, utc=True)
        result["windows"][label] = evaluate_window(frame, spec, pd.Timestamp(start), pd.Timestamp(end))
    if {name: sha(path) for name, path in files.items()} != before or sha(Path(__file__)) != source_hash:
        raise ValueError("inputs or experiment changed during run")
    frozen_policy_capture.verify_policy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False, default=str))


if __name__ == "__main__":
    main()
