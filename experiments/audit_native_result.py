"""Reconcile exported Freqtrade trades against already-audited local inputs.

This checks the framework's inclusive funding convention, not actual venue
settlement or the accuracy of historical execution and liquidation models.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

# Current assumed settlement cadence; historical cadence is not independently verified.
SYMBOLS = {"BTC": 8, "XAU": 4, "NDX100": 8}


def validate_trade_funding(symbol: str, funding: pd.DataFrame, reference: pd.DataFrame,
                           first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    """Validate assumed settlement grid including an event at the exit boundary."""
    interval = SYMBOLS[symbol]
    selected = funding[(funding.date >= first) & (funding.date <= last)].copy()
    if selected.date.duplicated().any():
        raise ValueError(f"duplicate funding timestamp for {symbol}")
    expected = pd.date_range(first.ceil(f"{interval}h"), last, freq=f"{interval}h")
    if len(expected) != len(selected) or len(expected.difference(pd.DatetimeIndex(selected.date))):
        raise ValueError(f"missing or unexpected funding observation for {symbol}")
    ref = reference.copy()
    ref["date"] = pd.to_datetime(ref["timestamp"], unit="ms", utc=True, errors="raise")
    if ref.date.duplicated().any():
        raise ValueError(f"duplicate reference timestamp for {symbol}")
    joined = selected.merge(ref[["date", "rate"]], on="date", how="left", validate="one_to_one")
    values = pd.to_numeric(joined.rate, errors="coerce")
    actual = pd.to_numeric(joined.funding_rate, errors="coerce")
    if any(not math.isfinite(float(v)) for v in values) or any(not math.isfinite(float(v)) for v in actual):
        raise ValueError(f"missing or invalid reference funding for {symbol}")
    if ((values - actual).abs() > 1e-12).any():
        raise ValueError(f"reference funding mismatch for {symbol}")
    return selected


def reconcile(result: dict, frames: dict[str, pd.DataFrame]) -> dict:
    trades = result["trades"]
    checks, boundary = [], []
    for index, trade in enumerate(trades):
        symbol = trade["pair"].split("/")[0]
        frame = frames[symbol]
        opened, closed = pd.Timestamp(trade["open_date"]), pd.Timestamp(trade["close_date"])
        if opened.tzinfo is None or closed.tzinfo is None or opened > closed:
            raise ValueError("invalid trade times")
        relevant = frame[(frame.date >= opened) & (frame.date <= closed)]
        if not all(math.isfinite(float(x)) for x in relevant[["funding_rate", "open"]].to_numpy().flat):
            raise ValueError("non-finite funding input")
        estimate = float((relevant.funding_rate * relevant.open).sum()) * float(trade["amount"])
        if not trade["is_short"]:
            estimate *= -1
        reported = float(trade["funding_fees"])
        if not math.isfinite(reported) or abs(estimate - reported) > 1e-8:
            raise ValueError(f"funding mismatch for trade {index}")
        sides = [label for label, time in (("open", opened), ("close", closed)) if (relevant.date == time).any()]
        if sides:
            boundary.append({"trade_index": index, "pair": trade["pair"], "at": sides})
        checks.append({"trade_index": index, "pair": trade["pair"], "events": len(relevant),
                       "recomputed_funding": estimate, "reported_funding": reported})
    profit = sum(float(t["profit_abs"]) for t in trades)
    delta = float(result["final_balance"]) - float(result["starting_balance"])
    if not math.isfinite(profit) or not math.isfinite(delta) or abs(profit - delta) > 1e-8:
        raise ValueError("balance mismatch")
    return {"trade_count": len(trades), "funding_checks": checks, "boundary_trades": boundary,
            "profit_sum": profit, "balance_change": delta,
            "caveat": "Reproduces Freqtrade inclusive timestamp convention; does not validate exchange settlement, slippage, or liquidation."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_zip", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("reference_dir", type=Path)
    args = parser.parse_args()
    with ZipFile(args.result_zip) as archive:
        names = [n for n in archive.namelist() if n.endswith(".json") and not n.endswith("_config.json")]
        if len(names) != 1:
            raise ValueError("ambiguous result archive")
        result = json.loads(archive.read(names[0]))["strategy"]["BitgetBaseline"]
    frames = {}
    for symbol in {trade["pair"].split("/")[0] for trade in result["trades"]}:
        paths = {kind: list(args.data_dir.rglob(f"{symbol}_USDT_USDT-1h-{kind}.feather"))
                 for kind in ("mark", "funding_rate")}
        if any(len(v) != 1 for v in paths.values()):
            raise ValueError(f"missing or ambiguous input for {symbol}")
        mark, funding = (pd.read_feather(paths[kind][0]) for kind in ("mark", "funding_rate"))
        symbol_trades = [t for t in result["trades"] if t["pair"].split("/")[0] == symbol]
        first = min(pd.Timestamp(t["open_date"]) for t in symbol_trades)
        last = max(pd.Timestamp(t["close_date"]) for t in symbol_trades)
        reference = pd.read_csv(args.reference_dir / f"{symbol}USDT-funding.csv")
        funding = validate_trade_funding(symbol, funding, reference, first, last)
        frames[symbol] = funding.merge(mark[["date", "open"]], on="date", validate="one_to_one")
        if len(frames[symbol]) != len(funding):
            raise ValueError(f"funding observations missing mark price for {symbol}")
    report = reconcile(result, frames)
    report["funding_reference"] = "separate download from same exchange; historical interval assumptions not independently verified"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
