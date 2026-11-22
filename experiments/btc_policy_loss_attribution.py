"""Descriptive attribution of fixed, previously viewed BTC price-only runs."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import pandas as pd

from experiments import btc_frozen_policy_history_check as base, frozen_policy_capture
from piia.intelligence.paper.specs import CONTRACTS
from trading_lab.perps.compare import simulate_book, summarize
from trading_lab.perps.mean_reversion import attach_range_bollinger
from trading_lab.perps.risk import Spec


def summarize_trades(trades: list[dict], *, final: float, fees: float, open_positions: int) -> dict:
    """Closed trade net already includes assumed fees/slippage; funding is unknown."""
    groups = {"long": {"count": 0, "net_pnl_usdt": 0.}, "short": {"count": 0, "net_pnl_usdt": 0.}}
    exits = {}
    for trade in trades:
        side, pnl, reason = trade["side"], float(trade["net_pnl"]), trade["exit_reason"]
        if type(side) is not int or side not in (-1, 1) or not math.isfinite(pnl) or not isinstance(reason, str) or not reason:
            raise ValueError("invalid trade")
        group = groups["long" if side == 1 else "short"]
        group["count"] += 1
        group["net_pnl_usdt"] += pnl
        item = exits.setdefault(reason, {"count": 0, "net_pnl_usdt": 0.})
        item["count"] += 1
        item["net_pnl_usdt"] += pnl
    closed_net = sum(x["net_pnl_usdt"] for x in groups.values())
    if not math.isfinite(final) or not math.isfinite(fees) or fees < 0 or open_positions not in (0, 1):
        raise ValueError("invalid summary")
    if open_positions == 0 and not math.isclose(final - 100., closed_net, abs_tol=1e-7):
        raise ValueError("closed trade attribution does not reconcile")
    return {"closed_trades_by_side": groups, "closed_trades_by_exit_reason": exits,
            "closed_trade_net_after_assumed_fees_and_slippage_usdt": closed_net,
            "assumed_total_fees_usdt": fees,
            "same_fills_before_assumed_fees_usdt": closed_net + fees if open_positions == 0 else None,
            "open_positions": open_positions, "funding": "unknown_excluded",
            "note": "Removing fees while holding fills fixed is attribution only, not a counterfactual backtest; fees affect size and path."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=Path("artifacts/btc_frozen_policy_history_check.json"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/btc_policy_loss_attribution.json"))
    args = parser.parse_args()
    frozen_policy_capture.verify_policy()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    if (reference.get("status") != "exploratory_seen_data_price_only_not_net_or_held_out"
            or reference.get("order_permissions") is not False or reference.get("funding") != "unknown_excluded"
            or reference.get("script_sha256") != base.sha(Path(base.__file__))):
        raise ValueError("reference report is invalid or from a different experiment")
    files = {name: args.data_dir / name for _, name, _, _ in base.WINDOWS}
    hashes = {name: base.sha(path) for name, path in files.items()}
    if hashes != reference.get("input_sha256"):
        raise ValueError("input files differ from reference")
    terms = CONTRACTS["BTCUSDT"]
    spec = Spec(base.SYMBOL, *(terms[k] for k in ("min_qty", "qty_step", "min_notional", "taker", "contract_size")))
    output = {"status": "exploratory_seen_data_trade_attribution_not_net_or_holdout", "order_permissions": False,
              "funding": "unknown_excluded", "reference_sha256": base.sha(args.reference), "windows": {},
              "limitations": ["Same previously viewed BTC-only runs; not out of sample.",
                              "Funding, historical specifications and liquidation are unverified.",
                              "Price fills include assumed slippage, not observed executions.",
                              "Fee attribution keeps the same fills and cannot establish a no-fee strategy outcome."]}
    for label, name, start, end in base.WINDOWS:
        frame = pd.read_csv(files[name], parse_dates=["date"]).set_index("date")
        frame.index = pd.to_datetime(frame.index, utc=True)
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        frame = frame.loc[frame.index < end]
        raw = simulate_book({base.SYMBOL: attach_range_bollinger(frame)}, {base.SYMBOL: spec}, capital=100,
                            leverage=2, start=start, end=end - pd.Timedelta(hours=1), slippage=.0005,
                            prepared=True, max_hold_hours=24)
        trades = raw.pop("trades")
        raw.pop("nav", None)
        actual = summarize({**raw, "trades": trades})
        if actual != reference["windows"][label]["accounts"]["frozen_range_candidate"]:
            raise ValueError("replay differs from reference report")
        output["windows"][label] = {"loss_halt": actual["loss_halt"],
            "attribution": summarize_trades(trades, final=actual["final"], fees=actual["fees"],
                                            open_positions=actual["open_positions"])}
    if {name: base.sha(path) for name, path in files.items()} != hashes or base.sha(args.reference) != output["reference_sha256"]:
        raise ValueError("input or reference changed during attribution")
    frozen_policy_capture.verify_policy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
