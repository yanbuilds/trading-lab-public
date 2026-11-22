"""Non-tradable exposure-matched direction diagnostic for native results.

The original exit timestamps are known only after the fact. This calculation
does not simulate an opposite-side strategy, stops, margin, or slippage.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments.compare_native_prefix import read_trades


def direction_diagnostic(trades: list[dict], initial: float = 100.) -> dict:
    if not trades or not math.isfinite(initial) or initial <= 0:
        raise ValueError("nonempty trades and positive finite capital required")
    actual, inverse, rows = 0., 0., []
    for t in trades:
        values = [float(t[k]) for k in ("amount", "open_rate", "close_rate", "fee_open", "fee_close", "funding_fees", "profit_abs")]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("nonfinite trade field")
        qty, entry, exit_, fee_open, fee_close, funding, net = values
        if qty <= 0 or entry <= 0 or exit_ <= 0 or min(fee_open, fee_close) < 0:
            raise ValueError("invalid amount, price or fee")
        if not isinstance(t["is_short"], bool):
            raise ValueError("invalid trade side")
        direction = -1 if t["is_short"] else 1
        fee = qty * (entry * fee_open + exit_ * fee_close)
        gross = direction * qty * (exit_ - entry)
        if abs((gross - fee + funding) - net) > 1e-7:
            raise ValueError("original fill cannot be reconciled under linear unit contract assumption")
        mirror = -gross - fee - funding
        actual += net
        inverse += mirror
        rows.append({"pair": t["pair"], "open_date": t["open_date"], "close_date": t["close_date"],
                     "actual_net": net, "opposite_hypothetical_net": mirror})
    return {"initial_usdt": initial, "actual_total_net": actual, "opposite_hypothetical_total": inverse,
            "opposite_hypothetical_final": initial + inverse, "trades": rows,
            "exposure": "identical original entry/exit times and quantities, opposite direction",
            "status": "non_tradable_retrospective_counterfactual",
            "limitations": ["exit times and fills use hindsight", "same prices used for both directions; no slippage",
                            "funding sign inverted at identical quantities and times", "no margin, liquidation or stop-path simulation",
                            "not an independently executable benchmark or evidence of predictive skill"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_zip", type=Path)
    args = parser.parse_args()
    print(json.dumps(direction_diagnostic(read_trades(args.result_zip)), indent=2))


if __name__ == "__main__":
    main()
