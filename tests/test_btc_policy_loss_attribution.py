"""Synthetic accounting only; not market results."""
import pytest

from experiments.btc_policy_loss_attribution import summarize_trades


def test_closed_trade_attribution_reconciles_without_claiming_funding():
    trades = [{"side": 1, "net_pnl": -1.2, "exit_reason": "atr_stop"},
              {"side": -1, "net_pnl": .2, "exit_reason": "reverse_signal"}]
    result = summarize_trades(trades, final=99., fees=.3, open_positions=0)
    assert result["closed_trades_by_side"]["long"] == {"count": 1, "net_pnl_usdt": -1.2}
    assert result["closed_trades_by_exit_reason"]["atr_stop"]["count"] == 1
    assert result["same_fills_before_assumed_fees_usdt"] == pytest.approx(-.7)
    assert result["funding"] == "unknown_excluded"
    with pytest.raises(ValueError, match="reconcile"):
        summarize_trades(trades, final=100., fees=.3, open_positions=0)
    assert summarize_trades(trades, final=100., fees=.3, open_positions=1)["same_fills_before_assumed_fees_usdt"] is None
