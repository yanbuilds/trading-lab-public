import pytest

from piia.intelligence.scenarios import position_scenarios, portfolio_stress


@pytest.mark.parametrize("side,stop,gap", [("long", 90, 80), ("short", 110, 120)])
def test_long_short_stop_and_adverse_gap_include_both_fees(side, stop, gap):
    result = position_scenarios(side=side, equity=100, entry=100, stop=stop, gap_exit=gap,
                                quantity=2, contract_size=0.5, leverage=2, fee_rate=0.001)
    a, b = result["stop_fill_assumption"], result["gap_fill_assumption"]
    assert a["gross_pnl_usdt"] == -10
    assert b["gross_pnl_usdt"] == -20
    assert a["net_pnl_excluding_funding_usdt"] == pytest.approx(-10 - 0.1 - stop * 0.001)
    assert b["loss_pct_of_entered_equity"] > a["loss_pct_of_entered_equity"]
    assert result["notional_divided_by_leverage_usdt"] == 50
    assert result["liquidation_model_supported"] is False
    assert result["funding"] == "unknown_excluded"
    assert result["order_permissions"] is False


def test_loss_may_exceed_input_equity_and_is_not_clamped():
    result = position_scenarios(side="short", equity=10, entry=100, stop=110, gap_exit=200,
                                quantity=1, contract_size=1, leverage=2, fee_rate=0)
    assert result["gap_fill_assumption"]["arithmetic_equity_after_pnl"] == -90
    assert result["gap_fill_assumption"]["loss_pct_of_entered_equity"] == 1000


@pytest.mark.parametrize("side,stop,gap", [("long", 110, 90), ("short", 90, 110), ("long", 90, 95)])
def test_rejects_non_adverse_scenarios(side, stop, gap):
    with pytest.raises(ValueError):
        position_scenarios(side=side, equity=100, entry=100, stop=stop, gap_exit=gap,
                           quantity=1, contract_size=1, leverage=2, fee_rate=.001)


def test_portfolio_joint_shock_counts_gross_without_offset():
    positions = [
        dict(symbol="btc", side="long", quantity=1, contract_size=1, mark_price=100,
             stress_price=90, exit_fee_rate=.001, slippage_rate=.01),
        dict(symbol="BTC", side="short", quantity=1, contract_size=1, mark_price=100,
             stress_price=110, exit_fee_rate=.001, slippage_rate=.01),
    ]
    result = portfolio_stress(equity=100, positions=positions)
    assert result["gross_notional_usdt"] == 200
    assert result["largest_symbol_notional_over_equity"] == 2
    assert result["gross_notional_over_equity"] == 2
    assert result["simultaneous_pnl_after_assumed_exit_cost_usdt"] == pytest.approx(-22.2002)
    assert result["arithmetic_equity_after_pnl"] == pytest.approx(77.7998)
    assert not result["correlation_estimated"] and not result["liquidation_model_supported"]
    assert result["funding"] == "unknown_excluded" and result["order_permissions"] is False


def test_portfolio_rejects_missing_nonfinite_and_empty():
    position = dict(symbol="BTC", side="long", quantity=1, contract_size=1, mark_price=100,
                    stress_price=90, exit_fee_rate=.001, slippage_rate=.01)
    for positions in ([], [{**position, "mark_price": float("nan")}], [{**position, "side": "flat"}],
                      [{**position, "quantity": True}], [{**position, "slippage_rate": 1.0}]):
        with pytest.raises(ValueError):
            portfolio_stress(equity=100, positions=positions)
