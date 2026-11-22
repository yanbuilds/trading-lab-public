import pytest

from experiments.direction_counterfactual import direction_diagnostic


def test_opposite_direction_preserves_costs_and_inverts_funding():
    trade = {"pair": "BTC/USDT:USDT", "open_date": "a", "close_date": "b", "amount": 1.,
             "open_rate": 100., "close_rate": 110., "fee_open": .001, "fee_close": .001,
             "funding_fees": -.5, "profit_abs": 9.29, "is_short": False}
    result = direction_diagnostic([trade])
    assert result["opposite_hypothetical_total"] == pytest.approx(-9.71)
    assert result["status"] == "non_tradable_retrospective_counterfactual"
    with pytest.raises(ValueError, match="cannot be reconciled"):
        direction_diagnostic([{**trade, "profit_abs": 9.}])
