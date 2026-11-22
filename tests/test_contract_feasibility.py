"""Synthetic examples only; not recommendations or current prices."""
import pytest

from experiments.contract_feasibility import minimum_feasibility


def test_minimum_stop_risk_is_not_fixed_by_upsizing():
    contract = {"symbol": "XAUUSDT", "status": "normal", "instrument_type": "perpetual", "min_qty": "0.01",
                "qty_step": "0.01", "min_notional": "5", "taker": "0.0006", "contract_size": "1"}
    out = minimum_feasibility("XAUUSDT", 5000., 100., contract)
    assert out["estimated_minimum_normal_stop_loss_usdt"] > out["risk_budget_usdt"]
    assert out["hypothetical_qty_rounded_down"] == 0 and out["reason"] == "minimum_order_exceeds_budget"
    assert out["order_permissions"] is False and out["exact_liquidation_verified"] is False


def test_cross_symbol_and_invalid_atr_are_rejected():
    contract = {"symbol": "BTCUSDT", "status": "normal", "instrument_type": "perpetual", "min_qty": ".0001",
                "qty_step": ".0001", "min_notional": "5", "taker": ".0006", "contract_size": "1"}
    with pytest.raises(ValueError, match="mismatch"):
        minimum_feasibility("XAUUSDT", 100., 1., contract)
    with pytest.raises(ValueError, match="invalid"):
        minimum_feasibility("BTCUSDT", 100., float("nan"), contract)
