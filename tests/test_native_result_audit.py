import pandas as pd
import pytest

from experiments.audit_native_result import reconcile, validate_trade_funding


def test_funding_reconciliation_flags_closing_boundary_and_rejects_tampering():
    t = pd.Timestamp("2026-01-01T08:00:00Z")
    frame = pd.DataFrame({"date": [t], "funding_rate": [.001], "open": [100.]})
    trade = {"pair": "BTC/USDT:USDT", "open_date": "2026-01-01T07:00:00Z",
             "close_date": t.isoformat(), "amount": 2, "is_short": False,
             "funding_fees": -.2, "profit_abs": -.2}
    result = {"trades": [trade], "starting_balance": 100., "final_balance": 99.8}
    output = reconcile(result, {"BTC": frame})
    assert output["boundary_trades"][0]["at"] == ["close"]
    trade["funding_fees"] = 0
    with pytest.raises(ValueError, match="funding mismatch"):
        reconcile(result, {"BTC": frame})


def test_funding_at_end_boundary_requires_matching_reference():
    first = pd.Timestamp("2026-01-01T07:00:00Z")
    last = pd.Timestamp("2026-01-01T08:00:00Z")
    funding = pd.DataFrame({"date": [last], "funding_rate": [.001]})
    reference = pd.DataFrame({"timestamp": [int(last.timestamp() * 1000)], "rate": [.001]})
    assert len(validate_trade_funding("BTC", funding, reference, first, last)) == 1
    with pytest.raises(ValueError, match="missing or unexpected"):
        validate_trade_funding("BTC", funding.iloc[:0], reference, first, last)
    reference.loc[0, "rate"] = .002
    with pytest.raises(ValueError, match="reference funding mismatch"):
        validate_trade_funding("BTC", funding, reference, first, last)
