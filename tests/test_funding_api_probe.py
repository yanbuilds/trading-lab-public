import pytest

from experiments.funding_api_probe import inspect_page


def test_inspect_page_does_not_treat_errors_or_wrong_symbol_as_data():
    row = {"symbol": "BTCUSDT", "fundingRateTimestamp": "1790006400000", "fundingRate": "0.0001"}
    assert inspect_page({"code": "00000", "data": {"resultList": [row]}}, "BTCUSDT")["record_count"] == 1
    assert inspect_page({"code": "00000", "data": {"resultList": []}}, "BTCUSDT")["oldest_utc"] is None
    with pytest.raises(ValueError):
        inspect_page({"code": "12345", "data": {"resultList": []}}, "BTCUSDT")
    for bad in ({**row, "symbol": "XAUUSDT"}, {**row, "fundingRate": "nan"},
                {**row, "fundingRateTimestamp": "bad"}):
        with pytest.raises(ValueError):
            inspect_page({"code": "00000", "data": {"resultList": [bad]}}, "BTCUSDT")
    with pytest.raises(ValueError, match="duplicate"):
        inspect_page({"code": "00000", "data": {"resultList": [row, row]}}, "BTCUSDT")
