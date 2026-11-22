from pathlib import Path

import pandas as pd
import pytest

from experiments.native_three_market_backtest import SYMBOLS, audit_data, write_prefix_data


def _fixture(root: Path):
    source, reference = root / "native", root / "reference"
    source.mkdir(); reference.mkdir()
    start = pd.Timestamp("2026-11-02T00:00:00Z")
    end = start + pd.Timedelta(hours=24)
    bars = pd.date_range(start - pd.Timedelta(hours=2), end - pd.Timedelta(hours=1), freq="1h")
    for symbol, interval in SYMBOLS.items():
        prices = pd.DataFrame({"date": bars, "open": 100., "high": 101., "low": 99., "close": 100., "volume": 1.})
        for kind in ("futures", "mark"):
            prices.to_feather(source / f"{symbol}_USDT_USDT-1h-{kind}.feather")
        dates = pd.date_range(start, end - pd.Timedelta(hours=1), freq=f"{interval}h")
        pd.DataFrame({"date": dates, "funding_rate": 0.0001}).to_feather(source / f"{symbol}_USDT_USDT-1h-funding_rate.feather")
        pd.DataFrame({"timestamp": [int(t.timestamp()*1000) for t in dates], "rate": 0.0001}).to_csv(reference / f"{symbol}USDT-funding.csv", index=False)
    return source, reference, start, end


def test_audit_rejects_missing_funding_and_mismatched_reference(tmp_path):
    source, reference, start, end = _fixture(tmp_path)
    out = audit_data(source, reference, start=start, end=end, warmup=2)
    assert out["files"]["BTC:funding_rate"]["rows_in_window"] == 3
    path = source / "BTC_USDT_USDT-1h-funding_rate.feather"
    good = pd.read_feather(path)
    good.iloc[:-1].to_feather(path)
    with pytest.raises(ValueError, match="missing or unexpected"):
        audit_data(source, reference, start=start, end=end, warmup=2)
    good.to_feather(path)
    path = reference / "BTCUSDT-funding.csv"
    changed = pd.read_csv(path); changed.loc[0, "rate"] = .5; changed.to_csv(path, index=False)
    with pytest.raises(ValueError, match="funding mismatch"):
        audit_data(source, reference, start=start, end=end, warmup=2)


def test_prefix_copy_contains_no_cutoff_or_future_rows_and_preserves_source(tmp_path):
    source, _, start, _ = _fixture(tmp_path)
    cutoff = start + pd.Timedelta(hours=8)
    original = (source / "BTC_USDT_USDT-1h-funding_rate.feather").read_bytes()
    output = tmp_path / "prefix"
    manifest = write_prefix_data(source, output, cutoff)
    assert len(manifest) == 9
    for path in output.rglob("*.feather"):
        data = pd.read_feather(path)
        assert (pd.to_datetime(data.date, utc=True) < cutoff).all()
    assert (source / "BTC_USDT_USDT-1h-funding_rate.feather").read_bytes() == original
