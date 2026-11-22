"""Synthetic data only; coverage is not trading performance."""
import json
import pandas as pd
import pytest

from experiments.history_coverage_audit import PAIRS, audit


def test_coverage_fails_closed_for_missing_bars_and_funding(tmp_path):
    start = pd.Timestamp("2026-06-01T00:00:00Z")
    stamps = pd.date_range(start - pd.Timedelta(hours=1), periods=4, freq="h")
    provenance = {"symbols": {pair: {"funding_interval_hours": 1} for pair in PAIRS.values()}}
    (tmp_path / "provenance.json").write_text(json.dumps(provenance))
    for symbol in PAIRS:
        pd.DataFrame({"date": stamps.astype(str), "open": 100, "high": 101, "low": 99,
                      "close": 100, "volume": 1}).to_csv(tmp_path / f"{symbol}.csv", index=False)
        pd.DataFrame({"timestamp": [int(t.timestamp() * 1000) for t in stamps[1:]],
                      "rate": [0.001] * 3}).to_csv(tmp_path / f"{symbol}-funding.csv", index=False)
    args = (tmp_path, start.isoformat(), (start + pd.Timedelta(hours=3)).isoformat())
    complete = audit(*args, warmup_hours=1)
    assert complete["all_symbols_locally_complete_under_assumptions"]
    assert complete["verified_net_performance_available"] is False
    assert complete["order_permissions"] is False
    assert not complete["symbols"]["NDX100USDT"]["listing_announcement"]["requested_start_definitely_before_listing_day"]
    older = audit(tmp_path, "2026-05-20T00:00:00Z", "2026-05-20T03:00:00Z", warmup_hours=0)
    assert older["symbols"]["NDX100USDT"]["listing_announcement"]["requested_start_definitely_before_listing_day"]
    assert not older["all_symbols_locally_complete_under_assumptions"]
    alternate = tmp_path / "alternate-btc.csv"
    alternate.write_bytes((tmp_path / "BTCUSDT.csv").read_bytes())
    alt_report = audit(*args, warmup_hours=1, btc_bars=alternate)
    assert alt_report["symbols"]["BTCUSDT"]["bars_complete_with_warmup"]
    assert not alt_report["symbols"]["BTCUSDT"]["using_cache_default_filename"]
    assert "does not establish" in alt_report["symbols"]["BTCUSDT"]["bars_provenance_note"]
    xau = pd.read_csv(tmp_path / "XAUUSDT.csv").drop(index=1)
    xau.to_csv(tmp_path / "XAUUSDT.csv", index=False)
    (tmp_path / "BTCUSDT-funding.csv").unlink()
    missing = audit(*args, warmup_hours=1)
    assert not missing["all_symbols_locally_complete_under_assumptions"]
    assert missing["symbols"]["XAUUSDT"]["missing_bars_with_warmup"] == 1
    assert not missing["symbols"]["BTCUSDT"]["funding_complete_under_recorded_interval_assumption"]
    ndx_path = tmp_path / "NDX100USDT.csv"
    ndx = pd.read_csv(ndx_path)
    ndx["high"] = ndx["high"].astype(float)
    ndx.loc[1, "high"] = float("inf")
    ndx.to_csv(ndx_path, index=False)
    invalid = audit(*args, warmup_hours=1)
    assert "bars_error" in invalid["symbols"]["NDX100USDT"]
    assert not invalid["symbols"]["NDX100USDT"]["bars_complete_with_warmup"]
    with pytest.raises(ValueError):
        audit(tmp_path, "2026-01-01", "2026-01-02", warmup_hours=1)
