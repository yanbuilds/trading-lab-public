from datetime import timezone
import sys

import pandas as pd
import pytest

from trading_lab.perps.risk import Spec
from experiments import portfolio_cost_stress


def test_missing_funding_blocks_comparison(monkeypatch, tmp_path):
    symbols = ("BTC/USDT:USDT", "XAU/USDT:USDT", "NDX100/USDT:USDT")
    idx = pd.date_range("2026-01-01", periods=260, freq="h", tz=timezone.utc)
    bars = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1.0}, index=idx)
    frames = {s: bars for s in symbols}
    specs = {s: Spec(s, 0.01, 0.01, 5) for s in symbols}
    funding = {s: [] for s in symbols}
    provenance = {"symbols": {s: {"funding_interval_hours": 8} for s in symbols}}
    monkeypatch.setattr(portfolio_cost_stress, "load_cache", lambda _: (frames, specs, funding, provenance))
    monkeypatch.setattr(portfolio_cost_stress, "portfolio_manifest", lambda _: {"synthetic_fixture": True})
    monkeypatch.setattr(sys, "argv", ["cost_stress", "--start", idx[240].isoformat(),
                                       "--end", (idx[-1] + pd.Timedelta(hours=1)).isoformat(),
                                       "--out", str(tmp_path / "result.json")])
    with pytest.raises(ValueError, match="incomplete funding"):
        portfolio_cost_stress.main()
    assert not (tmp_path / "result.json").exists()


def test_portfolio_manifest_tracks_each_contract_funding(tmp_path):
    (tmp_path / "BTCUSDT.csv").write_text("synthetic btc", encoding="utf-8")
    (tmp_path / "provenance.json").write_text("{}", encoding="utf-8")
    first = portfolio_cost_stress.portfolio_manifest(tmp_path)
    assert first["files"]["XAUUSDT-funding.csv"] is None
    (tmp_path / "XAUUSDT-funding.csv").write_text("synthetic funding", encoding="utf-8")
    second = portfolio_cost_stress.portfolio_manifest(tmp_path)
    assert second["files"]["XAUUSDT-funding.csv"]["sha256"]
    assert second != first
