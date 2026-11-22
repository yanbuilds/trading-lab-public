import json
from dataclasses import asdict

import pandas as pd
import pytest

from trading_lab.perps import research
from trading_lab.perps.risk import Spec


def test_backtest_rejects_actual_missing_funding_even_if_metadata_claims_complete(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    artifact = tmp_path / "artifacts"
    artifact.mkdir()
    marker = artifact / "perps-report.json"
    marker.write_text("previous report", encoding="utf-8")
    symbol = "BTC/USDT:USDT"
    spec = Spec(symbol, .0001, .0001, 5., .0006, 1.)
    (cache / "provenance.json").write_text(json.dumps({"symbols": {symbol: {
        "spec": asdict(spec), "funding_interval_hours": 8, "funding_complete": True}}}))
    dates = pd.date_range("2026-01-01", periods=80, freq="h", tz="UTC")
    pd.DataFrame({"date": dates, "open": 100., "high": 101., "low": 99., "close": 100.,
                  "volume": 1.}).to_csv(cache / "BTCUSDT.csv", index=False)
    pd.DataFrame({"timestamp": [], "rate": []}).to_csv(cache / "BTCUSDT-funding.csv", index=False)
    monkeypatch.setattr(research, "CACHE", cache)
    monkeypatch.setattr(research, "ART", artifact)
    with pytest.raises(ValueError, match="funding missing"):
        research.backtest()
    assert marker.read_text(encoding="utf-8") == "previous report"
