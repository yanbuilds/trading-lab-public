"""Synthetic fixture: checks causal windows, not model or market performance."""
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("chronos", reason="chronos is isolated to forecast-env, not piia-env; run this file with ../forecast-env/bin/python -m pytest tests/test_chronos_weekly_probe.py")
torch = pytest.importorskip("torch", reason="torch is isolated to forecast-env, not piia-env")

from experiments import chronos_probe


def test_weekly_probe_uses_only_predecision_history(tmp_path, monkeypatch):
    idx = pd.date_range("2025-12-01T00:00:00Z", "2026-11-10T23:00:00Z", freq="h", tz="UTC")
    prices = np.arange(len(idx), dtype=float) + 1000.
    bars = tmp_path / "synthetic_fixture_bars.csv"
    pd.DataFrame({"date": idx, "close": prices}).to_csv(bars, index=False)
    model = tmp_path / "synthetic_fixture_model"
    model.mkdir()
    weights = b"explicit synthetic test fixture; not model weights"
    (model / "model.safetensors").write_bytes(weights)
    monkeypatch.setattr(chronos_probe, "EXPECTED_SHA256", hashlib.sha256(weights).hexdigest())
    captured = []
    class FakePipeline:
        def predict_quantiles(self, series, prediction_length, quantile_levels, batch_size):
            assert prediction_length == 24 and quantile_levels == [.1, .5, .9]
            captured.extend(series)
            return [torch.tensor(np.tile([x[-1] - 1, x[-1], x[-1] + 1], (24, 1))) for x in series], None
    monkeypatch.setattr(chronos_probe.BaseChronosPipeline, "from_pretrained", lambda *a, **kw: FakePipeline())
    out = tmp_path / "synthetic_fixture_result.json"
    monkeypatch.setattr(sys, "argv", ["probe", "--model", str(model), "--bars", str(bars),
                                      "--weekly-fixed", "--out", str(out)])
    chronos_probe.main()
    result = json.loads(out.read_text())
    assert result["samples"] == 45
    assert len(captured) == 45 and all(len(series) == 512 for series in captured)
    for row, series in zip(result["forecasts"], captured, strict=True):
        decision = pd.Timestamp(row["decision_utc"])
        expected_last = prices[idx.get_loc(decision - pd.Timedelta(hours=1))]
        assert series[-1] == expected_last
        assert row["actual_24h"] == prices[idx.get_loc(decision + pd.Timedelta(hours=23))]
    captured.clear()
    out_common = tmp_path / "synthetic_fixture_common.json"
    monkeypatch.setattr(sys, "argv", ["probe", "--model", str(model), "--bars", str(bars),
                                      "--weekly-common", "--symbol", "XAUUSDT", "--out", str(out_common)])
    chronos_probe.main()
    common = json.loads(out_common.read_text())
    assert common["symbol_label"] == "XAUUSDT"
    assert common["samples"] == 22 and len(captured) == 22
    assert common["forecasts"][0]["decision_utc"].startswith("2026-06-15")
    assert common["forecasts"][-1]["decision_utc"].startswith("2026-11-09")
