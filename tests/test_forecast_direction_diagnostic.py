"""Synthetic fixtures only; these counts do not measure a real model."""
import pandas as pd
import pytest

from experiments.forecast_direction_diagnostic import evaluate


def _fixture():
    idx = pd.date_range("2026-06-13T00:00:00Z", "2026-11-10T23:00:00Z", freq="h")
    bars = pd.DataFrame({"close": 100.}, index=idx)
    decisions = pd.date_range("2026-06-15", "2026-11-09", freq="7D", tz="UTC")
    forecasts = []
    for i, t in enumerate(decisions):
        bars.loc[t - pd.Timedelta(hours=25), "close"] = 99.
        bars.loc[t + pd.Timedelta(hours=23), "close"] = 101.
        forecasts.append({"decision_utc": str(t), "last_close": 100., "actual_24h": 101.,
                          "median": 102. if i < 10 else 98. if i < 18 else 100.})
    report = {"sample_schedule": "fixed_mondays_2026_06_15_to_2026_11_09", "samples": 22,
              "symbol_label": "SYNTHETIC_FIXTURE", "forecasts": forecasts}
    return report, bars


def test_direction_counts_neutral_and_fixed_prior_momentum_separately():
    report, bars = _fixture()
    result = evaluate(report, bars)
    assert result["counts"]["model"] == {"correct": 10, "incorrect": 8, "neutral": 4}
    assert result["counts"]["prior_24h_momentum"] == {"correct": 22, "incorrect": 0, "neutral": 0}
    assert "not_a_trading_backtest" in result["status"]


def test_direction_diagnostic_rejects_changed_outcome_and_gaps():
    report, bars = _fixture()
    report["forecasts"][0]["actual_24h"] = 102.
    with pytest.raises(ValueError, match="does not match source bars"):
        evaluate(report, bars)
    report, bars = _fixture()
    with pytest.raises(ValueError, match="continuous"):
        evaluate(report, bars.iloc[:-2].drop(bars.index[25]))
