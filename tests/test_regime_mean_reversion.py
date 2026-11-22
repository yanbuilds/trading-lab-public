"""Synthetic tests for the frozen regime labels and mean-reversion candidate."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from trading_lab.perps.compare import simulate_book
from trading_lab.perps.mean_reversion import attach_range_bollinger
from trading_lab.perps.regime import (
    ADX_RANGE_BELOW, ADX_TREND_AT_OR_ABOVE, classify_bar, classify_regime,
    completed_4h_emas, load_manifest,
)
from trading_lab.perps.risk import Spec


MANIFEST = Path("config/piia_regime_mean_reversion_v1.yaml")


def _bars(n=800, start="2025-01-01", noise=0.2, drift=0.01):
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    close = 100 + np.arange(n) * drift + noise * np.sin(np.arange(n) / 8)
    return pd.DataFrame({"open": close, "high": close + 0.8, "low": close - 0.8,
                         "close": close, "volume": 1.0}, index=idx)


def test_manifest_is_frozen_and_not_optimized():
    data = load_manifest(MANIFEST)
    assert data["hypothesis_optimized"] is False
    assert data["held_out"] is False
    assert data["capital_primary_usdt"] == 100
    assert data["capital_optional_feasibility_usdt"] == 30
    assert data["time_stop_hours"] == 24
    assert data["mean_reversion"]["bollinger_period"] == 20
    raw = yaml.safe_load(MANIFEST.read_text())
    assert raw == data


@pytest.mark.parametrize("section,key,value", [
    (None, "time_stop_hours", 48), (None, "risk_fraction", .02),
    (None, "order_permissions", True), (None, "max_open_positions", 2),
    ("regime", "adx_range_below", 25), ("regime", "partial_4h_excluded", False),
    ("mean_reversion", "bollinger_std", 3.0), ("mean_reversion", "eligible_regime", "trend"),
    ("mean_reversion", "entry", "different entry rule"),
])
def test_changed_frozen_parameters_fail_closed(tmp_path, section, key, value):
    data = yaml.safe_load(MANIFEST.read_text())
    (data if section is None else data[section])[key] = value
    path = tmp_path / "altered_fixture.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen manifest mismatch"):
        load_manifest(path)


def test_manifest_identifies_each_market_and_candidate_code(tmp_path):
    from experiments.regime_mean_reversion import complete_manifest
    (tmp_path / "BTCUSDT.csv").write_text("synthetic", encoding="utf-8")
    (tmp_path / "provenance.json").write_text("{}", encoding="utf-8")
    first = complete_manifest(tmp_path)
    assert first["files"]["XAUUSDT.csv"] is None
    assert first["files"][str(Path("src/trading_lab/perps/mean_reversion.py"))]["sha256"]
    (tmp_path / "XAUUSDT.csv").write_text("synthetic gold", encoding="utf-8")
    assert complete_manifest(tmp_path) != first


def test_completed_4h_excludes_partial_current_bar():
    bars = _bars(n=900)
    # 02:00 is inside the 00:00-04:00 bin, which must stay incomplete.
    cut = bars.index[(bars.index.hour == 2)][20]
    prefix = completed_4h_emas(bars.loc[:cut])
    changed = bars.copy()
    # Mutate the remainder of the unfinished 4h bar and everything after it.
    changed.loc[changed.index > cut, ["open", "high", "low", "close"]] *= 4
    revised = completed_4h_emas(changed)
    pd.testing.assert_series_equal(prefix.loc[:cut, "trend_4h_fast"], revised.loc[:cut, "trend_4h_fast"])
    pd.testing.assert_series_equal(prefix.loc[:cut, "trend_4h_slow"], revised.loc[:cut, "trend_4h_slow"])
    assert prefix.loc[cut, "trend_4h_fast"] == prefix.loc[cut - pd.Timedelta(hours=2), "trend_4h_fast"]


def test_regime_and_signal_are_causal_under_future_price_changes():
    bars = _bars()
    cut = bars.index[600]
    original = classify_regime(bars)
    changed = bars.copy()
    changed.loc[changed.index > cut, ["open", "high", "low", "close"]] *= 3
    revised = classify_regime(changed)
    cols = ["regime", "regime_reason", "trend_4h_fast", "trend_4h_slow", "adx14"]
    pd.testing.assert_frame_equal(original.loc[:cut, cols], revised.loc[:cut, cols])
    sig_a = attach_range_bollinger(bars)
    sig_b = attach_range_bollinger(changed)
    pd.testing.assert_series_equal(sig_a.loc[:cut, "signal"], sig_b.loc[:cut, "signal"])


def test_classify_bar_order_is_frozen():
    assert classify_bar(adx=None, atr=1, atr_mean=1, ema_fast=2, ema_slow=1, htf_fast=2, htf_slow=1)[0] == "insufficient_data"
    assert classify_bar(adx=10, atr=4, atr_mean=1, ema_fast=2, ema_slow=1, htf_fast=2, htf_slow=1)[0] == "extreme_vol"
    assert classify_bar(adx=ADX_RANGE_BELOW - 1, atr=1, atr_mean=1, ema_fast=2, ema_slow=1, htf_fast=2, htf_slow=1)[0] == "range"
    assert classify_bar(adx=ADX_TREND_AT_OR_ABOVE, atr=1, atr_mean=1, ema_fast=2, ema_slow=1, htf_fast=2, htf_slow=1)[0] == "trend"
    assert classify_bar(adx=ADX_TREND_AT_OR_ABOVE, atr=1, atr_mean=1, ema_fast=2, ema_slow=1, htf_fast=1, htf_slow=2)[0] == "uncertain"
    assert classify_bar(adx=22, atr=1, atr_mean=1, ema_fast=2, ema_slow=1, htf_fast=2, htf_slow=1)[0] == "uncertain"


def test_mean_reversion_has_exits_and_never_fades_completed_4h_slope():
    bars = _bars(n=1200, drift=0.0, noise=1.5)
    prepared = attach_range_bollinger(bars)
    entries = prepared.loc[prepared["signal"] != 0]
    if not entries.empty:
        longs = entries[entries["signal"] == 1]
        shorts = entries[entries["signal"] == -1]
        assert (longs["regime"] == "range").all()
        assert (shorts["regime"] == "range").all()
        assert (longs["trend_4h_fast"] > longs["trend_4h_slow"]).all()
        assert (shorts["trend_4h_fast"] < shorts["trend_4h_slow"]).all()
    assert prepared["exit_long"].notna().any()
    assert prepared["atr"].notna().any()


def test_time_stop_and_default_book_unchanged(tmp_path):
    idx = pd.date_range("2026-01-01", periods=80, freq="h", tz="UTC")
    close = np.full(len(idx), 100.0)
    frame = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                          "close": close, "volume": 1.0, "atr": 1.0, "signal": 0,
                          "exit_long": False, "exit_short": False,
                          "trail_long": np.nan, "trail_short": np.nan}, index=idx)
    frame.loc[idx[10], "signal"] = 1
    spec = {"BTC/USDT:USDT": Spec("BTC/USDT:USDT", 0.0001, 0.0001, 5, 0.0006, 1)}
    funding = {"BTC/USDT:USDT": [
        {"timestamp": int(t.timestamp() * 1000), "rate": 0.0}
        for t in idx if t.hour % 8 == 0
    ]}
    intervals = {"BTC/USDT:USDT": 8}
    start, end = idx[5], idx[-2]
    base = simulate_book({"BTC/USDT:USDT": frame}, spec, funding, capital=100, start=start, end=end,
                         prepared=True, funding_intervals=intervals)
    again = simulate_book({"BTC/USDT:USDT": frame}, spec, funding, capital=100, start=start, end=end,
                          prepared=True, funding_intervals=intervals)
    assert base["return_pct"] == again["return_pct"]
    assert base["closed_trades"] == again["closed_trades"]
    timed = simulate_book({"BTC/USDT:USDT": frame}, spec, funding, capital=100, start=start, end=end,
                          prepared=True, funding_intervals=intervals, max_hold_hours=24)
    reasons = [t["exit_reason"] for t in timed["trades"]]
    assert "time_stop" in reasons
    assert timed["funding_coverage_checked"] is True
    assert timed["reconciliation_error"] < 1e-7


def test_minimum_order_is_rejected_never_upsized():
    idx = pd.date_range("2026-01-01", periods=80, freq="h", tz="UTC")
    close = np.full(len(idx), 100.0)
    frame = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                          "close": close, "volume": 1.0, "atr": 1.0, "signal": 0,
                          "exit_long": False, "exit_short": False,
                          "trail_long": np.nan, "trail_short": np.nan}, index=idx)
    frame.loc[idx[10], "signal"] = 1
    # Huge minimum quantity cannot be met by the 1% risk budget.
    spec = {"BTC/USDT:USDT": Spec("BTC/USDT:USDT", 10, 1, 5, 0.0006, 1)}
    funding = {"BTC/USDT:USDT": [{"timestamp": int(t.timestamp() * 1000), "rate": 0.0}
                                  for t in idx if t.hour % 8 == 0]}
    result = simulate_book({"BTC/USDT:USDT": frame}, spec, funding, capital=100,
                           start=idx[5], end=idx[-2], prepared=True,
                           funding_intervals={"BTC/USDT:USDT": 8})
    assert result["closed_trades"] == 0
    assert result["rejections"].get("minimum_order_exceeds_budget", 0) >= 1


def test_incomplete_funding_is_refused():
    idx = pd.date_range("2026-01-01", periods=80, freq="h", tz="UTC")
    close = np.full(len(idx), 100.0)
    frame = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                          "close": close, "volume": 1.0, "atr": 1.0, "signal": 0,
                          "exit_long": False, "exit_short": False,
                          "trail_long": np.nan, "trail_short": np.nan}, index=idx)
    spec = {"BTC/USDT:USDT": Spec("BTC/USDT:USDT", 0.0001, 0.0001, 5, 0.0006, 1)}
    with pytest.raises(ValueError, match="incomplete or invalid funding"):
        simulate_book({"BTC/USDT:USDT": frame}, spec, {"BTC/USDT:USDT": []}, capital=100,
                      start=idx[2], end=idx[-2], prepared=True,
                      funding_intervals={"BTC/USDT:USDT": 8})
