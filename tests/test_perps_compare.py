from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_lab.perps.compare import (
    PREPARERS,
    REUSED_LAST_BAR,
    REUSED_START,
    attach_ema,
    attach_tsmom_3m,
    attach_turtle_s1,
    simulate_book,
)
from trading_lab.perps.risk import Spec

S = Spec("BTC/USDT:USDT", 0.0001, 0.0001, 5.0, 0.0006, 1.0)


def _h1(rows: int = 200, start="2026-06-01", close0=100.0, drift=0.0, noise=0.0, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=rows, freq="h", tz="UTC")
    close = close0 + np.cumsum(np.full(rows, drift) + (rng.normal(0, noise, rows) if noise else 0))
    close = np.maximum(close, 1.0)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.2
    low = np.minimum(open_, close) - 0.2
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1.0}, index=idx)


def test_prefix_indicators_ignore_future_prices():
    d = _h1(80 * 24, drift=0.01)
    cut = d.index[40 * 24]
    for fn in (attach_ema, attach_turtle_s1, attach_tsmom_3m):
        a = fn(d)
        perturbed = d.copy()
        later = perturbed.index > cut
        perturbed.loc[later, ["open", "high", "low", "close"]] = [50.0, 51.0, 49.0, 50.0]
        b = fn(perturbed)
        cols = [c for c in ("signal", "atr", "fast", "slow", "exit_long", "exit_short", "tsmom_hold") if c in a.columns]
        pre = a.index <= cut
        for c in cols:
            left, right = a.loc[pre, c], b.loc[pre, c]
            if left.dtype == bool or right.dtype == bool:
                pd.testing.assert_series_equal(left.astype(bool), right.astype(bool), check_names=False)
            else:
                pd.testing.assert_series_equal(left, right, check_names=False, check_exact=False, atol=1e-9, rtol=0)


@pytest.mark.parametrize("cutoff", ["2026-09-15 12:00", "2026-09-30 22:00", "2026-09-30 23:00"])
def test_tsmom_truncation_does_not_invent_month_end_or_partial_day_signals(cutoff):
    # Synthetic prices exceed the 90-day warmup. Prefixes end mid-month,
    # just before the last bar of a month, and at a true month-end close.
    bars = _h1(180 * 24, drift=0.02, noise=0.1, seed=3)
    cutoff = pd.Timestamp(cutoff, tz="UTC")
    full = attach_tsmom_3m(bars)
    prefix = attach_tsmom_3m(bars.loc[:cutoff])
    for column in ("signal", "atr", "tsmom_hold", "exit_long", "exit_short", "vol60"):
        pd.testing.assert_series_equal(prefix[column], full.loc[:cutoff, column], check_names=False)
    if cutoff.hour != 23:
        assert prefix.iloc[-1]["signal"] == 0


def test_simulate_book_matches_legacy_ema_on_locked_cache():
    cache = Path("data/perps")
    if not (cache / "BTCUSDT.csv").exists():
        pytest.skip("locked cache missing")
    from trading_lab.perps.compare import load_cache
    from trading_lab.perps.research import simulate as legacy_simulate
    frames, specs, funding, _ = load_cache(cache)
    legacy = legacy_simulate(frames, specs, funding)
    book = simulate_book({k: attach_ema(v) for k, v in frames.items()}, specs, funding, prepared=True)
    assert book["closed_trades"] == legacy["closed_trades"]
    assert book["return_pct"] == pytest.approx(legacy["return_pct"], abs=1e-8)
    assert book["max_drawdown_pct"] == pytest.approx(legacy["max_drawdown_pct"], abs=1e-8)
    assert book["fees"] == pytest.approx(legacy["fees"], abs=1e-8)
    assert book["reconciliation_error"] < 1e-7


def test_turtle_and_tsmom_run_and_reconcile():
    d = _h1(150 * 24, drift=0.02, noise=0.4, seed=2)
    frames = {k: d.copy() for k in ("BTC/USDT:USDT", "XAU/USDT:USDT", "NDX100/USDT:USDT")}
    # XAU min qty is larger; use BTC-like specs so the synthetic path can fill
    specs = {k: S for k in frames}
    start = d.index[80 * 24]
    for name, fn in PREPARERS.items():
        r = simulate_book({k: fn(v) for k, v in frames.items()}, specs, prepared=True, start=start)
        assert r["reconciliation_error"] < 1e-7
        assert r["start"] is not None


def test_windows_are_precommitted():
    assert REUSED_START == pd.Timestamp("2026-09-17 06:00:00+00:00")
    assert REUSED_LAST_BAR == pd.Timestamp("2026-11-16 05:00:00+00:00")
    assert REUSED_START > pd.Timestamp("2026-06-01 00:00:00+00:00")


def test_exposure_counts_hours_with_open_or_new_position_conservatively():
    d = _h1(70)
    d["atr"] = 1.0
    d["signal"] = 0
    d["exit_long"] = False
    d["exit_short"] = False
    d.loc[d.index[60], "signal"] = 1
    d.loc[d.index[62], "exit_long"] = True
    result = simulate_book({S.symbol: d}, {S.symbol: S}, capital=100, prepared=True,
                           start=d.index[61], end=d.index[64])
    assert result["bars_processed"] == 4
    assert result["bars_with_position"] == 3  # includes the bar exited at its open
    assert result["exposure_bar_fraction"] == pytest.approx(.75)
    assert result["peak_end_bar_notional_over_initial"] > 0
    assert result["mean_end_bar_notional_over_initial"] == pytest.approx(result["peak_end_bar_notional_over_initial"] / 2)


def test_simultaneous_signals_respect_explicit_priority_and_default():
    bars = _h1(70)
    bars["atr"] = 1.0
    bars["signal"] = 0
    bars["exit_long"] = False
    bars["exit_short"] = False
    bars.loc[bars.index[60], "signal"] = 1
    bars.loc[bars.index[62], "exit_long"] = True
    symbols = ("BTC/USDT:USDT", "NDX100/USDT:USDT")
    frames = {s: bars for s in symbols}
    specs = {s: S for s in symbols}
    kwargs = dict(capital=100, prepared=True, start=bars.index[61], end=bars.index[64])
    assert simulate_book(frames, specs, **kwargs)["trades"][0]["symbol"] == symbols[0]
    assert simulate_book(frames, specs, entry_priority=symbols[::-1], **kwargs)["trades"][0]["symbol"] == symbols[1]
    for priority in ([symbols[0]], [symbols[0], symbols[0]], [symbols[0], "OTHER"]):
        with pytest.raises(ValueError, match="permutation"):
            simulate_book(frames, specs, entry_priority=priority, **kwargs)


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("stop_kind", ["gap", "intrabar"])
def test_book_stop_costs_and_gap_precedence(side, stop_kind):
    """Synthetic full-book stops, including adverse fills and both fees."""
    from trading_lab.perps.risk import size_order
    bars = _h1(70)
    bars["atr"], bars["signal"] = 1.0, 0
    bars["exit_long"], bars["exit_short"] = False, False
    bars.loc[bars.index[60], "signal"] = side
    if stop_kind == "gap":
        # A same-open exit signal must not mask the worse opening gap.
        bars.loc[bars.index[61], "exit_long" if side == 1 else "exit_short"] = True
        open_price = 96.0 if side == 1 else 104.0
        bars.loc[bars.index[62], ["open", "high", "low", "close"]] = [open_price, open_price + .2, open_price - .2, open_price]
    else:
        bars.loc[bars.index[62], ["open", "high", "low", "close"]] = [100., 103., 97., 100.]
    slip = .0005
    result = simulate_book({S.symbol: bars}, {S.symbol: S}, capital=100, prepared=True,
                           start=bars.index[61], end=bars.index[62], slippage=slip)
    assert result["closed_trades"] == 1 and result["open_positions"] == 0
    trade = result["trades"][0]
    assert trade["exit_reason"] == ("gap_stop" if stop_kind == "gap" else "atr_stop")
    entry = 100. * (1 + side * slip)
    qty, reason = size_order(100, 100, entry, 2., S, leverage=2., slippage=slip)
    assert reason == "ok"
    reference_exit = open_price if stop_kind == "gap" else entry - side * 2.
    actual_exit = reference_exit * (1 - side * slip)
    fees = qty * S.contract_size * S.taker * (entry + actual_exit)
    expected_pnl = side * qty * S.contract_size * (actual_exit - entry) - fees
    assert trade["net_pnl"] == pytest.approx(expected_pnl)
    assert result["fees"] == pytest.approx(fees)
    assert result["final"] == pytest.approx(100 + expected_pnl)
    assert result["funding_net_paid"] is None


def test_intrabar_stop_cannot_fund_other_contract_at_same_open():
    """Synthetic cross-market cash ordering in the prospective account engine."""
    a, b = _h1(70), _h1(70)
    for frame in (a, b):
        frame["atr"], frame["signal"] = 1., 0
        frame["exit_long"], frame["exit_short"] = False, False
    a.loc[a.index[60], "signal"] = 1
    # A is open at hour 62; its stop occurs somewhere inside that hour.
    a.loc[a.index[62], ["open", "high", "low", "close"]] = [100., 101., 97., 100.]
    # B has a signal at 61 (attempt at 62) and again at 62 (attempt at 63).
    b.loc[b.index[61:63], "signal"] = 1
    b.loc[b.index[63], "exit_long"] = True
    names = ("BTC/USDT:USDT", "XAU/USDT:USDT")
    specs = {name: Spec(name, .0001, .0001, 5., .0006, 1.) for name in names}
    result = simulate_book({names[0]: a, names[1]: b}, specs, capital=100, prepared=True,
                           start=a.index[61], end=a.index[64])
    assert len(result["trades"]) == 2
    first, second = result["trades"]
    assert first["symbol"] == names[0] and first["exit_reason"] == "atr_stop"
    assert second["symbol"] == names[1]
    assert pd.Timestamp(second["entry_time"]) == a.index[63]
    assert pd.Timestamp(second["entry_time"]) > a.index[62]
    assert result["reconciliation_error"] < 1e-7


def test_simulation_reports_unchecked_funding_and_rejects_missing_when_intervals_supplied():
    d = _h1(80)
    start, end = d.index[64], d.index[79]
    unchecked = simulate_book({S.symbol: d}, {S.symbol: S}, start=start, end=end)
    assert unchecked["funding_coverage_checked"] is False
    assert unchecked["funding_net_paid"] is None
    assert unchecked["funding_applied_unverified"] == 0.0
    assert "not validated net" in unchecked["funding_coverage_note"]
    with pytest.raises(ValueError, match="incomplete or invalid funding"):
        simulate_book({S.symbol: d}, {S.symbol: S}, {S.symbol: []}, start=start, end=end,
                      funding_intervals={S.symbol: 8})


def test_equity_halt_not_tripped_by_unrealized_gain():
    # Flat warmup then a persistent rally so an EMA long can show unrealized gains.
    d = _h1(200, start="2026-09-17", close0=100.0, drift=0.0)
    d.loc[d.index[70]:, ["open", "high", "low", "close"]] = None
    close = 100.0 + (pd.Series(range(len(d)), index=d.index) - 70).clip(lower=0) * 0.8
    open_ = close.shift(1).fillna(100.0)
    d["close"] = close
    d["open"] = open_
    d["high"] = d[["open", "close"]].max(axis=1) + 0.2
    d["low"] = d[["open", "close"]].min(axis=1) - 0.2
    frames = {"BTC/USDT:USDT": attach_ema(d)}
    specs = {"BTC/USDT:USDT": S}
    pre = frames["BTC/USDT:USDT"]
    assert int((pre["signal"] != 0).sum()) >= 1
    legacy = simulate_book(frames, specs, prepared=True, start=d.index[60], halt_mode="legacy_cash")
    fixed = simulate_book(frames, specs, prepared=True, start=d.index[60], halt_mode="equity")
    if fixed["max_drawdown_pct"] < 2.9:
        assert fixed["loss_halt"] is False
    # The legacy cash check is known to latch after ~3% unrealized gains.
    if legacy["max_drawdown_pct"] < 2.9 and legacy.get("final", 0) > 30.9:
        assert legacy["loss_halt"] is True


def test_funding_completeness_rejects_missing_duplicate_and_invalid_rates():
    from trading_lab.perps.compare import funding_complete_between
    start = pd.Timestamp("2026-09-17T00:00:00Z")
    end = start + pd.Timedelta(hours=16)
    rows = [{"timestamp": int(t.timestamp() * 1000), "rate": 0.0}
            for t in pd.date_range(start, end, freq="8h")]
    check = lambda data: funding_complete_between({"BTC": data}, {"BTC": 8}, start, end)
    assert check(rows)["complete"]
    assert not check(rows[:-1])["complete"]
    assert "duplicate_funding_timestamp" in check(rows + [rows[0]])["invalid"]["BTC"]
    assert "malformed_timestamp_or_rate" in check(rows + [{**rows[0], "rate": float("nan")}])["invalid"]["BTC"]
    off_cycle = {"timestamp": int((start + pd.Timedelta(hours=4)).timestamp() * 1000), "rate": 0.001}
    assert "unexpected_funding_timestamp" in check(rows + [off_cycle])["invalid"]["BTC"]


def test_funding_common_bounds_are_intersection_not_union():
    from trading_lab.perps.compare import funding_complete_between
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    def row(hour):
        return {"timestamp": int((start + pd.Timedelta(hours=hour)).timestamp() * 1000), "rate": 0.001}
    funding = {"BTC": [row(h) for h in (0, 8, 16, 24)], "XAU": [row(h) for h in (8, 16, 24, 32)]}
    report = funding_complete_between(funding, {"BTC": 8, "XAU": 8}, start, start + pd.Timedelta(hours=32))
    assert report["coverage_start"] == str(start + pd.Timedelta(hours=8))
    assert report["coverage_end"] == str(start + pd.Timedelta(hours=24))
    assert not report["complete"] and "not proof" in report["coverage_note"]
    funding["XAU"] = []
    report = funding_complete_between(funding, {"BTC": 8, "XAU": 8}, start, start + pd.Timedelta(hours=32))
    assert report["coverage_start"] is None and report["coverage_end"] is None
    funding["XAU"] = [row(40)]
    report = funding_complete_between(funding, {"BTC": 8, "XAU": 8}, start, start + pd.Timedelta(hours=32))
    assert report["coverage_start"] is None and report["coverage_end"] is None


def test_comparison_withholds_returns_when_funding_is_missing(monkeypatch, tmp_path):
    from trading_lab.perps import compare
    bars = _h1(300, start="2026-09-17 06:00")
    symbols = ("BTC/USDT:USDT", "XAU/USDT:USDT", "NDX100/USDT:USDT")
    frames = {s: bars for s in symbols}
    specs = {s: S for s in symbols}
    funding = {s: [] for s in symbols}
    provenance = {"symbols": {s: {"funding_interval_hours": 8} for s in symbols}}
    monkeypatch.setattr(compare, "load_cache", lambda _: (frames, specs, funding, provenance))
    monkeypatch.setattr(compare, "first_common_defined", lambda *args: bars.index[0])
    monkeypatch.setattr(compare, "first_signal_time", lambda *args: bars.index[0])
    monkeypatch.setattr(compare, "simulate_book", lambda *args, **kwargs: pytest.fail("must not calculate returns with missing funding"))
    report = compare.run_comparison(tmp_path)
    window = report["windows"]["reused_exploratory"]
    assert not window["funding_coverage"]["complete"]
    assert window["candidates"] == {}
    assert "withheld" in window["status"]
    assert "funding incomplete" in report["ema_locked_cache_replication"]["error"]
