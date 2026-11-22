from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("piia")

from piia.intelligence.paper.replay import (
    ACTOR,
    LABEL,
    atr,
    ema,
    run_btc_historical_replay,
)
from trading_lab.perps.risk import stop_fill

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "perps" / "BTCUSDT.csv"


def _write_synth(path: Path, rows: int = 120, start="2026-01-01") -> Path:
    idx = pd.date_range(start, periods=rows, freq="h", tz="UTC")
    # Flat then a sharp rally (EMA20 crosses above), then a gap down through a long stop,
    # then a downward drift (cross short), then a gap up through the short stop.
    close = np.full(rows, 100.0)
    close[60:80] = np.linspace(100, 130, 20)
    close[80:] = 90.0
    close[100:] = 80.0
    close[110:] = 120.0
    open_ = np.r_[100.0, close[:-1]]
    # After the rally, bar 80 opens at 90 (gap through any ~100 stop for a long)
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    # Make the gap bars truly gapping: bar 80 high also stays at ~90
    df = pd.DataFrame({"date": idx, "open": open_, "high": high, "low": low, "close": close, "volume": 1.0})
    path.mkdir(parents=True, exist_ok=True)
    csv = path / "BTCUSDT.csv"
    df.to_csv(csv, index=False)
    # Synthetic funding only: complete assumed 8-hour grid, one nonzero fixture.
    settlements = [t for t in idx if t.hour % 8 == 0]
    pd.DataFrame({"timestamp": [int(t.timestamp() * 1000) for t in settlements],
                  "rate": [0.0001 if t == idx[72] else 0.0 for t in settlements]}).to_csv(path / "BTCUSDT-funding.csv", index=False)
    return path


def test_ema_atr_match_prefix_no_lookahead():
    x = np.linspace(10, 20, 40)
    h = x + 1
    l = x - 1
    e = ema(x, 10)
    a = atr(h, l, x, 5)
    # perturb a later point; earlier values must stay identical
    x2 = x.copy(); x2[-1] = 999
    h2 = x2 + 1; l2 = x2 - 1
    e2 = ema(x2, 10)
    a2 = atr(h2, l2, x2, 5)
    np.testing.assert_allclose(e[:-1], e2[:-1], equal_nan=True)
    np.testing.assert_allclose(a[:-1], a2[:-1], equal_nan=True)


def test_synthetic_fill_gap_stop_recon_and_unused_quote(tmp_path):
    data = _write_synth(tmp_path / "data")
    result = run_btc_historical_replay(
        db_path=tmp_path / "replay.sqlite",
        evidence_store=tmp_path / "ev",
        data_dir=data,
        starting_cash=30.0,
        current_quote=999999.0,
        run_recorded_at=datetime(2026, 11, 16, 8, tzinfo=timezone.utc),
    )
    r = result.report
    assert r["label"] == LABEL
    assert r["approval_actor"] == ACTOR
    assert r["not_user_approval"] is True
    assert r["independent_llm_review"] == "NOT_WIRED"
    assert r["current_quote_provided_but_unused"] == 999999.0
    assert r["held_out"] is False
    assert r["reconciliation_ok"] is True
    assert r["live_orders"] is False
    # At least one closed path should exercise the engine. If min-size rejects everything, still record reasons.
    if r["closed_trades"] == 0 and r["open_positions"] == 0:
        assert r["rejections"], "expected either fills or explicit rejection reasons"
        return
    assert r["closed_trades"] + r["open_positions"] >= 1
    closed = r["closed_trade_summaries"]
    # Gap fills must be no better than the stop (long: fill <= stop; short: fill >= stop)
    for t in closed:
        assert t["sandbox_approval_actor"] == ACTOR
        if t["exit_reason"] == "gap_stop":
            if t["side"] == "long":
                assert t["exit"] <= t["stop"] + 1e-9
            else:
                assert t["exit"] >= t["stop"] - 1e-9


def test_future_append_invariance(tmp_path):
    data = _write_synth(tmp_path / "data", rows=120)
    cutoff = datetime(2026, 1, 4, 0, tzinfo=timezone.utc)  # 72 hours after start
    a = run_btc_historical_replay(
        db_path=tmp_path / "a.sqlite", evidence_store=tmp_path / "eva",
        data_dir=data, end_time=cutoff,
        run_recorded_at=datetime(2026, 11, 16, 8, tzinfo=timezone.utc),
    )
    from piia.intelligence.paper.risk_file import load_owner_risk
    assert a.report["starting_cash"] == load_owner_risk()[0]["virtual_equity_usdt"]
    # Perturb only bars after the cutoff
    csv = data / "BTCUSDT.csv"
    df = pd.read_csv(csv)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    later = df["date"] > cutoff
    df.loc[later, ["open", "high", "low", "close"]] = [50.0, 51.0, 40.0, 41.0]
    df.to_csv(csv, index=False)
    b = run_btc_historical_replay(
        db_path=tmp_path / "b.sqlite", evidence_store=tmp_path / "evb",
        data_dir=data,
        run_recorded_at=datetime(2026, 11, 16, 8, tzinfo=timezone.utc),
    )
    def key(t):
        return (t.side, t.signal_bar_open, t.entry_time, round(t.entry, 8), round(t.qty, 8), round(t.stop, 8))
    prefix_entries = [key(t) for t in a.trades if t.entry_time <= cutoff.isoformat()]
    full_entries = [key(t) for t in b.trades if t.entry_time <= cutoff.isoformat()]
    assert prefix_entries == full_entries
    for ta in a.trades:
        if ta.exit_time and ta.exit_time <= (cutoff.isoformat()):
            tb = next(x for x in b.trades if key(x) == key(ta))
            assert ta.exit_reason == tb.exit_reason
            assert ta.exit_time == tb.exit_time
            assert ta.net_usd == pytest.approx(tb.net_usd, abs=1e-9)


def test_stop_fill_gap_is_unfavorable():
    # long gap below stop uses the lower of stop and open
    assert stop_fill(1, 100, 90, 80, 91) == 90
    # short gap above stop uses the higher of stop and open
    assert stop_fill(-1, 100, 110, 99, 115) == 110


def test_missing_funding_rejects_replay_without_touching_database(tmp_path):
    data = _write_synth(tmp_path / "data")
    funding = data / "BTCUSDT-funding.csv"
    rows = pd.read_csv(funding)
    rows = rows.iloc[:-1]
    rows.to_csv(funding, index=False)
    db = tmp_path / "replay.sqlite"
    with pytest.raises(ValueError, match="funding data missing, incomplete or invalid"):
        run_btc_historical_replay(db_path=db, evidence_store=tmp_path / "ev", data_dir=data)
    assert not db.exists()


def test_existing_replay_database_is_never_replaced(tmp_path):
    data = _write_synth(tmp_path / "data")
    db = tmp_path / "existing.sqlite"
    db.write_bytes(b"existing user data")
    with pytest.raises(FileExistsError, match="already exists"):
        run_btc_historical_replay(db_path=db, evidence_store=tmp_path / "ev", data_dir=data)
    assert db.read_bytes() == b"existing user data"


@pytest.mark.skipif(not CACHE.exists(), reason="locked BTC cache not present")
def test_real_cache_replay_isolated_and_no_duplicate_positions(tmp_path):
    result = run_btc_historical_replay(
        db_path=tmp_path / "real.sqlite",
        evidence_store=tmp_path / "ev",
        data_dir=CACHE.parent,
        current_quote=1.0,  # would wreck sizing if used
        run_recorded_at=datetime(2026, 11, 16, 8, tzinfo=timezone.utc),
    )
    r = result.report
    assert r["reconciliation_ok"] is True
    assert r["current_quote_provided_but_unused"] == 1.0
    ids = [t.position_id for t in result.trades]
    assert len(ids) == len(set(ids))
    assert r["dataset"]["last"].startswith("2026-11-16")
    # Margin reserve must not be mistaken for a 3% daily equity loss.
    if r["max_drawdown_pct"] < 2.9:
        assert r["loss_halt"] is False
    assert r["signals_eligible"] >= 1

    # must not invent a contemporaneous publication
    for t in result.trades:
        assert t.sandbox_approval_actor == ACTOR
