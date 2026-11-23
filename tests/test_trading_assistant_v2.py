"""Comprehensive tests for Trading Assistant v2, Indicators v2, Adaptive Strategy v2,
Risk Ticket v2, Causal Walk-Forward v2, CLI, FastAPI endpoints, and Streamlit UI."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from trading_lab.indicators_v2 import (
    adx_di,
    atr,
    bollinger_bands,
    chandelier_stops,
    donchian_channels,
    ema,
    rsi,
)
from trading_lab.perps.adaptive_v2 import (
    attach_adaptive_v2,
    latest_bar_signal_summary,
    simulate_adaptive_v2,
)
from trading_lab.perps.risk_v2 import (
    compute_trade_ticket,
    fee_adjusted_breakeven,
    load_policy_v2,
)
from trading_lab.perps.walk_forward import run_walk_forward
from piia.intelligence.trade_assistant import (
    advise_open_position,
    build_trade_plan,
    format_scan_table,
    generate_demo_snapshot,
    scan_market_signals,
)


def _make_trending_h1(bars: int = 320, direction: int = 1, base: float = 50000.0) -> pd.DataFrame:
    idx = pd.date_range("2026-06-01", periods=bars, freq="1h", tz="UTC")
    rng = np.random.default_rng(42)
    drift = 0.0022 * direction
    rets = drift + rng.normal(0.0, 0.0012, size=bars)
    rets[-1] = 0.015 * direction  # clean breakout bar
    closes = base * np.cumprod(1.0 + rets)
    opens = np.roll(closes, 1)
    opens[0] = base
    span = np.abs(closes - opens) + closes * 0.0015
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) + 0.4 * span,
            "low": np.minimum(opens, closes) - 0.4 * span,
            "close": closes,
            "volume": 250.0,
        },
        index=idx,
    )


def test_pure_indicators_shapes_and_causality():
    df = _make_trending_h1(260, direction=1)
    c = df["close"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()

    e20 = ema(c, 20)
    a14 = atr(h, l, c, 14)
    r14 = rsi(c, 14)
    adx_v, pdi_v, mdi_v = adx_di(h, l, c, 14)
    mid, up, lo, bw, pctb = bollinger_bands(c, 20, 2.0)
    d_hi, d_lo = donchian_channels(h, l, 20, shift_bars=1)
    ch_l, ch_s = chandelier_stops(h, l, a14, 22, 2.5, shift_bars=1)

    assert np.isnan(e20[:19]).all() and np.isfinite(e20[19:]).all()
    assert np.isnan(a14[:14]).all() and np.isfinite(a14[14:]).all()
    assert ((r14[14:] >= 0) & (r14[14:] <= 100)).all()
    assert np.isfinite(adx_v[-1]) and pdi_v[-1] > mdi_v[-1]
    assert up[-1] > mid[-1] > lo[-1] and bw[-1] > 0 and np.isfinite(pctb[-1])
    # Causal check: modifying the final bar must NOT alter prior bars' indicators or shifted channels
    df2 = df.copy()
    df2.iloc[-1, df2.columns.get_loc("high")] *= 5.0
    df2.iloc[-1, df2.columns.get_loc("close")] *= 5.0
    e20_2 = ema(df2["close"].to_numpy(), 20)
    d_hi_2, _ = donchian_channels(df2["high"].to_numpy(), df2["low"].to_numpy(), 20, shift_bars=1)
    np.testing.assert_allclose(e20[:-1], e20_2[:-1])
    # Because shift_bars=1 uses strictly prior bars, even index -1 of d_hi is unchanged!
    assert d_hi[-1] == pytest.approx(d_hi_2[-1])
    assert np.isfinite(ch_l[-1]) and np.isfinite(ch_s[-1])


def test_fee_adjusted_breakeven_exact_zero_net_pnl():
    entry = 60000.0
    efee, xfee = 0.0002, 0.0006
    be_long = fee_adjusted_breakeven(entry, "long", entry_fee_rate=efee, exit_fee_rate=xfee)
    net_long = (be_long - entry) - entry * efee - be_long * xfee
    assert abs(net_long) < 1e-9 and be_long > entry

    be_short = fee_adjusted_breakeven(entry, "short", entry_fee_rate=efee, exit_fee_rate=xfee)
    net_short = (entry - be_short) - entry * efee - be_short * xfee
    assert abs(net_short) < 1e-9 and be_short < entry


def test_trade_ticket_v2_sizing_tp_ladder_and_liquidation_guard():
    policy = load_policy_v2()
    ticket = compute_trade_ticket(
        symbol="BTCUSDT",
        side="long",
        equity_usdt=1000.0,
        entry_price=65000.0,
        stop_loss=63700.0,
        take_profit=68250.0,
        leverage=3.0,
        risk_fraction=0.01,
        order_type="limit",
        funding_rate=0.0001,
        orderbook={
            "ok": True,
            "mid": "65000",
            "spread": "1.0",
            "asks": [{"price": "65000.5", "size": "2.0"}],
            "bids": [{"price": "64999.5", "size": "2.0"}],
        },
        policy=policy,
    )
    assert ticket["verdict"] == "READY_FOR_MANUAL_REVIEW"
    assert ticket["actionable_for_manual_review"] is True
    assert ticket["sizing"]["qty_contracts"] > 0
    assert ticket["risk_and_costs"]["estimated_stop_loss_usdt"] <= 10.0 + 1e-6
    assert len(ticket["take_profit_ladder"]) == 3
    assert ticket["take_profit_custom"]["net_reward_risk"] > 2.0
    assert ticket["orderbook_execution"]["ok"] is True

    # Stop beyond estimated liquidation at high leverage must REJECT
    bad = compute_trade_ticket(
        symbol="BTCUSDT",
        side="long",
        equity_usdt=1000.0,
        entry_price=65000.0,
        stop_loss=50000.0,  # -23% stop on 10x isolated leverage -> liquidates around -9.5%!
        leverage=10.0,
        risk_fraction=0.01,
        policy=policy,
    )
    assert bad["verdict"] == "REJECTED"
    assert "stop_at_or_beyond_estimated_liquidation" in bad["reasons"]


def test_adaptive_v2_prefix_invariance_and_simulation():
    up_df = _make_trending_h1(300, direction=1, base=60000.0)
    dn_df = _make_trending_h1(300, direction=-1, base=25000.0)

    full = attach_adaptive_v2(up_df)
    prefix = attach_adaptive_v2(up_df.iloc[:250])
    pd.testing.assert_series_equal(full["signal"].iloc[:250], prefix["signal"])
    pd.testing.assert_series_equal(full["regime"].iloc[:250], prefix["regime"])

    summary = latest_bar_signal_summary(up_df)
    assert summary["ok"] is True
    assert summary["regime"] == "trend_up"
    assert summary["signal"] == "LONG"

    sim = simulate_adaptive_v2({"BTCUSDT": up_df, "NDX100USDT": dn_df}, capital=500.0, leverage=2.0)
    assert sim["policy_id"] == "regime_adaptive_v2"
    assert sim["final"] > 0
    assert sim["closed_trades"] >= 1 or sim["open_positions"] == 1


def test_causal_walk_forward_v2_runs_without_lookahead():
    frames = {
        "BTCUSDT": _make_trending_h1(720, direction=1, base=60000.0),
        "NDX100USDT": _make_trending_h1(720, direction=-1, base=25000.0),
    }
    wf1 = run_walk_forward(frames, warmup_bars=200, fold_bars=200, step_bars=200, capital=500.0)
    assert wf1["folds_count"] == 2

    # Mutate data strictly inside fold 2 (bars 400+) and verify fold 1 result is 100% identical!
    mutated = {k: v.copy() for k, v in frames.items()}
    mutated["BTCUSDT"].iloc[450:, mutated["BTCUSDT"].columns.get_loc("close")] *= 0.5
    mutated["BTCUSDT"].iloc[450:, mutated["BTCUSDT"].columns.get_loc("low")] *= 0.4
    wf2 = run_walk_forward(mutated, warmup_bars=200, fold_bars=200, step_bars=200, capital=500.0)
    assert wf1["folds"][0]["adaptive_v2"] == wf2["folds"][0]["adaptive_v2"]


def test_scan_advise_cli_and_fastapi_routes():
    scan = scan_market_signals(demo_mode=True, equity_usdt=600.0, leverage=2.0, risk_fraction=0.01)
    assert scan["is_synthetic_demo"] is True
    assert set(scan["instruments"]) == {"BTCUSDT", "XAUUSDT", "NDX100USDT"}
    table = format_scan_table(scan)
    assert "BTCUSDT" in table and "XAUUSDT" in table

    advice = advise_open_position(
        symbol="BTCUSDT",
        side="long",
        entry_price=60000.0,
        qty=0.05,
        stop_loss=58800.0,
        leverage=2.0,
        mark_price=62000.0,  # > +1.5R (+2000 vs 1200 risk = +1.67R)
        candles_1h=generate_demo_snapshot()["instruments"]["BTCUSDT"]["candles_closed_1h"],
    )
    assert advice["pnl"]["current_r_multiple"] > 1.5
    assert advice["risk_levels"]["recommended_stop_loss"] > 60000.0

    # CLI test
    from piia.intelligence.cli import main as piia_main
    assert piia_main(["scan", "--demo", "--equity", "500"]) == 0
    assert piia_main([
        "plan-trade", "--symbol", "BTCUSDT", "--side", "long",
        "--equity", "500", "--entry", "64000", "--stop", "62800", "--tp", "67000",
    ]) == 0
    assert piia_main([
        "position-advice", "--symbol", "BTCUSDT", "--side", "long",
        "--entry", "64000", "--qty", "0.02", "--stop", "62800", "--mark", "66000",
    ]) == 0

    # FastAPI test
    from fastapi.testclient import TestClient
    from piia.intelligence.api import app
    with TestClient(app) as client:
        r_scan = client.get("/assistant/scan?demo=true&equity_usdt=500")
        assert r_scan.status_code == 200
        assert r_scan.json()["is_synthetic_demo"] is True

        r_plan = client.post("/assistant/plan-trade", json={
            "symbol": "BTCUSDT", "side": "long", "equity_usdt": 500.0,
            "entry_price": 64000.0, "stop_loss": 62800.0, "take_profit": 67000.0,
        })
        assert r_plan.status_code == 200
        assert r_plan.json()["actionable_for_manual_review"] is True

        r_pos = client.post("/assistant/position-advice", json={
            "symbol": "BTCUSDT", "side": "long", "entry_price": 64000.0,
            "qty": 0.02, "stop_loss": 62800.0, "mark_price": 66000.0,
        })
        assert r_pos.status_code == 200
        assert r_pos.json()["pnl"]["net_unrealized_after_fees_usdt"] > 0


def test_streamlit_v2_cockpit_renders_and_calculates():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    script = Path(__file__).resolve().parents[1] / "app" / "piia_app.py"
    at = AppTest.from_file(str(script), default_timeout=30).run()
    assert not at.exception

    # Click demo scan
    at.button(key="ta_btn_demo_scan").click()
    at.run()
    assert not at.exception
    assert at.session_state["ta_scan_result"]["is_synthetic_demo"] is True

    # Click Trade Ticket calculation
    at.button(key="ta_btn_calc_ticket").click()
    at.run()
    assert not at.exception
    assert at.session_state["ta_current_ticket"]["actionable_for_manual_review"] is True

    # Click Position Monitor diagnosis
    at.button(key="ta_btn_pos_advice").click()
    at.run()
    assert not at.exception
    assert at.session_state["ta_pos_advice"]["order_permissions"] is False
