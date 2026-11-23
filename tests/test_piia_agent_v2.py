"""Unit & integration tests for PIIA v2 (Synthetic fixtures only, zero live network)."""
from __future__ import annotations

from datetime import datetime, timezone
import sqlite3

import numpy as np
import pandas as pd
import pytest

from piia.intelligence.agent_v2.ablation import (
    ARMS_ORDER,
    audit_prefix_lookahead,
    run_ablation_tournament,
)
from piia.intelligence.agent_v2.continuous_paper import (
    _append_event,
    _init_db,
    _reconcile_open_position_with_bars,
    _reconstruct_book_state,
    approve_human_proposal,
    get_continuous_paper_status,
    run_prospective_cycle,
)
from piia.intelligence.agent_v2.manifest import verify_piia_v2_frozen
from piia.intelligence.agent_v2.pillars import (
    SYMBOLS,
    build_bar_evidence_for_symbol,
    load_v2_config,
    point_in_time_macro_context,
    prepare_market_feature_table,
)
from piia.intelligence.agent_v2.risk_gate import run_symbol_decision_trace
from piia.intelligence.agent_v2.roles import (
    run_independent_critic_pass1,
    run_independent_researcher,
)


def _synthetic_3contract_frames(bars: int = 500) -> dict[str, pd.DataFrame]:
    idx = pd.date_range("2026-10-01", periods=bars, freq="1h", tz="UTC")
    out = {}
    for seed, sym, base, drift in (
        (101, "BTCUSDT", 65000.0, 0.0012),
        (202, "XAUUSDT", 3850.0, 0.0002),
        (303, "NDX100USDT", 27500.0, -0.0009),
    ):
        rng = np.random.default_rng(seed)
        # Create structured regimes with clear breakouts and volatility shifts
        rets = drift + np.sin(np.linspace(0, 8 * np.pi, bars)) * 0.002 + rng.normal(0.0, 0.0022, size=bars)
        closes = base * np.cumprod(1.0 + rets)
        opens = np.roll(closes, 1)
        opens[0] = base
        span = np.abs(closes - opens) + closes * 0.0018
        out[sym] = pd.DataFrame(
            {
                "open": opens,
                "high": np.maximum(opens, closes) + 0.5 * span,
                "low": np.minimum(opens, closes) - 0.5 * span,
                "close": closes,
                "volume": 150.0,
            },
            index=idx,
        )
    return out


def test_frozen_manifest_matches_committed_v2_sources():
    res = verify_piia_v2_frozen()
    assert res["ok"] is True, f"Manifest mismatch: {res['issues']}"


def test_point_in_time_macro_strictly_blocks_future_or_unlagged_observations():
    obs = {
        "DFII10": [
            {"observation_date": "2026-11-05", "value": 1.80},
            {"observation_date": "2026-11-12", "value": 1.82},
            {"observation_date": "2026-11-13", "value": 1.84},
            {"observation_date": "2026-11-14", "value": 1.85},
            {"observation_date": "2026-11-16", "value": 1.88},
            {"observation_date": "2026-11-17", "value": 1.98},  # +0.18 jump vs 5 obs ago
            {"observation_date": "2026-11-19", "value": 2.25},  # future relative to 11-19T12:00Z!
        ],
        "DTWEXBGS": [
            {"observation_date": "2026-11-05", "value": 124.0},
            {"observation_date": "2026-11-12", "value": 124.2},
            {"observation_date": "2026-11-13", "value": 124.4},
            {"observation_date": "2026-11-14", "value": 124.5},
            {"observation_date": "2026-11-16", "value": 124.7},
            {"observation_date": "2026-11-17", "value": 125.5},  # +1.21% jump
            {"observation_date": "2026-11-19", "value": 130.0},  # future!
        ],
    }
    # At 2026-11-19 12:00 UTC, 2026-11-19 obs (+2d = 11-21) MUST NOT be used; 2026-11-17 (+2d = 11-19 00:00Z) IS usable!
    ctx = point_in_time_macro_context(obs, datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc))
    assert ctx["status"] == "available"
    assert ctx["series"]["DFII10"]["observation_date"] == "2026-11-17"
    assert ctx["macro_tightening_headwind"] is True


def test_role_context_isolation_and_critic_veto_trace():
    frames = _synthetic_3contract_frames(320)
    # Force strong uptrend + breakout on BTCUSDT last bar
    df = frames["BTCUSDT"].copy()
    anchor = float(df.iloc[249]["close"])
    for k in range(250, 320):
        df.iloc[k, df.columns.get_loc("close")] = anchor * (1.0 + 0.0025 * (k - 249))
        df.iloc[k, df.columns.get_loc("open")] = df.iloc[k - 1]["close"]
        df.iloc[k, df.columns.get_loc("high")] = df.iloc[k]["close"] * 1.0015
        df.iloc[k, df.columns.get_loc("low")] = df.iloc[k]["open"] * 0.999
    df.iloc[-1, df.columns.get_loc("close")] = float(df.iloc[-2]["high"]) * 1.008
    df.iloc[-1, df.columns.get_loc("high")] = df.iloc[-1]["close"] * 1.001

    prep = prepare_market_feature_table(df)
    bundle = build_bar_evidence_for_symbol(
        "BTCUSDT",
        feature_row=prep.iloc[-1],
        prev_close=float(prep.iloc[-2]["close"]),
        decision_utc=datetime(2026, 11, 18, 0, 0, tzinfo=timezone.utc),
        funding_rate_8h=0.0005,  # > 0.0003 crowding threshold -> Critic must veto!
        macro_ctx=None,
    )

    # Verify strict context isolation rejects contaminated bundles
    with pytest.raises(ValueError, match="Context isolation"):
        run_independent_researcher({**bundle, "critic_pass1": {}})
    with pytest.raises(ValueError, match="Context isolation"):
        run_independent_critic_pass1({**bundle, "researcher": {}})

    # Without Critic veto (M1), trade passes Risk Gate
    tr_no_veto = run_symbol_decision_trace(
        bundle, equity_usdt=500.0, free_cash_usdt=500.0,
        enable_macro=False, enable_forecast=False, enable_critic_veto=False, enable_live_external=False,
    )
    assert tr_no_veto["researcher"]["direction"] == "long"
    assert tr_no_veto["final_action"] == "long"
    assert tr_no_veto["programmatic_risk_gate"]["position_sizing"]["qty_contracts"] > 0

    # With Critic veto (M4), adverse funding crowding vetoes the long and preserves disagreement!
    tr_veto = run_symbol_decision_trace(
        bundle, equity_usdt=500.0, free_cash_usdt=500.0,
        enable_macro=True, enable_forecast=True, enable_critic_veto=True, enable_live_external=False,
    )
    assert tr_veto["critic_adversarial_pass2"]["vetoed"] is True
    assert "veto_adverse_funding_crowding" in tr_veto["critic_adversarial_pass2"]["veto_reasons"]
    assert tr_veto["final_action"] == "stand_aside"
    assert len(tr_veto["synthesizer"]["disagreements"]) >= 1


def test_ablation_tournament_comparability_and_zero_lookahead():
    frames = _synthetic_3contract_frames(460)
    idx = frames["BTCUSDT"].index
    start = idx[220]
    end = idx[-1]

    tourn = run_ablation_tournament(
        frames,
        start=start,
        end=end,
        starting_cash=500.0,
        held_out=True,
        market_data_hash="synthetic_sha256_demo",
        funding_complete=True,
    )
    assert tourn["comparison_protocol_audit"]["ok"] is True
    assert set(tourn["base_cost_results"]) == set(ARMS_ORDER)
    for arm, res in tourn["base_cost_results"].items():
        assert res["reconciliation_error"] < 1e-7

    # Automated prefix-truncation lookahead audit across 3 cutoffs
    la = audit_prefix_lookahead(
        frames,
        start=start,
        cutoffs=[idx[300], idx[360], idx[410]],
        end=end,
        starting_cash=500.0,
    )
    assert la["ok"] is True


def test_continuous_paper_append_only_triggers_and_human_approval(tmp_path):
    db_file = tmp_path / "piia_v2_test.sqlite"
    res = run_prospective_cycle(db_path=db_file, starting_cash=500.0, demo_mode=True)
    assert res["status"] == "recorded_prospective_cycle"
    assert res["order_permissions"] is False

    # Running again in the same hour returns `already_recorded_for_hour` (idempotent per hour)
    dup = run_prospective_cycle(db_path=db_file, starting_cash=500.0, demo_mode=True)
    assert dup["status"] == "already_recorded_for_hour"

    status = get_continuous_paper_status(db_file)
    assert status["initialized"] is True
    assert status["cycles_recorded"] == 1
    assert "M4_full_piia_v2" in status["books"]
    assert "human_approved_paper" in status["books"]

    # If a proposal was created in demo mode, test explicit human approval
    if status["pending_human_proposals"]:
        prop_id = status["pending_human_proposals"][0]["proposal_id"]
        approved = approve_human_proposal(prop_id, choice="approve", actor="unit-test-investor", db_path=db_file)
        assert approved["choice"] == "approve"
        assert approved["live_orders_submitted"] is False
        status2 = get_continuous_paper_status(db_file)
        assert status2["books"]["human_approved_paper"]["open_position"] is not None

    # Verify SQLite triggers block UPDATE and DELETE on append-only tables
    with sqlite3.connect(db_file) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM cycles")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("UPDATE decision_traces SET final_action = 'long'")


def test_agent_v2_cli_fastapi_and_synthetic_runner(tmp_path, monkeypatch):
    import json
    from fastapi.testclient import TestClient
    from piia.intelligence.api import app
    from piia.intelligence.cli import main as piia_main
    from experiments.run_piia_v2_oos_ablation import main as oos_main

    # 1. CLI freeze check + demo cycle + status
    cli_db = tmp_path / "cli_v2.sqlite"
    assert piia_main(["agent-verify-freeze"]) == 0
    assert piia_main(["agent-run", "--demo", "--starting-cash", "500", "--db", str(cli_db)]) == 0
    assert piia_main(["agent-status", "--db", str(cli_db)]) == 0

    cli_status = get_continuous_paper_status(cli_db)
    if cli_status["pending_human_proposals"]:
        pid = cli_status["pending_human_proposals"][0]["proposal_id"]
        assert piia_main([
            "agent-approve", "--proposal-id", pid, "--choice", "reject",
            "--actor", "cli-tester", "--note", "manual-pass", "--db", str(cli_db),
        ]) == 0

    # 2. FastAPI `/agent-v2/*` routes isolated via PIIA_V2_LEDGER_PATH
    api_db = tmp_path / "api_v2.sqlite"
    monkeypatch.setenv("PIIA_V2_LEDGER_PATH", str(api_db))
    with TestClient(app) as client:
        r_fz = client.get("/agent-v2/verify-freeze")
        assert r_fz.status_code == 200 and r_fz.json()["ok"] is True

        r_run = client.post("/agent-v2/run-cycle", json={"demo_mode": True, "starting_cash": 500.0})
        assert r_run.status_code == 200
        assert r_run.json()["order_permissions"] is False

        r_st = client.get("/agent-v2/status")
        assert r_st.status_code == 200
        st_json = r_st.json()
        assert st_json["initialized"] is True
        if st_json["pending_human_proposals"]:
            api_pid = st_json["pending_human_proposals"][0]["proposal_id"]
            r_dec = client.post(
                f"/agent-v2/proposals/{api_pid}/decision",
                json={"choice": "approve", "actor": "api-tester", "note": "approved-in-test"},
            )
            assert r_dec.status_code == 200
            assert r_dec.json()["live_orders_submitted"] is False

    # 3. Synthetic OOS ablation runner end-to-end check
    out_json = tmp_path / "synthetic_ablation.json"
    assert oos_main(["--demo-synthetic", "--out", str(out_json)]) == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["frozen_manifest_verification"]["ok"] is True
    assert payload["prefix_truncation_lookahead_audit"]["ok"] is True
    assert "unseen_post_2026_11_16_holdout" in payload["windows"]


def _df_to_candles(df: pd.DataFrame) -> list[dict]:
    from datetime import timedelta
    out = []
    for ts, row in df.iterrows():
        t_open = ts.to_pydatetime()
        out.append({
            "open_at_utc": t_open.isoformat(),
            "close_at_utc": (t_open + timedelta(hours=1)).isoformat(),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 100.0)),
        })
    return out


def _make_snapshot_from_frames(
    frames: dict[str, pd.DataFrame],
    *,
    settled_funding: dict | None = None,
) -> dict:
    last_close = max(df.index[-1] for df in frames.values()) + pd.Timedelta(hours=1)
    now_iso = last_close.isoformat()
    instruments = {}
    for sym, df in frames.items():
        last_c = float(df.iloc[-1]["close"])
        instruments[sym] = {
            "candles_closed_1h": _df_to_candles(df),
            "funding": {"funding_rate": "0.0001"},
            "ticker": {"mark": last_c, "last": last_c, "observed_at_utc": now_iso},
            "open_interest": None,
        }
    return {
        "mode": "supplied_snapshot",
        "collected_at_utc": now_iso,
        "market_snapshot": {"retrieved_at_utc": now_iso, "instruments": instruments},
        "books_snapshot": {"ok": False, "instruments": {}},
        "tiers_snapshot": {"ok": False, "instruments": {}},
        "settled_funding": settled_funding or {"ok": False, "instruments": {}},
        "macro_context": {
            "status": "unavailable",
            "historical_point_in_time_lagged": False,
            "macro_tightening_headwind": False,
            "macro_easing_tailwind": False,
        },
        "event_calendar_context": {"status": "unavailable", "historical_replay_supported": False},
        "flows_positioning_context": {"status": "unavailable", "liquidity_state": "unknown", "historical_replay_supported": False},
        "onchain_context": {"status": "unavailable", "historical_replay_supported": False},
        "polymarket_context": {"status": "unavailable", "historical_replay_supported": False},
        "chronos_forecast": {"weights_ran": False, "forecasts": {}},
    }


def test_orthogonal_single_switch_attribution_isolation():
    frames = _synthetic_3contract_frames(320)
    df = frames["BTCUSDT"].copy()
    anchor = float(df.iloc[249]["close"])
    for k in range(250, 320):
        # Tiny 0.04% moves + tiny range so ATR14/close is small enough to trigger friction veto
        df.iloc[k, df.columns.get_loc("close")] = anchor * (1.0 + 0.0004 * (k - 249))
        df.iloc[k, df.columns.get_loc("open")] = df.iloc[k - 1]["close"]
        df.iloc[k, df.columns.get_loc("high")] = df.iloc[k]["close"] * 1.0005
        df.iloc[k, df.columns.get_loc("low")] = df.iloc[k]["open"] * 0.9998
    df.iloc[-1, df.columns.get_loc("close")] = float(df.iloc[-2]["high"]) * 1.003
    df.iloc[-1, df.columns.get_loc("high")] = df.iloc[-1]["close"] * 1.0005

    prep = prepare_market_feature_table(df)
    # Force low atr14 so cost_gate ALSO fires when enabled, plus macro headwind & opposing stat forecast!
    row = prep.iloc[-1].copy()
    row["atr14"] = float(row["close"]) * 0.002  # 2*ATR=0.4%, round-trip cost=0.22% -> 55% > 25% cap
    row["atr14_mean50"] = row["atr14"]
    row["fc_point_24h"] = float(row["close"]) - 1.5 * float(row["atr14"])  # strong opposing stat drift!

    bundle = build_bar_evidence_for_symbol(
        "BTCUSDT",
        feature_row=row,
        prev_close=float(prep.iloc[-2]["close"]),
        decision_utc=datetime(2026, 11, 18, 0, 0, tzinfo=timezone.utc),
        funding_rate_8h=0.0001,
        macro_ctx={
            "status": "available",
            "historical_point_in_time_lagged": True,
            "macro_tightening_headwind": True,
            "macro_easing_tailwind": False,
        },
    )

    m1 = run_symbol_decision_trace(
        bundle, equity_usdt=500.0, free_cash_usdt=500.0,
        enable_cost_gate=False, enable_macro_filter=False, enable_stat_forecast_filter=False, enable_live_external=False,
    )
    assert m1["researcher"]["direction"] == "long"
    assert m1["critic_adversarial_pass2"]["veto_reasons"] == []
    assert m1["final_action"] == "long"
    assert m1["module_execution"]["neural_chronos_model"] == "not_executed"
    assert m1["module_execution"]["official_llm_review"] == "NOT_WIRED"

    # M2a: ONLY cost gate
    m2a = run_symbol_decision_trace(
        bundle, equity_usdt=500.0, free_cash_usdt=500.0,
        enable_cost_gate=True, enable_macro_filter=False, enable_stat_forecast_filter=False, enable_live_external=False,
    )
    assert m2a["critic_adversarial_pass2"]["veto_reasons"] == ["veto_cost_to_volatility_drag"]

    # M2b: ONLY macro filter (MUST fire macro veto WITHOUT activating cost_gate!)
    m2b = run_symbol_decision_trace(
        bundle, equity_usdt=500.0, free_cash_usdt=500.0,
        enable_cost_gate=False, enable_macro_filter=True, enable_stat_forecast_filter=False, enable_live_external=False,
    )
    assert m2b["critic_adversarial_pass2"]["veto_reasons"] == ["veto_macro_real_yield_dollar_headwind"]

    # M2c: ONLY statistical drift filter (MUST fire stat forecast veto WITHOUT activating cost_gate!)
    m2c = run_symbol_decision_trace(
        bundle, equity_usdt=500.0, free_cash_usdt=500.0,
        enable_cost_gate=False, enable_macro_filter=False, enable_stat_forecast_filter=True, enable_live_external=False,
    )
    assert m2c["critic_adversarial_pass2"]["veto_reasons"] == ["veto_stat_forecast_strong_opposing_drift"]


def test_continuous_paper_loss_halt_persists_across_restarts(tmp_path):
    from datetime import timedelta

    cfg = load_v2_config()
    frames = _synthetic_3contract_frames(320)
    # Shape BTCUSDT bar 299 (last bar of cycle 1 at cutoff index 300) into a clean breakout
    for sym in SYMBOLS:
        frames[sym] = frames[sym].iloc[:300].copy()
    btc = frames["BTCUSDT"]
    anchor = float(btc.iloc[249]["close"])
    for k in range(250, 300):
        btc.iloc[k, btc.columns.get_loc("close")] = anchor * (1.0 + 0.0025 * (k - 249))
        btc.iloc[k, btc.columns.get_loc("open")] = btc.iloc[k - 1]["close"]
        btc.iloc[k, btc.columns.get_loc("high")] = btc.iloc[k]["close"] * 1.0015
        btc.iloc[k, btc.columns.get_loc("low")] = btc.iloc[k]["open"] * 0.999
    btc.iloc[-1, btc.columns.get_loc("close")] = float(btc.iloc[-2]["high"]) * 1.008
    btc.iloc[-1, btc.columns.get_loc("high")] = btc.iloc[-1]["close"] * 1.001
    # Keep XAUUSDT & NDX100USDT flat/uncertain so M4 opens BTCUSDT cleanly at cycle 1
    for quiet in ("XAUUSDT", "NDX100USDT"):
        qdf = frames[quiet]
        q_px = float(qdf.iloc[-50]["close"])
        for k in range(len(qdf) - 50, len(qdf)):
            qdf.iloc[k, qdf.columns.get_loc("open")] = q_px
            qdf.iloc[k, qdf.columns.get_loc("high")] = q_px * 1.0005
            qdf.iloc[k, qdf.columns.get_loc("low")] = q_px * 0.9995
            qdf.iloc[k, qdf.columns.get_loc("close")] = q_px

    # Subcase A: 3% daily loss halt persists across DB restart & blocks new entries
    db_daily = tmp_path / "daily_halt.sqlite"
    snap1 = _make_snapshot_from_frames(frames)
    r1 = run_prospective_cycle(db_path=db_daily, starting_cash=500.0, snapshot_override=snap1)
    assert r1["decisions"]["BTCUSDT"]["final_action"] == "long"

    # Inject a severe adverse bar on BTCUSDT so open position gaps down -> >3% single-day equity loss (< 485U)
    full_frames = _synthetic_3contract_frames(320)
    t_next = btc.index[-1] + pd.Timedelta(hours=1)
    open_entry = get_continuous_paper_status(db_daily)["books"]["M4_full_piia_v2"]["open_position"]["entry"]
    crash_bar = pd.DataFrame(
        [{"open": open_entry * 0.85, "high": open_entry * 0.86, "low": open_entry * 0.84, "close": open_entry * 0.85, "volume": 500.0}],
        index=pd.DatetimeIndex([t_next], name="date", tz="UTC"),
    )
    frames2 = {s: frames[s].copy() for s in SYMBOLS}
    frames2["BTCUSDT"] = pd.concat([frames2["BTCUSDT"], crash_bar])
    for quiet in ("XAUUSDT", "NDX100USDT"):
        qdf = frames2[quiet]
        last_r = qdf.iloc[-1:].copy()
        last_r.index = pd.DatetimeIndex([t_next], name="date", tz="UTC")
        frames2[quiet] = pd.concat([qdf, last_r])
    # Add a valid settlement at t_next if boundary, or run cycle 2
    snap2 = _make_snapshot_from_frames(frames2)
    run_prospective_cycle(db_path=db_daily, starting_cash=500.0, snapshot_override=snap2)

    # Simulate process restart: read fresh status & run cycle 3 on a new strong breakout next day!
    st_after = get_continuous_paper_status(db_daily)
    assert st_after["books"]["M4_full_piia_v2"]["halted"] is True
    assert st_after["books"]["M4_full_piia_v2"]["halt_reason"] in {"daily_loss_halt_3pct", "total_loss_halt_10pct"}
    assert st_after["books"]["M4_full_piia_v2"]["open_position"] is None

    # Create another breakout bar 24h later -> M4 MUST refuse via `risk_gate:kill_switch_halted`
    frames3 = {s: frames2[s].copy() for s in SYMBOLS}
    t_day2 = t_next + pd.Timedelta(hours=25)
    for k in range(1, 26):
        t_k = t_next + pd.Timedelta(hours=k)
        px_b = float(frames["BTCUSDT"].iloc[-1]["close"]) * (1.0 + 0.002 * k)
        row_b = pd.DataFrame([{"open": px_b * 0.999, "high": px_b * 1.001, "low": px_b * 0.998, "close": px_b, "volume": 200.0}],
                             index=pd.DatetimeIndex([t_k], name="date", tz="UTC"))
        frames3["BTCUSDT"] = pd.concat([frames3["BTCUSDT"], row_b])
        for quiet in ("XAUUSDT", "NDX100USDT"):
            r_q = frames2[quiet].iloc[-1:].copy()
            r_q.index = pd.DatetimeIndex([t_k], name="date", tz="UTC")
            frames3[quiet] = pd.concat([frames3[quiet], r_q])
    r3 = run_prospective_cycle(db_path=db_daily, starting_cash=500.0, snapshot_override=_make_snapshot_from_frames(frames3))
    assert r3["decisions"]["BTCUSDT"]["stand_aside_reason"] == "risk_gate:kill_switch_halted"
    assert get_continuous_paper_status(db_daily)["books"]["M4_full_piia_v2"]["open_position"] is None

    # Subcase B: Verify exact 3% daily halt trigger AND exact 10% total halt trigger via book reconciliation
    db_tot = tmp_path / "total_halt.sqlite"
    conn = _init_db(db_tot)
    try:
        t0 = pd.Timestamp("2026-11-10T00:00:00+00:00")
        _append_event(conn, "M4_full_piia_v2", "init_book", None, {"starting_cash": 1000.0}, 1000.0, t0.isoformat())
        # Lose ~2.8% on 4 separate days -> single-day <3% each day, cumulative >10% (<= 900.0) triggers total_loss_halt_10pct!
        for d_off, loss in enumerate([27.0, 27.0, 27.0, 22.0]):
            d_ts = (t0 + timedelta(days=d_off)).isoformat()
            pos_id = f"p_{d_off}"
            pos_p = {
                "position_id": pos_id, "symbol": "BTCUSDT", "side": "long", "setup_type": "trend_breakout",
                "qty": 0.1, "contract_size": 1.0, "entry": 60000.0, "stop": 59000.0,
                "stop_effective_from_utc": d_ts, "max_hold_hours": 72, "entry_fee": 0.0,
                "opened_at_utc": d_ts, "entry_bar_open_utc": d_ts, "last_processed_bar_close_utc": d_ts,
            }
            cash_after_open = 1000.0 - d_off * 27.0
            _append_event(conn, "M4_full_piia_v2", "day_anchor", "BTCUSDT", {"utc_date": (t0 + timedelta(days=d_off)).date().isoformat(), "day_start_equity_usdt": cash_after_open}, cash_after_open, d_ts)
            _append_event(conn, "M4_full_piia_v2", "open_position", "BTCUSDT", pos_p, cash_after_open, d_ts)
            close_ts = (t0 + timedelta(days=d_off, hours=2)).isoformat()
            cash_after_close = cash_after_open - loss
            _append_event(conn, "M4_full_piia_v2", "close_position", "BTCUSDT", {
                "position_id": pos_id, "symbol": "BTCUSDT", "side": "long", "qty": 0.1,
                "entry": 60000.0, "exit": 60000.0 - loss * 10.0, "exit_fee": 0.0,
                "known_funding_paid_usdt": 0.0, "funding_complete": True,
                "pnl_ex_unknown_funding_usdt": -loss, "net_pnl": -loss, "exit_time_utc": close_ts, "exit_reason": "stop_loss",
            }, cash_after_close, close_ts)
        conn.commit()
        st_pre = _reconstruct_book_state(conn, "M4_full_piia_v2", 1000.0)
        assert st_pre["halted"] is False
        _reconcile_open_position_with_bars(
            conn, "M4_full_piia_v2", st_pre, {}, None, cfg, t0.isoformat(),
            current_cutoff_utc=pd.Timestamp("2026-11-13T05:00:00+00:00"),
        )
        conn.commit()
    finally:
        conn.close()
    st_tot = get_continuous_paper_status(db_tot)["books"]["M4_full_piia_v2"]
    assert st_tot["halted"] is True
    assert st_tot["halt_reason"] == "total_loss_halt_10pct"


def test_continuous_paper_funding_directions_and_missing_rate(tmp_path):
    cfg = load_v2_config()
    # Build bars covering 06:00Z .. 17:00Z across two 8h funding boundaries (08:00Z and 16:00Z)
    idx = pd.date_range("2026-11-20T06:00:00+00:00", periods=12, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "open": [60000.0] * 12,
        "high": [60100.0] * 12,
        "low": [59900.0] * 12,
        "close": [60000.0] * 12,
        "volume": [100.0] * 12,
    }, index=idx)
    prep = prepare_market_feature_table(df)
    t_08_ms = int(pd.Timestamp("2026-11-20T08:00:00+00:00").timestamp() * 1000)
    t_16_ms = int(pd.Timestamp("2026-11-20T16:00:00+00:00").timestamp() * 1000)

    # +0.0005 at 08:00Z, -0.0003 at 16:00Z
    settled_full = {
        "ok": True,
        "instruments": {
            "BTCUSDT": {
                "ok": True,
                "interval_hours": 8,
                "schedule_verified": True,
                "source_url": "https://api.bitget.com/api/v2/mix/market/history-fund-rate?symbol=BTCUSDT",
                "raw_sha256": "abc123provenance",
                "retrieved_at_utc": "2026-11-20T17:05:00+00:00",
                "records": [
                    {"timestamp": t_08_ms, "settlement_utc": "2026-11-20T08:00:00+00:00", "rate": 0.0005},
                    {"timestamp": t_16_ms, "settlement_utc": "2026-11-20T16:00:00+00:00", "rate": -0.0003},
                ],
            }
        },
    }

    db_dir = tmp_path / "funding_dir.sqlite"
    conn = _init_db(db_dir)
    try:
        for bname, side, stop_px in (("book_long", "long", 58000.0), ("book_short", "short", 62000.0)):
            _append_event(conn, bname, "init_book", None, {"starting_cash": 1000.0}, 1000.0, "2026-11-20T06:00:00+00:00")
            pos_p = {
                "position_id": f"pos_{bname}", "symbol": "BTCUSDT", "side": side, "setup_type": "trend_breakout",
                "qty": 0.01, "contract_size": 1.0, "entry": 60000.0, "stop": stop_px,
                "stop_effective_from_utc": "2026-11-20T06:00:00+00:00", "max_hold_hours": 72, "entry_fee": 0.36,
                "opened_at_utc": "2026-11-20T06:00:00+00:00", "entry_bar_open_utc": "2026-11-20T06:00:00+00:00",
                "last_processed_bar_close_utc": "2026-11-20T06:00:00+00:00",
            }
            _append_event(conn, bname, "open_position", "BTCUSDT", pos_p, 1000.0 - 0.36, "2026-11-20T06:00:00+00:00")
            st = _reconstruct_book_state(conn, bname, 1000.0)
            _reconcile_open_position_with_bars(conn, bname, st, {"BTCUSDT": prep}, settled_full, cfg, "2026-11-20T18:00:00+00:00")
        conn.commit()
        long_st = _reconstruct_book_state(conn, "book_long", 1000.0)
        short_st = _reconstruct_book_state(conn, "book_short", 1000.0)
    finally:
        conn.close()

    # Notional = 0.01 * 60000 = 600 USDT. Net rate paid by long = (+0.0005 - 0.0003) = +0.0002 -> +0.12 USDT
    assert long_st["known_funding_paid_usdt"] == pytest.approx(0.12)
    assert short_st["known_funding_paid_usdt"] == pytest.approx(-0.12)
    assert long_st["reconciliation_error_usdt"] < 1e-9
    assert short_st["reconciliation_error_usdt"] < 1e-9
    f_ev = long_st["open_pos"]["funding_records"][0]
    assert f_ev["raw_sha256"] == "abc123provenance"
    assert f_ev["estimation_basis"] == "public_settled_rate_times_1h_open_mark_proxy_not_venue_statement"

    # Now test MISSING funding rate (omit 16:00Z rate) -> closes trade cleanly but blocks net PnL claim!
    db_miss = tmp_path / "funding_miss.sqlite"
    settled_partial = {
        "ok": True,
        "instruments": {
            "BTCUSDT": {
                "ok": True, "interval_hours": 8, "schedule_verified": False,  # missing 16:00 record!
                "source_url": "https://api.bitget.com/api/v2/mix/market/history-fund-rate",
                "raw_sha256": "partial123", "retrieved_at_utc": "2026-11-20T18:00:00+00:00",
                "records": [{"timestamp": t_08_ms, "settlement_utc": "2026-11-20T08:00:00+00:00", "rate": 0.0005}],
            }
        },
    }
    # Last bar hits stop at 17:00Z to close the trade
    df_close = df.copy()
    df_close.iloc[-1, df_close.columns.get_loc("low")] = 57900.0
    prep_close = prepare_market_feature_table(df_close)
    conn2 = _init_db(db_miss)
    try:
        _append_event(conn2, "M4_full_piia_v2", "init_book", None, {"starting_cash": 1000.0}, 1000.0, "2026-11-20T06:00:00+00:00")
        _append_event(conn2, "M4_full_piia_v2", "open_position", "BTCUSDT", {
            "position_id": "pos_miss", "symbol": "BTCUSDT", "side": "long", "setup_type": "trend_breakout",
            "qty": 0.01, "contract_size": 1.0, "entry": 60000.0, "stop": 58000.0,
            "stop_effective_from_utc": "2026-11-20T06:00:00+00:00", "max_hold_hours": 72, "entry_fee": 0.36,
            "opened_at_utc": "2026-11-20T06:00:00+00:00", "entry_bar_open_utc": "2026-11-20T06:00:00+00:00",
            "last_processed_bar_close_utc": "2026-11-20T06:00:00+00:00",
        }, 1000.0 - 0.36, "2026-11-20T06:00:00+00:00")
        st2 = _reconstruct_book_state(conn2, "M4_full_piia_v2", 1000.0)
        _reconcile_open_position_with_bars(conn2, "M4_full_piia_v2", st2, {"BTCUSDT": prep_close}, settled_partial, cfg, "2026-11-20T18:00:00+00:00")
        conn2.commit()
    finally:
        conn2.close()
    miss_book = get_continuous_paper_status(db_miss)["books"]["M4_full_piia_v2"]
    assert miss_book["closed_trades"] == 1
    assert miss_book["funding_complete"] is False
    assert miss_book["unknown_funding_events"] >= 1
    assert miss_book["realized_net_pnl_usdt"] is None
    assert miss_book["net_return_pct"] is None
    assert miss_book["net_return_claim_allowed"] is False
    assert miss_book["reconciliation_error_usdt"] < 1e-9


def test_continuous_paper_cursor_prevents_double_settlement_and_retroactive_stop(tmp_path):
    cfg = load_v2_config()
    # Timeline: T0=07:00Z open (be_trigger=61800, be_stop=60075, initial_stop=58800)
    # Bar 07:00..08:00 dips to low=59500 (safe vs 58800, BELOW later BE 60075!)
    # Bar 08:00..09:00 settles funding (+0.0005 at 08:00Z) and rallies high=62000 (>= be_trigger -> ratchets BE=60075 at 09:00Z!)
    # Re-run / restart at 10:00Z passing ALL overlapping bars [06:00..09:00] -> MUST NOT double-charge 08:00Z OR trigger 60075 on 07:00Z bar!
    idx = pd.date_range("2026-11-20T06:00:00+00:00", periods=4, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "open": [60000.0, 60000.0, 60500.0, 61500.0],
        "high": [60200.0, 60400.0, 62000.0, 61800.0],
        "low": [59800.0, 59500.0, 60200.0, 61000.0],  # Bar 1 low=59500 < 60075; Bar 3 low=61000 > 60075
        "close": [60000.0, 60500.0, 61500.0, 61600.0],
        "volume": [100.0] * 4,
    }, index=idx)
    t_08_ms = int(pd.Timestamp("2026-11-20T08:00:00+00:00").timestamp() * 1000)
    settled = {
        "ok": True,
        "instruments": {
            "BTCUSDT": {
                "ok": True, "interval_hours": 8, "schedule_verified": True,
                "source_url": "https://api.bitget.com/test", "raw_sha256": "h1", "retrieved_at_utc": "2026-11-20T10:00:00+00:00",
                "records": [{"timestamp": t_08_ms, "settlement_utc": "2026-11-20T08:00:00+00:00", "rate": 0.0005}],
            }
        },
    }
    db_file = tmp_path / "cursor.sqlite"
    conn = _init_db(db_file)
    try:
        _append_event(conn, "M4_full_piia_v2", "init_book", None, {"starting_cash": 1000.0}, 1000.0, "2026-11-20T07:00:00+00:00")
        _append_event(conn, "M4_full_piia_v2", "open_position", "BTCUSDT", {
            "position_id": "pos_cur", "symbol": "BTCUSDT", "side": "long", "setup_type": "trend_breakout",
            "qty": 0.01, "contract_size": 1.0, "entry": 60000.0, "stop": 58800.0,
            "stop_effective_from_utc": "2026-11-20T07:00:00+00:00",
            "be_stop": 60075.0, "be_trigger": 61800.0, "be_locked": False, "max_hold_hours": 72,
            "entry_fee": 0.36, "opened_at_utc": "2026-11-20T07:00:00+00:00",
            "entry_bar_open_utc": "2026-11-20T07:00:00+00:00",
            "last_processed_bar_close_utc": "2026-11-20T07:00:00+00:00",
        }, 1000.0 - 0.36, "2026-11-20T07:00:00+00:00")
        # Step 1: Reconcile bars up to 09:00Z close ([07:00, 08:00])
        st = _reconstruct_book_state(conn, "M4_full_piia_v2", 1000.0)
        _reconcile_open_position_with_bars(conn, "M4_full_piia_v2", st, {"BTCUSDT": df.iloc[:3]}, settled, cfg, "2026-11-20T09:05:00+00:00")
        conn.commit()
        # Repeated call with identical bars -> NO duplicate funding or cursor event
        st_rep = _reconstruct_book_state(conn, "M4_full_piia_v2", 1000.0)
        _reconcile_open_position_with_bars(conn, "M4_full_piia_v2", st_rep, {"BTCUSDT": df.iloc[:3]}, settled, cfg, "2026-11-20T09:06:00+00:00")
        conn.commit()
        # Step 2 (restart): Reconcile overlapping full bars [06:00..09:00] -> ONLY 09:00..10:00 processes
        st_next = _reconstruct_book_state(conn, "M4_full_piia_v2", 1000.0)
        _reconcile_open_position_with_bars(conn, "M4_full_piia_v2", st_next, {"BTCUSDT": df}, settled, cfg, "2026-11-20T10:05:00+00:00")
        conn.commit()
    finally:
        conn.close()

    final_bk = get_continuous_paper_status(db_file)["books"]["M4_full_piia_v2"]
    assert final_bk["closed_trades"] == 0
    assert final_bk["open_position"] is not None
    assert final_bk["open_position"]["stop"] == pytest.approx(60075.0)
    assert final_bk["open_position"]["be_locked"] is True
    assert final_bk["known_funding_paid_usdt"] == pytest.approx(0.01 * 60500.0 * 0.0005)  # charged ONCE!
    assert final_bk["reconciled_bar_close_utc"] == "2026-11-20T10:00:00+00:00"
    assert final_bk["reconciliation_error_usdt"] < 1e-9


def test_human_approval_revalidates_expiry_quote_position_balance_and_halt(tmp_path):
    from datetime import timedelta

    db_file = tmp_path / "approval_reval.sqlite"
    res = run_prospective_cycle(db_path=db_file, starting_cash=500.0, demo_mode=True)
    assert res["status"] == "recorded_prospective_cycle"
    props = get_continuous_paper_status(db_file)["pending_human_proposals"]
    assert len(props) >= 1
    p0 = props[0]
    pid = p0["proposal_id"]
    expires = datetime.fromisoformat(p0["expires_at_utc"]).astimezone(timezone.utc)
    ref_px = float(p0["entry_plan"]["reference_price"])
    stop_px = float(p0["exit_plan"]["initial_stop_loss"])
    now_ok = expires - timedelta(minutes=5)

    # 1. Expired proposal rejected
    with pytest.raises(ValueError, match="proposal_expired"):
        approve_human_proposal(
            pid, choice="approve", db_path=db_file,
            now_override=expires + timedelta(seconds=10),
            current_quote={"price": ref_px, "observed_at_utc": (expires + timedelta(seconds=5)).isoformat()},
        )

    # 2. Stale quote (>5m) rejected
    with pytest.raises(ValueError, match="quote_stale"):
        approve_human_proposal(
            pid, choice="approve", db_path=db_file,
            now_override=now_ok,
            current_quote={"price": ref_px, "observed_at_utc": (now_ok - timedelta(minutes=6)).isoformat()},
        )

    # 3. Adversely drifted quote / crossed stop rejected
    with pytest.raises(ValueError, match="quote_crossed_stop|quote_drifted_adversely"):
        approve_human_proposal(
            pid, choice="approve", db_path=db_file,
            now_override=now_ok,
            current_quote={"price": stop_px, "observed_at_utc": now_ok.isoformat()},
        )

    # 4. Successful approval within bounds -> opens position; second approval fails with open_position_exists
    ok_res = approve_human_proposal(
        pid, choice="approve", db_path=db_file,
        now_override=now_ok,
        current_quote={"price": ref_px, "observed_at_utc": now_ok.isoformat()},
    )
    assert ok_res["choice"] == "approve"
    if len(props) >= 2:
        with pytest.raises(ValueError, match="open_position_exists"):
            approve_human_proposal(
                props[1]["proposal_id"], choice="approve", db_path=db_file,
                now_override=now_ok,
                current_quote={"price": float(props[1]["entry_plan"]["reference_price"]), "observed_at_utc": now_ok.isoformat()},
            )

    # 5. Depleted balance & halted book fail approval
    db_bal = tmp_path / "approval_bal.sqlite"
    run_prospective_cycle(db_path=db_bal, starting_cash=500.0, demo_mode=True)
    p_bal = get_continuous_paper_status(db_bal)["pending_human_proposals"][0]
    exp_b = datetime.fromisoformat(p_bal["expires_at_utc"]).astimezone(timezone.utc)
    t_b = exp_b - timedelta(minutes=5)
    conn = _init_db(db_bal)
    try:
        # Deplete balance slightly below threshold or trigger halt
        _append_event(conn, "human_approved_paper", "day_anchor", None, {"utc_date": t_b.date().isoformat(), "day_start_equity_usdt": 500.0}, 500.0, t_b.isoformat())
        _append_event(conn, "human_approved_paper", "bar_cursor", None, {"note": "simulate loss"}, 440.0, t_b.isoformat())
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(ValueError, match="kill_switch_halted"):
        approve_human_proposal(
            p_bal["proposal_id"], choice="approve", db_path=db_bal,
            now_override=t_b,
            current_quote={"price": float(p_bal["entry_plan"]["reference_price"]), "observed_at_utc": t_b.isoformat()},
        )

    # 6. Non-halted low balance failing minimum notional / order budget at revalidation
    db_low = tmp_path / "approval_low_cash.sqlite"
    conn_l = _init_db(db_low)
    try:
        # 15 USDT book (0% loss -> NOT halted), where 1% risk = 0.15U fails BTC min_qty 0.0001!
        t_now = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)
        p_id_low = "prop_low_bal"
        prop_low = {
            "proposal_id": p_id_low, "cycle_id": "cyc_low", "decision_id": "dec_low",
            "mode": "supplied_snapshot", "symbol": "BTCUSDT", "side": "long", "atr14": 2500.0,
            "entry_plan": {"reference_price": 65000.0},
            "exit_plan": {"initial_stop_loss": 60000.0, "trailing_or_target_rule": "trail", "invalidation": "inv", "max_hold_hours": 72},
            "position_sizing": {"qty_contracts": 0.0001},
            "cost_and_risk": {"entry_taker_fee_usdt": 0.01},
            "expires_at_utc": (t_now + timedelta(minutes=15)).isoformat(),
        }
        import json as _j
        conn_l.execute(
            "INSERT INTO human_proposals(proposal_id, cycle_id, decision_id, created_at_utc, expires_at_utc, symbol, side, proposal_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (p_id_low, "cyc_low", "dec_low", t_now.isoformat(), prop_low["expires_at_utc"], "BTCUSDT", "long", _j.dumps(prop_low)),
        )
        _append_event(conn_l, "human_approved_paper", "init_book", None, {"starting_cash": 15.0}, 15.0, t_now.isoformat())
        _append_event(conn_l, "human_approved_paper", "day_anchor", None, {"utc_date": t_now.date().isoformat(), "day_start_equity_usdt": 15.0}, 15.0, t_now.isoformat())
        conn_l.commit()
    finally:
        conn_l.close()
    with pytest.raises(ValueError, match="minimum_order_exceeds_budget"):
        approve_human_proposal(
            "prop_low_bal", choice="approve", db_path=db_low,
            now_override=t_now + timedelta(minutes=2),
            current_quote={"price": 65000.0, "observed_at_utc": t_now.isoformat()},
        )
