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
    approve_human_proposal,
    get_continuous_paper_status,
    run_prospective_cycle,
)
from piia.intelligence.agent_v2.manifest import verify_piia_v2_frozen
from piia.intelligence.agent_v2.pillars import (
    build_bar_evidence_for_symbol,
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
