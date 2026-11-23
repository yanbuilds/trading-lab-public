"""Comprehensive tests for PIIA Judgment v3: evidence integrity, tide, synthesis, chat Q&A, ablation, and non-interference."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import sqlite3

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from experiments.frozen_policy_capture import verify_policy
from piia.intelligence.agent_v2.manifest import verify_piia_v2_frozen
from piia.intelligence.judgment import evaluation as j_eval
from piia.intelligence.judgment.chat_workbench import answer_judgment_query
from piia.intelligence.judgment.evaluation import (
    JudgmentLedgerKindError,
    get_judgment_ledger_status,
    load_latest_two_packets,
    record_judgment_cycle,
    run_domain_ablation,
    run_four_way_comparison,
)
from piia.intelligence.judgment.evidence import generate_demo_judgment_inputs
from piia.intelligence.judgment.manifest import verify_judgment_v3_frozen
from piia.intelligence.judgment.synthesizer import synthesize_judgment_packet


def test_frozen_v1_and_v2_2_1_controls_remain_untouched_and_judgment_v3_passes():
    verify_policy()  # Frozen policy v1 untouched
    v2 = verify_piia_v2_frozen()
    assert v2["ok"] is True
    assert v2["manifest_sha256"] == "0c8358c94e3cf12beb0a719fef4c7227516bc77bce2a7528c0e3343cbdc47578"
    v3 = verify_judgment_v3_frozen()
    assert v3["ok"] is True


def test_future_stale_duplicate_and_unverified_news_filtering():
    bundle = generate_demo_judgment_inputs()
    cutoff = pd.Timestamp(bundle["cutoff_utc"]).to_pydatetime()

    bundle["raw_news_items"] = [
        # 1. Verified primary release 2h before cutoff
        {
            "news_id": "n_primary",
            "title": "Federal Reserve Press Release: FOMC signals dovish rate cut path as inflation cools",
            "url": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20261124a.htm",
            "published_at_utc": (cutoff - timedelta(hours=2)).isoformat(),
            "source_name": "Federal Reserve",
            "source_kind": "primary_official",
        },
        # 2. Duplicate repost 30m later (same headline wording -> zero weight)
        {
            "news_id": "n_dup",
            "title": "Federal Reserve Press Release: FOMC signals dovish rate cut path as inflation cools!",
            "url": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20261124a-copy.htm",
            "published_at_utc": (cutoff - timedelta(hours=1, minutes=30)).isoformat(),
            "source_name": "Mirror Feed",
            "source_kind": "primary_official",
        },
        # 3. Future news (published AFTER cutoff -> strictly future_excluded)
        {
            "news_id": "n_future",
            "title": "BLS CPI Inflation shocks higher after decision cutoff",
            "url": "https://www.bls.gov/cpi/future_release.htm",
            "published_at_utc": (cutoff + timedelta(minutes=10)).isoformat(),
            "source_name": "BLS",
            "source_kind": "primary_official",
        },
        # 4. Stale news (90h old > 72h -> strictly stale_excluded)
        {
            "news_id": "n_stale",
            "title": "FOMC minutes old release",
            "url": "https://www.federalreserve.gov/old.htm",
            "published_at_utc": (cutoff - timedelta(hours=90)).isoformat(),
            "source_name": "Federal Reserve",
            "source_kind": "primary_official",
        },
        # 5. Unverified rumor (no URL / no timestamp -> unverified_unknown)
        {
            "news_id": "n_unverified",
            "title": "Unverified rumor about bitcoin etf inflow",
            "url": None,
            "published_at_utc": None,
            "verified": False,
        },
        # 6. Swing-only news (36h old -> eligible for swing_1_3d, excluded from intraday)
        {
            "news_id": "n_swing_only",
            "title": "SEC approves spot bitcoin etf custody expansion filing",
            "url": "https://www.sec.gov/Archives/edgar/data/0001/btc.htm",
            "published_at_utc": (cutoff - timedelta(hours=36)).isoformat(),
            "source_name": "SEC",
            "source_kind": "primary_official",
        },
    ]

    pkt = synthesize_judgment_packet(bundle, include_v2_control=False)
    by_id = {n["news_id"]: n for n in pkt["news_audit"]}
    assert by_id["n_primary"]["news_kind"] == "primary_release"
    assert by_id["n_primary"]["freshness_status"] == "fresh"
    assert by_id["n_dup"]["news_kind"] == "duplicate_repost"
    assert by_id["n_dup"]["duplicate_of"] == "n_primary"
    assert by_id["n_future"]["freshness_status"] == "future_excluded"
    assert by_id["n_stale"]["freshness_status"] == "stale_excluded"
    assert by_id["n_unverified"]["verification_status"] == "unverified_unknown"
    assert by_id["n_unverified"]["freshness_status"] == "missing"
    assert by_id["n_swing_only"]["freshness_status"] == "lagged_usable"
    assert by_id["n_swing_only"]["eligible_horizons"] == ["swing_1_3d"]

    atom_map = {a["evidence_id"]: a for a in pkt["evidence_atoms"]}
    assert atom_map["news:n_future"]["role_by_symbol_horizon"].get("BTCUSDT:intraday") == "excluded_stale_or_future"
    assert atom_map["news:n_stale"]["role_by_symbol_horizon"].get("BTCUSDT:swing_1_3d") == "excluded_stale_or_future"
    assert atom_map["news:n_swing_only"]["role_by_symbol_horizon"].get("BTCUSDT:intraday") == "other_horizon_only"


def test_cross_timestamp_mixing_and_etf_dashes_and_unrun_chronos():
    # 1. Mismatched contract timestamps force coherence=False and all judgments stand aside
    bad_bundle = generate_demo_judgment_inputs()
    bad_bundle["market_snapshot"]["instruments"]["XAUUSDT"]["candles_closed_1h"].pop()  # 1h lagging!
    pkt_bad = synthesize_judgment_packet(bad_bundle, include_v2_control=False)
    assert pkt_bad["coherence"]["aligned"] is False
    for sym in ("XAUUSDT", "BTCUSDT", "NDX100USDT"):
        assert pkt_bad["judgments"][sym]["intraday"]["final_bias"] == "neutral_stand_aside"
        assert pkt_bad["judgments"][sym]["swing_1_3d"]["final_bias"] == "neutral_stand_aside"

    # 2. BTC ETF dashes ('-') / incomplete latest day MUST remain unknown/not zero + DXY/DTWEXBGS separation
    bundle = generate_demo_judgment_inputs()
    bundle["etf_flows"] = {
        "usable_for_btc_research": False,
        "latest_row_is_partial": True,
        "latest_row": {"flow_date": "2026-11-23", "unreported_fund_count": 3, "complete": False},
        "complete_daily_flows": [],
    }
    # Expired Chronos from an earlier cutoff (cutoff - 1h) MUST NOT be borrowed!
    earlier = (pd.Timestamp(bundle["cutoff_utc"]) - pd.Timedelta(hours=1)).isoformat()
    bundle["chronos_forecast"] = {
        "weights_ran": True,
        "model_id": "amazon/chronos-t5-tiny",
        "revision": "29d808298f1a62493e7b9a5e08529d0d930fa189",
        "run_at_utc": earlier,
        "forecasts": {
            "BTCUSDT": {
                "ok": True,
                "last_close_at_utc": earlier,
                "quantile_10": [90000.0] * 24,
                "quantile_50": [99000.0] * 24,
                "quantile_90": [105000.0] * 24,
            }
        },
    }
    pkt = synthesize_judgment_packet(bundle, include_v2_control=False)
    atoms = {a["evidence_id"]: a for a in pkt["evidence_atoms"]}

    # Verify strict data-tier distinctions
    assert atoms["macro:ice_dxy"]["data_tier"] == "unavailable"
    assert atoms["macro:ice_dxy"]["freshness_status"] == "missing"
    assert atoms["macro:broad_dollar_proxy:DTWEXBGS"]["data_tier"] == "proxy_series"
    assert atoms["macro:broad_dollar_proxy:DTWEXBGS"]["value_summary"]["is_ice_dxy"] is False
    assert atoms["macro:real_yield_daily:DFII10"]["data_tier"] == "latest_daily_vintage"
    assert atoms["macro:real_yield_daily:DFII10"]["value_summary"]["intraday_realtime"] is False
    assert atoms["sentiment:xau:cftc_cot_lagged"]["data_tier"] == "lagged_official_report"
    assert atoms["sentiment:btc:onchain_network_activity"]["value_summary"]["measure"] == "network_activity_and_fee_congestion_not_exchange_flow"
    assert atoms["sentiment:btc:onchain_exchange_netflow"]["freshness_status"] == "missing"

    # ETF incomplete/dashes must stay missing (NOT coerced to 0.0!)
    etf_atom = atoms["sentiment:btc:spot_etf_flows"]
    assert etf_atom["freshness_status"] == "missing"
    assert etf_atom["direction_by_symbol"]["BTCUSDT"] == "unknown"
    assert etf_atom["value_summary"]["dashes_treated_as_zero"] is False
    assert etf_atom["value_summary"]["sum_3_complete_days_musd"] is None

    # Expired Chronos rejected cleanly; AI flags never forged
    fc_btc = pkt["forecast_and_models"]["by_symbol"]["BTCUSDT"]["chronos_neural"]
    assert fc_btc["weights_ran"] is False
    assert fc_btc["status"] == "expired_excluded"
    assert pkt["forecast_and_models"]["ai_provenance"]["llm_research_opinion"]["executed"] is False
    assert pkt["forecast_and_models"]["ai_provenance"]["independent_ai_critic"]["status"] == "NOT_WIRED"

    # Gold vs NDX divergence explanation present
    assert "黄金" in pkt["macro_summary"]["gold_vs_ndx_divergence_note"]
    assert "纳指" in pkt["macro_summary"]["gold_vs_ndx_divergence_note"]


def test_tide_missing_data_stays_unknown_and_vix_alone_never_decides_ndx_direction():
    # Remove all macro ALFRED series -> Tide macro_backdrop MUST become "unknown" & cycle_stage "insufficient_evidence"
    bundle = generate_demo_judgment_inputs()
    bundle["alfred_series"] = {}
    bundle["etf_flows"] = {}
    pkt_miss = synthesize_judgment_packet(bundle, include_v2_control=False)
    assert pkt_miss["tide"]["macro_backdrop"]["state"] == "unknown"
    assert pkt_miss["tide"]["cycle_stage"]["stage"] == "insufficient_evidence"

    # High VIX spike alone when NDX technicals are neutral/unknown and macro is neutral -> VIX must NOT force bearish/bullish alone!
    bundle_vix = generate_demo_judgment_inputs()
    dates = [row["observation_date"] for row in bundle_vix["alfred_series"]["VIXCLS"]["observations"]]
    for sid, flat_val in (("DFII10", 2.0), ("DTWEXBGS", 124.0), ("DGS10", 4.3), ("DGS2", 4.25), ("DFF", 4.5), ("WALCL", 6980000.0), ("WDTGAL", 800000.0), ("RRPONTSYD", 190.0), ("NASDAQ100", 20500.0), ("SP500", 5900.0)):
        bundle_vix["alfred_series"][sid]["observations"] = [{"observation_date": d, "value": flat_val} for d in dates]
    bundle_vix["alfred_series"]["VIXCLS"]["observations"] = [
        {"observation_date": d, "value": 32.0 if i == len(dates) - 1 else 18.0}
        for i, d in enumerate(dates)
    ]
    bundle_vix["raw_news_items"] = []
    # Flatten NDX 1h candles into sideways range so technical_only is neutral_stand_aside
    ndx_candles = bundle_vix["market_snapshot"]["instruments"]["NDX100USDT"]["candles_closed_1h"]
    for c in ndx_candles:
        c["open"] = c["close"] = 20500.0
        c["high"] = 20520.0
        c["low"] = 20480.0
    pkt_vix = synthesize_judgment_packet(bundle_vix, include_v2_control=False)
    assert pkt_vix["sentiment_summary"]["vix"]["elevated_volatility_regime"] is True
    assert pkt_vix["judgments"]["NDX100USDT"]["intraday"]["final_bias"] == "neutral_stand_aside"
    assert pkt_vix["judgments"]["NDX100USDT"]["swing_1_3d"]["final_bias"] == "neutral_stand_aside"

    # Deterministic bit-for-bit reproducibility check
    pkt_vix_2 = synthesize_judgment_packet(deepcopy(bundle_vix), include_v2_control=False)
    assert pkt_vix["judgment_hash"] == pkt_vix_2["judgment_hash"]


def test_imminent_macro_event_triggers_hard_veto_and_driver_deduplication():
    bundle = generate_demo_judgment_inputs()
    cutoff = pd.Timestamp(bundle["cutoff_utc"]).to_pydatetime()
    bundle["event_calendar"] = {
        "present_sources": ["bls"],
        "retrieved_at_utc": cutoff.isoformat(),
        "sources": {
            "bls": {
                "source_url": "https://www.bls.gov/schedule/news_release/",
                "window_events": [
                    {
                        "agency": "bls",
                        "title": "Consumer Price Index (CPI)",
                        "scheduled_at_utc": (cutoff + timedelta(minutes=45)).isoformat(),
                        "source_url": "https://www.bls.gov/schedule/news_release/",
                    }
                ],
            }
        },
    }
    pkt = synthesize_judgment_packet(bundle, include_v2_control=False)
    # BTC intraday had bullish technical trend, MUST be vetoed to stand_aside by imminent CPI (<2h)
    btc_intra = pkt["judgments"]["BTCUSDT"]["intraday"]
    assert btc_intra["technical_only"]["bias"] == "bullish_bias"
    assert btc_intra["final_bias"] == "neutral_stand_aside"
    assert any("imminent_tier1_macro_event" in g for g in btc_intra["hard_gates_triggered"])
    assert any(ev["consensus"]["status"] == "unavailable_in_official_public_feed" for ev in pkt["macro_summary"]["events"])


def test_chat_workbench_qa_whatif_diff_and_isolated_shadow_ledger(tmp_path, monkeypatch):
    real_db = tmp_path / "j3_live.sqlite"
    demo_db = tmp_path / "j3_demo.sqlite"
    monkeypatch.setattr(j_eval, "REAL_JUDGMENT_LEDGER_PATH", real_db)
    monkeypatch.setattr(j_eval, "DEMO_JUDGMENT_LEDGER_PATH", demo_db)

    b1 = generate_demo_judgment_inputs(step_shift_hours=0)
    # First run with test_or_injected isolated ledger via demo=True (synthetic demo)
    rec1 = record_judgment_cycle(b1, db_path=demo_db)
    assert rec1["status"] == "recorded_judgment_cycle"

    # Cross-kind write refused into live shadow ledger!
    with pytest.raises(JudgmentLedgerKindError):
        record_judgment_cycle(b1, db_path=real_db)

    # SQLite triggers forbid mutating past judgment packets or arm_judgments
    with sqlite3.connect(demo_db) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("UPDATE arm_judgments SET predicted_bias = 'bullish_bias'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM judgment_packets")

    # Second cycle 2 hours later: also resolves nothing mature yet (6h > 2h) & populates previous_packet
    b2 = generate_demo_judgment_inputs(step_shift_hours=2)
    rec2 = record_judgment_cycle(b2, db_path=demo_db)
    assert rec2["status"] == "recorded_judgment_cycle"
    cur_pkt, prev_pkt = load_latest_two_packets(demo_db)
    assert cur_pkt is not None and prev_pkt is not None

    # Q1: Main 3-contract overview
    q1 = answer_judgment_query("今天黄金、BTC、纳指 100 看多还是看空？", packet=cur_pkt, previous_packet=prev_pkt)
    assert q1["intent"] == "overview"
    assert len(q1["summary_table"]) == 3
    assert "v2.2.1 正式信号" in q1["markdown"]

    # Q2: Why not short?
    q2 = answer_judgment_query("BTC为什么不做空？", packet=cur_pkt, previous_packet=prev_pkt)
    assert q2["intent"] == "why_not_direction"
    assert "BTCUSDT" in q2["details_by_symbol"]

    # Q3: Counterfactual sensitivity: what if dollar & real yields rise vs drop?
    q3_up = answer_judgment_query("如果美元上升且实际利率走高会改判吗？", packet=cur_pkt, previous_packet=prev_pkt)
    assert q3_up["intent"] == "what_if_sensitivity"
    assert q3_up["counterfactual_simulation"] is True
    assert len(q3_up["scenario"]) == 2
    assert any(d["direction_changed"] or d["conviction_changed"] for d in q3_up["diffs"])

    q3_down = answer_judgment_query("如果美元走弱且降息预期升温会改判吗？", packet=cur_pkt, previous_packet=prev_pkt)
    assert q3_down["intent"] == "what_if_sensitivity"
    assert q3_down["counterfactual_simulation"] is True
    assert len(q3_down["diffs"]) == 6

    # Q4: Diff since last snapshot
    q4 = answer_judgment_query("相比上次，哪条证据变了？", packet=cur_pkt, previous_packet=prev_pkt)
    assert q4["intent"] == "diff_since_last"
    assert len(q4["evidence_changes"]) >= 1

    st = get_judgment_ledger_status(demo_db, demo=True)
    assert st["packets_recorded"] == 2
    assert st["validation_status"] == "NOT_A_PROSPECTIVE_LEDGER"
    assert not real_db.exists()


def test_four_way_ai_comparison_five_domain_ablation_cli_and_api(tmp_path, monkeypatch):
    from piia.intelligence.api import app
    from piia.intelligence.cli import main as piia_main

    # 1. Four-way AI vs Rules comparison: unconfigured first, then with deterministic mock AI verifying A1 vs A2 isolation
    bundle = generate_demo_judgment_inputs()
    comp_unconf = run_four_way_comparison(bundle)
    assert comp_unconf["arms"]["A1_direct_ai_candles_only"]["status"] == "NOT_RUN_NEEDS_MODEL_CONFIG"
    assert comp_unconf["arms"]["A2_ai_freeform_same_packet"]["status"] == "NOT_RUN_NEEDS_MODEL_CONFIG"
    assert comp_unconf["arms"]["A1_direct_ai_candles_only"]["payload_keys"] == ["cutoff_utc", "instruments", "task"]
    assert "evidence_atoms" in comp_unconf["arms"]["A2_ai_freeform_same_packet"]["payload_keys"]

    class _MockAI:
        def compare(self, arm_name: str, payload: dict) -> dict:
            if arm_name == "A1_direct_ai_candles_only":
                assert "evidence_atoms" not in payload
                return {"status": "completed", "decisions": {"BTCUSDT": {"intraday": "bullish_bias", "swing_1_3d": "bullish_bias"}}}
            assert "evidence_atoms" in payload
            return {"status": "completed", "decisions": {"BTCUSDT": {"intraday": "neutral_stand_aside", "swing_1_3d": "bearish_bias"}}}

    comp_ai = run_four_way_comparison(bundle, ai_provider=_MockAI())
    assert comp_ai["arms"]["A1_direct_ai_candles_only"]["status"] == "completed"
    assert comp_ai["arms"]["A2_ai_freeform_same_packet"]["status"] == "completed"

    # 2. Five-domain leave-one-out ablation across a 3-hour sequence (tests shadow paper metrics too)
    seq = [generate_demo_judgment_inputs(step_shift_hours=h) for h in range(3)]
    abl = run_domain_ablation(seq[0], replay_sequence=seq)
    for arm in ("Full", "No_Macro", "No_Tide", "No_Sentiment_Flow", "No_News", "No_Forecast", "Technical_Only"):
        assert arm in abl["arms"]
        assert "shadow_paper_1x_slip" in abl["arms"][arm]
        assert abl["arms"][arm]["shadow_paper_1x_slip"]["net_return_pct_with_funding"] is None  # unknown funding stays None

    # 3. CLI & FastAPI end-to-end on demo ledger in tmp_path
    cli_demo_db = tmp_path / "cli_j3_demo.sqlite"
    assert piia_main(["judgment-verify-freeze"]) == 0
    assert piia_main(["judgment-run", "--demo", "--db", str(cli_demo_db)]) == 0
    assert piia_main(["judgment-ask", "今天黄金、BTC、纳指 100 看多还是看空？", "--demo", "--db", str(cli_demo_db)]) == 0
    assert piia_main(["judgment-status", "--demo", "--db", str(cli_demo_db)]) == 0

    api_demo_db = tmp_path / "api_j3_demo.sqlite"
    monkeypatch.setenv("PIIA_JUDGMENT_V3_DEMO_LEDGER_PATH", str(api_demo_db))
    with TestClient(app) as client:
        assert client.get("/judgment-v3/verify-freeze").json()["ok"] is True
        r_run = client.post("/judgment-v3/run", json={"demo_mode": True, "starting_cash": 500.0})
        assert r_run.status_code == 200
        r_ask = client.post("/judgment-v3/ask", json={"question": "如果美元上升会改判吗？", "demo_mode": True})
        assert r_ask.status_code == 200
        assert r_ask.json()["counterfactual_simulation"] is True
        r_abl = client.get("/judgment-v3/ablation?demo=true")
        assert r_abl.status_code == 200
        assert "Full" in r_abl.json()["domain_ablation"]["arms"]
