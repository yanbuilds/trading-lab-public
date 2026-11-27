"""Comprehensive tests for PIIA Judgment v3: evidence integrity, tide, synthesis, chat Q&A, ablation, and non-interference."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import sqlite3

from fastapi.testclient import TestClient
import httpx
import pandas as pd
import pytest

from piia.intelligence.agent_v2.manifest import verify_piia_v2_frozen
from piia.intelligence.judgment import evaluation as j_eval
from piia.intelligence.judgment.chat_workbench import answer_judgment_query
from piia.intelligence.judgment.evaluation import (
    JudgmentLedgerKindError,
    get_judgment_ledger_status,
    load_latest_two_packets,
    record_judgment_cycle,
    replay_archived_judgment_ledger,
    run_domain_ablation,
    run_four_way_comparison,
)
from piia.intelligence.judgment.evidence import (
    build_forecast_and_model_evidence,
    build_macro_evidence,
    build_sentiment_flow_evidence,
    fetch_bitget_venue_microstructure,
    fetch_coinmetrics_btc_exchange_flows,
    fetch_spdr_gld_etf_flows,
    fetch_us_treasury_daily_real_yield,
    fetch_yahoo_intraday_macro_and_breadth,
    generate_demo_judgment_inputs,
    load_judgment_config,
    validate_atom_temporal_bounds,
    validate_market_snapshot_coherence,
)
from piia.intelligence.judgment.manifest import verify_judgment_v3_frozen
from piia.intelligence.judgment.synthesizer import (
    _apply_v3_paper_review_gate,
    _audit_venue_microstructure,
    synthesize_judgment_packet,
)


def test_frozen_v1_and_v2_2_1_controls_remain_untouched_and_judgment_v3_passes():
    try:
        from experiments.frozen_policy_capture import verify_policy
    except ModuleNotFoundError as exc:
        if exc.name != "talib":
            raise
    else:
        verify_policy()  # Frozen policy v1 untouched (verified when TA-Lib is installed)
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
    assert "`v2.2.1` 信号" in q1["markdown"] or "v2.2.1 信号" in q1["markdown"]
    assert "dir=" not in q1["markdown"] and "role=" not in q1["markdown"]
    assert "[forecast:" not in q1["markdown"] and "[sentiment:" not in q1["markdown"] and "[macro:" not in q1["markdown"]

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
        r_abl = client.get("/judgment-v3/ablation?demo=true&starting_cash=100")
        assert r_abl.status_code == 200
        assert r_abl.json()["starting_cash_usdt"] == 100.0
        assert "Full" in r_abl.json()["domain_ablation"]["arms"]
        r_nv = client.get("/judgment-v3/replay-nvidia?starting_cash=100")
        assert r_nv.status_code == 200
        assert r_nv.json()["scenarios"]["2_news_missed_anomaly_breakout_only"]["v3_paper_status"] == "pending_review"


def test_temporal_sanitization_a0_a1_a2_a3_macro_polymarket_and_post_availability_ledger(tmp_path, monkeypatch):
    from piia.intelligence.judgment.news import parse_aware_timestamp

    bundle = generate_demo_judgment_inputs(step_shift_hours=0)
    cutoff_dt = parse_aware_timestamp(bundle["cutoff_utc"])
    assert cutoff_dt is not None

    # Inject dirty candles into NDX100USDT: out-of-order, duplicate, missing-ts, unclosed, future
    clean_rows = list(bundle["market_snapshot"]["instruments"]["NDX100USDT"]["candles_closed_1h"])
    last_bar = deepcopy(clean_rows[-1])
    dup_bar = deepcopy(clean_rows[-1])
    swap_a, swap_b = deepcopy(clean_rows[-3]), deepcopy(clean_rows[-2])
    missing_ts_bar = {"open": 25000.0, "high": 25010.0, "low": 24990.0, "close": 25005.0, "volume": 1.0}
    unclosed_bar = deepcopy(last_bar)
    unclosed_bar["open_at_utc"] = cutoff_dt.isoformat()
    unclosed_bar["close_at_utc"] = (cutoff_dt + timedelta(hours=1)).isoformat()
    unclosed_bar["is_closed"] = False
    future_bar = deepcopy(last_bar)
    future_bar["open_at_utc"] = (cutoff_dt + timedelta(hours=1)).isoformat()
    future_bar["close_at_utc"] = (cutoff_dt + timedelta(hours=2)).isoformat()
    dirty_rows = clean_rows[:-3] + [swap_b, swap_a, dup_bar, missing_ts_bar, last_bar, unclosed_bar, future_bar]
    bundle["market_snapshot"]["instruments"]["NDX100USDT"]["candles_closed_1h"] = dirty_rows

    # Inject future vintage ALFRED macro & stale/future Polymarket events
    bundle["alfred_series"]["DTWEXBGS"]["vintage_date"] = (cutoff_dt + timedelta(days=2)).date().isoformat()
    bundle["polymarket"] = {
        "usable_for_research": True,
        "asof_utc": cutoff_dt.isoformat(),
        "retrieved_at_utc": cutoff_dt.isoformat(),
        "events": [
            {
                "id": "pm_future",
                "title": "Future close test market",
                "asset_mapping": {"relevant_instruments": ["BTCUSDT"]},
                "markets": [
                    {
                        "closed": False,
                        "question": "Future book timestamp test",
                        "outcomes": [
                            {
                                "name": "Yes",
                                "book": {
                                    "asof_utc": (cutoff_dt + timedelta(minutes=10)).isoformat(),
                                    "usable_for_implied_probability": True,
                                    "official_midpoint": 0.72,
                                    "official_spread": 0.02,
                                },
                            }
                        ],
                    }
                ],
            },
            {
                "id": "pm_stale",
                "title": "Stale market event",
                "asset_mapping": {"relevant_instruments": ["BTCUSDT"]},
                "markets": [
                    {
                        "closed": False,
                        "question": "Stale book timestamp test",
                        "outcomes": [
                            {
                                "name": "Yes",
                                "book": {
                                    "asof_utc": (cutoff_dt - timedelta(minutes=45)).isoformat(),
                                    "usable_for_implied_probability": True,
                                    "official_midpoint": 0.55,
                                    "official_spread": 0.02,
                                },
                            }
                        ],
                    }
                ],
            },
        ],
    }

    captured: dict[str, dict] = {}

    class _VerifySanitizedAI:
        def compare(self, arm_name: str, payload: dict) -> dict:
            captured[arm_name] = payload
            return {"status": "completed", "decisions": {"BTCUSDT": {"intraday": "bullish_bias", "swing_1_3d": "bullish_bias"}}}

    comp = run_four_way_comparison(bundle, ai_provider=_VerifySanitizedAI(), starting_cash_usdt=100.0)
    a1_bars = captured["A1_direct_ai_candles_only"]["instruments"]["NDX100USDT"]["recent_closed_1h_tail"]
    assert len(a1_bars) == 48
    opens = [parse_aware_timestamp(b["open_at_utc"]) for b in a1_bars]
    closes = [parse_aware_timestamp(b["close_at_utc"]) for b in a1_bars]
    assert all(o is not None and c is not None and c <= cutoff_dt for o, c in zip(opens, closes))
    assert opens == sorted(opens) and len(set(opens)) == 48
    audit_ndx = comp["input_audit"]["arms"]["A1_direct_ai_candles_only"]["sanitization_audit"]["NDX100USDT"]
    assert audit_ndx["dropped_invalid"] >= 1
    assert audit_ndx["dropped_unclosed"] >= 1
    assert audit_ndx["dropped_future"] >= 1
    assert audit_ndx["dropped_duplicate"] >= 1
    assert audit_ndx["was_out_of_order"] is True
    assert comp["input_audit"]["arms"]["A1_direct_ai_candles_only"]["model_config"] == comp["input_audit"]["arms"]["A2_ai_freeform_same_packet"]["model_config"]

    pkt = synthesize_judgment_packet(bundle, include_v2_control=False)
    atoms = {a["evidence_id"]: a for a in pkt["evidence_atoms"]}
    assert atoms["macro:broad_dollar_proxy:DTWEXBGS"]["freshness_status"] == "future_excluded"
    assert atoms["sentiment:polymarket:filtered_events"]["freshness_status"] == "stale_excluded"
    lc = pkt["lifecycle_timestamps"]
    for req_ts in ("bar_cutoff_utc", "latest_announcement_published_at_utc", "first_retrieved_at_utc", "judgment_generated_at_utc", "earliest_executable_at_utc"):
        assert req_ts in lc
    assert parse_aware_timestamp(lc["earliest_executable_at_utc"]) >= parse_aware_timestamp(lc["judgment_generated_at_utc"])

    # Test non-contiguous snapshots: h=0, h=1, and h=3 (gap at h=2) -> 3 snapshots, max_contiguous=2, missing=1
    demo_db = tmp_path / "non_contig.sqlite"
    for h_shift in (0, 1, 3):
        rec = record_judgment_cycle(generate_demo_judgment_inputs(step_shift_hours=h_shift), db_path=demo_db)
        assert rec["status"] == "recorded_judgment_cycle"
    st = get_judgment_ledger_status(demo_db, demo=True)
    assert st["packets_recorded"] == 3
    assert st["max_contiguous_hours"] == 2
    assert st["missing_hours_in_span"] == 1
    assert st["arm_outcomes"] == {}  # post-availability 6h hasn't matured yet!

    # Advance to +7h -> cycle h=0 had earliest_exec at h=1, so +6h from earliest_exec = h=7 (now matures!)
    # Whereas at h=6 (cutoff + 6h BEFORE earliest_exec + 6h), outcome must remain unresolved!
    rec_h6 = record_judgment_cycle(generate_demo_judgment_inputs(step_shift_hours=6), db_path=demo_db)
    assert rec_h6["outcomes_resolved"] == 0
    rec_h7 = record_judgment_cycle(generate_demo_judgment_inputs(step_shift_hours=7), db_path=demo_db)
    assert rec_h7["outcomes_resolved"] > 0


def test_nvidia_case_4_scenarios_verified_volume_and_corporate_ir_edgar(monkeypatch):
    import sys
    from piia.intelligence.judgment.manifest import ROOT
    from piia.intelligence.judgment.news import classify_and_filter_news_items, fetch_ndx_sec_edgar_filings, parse_rss_or_atom

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from experiments.replay_nvidia_20261123_case import _build_base_bundle, replay_nvidia_20261123_case

    rep = replay_nvidia_20261123_case(starting_cash_usdt=100.0)
    sc = rep["scenarios"]
    assert sc["1a_timely_official_announcement_1402z"]["official_v2_2_1_signal"] == "short"
    assert sc["1a_timely_official_announcement_1402z"]["v3_paper_status"] == "pending_review"
    assert sc["1a_timely_official_announcement_1402z"]["volume_ma20_verified_surge"] is True
    assert "放量" in sc["1a_timely_official_announcement_1402z"]["tide_structure_cn"]

    assert sc["1b_delayed_live_retrieval_1449z"]["lifecycle_timestamps"]["earliest_executable_at_utc"] == "2026-11-23T15:00:00+00:00"
    assert sc["1b_delayed_live_retrieval_1449z"]["v3_paper_status"] == "pending_review"

    assert sc["2_news_missed_anomaly_breakout_only"]["official_v2_2_1_signal"] == "short"
    assert sc["2_news_missed_anomaly_breakout_only"]["v3_intraday_final_bias"] == "neutral_stand_aside"
    assert sc["2_news_missed_anomaly_breakout_only"]["v3_paper_status"] == "pending_review"
    assert "shock_breakout_vs_counter_trend_short" in sc["2_news_missed_anomaly_breakout_only"]["v3_review_triggers"]

    assert sc["3_symmetric_downside_breakdown_vs_long"]["official_v2_2_1_signal"] == "long"
    assert sc["3_symmetric_downside_breakdown_vs_long"]["v3_paper_status"] == "pending_review"
    assert "shock_breakdown_vs_counter_trend_long" in sc["3_symmetric_downside_breakdown_vs_long"]["v3_review_triggers"]

    assert sc["4_ordinary_range_market_not_intercepted"]["official_v2_2_1_signal"] == "short"
    assert sc["4_ordinary_range_market_not_intercepted"]["v3_paper_eligible"] is True
    assert sc["4_ordinary_range_market_not_intercepted"]["v3_paper_status"] == "eligible"

    # Verified volume wording test: same breakout geometry with normal volume must NOT emit "放量"
    b_low_vol = _build_base_bundle(cutoff_iso="2026-11-23T14:00:00+00:00", retrieved_iso="2026-11-23T14:02:00+00:00", upside_shock=True)
    b_low_vol["market_snapshot"]["instruments"]["NDX100USDT"]["candles_closed_1h"][-1]["volume"] = 2.0
    pkt_lv = synthesize_judgment_packet(b_low_vol, include_v2_control=False)
    pos_lv = pkt_lv["tide"]["instrument_positions"]["NDX100USDT"]
    assert pos_lv["metrics"]["volume_surge_verified"] is False
    assert "放量" not in pos_lv["structure_kind_cn"]

    # Corporate IR RSS (NVIDIA Newsroom) + SEC EDGAR items list parsing offline
    rss_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel><title>NVIDIA Newsroom</title>
    <item>
      <title>NVIDIA Board Reinstates Jensen Huang as President, CEO and Board Director</title>
      <link>https://nvidianews.nvidia.com/news/nvidia-board-reinstates-jensen-huang-as-president-ceo-and-board-director</link>
      <pubDate>Mon, 23 Nov 2026 13:55:00 GMT</pubDate>
    </item></channel></rss>"""
    parsed_rss = parse_rss_or_atom(
        rss_xml,
        feed_id="nvda_newsroom",
        feed_name="NVIDIA Newsroom RSS",
        retrieved_at=datetime(2026, 11, 23, 14, 2, tzinfo=timezone.utc),
    )
    assert len(parsed_rss) == 1
    assert "13:55:00" in parsed_rss[0]["published_at_raw"]
    cls_nvda = classify_and_filter_news_items(parsed_rss, cutoff_utc=datetime(2026, 11, 23, 14, 0, tzinfo=timezone.utc), replay_mode=True)
    assert cls_nvda[0]["published_at_utc"] == "2026-11-23T13:55:00+00:00"
    assert cls_nvda[0]["verification_status"] == "verified"
    assert cls_nvda[0]["news_kind"] == "primary_release"

    class _FakeResp:
        def __init__(self, payload: dict):
            self._payload = payload
            self.content = json.dumps(payload).encode("utf-8")
            self.status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._payload

    class _FakeClient:
        def __init__(self, **_kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, **_kw):
            return _FakeResp(
                {
                    "name": "NVIDIA CORP",
                    "filings": {
                        "recent": {
                            "form": ["8-K", "10-Q"],
                            "filingDate": ["2026-11-23", "2026-11-18"],
                            "acceptanceDateTime": ["2026-11-23T13:57:54.000Z", "2026-11-18T21:00:00.000Z"],
                            "accessionNumber": ["0001045810-26-000123", "0001045810-26-000100"],
                            "primaryDocument": ["nvda8k.htm", "nvda10q.htm"],
                            "items": [["5.02", "9.01"], []],
                        }
                    },
                }
            )

    filings_out, edgar_errs = fetch_ndx_sec_edgar_filings(
        cutoff_utc=datetime(2026, 11, 23, 14, 0, tzinfo=timezone.utc),
        client=_FakeClient(),
    )
    assert edgar_errs == {}
    assert len(filings_out) >= 1
    assert filings_out[0]["items"] == "5.02,9.01"
    assert "items=5.02,9.01" in filings_out[0]["title"]


def test_grokbot_factor_usage_ledger_adapters_supplement_and_ai_chronos_paths(monkeypatch):
    bundle = generate_demo_judgment_inputs()
    cutoff = pd.Timestamp(bundle["cutoff_utc"]).to_pydatetime()

    # Live TFTC schema: latest_complete_row without complete_daily_flows + real Polymarket schema
    bundle["etf_flows"] = {
        "usable_for_btc_research": True,
        "asof_complete_flow_date": cutoff.date().isoformat(),
        "date_modified": cutoff.isoformat(),
        "latest_complete_row": {"flow_date": cutoff.date().isoformat(), "table_total_millions_usd": -669.16, "complete": True},
        "source_url": "https://bitcoinetf.tftc.io/",
    }
    bundle["polymarket"] = {
        "usable_for_research": True,
        "asof_utc": cutoff.isoformat(),
        "retrieved_at_utc": cutoff.isoformat(),
        "events": [
            {
                "id": "pm_live_ok",
                "title": "Bitcoin above 100k by Dec 31?",
                "category": "crypto_price",
                "asset_mapping": {"relevant_instruments": ["BTCUSDT", "NDX100USDT"], "mapping_kind": "direct"},
                "markets": [
                    {
                        "closed": False,
                        "question": "BTC > 100k?",
                        "outcomes": [
                            {
                                "name": "Yes",
                                "book": {
                                    "asof_utc": (cutoff - timedelta(seconds=30)).isoformat(),
                                    "usable_for_implied_probability": True,
                                    "official_midpoint": 0.63,
                                    "official_spread": 0.02,
                                    "change_24h": {"change_24h_pp": 4.0, "to_timestamp_utc": cutoff.isoformat()},
                                },
                            }
                        ],
                    }
                ],
            }
        ],
    }
    bundle["chronos_forecast"] = {
        "weights_ran": True,
        "model_id": "amazon/chronos-t5-tiny",
        "revision": "29d808298f1a62493e7b9a5e08529d0d930fa189",
        "run_at_utc": cutoff.isoformat(),
        "forecasts": {
            "BTCUSDT": {
                "ok": True,
                "last_close_at_utc": cutoff.isoformat(),
                "quantile_10": [98000.0] * 24,
                "quantile_50": [103500.0] * 24,
                "quantile_90": [108000.0] * 24,
            }
        },
    }
    bundle["agent_or_human_supplement"] = "Agent noted desk liquidity thinner ahead of holiday."

    pkt = synthesize_judgment_packet(bundle, starting_cash_usdt=100.0)
    assert set(pkt["factor_usage_ledger"].keys()) == {
        "macro",
        "tide",
        "sentiment_positioning",
        "etf_and_onchain",
        "polymarket",
        "news",
        "forecast_models",
    }
    etf_entries = {e["factor_id"]: e for e in pkt["factor_usage_ledger"]["etf_and_onchain"]}
    assert etf_entries["sentiment:btc:spot_etf_flows"]["api_responded_ok"] is True
    assert etf_entries["sentiment:btc:spot_etf_flows"]["participated_in_judgment"] is True
    assert etf_entries["sentiment:xau:gold_etf_flows"]["participation_status"] == "missing"

    pm_entry = pkt["factor_usage_ledger"]["polymarket"][0]
    assert pm_entry["api_responded_ok"] is True
    assert pm_entry["participated_in_judgment"] is True
    assert pm_entry["participation_status"] == "background_only"

    macro_entries = {e["factor_id"]: e for e in pkt["factor_usage_ledger"]["macro"]}
    assert macro_entries["macro:real_yield_intraday"]["participation_status"] == "missing"
    assert macro_entries["macro:real_yield_daily:DFII10"]["participated_in_judgment"] is True

    fc_entries = {e["factor_id"]: e for e in pkt["factor_usage_ledger"]["forecast_models"]}
    assert fc_entries["forecast:btcusdt:chronos_neural"]["participated_in_judgment"] is True
    assert fc_entries["model:llm_research_opinion"]["participation_status"] == "not_run"

    assert pkt["agent_or_human_supplement"]["provided"] is True
    assert pkt["agent_or_human_supplement"]["can_override_risk_gate"] is False

    # Offline mock HTTP tests for OpenAI & Anthropic official AI calls inside _call_official_ai_for_comparison
    for prov, resp_json in (
        ("openai", {"choices": [{"message": {"content": "```json\n{\"BTCUSDT\": {\"intraday\": \"bullish_bias\", \"swing_1_3d\": \"bullish_bias\"}}\n```"}}], "usage": {"total_tokens": 42}}),
        ("anthropic", {"content": [{"type": "text", "text": "{\"BTCUSDT\": {\"intraday\": \"bearish_bias\", \"swing_1_3d\": \"neutral_stand_aside\"}}"}], "usage": {"input_tokens": 10, "output_tokens": 10}}),
    ):
        monkeypatch.setattr(j_eval, "model_configured", lambda: True)
        monkeypatch.setattr(j_eval, "model_provider", lambda p=prov: p)
        monkeypatch.setattr(j_eval, "model_base_url", lambda: "https://api.example.invalid")
        monkeypatch.setattr(j_eval, "model_name", lambda: "test-model")
        monkeypatch.setenv("PIIA_MODEL_API_KEY", "sk-offline-dummy")

        class _MockAIResp:
            def __init__(self, d):
                self._d = d

            def raise_for_status(self):
                return None

            def json(self):
                return self._d

        class _MockAIClient:
            def __init__(self, **_kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def post(self, *_args, **kwargs):
                assert kwargs["json"]["temperature"] == 0
                return _MockAIResp(resp_json)

        monkeypatch.setattr(j_eval.httpx, "Client", _MockAIClient)
        out = j_eval._call_official_ai_for_comparison("A1_direct_ai_candles_only", {"cutoff_utc": bundle["cutoff_utc"]})
        assert out["status"] == "completed"
        assert "BTCUSDT" in out["decisions"]


def test_fair_100_usdt_shadow_simulation_min_qty_funding_portfolio_audit_and_cli_nvidia(tmp_path):
    from piia.intelligence.cli import main as piia_main
    from piia.intelligence.judgment.evaluation import _simulate_shadow_paper_sequence, evaluate_portfolio_risk_audit

    seq_bundles = [generate_demo_judgment_inputs(step_shift_hours=h) for h in range(10)]
    pairs = [(synthesize_judgment_packet(b, starting_cash_usdt=100.0), b["market_snapshot"]) for b in seq_bundles]

    # 1. Verify funding missing -> None, funding schedule covering window -> verified return
    sim_unknown = _simulate_shadow_paper_sequence(pairs, arm_selector="full", starting_cash=100.0, funding_by_symbol=None)
    assert sim_unknown["starting_cash_usdt"] == 100.0
    assert sim_unknown["net_return_pct_with_funding"] is None

    # Build full 8h funding schedule covering all timestamps across seq_bundles
    t0 = pd.Timestamp(seq_bundles[0]["cutoff_utc"]).to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    full_funding = {
        sym: [
            {"timestamp": int((t0 + timedelta(hours=8 * k)).timestamp() * 1000), "rate": 0.0001}
            for k in range(20)
        ]
        for sym in ("XAUUSDT", "BTCUSDT", "NDX100USDT")
    }
    sim_verified = _simulate_shadow_paper_sequence(pairs, arm_selector="full", starting_cash=100.0, funding_by_symbol=full_funding)
    assert sim_verified["funding_status"] in {"verified_from_historical_schedule", "verified_zero_exposure"}
    assert sim_verified["net_return_pct_with_funding"] is not None

    # 2. Verify pending_review_skips when first packet has NDX pending_review & high min_qty rejection when ATR huge
    huge_atr_pairs = deepcopy(pairs)
    for pkt, mkt in huge_atr_pairs:
        for sym in ("XAUUSDT", "BTCUSDT", "NDX100USDT"):
            pkt["judgments"][sym]["swing_1_3d"]["final_bias"] = "bullish_bias"
            pkt["judgments"][sym]["paper_eligibility"]["status"] = "eligible"
            for r in mkt["instruments"][sym]["candles_closed_1h"][-14:]:
                r["high"] = float(r["close"]) * 1.15
                r["low"] = float(r["close"]) * 0.85
    sim_reject = _simulate_shadow_paper_sequence(huge_atr_pairs, arm_selector="full", starting_cash=100.0)
    assert sim_reject["min_qty_budget_rejections"] >= 1

    # 3. Portfolio risk audit strictly denies single-position sufficiency
    pa = evaluate_portfolio_risk_audit(
        {sym: seq_bundles[-1]["market_snapshot"]["instruments"][sym]["candles_closed_1h"] for sym in ("XAUUSDT", "BTCUSDT", "NDX100USDT")},
        [{"symbol": "BTCUSDT", "side": "long", "notional_usdt": 150.0}, {"symbol": "NDX100USDT", "side": "long", "notional_usdt": 120.0}],
        equity_usdt=100.0,
    )
    assert pa["single_position_sim_proves_portfolio_safety"] is False
    assert pa["portfolio_risk_ok"] is False
    assert any("gross_leverage_exceeds_cap" in w for w in pa["warnings"])

    # 4. CLI judgment-replay-nvidia + supplement note
    nv_out = tmp_path / "nv_case.json"
    assert piia_main(["judgment-replay-nvidia", "--starting-cash", "100", "--out", str(nv_out)]) == 0
    assert json.loads(nv_out.read_text(encoding="utf-8"))["scenarios"]["3_symmetric_downside_breakdown_vs_long"]["v3_paper_status"] == "pending_review"

    cli_db = tmp_path / "cli_supp.sqlite"
    assert piia_main(["judgment-ask", "今天黄金、BTC、纳指 100 看多还是看空？", "--demo", "--starting-cash", "100", "--supplement-note", "CLI test supplement", "--db", str(cli_db)]) == 0
    cur_p, _ = load_latest_two_packets(cli_db)
    assert cur_p is not None and cur_p["starting_cash_usdt"] == 100.0
    assert cur_p["agent_or_human_supplement"]["note"] == "CLI test supplement"
    assert cur_p["agent_or_human_supplement"]["can_override_risk_gate"] is False


def test_strict_calendar_release_matching_and_post_release_reaction():
    cfg = load_judgment_config()
    bundle = generate_demo_judgment_inputs()
    cutoff = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
    demo_base = pd.Timestamp(bundle["cutoff_utc"]).to_pydatetime()
    frames_1h = validate_market_snapshot_coherence(bundle["market_snapshot"], cutoff_utc=demo_base)["frames_1h"]
    # Shift one frame close index around our synthetic 10:00Z..12:00Z release window
    shift_delta = cutoff - demo_base
    shifted_frames = {}
    for sym, df in frames_1h.items():
        sdf = df.copy()
        sdf.index = sdf.index + shift_delta
        shifted_frames[sym] = sdf

    cal = {
        "present_sources": ["bea", "bls"],
        "retrieved_at_utc": cutoff.isoformat(),
        "sources": {
            "bea": {
                "source_url": "https://www.bea.gov/news/schedule",
                "window_events": [
                    {
                        "agency": "BEA",
                        "title": "Gross Domestic Product (Second Estimate), Q3",
                        "scheduled_at_utc": (cutoff + timedelta(hours=3)).isoformat(),
                    },
                    {
                        "agency": "BEA",
                        "title": "Personal Income and Outlays (PCE)",
                        "scheduled_at_utc": (cutoff - timedelta(hours=2)).isoformat(),
                    },
                ],
            },
            "bls": {
                "source_url": "https://www.bls.gov/schedule/news_release/",
                "window_events": [
                    {
                        "agency": "BLS",
                        "title": "Employment Situation",
                        "scheduled_at_utc": (cutoff - timedelta(hours=1)).isoformat(),
                    }
                ],
            },
        },
    }
    verified_news = [
        # Older Sep 16 Fed FOMC release MUST NOT match future Nov 25 BEA event
        {
            "news_id": "fed_old",
            "title": "Federal Reserve issues FOMC statement",
            "url": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260916a.htm",
            "published_at_utc": "2026-09-16T18:00:00+00:00",
            "retrieved_at_utc": cutoff.isoformat(),
            "source_name": "Federal Reserve",
            "verification_status": "verified",
            "freshness_status": "stale_excluded",
            "news_kind": "primary_release",
            "affected_symbols": ["BTCUSDT", "XAUUSDT", "NDX100USDT"],
            "eligible_horizons": ["intraday", "swing_1_3d"],
            "bias_by_symbol": {"BTCUSDT": "neutral", "XAUUSDT": "neutral", "NDX100USDT": "neutral"},
        },
        # Genuine BEA Personal Income & Outlays release right after scheduled time
        {
            "news_id": "bea_pce",
            "title": "Personal Income and Outlays, October",
            "url": "https://www.bea.gov/news/2026/personal-income-and-outlays",
            "published_at_utc": (cutoff - timedelta(hours=1, minutes=55)).isoformat(),
            "retrieved_at_utc": cutoff.isoformat(),
            "source_name": "Bureau of Economic Analysis",
            "verification_status": "verified",
            "freshness_status": "fresh",
            "news_kind": "primary_release",
            "affected_symbols": ["BTCUSDT", "XAUUSDT", "NDX100USDT"],
            "eligible_horizons": ["intraday", "swing_1_3d"],
            "bias_by_symbol": {"BTCUSDT": "neutral", "XAUUSDT": "neutral", "NDX100USDT": "neutral"},
        },
        # Unrelated Fed press release at BLS event timestamp MUST NOT match BEA/BLS
        {
            "news_id": "fed_unrel",
            "title": "Federal Reserve Board announces supervisory update",
            "url": "https://www.federalreserve.gov/newsevents/pressreleases/other20261125a.htm",
            "published_at_utc": (cutoff - timedelta(minutes=50)).isoformat(),
            "retrieved_at_utc": cutoff.isoformat(),
            "source_name": "Federal Reserve",
            "verification_status": "verified",
            "freshness_status": "fresh",
            "news_kind": "primary_release",
            "affected_symbols": ["NDX100USDT"],
            "eligible_horizons": ["intraday"],
            "bias_by_symbol": {"BTCUSDT": "neutral", "XAUUSDT": "neutral", "NDX100USDT": "neutral"},
        },
    ]

    atoms, macro_sum = build_macro_evidence(
        alfred_series={},
        dxy_live=None,
        event_calendar=cal,
        verified_news=verified_news,
        vix_summary={},
        cutoff_utc=cutoff,
        cfg=cfg,
        frames_1h=shifted_frames,
    )
    by_title = {e["title"]: e for e in macro_sum["events"]}
    assert by_title["Gross Domestic Product (Second Estimate), Q3"]["actual"]["status"] == "not_yet_released"
    assert by_title["Gross Domestic Product (Second Estimate), Q3"]["post_release_market_reaction"] is None
    assert by_title["Personal Income and Outlays (PCE)"]["actual"]["status"] == "released_verified"
    assert by_title["Personal Income and Outlays (PCE)"]["consensus"]["status"] == "unavailable_in_official_public_feed"
    rxn = by_title["Personal Income and Outlays (PCE)"]["post_release_market_reaction"]
    assert rxn is not None and rxn["status"] == "observed_post_release_price_impulse"
    assert "NDX100USDT" in rxn["by_symbol"] and "ret_since_release_pct" in rxn["by_symbol"]["NDX100USDT"]
    assert by_title["Employment Situation"]["actual"]["status"] == "release_unverified"
    cal_atom = next(a for a in atoms if a["evidence_id"] == "macro:economic_events_calendar")
    assert cal_atom["freshness_status"] == "fresh"


def test_markdown_dedup_setup_bridge_and_chronos_run_window():
    cfg = load_judgment_config()
    bundle = generate_demo_judgment_inputs()
    cutoff = pd.Timestamp(bundle["cutoff_utc"]).to_pydatetime()
    # Run timestamp +25m after closed-bar cutoff inside the same hour MUST remain fresh (not future_excluded!)
    bundle["chronos_forecast"] = {
        "weights_ran": True,
        "model_id": "amazon/chronos-t5-tiny",
        "revision": "29d808298f1a62493e7b9a5e08529d0d930fa189",
        "run_at_utc": (cutoff + timedelta(minutes=25)).isoformat(),
        "forecasts": {
            sym: {
                "ok": True,
                "last_close_at_utc": cutoff.isoformat(),
                "quantile_10": [float(bundle["market_snapshot"]["instruments"][sym]["candles_closed_1h"][-1]["close"]) * 0.99] * 24,
                "quantile_50": [float(bundle["market_snapshot"]["instruments"][sym]["candles_closed_1h"][-1]["close"]) * 1.03] * 24,
                "quantile_90": [float(bundle["market_snapshot"]["instruments"][sym]["candles_closed_1h"][-1]["close"]) * 1.06] * 24,
            }
            for sym in ("BTCUSDT", "XAUUSDT", "NDX100USDT")
        },
    }
    frames_1h = validate_market_snapshot_coherence(bundle["market_snapshot"], cutoff_utc=cutoff)["frames_1h"]
    fc_atoms, _ = build_forecast_and_model_evidence(
        frames_1h=frames_1h,
        chronos_payload=bundle["chronos_forecast"],
        llm_opinion=None,
        cutoff_utc=cutoff,
        cfg=cfg,
    )
    btc_fc = next(a for a in fc_atoms if a["evidence_id"] == "forecast:btcusdt:chronos_neural")
    assert btc_fc["freshness_status"] == "fresh"
    assert validate_atom_temporal_bounds(btc_fc, cutoff_utc=cutoff)["freshness_status"] == "fresh"

    pkt = synthesize_judgment_packet(bundle, starting_cash_usdt=100.0)
    for sym in ("BTCUSDT", "XAUUSDT", "NDX100USDT"):
        assert "setup_bridge" in pkt["judgments"][sym]
        assert pkt["judgments"][sym]["setup_bridge"]["summary_cn"]

    ans = answer_judgment_query("今天黄金、BTC、纳指 100 看多还是看空？", packet=pkt)
    md = ans["markdown"]
    assert "今日结论总览" in md
    assert "可执行纸面交易计划" in md
    assert "研究→开仓等待条件 (`v2.2.1`)" in md
    assert "交易所微观结构盘口/资金费率" in md
    assert "Chronos 6h" in md
    assert "dir=" not in md and "role=" not in md
    assert "[forecast:" not in md and "[sentiment:" not in md and "[macro:" not in md

    # Verify explicit entry/stop-loss/take-profit/hold-hours rendering when paper_eligibility is eligible
    pkt_trade = deepcopy(pkt)
    pkt_trade["judgments"]["BTCUSDT"]["official_v2_2_1_signal"] = "short"
    pkt_trade["judgments"]["BTCUSDT"]["official_v2_2_1_setup_type"] = "range_reversion"
    pkt_trade["tide"]["instrument_positions"]["BTCUSDT"]["invalidation_levels"]["bb_mid"] = 60735.38
    pkt_trade["judgments"]["BTCUSDT"]["paper_eligibility"] = {
        "eligible": True,
        "status": "eligible",
        "label_cn": "是",
        "review_required": False,
        "review_triggers": [],
        "side": "short",
        "entry_plan": {"reference_price": 61466.8, "assumed_fill_with_slippage": 61436.07},
        "exit_plan": {
            "initial_stop_loss": 62231.23,
            "assumed_stop_fill": 62262.34,
            "fee_adjusted_breakeven_stop": 61362.39,
            "breakeven_ratchet_trigger_price": 60243.33,
            "trailing_or_target_rule": "bb_sma20_mid_or_18h",
            "max_hold_hours": 18,
        },
        "position_sizing": {"qty_contracts": 0.0011, "notional_usdt": 67.58, "leverage": 2.0, "initial_margin_usdt": 33.79, "free_cash_after_entry_usdt": 66.17},
        "cost_and_risk": {"risk_budget_usdt": 1.0, "estimated_normal_stop_loss_usdt": 0.99, "estimated_normal_stop_loss_pct_equity": 0.99, "estimated_isolated_liquidation": 91732.13},
    }
    md_trade = answer_judgment_query("今天黄金、BTC、纳指 100 看多还是看空？", packet=pkt_trade)["markdown"]
    assert "62231.23" in md_trade and "初始止损" in md_trade
    assert "60735.38" in md_trade and "止盈与保本" in md_trade
    assert "18 小时" in md_trade and "最长持仓" in md_trade
    assert "67.58" in md_trade and "USDT" in md_trade

    why = answer_judgment_query("为什么不做多？", packet=pkt)
    assert "距离 v2.2.1 正式触发条件" in why["markdown"]


def test_intraday_yahoo_coinmetrics_status_time_and_venue_microstructure_gate():
    from urllib.parse import unquote

    cfg = load_judgment_config()
    cutoff = datetime(2026, 11, 25, 10, 0, tzinfo=timezone.utc)
    ts_list = [int((cutoff - timedelta(hours=h)).timestamp()) for h in range(48, -1, -1)] + [
        int((cutoff + timedelta(hours=2)).timestamp()),  # Future bar MUST be stripped!
    ]

    def _yahoo_closes(start: float, step_last24: float, fut_val: float) -> list[float]:
        arr = [start for _ in range(len(ts_list) - 1)]
        for idx in range(len(arr) - 24, len(arr)):
            arr[idx] = start + step_last24 * ((idx - (len(arr) - 25)) / 24.0)
        arr.append(fut_val)
        return arr

    yahoo_map = {
        "DX-Y.NYB": _yahoo_closes(102.0, +1.5, 90.0),
        "^TNX": _yahoo_closes(41.5, +1.2, 30.0),  # +12 bps 24h (>6 bps threshold)
        "TIP": _yahoo_closes(108.0, -0.5, 120.0),
        "QQQ": _yahoo_closes(500.0, +2.0, 400.0),
        "QQEW": _yahoo_closes(120.0, +3.0, 90.0),
    }

    def yahoo_handler(request: httpx.Request) -> httpx.Response:
        sym = unquote(request.url.path.rsplit("/", 1)[-1])
        return httpx.Response(
            200,
            json={
                "chart": {
                    "result": [
                        {
                            "timestamp": ts_list,
                            "indicators": {"quote": [{"close": yahoo_map[sym]}]},
                        }
                    ]
                }
            },
        )

    with httpx.Client(transport=httpx.MockTransport(yahoo_handler)) as y_cli:
        im = fetch_yahoo_intraday_macro_and_breadth(bar_cutoff_utc=cutoff, client=y_cli)
    assert im["ok"] is True
    assert im["series"]["DX-Y.NYB"]["obs_at_utc"] == cutoff.isoformat()
    assert im["series"]["DX-Y.NYB"]["close"] == 103.5
    assert im["series"]["^TNX"]["close"] == 42.7
    assert im["qqew_minus_qqq_5d_pct"] is not None

    def cm_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "time": "2026-11-24T00:00:00.000000000Z",
                        "FlowInExNtv": "1000.0",
                        "FlowInExNtv-status": "flash",
                        "FlowInExNtv-status-time": (cutoff - timedelta(hours=2)).isoformat(),
                        "FlowOutExNtv": "3500.0",
                        "FlowOutExNtv-status": "flash",
                        "FlowOutExNtv-status-time": (cutoff - timedelta(hours=1)).isoformat(),
                        "FlowInExUSD": "90000000.0",
                        "FlowOutExUSD": "315000000.0",
                    },
                    {
                        # Future status-time MUST be rejected!
                        "time": "2026-11-25T00:00:00.000000000Z",
                        "FlowInExNtv": "9000.0",
                        "FlowInExNtv-status": "flash",
                        "FlowInExNtv-status-time": (cutoff + timedelta(hours=1)).isoformat(),
                        "FlowOutExNtv": "1000.0",
                        "FlowOutExNtv-status": "flash",
                        "FlowOutExNtv-status-time": (cutoff + timedelta(hours=1)).isoformat(),
                    },
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(cm_handler)) as cm_cli:
        ex_nf = fetch_coinmetrics_btc_exchange_flows(decision_cutoff_utc=cutoff, client=cm_cli)
    assert ex_nf["usable"] is True
    assert ex_nf["netflow_btc"] == -2500.0

    macro_atoms, macro_sum = build_macro_evidence(
        alfred_series={},
        dxy_live={"ok": False, "reason": "no_ice"},
        event_calendar=None,
        verified_news=[],
        vix_summary={},
        cutoff_utc=cutoff,
        cfg=cfg,
        intraday_macro=im,
    )
    m_by_id = {a["evidence_id"]: a for a in macro_atoms}
    assert m_by_id["macro:ice_dxy"]["freshness_status"] == "missing"
    assert m_by_id["macro:real_yield_intraday"]["freshness_status"] == "missing"
    assert m_by_id["macro:intraday_dollar_proxy:DX_Y_NYB"]["freshness_status"] == "fresh"
    assert m_by_id["macro:intraday_dollar_proxy:DX_Y_NYB"]["direction_by_symbol"]["BTCUSDT"] == "bearish"
    assert m_by_id["macro:intraday_rates_tips_proxy:TNX_TIP"]["freshness_status"] == "fresh"
    assert macro_sum["ice_dxy_available"] is False
    assert macro_sum["intraday_dollar_dx_y_nyb"] == 103.5
    assert macro_sum["intraday_tnx_pct"] == 4.27
    assert macro_sum["intraday_tnx_delta_24h_bps"] == 12.0

    # Also verify direct percent-scale ^TNX quotes (e.g. 4.904% with +0.12 = +12 bps) normalize identically
    from piia.intelligence.judgment.evidence import _normalize_tnx_yield
    assert _normalize_tnx_yield({"ok": True, "close": 4.904, "delta_24h": 0.12}) == (4.904, 12.0)

    bundle = generate_demo_judgment_inputs(now=cutoff)
    sent_atoms, _, _ = build_sentiment_flow_evidence(
        market_snapshot=bundle["market_snapshot"],
        alfred_series={},
        fear_greed_ctx=None,
        etf_ctx=None,
        gold_cot_ctx=None,
        onchain_ctx=None,
        exchange_netflow_ctx=ex_nf,
        gold_etf_ctx=None,
        ndx_edgar_filings=[],
        polymarket_ctx=None,
        cutoff_utc=cutoff,
        cfg=cfg,
        intraday_macro=im,
    )
    s_by_id = {a["evidence_id"]: a for a in sent_atoms}
    assert s_by_id["sentiment:btc:onchain_exchange_netflow"]["freshness_status"] == "fresh"
    assert s_by_id["sentiment:btc:onchain_exchange_netflow"]["direction_by_symbol"]["BTCUSDT"] == "bullish"
    assert s_by_id["sentiment:btc:onchain_network_activity"]["freshness_status"] == "missing"
    assert s_by_id["sentiment:ndx:tech_relative_strength"]["value_summary"]["qqew_minus_qqq_5d_pct"] == im["qqew_minus_qqq_5d_pct"]

    # Bitget venue microstructure: XAUUSDT 4h default is normal; BTCUSDT mark/last dislocation >= 50 bps triggers pending_review
    def bg_handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "tickers" in url:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"symbol": "BTCUSDT", "lastPr": "100000", "markPrice": "100650", "indexPrice": "100100"},
                        {"symbol": "XAUUSDT", "lastPr": "2650", "markPrice": "2651", "indexPrice": "2650.5"},
                        {"symbol": "NDX100USDT", "lastPr": "21000", "markPrice": "21002", "indexPrice": "21001"},
                    ]
                },
            )
        sym = dict(request.url.params).get("symbol")
        intervals = {"BTCUSDT": "2", "XAUUSDT": "4", "NDX100USDT": "8"}
        return httpx.Response(200, json={"data": [{"symbol": sym, "fundingRate": "0.0001", "fundingRateInterval": intervals[sym], "nextUpdate": "1795651200000"}]})

    with httpx.Client(transport=httpx.MockTransport(bg_handler)) as bg_cli:
        vm = fetch_bitget_venue_microstructure(("BTCUSDT", "XAUUSDT", "NDX100USDT"), decision_cutoff_utc=cutoff, client=bg_cli)
    assert vm["ok"] is True

    xau_audit = _audit_venue_microstructure("XAUUSDT", venue_micro=vm, cfg=cfg)
    assert xau_audit["dynamic_interval_changed"] is False
    assert xau_audit["review_triggers"] == []

    btc_audit = _audit_venue_microstructure("BTCUSDT", venue_micro=vm, cfg=cfg)
    assert btc_audit["severe_dislocation"] is True
    assert btc_audit["dynamic_interval_changed"] is True
    v2_cand = {
        "official_v2_2_1_signal": "long",
        "setup_type": "trend_pullback",
        "paper_eligibility": {"eligible": True, "status": "eligible", "label_cn": "是", "review_required": False, "review_triggers": [], "side": "long", "blocking_reasons": []},
    }
    pe_gated, trigs = _apply_v3_paper_review_gate(
        "BTCUSDT",
        v2_ctrl=v2_cand,
        pos={"close": 100000.0, "metrics": {}, "invalidation_levels": {}},
        atoms=[],
        news_enabled=True,
        venue_audit=btc_audit,
    )
    assert pe_gated["status"] == "pending_review"
    assert pe_gated["eligible"] is False
    assert any("venue_mark_last_dislocation" == t["trigger"] for t in trigs)
    assert any("venue_dynamic_funding_interval_changed" == t["trigger"] for t in trigs)


def test_archived_5_domain_bundles_append_only_replay_cli_and_api(tmp_path, monkeypatch):
    from piia.intelligence.api import app
    from piia.intelligence.cli import main as piia_main

    arch_db = tmp_path / "arch_replay.sqlite"
    for h in (0, 1):
        b = generate_demo_judgment_inputs(step_shift_hours=h)
        b["is_synthetic_demo"] = False
        b["intraday_macro"] = {"ok": True, "retrieved_at_utc": b["cutoff_utc"], "series": {}, "qqew_minus_qqq_5d_pct": 0.9}
        b["exchange_netflow"] = {"usable": False, "reason": "none_in_fixture"}
        b["venue_microstructure"] = {"ok": True, "symbols": {"XAUUSDT": {"funding_interval_hours": 4, "mark_vs_last_bps": 1.0}}}
        res = record_judgment_cycle(b, db_path=arch_db, starting_cash=100.0)
        assert res["archived_bundle_sha256"]

    # Calling again for hour 0 stays idempotent and append-only triggers protect archived_input_bundles
    b0_again = generate_demo_judgment_inputs(step_shift_hours=0)
    b0_again["is_synthetic_demo"] = False
    res_dup = record_judgment_cycle(b0_again, db_path=arch_db, starting_cash=100.0)
    assert res_dup["status"] == "already_recorded_for_hour"

    conn = sqlite3.connect(arch_db)
    try:
        cnt = conn.execute("SELECT COUNT(*) FROM archived_input_bundles").fetchone()[0]
        assert cnt == 2
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("UPDATE archived_input_bundles SET bundle_sha256='tamper'")
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("DELETE FROM archived_input_bundles")
    finally:
        conn.close()

    rep = replay_archived_judgment_ledger(arch_db, starting_cash=100.0)
    assert rep["status"] == "replayed_archived_5_domain_bundles"
    assert rep["replay_non_blind_5_domains"] is True
    assert rep["archived_bundles_count"] == 2
    assert rep["starting_cash_usdt"] == 100.0
    assert "domain_ablation" in rep

    cli_db = tmp_path / "cli_arch.sqlite"
    cli_out = tmp_path / "cli_rep.json"
    assert piia_main(["judgment-capture-hourly", "--demo", "--starting-cash", "100", "--db", str(cli_db)]) == 0
    assert piia_main(["judgment-replay-archived", "--demo", "--starting-cash", "100", "--db", str(cli_db), "--out", str(cli_out)]) == 0
    assert json.loads(cli_out.read_text(encoding="utf-8"))["archived_bundles_count"] == 1

    monkeypatch.setenv("PIIA_JUDGMENT_V3_DEMO_LEDGER_PATH", str(cli_db))
    monkeypatch.setenv("PIIA_DATABASE_URL", f"sqlite:///{(tmp_path / 'api_arch.sqlite').as_posix()}")
    with TestClient(app) as client:
        r = client.get("/judgment-v3/replay-archived", params={"demo": "true", "starting_cash": 100})
        assert r.status_code == 200
        assert r.json()["archived_bundles_count"] == 1


def _build_minimal_gld_xlsx(rows: list[list[str]]) -> bytes:
    """Create a valid minimal OOXML XLSX matching State Street GLD NAV History layout."""
    import io
    import zipfile
    from xml.sax.saxutils import escape

    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/sharedStrings.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    wb = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="NAV History" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )
    wb_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" '
        'Target="sharedStrings.xml"/>'
        "</Relationships>"
    )
    shared: list[str] = []
    shared_idx: dict[str, int] = {}

    def _s_idx(val: str) -> int:
        if val not in shared_idx:
            shared_idx[val] = len(shared)
            shared.append(val)
        return shared_idx[val]

    sheet_rows_xml = []
    for r_i, row in enumerate(rows, start=1):
        cells_xml = []
        for c_i, cell_val in enumerate(row):
            col_letter = chr(ord("A") + c_i)
            idx = _s_idx(str(cell_val))
            cells_xml.append(f'<c r="{col_letter}{r_i}" t="s"><v>{idx}</v></c>')
        sheet_rows_xml.append(f'<row r="{r_i}">{"".join(cells_xml)}</row>')

    ss_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="{len(shared)}" uniqueCount="{len(shared)}">'
        + "".join(f"<si><t>{escape(s)}</t></si>" for s in shared)
        + "</sst>"
    )
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(sheet_rows_xml)}</sheetData>"
        "</worksheet>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", wb)
        zf.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        zf.writestr("xl/sharedStrings.xml", ss_xml)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)
    return buf.getvalue()


def test_spdr_gld_xlsx_fetcher_and_gold_etf_sentiment_direction():
    import httpx

    cfg = load_judgment_config()
    cutoff = datetime(2026, 11, 26, 6, 0, tzinfo=timezone.utc)
    xlsx_bytes = _build_minimal_gld_xlsx(
        [
            ["Fund Name", "SPDR Gold Shares", "", ""],
            ["Date", "NAV", "Shares Outstanding", "Total Net Assets"],
            ["26-Nov-2026", "320.0", "362000000", "115840000000"],  # 22:00 UTC > cutoff -> excluded!
            ["25-Nov-2026", "317.5", "360600000", "114490500000"],
            ["24-Nov-2026", "315.0", "360200000", "113463000000"],
            ["21-Nov-2026", "314.0", "360000000", "113040000000"],
            ["20-Nov-2026", "312.0", "359800000", "112257600000"],
        ]
    )
    transport = httpx.MockTransport(lambda req: httpx.Response(200, content=xlsx_bytes))
    with httpx.Client(transport=transport) as client:
        gld = fetch_spdr_gld_etf_flows(decision_cutoff_utc=cutoff, client=client)

    assert gld["usable"] is True
    assert gld["asof_date"] == "2026-11-25"
    assert gld["coverage_scope"] == "spdr_gld_single_flagship_etf"
    assert gld["delta_1d_shares"] == 400000
    assert gld["delta_3d_shares"] == 800000
    assert gld["net_flow_1d_musd"] == 127.0
    assert gld["net_flow_3d_musd"] == 254.0

    inputs = generate_demo_judgment_inputs(now=cutoff)
    common_kw = dict(
        market_snapshot=inputs["market_snapshot"],
        alfred_series=inputs["alfred_series"],
        fear_greed_ctx=inputs["fear_greed"],
        etf_ctx=inputs["etf_flows"],
        gold_cot_ctx=inputs["gold_cot"],
        onchain_ctx=inputs["onchain"],
        exchange_netflow_ctx=inputs["exchange_netflow"],
        ndx_edgar_filings=inputs["ndx_edgar_filings"],
        polymarket_ctx=inputs["polymarket"],
        cutoff_utc=cutoff,
        cfg=cfg,
    )
    atoms, _, sent_summary = build_sentiment_flow_evidence(gold_etf_ctx=gld, **common_kw)
    gld_atom = next(a for a in atoms if a["evidence_id"] == "sentiment:xau:gold_etf_flows")
    assert gld_atom["freshness_status"] == "lagged_usable"
    assert gld_atom["direction_by_symbol"]["XAUUSDT"] == "bullish"
    assert sent_summary["gold_etf_3d_musd"] == 254.0

    # Test bearish and neutral directions around gold_etf_3d_musd_min (150M)
    atoms_bear, _, _ = build_sentiment_flow_evidence(
        gold_etf_ctx=dict(gld, net_flow_3d_musd=-220.0, net_tonnage_or_flow=-220.0),
        **common_kw,
    )
    assert next(a for a in atoms_bear if a["evidence_id"] == "sentiment:xau:gold_etf_flows")["direction_by_symbol"]["XAUUSDT"] == "bearish"

    atoms_flat, _, _ = build_sentiment_flow_evidence(
        gold_etf_ctx=dict(gld, net_flow_3d_musd=40.0, net_tonnage_or_flow=40.0),
        **common_kw,
    )
    assert next(a for a in atoms_flat if a["evidence_id"] == "sentiment:xau:gold_etf_flows")["direction_by_symbol"]["XAUUSDT"] == "neutral"


def test_us_treasury_real_yield_csv_pit_bounds_and_audit_gap_split():
    import httpx
    from piia.intelligence.judgment.chat_workbench import answer_judgment_query
    from piia.intelligence.judgment.synthesizer import synthesize_judgment_packet

    cfg = load_judgment_config()
    cutoff = datetime(2026, 11, 26, 6, 0, tzinfo=timezone.utc)
    csv_body = (
        'Date,"5 YR","7 YR","10 YR","20 YR","30 YR"\n'
        '11/26/2026,2.10,2.40,2.80,2.90,2.95\n'  # 22:00 UTC > cutoff -> excluded!
        '11/25/2026,2.05,2.35,2.73,2.88,2.92\n'
        '11/24/2026,2.06,2.36,2.75,2.89,2.93\n'
    )
    transport = httpx.MockTransport(lambda req: httpx.Response(200, text=csv_body))
    with httpx.Client(transport=transport) as client:
        ry = fetch_us_treasury_daily_real_yield(decision_cutoff_utc=cutoff, client=client)

    assert ry["ok"] is True
    assert ry["asof_date"] == "2026-11-25"
    assert [o["observation_date"] for o in ry["observations"]] == ["2026-11-24", "2026-11-25"]

    inputs = generate_demo_judgment_inputs(now=cutoff)
    # Inject Treasury observations with explicit available_at_utc on top of older DFII10 points
    base_obs = list((inputs["alfred_series"]["DFII10"].get("observations") or [])[:-2])
    inputs["alfred_series"]["DFII10"]["observations"] = base_obs + ry["observations"]
    macro_atoms, macro_summary = build_macro_evidence(
        alfred_series=inputs["alfred_series"],
        dxy_live=inputs["dxy_live"],
        event_calendar=inputs["event_calendar"],
        verified_news=[],
        vix_summary={},
        cutoff_utc=cutoff,
        cfg=cfg,
    )

    dfii_atom = next(a for a in macro_atoms if a["evidence_id"] == "macro:real_yield_daily:DFII10")
    ry_intraday_atom = next(a for a in macro_atoms if a["evidence_id"] == "macro:real_yield_intraday")
    assert dfii_atom["freshness_status"] == "lagged_usable"
    assert dfii_atom["value_summary"]["observation_date"] == "2026-11-25"
    assert dfii_atom["value_summary"]["value_pct"] == 2.73
    assert macro_summary["real_yield_dfii10_pct"] == 2.73
    # Daily Treasury DFII10 must never falsely claim to be intraday TIPS realtime
    assert ry_intraday_atom["freshness_status"] == "missing"

    packet = synthesize_judgment_packet(inputs, starting_cash_usdt=100.0)
    ans = answer_judgment_query("今天黄金、BTC、纳指 100 看多还是看空？", packet=packet)
    md = ans["markdown"]
    assert "已用代理替代(无付费直连)" in md
    assert "ICE DXY(用DX-Y.NYB代理)" in md
    assert "盘中TIPS(用^TNX/TIP代理)" in md
    assert "真缺失：" in md

