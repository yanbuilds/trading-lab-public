"""Synthetic gate tests; these fixtures are not market evidence."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone

from piia.intelligence import chat_decision_gate as gate


def test_gate_rechecks_time_binding_risk_and_forecast(monkeypatch):
    now = datetime(2026, 11, 16, tzinfo=timezone.utc)
    request = {"snapshot_ok": True, "cutoff_at_utc": now.isoformat(), "request_hash": "hash",
               "paper_risk_context": {"available": True, "owner_config_hash": "risk"},
               "instruments": {"BTCUSDT": {"ticker": {"observed_at_utc": now.isoformat()},
                                           "all_closed_candles": [{"fixture": True}]}}}
    state = {"request_hash": "hash", "run_id": "fixture", "forecast_weights_ran": True,
             "packet_recommendation_blockers": [], "instrument_recommendation_blockers": {"BTCUSDT": []}}
    archive = {"request": deepcopy(request), "run_id": "fixture", "order_permissions": False,
               "created_at_utc": now.isoformat()}
    monkeypatch.setattr(gate, "candle_quality", lambda *_: [])
    monkeypatch.setattr(gate, "_verify_cycle_binding", lambda *_: None)
    def check(**kwargs):
        args = dict(now=now, state=state, restricted={}, archive=archive,
                    current_risk_hash="risk", forecast_bound=True)
        args.update(kwargs)
        return gate.assess_chat_evidence(request, "BTCUSDT", **args)
    assert not check()["review_blocked"]
    assert "snapshot_missing_or_stale" in check(now=now + timedelta(minutes=6))["blockers"]
    assert "paper_risk_changed_or_unavailable" in check(current_risk_hash="changed")["blockers"]
    assert "full_research_evidence_unavailable" in check(archive=None)["blockers"]
    assert "chronos_forecast_unbound" in check(forecast_bound=False)["blockers"]
    altered = {**archive, "request": {**request, "request_hash": "other"}}
    assert "research_evidence_binding_invalid" in check(archive=altered)["blockers"]
    early = {**archive, "created_at_utc": (now - timedelta(seconds=1)).isoformat()}
    assert "research_evidence_binding_invalid" in check(archive=early)["blockers"]
    request["instruments"]["BTCUSDT"]["ticker"]["observed_at_utc"] = (now + timedelta(seconds=1)).isoformat()
    archive["request"] = deepcopy(request)
    assert "research_evidence_binding_invalid" in check(now=now + timedelta(seconds=2))["blockers"]


def test_gate_downgrades_review_without_mutating_model_result():
    model = {"synthesizer": {"suggested_action": "review_paper_plan", "summary": "fixture"}}
    result = gate.apply_chat_gate(model, {"review_blocked": True, "order_permissions": False})
    assert model["synthesizer"]["suggested_action"] == "review_paper_plan"
    assert result["synthesizer"]["suggested_action"] == "need_more_evidence"
    assert result["model_original_suggested_action"] == "review_paper_plan"
    assert result["deterministic_review_gate"]["order_permissions"] is False


def test_unverified_claims_block_review_even_when_market_gate_passed():
    model = {"blocked_claims": [], "injection_flags": [],
             "researcher": {"claims": [{"status": "unverified"}]},
             "critic": {"claims": [{"status": "unverified"}]},
             "synthesizer": {"claims": [{"status": "unverified"}], "suggested_action": "review_paper_plan"}}
    result = gate.apply_chat_gate(model, {"review_blocked": False, "blockers": [], "order_permissions": False})
    assert result["deterministic_review_gate"]["review_blocked"]
    assert "model_claims_not_verified:researcher" in result["deterministic_review_gate"]["blockers"]
    assert result["synthesizer"]["suggested_action"] == "need_more_evidence"
    model["blocked_claims"] = [{"reason": "evidence_id_not_on_revision"}]
    model["injection_flags"] = ["fixture warning"]
    result = gate.apply_chat_gate(model, {"review_blocked": False, "blockers": []})
    assert "model_blocked_claims_present" in result["deterministic_review_gate"]["blockers"]
    assert "model_injection_flags_present" in result["deterministic_review_gate"]["blockers"]


def test_usable_external_context_must_not_postdate_archive(monkeypatch):
    now = datetime(2026, 11, 16, tzinfo=timezone.utc)
    request = {"snapshot_ok": True, "cutoff_at_utc": now.isoformat(), "request_hash": "hash",
               "paper_risk_context": {"available": True, "owner_config_hash": "risk"},
               "instruments": {"BTCUSDT": {"ticker": {"observed_at_utc": now.isoformat()}}},
               "fear_greed_context": {"usable_for_research": True,
                                      "retrieved_at_utc": (now + timedelta(seconds=1)).isoformat()}}
    state = {"request_hash": "hash", "run_id": "fixture", "packet_recommendation_blockers": [],
             "instrument_recommendation_blockers": {"BTCUSDT": []}}
    archive = {"request": deepcopy(request), "run_id": "fixture", "order_permissions": False,
               "created_at_utc": now.isoformat()}
    monkeypatch.setattr(gate, "candle_quality", lambda *_: [])
    monkeypatch.setattr(gate, "_verify_cycle_binding", lambda *_: None)
    result = gate.assess_chat_evidence(request, "BTCUSDT", now=now + timedelta(seconds=2), state=state,
                                       restricted={}, archive=archive, current_risk_hash="risk", forecast_bound=False)
    assert "external_evidence_time_invalid:fear_greed_context" in result["blockers"]
    context = request["fear_greed_context"]
    context["retrieved_at_utc"] = (now - timedelta(seconds=2)).isoformat()
    context["completed_at_utc"] = (now + timedelta(seconds=1)).isoformat()
    archive["request"] = deepcopy(request)
    result = gate.assess_chat_evidence(request, "BTCUSDT", now=now + timedelta(seconds=2), state=state,
                                       restricted={}, archive=archive, current_risk_hash="risk", forecast_bound=False)
    assert "external_evidence_time_invalid:fear_greed_context" in result["blockers"]
    archive["created_at_utc"] = (now + timedelta(seconds=2)).isoformat()
    result = gate.assess_chat_evidence(request, "BTCUSDT", now=now + timedelta(seconds=3), state=state,
                                       restricted={}, archive=archive, current_risk_hash="risk", forecast_bound=False)
    assert "external_evidence_time_invalid:fear_greed_context" in result["blockers"]
    archive["created_at_utc"] = now.isoformat()
    context["completed_at_utc"] = (now - timedelta(seconds=1)).isoformat()
    archive["request"] = deepcopy(request)
    result = gate.assess_chat_evidence(request, "BTCUSDT", now=now, state=state,
                                       restricted={}, archive=archive, current_risk_hash="risk", forecast_bound=False)
    assert "external_evidence_time_invalid:fear_greed_context" not in result["blockers"]


def test_unverified_latest_etf_session_blocks_btc_only(monkeypatch):
    from piia.intelligence.research_cycle import _critical_blockers
    request = {"snapshot_ok": True, "instruments": {}, "missing_datasets": []}
    etf = {"usable_for_btc_research": True, "latest_exchange_session_verified": False}
    result = _critical_blockers(request, {"weights_ran": False}, None, etf=etf)
    reason = "btc_etf_latest_exchange_session_unverified"
    assert reason in result["instruments"]["BTCUSDT"]


    assert reason not in result["instruments"]["XAUUSDT"]
    assert reason not in result["instruments"]["NDX100USDT"]
    now = datetime(2026, 11, 16, tzinfo=timezone.utc)
    request.update(cutoff_at_utc=now.isoformat(), etf_flow_context=etf,
                   instruments={"BTCUSDT": {"ticker": {"observed_at_utc": now.isoformat()}}},
                   paper_risk_context={"available": True, "owner_config_hash": "risk"})
    state = {"request_hash": "hash", "run_id": "fixture", "packet_recommendation_blockers": [],
             "instrument_recommendation_blockers": {"BTCUSDT": []}}
    request["request_hash"] = "hash"
    archive = {"request": deepcopy(request), "run_id": "fixture", "order_permissions": False,
               "created_at_utc": now.isoformat()}
    monkeypatch.setattr(gate, "candle_quality", lambda *_: [])
    monkeypatch.setattr(gate, "_verify_cycle_binding", lambda *_: None)
    result = gate.assess_chat_evidence(request, "BTCUSDT", now=now, state=state, restricted={},
                                       archive=archive, current_risk_hash="risk", forecast_bound=False)
    assert reason in result["blockers"]


def test_btc_sentiment_requires_saved_reproducible_snapshot(monkeypatch, tmp_path):
    import httpx
    from piia.intelligence.fear_greed import fetch_fear_greed
    now = datetime(2026, 11, 16, tzinfo=timezone.utc)
    row = {"value": "25", "value_classification": "Fear", "timestamp": str(int(now.timestamp()))}
    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"data": [row]}, request=req)))
    context = fetch_fear_greed(now=now, client=client, snapshot_dir=tmp_path / "sentiment")
    request = {"snapshot_ok": True, "cutoff_at_utc": now.isoformat(), "request_hash": "fixture",
               "paper_risk_context": {"available": True, "owner_config_hash": "risk"},
               "instruments": {"BTCUSDT": {"ticker": {"observed_at_utc": now.isoformat()}}},
               "fear_greed_context": context}
    state = {"request_hash": "fixture", "run_id": "fixture", "paths": {"run_dir": str(tmp_path)},
             "packet_recommendation_blockers": [], "instrument_recommendation_blockers": {"BTCUSDT": []}}
    archive = {"request": deepcopy(request), "run_id": "fixture", "order_permissions": False, "created_at_utc": now.isoformat()}
    monkeypatch.setattr(gate, "candle_quality", lambda *_: [])
    monkeypatch.setattr(gate, "_verify_cycle_binding", lambda *_: None)
    def check():
        return gate.assess_chat_evidence(request, "BTCUSDT", now=now, state=state, restricted={},
                                         archive=archive, current_risk_hash="risk", forecast_bound=False)
    assert "btc_sentiment_raw_snapshot_invalid_or_stale" not in check()["blockers"]
    (tmp_path / "sentiment" / "fear_greed.json").unlink()
    assert "btc_sentiment_raw_snapshot_invalid_or_stale" in check()["blockers"]


def test_liquidity_requires_collection_and_assessment_times_before_archive(monkeypatch):
    now = datetime(2026, 11, 16, 12, tzinfo=timezone.utc)
    context = {"retrieved_at_utc": (now - timedelta(minutes=3)).isoformat(),
               "inputs_collected_at_utc": (now - timedelta(minutes=1)).isoformat(),
               "cutoff_at_utc": (now - timedelta(minutes=1)).isoformat(),
               "completed_at_utc": (now - timedelta(seconds=30)).isoformat()}
    request = {"snapshot_ok": True, "cutoff_at_utc": now.isoformat(), "request_hash": "fixture",
               "paper_risk_context": {"available": True, "owner_config_hash": "risk"},
               "instruments": {"BTCUSDT": {"ticker": {"observed_at_utc": now.isoformat()}}},
               "liquidity_positioning_context": context}
    state = {"request_hash": "fixture", "run_id": "fixture", "packet_recommendation_blockers": [],
             "instrument_recommendation_blockers": {"BTCUSDT": []}}
    archive = {"request": deepcopy(request), "run_id": "fixture", "order_permissions": False, "created_at_utc": now.isoformat()}
    monkeypatch.setattr(gate, "candle_quality", lambda *_: [])
    monkeypatch.setattr(gate, "_verify_cycle_binding", lambda *_: None)
    def check():
        archive["request"] = deepcopy(request)
        return gate.assess_chat_evidence(request, "BTCUSDT", now=now, state=state, restricted={},
                                         archive=archive, current_risk_hash="risk", forecast_bound=False)
    assert "liquidity_evidence_time_unverified" not in check()["blockers"]
    del context["inputs_collected_at_utc"]
    assert "liquidity_evidence_time_unverified" in check()["blockers"]
    context["inputs_collected_at_utc"] = now.isoformat()
    assert "liquidity_evidence_time_unverified" in check()["blockers"]


def test_etf_freshness_is_rechecked_after_market_close(monkeypatch):
    captured = datetime(2026, 11, 27, 17, 59, tzinfo=timezone.utc)
    now = captured + timedelta(minutes=2)
    request = {"snapshot_ok": True, "cutoff_at_utc": captured.isoformat(), "request_hash": "hash",
               "paper_risk_context": {"available": True, "owner_config_hash": "risk"},
               "instruments": {"BTCUSDT": {"ticker": {"observed_at_utc": captured.isoformat()}}},
               "etf_flow_context": {"usable_for_btc_research": True, "latest_exchange_session_verified": True,
                                    "asof_complete_flow_date": "2026-11-25", "retrieved_at_utc": captured.isoformat(),
                                    "completed_at_utc": captured.isoformat()}}
    state = {"request_hash": "hash", "run_id": "fixture", "packet_recommendation_blockers": [],
             "instrument_recommendation_blockers": {"BTCUSDT": []}}
    archive = {"request": deepcopy(request), "run_id": "fixture", "order_permissions": False,
               "created_at_utc": captured.isoformat()}
    monkeypatch.setattr(gate, "candle_quality", lambda *_: [])
    monkeypatch.setattr(gate, "_verify_cycle_binding", lambda *_: None)
    result = gate.assess_chat_evidence(request, "BTCUSDT", now=now, state=state, restricted={},
                                       archive=archive, current_risk_hash="risk", forecast_bound=False)
    assert "btc_etf_latest_exchange_session_unverified" in result["blockers"]


def test_fresh_packet_cannot_hide_old_onchain_snapshot(monkeypatch):
    now = datetime(2026, 11, 16, tzinfo=timezone.utc)
    request = {"snapshot_ok": True, "cutoff_at_utc": now.isoformat(), "request_hash": "hash",
               "paper_risk_context": {"available": True, "owner_config_hash": "risk"},
               "instruments": {"BTCUSDT": {"ticker": {"observed_at_utc": now.isoformat()}},
                               "XAUUSDT": {"ticker": {"observed_at_utc": now.isoformat()}}},
               "onchain_context": {"usable_for_btc_research": True,
                    "retrieved_at_utc": (now - timedelta(minutes=6)).isoformat(),
                    "completed_at_utc": (now - timedelta(minutes=5)).isoformat(),
                    "metrics": {"last_block": {"observation_at_utc": (now - timedelta(minutes=10)).isoformat()}}}}
    state = {"request_hash": "hash", "run_id": "fixture", "packet_recommendation_blockers": [],
             "instrument_recommendation_blockers": {"BTCUSDT": [], "XAUUSDT": []}}
    archive = {"request": deepcopy(request), "run_id": "fixture", "order_permissions": False,
               "created_at_utc": now.isoformat()}
    monkeypatch.setattr(gate, "candle_quality", lambda *_: [])
    monkeypatch.setattr(gate, "_verify_cycle_binding", lambda *_: None)
    args = dict(now=now, state=state, restricted={}, archive=archive, current_risk_hash="risk", forecast_bound=False)
    assert "btc_onchain_snapshot_or_tip_stale" in gate.assess_chat_evidence(request, "BTCUSDT", **args)["blockers"]
    assert "btc_onchain_snapshot_or_tip_stale" not in gate.assess_chat_evidence(request, "XAUUSDT", **args)["blockers"]
    request["onchain_context"]["retrieved_at_utc"] = (now - timedelta(minutes=2)).isoformat()
    request["onchain_context"]["completed_at_utc"] = (now - timedelta(minutes=1)).isoformat()
    archive["request"] = deepcopy(request)
    assert "btc_onchain_snapshot_or_tip_stale" not in gate.assess_chat_evidence(request, "BTCUSDT", **args)["blockers"]


def test_macro_raw_snapshot_binding_is_required(monkeypatch, tmp_path):
    from piia.intelligence.macro_data import parse_vintage_csv, _save_raw_snapshot
    now = datetime(2026, 11, 16, 10, tzinfo=timezone.utc)
    raw = b"observation_date,DGS10_20261114\n2026-11-12,4.2\n"
    item = parse_vintage_csv(raw, "DGS10", "2026-11-14", now)
    raw_dir = tmp_path / "macro" / "raw"
    _save_raw_snapshot(item, raw, raw_dir)
    request = {"snapshot_ok": True, "cutoff_at_utc": now.isoformat(), "request_hash": "fixture",
               "paper_risk_context": {"available": True, "owner_config_hash": "risk"},
               "instruments": {"BTCUSDT": {"ticker": {"observed_at_utc": now.isoformat()}}},
               "macro_context": {"series": {"DGS10": item}}}
    state = {"request_hash": "fixture", "run_id": "fixture", "paths": {"run_dir": str(tmp_path)},
             "packet_recommendation_blockers": [], "instrument_recommendation_blockers": {"BTCUSDT": []}}
    archive = {"request": deepcopy(request), "run_id": "fixture", "order_permissions": False,
               "created_at_utc": now.isoformat()}
    monkeypatch.setattr(gate, "candle_quality", lambda *_: [])
    monkeypatch.setattr(gate, "_verify_cycle_binding", lambda *_: None)
    def check():
        return gate.assess_chat_evidence(request, "BTCUSDT", now=now, state=state, restricted={},
                                         archive=archive, current_risk_hash="risk", forecast_bound=False)
    assert not check()["review_blocked"]
    old_time = now - timedelta(days=20)
    old_raw = b"observation_date,DGS10_20261025\n2026-10-24,4.2\n"
    old_item = parse_vintage_csv(old_raw, "DGS10", "2026-10-25", old_time)
    _save_raw_snapshot(old_item, old_raw, raw_dir)
    assert not old_item["stale"]  # Fresh when fetched, stale at review.
    request["macro_context"]["series"]["DGS10"] = old_item
    archive["request"] = deepcopy(request)
    assert "macro_raw_snapshot_unavailable_or_invalid" in check()["blockers"]
    later_item = dict(item, retrieved_at_utc=(now + timedelta(seconds=1)).isoformat())
    request["macro_context"]["series"]["DGS10"] = later_item
    archive["request"] = deepcopy(request)
    assert "macro_raw_snapshot_unavailable_or_invalid" in check()["blockers"]
    request["macro_context"]["series"]["DGS10"] = item
    archive["request"] = deepcopy(request)
    (raw_dir / item["raw_snapshot_file"]).write_bytes(b"modified")
    assert "macro_raw_snapshot_unavailable_or_invalid" in check()["blockers"]
    (raw_dir / item["raw_snapshot_file"]).unlink()
    assert "macro_raw_snapshot_unavailable_or_invalid" in check()["blockers"]
