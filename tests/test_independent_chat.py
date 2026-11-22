"""Synthetic chat fixtures; no paid model or live trade calls."""
import copy
import json
from piia.intelligence.agents.io import RoleOutput
from piia.intelligence.agents import independent_context as module
from piia.intelligence.chat_context import compact_contexts, compact_forecast, compact_macro


def test_same_model_first_passes_share_evidence_not_each_others_opinions(monkeypatch):
    calls = []
    monkeypatch.setattr(module, "model_configured", lambda: True)
    class Provider:
        usage_records = []
        def complete(self, role, payload):
            calls.append((role, copy.deepcopy(payload)))
            return RoleOutput(role=role, actor="fixture", generation="model", summary=role,
                              claims=[{"text": "claim", "evidence_ids": ["BTC"], "status": "supported"}])
    monkeypatch.setattr(module, "OfficialAPIProvider", Provider)
    payload = {"statement": "question", "evidence": [{"id": "BTC", "raw": "fixture"}],
               "previous_discussion": [{"question": "prior", "prior_model_opinions_not_evidence": {"critic": "secret prior opinion"}}],
               "researcher": {"summary": "must not leak"}}
    out = module.run_independent_context_research(payload=payload, legal_evidence_ids=["BTC"])
    assert out.status == "completed" and out.independent_llm_review == "NOT_WIRED"
    assert calls[0][1] == calls[1][1]
    assert "researcher" not in calls[1][1] and "critic" not in calls[1][1]
    assert calls[1][1]["previous_discussion"] == [{"question": "prior"}]
    assert calls[2][1]["researcher"]["summary"] == "researcher"
    assert calls[2][1]["critic"]["summary"] == "critic"
    assert out.critic.claims[0].status == "unverified"
    assert payload["researcher"]["summary"] == "must not leak"


def test_model_unconfigured_and_partial_failure_remain_explicit(monkeypatch):
    monkeypatch.setattr(module, "model_configured", lambda: False)
    monkeypatch.setattr(module, "OfficialAPIProvider", lambda: (_ for _ in ()).throw(AssertionError("called")))
    result = module.run_independent_context_research(payload={"evidence": []}, legal_evidence_ids=[])
    assert result.status == "needs_model_configuration" and result.independent_llm_review == "NOT_WIRED"
    monkeypatch.setattr(module, "model_configured", lambda: True)
    class Failed:
        usage_records = [{"role": "researcher", "input_tokens": 5, "output_tokens": 2}]
        def complete(self, role, payload):
            raise RuntimeError("secret should not be exposed")
    monkeypatch.setattr(module, "OfficialAPIProvider", Failed)
    result = module.run_independent_context_research(payload={"evidence": []}, legal_evidence_ids=[])
    assert result.status == "model_call_failed" and result.usage_complete is False
    assert "secret" not in result.message and result.token_cost is None


def test_context_summaries_preserve_source_timing_and_limit_samples():
    r = {"etf_flow_context": {"kind": "fixture", "latest_complete_age_calendar_days": 4, "latest_exchange_session_verified": False},
         "fear_greed_context": {"kind": "fixture", "retrieved_at_utc": "2026-01-01T00:00:00Z", "stale": False,
                                 "observations": [{"value": 12, "observed_at_utc": "2026-01-01T00:00:00Z"}]},
         "prediction_market_context": {"asof_utc": "2026-01-01T00:00:00Z", "events": [
             {"title": "fixture", "markets": [{"question": "a", "outcomes": []}, {"question": "b", "outcomes": []}]}]}}
    out = {x["id"]: json.loads(x["raw"]) for x in compact_contexts(r)}
    assert out["BTC_ETF_FLOWS"]["latest_complete_age_calendar_days"] == 4
    assert out["BTC_ETF_FLOWS"]["latest_exchange_session_verified"] is False
    assert out["BTC_FEAR_GREED"]["observations"][0]["value"] == 12
    assert out["POLYMARKET_SAMPLE"]["events_sample"][0]["markets_omitted_by_chat"] == 1
    assert "not facts" in out["POLYMARKET_SAMPLE"]["note"]
    assert compact_contexts({}) == []
    macro = compact_macro({"series": {"DFII10": {"value": 2.8, "observation_date": "2026-01-01"}}, "intraday_release_verified": False})
    assert json.loads(macro["raw"])["intraday_release_verified"] is False


def test_forecast_must_match_archived_research_packet():
    stamp = "2026-01-01T00:00:00+00:00"
    req = {"request_hash": "fixture", "instruments": {"BTCUSDT": {"last_closed_close_at_utc": stamp,
           "indicators_prefix_only": {"last_close": 100.}}}}
    state = {"request_hash": "fixture", "forecast_weights_ran": True,
             "exploratory_evidence_archive": {"recorded_at_utc": stamp}}
    item = {"ok": True, "weights_ran": True, "timeframe": "1H_closed", "last_close": 100., "last_close_at_utc": stamp,
            "prediction_length": 1, "quantile_10": [99.], "quantile_50": [100.], "quantile_90": [101.]}
    forecast = {"weights_ran": True, "forecasts": {"BTCUSDT": item}}
    archive = {"request": copy.deepcopy(req), "forecast": copy.deepcopy(forecast), "created_at_utc": stamp}
    assert compact_forecast(req, state, forecast, archive)["id"] == "CHRONOS_RESEARCH_FORECAST"
    forecast["forecasts"]["BTCUSDT"]["quantile_50"] = [100.5]
    assert compact_forecast(req, state, forecast, archive) is None
