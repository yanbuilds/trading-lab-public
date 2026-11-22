from datetime import datetime, timezone
import json
import math

import pytest

from piia.intelligence.agents.providers import build_official_request, parse_official_response
from piia.intelligence.forecast_benchmark import neural_forecast_status, random_walk_drift_forecast
from piia.intelligence.macro_regime import classify_macro_snapshot
from piia.intelligence.research_fusion import build_research_bundle, default_bundle_from_closes, main


def test_random_walk_benchmark_is_labelled_and_rejects_bad_input():
    closes = [100.0 + i * 0.01 for i in range(200)]
    out = random_walk_drift_forecast(closes)
    assert out["available"] is True
    assert out["kind"] == "statistical_benchmark_not_neural_model"
    assert out["calibration_status"] == "not_calibrated"
    assert out["interval_low"] < out["point_horizon"] < out["interval_high"] or out["residual_sigma"] == 0
    assert random_walk_drift_forecast([1, 2, 3])["available"] is False
    assert random_walk_drift_forecast([-1] * 200)["reason"] == "non_positive_or_non_finite_close"
    assert random_walk_drift_forecast([True] * 200)["available"] is False
    assert random_walk_drift_forecast(closes, horizon=True)["available"] is False
    assert random_walk_drift_forecast([1e-300, 1e300] * 100, horizon=10000)["available"] is False


def test_neural_status_does_not_download_or_claim_weights():
    status = neural_forecast_status()
    assert status["weights_loaded"] is False
    assert status["order_permissions"] is False
    assert status["chronos"] in {"unavailable", "library_present_weights_not_loaded"}


def test_macro_snapshot_blocks_historical_alpha_and_handles_missing():
    missing = classify_macro_snapshot(None)
    assert missing["status"] == "missing"
    assert missing["macro_enhanced_historical_alpha"] == "blocked"
    assert missing["historical_replay_supported"] is False
    context = {
        "kind": "alfred_conservative_date_vintage",
        "historical_replay_supported": False,
        "retrieved_at_utc": "2026-11-17T01:00:00+00:00",
        "vintage_date": "2026-11-15",
        "series": {
            "DGS10": {"value": 4.2, "stale": False},
            "DGS2": {"value": 3.5, "stale": False},
            "DFII10": {"value": 1.8, "stale": False},
            "DFF": {"value": 4.0, "stale": False},
        },
    }
    labelled = classify_macro_snapshot(context)
    assert labelled["status"] == "available"
    assert labelled["curve_label"] == "upward"
    assert math.isclose(labelled["curve_10y_minus_2y"], 0.7)
    assert labelled["real_rate_label"] == "positive"
    assert labelled["macro_enhanced_historical_alpha"] == "blocked"
    inverted = classify_macro_snapshot({**context, "series": {**context["series"], "DGS2": {"value": 5.0, "stale": False}}})
    assert inverted["curve_label"] == "inverted"


def test_fusion_retains_contradictions_and_does_not_vote():
    closes = [100.0 + i * 0.2 for i in range(200)]
    forecast = random_walk_drift_forecast(closes)
    bundle = build_research_bundle(
        as_of_utc="2026-11-16T00:00:00+00:00",
        symbol="BTCUSDT",
        technical_regime={"kind": "deterministic_closed_bar_classifier_not_agent",
                          "regime": "trend", "completed_4h_ema20_vs_ema50": "down",
                          "order_permissions": False},
        macro_regime=classify_macro_snapshot(None),
        statistical_forecast=forecast,
        neural_status=neural_forecast_status(),
        ai_adapter={"configured": False, "independent_llm_review": "NOT_WIRED"},
    )
    assert bundle["voting"] is False
    assert bundle["order_permissions"] is False
    assert bundle["suggested_action"] == "research_only"
    assert bundle["confidence_is_not_accuracy"] is True
    if forecast["available"] and forecast["point_horizon"] > forecast["last_close"]:
        assert "technical_downtrend_vs_statistical_forecast_up" in bundle["contradictions_retained"]
    assert bundle["independent_sections"]["macro_regime"]["macro_enhanced_historical_alpha"] == "blocked"


def test_official_request_schemas_without_network():
    payload = {"statement": "example", "evidence": []}
    openai = build_official_request("openai", "https://api.openai.com", "dummy-key", "demo-model",
                                    "researcher", payload)
    assert openai["url"] == "https://api.openai.com/v1/chat/completions"
    assert openai["headers"]["Authorization"] == "Bearer dummy-key"
    assert "temperature" not in openai["body"]
    anthropic = build_official_request("anthropic", "https://api.anthropic.com", "dummy-key", "demo-model",
                                       "critic", payload)
    assert anthropic["url"] == "https://api.anthropic.com/v1/messages"
    assert anthropic["headers"]["x-api-key"] == "dummy-key"
    assert anthropic["headers"]["anthropic-version"] == "2023-06-01"
    assert anthropic["body"]["messages"][0]["role"] == "user"
    assert "system" in anthropic["body"]
    assert "Authorization" not in anthropic["headers"]
    text, usage = parse_official_response("anthropic", {
        "content": [{"type": "text", "text": '{"summary":"ok"}'}],
        "usage": {"input_tokens": 9, "output_tokens": 2},
    })
    assert text == '{"summary":"ok"}'
    assert usage == {"input_tokens": 9, "output_tokens": 2}
    with pytest.raises(ValueError):
        build_official_request("unknown", "https://example.com", "x", "m", "researcher", payload)


def test_anthropic_parse_and_openai_parse_keep_usage_unknown_when_missing():
    _, usage = parse_official_response("openai", {"choices": [{"message": {"content": "{}"}}]})
    assert usage["input_tokens"] is None
    content, usage_a = parse_official_response("anthropic", {"content": [{"type": "text", "text": "{}"}]})
    assert content == "{}"
    assert usage_a["output_tokens"] is None


def test_fusion_cli_writes_bundle(tmp_path):
    closes = tmp_path / "closes.json"
    closes.write_text(json.dumps([100 + i * 0.01 for i in range(200)]), encoding="utf-8")
    out = tmp_path / "bundle.json"
    assert main(["--closes-json", str(closes), "--symbol", "BTCUSDT", "--out", str(out)]) == 0
    bundle = json.loads(out.read_text(encoding="utf-8"))
    assert bundle["kind"] == "research_evidence_bundle_not_trade"
    assert bundle["independent_sections"]["statistical_forecast"]["available"] is True
    assert bundle["independent_sections"]["ai_adapter_status"]["independent_llm_review"] == "NOT_WIRED"
