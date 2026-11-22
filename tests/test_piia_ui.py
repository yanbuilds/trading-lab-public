from __future__ import annotations
from pathlib import Path
import pytest
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

def test_ui_renders():
    script = Path(__file__).resolve().parents[1] / "app" / "piia_app.py"
    at = AppTest.from_file(str(script), default_timeout=20)
    at.run()
    assert not at.exception
    assert any("没有保存规格原文" in item.value for item in at.caption)
    fields = {item.label: item.value for item in at.text_input}
    assert fields["Start quote time (ISO 8601 with UTC offset; optional)"] == ""
    assert fields["End quote time (ISO 8601 with UTC offset; optional)"] == ""


def test_chronos2_button_is_explicit_and_does_not_require_model_api(monkeypatch):
    from piia.intelligence import manual_observation
    calls = []
    def fake_capture():
        calls.append(True)
        return {"status": "no_candidate", "model_process_attempted": False,
                "weights_ran_verified": False, "order_permissions": False}
    monkeypatch.setattr(manual_observation, "capture_chronos2_once", fake_capture)
    script = Path(__file__).resolve().parents[1] / "app" / "piia_app.py"
    at = AppTest.from_file(str(script), default_timeout=60).run()
    assert calls == []
    next(button for button in at.button if button.label == "记录 Chronos-2 影子观察").click()
    at.run()
    assert not at.exception and len(calls) == 1


def test_segmented_report_is_run_only_when_clicked(monkeypatch):
    from piia.intelligence import manual_observation
    calls = []
    def fake_report():
        calls.append(True)
        return {"status": "all_segments_no_aggregate", "segments": [], "aggregate_return": None, "order_permissions": False}
    monkeypatch.setattr(manual_observation, "run_segmented_report", fake_report)
    script = Path(__file__).resolve().parents[1] / "app" / "piia_app.py"
    at = AppTest.from_file(str(script), default_timeout=30).run()
    assert calls == []
    next(button for button in at.button if button.label == "复算所有连续区间").click()
    at.run()
    assert not at.exception and len(calls) == 1


def test_funding_comparison_runs_only_when_clicked(monkeypatch):
    from piia.intelligence import manual_observation
    calls = []
    def fake_report():
        calls.append(True)
        return {"status": "all_segments_no_aggregate", "segments": [{"status": "unavailable"}],
                "aggregate_return": None, "order_permissions": False}
    monkeypatch.setattr(manual_observation, "run_funding_segment_report", fake_report)
    script = Path(__file__).resolve().parents[1] / "app" / "piia_app.py"
    at = AppTest.from_file(str(script), default_timeout=30).run()
    assert calls == []
    next(button for button in at.button if button.label == "复算各区间公共资金费估算").click()
    at.run()
    assert not at.exception and len(calls) == 1
    assert any("部分区间的数据或核验不足" in warning.value for warning in at.warning)


def test_ai_shadow_runs_only_on_explicit_button(monkeypatch):
    from piia.intelligence import ai_shadow
    calls = []
    def fake_run():
        calls.append(True)
        return {"status": "needs_model_configuration", "model_called": False, "order_permissions": False}
    monkeypatch.setattr(ai_shadow, "run", fake_run)
    script = Path(__file__).resolve().parents[1] / "app" / "piia_app.py"
    at = AppTest.from_file(str(script), default_timeout=60).run()
    assert calls == []
    next(button for button in at.button if button.label == "手动运行 AI 影子研究（可能收费）").click()
    at.run()
    assert not at.exception and len(calls) == 1
    assert any("AI 影子记录未完整确认" in warning.value for warning in at.warning)


def test_manual_portfolio_stress_requires_user_input():
    import json
    script = Path(__file__).resolve().parents[1] / "app" / "piia_app.py"
    at = AppTest.from_file(str(script), default_timeout=30).run()
    field = next(x for x in at.text_area if x.label == "假设持仓 JSON 数组")
    assert field.value == "[]"
    fixture = [{"symbol": "BTC", "side": "long", "quantity": 1, "contract_size": 1,
                "mark_price": 100, "stress_price": 90, "exit_fee_rate": 0, "slippage_rate": 0}]
    field.set_value(json.dumps(fixture))
    next(button for button in at.button if button.label == "计算组合压力情景").click()
    at.run()
    assert not at.exception
    result = at.session_state["manual_portfolio_stress"]
    assert result["simultaneous_pnl_after_assumed_exit_cost_usdt"] == -10
    assert result["order_permissions"] is False


def test_report_hides_model_result_when_claim_audit_fails(tmp_path, monkeypatch):
    import json
    from piia.intelligence import ai_shadow_source_audit, paths, report_ledger_check, shadow_claim_audit
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(report_ledger_check, "check_report_ledger", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(ai_shadow_source_audit, "gate_report", lambda value: (value, None))
    monkeypatch.setattr(shadow_claim_audit, "audit_claims", lambda *a, **k: {"ok": False})
    out = tmp_path / "artifacts" / "prospective_paper_report.json"
    out.parent.mkdir()
    out.write_text(json.dumps({"status": "local_prospective_paper_approximation_not_independently_verified",
        "order_permissions": False, "ai_shadow": {"status": "complete", "result": {"final": 999}},
        "chronos_shadow": {"status": "complete", "result": {"final": 999}}}), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "app" / "piia_app.py"
    at = AppTest.from_file(str(script), default_timeout=30).run()
    assert not at.exception
    assert any("ai_shadow 的每小时调用记录" in warning.value for warning in at.warning)
    assert any("chronos_shadow 的每小时调用记录" in warning.value for warning in at.warning)
    assert not any('"final": 999' in item.value for item in at.json)


def test_failed_full_capture_does_not_leave_prior_snapshot_selected(monkeypatch):
    from piia.intelligence import research_cycle
    def fail(*args, **kwargs):
        raise RuntimeError("fixture network failure")
    monkeypatch.setattr(research_cycle, "prepare_run", fail)
    script = Path(__file__).resolve().parents[1] / "app" / "piia_app.py"
    at = AppTest.from_file(str(script), default_timeout=20).run()
    at.session_state["chat_request_path"] = "old_fixture"
    at.session_state["chat_cycle_state"] = {"run_id": "old_fixture"}
    next(b for b in at.button if b.label == "获取完整跨市场研究包").click()
    at.run()
    assert not at.exception
    assert any("完整研究包生成失败" in e.value for e in at.error)
    assert "chat_cycle_state" not in at.session_state


def test_full_capture_displays_available_context_and_missing_forecast(tmp_path, monkeypatch):
    import json
    from datetime import datetime, timezone
    from piia.intelligence import research_cycle
    now = datetime.now(timezone.utc).isoformat()
    request = {"snapshot_ok": True, "cutoff_at_utc": now, "errors": [], "missing_datasets": ["btc_etf_flows"],
               "request_hash": "synthetic_fixture", "fear_greed_context": {"kind": "synthetic", "observations": [{"value": 12}]},
               "instruments": {"BTCUSDT": {"indicators_prefix_only": {"ok": False}, "all_closed_candles": []}}}
    path = tmp_path / "synthetic-request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    state = {"run_id": "synthetic", "paths": {"full_request": str(path), "forecast": str(tmp_path / "missing.json")},
             "forecast_weights_ran": False, "trading_recommendation_blockers": ["btc_etf_flows"]}
    monkeypatch.setattr(research_cycle, "prepare_run", lambda *_args, **_kwargs: state)
    script = Path(__file__).resolve().parents[1] / "app" / "piia_app.py"
    at = AppTest.from_file(str(script), default_timeout=20).run()
    next(b for b in at.button if b.label == "获取完整跨市场研究包").click()
    at.run()
    assert not at.exception
    assert at.session_state["chat_request"]["request_hash"] == "synthetic_fixture"
    assert any("预测缺失" in w.value for w in at.warning)


def test_prospective_report_display_is_explicitly_local_and_not_live(tmp_path, monkeypatch):
    import json
    from piia.intelligence import paths
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    output = tmp_path / "artifacts" / "prospective_paper_report.json"
    output.parent.mkdir()
    output.write_text(json.dumps({"status": "local_prospective_paper_approximation_not_independently_verified",
        "order_permissions": False, "generated_at_utc": "2026-11-19T00:00:00+00:00",
        "last_observation_cutoff_utc": "2026-11-18T23:00:00+00:00", "funding": "unknown_excluded",
        "candidate": {"initial": 100}, "ema": {"initial": 100}, "ai_shadow": {"status": "no_ai_shadow_records"}}), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "app" / "piia_app.py"
    at = AppTest.from_file(str(script), default_timeout=20).run()
    assert not at.exception
    captions = " ".join(c.value for c in at.caption)
    assert "2026-11-19T00:00:00+00:00" in captions
    assert "资金费未知" in captions
    assert any("账本不一致或无法核对" in w.value for w in at.warning)


def test_funding_stress_ui_is_labeled_hypothetical(tmp_path, monkeypatch):
    import json
    from piia.intelligence import paths
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    output = tmp_path / "artifacts" / "prospective_funding_stress.json"
    output.parent.mkdir()
    output.write_text(json.dumps({"status": "hypothetical_funding_stress_not_actual_net_performance",
                                  "order_permissions": False, "observations": 3, "scenarios": {}}), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "app" / "piia_app.py"
    at = AppTest.from_file(str(script), default_timeout=20).run()
    assert not at.exception
    assert "不是真实结算" in " ".join(c.value for c in at.caption)


def test_funding_estimate_ui_discloses_proxy(tmp_path, monkeypatch):
    import json
    from piia.intelligence import paths
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    output = tmp_path / "artifacts" / "prospective_funding_estimate.json"
    output.parent.mkdir()
    output.write_text(json.dumps({"status": "public_funding_paper_estimate_not_actual_net_performance",
                                  "order_permissions": False, "observations": 3, "accounts": {}}), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "app" / "piia_app.py"
    at = AppTest.from_file(str(script), default_timeout=20).run()
    assert not at.exception
    assert "开盘价代替标记价" in " ".join(c.value for c in at.caption)


def test_model_configuration_error_does_not_display_key(monkeypatch):
    monkeypatch.setenv("PIIA_MODEL_PROVIDER", "xai")
    monkeypatch.setenv("PIIA_MODEL_BASE_URL", "https://api.openai.com")
    monkeypatch.setenv("PIIA_MODEL_NAME", "fake-model")
    monkeypatch.setenv("PIIA_MODEL_API_KEY", "secret-test-value")
    script = Path(__file__).resolve().parents[1] / "app" / "piia_app.py"
    at = AppTest.from_file(str(script), default_timeout=20).run()
    assert not at.exception
    captions = " ".join(c.value for c in at.caption)
    assert "模型服务商与 API 地址不匹配" in captions
    assert "secret-test-value" not in captions


def test_manual_scenario_form_calculates_without_order():
    script = Path(__file__).resolve().parents[1] / "app" / "piia_app.py"
    at = AppTest.from_file(str(script), default_timeout=20).run()
    at.number_input(key="scenario_entry").set_value(100.0)
    at.number_input(key="scenario_stop").set_value(90.0)
    at.number_input(key="scenario_gap").set_value(80.0)
    at.number_input(key="scenario_qty").set_value(1.0)
    next(b for b in at.button if b.label == "计算两种情景").click()
    at.run()
    assert not at.exception
    result = at.session_state["manual_risk_scenario"]
    assert result["gap_fill_assumption"]["gross_pnl_usdt"] == -20
    assert result["liquidation_price"] is None
    assert result["order_permissions"] is False


def test_model_question_uses_current_snapshot_without_real_api(monkeypatch):
    from datetime import datetime, timedelta, timezone
    from piia.intelligence import settings
    from piia.intelligence.agents import independent_context
    from piia.intelligence.research_packet import capture_paper_risk_context

    monkeypatch.setattr(settings, "model_status", lambda: {"configured": True})
    calls = []
    class Result:
        def model_dump(self):
            return {"status": "completed", "mode": "model", "researcher": {"summary": "示例观点"},
                    "critic": {"summary": "示例反方"}, "synthesizer": {"summary": "示例综合", "suggested_action": "review_paper_plan"}}
    def fake_workflow(**kwargs):
        calls.append(kwargs)
        return Result()
    monkeypatch.setattr(independent_context, "run_independent_context_research", fake_workflow)
    now_dt = datetime.now(timezone.utc)
    now = now_dt.isoformat()
    hour = now_dt.replace(minute=0, second=0, microsecond=0)
    candles = [{"open_at_utc": (hour - timedelta(hours=i)).isoformat(),
                "close_at_utc": (hour - timedelta(hours=i-1)).isoformat(),
                "open": 100., "high": 101., "low": 99., "close": 100., "volume": 1.}
               for i in range(60, 0, -1)]
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "piia_app.py"), default_timeout=60)
    at.session_state["chat_request"] = {
        "snapshot_ok": True, "cutoff_at_utc": now, "missing_datasets": ["macro"],
        "paper_risk_context": capture_paper_risk_context(),
        "macro_context": {"kind": "test_fixture", "series": {"DGS10": {"value": 4.2}}},
        "fear_greed_context": {"kind": "test_fixture", "observations": [{"value": 12, "observed_at_utc": now}]},
        "prediction_market_context": {"kind": "test_fixture", "asof_utc": now, "events": []},
        "instruments": {"BTCUSDT": {"ticker": {"observed_at_utc": now, "last": 100, "mark": 100, "funding_rate": "0.123"},
                                      "open_interest": {"period_end_utc": hour.isoformat(), "total_open_interest_base_units": 100.0},
                                      "funding": None, "indicators_prefix_only": {"ok": True},
                                      "candles_closed_1h_tail": candles[-8:], "all_closed_candles": candles}},
    }
    at.run()
    at.number_input(key="sizing_stop").set_value(95.0)
    at.number_input(key="sizing_gap").set_value(80.0)
    next(b for b in at.button if b.label == "计算虚拟风险示例").click()
    at.run()
    assert at.session_state["virtual_sizing_example"]["result"]["reason"] == "ok"
    next(c for c in at.checkbox if c.label == "将虚拟风险估算发送给模型").check()
    at.text_area(key="chat_question").set_value("有哪些反方观点？")
    next(b for b in at.button if b.label == "让模型分析这份快照").click()
    at.run()
    assert not at.exception
    assert len(calls) == 1
    assert calls[0]["payload"]["statement"] == "有哪些反方观点？"
    assert calls[0]["payload"]["focus_symbol"] == "BTCUSDT"
    assert "thesis_direction" not in calls[0]["payload"]
    assert calls[0]["legal_evidence_ids"] == ["BTCUSDT", "ALFRED_MACRO", "BTC_FEAR_GREED", "POLYMARKET_SAMPLE"]
    assert calls[0]["payload"]["risk_context"]["kind"] == "virtual_example_not_actual_account"
    assert calls[0]["payload"]["risk_context"]["actual_positions"] == "unknown"
    assert calls[0]["payload"]["deterministic_review_gate"]["review_blocked"] is True
    analysis = at.session_state["chat_dialogue"][0]["analysis"]
    assert analysis["synthesizer"]["suggested_action"] == "need_more_evidence"
    assert analysis["deterministic_review_gate"]["order_permissions"] is False
    assert calls[0]["payload"]["risk_context"]["owner_config"]["specs"]["BTCUSDT"]["min_qty"] == 0.0001
    assert calls[0]["payload"]["deterministic_virtual_sizing_example"]["order_permissions"] is False
    assert calls[0]["payload"]["deterministic_virtual_sizing_example"]["gap_scenario"]["exceeds_stop_risk_budget"] is True
    import json
    model_evidence = json.loads(calls[0]["payload"]["evidence"][0]["raw"])
    assert model_evidence["open_interest"]["total_open_interest_base_units"] == 100.0
    assert model_evidence["funding"] is None and "funding_rate" not in model_evidence["ticker"]
    assert "current_funding:BTCUSDT" in calls[0]["payload"]["missing_datasets"]
    assert "user_provided_scenario_unverified" not in calls[0]["payload"]
    assert calls[0]["payload"]["previous_discussion"] == []
    at.text_area(key="chat_question").set_value("如果反转呢？")
    next(b for b in at.button if b.label == "让模型分析这份快照").click()
    at.run()
    assert not at.exception
    assert len(calls) == 2
    prior = calls[1]["payload"]["previous_discussion"]
    assert prior[0]["question"] == "有哪些反方观点？"
    assert prior[0]["prior_model_opinions_not_evidence"]["critic"] == "示例反方"
    assert len(at.session_state["chat_dialogue"]) == 2
    at.session_state["manual_risk_scenario"] = {"kind": "manual_hypothetical_linear_usdt_not_order", "entry_price": 100}
    next(c for c in at.checkbox if c.label == "将本会话的手工风险情景发送给模型").check()
    at.text_area(key="chat_question").set_value("评估这个情景")
    next(b for b in at.button if b.label == "让模型分析这份快照").click()
    at.run()
    assert calls[2]["payload"]["user_provided_scenario_unverified"]["entry_price"] == 100
    stale_snapshot = dict(at.session_state["chat_request"])
    stale_snapshot["instruments"] = {"BTCUSDT": {**stale_snapshot["instruments"]["BTCUSDT"],
                                                    "all_closed_candles": candles[:-3]}}
    at.session_state["chat_request"] = stale_snapshot
    next(b for b in at.button if b.label == "让模型分析这份快照").click()
    at.run()
    assert len(calls) == 3
    assert any("K 线不完整或已过期" in e.value for e in at.error)
    new_snapshot = dict(at.session_state["chat_request"])
    new_snapshot["request_hash"] = "new-snapshot"
    at.session_state["chat_request"] = new_snapshot
    at.run()
    assert at.session_state["chat_dialogue"] == []


def test_model_dialogue_is_scoped_to_selected_contract(monkeypatch):
    from datetime import datetime, timedelta, timezone
    from piia.intelligence import settings
    from piia.intelligence.agents import independent_context

    monkeypatch.setattr(settings, "model_status", lambda: {"configured": True})
    calls = []
    class Result:
        def model_dump(self):
            return {"status": "completed", "researcher": {"summary": "test"}, "critic": {}, "synthesizer": {}}
    def fake_workflow(**kwargs):
        calls.append(kwargs)
        return Result()
    monkeypatch.setattr(independent_context, "run_independent_context_research", fake_workflow)
    now_dt = datetime.now(timezone.utc)
    hour = now_dt.replace(minute=0, second=0, microsecond=0)
    candles = [{"open_at_utc": (hour - timedelta(hours=i)).isoformat(),
                "close_at_utc": (hour - timedelta(hours=i-1)).isoformat(),
                "open": 100., "high": 101., "low": 99., "close": 100., "volume": 1.}
               for i in range(60, 0, -1)]
    def instrument(name):
        return {"ticker": {"observed_at_utc": now_dt.isoformat(), "last": 100, "mark": 100},
                "indicators_prefix_only": {"ok": True}, "market_context": {"fixture_symbol": name},
                "all_closed_candles": candles, "candles_closed_1h_tail": candles[-8:]}
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "piia_app.py"), default_timeout=20)
    at.session_state["chat_request"] = {"snapshot_ok": True, "cutoff_at_utc": now_dt.isoformat(),
        "missing_datasets": [], "instruments": {"BTCUSDT": instrument("BTCUSDT"), "XAUUSDT": instrument("XAUUSDT")}}
    at.run()
    at.text_area(key="chat_question").set_value("BTC question")
    next(b for b in at.button if b.label == "让模型分析这份快照").click()
    at.run()
    at.selectbox(key="chat_symbol").select("XAUUSDT")
    at.text_area(key="chat_question").set_value("gold question")
    next(b for b in at.button if b.label == "让模型分析这份快照").click()
    at.run()
    assert not at.exception
    assert calls[1]["payload"]["focus_symbol"] == "XAUUSDT"
    import json
    focus = calls[1]["payload"]["focus_market_context"]
    evidence = {entry["id"]: json.loads(entry["raw"]) for entry in calls[1]["payload"]["evidence"]}
    assert focus == evidence["XAUUSDT"]["market_context"]
    assert focus["timeframe_alignment"] == "incomplete"
    assert "daily_context:XAUUSDT" in calls[1]["payload"]["missing_datasets"]
    assert calls[1]["payload"]["previous_discussion"] == []
    at.selectbox(key="chat_symbol").select("BTCUSDT")
    at.text_area(key="chat_question").set_value("follow up")
    next(b for b in at.button if b.label == "让模型分析这份快照").click()
    at.run()
    assert calls[2]["payload"]["focus_symbol"] == "BTCUSDT"
    assert calls[2]["payload"]["previous_discussion"][0]["question"] == "BTC question"
