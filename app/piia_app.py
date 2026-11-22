"""Minimal local PIIA console. Uses the same IntelligenceService as the API."""
from __future__ import annotations
import os
import sqlite3
from datetime import datetime, timedelta, timezone

import streamlit as st

from piia.intelligence.clock import Clock
from piia.intelligence.db import create_all, make_engine, session_factory
from piia.intelligence.paths import default_database_url
from piia.intelligence.schemas import (
    ApprovalIn, CounterIn, EvidenceIn, OutcomeIn, PlanIn, RiskConfigIn, ThesisCreate, ThesisRevise,
)
from piia.intelligence.service import IntelligenceService

st.set_page_config(page_title="PIIA Phase 1", layout="wide")
st.title("PIIA — 合约研究与本地纸面交易")
from piia.intelligence.settings import model_configuration_reasons, model_status
from piia.intelligence.research_service import ExtendedService
from piia.intelligence.paper.specs import SAMPLE_MARKS, CONTRACTS

ms = model_status()
st.caption(
    "仅用于本地研究与简化纸面交易，不能连接真实账户或交易所模拟盘下单。"
    "样例观点会标注为演示；此页面没有登录认证，只能在本机使用。"
)
if not ms["configured"]:
    st.warning("尚未配置模型 API；独立模型审查未接入。可使用演示，或让当前聊天助手读取研究包。")
    reasons = model_configuration_reasons(ms)
    if reasons:
        st.caption("模型配置：" + " ".join(reasons))
else:
    st.info("已检测到模型环境配置；实际连接和费用需以服务商结果为准。")
st.caption("固定历史样例行情，非实时：观察时间约 2026-11-16 07:25 UTC。不要据此判断当前价格： " + ", ".join(
    f"{k} last={v['last']}" for k,v in SAMPLE_MARKS.items()
) + ". Exact liquidation is unverified.")

@st.cache_resource
def _factory():
    url = os.environ.get("PIIA_DATABASE_URL") or default_database_url()
    engine = make_engine(url)
    create_all(engine)
    fac = session_factory(engine)
    with fac() as s:
        IntelligenceService(s).seed_instruments()
    return fac

def service() -> IntelligenceService:
    return IntelligenceService(_factory()(), Clock())

svc = service()
try:
    instruments = {i.id: i.display_name for i in svc.list_instruments()}
    tab_chat, tab_scenario, tab_create, tab_view, tab_risk, tab_eval, tab_research, tab_paper = st.tabs(["聊天辅助研究", "仓位风险情景", "Create thesis", "Revisions & countercase", "Sample risk / plan", "Outcome", "Research run", "Paper account"])

    with tab_chat:
        import json
        from pathlib import Path
        from piia.intelligence.paths import ROOT
        from piia.intelligence.public_data import fetch_public_snapshot, candle_quality, current_funding_for_research
        from piia.intelligence.research_packet import build_request, capture_paper_risk_context, import_response, write_json
        from piia.intelligence.errors import PiiaError

        st.subheader("聊天辅助研究")
        st.caption("冻结公开数据后分析；研究方和反方可使用同一模型的独立请求上下文，不代表独立模型。不会自动批准或下单。")
        with st.expander("前瞻纸面对照（本地报告）"):
            st.caption("手动采集连续小时观察并生成报告后显示；资金费未知，模拟开盘和止损不代表真实成交。")
            st.caption("冻结采集器保存合约规格解析值及原文哈希，但没有保存规格原文；因此无法重新解析原始响应核验规格。")
            st.caption("政策开始后可按需记录当前小时；不会自动定时或发送订单。此按钮不调用付费模型 API。")
            from piia.intelligence.prospective_status import collection_status
            local_status = collection_status(path=ROOT / "data/piia/frozen_policy_v1.sqlite")
            st.caption("本地采集状态：仅核对记录覆盖与基本绑定，不代表策略收益或独立审计。")
            st.json(local_status)
            if any(value in {"attempted_without_result", "invalid_binding", "recorded_model_failed", "recorded_incomplete"}
                   for value in (local_status.get("latest_steps") or {}).values()):
                st.warning("部分步骤失败、未完成或绑定异常；请检查账本，不要重试挑选结果。")
            if st.button("手动记录本小时研究观察"):
                from piia.intelligence.manual_observation import capture_once
                with st.spinner("正在记录机械观察、公开资金费快照和 Chronos 影子结果。"):
                    observation_result = capture_once()
                st.json(observation_result)
                if observation_result.get("status") != "captured_research_only":
                    st.warning("本小时观察未确认；可能尚未到政策开始时间，或数据/本地环境不可用。请检查本地账本后再操作。")
                elif any(step.get("status") not in {"captured_research_only", "complete", "completed", "no_candidate"}
                         for step in observation_result.get("steps", {}).values()):
                    st.warning("机械观察已记录，但部分资金费或模型记录缺失；不能将缺失时段补作完整对照。")
            if st.button("记录 Chronos-2 影子观察"):
                from piia.intelligence.manual_observation import capture_chronos2_once
                with st.spinner("正在运行本地 Chronos-2；首次加载可能较慢，不会自动重试。"):
                    chronos2_result = capture_chronos2_once()
                st.json(chronos2_result)
                if chronos2_result.get("status") not in {"completed", "no_candidate"}:
                    st.warning("Chronos-2 影子记录未完整确认；请先检查本地账本。")
            st.caption("AI 影子使用已配置模型，对非零候选发起请求，可能产生 API 费用；研究方与反方不是独立模型。当前协议仅含机械信号和最近八根 K 线，不含完整宏观或资金流。")
            if st.button("手动运行 AI 影子研究（可能收费）"):
                from piia.intelligence.ai_shadow import run as run_ai_shadow
                try:
                    with st.spinner("正在运行 AI 影子研究，不下单、不自动重试。"):
                        ai_result = run_ai_shadow()
                    st.json(ai_result)
                    if ai_result.get("status") not in {"completed", "no_candidate"}:
                        st.warning("AI 影子记录未完整确认；模型配置、失败或缺失不算完成。请检查本地账本。")
                except Exception:
                    st.warning("AI 影子运行未确认；请检查本地账本，不要重复运行以挑选结果。")
            st.caption("断档报告 v2 从 2026-11-18 07:00 UTC 起按预先固定的规则分段，加入 24 小时动量及 168 小时漂移基准；各段单独起算，不能合并收益。v1 因 Windows 临时文件问题不可用。")
            if st.button("复算所有连续区间"):
                from piia.intelligence.manual_observation import run_segmented_report
                with st.spinner("正在本地复算，不连接交易所或模型。"):
                    segmented = run_segmented_report()
                if segmented.get("status") != "all_segments_no_aggregate":
                    st.warning("分段报告尚未生效，或本地复算不可用；不能用缺失数据替代。")
                st.json(segmented)
            st.caption("公共资金费对照会请求公开历史费率；它是估算，实际结算、成交和强平仍未验证。")
            if st.button("复算各区间公共资金费估算"):
                from piia.intelligence.manual_observation import run_funding_segment_report
                with st.spinner("正在检查本地观察并请求公开资金费历史；不连接账户或模型。"):
                    funding_report = run_funding_segment_report()
                if funding_report.get("status") != "all_segments_no_aggregate":
                    st.warning("资金费估算尚未生效或不可用；缺失数据不会按零处理。")
                elif any(section.get("status") == "unavailable" for section in funding_report.get("segments", [])):
                    st.warning("部分区间的数据或核验不足，未输出该区间估算。")
                st.json(funding_report)
            report_path = ROOT / "artifacts" / "prospective_paper_report.json"
            if not report_path.is_file():
                st.info("暂无前瞻对照报告；历史回测不会自动补作前瞻结果。")
            else:
                if st.button("复算并核对前瞻报告"):
                    from piia.intelligence.paper_report_replay import verify_saved_report
                    with st.spinner("正在使用本地冻结规则复算；不调用模型或交易所。"):
                        verification = verify_saved_report(root=ROOT)
                    if verification["status"] == "matched_local_replay":
                        st.success("报告数值与本地复算一致；这仍是未验证净收益的研究近似。")
                    else:
                        st.warning("无法复算或报告与复算结果不一致。")
                    st.json(verification)
                try:
                    if report_path.stat().st_size > 2_000_000:
                        raise ValueError("report too large")
                    paper_report = json.loads(report_path.read_text(encoding="utf-8"))
                    if (not isinstance(paper_report, dict) or paper_report.get("status") != "local_prospective_paper_approximation_not_independently_verified"
                            or paper_report.get("order_permissions") is not False):
                        raise ValueError("unexpected report")
                    st.caption(f"生成 UTC：{paper_report.get('generated_at_utc', '未知')}；"
                               f"最后观察截至 UTC：{paper_report.get('last_observation_cutoff_utc', '未知')}。")
                    from piia.intelligence.report_ledger_check import check_report_ledger
                    ledger_check = check_report_ledger(paper_report, ledger=ROOT / "data/piia/frozen_policy_v1.sqlite")
                    if not ledger_check["ok"]:
                        st.warning("报告与当前前瞻账本不一致或无法核对，已隐藏结果。")
                        raise ValueError("report ledger mismatch")
                    st.caption("已核对账本时间范围和数量；未在界面重新计算报告收益。")
                    from piia.intelligence.ai_shadow_source_audit import gate_report
                    paper_report, audit = gate_report(paper_report)
                    if audit is not None:
                        if not audit["ok"]:
                            st.warning("AI 对照依赖代码未能证明在观察前固定，已隐藏 AI 对照结果。")
                            st.json(audit["issues"])
                        else:
                            st.caption("AI 依赖代码通过本地版本核查；云模型版本及独立时间仍未验证。")
                    from piia.intelligence.shadow_claim_audit import audit_claims
                    for arm in ("ai_shadow", "chronos_shadow"):
                        model_report = paper_report.get(arm)
                        if isinstance(model_report, dict) and model_report.get("status") == "complete":
                            if not audit_claims(ROOT / "data/piia/frozen_policy_v1.sqlite", arm)["ok"]:
                                paper_report[arm] = {"status": "claim_audit_failed", "result": None}
                                st.warning(f"{arm} 的每小时调用记录未通过本地绑定检查，已隐藏该对照。")
                    st.json({k: paper_report.get(k) for k in ("observations", "capital_usdt", "funding", "candidate", "ema", "cash_reference", "ai_shadow", "chronos_shadow", "limitations")})
                except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                    st.warning("本地报告不可用或格式不正确，请重新生成；未执行交易。")
        with st.expander("Chronos-2 前瞻影子对照（本地报告）"):
            st.caption("需要每小时完整机械与 Chronos-2 影子记录；仅价格诊断，资金费未知，未证明模型优势。")
            chronos2_path = ROOT / "artifacts" / "chronos2_shadow_report.json"
            if not chronos2_path.is_file():
                st.info("暂无 Chronos-2 前瞻对照报告；旧历史不能补作前瞻记录。")
            else:
                try:
                    if chronos2_path.stat().st_size > 4_000_000:
                        raise ValueError("report too large")
                    chronos2_report = json.loads(chronos2_path.read_text(encoding="utf-8"))
                    if (not isinstance(chronos2_report, dict)
                            or chronos2_report.get("status") != "local_chronos2_shadow_price_only_not_validated_net"
                            or chronos2_report.get("order_permissions") is not False):
                        raise ValueError("unexpected report")
                    from piia.intelligence.report_ledger_check import check_report_ledger
                    if not check_report_ledger(chronos2_report, ledger=ROOT / "data/piia/frozen_policy_v1.sqlite")["ok"]:
                        raise ValueError("report ledger mismatch")
                    from piia.intelligence.shadow_claim_audit import audit_claims
                    if not audit_claims(ROOT / "data/piia/frozen_policy_v1.sqlite", "chronos2_shadow")["ok"]:
                        raise ValueError("chronos2 claim audit failed")
                    tiny = chronos2_report.get("chronos_tiny_shadow")
                    if isinstance(tiny, dict) and tiny.get("status") == "complete":
                        if not audit_claims(ROOT / "data/piia/frozen_policy_v1.sqlite", "chronos_shadow")["ok"]:
                            chronos2_report["chronos_tiny_shadow"] = {"status": "claim_audit_failed", "result": None}
                            st.warning("Chronos-tiny 调用记录未通过本地绑定检查，已隐藏该对照。")
                    st.caption("账本时间范围与数量一致；界面未重新计算报告收益。")
                    st.json({k: chronos2_report.get(k) for k in ("generated_at_utc", "observations", "active_model_signals",
                        "funding", "candidate", "ema", "chronos_tiny_shadow", "chronos2_shadow", "cash_reference", "limitations")})
                    funding_estimate = chronos2_report.get("public_funding_estimate")
                    if isinstance(funding_estimate, dict):
                        st.caption("以下为公开费率与已观察周期下的估算，真实结算及账户费用未验证。")
                        st.json({k: funding_estimate.get(k) for k in ("status", "reason", "funding_not_replaced_with_zero",
                                                                      "accounts", "chronos_tiny_status", "limitations")})
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    st.warning("Chronos-2 对照报告无法核对或与账本不一致；已隐藏结果。")
        with st.expander("资金费假设压力情景（本地报告）"):
            st.caption("固定假设费率，不是真实结算或已验证净收益；不同情景的仓位和停机路径可能不同。")
            stress_path = ROOT / "artifacts" / "prospective_funding_stress.json"
            if not stress_path.is_file():
                st.info("暂无资金费压力报告；不足的前瞻数据不会用历史数据补齐。")
            else:
                try:
                    if stress_path.stat().st_size > 2_000_000:
                        raise ValueError("report too large")
                    stress_report = json.loads(stress_path.read_text(encoding="utf-8"))
                    if (not isinstance(stress_report, dict)
                            or stress_report.get("status") != "hypothetical_funding_stress_not_actual_net_performance"
                            or stress_report.get("order_permissions") is not False):
                        raise ValueError("unexpected report")
                    from piia.intelligence.report_ledger_check import check_report_ledger
                    if not check_report_ledger(stress_report, ledger=ROOT / "data/piia/frozen_policy_v1.sqlite")["ok"]:
                        st.warning("压力报告与当前前瞻账本不一致或无法核对，已隐藏结果。")
                        raise ValueError("report ledger mismatch")
                    from piia.intelligence.ai_shadow_source_audit import gate_report
                    stress_report, audit = gate_report(stress_report)
                    if audit is not None and not audit["ok"]:
                        st.warning("AI 对照依赖版本未通过本地核查，已隐藏 AI 压力结果。")
                    st.json({k: stress_report.get(k) for k in ("generated_at_utc", "observations", "capital_usdt", "scenarios", "limitations")})
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    st.warning("本地压力报告不可用或格式不正确；未执行交易。")
        with st.expander("公开资金费纸面估算（本地报告）"):
            st.caption("官方历史费率结合已记录周期估算；开盘价代替标记价，实际账户费用和结算完整性未验证。")
            estimate_path = ROOT / "artifacts" / "prospective_funding_estimate.json"
            if not estimate_path.is_file():
                st.info("暂无公开资金费纸面估算；缺失观察不补零。")
            else:
                try:
                    if estimate_path.stat().st_size > 4_000_000:
                        raise ValueError("report too large")
                    estimate = json.loads(estimate_path.read_text(encoding="utf-8"))
                    if (not isinstance(estimate, dict)
                            or estimate.get("status") != "public_funding_paper_estimate_not_actual_net_performance"
                            or estimate.get("order_permissions") is not False):
                        raise ValueError("unexpected report")
                    from piia.intelligence.report_ledger_check import check_report_ledger
                    if not check_report_ledger(estimate, ledger=ROOT / "data/piia/frozen_policy_v1.sqlite")["ok"]:
                        st.warning("资金费估算与当前前瞻账本不一致或无法核对，已隐藏结果。")
                        raise ValueError("report ledger mismatch")
                    from piia.intelligence.ai_shadow_source_audit import gate_report
                    estimate, audit = gate_report(estimate)
                    if audit is not None and not audit["ok"]:
                        st.warning("AI 对照依赖版本未通过本地核查，已隐藏 AI 估算结果。")
                    st.json({k: estimate.get(k) for k in ("generated_at_utc", "observations", "capital_usdt", "accounts",
                                                          "funding_intervals_observed_hours", "limitations")})
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    st.warning("本地资金费估算不可用或格式不正确；未执行交易。")
        include_macro = st.checkbox("同时获取 ALFRED 宏观日期版本", value=False,
                                    help="包括部分美债收益率及联邦基金有效利率；发布时间的日内精度未验证。")
        if st.button("获取最新公开行情并生成研究包"):
            risk_context = capture_paper_risk_context()
            snapshot = fetch_public_snapshot()
            snapshot["paper_risk_context"] = risk_context
            if include_macro:
                from piia.intelligence.macro_data import fetch_alfred_context
                snapshot["macro_context"] = fetch_alfred_context()
            request = build_request(snapshot)
            path = ROOT / "artifacts" / "research_packets" / "current_request.json"
            write_json(path, request)
            st.session_state["chat_request_path"] = str(path)
            st.session_state["chat_request"] = request
            st.session_state.pop("chat_cycle_state", None)
            if request["snapshot_ok"]:
                st.success("研究包已保存。请让聊天助手读取下方文件并完成分析。")
            else:
                st.error("行情不完整：可以排查数据，但不可据此生成纸面计划。")
            st.json({"cutoff_at_utc": request["cutoff_at_utc"], "errors": request["errors"],
                     "missing_datasets": request["missing_datasets"]})
        if st.button("获取完整跨市场研究包"):
            from piia.intelligence.research_cycle import prepare_run
            for key in ("chat_request", "chat_request_path", "chat_cycle_state"):
                st.session_state.pop(key, None)
            try:
                with st.spinner("正在抓取公开行情、宏观和其他研究数据并尝试本地预测；可能需要几分钟。"):
                    state = prepare_run(ROOT / "artifacts" / "research_cycle")
                path = Path(state["paths"]["full_request"])
                request = json.loads(path.read_text(encoding="utf-8"))
                st.session_state["chat_request_path"] = str(path)
                st.session_state["chat_request"] = request
                st.session_state["chat_cycle_state"] = state
                if request["snapshot_ok"]:
                    st.success("完整研究包已保存；各数据源的缺失、延迟和限制见下方。")
                else:
                    st.error("行情不完整：可以排查数据，但不可据此生成纸面计划。")
                st.json({"cutoff_at_utc": request["cutoff_at_utc"], "errors": request["errors"],
                         "missing_datasets": request["missing_datasets"],
                         "forecast_ran": state.get("forecast_weights_ran"),
                         "research_blockers": state.get("trading_recommendation_blockers")})
            except Exception:
                st.error("完整研究包生成失败，请检查公开数据连接、本地预测环境及日志；未执行交易。")
        st.code(st.session_state.get("chat_request_path", str(ROOT / "artifacts" / "research_packets" / "current_request.json")))
        chat_request = st.session_state.get("chat_request")
        if chat_request:
            import altair as alt
            import pandas as pd
            snapshot_id = chat_request.get("request_hash") or chat_request["cutoff_at_utc"]
            if st.session_state.get("chat_dialogue_snapshot") != snapshot_id:
                st.session_state["chat_dialogue_snapshot"] = snapshot_id
                st.session_state["chat_dialogue"] = []
                st.session_state.pop("virtual_sizing_example", None)
                st.session_state.pop("model_last_failure", None)
            sym = st.selectbox("查看研究快照", list(chat_request["instruments"]), key="chat_symbol")
            snapshot = chat_request["instruments"][sym]
            st.caption(f"冻结快照，非实时推送；截至 UTC {chat_request['cutoff_at_utc']}。仅包含已收盘 1 小时 K 线。")
            if "paper_risk_context" in chat_request:
                st.caption("本地虚拟纸面风险和合约规格快照；不是你的真实账户，规格可能过期。")
                st.json(chat_request["paper_risk_context"])
            st.json(snapshot["indicators_prefix_only"])
            if snapshot.get("market_context"):
                st.caption("多周期均线和已收盘小时相对过去 20 个完整日线区间的位置：历史背景，不是交易信号或预测。")
                st.json(snapshot["market_context"])
            st.caption("资金费率：公开快照及本地采集时间，不是已结算费用或未来资金费预测。")
            st.json(snapshot.get("funding") or {"available": False})
            if current_funding_for_research(snapshot.get("funding"), datetime.now(timezone.utc)) is None:
                st.caption("资金费率缺失、时间不可用或已超过 5 分钟；模型分析时按缺失处理。")
            st.caption("Bitget 小时未平仓量背景：多空合计，不表示方向；数据可能缺失或延迟。")
            st.json(snapshot.get("open_interest") or {"available": False})
            st.caption("已收盘 UTC 日线背景：")
            st.json(snapshot.get("daily_context") or {"ok": False, "reason": "snapshot predates daily context"})
            if "macro_context" in chat_request:
                st.caption("部分宏观背景：日期版本，非实时；缺失项目仍显示为缺失。")
                st.json(chat_request["macro_context"])
            bound_forecast = None
            cycle_state = st.session_state.get("chat_cycle_state")
            archive = restricted = None
            def reload_chat_binding():
                if not cycle_state:
                    return None, None, None, None
                try:
                    from piia.intelligence.forward_archive import read_event
                    from piia.intelligence.chat_context import compact_forecast
                    run_dir = Path(cycle_state["paths"]["run_dir"])
                    current_state = json.loads((run_dir / "STATE.json").read_text(encoding="utf-8"))
                    current_restricted = json.loads((run_dir / "dispatch" / "RESTRICTED_EVIDENCE.json").read_text(encoding="utf-8"))
                    current_archive = read_event(current_state["run_id"])
                    try:
                        forecast = json.loads(Path(current_state["paths"]["forecast"]).read_text(encoding="utf-8"))
                        current_forecast = compact_forecast(chat_request, current_state, forecast, current_archive)
                    except (OSError, ValueError, KeyError, TypeError):
                        current_forecast = None
                    return current_state, current_restricted, current_archive, current_forecast
                except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
                    return None, None, None, None
            if cycle_state:
                cycle_state, restricted, archive, bound_forecast = reload_chat_binding()
                with st.expander("查看跨市场证据摘要"):
                    from piia.intelligence.chat_context import compact_contexts
                    contexts = compact_contexts(chat_request)
                    st.caption("日历和预测市场为有数量限制的样本；BTC 专属指标不能直接解释黄金。")
                    for context in contexts:
                        st.write(context["id"])
                        st.json(json.loads(context["raw"]))
                    if bound_forecast:
                        st.write("CHRONOS_RESEARCH_FORECAST")
                        st.json(json.loads(bound_forecast["raw"]))
                    else:
                        st.warning("本地预测缺失或与存档不一致，不能作为模型证据。")
                    st.caption("未配置模型服务时仍可让当前聊天助手读取研究包；预测摘要是未校准研究信息。")
            candle_rows = snapshot.get("all_closed_candles") or []
            if candle_rows:
                series = pd.DataFrame(candle_rows)
                series["open_at_utc"] = pd.to_datetime(series["open_at_utc"], utc=True)
                series["direction"] = series.apply(lambda row: "上涨" if row["close"] >= row["open"] else "下跌", axis=1)
                base = alt.Chart(series).encode(x=alt.X("open_at_utc:T", title="开盘时间 UTC"),
                                                color=alt.Color("direction:N", scale=alt.Scale(domain=["上涨", "下跌"], range=["#20a879", "#d9534f"])))
                candles = base.mark_rule().encode(y=alt.Y("low:Q", scale=alt.Scale(zero=False), title="价格"), y2="high:Q")
                candles += base.mark_bar(size=5).encode(y="open:Q", y2="close:Q",
                    tooltip=["open_at_utc:T", "open:Q", "high:Q", "low:Q", "close:Q", "volume:Q"])
                st.altair_chart(candles, width="stretch")
            st.subheader("虚拟账户止损风险估算")
            st.caption("输入假设止损价；使用纸面配置本金、快照标记价和本地规格计算，不读取真实持仓、不下单。")
            sizing_side = st.selectbox("假设方向", ["long", "short"], key="sizing_side")
            sizing_stop = st.number_input("假设止损价", min_value=0.0, value=0.0, format="%.4f", key="sizing_stop")
            sizing_gap = st.number_input("可选：更不利的跳空参考价（0 表示不计算）", min_value=0.0,
                                         value=0.0, format="%.4f", key="sizing_gap")
            if st.button("计算虚拟风险示例"):
                from piia.intelligence.paper.preflight import virtual_sizing_example
                from piia.intelligence.paper.risk_file import load_owner_risk
                try:
                    _, owner_hash = load_owner_risk()
                    example = virtual_sizing_example(chat_request, sym, sizing_side, sizing_stop, current_risk_hash=owner_hash,
                                                     gap_exit=sizing_gap if sizing_gap else None)
                    st.session_state["virtual_sizing_example"] = {"snapshot_id": snapshot_id, "result": example}
                except ValueError as exc:
                    st.session_state.pop("virtual_sizing_example", None)
                    st.error(str(exc))
            example_state = st.session_state.get("virtual_sizing_example")
            if (example_state and example_state["snapshot_id"] == snapshot_id
                    and example_state["result"]["symbol"] == sym
                    and example_state["result"]["side"] == sizing_side
                    and example_state["result"]["assumed_stop"] == sizing_stop
                    and (example_state["result"].get("gap_scenario") or {}).get("gap_reference_price") == (sizing_gap if sizing_gap else None)):
                try:
                    quote_time = datetime.fromisoformat(example_state["result"]["reference_at_utc"])
                    quote_recent = 0 <= (datetime.now(timezone.utc) - quote_time).total_seconds() <= 300
                except (KeyError, TypeError, ValueError):
                    quote_recent = False
                if not quote_recent:
                    st.warning("此虚拟估算使用的报价已过期；请重新获取行情并计算。")
                try:
                    spec_time = datetime.fromisoformat(example_state["result"]["spec_snapshot_at_utc"])
                    specs_recent = 0 <= (datetime.now(timezone.utc) - spec_time).total_seconds() <= 300
                except (KeyError, TypeError, ValueError):
                    specs_recent = False
                if not specs_recent:
                    st.warning("合约规格快照已过期或时间未知；数量仅供情景研究，不能用于判断当前订单是否有效。")
                st.json(example_state["result"])
            st.subheader("向配置的模型提问")
            st.caption(f"本次分析对象：{sym}。按下按钮后会发送问题、行情摘要及此合约最近两轮讨论，可能产生 API 费用；其他合约仅作跨市场背景。三个角色使用同一模型，并非独立模型审查。")
            st.caption("模型观点和任何数值预测均未经过留出集校准；历史指标和假设情景不是预测目标。")
            include_scenario = st.checkbox("将本会话的手工风险情景发送给模型", value=False,
                                           help="仅勾选并提交时，手工输入的资金、数量和价格等情景数据会发送给模型服务商。")
            include_portfolio = st.checkbox("将本会话的组合压力情景发送给模型", value=False,
                                            help="仅勾选并提交时，手工输入的组合权益、标的、数量和价格会发送给模型服务商。")
            include_sizing = st.checkbox("将虚拟风险估算发送给模型", value=False,
                                         help="仅在勾选并提交问题时发送本次虚拟账户示例。")
            question = st.text_area("你的交易研究问题", key="chat_question", max_chars=2000)
            if st.button("让模型分析这份快照"):
                from piia.intelligence.agents.independent_context import run_independent_context_research
                try:
                    cutoff = datetime.fromisoformat(chat_request["cutoff_at_utc"])
                    age = (datetime.now(timezone.utc) - cutoff).total_seconds()
                    if not ms["configured"]:
                        st.error("尚未配置模型 API；你也可以让当前聊天助手读取上面的研究包。")
                    elif not chat_request["snapshot_ok"] or not 0 <= age <= 300:
                        st.error("行情不完整或快照超过 5 分钟，请先重新获取。")
                    elif not question.strip():
                        st.error("请先输入研究问题。")
                    else:
                        from piia.intelligence.market_context import refresh_research_context
                        from piia.intelligence.chat_context import compact_instrument
                        evidence = []
                        runtime_missing = list(chat_request["missing_datasets"])
                        runtime_contexts = {}
                        for symbol, item in chat_request["instruments"].items():
                            ticker = item.get("ticker") or {}
                            observed = ticker.get("observed_at_utc")
                            if not observed or not 0 <= (datetime.now(timezone.utc) - datetime.fromisoformat(observed)).total_seconds() <= 300:
                                raise ValueError(f"{symbol} 报价过期，请重新获取行情")
                            try:
                                candle_problems = candle_quality(item.get("all_closed_candles") or [], datetime.now(timezone.utc))
                            except (KeyError, TypeError, ValueError):
                                candle_problems = ["invalid_candles"]
                            if candle_problems:
                                raise ValueError(f"{symbol} K 线不完整或已过期，请重新获取行情")
                            oi = item.get("open_interest")
                            try:
                                if not oi or not timedelta(0) <= datetime.now(timezone.utc) - datetime.fromisoformat(oi["period_end_utc"]) <= timedelta(hours=2):
                                    raise ValueError("missing or stale")
                            except (KeyError, TypeError, ValueError):
                                oi = None
                                runtime_missing.append(f"open_interest_history:{symbol}")
                            funding = current_funding_for_research(item.get("funding"), datetime.now(timezone.utc))
                            if funding is None:
                                runtime_missing.append(f"current_funding:{symbol}")
                            daily, market_context = refresh_research_context(item, datetime.now(timezone.utc))
                            if not daily.get("ok"):
                                runtime_missing.append(f"daily_context:{symbol}")
                            # The ticker can carry another unlabeled funding value; do not bypass freshness checks.
                            ticker_for_model = {k: v for k, v in ticker.items() if k != "funding_rate"}
                            compact = compact_instrument(ticker=ticker_for_model, funding=funding, open_interest=oi,
                                indicators=item.get("indicators_prefix_only"), daily=daily, market_context=market_context,
                                candles=item.get("candles_closed_1h_tail") if symbol == sym else None)
                            runtime_contexts[symbol] = compact["market_context"]
                            evidence.append({"id": symbol, "raw": json.dumps(compact, ensure_ascii=False, default=str)})
                        if "macro_context" in chat_request:
                            from piia.intelligence.chat_context import compact_macro
                            evidence.append(compact_macro(chat_request["macro_context"]))
                        from piia.intelligence.chat_context import compact_contexts
                        evidence.extend(compact_contexts(chat_request))
                        if st.session_state.get("chat_cycle_state"):
                            if bound_forecast:
                                evidence.append(bound_forecast)
                            else:
                                runtime_missing.append("chronos_forecast_unavailable_or_unbound")
                        from piia.intelligence.paper.risk_file import load_owner_risk
                        owner_risk, risk_hash = load_owner_risk()
                        frozen_risk = chat_request.get("paper_risk_context")
                        if frozen_risk and frozen_risk.get("available") and frozen_risk.get("owner_config_hash") != risk_hash:
                            raise ValueError("纸面风控配置已变化，请重新获取研究包。")
                        dialogue = st.session_state.get("chat_dialogue", [])
                        previous = []
                        for turn in [t for t in dialogue if t.get("symbol") == sym][-2:]:
                            previous.append({"question": turn["question"][:600],
                                "prior_model_opinions_not_evidence": {
                                    role: str((turn["analysis"].get(role) or {}).get("summary") or "")[:400]
                                    for role in ("researcher", "critic", "synthesizer")}})
                        payload = {"statement": question, "focus_symbol": sym,
                                   "focus_market_context": runtime_contexts[sym],
                                   "invalidation": "须结合新证据重新评估",
                                   "previous_discussion": previous,
                                   "risk_context": {"kind": "virtual_example_not_actual_account",
                                                    "owner_config": frozen_risk if frozen_risk is not None else owner_risk,
                                                    "owner_config_hash": risk_hash,
                                                    "actual_positions": "unknown",
                                                    "exact_liquidation": "unverified",
                                                    "instruction": "Review downside, invalidation and missing evidence; do not assume account size or current positions."},
                                   "evidence": evidence,
                                   "missing_datasets": list(dict.fromkeys(runtime_missing))}
                        from piia.intelligence.chat_decision_gate import assess_chat_evidence, apply_chat_gate
                        payload["deterministic_review_gate"] = assess_chat_evidence(
                            chat_request, sym, now=datetime.now(timezone.utc), state=cycle_state,
                            restricted=restricted, archive=archive, current_risk_hash=risk_hash,
                            forecast_bound=bool(bound_forecast))
                        if include_scenario:
                            scenario = st.session_state.get("manual_risk_scenario")
                            if not scenario:
                                raise ValueError("请先在仓位风险情景中计算，然后再选择发送。")
                            payload["user_provided_scenario_unverified"] = scenario
                        if include_portfolio:
                            portfolio_scenario = st.session_state.get("manual_portfolio_stress")
                            if not portfolio_scenario:
                                raise ValueError("请先计算组合压力情景，然后再选择发送。")
                            payload["user_provided_portfolio_stress_unverified"] = portfolio_scenario
                        if include_sizing:
                            from piia.intelligence.paper.preflight import virtual_sizing_example
                            saved = st.session_state.get("virtual_sizing_example")
                            if (not saved or saved["snapshot_id"] != snapshot_id or saved["result"]["symbol"] != sym
                                    or saved["result"]["side"] != sizing_side
                                    or saved["result"]["assumed_stop"] != sizing_stop
                                    or (saved["result"].get("gap_scenario") or {}).get("gap_reference_price") != (sizing_gap if sizing_gap else None)):
                                raise ValueError("请先按当前标的、方向和止损重新计算虚拟风险示例。")
                            old = saved["result"]
                            payload["deterministic_virtual_sizing_example"] = virtual_sizing_example(
                                chat_request, sym, old["side"], old["assumed_stop"], current_risk_hash=risk_hash,
                                gap_exit=sizing_gap if sizing_gap else None)
                        result = run_independent_context_research(payload=payload,
                                                                  legal_evidence_ids=[e["id"] for e in evidence])
                        analysis = result.model_dump()
                        if analysis.get("status") != "completed":
                            st.session_state["model_last_failure"] = {
                                "symbol": sym, "status": analysis.get("status"), "usage_records": analysis.get("usage_records", []),
                                "usage_complete": False, "monetary_cost": "unknown"}
                            raise ValueError("模型未完成分析，请检查配置。")
                        current_state, current_restricted, current_archive, current_forecast = reload_chat_binding()
                        try:
                            _, current_risk_hash = load_owner_risk()
                        except (OSError, ValueError, KeyError, TypeError):
                            current_risk_hash = None
                        gate = assess_chat_evidence(chat_request, sym, now=datetime.now(timezone.utc),
                            state=current_state, restricted=current_restricted, archive=current_archive,
                            current_risk_hash=current_risk_hash, forecast_bound=bool(current_forecast))
                        analysis = apply_chat_gate(analysis, gate)
                        if analysis["deterministic_review_gate"]["review_blocked"]:
                            st.warning("程序证据门控未通过：仅供研究，不可作为纸面计划审查依据。")
                        st.session_state.pop("model_last_failure", None)
                        st.session_state["chat_dialogue"] = (dialogue + [{"symbol": sym, "question": question,
                            "analysis": analysis, "cutoff": chat_request["cutoff_at_utc"]}])[-5:]
                except ValueError as exc:
                    st.error(str(exc))
                except Exception:
                    st.error("模型调用失败；请检查配置和网络。未执行交易。")
            if (st.session_state.get("model_last_failure") or {}).get("symbol") == sym:
                st.warning("模型调用未完成。已知用量如下；未返回的用量和费用未知。")
                st.json(st.session_state["model_last_failure"])
            matching_dialogue = [turn for turn in st.session_state.get("chat_dialogue", []) if turn.get("symbol") == sym]
            if matching_dialogue:
                st.caption("本次快照的讨论（可能已经过期；不是交易指令；关闭页面后不保留）")
                for turn in matching_dialogue:
                    with st.chat_message("user"):
                        st.write(turn["question"])
                    with st.chat_message("assistant"):
                        st.caption(f"依据的快照截止 UTC：{turn['cutoff']}")
                        try:
                            fresh = 0 <= (datetime.now(timezone.utc) - datetime.fromisoformat(turn["cutoff"])).total_seconds() <= 300
                        except (ValueError, TypeError, KeyError):
                            fresh = False
                        if not fresh:
                            st.warning("这份分析的快照已过期，不能作为当前计划依据。")
                        st.json(turn["analysis"])
        st.caption("分析完成后，导入与研究包匹配的回复 JSON。导入使用独立数据库，不会写入默认研究库。")
        response_upload = st.file_uploader("选择助手生成的回复 JSON", type=["json"], key="chat_response")
        if st.button("导入聊天分析"):
            path = Path(st.session_state.get("chat_request_path", str(ROOT / "artifacts" / "research_packets" / "current_request.json")))
            try:
                if response_upload is None:
                    raise ValueError("请先选择回复 JSON")
                request = json.loads(path.read_text(encoding="utf-8"))
                response = json.loads(response_upload.getvalue().decode("utf-8"))
                db = ROOT / "data" / "piia" / "piia.research_packets.sqlite"
                result = import_response(request, response, db_path=db,
                                         evidence_store=db.parent / "research_packet_evidence")
                st.json(result)
            except (OSError, ValueError, PiiaError) as exc:
                st.error(f"导入失败：{exc}")

        st.subheader("已保存的聊天研究")
        st.caption("历史记录按导入时间排列；快照时间不是当前行情。研究员、反方和综合意见可能来自同一助手，不能视为独立复核。")
        try:
            from piia.intelligence.chat_history import recent_chat_research
            history = recent_chat_research(ROOT / "data" / "piia" / "piia.research_packets.sqlite")
            if history:
                for record in history:
                    with st.expander(f"{record['symbol']} · 快照 UTC {record['snapshot_cutoff_utc']} · {record['suggested_action']}"):
                        st.json(record)
            else:
                st.info("尚无已导入的聊天研究。")
        except (OSError, ValueError, TypeError, KeyError, sqlite3.Error):
            st.error("无法读取聊天研究记录，请检查本地数据库。")

    with tab_scenario:
        from piia.intelligence.scenarios import position_scenarios
        st.subheader("手工输入：线性 USDT 合约风险情景")
        st.caption("仅计算假设损益，不读取账户、不下单。请输入自行核实的数据；计算结果仅保存在当前页面会话。")
        st.warning("不计算强平价、资金费或其他仓位。价格可能跳过止损，也可能先于止损触发强平。")
        with st.form("manual_risk_form"):
            side_label = st.selectbox("方向", ["多", "空"])
            eq = st.number_input("用于比较的账户权益（USDT）", min_value=0.0, value=100.0, key="scenario_equity")
            entry_px = st.number_input("假设入场成交价", min_value=0.0, value=0.0, format="%.6f", key="scenario_entry")
            stop_px = st.number_input("假设止损成交价", min_value=0.0, value=0.0, format="%.6f", key="scenario_stop")
            gap_px = st.number_input("更不利的跳空成交价", min_value=0.0, value=0.0, format="%.6f", key="scenario_gap")
            qty = st.number_input("合约张数", min_value=0.0, value=0.0, format="%.6f", key="scenario_qty")
            contract_size = st.number_input("每张合约对应的标的数量", min_value=0.0, value=1.0, format="%.6f", key="scenario_contract_size")
            leverage = st.number_input("假设杠杆", min_value=1.0, max_value=150.0, value=2.0, key="scenario_leverage")
            fee_pct = st.number_input("单边手续费率（%）", min_value=0.0, max_value=99.0, value=0.06, format="%.4f", key="scenario_fee")
            submitted = st.form_submit_button("计算两种情景")
        if submitted:
            try:
                st.session_state["manual_risk_scenario"] = position_scenarios(
                    side="long" if side_label == "多" else "short", equity=eq, entry=entry_px,
                    stop=stop_px, gap_exit=gap_px, quantity=qty, contract_size=contract_size,
                    leverage=leverage, fee_rate=fee_pct / 100)
            except ValueError as exc:
                st.session_state.pop("manual_risk_scenario", None)
                st.error(str(exc))
        if st.session_state.get("manual_risk_scenario"):
            st.json(st.session_state["manual_risk_scenario"])

        st.subheader("手工输入：多持仓同步压力情景")
        st.caption("输入当前权益和假设持仓；每个持仓分别填写方向、标记价及同一情景下的价格。不会读取账户、估计相关性或下单。")
        st.caption('JSON 字段：symbol、side（long/short）、quantity、contract_size、mark_price、stress_price、exit_fee_rate、slippage_rate；费率用小数，例如 0.0006。')
        with st.form("manual_portfolio_stress_form"):
            portfolio_equity = st.number_input("当前组合权益（USDT）", min_value=0.0, value=100.0, key="portfolio_stress_equity")
            portfolio_positions = st.text_area("假设持仓 JSON 数组", value="[]", key="portfolio_stress_positions")
            portfolio_submitted = st.form_submit_button("计算组合压力情景")
        if portfolio_submitted:
            try:
                from piia.intelligence.scenarios import portfolio_stress
                st.session_state["manual_portfolio_stress"] = portfolio_stress(equity=portfolio_equity, positions=json.loads(portfolio_positions))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                st.session_state.pop("manual_portfolio_stress", None)
                st.error(f"无法计算组合情景：{exc}")
        if st.session_state.get("manual_portfolio_stress"):
            st.json(st.session_state["manual_portfolio_stress"])

    with tab_create:
        st.subheader("Create a labelled example thesis")
        inst = st.selectbox("Instrument", list(instruments.keys()), format_func=lambda k: f"{k} — {instruments[k]}")
        author = st.text_input("Author", "yan-example")
        direction = st.selectbox("Direction", ["long", "short", "flat"])
        horizon = st.number_input("Horizon days", 1, 365, 7)
        statement = st.text_area("Statement", "LABELED_FIXTURE_EXAMPLE: BTC may rise over the next week. Not a recommendation or a historical track record.")
        invalidation = st.text_area("Invalidation", "Close below the stated evidence price by more than 8% before expiry.")
        confidence = st.slider("Subjective confidence (not calibrated accuracy)", 0.0, 1.0, 0.4)
        raw = st.text_area("Paste raw evidence snapshot", "example fixture snapshot: {\"pair\":\"BTC-USD\",\"note\":\"labelled demo, not live market data\"}")
        provider = st.text_input("Provider", "manual-paste")
        if st.button("Save thesis"):
            ev = EvidenceIn(raw=raw, provider=provider, data_quality="unverified", label="user_provided", relation="supports")
            payload = ThesisCreate(
                instrument_id=inst, author=author, direction=direction, horizon_days=int(horizon),
                invalidation=invalidation, subjective_confidence=float(confidence), statement=statement,
                label="LABELED_FIXTURE_EXAMPLE", is_example=True, evidence=[ev],
            )
            result = svc.create_thesis(payload)
            st.session_state["thesis_id"] = result["id"]
            st.success(f"Saved thesis {result['id']}")
            st.json(result)

    thesis_id = st.text_input("Thesis id", st.session_state.get("thesis_id", ""))

    with tab_view:
        if st.button("Load thesis") and thesis_id:
            st.session_state["loaded"] = svc.get_thesis(thesis_id)
        loaded = st.session_state.get("loaded")
        if loaded:
            st.json(loaded)
            current = loaded["current_revision_id"]
            body = st.text_area("Manual counterargument", "Rates stay restrictive and ETF flows stay negative.")
            if st.button("Add manual countercase"):
                st.write(svc.add_counterargument(current, CounterIn(actor="user", body=body, generation="manual")))
            if st.button("Add labelled deterministic demo countercase"):
                st.warning("This is a template, not independent LLM research.")
                st.write(svc.deterministic_demo_counter(current))
            reason = st.text_input("Revision change reason")
            new_stmt = st.text_area("Revised statement", loaded["revisions"][-1]["statement"] if loaded["revisions"] else "")
            if st.button("Append revision"):
                prev = loaded["revisions"][-1]
                st.write(svc.revise_thesis(thesis_id, ThesisRevise(
                    author=author if "author" in dir() else "yan-example",
                    direction=prev["direction"], horizon_days=prev["horizon_days"],
                    invalidation=prev["invalidation"], subjective_confidence=prev["subjective_confidence"],
                    statement=new_stmt, change_reason=reason, expected_revision_id=current, is_example=True,
                )))

    with tab_risk:
        st.caption("Sample fixture portfolio only. Leveraged or perpetual executable plans are blocked because the exact margin and liquidation model is not implemented.")
        mark = st.number_input("Fixture BTC spot mark", 1.0, 1_000_000.0, 90000.0)
        qty = st.number_input("Fixture quantity", 0.0, 10.0, 0.0004, format="%.6f")
        if st.button("Build sample portfolio, risk config and plan"):
            if not thesis_id:
                st.error("Enter a thesis id first")
            else:
                th = svc.get_thesis(thesis_id)
                port = svc.create_portfolio(svc.default_sample_portfolio(mark=float(mark), quantity=float(qty)))
                cfg = svc.create_risk_config(RiskConfigIn(payload={}, notes="default phase-1 limits"))
                plan = svc.create_plan(PlanIn(revision_id=th["current_revision_id"], portfolio_id=port["id"], risk_config_id=cfg["id"], side="long", leverage=1.0))
                st.session_state["plan_id"] = plan["id"]
                st.json({"portfolio": port, "risk_config": cfg, "plan": plan})
        plan_id = st.text_input("Plan id", st.session_state.get("plan_id", ""))
        choice = st.selectbox("User choice", ["approve", "reject"])
        reject_reason = st.text_input("Rejection reason", "")
        if st.button("Submit explicit approval decision"):
            st.caption("This is a user action in the local app, not a model auto-approval, and not a live trade.")
            st.json(svc.approve_plan(plan_id, ApprovalIn(actor="local-user", user_choice=choice, rejection_reason=reject_reason or None)))

    with tab_eval:
        st.caption("手工报价未独立验证，不能用于统计准确率。需填写真实报价时间：起点为观点创建前 5 分钟内，终点为到期后 5 分钟内；数据不合适时结果为未知。")
        start_px = st.number_input("Start price (optional)", 0.0, 1_000_000.0, 0.0)
        end_px = st.number_input("End price (optional)", 0.0, 1_000_000.0, 0.0)
        start_source = st.text_input("Start quote source (optional)", "")
        start_time = st.text_input("Start quote time (ISO 8601 with UTC offset; optional)", "")
        end_source = st.text_input("End quote source (optional)", "")
        end_time = st.text_input("End quote time (ISO 8601 with UTC offset; optional)", "")
        missing = st.checkbox("Treat end quote as missing")
        if st.button("Evaluate current revision") and thesis_id:
            try:
                th = svc.get_thesis(thesis_id)
                rev = next(r for r in th["revisions"] if r["id"] == th["current_revision_id"])
                st.json(svc.evaluate_outcome(OutcomeIn(
                    revision_id=rev["id"],
                    start={"price": start_px or None, "source": start_source or None,
                           "observed_at": start_time.strip() or None},
                    end={"price": None if missing else end_px or None, "source": end_source or None,
                         "observed_at": end_time.strip() or None},
                )))
            except (ValueError, PiiaError) as exc:
                st.error(f"无法评估：{exc}")
        if thesis_id and st.button("Show audit trace"):
            st.json(svc.audit_trace(thesis_id))

    ext = ExtendedService(svc)
    with tab_research:
        st.caption("On-demand researcher → critic → synthesizer. Model roles use the same configured model; independent review is not wired. Fixture mode is labelled and cannot auto-approve a plan.")
        st.info("研究员整理证据，反方检查漏洞，综合角色汇总分歧。先创建观点并附证据；真实模型模式需要配置服务，所有结论都需人工判断。")
        mode = st.selectbox("Run mode", ["fixture", "model"])
        if st.button("Run research") and thesis_id:
            run = ext.run_research(thesis_id, mode=mode, actor="local-user")
            if run["status"] == "completed":
                st.session_state["research_run_id"] = run["id"]
            st.json(run)
        st.caption("Disagreements, blocked claims, and injection flags appear in the run result. Subjective confidence is not a forecast accuracy.")

    with tab_paper:
        st.caption("Local simplified paper. The prefilled prices below are fixed historical examples, not live quotes. Default starting cash comes from the owner risk file. Venue margin/liquidation are unsupported. Not real performance.")
        if st.button("Open paper account"):
            acc = ext.paper.open_account()
            st.session_state["account_id"] = acc["id"]
            st.json(acc)
        account_id = st.text_input("Paper account id", st.session_state.get("account_id", ""))
        symbol = st.selectbox("Contract", list(CONTRACTS))
        side = st.selectbox("Side", ["long", "short"], key="paper_side")
        entry = st.number_input("Entry price", 0.01, 1_000_000.0, float(SAMPLE_MARKS[symbol]["mark"]))
        if side == "long":
            default_stop = float(SAMPLE_MARKS[symbol]["mark"]) * 0.98
        else:
            default_stop = float(SAMPLE_MARKS[symbol]["mark"]) * 1.02
        stop = st.number_input("Stop loss", 0.01, 1_000_000.0, default_stop)
        lev = st.number_input("Leverage (owner cap 2)", 1.0, 10.0, 2.0)
        run_id = st.text_input("Research run id (optional)", st.session_state.get("research_run_id", ""))
        if st.button("Build sandbox paper plan") and thesis_id and account_id:
            plan = ext.create_sandbox_plan(
                thesis_id=thesis_id, account_id=account_id, symbol=symbol, side=side,
                entry_price=float(entry), stop_loss=float(stop), leverage=float(lev),
                research_run_id=run_id or None,
            )
            st.session_state["paper_plan_id"] = plan["id"]
            st.json(plan)
        paper_plan_id = st.text_input("Paper plan id", st.session_state.get("paper_plan_id", ""))
        if st.button("Explicitly approve paper plan") and paper_plan_id:
            st.caption("Human action only. Not a live order.")
            dec = ext.approve_paper(paper_plan_id, ApprovalIn(actor="local-user", user_choice="approve"))
            st.session_state["decision_id"] = dec["id"]
            st.json(dec)
        decision_id = st.text_input("Decision id", st.session_state.get("decision_id", ""))
        if st.button("Execute local paper fill") and account_id and decision_id:
            filled = ext.paper.execute_approved(account_id=account_id, decision_id=decision_id)
            if filled.get("position"):
                st.session_state["position_id"] = filled["position"]["id"]
            st.json(filled)
        position_id = st.text_input("Position id", st.session_state.get("position_id", ""))
        c1,c2,c3,c4 = st.columns(4)
        o=c1.number_input("Candle open", 0.01, 1_000_000.0, float(SAMPLE_MARKS[symbol]["mark"]))
        h=c2.number_input("High", 0.01, 1_000_000.0, float(SAMPLE_MARKS[symbol]["mark"])*1.01)
        l=c3.number_input("Low", 0.01, 1_000_000.0, float(SAMPLE_MARKS[symbol]["mark"])*0.97)
        c=c4.number_input("Close", 0.01, 1_000_000.0, float(SAMPLE_MARKS[symbol]["mark"]))
        if st.button("On-demand stop/gap check") and position_id:
            st.json(ext.paper.check_stop(position_id, {"open":o,"high":h,"low":l,"close":c}))
        kill_on = st.checkbox("Enable kill switch")
        if st.button("Set kill switch"):
            st.json(ext.paper.set_kill("local-user", kill_on, "manual dashboard action"))
        if account_id and st.button("Refresh account"):
            st.json(ext.paper.account_dict(ext.paper.get_account(account_id)))
finally:
    svc.session.close()
