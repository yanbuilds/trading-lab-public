"""Deterministic, traceable multi-horizon judgment synthesis engine (PIIA Judgment v3).

Combines verified evidence across 5 external domains + causal market bars into per-symbol
`intraday` and `swing_1_3d` judgments, keeps pure technical direction alongside final judgment,
deduplicates collinear drivers by `causal_driver_group`, distinguishes hard gates from background
modifiers, and displays frozen PIIA v2.2.1 formal signals & paper eligibility separately.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from typing import Any

from piia.intelligence.agent_v2.pillars import (
    SYMBOLS,
    build_bar_evidence_for_symbol,
    canonical_sha256,
    load_v2_config,
    point_in_time_macro_context,
)
from piia.intelligence.agent_v2.risk_gate import evaluate_programmatic_risk, run_symbol_decision_trace
from piia.intelligence.judgment.evidence import (
    build_forecast_and_model_evidence,
    build_macro_evidence,
    build_sentiment_flow_evidence,
    load_judgment_config,
    make_atom,
    validate_market_snapshot_coherence,
)
from piia.intelligence.judgment.news import classify_and_filter_news_items, parse_aware_timestamp
from piia.intelligence.judgment.tide import compute_market_tide

BIAS_CN = {
    "bullish_bias": "偏多",
    "bearish_bias": "偏空",
    "neutral_stand_aside": "观望",
}
HORIZONS = ("intraday", "swing_1_3d")
PARTICIPATION_CN = {
    "changed_judgment": "改变判断",
    "background_only": "仅作背景",
    "expired": "过期",
    "missing": "缺失",
    "not_run": "未运行",
}


def compute_earliest_executable_utc(bar_cutoff_dt: datetime, judgment_generated_dt: datetime) -> datetime:
    """Return earliest 1h bar open >= max(bar_cutoff_utc, judgment_generated_at_utc)."""
    ref = max(bar_cutoff_dt.astimezone(timezone.utc), judgment_generated_dt.astimezone(timezone.utc))
    floored = ref.replace(minute=0, second=0, microsecond=0)
    return floored if ref == floored else floored + timedelta(hours=1)


def _safe_f(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        fv = float(v)
        return fv if math.isfinite(fv) else None
    except (TypeError, ValueError):
        return None


def _pure_technical_judgment(pos: dict[str, Any], *, horizon: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Compute baseline pure-technical directional judgment per (symbol, horizon)."""
    tcfg = cfg["technical_thresholds"]
    if pos.get("position_state") == "unknown":
        return {"bias": "neutral_stand_aside", "label_cn": BIAS_CN["neutral_stand_aside"], "rationale": "K 线或均线预热不足，技术面观望"}
    m = pos["metrics"]
    lvl = pos["invalidation_levels"]
    close_px = float(pos["close"])
    h1_dir, h4_dir, d_dir = pos["h1_dir"], pos["h4_dir"], pos["daily_dir"]
    adx14 = float(m["adx14"])
    rsi14 = _safe_f(m.get("rsi14"))
    atr_ratio = _safe_f(m.get("atr_ratio"))
    e20 = float(m["h1_ema20"])

    if atr_ratio is not None and atr_ratio >= float(tcfg["extreme_vol_atr_ratio"]) and horizon == "intraday":
        return {"bias": "neutral_stand_aside", "label_cn": BIAS_CN["neutral_stand_aside"], "rationale": f"日内极端波动 (ATR比={atr_ratio:.2f}>=2.2)，技术面日内观望"}

    if horizon == "intraday":
        if h4_dir == "up" and h1_dir == "up" and close_px >= e20 and adx14 >= float(tcfg["adx_range_max"]):
            return {"bias": "bullish_bias", "label_cn": BIAS_CN["bullish_bias"], "rationale": f"4H与1H双均线多头一致，现价站上 1H EMA20 ({e20:.2f}), ADX={adx14:.1f}"}
        if h4_dir == "down" and h1_dir == "down" and close_px <= e20 and adx14 >= float(tcfg["adx_range_max"]):
            return {"bias": "bearish_bias", "label_cn": BIAS_CN["bearish_bias"], "rationale": f"4H与1H双均线空头一致，现价位于 1H EMA20 ({e20:.2f}) 下方, ADX={adx14:.1f}"}
        bb_lo, bb_up = _safe_f(lvl.get("bb_lower")), _safe_f(lvl.get("bb_upper"))
        in_pure_range = pos.get("position_state") == "range_consolidation"
        if in_pure_range and adx14 < float(tcfg["adx_range_max"]) and rsi14 is not None and bb_lo is not None and bb_up is not None:
            if close_px < bb_lo and rsi14 < float(tcfg["rsi_oversold"]) and h4_dir == "up" and not m.get("broke_donchian_low") and not m.get("below_h4_ema50"):
                return {"bias": "bullish_bias", "label_cn": BIAS_CN["bullish_bias"], "rationale": f"震荡下轨超卖反弹 (RSI14={rsi14:.1f}<35 且未向下破位)"}
            if close_px > bb_up and rsi14 > float(tcfg["rsi_overbought"]) and h4_dir == "down" and not m.get("broke_donchian_high") and not m.get("above_h4_ema50"):
                return {"bias": "bearish_bias", "label_cn": BIAS_CN["bearish_bias"], "rationale": f"震荡上轨超买回落 (RSI14={rsi14:.1f}>65 且未向上突破)"}
        return {"bias": "neutral_stand_aside", "label_cn": BIAS_CN["neutral_stand_aside"], "rationale": f"日内 1H/4H 结构未共振 ({pos['structure_kind_cn']})，纯技术面观望"}

    # swing_1_3d technical baseline uses 1D + 4H + ADX structure
    if d_dir == "up" and h4_dir == "up":
        return {"bias": "bullish_bias", "label_cn": BIAS_CN["bullish_bias"], "rationale": "1D 与 4H 趋势多头对齐（日线EMA20>50/SMA200 且 4H EMA20>50）"}
    if d_dir == "down" and h4_dir == "down":
        return {"bias": "bearish_bias", "label_cn": BIAS_CN["bearish_bias"], "rationale": "1D 与 4H 趋势空头对齐（日线EMA20<50/SMA200 且 4H EMA20<50）"}
    if d_dir == "unknown" and h4_dir == h1_dir and adx14 >= float(tcfg["adx_trend_min"]):
        b = "bullish_bias" if h4_dir == "up" else "bearish_bias"
        return {"bias": b, "label_cn": BIAS_CN[b], "rationale": f"日线历史不足，暂以 4H+1H 同向趋势且 ADX={adx14:.1f}>=22 代理（已披露日线缺口）"}
    return {"bias": "neutral_stand_aside", "label_cn": BIAS_CN["neutral_stand_aside"], "rationale": f"1D({d_dir})/4H({h4_dir}) 方向分化或震荡，纯技术面 1-3 日观望"}


def _news_items_to_atoms(news_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for item in news_items:
        ver_ok = item.get("verification_status") == "verified"
        fresh = item.get("freshness_status") or "missing"
        nkind = item.get("news_kind") or "unverified"
        is_primary = ver_ok and nkind == "primary_release" and fresh in {"fresh", "lagged_usable"}
        syms = item.get("affected_symbols") or []
        horizons = item.get("eligible_horizons") or []
        chan = str(item.get("transmission_channel") or "")
        topic_group = "news_primary_policy" if "利率" in chan or "政策" in chan else (
            "news_primary_geopolitics" if "避险" in chan else (
                "news_primary_ndx_event" if syms == ["NDX100USDT"] else "news_primary_btc_reg"
            )
        )
        bias_map = dict(item.get("bias_by_symbol") or {})
        age_h = item.get("age_hours")
        atoms.append(
            make_atom(
                evidence_id=f"news:{item['news_id']}",
                domain="news",
                factor_name=f"{'[原始]' if nkind == 'primary_release' else ('[重复转载]' if nkind == 'duplicate_repost' else ('[解读]' if nkind == 'secondary_interpretation' else '[未核实]'))} {item['title'][:90]}",
                source_name=str(item.get("source_name") or "unknown"),
                source_url=item.get("url"),
                raw_sha256=item.get("raw_sha256"),
                data_tier="realtime_verified" if is_primary else ("lagged_official_report" if ver_ok else "unavailable"),
                observed_at_utc=item.get("published_at_utc"),
                published_at_utc=item.get("published_at_utc"),
                available_at_utc=item.get("published_at_utc"),
                available_at_known=ver_ok,
                retrieved_at_utc=str(item.get("retrieved_at_utc") or ""),
                freshness_status=fresh,
                applies_to_symbols=syms,
                applies_to_horizons=horizons,
                value_summary={
                    "news_kind": nkind,
                    "duplicate_of": item.get("duplicate_of"),
                    "verification_status": item.get("verification_status"),
                    "transmission_channel": chan,
                    "age_hours": age_h,
                },
                direction_by_symbol=bias_map if is_primary else {s: "unknown" for s in SYMBOLS},
                causal_driver_group=topic_group if is_primary else f"news_context_{item['news_id']}",
                is_hard_gate_candidate=bool(is_primary and age_h is not None and float(age_h) <= 12.0),
                can_decide_direction_alone=False,
                limitation_note=str(item.get("limitation_note") or ""),
            )
        )
    return atoms


def _official_v2_signal_for_symbol(
    sym: str,
    *,
    prep_df: Any,
    market_snapshot: dict[str, Any],
    books_snapshot: dict[str, Any],
    tiers_snapshot: dict[str, Any],
    alfred_series: dict[str, dict[str, Any]],
    macro_summary: dict[str, Any],
    sent_summary: dict[str, Any],
    cutoff_utc: datetime,
    chronos_payload: dict[str, Any] | None,
    starting_cash_usdt: float = 500.0,
) -> dict[str, Any]:
    """Evaluate frozen PIIA v2.2.1 signal & paper eligibility read-only."""
    v2_cfg = load_v2_config()
    if prep_df is None or len(prep_df) < 2:
        return {
            "official_v2_2_1_signal": "stand_aside",
            "stand_aside_reason": "insufficient_1h_bars",
            "setup_type": "none",
            "disagreements": [],
            "paper_eligibility": {"eligible": False, "status": "blocked", "label_cn": "否", "review_required": False, "side": None, "blocking_reasons": ["insufficient_1h_bars"], "review_triggers": [], "sizing_preview": None},
        }
    row_sig = prep_df.iloc[-1]
    prev_c = float(prep_df.iloc[-2]["close"])
    inst = (market_snapshot.get("instruments") or {}).get(sym) or {}
    f_raw = (inst.get("funding") or {}).get("funding_rate")
    try:
        f_rate = float(f_raw) if f_raw is not None else None
    except (TypeError, ValueError):
        f_rate = None

    macro_hist = {sid: obj.get("observations") or [] for sid, obj in (alfred_series or {}).items()}
    v2_macro = point_in_time_macro_context(macro_hist, cutoff_utc, cfg=v2_cfg)
    ext = {
        "orderbook": ((books_snapshot.get("instruments") or {}).get(sym) or {}).get("orderbook"),
        "tiers": ((tiers_snapshot.get("instruments") or {}).get(sym) or {}).get("tiers"),
        "open_interest": inst.get("open_interest"),
        "flows_positioning": {
            "status": "collected_partial_or_complete",
            "btc_etf_flows": {"usable": sent_summary.get("btc_etf_3d_musd") is not None, "latest_complete_total_musd": sent_summary.get("btc_etf_3d_musd")},
            "liquidity_state": "normal",
        },
        "chronos_forecast": chronos_payload or {"weights_ran": False, "forecasts": {}},
    }
    bundle = build_bar_evidence_for_symbol(
        sym,
        feature_row=row_sig,
        prev_close=prev_c,
        decision_utc=cutoff_utc,
        funding_rate_8h=f_rate,
        macro_ctx=v2_macro,
        external_live_pillars=ext,
    )
    trace = run_symbol_decision_trace(
        bundle,
        equity_usdt=float(starting_cash_usdt),
        free_cash_usdt=float(starting_cash_usdt),
        open_positions_count=0,
        kill_switch_halted=False,
        cfg=v2_cfg,
    )
    rk = trace["programmatic_risk_gate"]
    final_act = rk["final_action"]
    setup_type = trace["synthesizer"].get("setup_type")
    blockers = []
    if final_act == "stand_aside" and rk.get("stand_aside_reason"):
        blockers.append(str(rk["stand_aside_reason"]))
    is_ok = bool(final_act in {"long", "short"} and rk.get("approved_by_risk_gate"))
    return {
        "official_v2_2_1_signal": final_act,
        "stand_aside_reason": rk.get("stand_aside_reason"),
        "setup_type": setup_type,
        "disagreements": trace["synthesizer"].get("disagreements") or [],
        "paper_eligibility": {
            "eligible": is_ok,
            "status": "eligible" if is_ok else "blocked",
            "label_cn": "是" if is_ok else "否",
            "review_required": False,
            "review_triggers": [],
            "candidate_side": final_act if final_act in {"long", "short"} else None,
            "side": final_act if final_act in {"long", "short"} else None,
            "blocking_reasons": blockers,
            "entry_plan": rk.get("entry_plan"),
            "exit_plan": rk.get("exit_plan"),
            "position_sizing": rk.get("position_sizing"),
            "cost_and_risk": rk.get("cost_and_risk"),
            "equity_reference_usdt": float(starting_cash_usdt),
        },
    }


def _apply_v3_paper_review_gate(
    sym: str,
    *,
    v2_ctrl: dict[str, Any],
    pos: dict[str, Any],
    atoms: list[dict[str, Any]],
    news_enabled: bool,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Screen counter-trend v2 signals against shock structural breakouts and opposing primary news."""
    pe = dict(v2_ctrl["paper_eligibility"])
    cand_side = v2_ctrl["official_v2_2_1_signal"]
    setup_type = str(v2_ctrl.get("setup_type") or "")
    if cand_side not in {"long", "short"} or not pe.get("eligible"):
        return pe, []

    m = pos.get("metrics") or {}
    lvl = pos.get("invalidation_levels") or {}
    triggers: list[dict[str, str]] = []
    shock = bool(m.get("shock_volatility"))
    is_mr = setup_type == "range_reversion"

    if cand_side == "short":
        if bool(m.get("broke_donchian_high")) or (shock and bool(m.get("above_h4_ema50")) and is_mr):
            triggers.append(
                {
                    "trigger": "shock_breakout_vs_counter_trend_short",
                    "detail": (
                        f"{sym} {setup_type} 触发做空，但收盘 {pos.get('close')} 已突破 "
                        f"48h上轨({lvl.get('donchian_48h_high')}) / 4H_EMA50({lvl.get('h4_ema50')}) "
                        f"且波动放大(bar_range_atr={m.get('bar_range_atr')}, atr_ratio={m.get('atr_ratio')})"
                    ),
                }
            )
    elif cand_side == "long":
        if bool(m.get("broke_donchian_low")) or (shock and bool(m.get("below_h4_ema50")) and is_mr):
            triggers.append(
                {
                    "trigger": "shock_breakdown_vs_counter_trend_long",
                    "detail": (
                        f"{sym} {setup_type} 触发做多，但收盘 {pos.get('close')} 已跌破 "
                        f"48h下轨({lvl.get('donchian_48h_low')}) / 4H_EMA50({lvl.get('h4_ema50')}) "
                        f"且波动放大(bar_range_atr={m.get('bar_range_atr')}, atr_ratio={m.get('atr_ratio')})"
                    ),
                }
            )

    if news_enabled:
        for atom in atoms:
            if atom.get("domain") != "news" or sym not in (atom.get("applies_to_symbols") or []):
                continue
            if atom.get("freshness_status") != "fresh":
                continue
            if (atom.get("value_summary") or {}).get("news_kind") != "primary_release":
                continue
            age_h = _safe_f((atom.get("value_summary") or {}).get("age_hours"))
            if age_h is None or age_h > 12.0:
                continue
            d = (atom.get("direction_by_symbol") or {}).get(sym)
            opposes = (cand_side == "short" and d == "bullish") or (cand_side == "long" and d == "bearish")
            aligned_dir = (cand_side == "long" and d == "bullish") or (cand_side == "short" and d == "bearish")
            if opposes or (bool(atom.get("is_hard_gate_candidate")) and not aligned_dir):
                triggers.append(
                    {
                        "trigger": f"opposing_or_material_primary_catalyst:{atom['evidence_id']}",
                        "detail": f"{atom['factor_name']} ({atom['source_name']}, {atom.get('published_at_utc')}, dir={d})",
                    }
                )

    if triggers:
        trig_ids = [t["trigger"] for t in triggers]
        pe["eligible"] = False
        pe["status"] = "pending_review"
        pe["label_cn"] = "待复核"
        pe["review_required"] = True
        pe["review_triggers"] = trig_ids
        pe["review_trigger_details"] = triggers
        pe["side"] = None
        pe["blocking_reasons"] = [f"pending_review:{','.join(trig_ids)}"]
    return pe, triggers


def _build_factor_usage_ledger(
    *,
    atoms: list[dict[str, Any]],
    tide: dict[str, Any],
    forecast_summary: dict[str, Any],
    judgments: dict[str, Any],
    ablation: dict[str, bool],
) -> dict[str, list[dict[str, Any]]]:
    """Build 7-category factor audit ledger strictly distinguishing API OK from factor participation."""
    categories: dict[str, list[dict[str, Any]]] = {
        "macro": [],
        "tide": [],
        "sentiment_positioning": [],
        "etf_and_onchain": [],
        "polymarket": [],
        "news": [],
        "forecast_models": [],
    }
    changed_eids: set[str] = set()
    changed_domains: set[str] = set()
    for sym in SYMBOLS:
        j = judgments.get(sym) or {}
        if j.get("paper_eligibility", {}).get("review_required"):
            changed_domains.add("tide")
        for h in HORIZONS:
            hd = j.get(h) or {}
            if hd.get("changed_vs_technical"):
                for drv in hd.get("active_causal_drivers") or []:
                    changed_domains.add(str(drv.get("domain") or ""))
                    if drv.get("summary", "").startswith("["):
                        changed_eids.add(drv["summary"].split("]", 1)[0].lstrip("["))

    def _category_for_atom(atom: dict[str, Any]) -> str:
        eid = atom["evidence_id"]
        dom = atom["domain"]
        if dom == "macro":
            return "macro"
        if dom == "news" or eid == "sentiment:ndx:sec_edgar_constituents":
            return "news"
        if dom == "forecast":
            return "forecast_models"
        if eid == "sentiment:polymarket:filtered_events":
            return "polymarket"
        if eid in {
            "sentiment:btc:spot_etf_flows",
            "sentiment:xau:gold_etf_flows",
            "sentiment:btc:onchain_network_activity",
            "sentiment:btc:onchain_exchange_netflow",
        }:
            return "etf_and_onchain"
        return "sentiment_positioning"

    # Tide entries first
    mb = tide.get("macro_backdrop") or {}
    cs = tide.get("cycle_stage") or {}
    if not ablation["enable_tide"]:
        mb_st, cs_st = "not_run", "not_run"
    else:
        mb_st = "missing" if mb.get("state") == "unknown" else ("changed_judgment" if "tide" in changed_domains else "background_only")
        cs_st = "missing" if cs.get("stage") == "insufficient_evidence" else ("changed_judgment" if "tide" in changed_domains else "background_only")
    categories["tide"].append(
        {
            "factor_id": "tide:macro_backdrop",
            "factor_name": "市场潮汐第一层：宏观流动性大环境",
            "participation_status": mb_st,
            "participation_status_cn": PARTICIPATION_CN[mb_st],
            "participated_in_judgment": mb_st in {"changed_judgment", "background_only"},
            "api_responded_ok": mb.get("known_macro_pillars", 0) >= 2,
            "source_name": "ALFRED / DTWEXBGS / DFII10 / Net Liquidity / VIX",
            "source_url": None,
            "timestamp_utc": tide.get("macro_backdrop", {}).get("asof_utc"),
            "detail_reason": mb.get("state_cn") or mb.get("state"),
        }
    )
    categories["tide"].append(
        {
            "factor_id": "tide:cycle_stage",
            "factor_name": "市场潮汐第二层：周期阶段",
            "participation_status": cs_st,
            "participation_status_cn": PARTICIPATION_CN[cs_st],
            "participated_in_judgment": cs_st in {"changed_judgment", "background_only"},
            "api_responded_ok": cs.get("stage") != "insufficient_evidence",
            "source_name": "Cross-market Bars + OI + ETF Flow",
            "source_url": None,
            "timestamp_utc": None,
            "detail_reason": cs.get("stage_cn") or cs.get("stage"),
        }
    )
    for sym in SYMBOLS:
        pos = (tide.get("instrument_positions") or {}).get(sym) or {}
        if not ablation["enable_tide"]:
            pst = "not_run"
        elif pos.get("position_state") == "unknown":
            pst = "missing"
        elif pos.get("position_state") in {"bounce_in_downtrend", "potential_bullish_reversal", "potential_bearish_reversal", "pullback_in_uptrend"}:
            pst = "changed_judgment" if "tide" in changed_domains else "background_only"
        else:
            pst = "background_only"
        categories["tide"].append(
            {
                "factor_id": f"tide:position:{sym}",
                "factor_name": f"{sym} 多周期位置 (1D/4H/1H)",
                "participation_status": pst,
                "participation_status_cn": PARTICIPATION_CN[pst],
                "participated_in_judgment": pst in {"changed_judgment", "background_only"},
                "api_responded_ok": pos.get("position_state") != "unknown",
                "source_name": "Bitget Closed 1D/4H/1H Klines",
                "source_url": None,
                "timestamp_utc": None,
                "detail_reason": pos.get("structure_kind_cn"),
            }
        )

    for atom in atoms:
        cat = _category_for_atom(atom)
        eid = atom["evidence_id"]
        fresh = atom["freshness_status"]
        roles = set((atom.get("role_by_symbol_horizon") or {}).values())
        val_sum = atom.get("value_summary") or {}
        api_ok = val_sum.get("api_responded_ok")
        if api_ok is None:
            api_ok = fresh in {"fresh", "lagged_usable"}

        if "disabled_by_ablation" in roles:
            p_st = "not_run"
        elif eid.endswith(":chronos_neural") and fresh == "missing":
            p_st = "not_run"
        elif fresh in {"stale_excluded", "future_excluded"} or val_sum.get("status") in {"expired_excluded"}:
            p_st = "expired"
        elif fresh == "missing":
            p_st = "missing"
        elif eid in changed_eids or bool(roles & {"changed_direction", "triggered_veto"}):
            p_st = "changed_judgment"
        else:
            p_st = "background_only"

        categories[cat].append(
            {
                "factor_id": eid,
                "factor_name": atom["factor_name"],
                "participation_status": p_st,
                "participation_status_cn": PARTICIPATION_CN[p_st],
                "participated_in_judgment": p_st in {"changed_judgment", "background_only"},
                "api_responded_ok": bool(api_ok),
                "source_name": atom.get("source_name"),
                "source_url": atom.get("source_url"),
                "timestamp_utc": atom.get("published_at_utc") or atom.get("observed_at_utc") or atom.get("retrieved_at_utc"),
                "detail_reason": val_sum.get("missing_or_excluded_reason") or val_sum.get("reason") or atom.get("limitation_note"),
            }
        )

    ai_prov = (forecast_summary or {}).get("ai_provenance") or {}
    for ai_key, ai_label in (("llm_research_opinion", "外部大模型研究意见"), ("independent_ai_critic", "独立 AI 反方审查")):
        info = ai_prov.get(ai_key) or {}
        ran = bool(info.get("executed"))
        st = "background_only" if ran else "not_run"
        categories["forecast_models"].append(
            {
                "factor_id": f"model:{ai_key}",
                "factor_name": ai_label,
                "participation_status": st,
                "participation_status_cn": PARTICIPATION_CN[st],
                "participated_in_judgment": ran,
                "api_responded_ok": ran,
                "source_name": info.get("provider") or info.get("status"),
                "source_url": None,
                "timestamp_utc": info.get("run_at_utc"),
                "detail_reason": info.get("note") or info.get("status"),
            }
        )
    return categories


def _synthesize_one(
    sym: str,
    horizon: str,
    *,
    pos: dict[str, Any],
    tide: dict[str, Any],
    macro_summary: dict[str, Any],
    vix_summary: dict[str, Any],
    atoms: list[dict[str, Any]],
    ablation: dict[str, bool],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    key = f"{sym}:{horizon}"
    tech = _pure_technical_judgment(pos, horizon=horizon, cfg=cfg)
    tech_bias = tech["bias"]
    tech_vote = 2 if tech_bias == "bullish_bias" else (-2 if tech_bias == "bearish_bias" else 0)

    hard_gates: list[str] = []
    supporting: list[str] = []
    counter: list[str] = []
    gaps: list[str] = list(pos.get("missing_inputs") or [])

    if ablation["enable_macro"] and horizon == "intraday" and macro_summary.get("imminent_tier1_event"):
        hard_gates.append(f"imminent_tier1_macro_event:{macro_summary['imminent_event_title'] if 'imminent_event_title' in macro_summary else macro_summary['imminent_tier1_event']}")
    if horizon == "intraday":
        ar = _safe_f((pos.get("metrics") or {}).get("atr_ratio"))
        if ar is not None and ar >= float(cfg["technical_thresholds"]["extreme_vol_atr_ratio"]):
            hard_gates.append(f"extreme_intraday_volatility:atr_ratio={ar:.2f}")

    cycle_stage = (tide.get("cycle_stage") or {}).get("stage")
    macro_backdrop = tide.get("macro_backdrop") or {}
    sym_backdrop = (macro_backdrop.get("backdrop_by_symbol") or {}).get(sym, "unknown")

    if ablation["enable_tide"] and cycle_stage == "deleveraging" and sym in {"BTCUSDT", "NDX100USDT"}:
        if tech_bias == "bullish_bias":
            hard_gates.append("tide_deleveraging_long_block")

    # Collect causal driver votes deduplicated by causal_driver_group
    driver_votes: dict[str, tuple[int, str, str]] = {}  # group -> (vote, label, domain)

    def _register_vote(group: str, vote: int, desc: str, domain: str) -> None:
        cur = driver_votes.get(group)
        if cur is None or abs(vote) > abs(cur[0]):
            driver_votes[group] = (vote, desc, domain)

    if ablation["enable_tide"]:
        if macro_backdrop.get("state") == "unknown":
            gaps.append("tide_macro_backdrop_unknown")
        elif sym_backdrop == "tailwind":
            w = 2 if horizon == "swing_1_3d" else 1
            _register_vote("market_tide_layer1", +w, f"潮汐大环境对 {sym} 为顺风 ({macro_backdrop['state_cn']})", "tide")
        elif sym_backdrop == "headwind":
            w = 2 if horizon == "swing_1_3d" else 1
            _register_vote("market_tide_layer1", -w, f"潮汐大环境对 {sym} 为逆风 ({macro_backdrop['state_cn']})", "tide")

        pstate = pos.get("position_state")
        if pstate == "bounce_in_downtrend":
            _register_vote("market_tide_structure", -1, f"标的位置属于下降趋势中的逆势反弹 ({pos['structure_kind_cn']})", "tide")
        elif pstate == "pullback_in_uptrend" and horizon == "swing_1_3d":
            _register_vote("market_tide_structure", +1, f"标的位置属于上升趋势回踩 ({pos['structure_kind_cn']})", "tide")
        elif pstate == "potential_bullish_reversal" and horizon == "swing_1_3d":
            _register_vote("market_tide_structure", +1, f"存在向上突破反转候选 ({pos['structure_kind_cn']})", "tide")
        elif pstate == "potential_bearish_reversal" and horizon == "swing_1_3d":
            _register_vote("market_tide_structure", -1, f"存在向下破位反转候选 ({pos['structure_kind_cn']})", "tide")

    domain_enabled = {
        "macro": ablation["enable_macro"],
        "sentiment_flow": ablation["enable_sentiment_flow"],
        "news": ablation["enable_news"],
        "forecast": ablation["enable_forecast"],
    }

    for atom in atoms:
        if sym not in atom["applies_to_symbols"]:
            continue
        dom = atom["domain"]
        if not domain_enabled.get(dom, False):
            atom["role_by_symbol_horizon"][key] = "disabled_by_ablation"
            continue
        fresh = atom["freshness_status"]
        if fresh in {"future_excluded", "stale_excluded"}:
            atom["role_by_symbol_horizon"][key] = "excluded_stale_or_future"
            continue
        if fresh == "missing" and not atom["applies_to_horizons"]:
            atom["role_by_symbol_horizon"][key] = "missing_gap"
            gaps.append(f"{atom['evidence_id']}:{atom['factor_name']}")
            continue
        if horizon not in atom["applies_to_horizons"]:
            atom["role_by_symbol_horizon"][key] = "other_horizon_only"
            continue
        if fresh == "missing":
            atom["role_by_symbol_horizon"][key] = "missing_gap"
            gaps.append(f"{atom['evidence_id']}:{atom['factor_name']}")
            continue

        d = atom["direction_by_symbol"].get(sym, "unknown")
        # Determine weight by domain discipline (deduplicated by causal_driver_group!)
        weight = 0
        if dom == "macro" and horizon == "swing_1_3d" and not ablation["enable_tide"]:
            weight = 1 if d == "bullish" else (-1 if d == "bearish" else 0)
        elif dom == "macro" and atom["evidence_id"] == "macro:ice_dxy" and d in {"bullish", "bearish"}:
            weight = 1 if d == "bullish" else -1
        elif dom == "news" and (atom["value_summary"] or {}).get("news_kind") == "primary_release":
            age_h = _safe_f((atom["value_summary"] or {}).get("age_hours"))
            recent_h = float(cfg["macro_and_tide_thresholds"]["news_primary_recent_hours"])
            mag = 2 if (age_h is not None and age_h <= recent_h) else 1
            weight = mag if d == "bullish" else (-mag if d == "bearish" else 0)
        elif dom == "sentiment_flow":
            if atom["evidence_id"] in {"sentiment:btc:spot_etf_flows", "sentiment:ndx:tech_relative_strength", "sentiment:xau:gold_etf_flows", "sentiment:btc:onchain_exchange_netflow"}:
                weight = 1 if d == "bullish" else (-1 if d == "bearish" else 0)
            else:
                weight = 0  # VIX, COT, Polymarket, Fear&Greed, mempool cannot directly vote direction
        elif dom == "forecast":
            # Only verified aligned Chronos or 168h stat drift contributes (+1/-1)
            weight = 1 if d == "bullish" else (-1 if d == "bearish" else 0)

        if weight != 0:
            _register_vote(
                atom["causal_driver_group"],
                weight,
                f"[{atom['evidence_id']}] {atom['factor_name']} ({'偏多' if weight > 0 else '偏空'})",
                dom,
            )

    ext_score = sum(v[0] for v in driver_votes.values())
    conv_reduced = False

    # Conflict & Final Synthesis Rules
    if hard_gates:
        final_bias = "neutral_stand_aside"
        conviction = "stand_aside_gated"
    elif tech_vote == 0:
        if ext_score >= 2:
            final_bias = "bullish_bias"
            conviction = "moderate_external_alignment"
        elif ext_score <= -2:
            final_bias = "bearish_bias"
            conviction = "moderate_external_alignment"
        else:
            final_bias = "neutral_stand_aside"
            conviction = "insufficient_edge_or_neutral"
    else:
        same_sign = (tech_vote > 0 and ext_score >= 0) or (tech_vote < 0 and ext_score <= 0)
        if same_sign:
            final_bias = tech_bias
            conviction = "moderate_aligned"
        elif abs(ext_score) <= 1:
            final_bias = tech_bias
            conviction = "reduced_by_counterevidence"
            conv_reduced = True
        elif abs(ext_score) >= 3 and (ablation["enable_tide"] or ablation["enable_macro"]):
            # Strong aligned external tide + macro + primary news flips judgment against technicals
            final_bias = "bearish_bias" if ext_score < 0 else "bullish_bias"
            conviction = "reduced_by_technical_conflict"
            conv_reduced = True
        else:
            final_bias = "neutral_stand_aside"
            conviction = "stand_aside_macro_tech_conflict"

    m_pos = pos.get("metrics") or {}
    if horizon == "intraday" and m_pos.get("shock_volatility"):
        if pos.get("position_state") == "potential_bullish_reversal" and m_pos.get("above_h4_ema50") and final_bias == "bearish_bias":
            hard_gates.append("shock_upside_breakout_blocks_intraday_short")
            final_bias = "neutral_stand_aside"
            conviction = "stand_aside_gated"
        elif pos.get("position_state") == "potential_bearish_reversal" and m_pos.get("below_h4_ema50") and final_bias == "bullish_bias":
            hard_gates.append("shock_downside_breakdown_blocks_intraday_long")
            final_bias = "neutral_stand_aside"
            conviction = "stand_aside_gated"

    # Background risk modifiers (VIX elevated, extreme funding+OI, extreme F&G)
    if vix_summary.get("elevated_volatility_regime") and sym in {"NDX100USDT", "BTCUSDT"} and final_bias != "neutral_stand_aside":
        conviction = "reduced_by_volatility_backdrop"
        conv_reduced = True
        counter.append(f"[sentiment:ndx:vix_volatility_backdrop] VIX={vix_summary.get('value')} 处于高波动区间，降低把握并提示双边风险（非独立空头触发器）")

    # Populate semantic role_by_symbol_horizon + supporting/counterevidence lists cleanly
    final_dir_token = "bullish" if final_bias == "bullish_bias" else ("bearish" if final_bias == "bearish_bias" else "neutral")
    for atom in atoms:
        if sym not in atom["applies_to_symbols"] or horizon not in atom["applies_to_horizons"]:
            continue
        if atom["role_by_symbol_horizon"].get(key) in {"excluded_stale_or_future", "missing_gap", "disabled_by_ablation"}:
            continue
        if any(gate.startswith("imminent_tier1_macro_event") for gate in hard_gates) and atom["evidence_id"] == "macro:economic_events_calendar":
            atom["role_by_symbol_horizon"][key] = "triggered_veto"
            continue
        d = atom["direction_by_symbol"].get(sym, "unknown")
        if hard_gates and d != "neutral" and atom.get("is_hard_gate_candidate"):
            atom["role_by_symbol_horizon"][key] = "triggered_veto"
        elif final_bias != tech_bias and d in {"bullish", "bearish"} and (
            (final_bias == "bullish_bias" and d == "bullish")
            or (final_bias == "bearish_bias" and d == "bearish")
            or (final_bias == "neutral_stand_aside" and ((tech_bias == "bullish_bias" and d == "bearish") or (tech_bias == "bearish_bias" and d == "bullish")))
        ):
            atom["role_by_symbol_horizon"][key] = "changed_direction"
        elif conv_reduced and ((final_dir_token == "bullish" and d == "bearish") or (final_dir_token == "bearish" and d == "bullish") or d == "two_sided_risk"):
            atom["role_by_symbol_horizon"][key] = "reduced_conviction"
        elif final_dir_token in {"bullish", "bearish"} and d == final_dir_token:
            atom["role_by_symbol_horizon"][key] = "confirmed_direction"
        else:
            atom["role_by_symbol_horizon"][key] = "context_only"

        stamp = atom.get("published_at_utc") or atom.get("observed_at_utc") or atom.get("retrieved_at_utc")
        summary_line = f"[{atom['evidence_id']}] {atom['factor_name']} ({atom['source_name']}, {stamp}): dir={d}, role={atom['role_by_symbol_horizon'][key]}"
        if d in {"bullish", "bearish"} and d == final_dir_token:
            supporting.append(summary_line)
        elif d in {"bullish", "bearish", "two_sided_risk"} and (d != final_dir_token or final_bias == "neutral_stand_aside"):
            counter.append(summary_line)

    if ablation["enable_tide"] and sym_backdrop in {"tailwind", "headwind"}:
        tide_line = f"[tide:macro_backdrop:{sym}] {macro_backdrop['state_cn']} -> {sym} {sym_backdrop}"
        if (final_bias == "bullish_bias" and sym_backdrop == "tailwind") or (final_bias == "bearish_bias" and sym_backdrop == "headwind"):
            supporting.insert(0, tide_line)
        else:
            counter.insert(0, tide_line)

    changed_vs_tech = final_bias != tech_bias
    active_desc = "; ".join(v[1] for v in driver_votes.values()) or "无额外同向/反向有效外部驱动"
    if hard_gates:
        why_tech_diff = f"触发硬门槛阻断（{', '.join(hard_gates)}），将原纯技术面 {tech['label_cn']} 强制调整为 观望。"
    elif changed_vs_tech:
        why_tech_diff = f"纯技术面为 {tech['label_cn']}，叠加外部多维因子（{active_desc}）后按因果去重与仲裁规则改判为 {BIAS_CN[final_bias]}。"
    else:
        why_tech_diff = (
            f"综合结论与纯技术面相同（同为 {BIAS_CN[final_bias]}），原因："
            + (
                f"外部驱动（{active_desc}）与技术面一致确认方向，未触及冲突反转阈值。"
                if final_bias != "neutral_stand_aside"
                else f"外部因子净偏向不足以打破中性阈值（ext_score={ext_score}, {active_desc}），继续观望。"
            )
        )

    lvl = pos.get("invalidation_levels") or {}
    invalidation_list = []
    if lvl.get("h4_ema50") is not None:
        invalidation_list.append(f"价格失效参考：4H EMA50 ({lvl['h4_ema50']}) 或 48h Donchian ({lvl.get('donchian_48h_low')} ~ {lvl.get('donchian_48h_high')}) 被收盘跌破/突破。")
    invalidation_list.extend(macro_backdrop.get("flip_conditions") or [])
    if macro_summary.get("events"):
        ev0 = macro_summary["events"][0]
        invalidation_list.append(f"事件失效条件：即将到来的 `{ev0['title']}` ({ev0['schedule']['scheduled_at_utc']}) 若出现反向超预期公布，须即时重评。")

    flip_list = [
        f"改判门槛：若价格收于 1H EMA50 ({lvl.get('h1_ema50')}) 与 4H EMA50 ({lvl.get('h4_ema50')}) 同一侧且 ADX>=22，叠加宏观/新闻同向，将触发方向重估。",
    ]

    return {
        "symbol": sym,
        "horizon": horizon,
        "technical_only": tech,
        "final_bias": final_bias,
        "final_label_cn": BIAS_CN[final_bias],
        "conviction_note": conviction,
        "changed_vs_technical": changed_vs_tech,
        "why_changed_or_unchanged_vs_technical": why_tech_diff,
        "hard_gates_triggered": hard_gates,
        "external_net_score_dedup": ext_score,
        "active_causal_drivers": [{"group": g, "vote": v[0], "summary": v[1], "domain": v[2]} for g, v in driver_votes.items()],
        "supporting_evidence": supporting[:6] or [tech["rationale"]],
        "strongest_counterevidence": counter[:5] or ["当前未发现足以直接否决结论的已核验强反证（仍受未校准概率与缺失数据限制）"],
        "invalidation_conditions": invalidation_list,
        "flip_conditions": flip_list,
        "data_gaps": sorted(set(gaps)),
    }


def synthesize_judgment_packet(
    inputs_bundle: dict[str, Any],
    *,
    ablation_switches: dict[str, bool] | None = None,
    include_v2_control: bool = True,
    starting_cash_usdt: float | None = None,
    replay_mode: bool | None = None,
) -> dict[str, Any]:
    """Synthesize a completely traceable, bit-for-bit reproducible multi-horizon JudgmentPacket."""
    cfg = load_judgment_config()
    sw = {
        "enable_macro": True,
        "enable_tide": True,
        "enable_sentiment_flow": True,
        "enable_news": True,
        "enable_forecast": True,
    }
    if ablation_switches:
        for k, v in ablation_switches.items():
            if k in sw:
                sw[k] = bool(v)

    is_replay = bool(inputs_bundle.get("replay_mode")) if replay_mode is None else bool(replay_mode)
    cash_ref = float(
        starting_cash_usdt
        if starting_cash_usdt is not None
        else (inputs_bundle.get("starting_cash_usdt") or cfg["shadow_paper"]["starting_cash_usdt"])
    )

    bar_cutoff_dt = parse_aware_timestamp(inputs_bundle.get("bar_cutoff_utc") or inputs_bundle.get("cutoff_utc")) or datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    decision_cutoff_dt = parse_aware_timestamp(inputs_bundle.get("decision_cutoff_utc") or inputs_bundle.get("cutoff_utc")) or bar_cutoff_dt
    bar_cutoff_iso = bar_cutoff_dt.isoformat()
    mkt_snap = inputs_bundle["market_snapshot"]

    coherence = validate_market_snapshot_coherence(mkt_snap, cutoff_utc=bar_cutoff_dt)
    fcfg = cfg["freshness_limits"]
    mcfg = cfg["macro_and_tide_thresholds"]

    ab_alfred = dict(inputs_bundle.get("alfred_series") or {})
    if not sw["enable_macro"]:
        ab_alfred = {k: v for k, v in ab_alfred.items() if k in {"VIXCLS", "NASDAQ100", "SP500"}}
    if not sw["enable_sentiment_flow"]:
        ab_alfred = {k: v for k, v in ab_alfred.items() if k not in {"VIXCLS", "NASDAQ100", "SP500"}}

    verified_news = classify_and_filter_news_items(
        inputs_bundle.get("raw_news_items") or [],
        cutoff_utc=decision_cutoff_dt,
        intraday_max_hours=float(fcfg["news_intraday_max_age_hours"]),
        swing_max_hours=float(fcfg["news_swing_max_age_hours"]),
        duplicate_jaccard_min=float(mcfg["news_duplicate_jaccard_min"]),
        replay_mode=is_replay,
    )
    active_news_for_macro = verified_news if sw["enable_news"] else []

    sent_atoms, vix_summary, sent_summary = build_sentiment_flow_evidence(
        market_snapshot=mkt_snap,
        alfred_series=ab_alfred,
        fear_greed_ctx=inputs_bundle.get("fear_greed") if sw["enable_sentiment_flow"] else None,
        etf_ctx=inputs_bundle.get("etf_flows") if sw["enable_sentiment_flow"] else None,
        gold_cot_ctx=inputs_bundle.get("gold_cot") if sw["enable_sentiment_flow"] else None,
        onchain_ctx=inputs_bundle.get("onchain") if sw["enable_sentiment_flow"] else None,
        exchange_netflow_ctx=inputs_bundle.get("exchange_netflow") if sw["enable_sentiment_flow"] else None,
        gold_etf_ctx=inputs_bundle.get("gold_etf_flows") if sw["enable_sentiment_flow"] else None,
        ndx_edgar_filings=(inputs_bundle.get("ndx_edgar_filings") or []) if sw["enable_sentiment_flow"] else [],
        polymarket_ctx=inputs_bundle.get("polymarket") if sw["enable_sentiment_flow"] else None,
        cutoff_utc=decision_cutoff_dt,
        cfg=cfg,
        replay_mode=is_replay,
    )
    macro_atoms, macro_summary = build_macro_evidence(
        alfred_series=ab_alfred,
        dxy_live=inputs_bundle.get("dxy_live") if sw["enable_macro"] else None,
        event_calendar=inputs_bundle.get("event_calendar") if sw["enable_macro"] else None,
        verified_news=active_news_for_macro,
        vix_summary=vix_summary,
        cutoff_utc=decision_cutoff_dt,
        cfg=cfg,
    )
    tide, prepared_1h = compute_market_tide(
        frames_1h=coherence["frames_1h"],
        frames_1d=coherence["frames_1d"],
        market_snapshot=mkt_snap,
        macro_summary=macro_summary,
        vix_summary=vix_summary,
        sent_summary=sent_summary,
        cfg=cfg,
    )
    fc_atoms, forecast_summary = build_forecast_and_model_evidence(
        frames_1h=coherence["frames_1h"],
        chronos_payload=inputs_bundle.get("chronos_forecast"),
        llm_opinion=inputs_bundle.get("llm_opinion"),
        cutoff_utc=bar_cutoff_dt,
        cfg=cfg,
    )
    news_atoms = _news_items_to_atoms(verified_news)
    all_atoms = [*macro_atoms, *sent_atoms, *news_atoms, *fc_atoms]

    supp_raw = (
        inputs_bundle.get("agent_or_human_supplement")
        or inputs_bundle.get("agent_supplement")
        or inputs_bundle.get("human_supplement")
    )
    if isinstance(supp_raw, dict) and supp_raw.get("note"):
        supp_block = {
            "provided": True,
            "actor": str(supp_raw.get("actor") or "agent_or_human"),
            "note": str(supp_raw["note"]),
            "recorded_at_utc": str(supp_raw.get("recorded_at_utc") or decision_cutoff_dt.isoformat()),
            "can_override_risk_gate": False,
        }
    elif isinstance(supp_raw, str) and supp_raw.strip():
        supp_block = {
            "provided": True,
            "actor": "agent_or_human",
            "note": supp_raw.strip(),
            "recorded_at_utc": decision_cutoff_dt.isoformat(),
            "can_override_risk_gate": False,
        }
    else:
        supp_block = {"provided": False, "actor": None, "note": None, "recorded_at_utc": None, "can_override_risk_gate": False}

    judgments: dict[str, Any] = {}
    for sym in SYMBOLS:
        pos = tide["instrument_positions"][sym]
        if not coherence["aligned"]:
            forced = {
                "symbol": sym,
                "technical_only": {"bias": "neutral_stand_aside", "label_cn": "观望", "rationale": "合约收盘时间戳未对齐或包含未来K线，禁止交叉时间点混合判断"},
                "final_bias": "neutral_stand_aside",
                "final_label_cn": "观望",
                "conviction_note": "stand_aside_alignment_gate",
                "changed_vs_technical": False,
                "why_changed_or_unchanged_vs_technical": f"三合约行情对齐门控未通过（{', '.join(coherence['reasons'])}），全部观望。",
                "hard_gates_triggered": list(coherence["reasons"]),
                "external_net_score_dedup": 0,
                "active_causal_drivers": [],
                "supporting_evidence": [],
                "strongest_counterevidence": list(coherence["reasons"]),
                "invalidation_conditions": ["等待 BTCUSDT / XAUUSDT / NDX100USDT 完整闭合 1H K 线同步到同一整点"],
                "flip_conditions": [],
                "data_gaps": list(coherence["reasons"]),
            }
            sym_horizons = {"intraday": dict(forced, horizon="intraday"), "swing_1_3d": dict(forced, horizon="swing_1_3d")}
        else:
            sym_horizons = {
                h: _synthesize_one(
                    sym, h, pos=pos, tide=tide, macro_summary=macro_summary, vix_summary=vix_summary,
                    atoms=all_atoms, ablation=sw, cfg=cfg,
                )
                for h in HORIZONS
            }

        v2_ctrl = (
            _official_v2_signal_for_symbol(
                sym,
                prep_df=prepared_1h.get(sym),
                market_snapshot=mkt_snap,
                books_snapshot=inputs_bundle.get("books_snapshot") or {},
                tiers_snapshot=inputs_bundle.get("tiers_snapshot") or {},
                alfred_series=inputs_bundle.get("alfred_series") or {},
                macro_summary=macro_summary,
                sent_summary=sent_summary,
                cutoff_utc=bar_cutoff_dt,
                chronos_payload=inputs_bundle.get("chronos_forecast"),
                starting_cash_usdt=cash_ref,
            )
            if (include_v2_control and coherence["aligned"])
            else {
                "official_v2_2_1_signal": "stand_aside",
                "stand_aside_reason": "input_alignment_rejected" if not coherence["aligned"] else "v2_control_skipped",
                "setup_type": "none",
                "disagreements": [],
                "paper_eligibility": {"eligible": False, "status": "blocked", "label_cn": "否", "review_required": False, "review_triggers": [], "side": None, "blocking_reasons": list(coherence["reasons"])},
            }
        )
        v3_pe, review_triggers = _apply_v3_paper_review_gate(
            sym,
            v2_ctrl=v2_ctrl,
            pos=pos,
            atoms=all_atoms,
            news_enabled=sw["enable_news"],
        ) if coherence["aligned"] else (v2_ctrl["paper_eligibility"], [])

        changed_evidence_items: list[dict[str, Any]] = []
        for h in HORIZONS:
            hd = sym_horizons[h]
            if hd["changed_vs_technical"] or hd["hard_gates_triggered"]:
                for g in hd["hard_gates_triggered"]:
                    changed_evidence_items.append({"horizon": h, "kind": "hard_gate", "detail": g})
                for drv in hd["active_causal_drivers"]:
                    changed_evidence_items.append({"horizon": h, "kind": "causal_driver", "domain": drv["domain"], "vote": drv["vote"], "detail": drv["summary"]})
        for rt in review_triggers:
            changed_evidence_items.append({"horizon": "paper_execution", "kind": "review_gate", "detail": f"{rt['trigger']}: {rt['detail']}"})

        judgments[sym] = {
            "symbol": sym,
            "last_close": pos.get("close"),
            "intraday": sym_horizons["intraday"],
            "swing_1_3d": sym_horizons["swing_1_3d"],
            "evidence_that_changed_conclusion": changed_evidence_items,
            "official_v2_2_1_signal": v2_ctrl["official_v2_2_1_signal"],
            "official_v2_2_1_setup_type": v2_ctrl.get("setup_type"),
            "official_v2_2_1_reason": v2_ctrl["stand_aside_reason"],
            "official_v2_2_1_disagreements": v2_ctrl["disagreements"],
            "official_v2_2_1_paper_eligibility": v2_ctrl["paper_eligibility"],
            "paper_eligibility": v3_pe,
            "agent_or_human_supplement": supp_block,
        }

    factors_used = [a["evidence_id"] for a in all_atoms if a["freshness_status"] in {"fresh", "lagged_usable"}]
    factors_missing = [a["evidence_id"] for a in all_atoms if a["freshness_status"] == "missing"]
    factors_excluded = [
        {"evidence_id": a["evidence_id"], "status": a["freshness_status"]}
        for a in all_atoms
        if a["freshness_status"] in {"future_excluded", "stale_excluded"}
    ]

    valid_pub_dts = [
        parse_aware_timestamp(n.get("published_at_utc"))
        for n in verified_news
        if n.get("verification_status") == "verified" and n.get("freshness_status") in {"fresh", "lagged_usable"}
    ]
    latest_ann_iso = max((dt for dt in valid_pub_dts if dt is not None), default=None)
    first_ret_iso = str(inputs_bundle.get("first_retrieved_at_utc") or inputs_bundle.get("collected_at_utc") or decision_cutoff_dt.isoformat())
    gen_dt = parse_aware_timestamp(inputs_bundle.get("judgment_generated_at_utc")) or parse_aware_timestamp(inputs_bundle.get("collected_at_utc")) or decision_cutoff_dt
    earliest_exec_dt = compute_earliest_executable_utc(bar_cutoff_dt, gen_dt)

    factor_ledger = _build_factor_usage_ledger(
        atoms=all_atoms,
        tide=tide,
        forecast_summary=forecast_summary,
        judgments=judgments,
        ablation=sw,
    )

    lifecycle = {
        "bar_cutoff_utc": bar_cutoff_iso,
        "decision_cutoff_utc": decision_cutoff_dt.isoformat(),
        "latest_announcement_published_at_utc": latest_ann_iso.isoformat() if latest_ann_iso else None,
        "first_retrieved_at_utc": first_ret_iso,
        "judgment_generated_at_utc": gen_dt.isoformat(),
        "earliest_executable_at_utc": earliest_exec_dt.isoformat(),
    }

    packet = {
        "protocol_id": cfg["protocol_id"],
        "version": cfg["version"],
        "baseline_control_version": cfg["baseline_control_version"],
        "starting_cash_usdt": cash_ref,
        "replay_mode": is_replay,
        "is_synthetic_demo": bool(inputs_bundle.get("is_synthetic_demo")),
        "cutoff_utc": bar_cutoff_iso,
        "collected_at_utc": inputs_bundle.get("collected_at_utc") or bar_cutoff_iso,
        "lifecycle_timestamps": lifecycle,
        "coherence": {
            "aligned": coherence["aligned"],
            "reasons": coherence["reasons"],
            "last_close_by_symbol": coherence["last_close_by_symbol"],
            "sanitization_audit": coherence["sanitization_audit"],
        },
        "ablation_switches": sw,
        "macro_summary": macro_summary,
        "sentiment_summary": sent_summary,
        "tide": tide,
        "forecast_and_models": forecast_summary,
        "news_audit": verified_news,
        "evidence_atoms": all_atoms,
        "factor_usage_ledger": factor_ledger,
        "judgments": judgments,
        "agent_or_human_supplement": supp_block,
        "audit_coverage": {
            "factors_used": factors_used,
            "factors_missing": factors_missing,
            "factors_excluded": factors_excluded,
            "collection_errors": inputs_bundle.get("collection_errors") or {},
        },
        "order_permissions": False,
        "calibrated_probabilities": False,
        "validation_status": "PENDING_INSUFFICIENT_PROSPECTIVE_SAMPLE",
    }
    packet["judgment_hash"] = canonical_sha256(
        {
            "cutoff_utc": packet["cutoff_utc"],
            "starting_cash_usdt": packet["starting_cash_usdt"],
            "earliest_executable_at_utc": lifecycle["earliest_executable_at_utc"],
            "coherence": packet["coherence"],
            "ablation_switches": packet["ablation_switches"],
            "judgments": packet["judgments"],
            "tide": packet["tide"],
            "evidence_atoms": packet["evidence_atoms"],
        }
    )
    return packet
