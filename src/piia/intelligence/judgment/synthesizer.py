"""Deterministic, traceable multi-horizon judgment synthesis engine (PIIA Judgment v3).

Combines verified evidence across 5 external domains + causal market bars into per-symbol
`intraday` and `swing_1_3d` judgments, keeps pure technical direction alongside final judgment,
deduplicates collinear drivers by `causal_driver_group`, distinguishes hard gates from background
modifiers, and displays frozen PIIA v2.2.1 formal signals & paper eligibility separately.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any

from piia.intelligence.agent_v2.pillars import (
    SYMBOLS,
    build_bar_evidence_for_symbol,
    canonical_sha256,
    load_v2_config,
    point_in_time_macro_context,
)
from piia.intelligence.agent_v2.risk_gate import run_symbol_decision_trace
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
        if adx14 < float(tcfg["adx_range_max"]) and rsi14 is not None and bb_lo is not None and bb_up is not None:
            if close_px < bb_lo and rsi14 < float(tcfg["rsi_oversold"]) and h4_dir == "up":
                return {"bias": "bullish_bias", "label_cn": BIAS_CN["bullish_bias"], "rationale": f"震荡下轨超卖反弹 (RSI14={rsi14:.1f}<35 且 4H 向上)"}
            if close_px > bb_up and rsi14 > float(tcfg["rsi_overbought"]) and h4_dir == "down":
                return {"bias": "bearish_bias", "label_cn": BIAS_CN["bearish_bias"], "rationale": f"震荡上轨超买回落 (RSI14={rsi14:.1f}>65 且 4H 向下)"}
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
) -> dict[str, Any]:
    """Evaluate frozen PIIA v2.2.1 signal & paper eligibility read-only (500 USDT control)."""
    v2_cfg = load_v2_config()
    if prep_df is None or len(prep_df) < 2:
        return {
            "official_v2_2_1_signal": "stand_aside",
            "stand_aside_reason": "insufficient_1h_bars",
            "setup_type": "none",
            "disagreements": [],
            "paper_eligibility": {"eligible": False, "side": None, "blocking_reasons": ["insufficient_1h_bars"], "sizing_preview": None},
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
        equity_usdt=500.0,
        free_cash_usdt=500.0,
        open_positions_count=0,
        kill_switch_halted=False,
        cfg=v2_cfg,
    )
    rk = trace["programmatic_risk_gate"]
    final_act = rk["final_action"]
    blockers = []
    if final_act == "stand_aside" and rk.get("stand_aside_reason"):
        blockers.append(str(rk["stand_aside_reason"]))
    return {
        "official_v2_2_1_signal": final_act,
        "stand_aside_reason": rk.get("stand_aside_reason"),
        "setup_type": trace["synthesizer"].get("setup_type"),
        "disagreements": trace["synthesizer"].get("disagreements") or [],
        "paper_eligibility": {
            "eligible": bool(final_act in {"long", "short"} and rk.get("approved_by_risk_gate")),
            "side": final_act if final_act in {"long", "short"} else None,
            "blocking_reasons": blockers,
            "entry_plan": rk.get("entry_plan"),
            "exit_plan": rk.get("exit_plan"),
            "position_sizing": rk.get("position_sizing"),
            "cost_and_risk": rk.get("cost_and_risk"),
        },
    }


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

    cutoff_dt = parse_aware_timestamp(inputs_bundle.get("cutoff_utc")) or datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    cutoff_iso = cutoff_dt.isoformat()
    mkt_snap = inputs_bundle["market_snapshot"]

    coherence = validate_market_snapshot_coherence(mkt_snap, cutoff_utc=cutoff_dt)
    fcfg = cfg["freshness_limits"]
    mcfg = cfg["macro_and_tide_thresholds"]

    ab_alfred = dict(inputs_bundle.get("alfred_series") or {})
    if not sw["enable_macro"]:
        ab_alfred = {k: v for k, v in ab_alfred.items() if k in {"VIXCLS", "NASDAQ100", "SP500"}}
    if not sw["enable_sentiment_flow"]:
        ab_alfred = {k: v for k, v in ab_alfred.items() if k not in {"VIXCLS", "NASDAQ100", "SP500"}}

    verified_news = classify_and_filter_news_items(
        inputs_bundle.get("raw_news_items") or [],
        cutoff_utc=cutoff_dt,
        intraday_max_hours=float(fcfg["news_intraday_max_age_hours"]),
        swing_max_hours=float(fcfg["news_swing_max_age_hours"]),
        duplicate_jaccard_min=float(mcfg["news_duplicate_jaccard_min"]),
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
        cutoff_utc=cutoff_dt,
        cfg=cfg,
    )
    macro_atoms, macro_summary = build_macro_evidence(
        alfred_series=ab_alfred,
        dxy_live=inputs_bundle.get("dxy_live") if sw["enable_macro"] else None,
        event_calendar=inputs_bundle.get("event_calendar") if sw["enable_macro"] else None,
        verified_news=active_news_for_macro,
        vix_summary=vix_summary,
        cutoff_utc=cutoff_dt,
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
        cutoff_utc=cutoff_dt,
        cfg=cfg,
    )
    news_atoms = _news_items_to_atoms(verified_news)
    all_atoms = [*macro_atoms, *sent_atoms, *news_atoms, *fc_atoms]

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
                cutoff_utc=cutoff_dt,
                chronos_payload=inputs_bundle.get("chronos_forecast"),
            )
            if (include_v2_control and coherence["aligned"])
            else {
                "official_v2_2_1_signal": "stand_aside",
                "stand_aside_reason": "input_alignment_rejected" if not coherence["aligned"] else "v2_control_skipped",
                "setup_type": "none",
                "disagreements": [],
                "paper_eligibility": {"eligible": False, "side": None, "blocking_reasons": list(coherence["reasons"])},
            }
        )

        judgments[sym] = {
            "symbol": sym,
            "last_close": pos.get("close"),
            "intraday": sym_horizons["intraday"],
            "swing_1_3d": sym_horizons["swing_1_3d"],
            "official_v2_2_1_signal": v2_ctrl["official_v2_2_1_signal"],
            "official_v2_2_1_reason": v2_ctrl["stand_aside_reason"],
            "official_v2_2_1_disagreements": v2_ctrl["disagreements"],
            "paper_eligibility": v2_ctrl["paper_eligibility"],
        }

    factors_used = [a["evidence_id"] for a in all_atoms if a["freshness_status"] in {"fresh", "lagged_usable"}]
    factors_missing = [a["evidence_id"] for a in all_atoms if a["freshness_status"] == "missing"]
    factors_excluded = [
        {"evidence_id": a["evidence_id"], "status": a["freshness_status"]}
        for a in all_atoms
        if a["freshness_status"] in {"future_excluded", "stale_excluded"}
    ]

    packet = {
        "protocol_id": cfg["protocol_id"],
        "version": cfg["version"],
        "baseline_control_version": cfg["baseline_control_version"],
        "is_synthetic_demo": bool(inputs_bundle.get("is_synthetic_demo")),
        "cutoff_utc": cutoff_iso,
        "collected_at_utc": inputs_bundle.get("collected_at_utc") or cutoff_iso,
        "coherence": {
            "aligned": coherence["aligned"],
            "reasons": coherence["reasons"],
            "last_close_by_symbol": coherence["last_close_by_symbol"],
        },
        "ablation_switches": sw,
        "macro_summary": macro_summary,
        "sentiment_summary": sent_summary,
        "tide": tide,
        "forecast_and_models": forecast_summary,
        "news_audit": verified_news,
        "evidence_atoms": all_atoms,
        "judgments": judgments,
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
            "coherence": packet["coherence"],
            "ablation_switches": packet["ablation_switches"],
            "judgments": packet["judgments"],
            "tide": packet["tide"],
            "evidence_atoms": packet["evidence_atoms"],
        }
    )
    return packet
