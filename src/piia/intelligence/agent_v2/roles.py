"""Isolated Researcher, Independent Critic, and Synthesizer Roles (PIIA v2).

Context-isolation invariant:
- `run_independent_researcher(bundle)` receives ONLY `bundle` (no Critic fields).
- `run_independent_critic_pass1(bundle)` receives ONLY `bundle` (no Researcher fields).
- `run_critic_adversarial_veto(bundle, researcher_out)` applies pre-frozen deterministic
  veto checks against the Researcher's proposal without altering Pass-1 blind findings.
- `run_synthesizer(bundle, researcher_out, critic_pass1, critic_veto)` retains ALL
  disagreements verbatim, never majority-votes, and outputs a traceable
  `long` | `short` | `stand_aside` decision candidate for the Programmatic Risk Gate.

Module ablation flags (`enable_macro`, `enable_forecast`, `enable_critic_veto`,
`enable_live_external`) allow exact apples-to-apples ablation studies without
changing the underlying codebase.
"""
from __future__ import annotations

import math
from typing import Any

from piia.intelligence.agent_v2.pillars import canonical_sha256, load_v2_config


def run_independent_researcher(
    bundle: dict[str, Any],
    *,
    cfg: dict[str, Any] | None = None,
    enable_macro: bool = True,
    enable_stat_forecast: bool | None = None,
    enable_neural_forecast: bool = False,
    enable_forecast: bool | None = None,
    enable_live_external: bool = True,
) -> dict[str, Any]:
    """Form an isolated directional hypothesis from frozen evidence ONLY."""
    if any(k in bundle for k in ("critic_pass1", "critic_veto", "synthesizer")):
        raise ValueError("Context isolation violation: researcher received critic/synthesizer state")
    cfg = cfg or load_v2_config()
    mcfg = cfg["evidence_pillars"]["market"]
    rcfg = cfg["roles_and_gates"]
    use_stat_forecast = (
        enable_stat_forecast
        if enable_stat_forecast is not None
        else (True if enable_forecast is None else bool(enable_forecast))
    )
    use_neural_forecast = bool(enable_neural_forecast)

    pillars = bundle["pillars"]
    mkt = pillars["market"]
    ohlc = mkt["ohlc"]
    close_px = float(ohlc["close"])
    open_px = float(ohlc["open"])
    high_px = float(ohlc["high"])
    low_px = float(ohlc["low"])
    prev_close = mkt.get("prev_close")

    adx14 = mkt.get("adx14")
    pdi = mkt.get("plus_di14")
    mdi = mkt.get("minus_di14")
    atr14 = mkt.get("atr14")
    atr_mean = mkt.get("atr14_mean50")
    ema20 = mkt.get("ema20_1h")
    ema50 = mkt.get("ema50_1h")
    htf20 = mkt.get("htf_ema20_4h")
    htf50 = mkt.get("htf_ema50_4h")
    rsi14 = mkt.get("rsi14")
    bb_lo = mkt.get("bb_lower")
    bb_up = mkt.get("bb_upper")
    don_hi = mkt.get("donchian_high")
    don_lo = mkt.get("donchian_low")

    citations = [mkt["evidence_hash"]]
    supporting_notes: list[str] = []
    missing_notes: list[str] = []

    required = (adx14, pdi, mdi, atr14, atr_mean, ema20, ema50, htf20, htf50)
    if any(v is None or not math.isfinite(float(v)) for v in required) or float(atr_mean) <= 0 or float(atr14) <= 0:
        return {
            "role": "researcher",
            "context_isolated": True,
            "symbol": bundle["symbol"],
            "regime": "insufficient_data",
            "direction": "stand_aside",
            "setup_type": "none",
            "alignment_score": 0.0,
            "evidence_citations": citations,
            "supporting_notes": [],
            "missing_data": ["market_indicator_warmup_incomplete"],
            "proposed_stop_atr_mult": None,
            "proposed_exit_rule": None,
            "invalidation": "Await complete closed-bar indicator warmup.",
        }

    atr_ratio = float(atr14) / float(atr_mean)
    htf_dir = 1 if float(htf20) > float(htf50) else (-1 if float(htf20) < float(htf50) else 0)
    h1_dir = 1 if float(ema20) > float(ema50) else (-1 if float(ema20) < float(ema50) else 0)

    if atr_ratio >= float(mcfg["extreme_vol_atr_ratio"]):
        regime = "extreme_vol"
    elif float(adx14) >= float(mcfg["adx_trend_min"]) and htf_dir == 1 and h1_dir == 1 and float(pdi) > float(mdi):
        regime = "trend_up"
    elif float(adx14) >= float(mcfg["adx_trend_min"]) and htf_dir == -1 and h1_dir == -1 and float(mdi) > float(pdi):
        regime = "trend_down"
    elif float(adx14) < float(mcfg["adx_range_max"]):
        regime = "range"
    else:
        regime = "uncertain"

    direction = "stand_aside"
    setup_type = "none"
    score = 0.0
    stop_mult = None
    exit_rule = None
    invalidation = "No actionable setup on closed bar."

    if regime == "trend_up":
        brk = (don_hi is not None and prev_close is not None and close_px > float(don_hi) and float(prev_close) <= float(don_hi))
        pb = (
            float(adx14) >= float(mcfg["adx_pullback_min"])
            and low_px <= float(ema20)
            and close_px > float(ema20)
            and close_px > open_px
            and rsi14 is not None
            and float(rsi14) <= 56.0
        )
        if brk or pb:
            direction = "long"
            setup_type = "trend_breakout" if brk else "trend_pullback"
            score = float(adx14) / 25.0 + (0.35 if brk else 0.15)
            stop_mult = float(rcfg["trend_stop_atr_mult"])
            exit_rule = f"chandelier_22h_2.5atr_or_1h4h_flip_or_{int(rcfg['trend_max_hold_hours'])}h"
            invalidation = f"Price falls {stop_mult}x ATR14 below entry or 1h+4h EMAs flip bearish."
            supporting_notes.append(f"1h/4h uptrend aligned (ADX14={float(adx14):.1f}, +DI>{float(mdi):.1f}) via {setup_type}.")

    elif regime == "trend_down":
        brk = (don_lo is not None and prev_close is not None and close_px < float(don_lo) and float(prev_close) >= float(don_lo))
        pb = (
            float(adx14) >= float(mcfg["adx_pullback_min"])
            and high_px >= float(ema20)
            and close_px < float(ema20)
            and close_px < open_px
            and rsi14 is not None
            and float(rsi14) >= 44.0
        )
        if brk or pb:
            direction = "short"
            setup_type = "trend_breakout" if brk else "trend_pullback"
            score = float(adx14) / 25.0 + (0.35 if brk else 0.15)
            stop_mult = float(rcfg["trend_stop_atr_mult"])
            exit_rule = f"chandelier_22h_2.5atr_or_1h4h_flip_or_{int(rcfg['trend_max_hold_hours'])}h"
            invalidation = f"Price rises {stop_mult}x ATR14 above entry or 1h+4h EMAs flip bullish."
            supporting_notes.append(f"1h/4h downtrend aligned (ADX14={float(adx14):.1f}, -DI>{float(pdi):.1f}) via {setup_type}.")

    elif regime == "range" and atr_ratio <= float(mcfg["range_max_atr_ratio"]):
        if rsi14 is not None and bb_lo is not None and bb_up is not None:
            if close_px < float(bb_lo) and float(rsi14) < float(mcfg["rsi_oversold"]) and htf_dir == 1:
                direction = "long"
                setup_type = "range_reversion"
                score = (float(mcfg["rsi_oversold"]) - float(rsi14)) / 15.0 + 0.50
                stop_mult = float(rcfg["range_stop_atr_mult"])
                exit_rule = f"bb_sma20_mid_or_{int(rcfg['range_max_hold_hours'])}h"
                invalidation = f"Price drops {stop_mult}x ATR14 below entry after lower-band excursion."
                supporting_notes.append(f"Low-ADX range oversold excursion (RSI14={float(rsi14):.1f}<{mcfg['rsi_oversold']}) with 4h upward slope guard.")
            elif close_px > float(bb_up) and float(rsi14) > float(mcfg["rsi_overbought"]) and htf_dir == -1:
                direction = "short"
                setup_type = "range_reversion"
                score = (float(rsi14) - float(mcfg["rsi_overbought"])) / 15.0 + 0.50
                stop_mult = float(rcfg["range_stop_atr_mult"])
                exit_rule = f"bb_sma20_mid_or_{int(rcfg['range_max_hold_hours'])}h"
                invalidation = f"Price rises {stop_mult}x ATR14 above entry after upper-band excursion."
                supporting_notes.append(f"Low-ADX range overbought excursion (RSI14={float(rsi14):.1f}>{mcfg['rsi_overbought']}) with 4h downward slope guard.")

    # Incorporate non-market pillars into Researcher evidence context (without peeking at Critic)
    if enable_macro:
        mac = pillars["macro"]
        if mac.get("status") == "available":
            citations.append(mac["evidence_hash"])
            if direction == "long" and mac.get("macro_easing_tailwind"):
                score += 0.20
                supporting_notes.append("Point-in-time ALFRED real yield & broad dollar easing tailwind.")
            elif direction == "short" and mac.get("macro_tightening_headwind"):
                score += 0.20
                supporting_notes.append("Point-in-time ALFRED real yield & broad dollar tightening supports short bias.")
        else:
            missing_notes.append("macro_context_unavailable")

    if use_stat_forecast or use_neural_forecast:
        fc = pillars["forecast"]
        citations.append(fc["evidence_hash"])
        neural = fc.get("neural_chronos") or {}
        stat = fc.get("statistical_drift") or {}
        if use_neural_forecast and neural.get("available") and direction in {"long", "short"}:
            n_est = neural.get("q50_24h")
            if n_est is not None and (
                (direction == "long" and float(n_est) > close_px)
                or (direction == "short" and float(n_est) < close_px)
            ):
                score += 0.15
                supporting_notes.append("Neural Chronos 24h q50 aligns with proposed direction.")
        elif use_stat_forecast:
            s_est = stat.get("point_24h") if stat.get("available") else None
            if s_est is not None and direction in {"long", "short"}:
                if (direction == "long" and float(s_est) > close_px) or (direction == "short" and float(s_est) < close_px):
                    score += 0.15
                    supporting_notes.append("24h uncalibrated statistical drift benchmark aligns with proposed direction.")
            elif not stat.get("available"):
                missing_notes.append("statistical_forecast_history_insufficient")

    if enable_live_external:
        for pname in ("flows_positioning", "onchain", "polymarket", "event_calendar"):
            p_obj = pillars.get(pname) or {}
            st_val = p_obj.get("status")
            if st_val in {"available", "collected_partial_or_complete"}:
                citations.append(p_obj["evidence_hash"])
            elif st_val != "not_applicable":
                missing_notes.append(f"{pname}:{st_val}")

    out = {
        "role": "researcher",
        "context_isolated": True,
        "symbol": bundle["symbol"],
        "regime": regime,
        "direction": direction,
        "setup_type": setup_type,
        "alignment_score": round(score, 4),
        "evidence_citations": list(dict.fromkeys(citations)),
        "supporting_notes": supporting_notes,
        "missing_data": missing_notes,
        "proposed_stop_atr_mult": stop_mult,
        "proposed_exit_rule": exit_rule,
        "invalidation": invalidation,
    }
    out["output_hash"] = canonical_sha256(out)
    return out


def run_independent_critic_pass1(
    bundle: dict[str, Any],
    *,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Blind Pass-1 Critic: inspects ONLY frozen evidence, ZERO access to Researcher output."""
    if "researcher" in bundle or "synthesizer" in bundle:
        raise ValueError("Context isolation violation: blind critic received researcher/synthesizer output")
    cfg = cfg or load_v2_config()
    mcfg = cfg["evidence_pillars"]["market"]
    rcfg = cfg["roles_and_gates"]

    pillars = bundle["pillars"]
    mkt = pillars["market"]
    mac = pillars["macro"]
    fc = pillars["forecast"]
    flows = pillars["flows_positioning"]

    risk_flags: list[str] = []
    missing_gaps: list[str] = []
    citations = [mkt["evidence_hash"]]

    atr14 = mkt.get("atr14")
    atr_mean = mkt.get("atr14_mean50")
    close_px = float(mkt["ohlc"]["close"])
    if atr14 is not None and atr_mean is not None and float(atr_mean) > 0:
        if float(atr14) / float(atr_mean) >= float(mcfg["extreme_vol_atr_ratio"]):
            risk_flags.append("extreme_volatility_regime")
        # Friction share of 1R check (using 2*ATR14 reference)
        round_trip_rate = 2.0 * (float(cfg["risk_and_costs"]["taker_fee_fallback"]) + float(cfg["risk_and_costs"]["slippage_each_side_base"]))
        one_r_pct = (2.0 * float(atr14)) / close_px if close_px > 0 else 0.0
        if one_r_pct > 0 and (round_trip_rate / one_r_pct) > float(rcfg["critic_max_friction_share_of_1r"]):
            risk_flags.append("high_friction_to_atr_ratio")

    f_rate = mkt.get("funding_rate_8h")
    if f_rate is None:
        missing_gaps.append("funding_rate_8h_missing")
    elif abs(float(f_rate)) > float(rcfg["critic_max_adverse_funding_8h"]):
        risk_flags.append(f"elevated_funding_crowding:{float(f_rate):+.6f}")

    if mac.get("status") == "available":
        citations.append(mac["evidence_hash"])
        if mac.get("macro_tightening_headwind"):
            risk_flags.append("macro_real_yield_and_dollar_tightening_headwind")
    else:
        missing_gaps.append("macro_unavailable")

    if flows.get("liquidity_state") == "deleveraging":
        citations.append(flows["evidence_hash"])
        risk_flags.append("liquidity_state_deleveraging")

    for pname in ("flows_positioning", "onchain", "polymarket", "event_calendar"):
        st_v = (pillars.get(pname) or {}).get("status")
        if st_v in {"not_available_point_in_time", "unavailable", "unavailable_or_empty", "incomplete_or_stale"}:
            missing_gaps.append(f"{pname}:{st_v}")

    stat_fc = (fc.get("statistical_drift") or {})
    if stat_fc.get("available") and atr14 and float(atr14) > 0:
        citations.append(fc["evidence_hash"])
        band = float(stat_fc["interval_high"]) - float(stat_fc["interval_low"])
        if band / float(atr14) > float(cfg["evidence_pillars"]["forecast"]["wide_uncertainty_q90_q10_atr_mult"]):
            risk_flags.append("wide_forecast_uncertainty_band")

    out = {
        "role": "critic_independent_pass1",
        "context_isolated": True,
        "symbol": bundle["symbol"],
        "risk_flags": risk_flags,
        "missing_gaps": missing_gaps,
        "evidence_citations": list(dict.fromkeys(citations)),
    }
    out["output_hash"] = canonical_sha256(out)
    return out


def run_critic_adversarial_veto(
    bundle: dict[str, Any],
    researcher_out: dict[str, Any],
    critic_pass1: dict[str, Any],
    *,
    cfg: dict[str, Any] | None = None,
    enable_cost_gate: bool | None = None,
    enable_macro_filter: bool | None = None,
    enable_stat_forecast_filter: bool | None = None,
    enable_neural_forecast_filter: bool = False,
    enable_macro: bool | None = None,
    enable_forecast: bool | None = None,
    enable_critic_veto: bool | None = None,
    enable_live_external: bool = True,
) -> dict[str, Any]:
    """Pass-2 Adversarial Critic: applies orthogonal, independently toggleable veto checks."""
    cfg = cfg or load_v2_config()
    rcfg = cfg["roles_and_gates"]
    fcfg = cfg["evidence_pillars"]["forecast"]
    pillars = bundle["pillars"]
    mkt = pillars["market"]
    mac = pillars["macro"]
    fc = pillars["forecast"]
    flows = pillars["flows_positioning"]

    use_cost_gate = (
        enable_cost_gate
        if enable_cost_gate is not None
        else (True if enable_critic_veto is None else bool(enable_critic_veto))
    )
    use_macro_filter = (
        enable_macro_filter
        if enable_macro_filter is not None
        else (True if enable_macro is None else bool(enable_macro))
    )
    use_stat_fc_filter = (
        enable_stat_forecast_filter
        if enable_stat_forecast_filter is not None
        else (True if enable_forecast is None else bool(enable_forecast))
    )
    use_neural_fc_filter = bool(enable_neural_forecast_filter)

    direction = researcher_out.get("direction") or "stand_aside"
    veto_reasons: list[str] = []
    cautions: list[str] = list(critic_pass1.get("risk_flags") or [])

    if direction in {"long", "short"}:
        sign = 1.0 if direction == "long" else -1.0
        close_px = float(mkt["ohlc"]["close"])
        atr14 = float(mkt.get("atr14") or 0.0)
        stop_mult = float(researcher_out.get("proposed_stop_atr_mult") or 2.0)

        if use_cost_gate:
            # 1. Extreme volatility veto
            if "extreme_volatility_regime" in cautions:
                veto_reasons.append("veto_extreme_volatility")

            # 2. Friction share of 1R veto
            round_trip_cost_pct = 2.0 * (
                float(cfg["risk_and_costs"]["taker_fee_fallback"])
                + float(cfg["risk_and_costs"]["slippage_each_side_base"])
            )
            stop_pct = (stop_mult * atr14 / close_px) if (close_px > 0 and atr14 > 0) else 0.0
            if stop_pct <= 0 or (round_trip_cost_pct / stop_pct) > float(rcfg["critic_max_friction_share_of_1r"]):
                veto_reasons.append("veto_cost_to_volatility_drag")

            # 3. Adverse contemporaneous funding crowding veto
            f_rate = mkt.get("funding_rate_8h")
            if f_rate is not None:
                adverse_funding = sign * float(f_rate)  # >0 means trader pays funding
                if adverse_funding > float(rcfg["critic_max_adverse_funding_8h"]):
                    veto_reasons.append("veto_adverse_funding_crowding")

            # 4. Higher-timeframe conflict veto
            htf20, htf50 = mkt.get("htf_ema20_4h"), mkt.get("htf_ema50_4h")
            if htf20 is not None and htf50 is not None:
                if (direction == "long" and float(htf20) < float(htf50)) or (
                    direction == "short" and float(htf20) > float(htf50)
                ):
                    veto_reasons.append("veto_higher_timeframe_conflict")

        # 5. Point-in-time Macro headwind veto (independent toggle!)
        if use_macro_filter and mac.get("status") == "available":
            if direction == "long" and mac.get("macro_tightening_headwind"):
                veto_reasons.append("veto_macro_real_yield_dollar_headwind")
            elif direction == "short" and mac.get("macro_easing_tailwind"):
                veto_reasons.append("veto_macro_real_yield_dollar_tailwind_against_short")

        # 6a. Statistical 168h drift opposing move veto (independent toggle!)
        if use_stat_fc_filter and atr14 > 0:
            stat = fc.get("statistical_drift") or {}
            pt = stat.get("point_24h") if stat.get("available") else None
            if pt is not None:
                expected_delta = sign * (float(pt) - close_px)
                if expected_delta < -float(fcfg["opposing_move_veto_atr_mult"]) * atr14:
                    veto_reasons.append("veto_stat_forecast_strong_opposing_drift")

        # 6b. Neural Chronos opposing move veto (only if Chronos weights actually executed!)
        if use_neural_fc_filter and atr14 > 0:
            neural = fc.get("neural_chronos") or {}
            n_pt = neural.get("q50_24h") if neural.get("available") else None
            if n_pt is not None:
                expected_delta_n = sign * (float(n_pt) - close_px)
                if expected_delta_n < -float(fcfg["opposing_move_veto_atr_mult"]) * atr14:
                    veto_reasons.append("veto_neural_chronos_strong_opposing_drift")

        # 7. Live external liquidity / ETF outflow / deleveraging checks
        if enable_live_external:
            if direction == "long" and flows.get("liquidity_state") == "deleveraging":
                veto_reasons.append("veto_liquidity_deleveraging_long_block")
            if bundle["symbol"] == "BTCUSDT" and direction == "long":
                etf_f = flows.get("btc_etf_flows") or {}
                flow_musd = etf_f.get("latest_complete_total_musd")
                if etf_f.get("usable") and flow_musd is not None:
                    try:
                        if float(flow_musd) <= float(cfg["evidence_pillars"]["flows_and_positioning"]["btc_etf_outflow_caution_musd"]):
                            veto_reasons.append("veto_heavy_btc_etf_net_outflow")
                    except (TypeError, ValueError):
                        pass

    out = {
        "role": "critic_adversarial_pass2",
        "symbol": bundle["symbol"],
        "reviewed_researcher_hash": researcher_out["output_hash"],
        "switches_applied": {
            "cost_gate": use_cost_gate,
            "macro_filter": use_macro_filter,
            "stat_forecast_filter": use_stat_fc_filter,
            "neural_forecast_filter": use_neural_fc_filter,
            "live_external": bool(enable_live_external),
        },
        "vetoed": bool(veto_reasons),
        "veto_reasons": veto_reasons,
        "cautions_retained": cautions,
    }
    out["output_hash"] = canonical_sha256(out)
    return out


def run_synthesizer(
    bundle: dict[str, Any],
    researcher_out: dict[str, Any],
    critic_pass1: dict[str, Any],
    critic_veto: dict[str, Any],
) -> dict[str, Any]:
    """Synthesize Researcher and Critic without erasing disagreements or blind voting."""
    disagreements: list[str] = []
    r_dir = researcher_out.get("direction") or "stand_aside"

    for flag in critic_pass1.get("risk_flags") or []:
        if r_dir in {"long", "short"}:
            disagreements.append(f"Researcher proposed {r_dir} ({researcher_out.get('setup_type')}), while Critic flagged {flag}.")

    for v in critic_veto.get("veto_reasons") or []:
        disagreements.append(f"Critic vetoed {r_dir} due to {v}.")

    if r_dir not in {"long", "short"}:
        final_dir = "stand_aside"
        decision_stage = "researcher_no_setup"
        rationale = f"Researcher found no valid setup (regime={researcher_out.get('regime')})."
    elif critic_veto.get("vetoed"):
        final_dir = "stand_aside"
        decision_stage = "critic_veto"
        rationale = f"Researcher {r_dir} thesis blocked by Independent Critic: {', '.join(critic_veto['veto_reasons'])}."
    else:
        final_dir = r_dir
        decision_stage = "passed_to_risk_gate"
        rationale = (
            f"Researcher {r_dir} ({researcher_out.get('setup_type')}) survived Independent Critic review; "
            f"forwarding to deterministic cost & risk gate."
        )

    out = {
        "role": "synthesizer",
        "voting_used": False,
        "symbol": bundle["symbol"],
        "bundle_hash": bundle["bundle_hash"],
        "researcher_hash": researcher_out["output_hash"],
        "critic_pass1_hash": critic_pass1["output_hash"],
        "critic_veto_hash": critic_veto["output_hash"],
        "candidate_direction": final_dir,
        "decision_stage": decision_stage,
        "setup_type": researcher_out.get("setup_type") if final_dir in {"long", "short"} else "none",
        "alignment_score": researcher_out.get("alignment_score", 0.0) if final_dir in {"long", "short"} else 0.0,
        "proposed_stop_atr_mult": researcher_out.get("proposed_stop_atr_mult") if final_dir in {"long", "short"} else None,
        "proposed_exit_rule": researcher_out.get("proposed_exit_rule") if final_dir in {"long", "short"} else None,
        "invalidation": researcher_out.get("invalidation"),
        "rationale": rationale,
        "disagreements": disagreements,
        "missing_data_disclosed": sorted(set((researcher_out.get("missing_data") or []) + (critic_pass1.get("missing_gaps") or []))),
    }
    out["output_hash"] = canonical_sha256(out)
    return out
