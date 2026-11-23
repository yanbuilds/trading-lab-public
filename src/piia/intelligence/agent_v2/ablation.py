"""Unified Out-of-Sample & Module Ablation Tournament Engine (PIIA v2).

Evaluates 5 simple baselines and 4 progressive PIIA v2 module-ablation arms on
identical bars, identical starting capital, identical single-position isolated
risk limits, identical Bitget contract specs, identical taker fees, identical
slippage (1x base + 2x stress), and identical contemporaneous funding rates.

Arms:
- `B0_cash`: Zero-exposure cash reference.
- `B1_simple_ema_1h`: 1h EMA20/50 cross, 2*ATR14 stop, reverse-cross exit.
- `B2_aligned_ema_4h`: 1h EMA20/50 cross gated by completed 4h EMA20/50.
- `B3_donchian_breakout_4h`: 48h Donchian breakout gated by completed 4h EMA20/50,
  Chandelier(22h, 2.5*ATR14) trailing stop.
- `B4_v1_range_bollinger`: Original v1 ADX<20 Bollinger(20, 2) mean-reversion
  with 4h trend guard, SMA20 mid exit, and 24h time stop (pure-Python replica
  so evaluation runs with or without C-extension TA-Lib).
- `M1_researcher_market_only`: PIIA v2 Independent Researcher + Risk Gate ONLY
  (Macro=OFF, Forecast=OFF, Critic Veto=OFF).
- `M2_plus_macro_vintage`: `M1` + Point-in-time ALFRED Macro (`available_at <= t`)
  with Macro headwind veto (Forecast=OFF, full Critic=partial macro only).
- `M3_plus_forecast_module`: `M1` + Causal 168h drift Forecast alignment &
  opposing-drift veto (Macro=OFF).
- `M4_full_piia_v2`: Full PIIA v2 (`Researcher + Point-in-time Macro + Forecast
  + Independent Critic Veto + Synthesizer + Programmatic Risk Gate`).
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import math
from typing import Any

import numpy as np
import pandas as pd

from piia.intelligence.agent_v2.pillars import (
    build_bar_evidence_for_symbol,
    canonical_sha256,
    load_v2_config,
    point_in_time_macro_context,
    prepare_market_feature_table,
)
from piia.intelligence.agent_v2.risk_gate import (
    evaluate_programmatic_risk,
    run_symbol_decision_trace,
)
from piia.intelligence.research_comparison import audit_comparison
from trading_lab.perps.risk import exit_pnl, stop_fill
from trading_lab.perps.risk_v2 import resolve_spec

ARMS_ORDER = (
    "B0_cash",
    "B1_simple_ema_1h",
    "B2_aligned_ema_4h",
    "B3_donchian_breakout_4h",
    "B4_v1_range_bollinger",
    "M1_researcher_market_only",
    "M2_plus_macro_vintage",
    "M3_plus_forecast_module",
    "M4_full_piia_v2",
)


def _baseline_candidate_from_row(
    arm: str,
    symbol: str,
    row: pd.Series,
    prev_row: pd.Series | None,
) -> dict[str, Any]:
    """Emit a synthetic `synth_out` compatible with `evaluate_programmatic_risk` for baselines."""
    no_trade = {"candidate_direction": "stand_aside", "decision_stage": "baseline_no_signal"}
    if arm == "B0_cash" or prev_row is None:
        return no_trade

    e20, e50 = row.get("ema20"), row.get("ema50")
    pe20, pe50 = prev_row.get("ema20"), prev_row.get("ema50")
    htf20, htf50 = row.get("htf_ema20"), row.get("htf_ema50")
    atr14 = row.get("atr14")
    atr_mean = row.get("atr14_mean50")
    adx14 = row.get("adx14")
    close_px = float(row["close"])
    prev_close = float(prev_row["close"])

    if any(v is None or not math.isfinite(float(v)) for v in (e20, e50, pe20, pe50, atr14)) or float(atr14) <= 0:
        return no_trade

    if arm == "B1_simple_ema_1h":
        if float(e20) > float(e50) and float(pe20) <= float(pe50):
            return {"candidate_direction": "long", "setup_type": "ema_1h", "alignment_score": 1.0,
                    "proposed_stop_atr_mult": 2.0, "proposed_exit_rule": "ema_reverse", "invalidation": "EMA20 < EMA50"}
        if float(e20) < float(e50) and float(pe20) >= float(pe50):
            return {"candidate_direction": "short", "setup_type": "ema_1h", "alignment_score": 1.0,
                    "proposed_stop_atr_mult": 2.0, "proposed_exit_rule": "ema_reverse", "invalidation": "EMA20 > EMA50"}
        return no_trade

    if any(v is None or not math.isfinite(float(v)) for v in (htf20, htf50)):
        return no_trade

    if arm == "B2_aligned_ema_4h":
        if float(e20) > float(e50) and float(pe20) <= float(pe50) and float(htf20) > float(htf50):
            return {"candidate_direction": "long", "setup_type": "ema_4h_aligned", "alignment_score": 1.0,
                    "proposed_stop_atr_mult": 2.0, "proposed_exit_rule": "ema_reverse", "invalidation": "EMA20 < EMA50"}
        if float(e20) < float(e50) and float(pe20) >= float(pe50) and float(htf20) < float(htf50):
            return {"candidate_direction": "short", "setup_type": "ema_4h_aligned", "alignment_score": 1.0,
                    "proposed_stop_atr_mult": 2.0, "proposed_exit_rule": "ema_reverse", "invalidation": "EMA20 > EMA50"}
        return no_trade

    if arm == "B3_donchian_breakout_4h":
        d_hi, d_lo = row.get("donchian_high"), row.get("donchian_low")
        if d_hi is not None and math.isfinite(float(d_hi)) and close_px > float(d_hi) and prev_close <= float(d_hi) and float(htf20) > float(htf50):
            return {"candidate_direction": "long", "setup_type": "donchian_breakout", "alignment_score": 1.2,
                    "proposed_stop_atr_mult": 2.0, "proposed_exit_rule": "chandelier_trail", "invalidation": "Chandelier stop"}
        if d_lo is not None and math.isfinite(float(d_lo)) and close_px < float(d_lo) and prev_close >= float(d_lo) and float(htf20) < float(htf50):
            return {"candidate_direction": "short", "setup_type": "donchian_breakout", "alignment_score": 1.2,
                    "proposed_stop_atr_mult": 2.0, "proposed_exit_rule": "chandelier_trail", "invalidation": "Chandelier stop"}
        return no_trade

    if arm == "B4_v1_range_bollinger":
        bb_lo, bb_up = row.get("bb_lower"), row.get("bb_upper")
        if any(v is None or not math.isfinite(float(v)) for v in (adx14, atr_mean, bb_lo, bb_up)) or float(atr_mean) <= 0:
            return no_trade
        if float(atr14) >= 2.0 * float(atr_mean) or float(adx14) >= 20.0:
            return no_trade
        if close_px < float(bb_lo) and float(htf20) > float(htf50):
            return {"candidate_direction": "long", "setup_type": "v1_range", "alignment_score": 1.0,
                    "proposed_stop_atr_mult": 2.0, "proposed_exit_rule": "bb_mid_or_24h", "invalidation": "2*ATR14"}
        if close_px > float(bb_up) and float(htf20) < float(htf50):
            return {"candidate_direction": "short", "setup_type": "v1_range", "alignment_score": 1.0,
                    "proposed_stop_atr_mult": 2.0, "proposed_exit_rule": "bb_mid_or_24h", "invalidation": "2*ATR14"}
        return no_trade

    raise ValueError(f"Unknown baseline arm: {arm}")


def simulate_single_arm(
    arm: str,
    prepared_tables: dict[str, pd.DataFrame],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    starting_cash: float = 100.0,
    slippage: float | None = None,
    funding: dict[str, list[dict[str, Any]]] | None = None,
    macro_observations: dict[str, list[dict[str, Any]]] | None = None,
    specs_override: dict[str, dict[str, Any]] | None = None,
    cfg: dict[str, Any] | None = None,
    record_traces: bool = False,
) -> dict[str, Any]:
    """Simulate one tournament arm on common hourly timestamps in `[start, end]`."""
    cfg = cfg or load_v2_config()
    rc = cfg["risk_and_costs"]
    rg = cfg["roles_and_gates"]
    slip = float(rc["slippage_each_side_base"] if slippage is None else slippage)
    daily_halt = float(rc["daily_loss_halt_fraction"])
    total_halt = float(rc["total_loss_halt_fraction"])
    be_r = float(rg["breakeven_ratchet_r"])

    common = sorted(set.intersection(*(set(df.index) for df in prepared_tables.values())))
    eval_bars = [ts for ts in common if start <= ts <= end]
    if not eval_bars:
        raise ValueError("No common bars inside evaluation window")

    v1_specs = {
        sym: resolve_spec(sym, (specs_override or {}).get(sym)).to_v1_spec()
        for sym in prepared_tables
    }
    fmaps = {
        sym: {pd.Timestamp(int(r["timestamp"]), unit="ms", tz="UTC"): float(r["rate"]) for r in rows}
        for sym, rows in (funding or {}).items()
    }

    cash = float(starting_cash)
    pos: dict[str, Any] | None = None
    closed_trades: list[dict[str, Any]] = []
    nav: list[tuple[str, float]] = []
    stand_aside_counts: Counter[str] = Counter()
    veto_counts: Counter[str] = Counter()
    traces_sample: list[dict[str, Any]] = []
    fees_paid = 0.0
    funding_paid = 0.0
    kill = False
    day = None
    day_start = cash
    bars_in_pos = 0

    # Pre-cache macro context per UTC calendar day for speed (since lag is calendar-day based)
    macro_by_day: dict[Any, dict[str, Any]] = {}

    def _get_macro(dt_utc: datetime) -> dict[str, Any]:
        d_key = dt_utc.date()
        if d_key not in macro_by_day:
            macro_by_day[d_key] = point_in_time_macro_context(macro_observations, dt_utc, cfg=cfg)
        return macro_by_day[d_key]

    def _close_position(raw_px: float, ts_exit: pd.Timestamp, reason: str) -> None:
        nonlocal cash, pos, fees_paid
        assert pos is not None
        sp = v1_specs[pos["symbol"]]
        side_i = 1 if pos["side"] == "long" else -1
        qty = pos["qty"]
        fill_px = raw_px * (1.0 - side_i * slip)
        exit_fee = qty * sp.contract_size * fill_px * sp.taker
        gross = exit_pnl(side_i, qty, pos["entry"], fill_px, sp)
        cash += gross - exit_fee
        fees_paid += exit_fee
        closed_trades.append({
            "symbol": pos["symbol"],
            "side": pos["side"],
            "setup_type": pos["setup_type"],
            "entry_time": str(pos["entry_time"]),
            "exit_time": str(ts_exit),
            "entry": round(pos["entry"], 6),
            "exit": round(fill_px, 6),
            "qty": qty,
            "net_pnl": gross - pos["entry_fee"] - exit_fee - pos["funding"],
            "entry_fee": pos["entry_fee"],
            "exit_fee": exit_fee,
            "funding_paid": pos["funding"],
            "exit_reason": reason,
            "decision_id": pos.get("decision_id"),
        })
        pos = None

    common_pos = {ts: i for i, ts in enumerate(common)}

    for ts in eval_bars:
        idx = common_pos[ts]
        if idx == 0:
            continue
        prev_ts = common[idx - 1]
        prev2_ts = common[idx - 2] if idx >= 2 else None
        if ts - prev_ts != pd.Timedelta(hours=1):
            raise ValueError(f"Gapped hourly timeline at {ts}")

        rows_now = {sym: df.loc[ts] for sym, df in prepared_tables.items()}
        rows_sig = {sym: df.loc[prev_ts] for sym, df in prepared_tables.items()}
        rows_sig_prev = {sym: df.loc[prev2_ts] for sym, df in prepared_tables.items()} if prev2_ts is not None else {}

        if ts.date() != day:
            day = ts.date()
            if pos is not None:
                sp_d = v1_specs[pos["symbol"]]
                day_start = cash + exit_pnl(1 if pos["side"] == "long" else -1, pos["qty"], pos["entry"], float(rows_now[pos["symbol"]]["open"]), sp_d)
            else:
                day_start = cash

        # 1. Manage existing position at bar open (funding -> trailing update -> gap stop -> signal/time exit)
        if pos is not None:
            k = pos["symbol"]
            sp = v1_specs[k]
            side_i = 1 if pos["side"] == "long" else -1
            open_now = float(rows_now[k]["open"])
            sig_row = rows_sig[k]

            rate = fmaps.get(k, {}).get(ts, 0.0)
            if rate:
                f_amt = side_i * pos["qty"] * sp.contract_size * open_now * rate
                cash -= f_amt
                funding_paid += f_amt
                pos["funding"] += f_amt

            # Update Chandelier trailing stop for trend/breakout setups using `rows_now[k]` (which has shift_bars=1!)
            if pos["setup_type"] in {"trend_breakout", "trend_pullback", "donchian_breakout"}:
                ch_col = "chandelier_long" if side_i == 1 else "chandelier_short"
                ch_val = rows_now[k].get(ch_col)
                if ch_val is not None and math.isfinite(float(ch_val)):
                    if side_i == 1:
                        pos["stop"] = max(pos["stop"], float(ch_val))
                    else:
                        pos["stop"] = min(pos["stop"], float(ch_val))

            gap = (side_i == 1 and open_now <= pos["stop"]) or (side_i == -1 and open_now >= pos["stop"])
            if gap:
                _close_position(open_now, ts, "gap_stop")
            else:
                st_type = pos["setup_type"]
                if st_type in {"ema_1h", "ema_4h_aligned"}:
                    rev = (side_i == 1 and float(sig_row["ema20"]) < float(sig_row["ema50"])) or (
                        side_i == -1 and float(sig_row["ema20"]) > float(sig_row["ema50"])
                    )
                    if rev:
                        _close_position(open_now, ts, "reverse_signal")
                elif st_type == "v1_range":
                    bb_m = sig_row.get("bb_mid")
                    mid_hit = bb_m is not None and math.isfinite(float(bb_m)) and (
                        (side_i == 1 and float(sig_row["close"]) >= float(bb_m))
                        or (side_i == -1 and float(sig_row["close"]) <= float(bb_m))
                    )
                    if mid_hit:
                        _close_position(open_now, ts, "bb_mid_exit")
                    elif ts - pos["entry_time"] >= pd.Timedelta(hours=24):
                        _close_position(open_now, ts, "time_stop_24h")
                elif st_type == "range_reversion":
                    bb_m = sig_row.get("bb_mid")
                    mid_hit = bb_m is not None and math.isfinite(float(bb_m)) and (
                        (side_i == 1 and float(sig_row["close"]) >= float(bb_m))
                        or (side_i == -1 and float(sig_row["close"]) <= float(bb_m))
                    )
                    if mid_hit:
                        _close_position(open_now, ts, "range_mid_target")
                    elif ts - pos["entry_time"] >= pd.Timedelta(hours=int(rg["range_max_hold_hours"])):
                        _close_position(open_now, ts, "range_time_stop")
                else:
                    # Trend leg (`trend_breakout`, `trend_pullback`, `donchian_breakout`)
                    rev = (
                        (side_i == 1 and float(sig_row["ema20"]) < float(sig_row["ema50"]) and float(sig_row["htf_ema20"]) < float(sig_row["htf_ema50"]))
                        or (side_i == -1 and float(sig_row["ema20"]) > float(sig_row["ema50"]) and float(sig_row["htf_ema20"]) > float(sig_row["htf_ema50"]))
                    )
                    if rev:
                        _close_position(open_now, ts, "htf_trend_reversal")
                    elif st_type != "donchian_breakout" and ts - pos["entry_time"] >= pd.Timedelta(hours=int(rg["trend_max_hold_hours"])):
                        _close_position(open_now, ts, "trend_time_stop")

        eq_open = cash + (
            exit_pnl(1 if pos["side"] == "long" else -1, pos["qty"], pos["entry"], float(rows_now[pos["symbol"]]["open"]), v1_specs[pos["symbol"]])
            if pos is not None else 0.0
        )
        if eq_open <= day_start * (1.0 - daily_halt) or eq_open <= starting_cash * (1.0 - total_halt):
            kill = True

        # 2. Evaluate new entry candidates when flat and not halted
        pos_at_open = pos is not None
        if pos is None and not kill and arm != "B0_cash":
            decision_utc = ts.to_pydatetime()  # prev_ts bar closes at `ts`!
            macro_ctx = _get_macro(decision_utc) if arm in {"M2_plus_macro_vintage", "M4_full_piia_v2"} else None

            actionable_traces: list[tuple[float, str, dict[str, Any]]] = []
            for sym in sorted(prepared_tables):
                sig_row = rows_sig[sym]
                prev_c = float(rows_sig_prev[sym]["close"]) if sym in rows_sig_prev else None
                next_open_px = float(rows_now[sym]["open"])
                # Last observed funding rate at or before `prev_ts` (no peeking at future funding!)
                last_f_rate = None
                sym_fmap = fmaps.get(sym) or {}
                for look_h in range(0, 9):
                    chk = prev_ts - pd.Timedelta(hours=look_h)
                    if chk in sym_fmap:
                        last_f_rate = sym_fmap[chk]
                        break

                if arm.startswith("B"):
                    synth_cand = _baseline_candidate_from_row(
                        arm, sym, sig_row, rows_sig_prev.get(sym)
                    )
                    mkt_pil = {
                        "ohlc": {"close": float(sig_row["close"])},
                        "atr14": float(sig_row["atr14"]) if pd.notna(sig_row["atr14"]) else 0.0,
                        "funding_rate_8h": last_f_rate,
                    }
                    risk_res = evaluate_programmatic_risk(
                        symbol=sym,
                        synth_out=synth_cand,
                        market_pillar=mkt_pil,
                        equity_usdt=cash,
                        free_cash_usdt=cash,
                        open_positions_count=0,
                        kill_switch_halted=kill,
                        planned_entry_price=next_open_px,
                        slippage_override=slip,
                        spec_override=(specs_override or {}).get(sym),
                        cfg=cfg,
                    )
                    if risk_res["approved_by_risk_gate"]:
                        actionable_traces.append((float(synth_cand.get("alignment_score", 1.0)), sym, {
                            "decision_id": f"{arm}:{sym}:{ts.isoformat()}",
                            "setup_type": synth_cand["setup_type"],
                            "programmatic_risk_gate": risk_res,
                        }))
                    else:
                        stand_aside_counts[str(risk_res["stand_aside_reason"])] += 1
                else:
                    # PIIA v2 Module Ablation Arms (`M1`..`M4`)
                    en_macro = arm in {"M2_plus_macro_vintage", "M4_full_piia_v2"}
                    en_fc = arm in {"M3_plus_forecast_module", "M4_full_piia_v2"}
                    en_critic = arm in {"M2_plus_macro_vintage", "M3_plus_forecast_module", "M4_full_piia_v2"}
                    bundle = build_bar_evidence_for_symbol(
                        sym,
                        feature_row=sig_row,
                        prev_close=prev_c,
                        decision_utc=decision_utc,
                        funding_rate_8h=last_f_rate,
                        macro_ctx=macro_ctx if en_macro else None,
                    )
                    # Note: for M2 (macro only) & M3 (forecast only), isolate veto to ONLY that module vs M4 full critic!
                    if arm == "M4_full_piia_v2":
                        trace = run_symbol_decision_trace(
                            bundle,
                            equity_usdt=cash,
                            free_cash_usdt=cash,
                            open_positions_count=0,
                            kill_switch_halted=kill,
                            planned_entry_price=next_open_px,
                            slippage_override=slip,
                            spec_override=(specs_override or {}).get(sym),
                            enable_macro=True,
                            enable_forecast=True,
                            enable_critic_veto=True,
                            enable_live_external=False,
                            cfg=cfg,
                        )
                    elif arm == "M1_researcher_market_only":
                        trace = run_symbol_decision_trace(
                            bundle,
                            equity_usdt=cash,
                            free_cash_usdt=cash,
                            open_positions_count=0,
                            kill_switch_halted=kill,
                            planned_entry_price=next_open_px,
                            slippage_override=slip,
                            spec_override=(specs_override or {}).get(sym),
                            enable_macro=False,
                            enable_forecast=False,
                            enable_critic_veto=False,
                            enable_live_external=False,
                            cfg=cfg,
                        )
                    else:
                        # M2 or M3: run trace with only macro or only forecast enabled, filtering critic veto to that module
                        trace = run_symbol_decision_trace(
                            bundle,
                            equity_usdt=cash,
                            free_cash_usdt=cash,
                            open_positions_count=0,
                            kill_switch_halted=kill,
                            planned_entry_price=next_open_px,
                            slippage_override=slip,
                            spec_override=(specs_override or {}).get(sym),
                            enable_macro=en_macro,
                            enable_forecast=en_fc,
                            enable_critic_veto=True,
                            enable_live_external=False,
                            cfg=cfg,
                        )
                    for vr in trace["critic_adversarial_pass2"]["veto_reasons"]:
                        veto_counts[vr] += 1
                    if record_traces and (trace["final_action"] in {"long", "short"} or trace["critic_adversarial_pass2"]["vetoed"]):
                        if len(traces_sample) < 25:
                            traces_sample.append(trace)
                    rg_res = trace["programmatic_risk_gate"]
                    if rg_res["approved_by_risk_gate"]:
                        score = float(trace["synthesizer"]["alignment_score"])
                        actionable_traces.append((score, sym, {
                            "decision_id": trace["decision_id"],
                            "setup_type": trace["synthesizer"]["setup_type"],
                            "programmatic_risk_gate": rg_res,
                        }))
                    else:
                        stand_aside_counts[str(trace["stand_aside_reason"])] += 1

            if actionable_traces:
                # Sort by highest alignment score (deterministic tie-break by symbol)
                actionable_traces.sort(key=lambda item: (-item[0], item[1]))
                _best_score, chosen_sym, chosen_meta = actionable_traces[0]
                rg_ok = chosen_meta["programmatic_risk_gate"]
                sz = rg_ok["position_sizing"]
                ep = rg_ok["entry_plan"]
                xp = rg_ok["exit_plan"]
                cr = rg_ok["cost_and_risk"]
                entry_fee = float(cr["entry_taker_fee_usdt"])
                cash -= entry_fee
                fees_paid += entry_fee
                pos = {
                    "symbol": chosen_sym,
                    "side": sz["side"],
                    "setup_type": chosen_meta["setup_type"],
                    "qty": float(sz["qty_contracts"]),
                    "entry": float(ep["assumed_fill_with_slippage"]),
                    "stop": float(xp["initial_stop_loss"]),
                    "be_stop": float(xp["fee_adjusted_breakeven_stop"]),
                    "be_trigger": float(xp["breakeven_ratchet_trigger_price"]),
                    "be_locked": False,
                    "entry_fee": entry_fee,
                    "funding": 0.0,
                    "entry_time": ts,
                    "decision_id": chosen_meta["decision_id"],
                }

        if pos_at_open or pos is not None:
            bars_in_pos += 1

        # 3. Intrabar stop check, then end-of-bar breakeven ratchet for M-arms on NEXT bar
        if pos is not None:
            k = pos["symbol"]
            r_now = rows_now[k]
            side_i = 1 if pos["side"] == "long" else -1
            hit = stop_fill(side_i, pos["stop"], float(r_now["open"]), float(r_now["low"]), float(r_now["high"]))
            if hit is not None:
                _close_position(hit, ts + pd.Timedelta(hours=1), "stop_loss")
            elif arm.startswith("M") and not pos["be_locked"]:
                reached = (side_i == 1 and float(r_now["high"]) >= pos["be_trigger"]) or (
                    side_i == -1 and float(r_now["low"]) <= pos["be_trigger"]
                )
                if reached:
                    if (side_i == 1 and pos["be_stop"] > pos["stop"]) or (side_i == -1 and pos["be_stop"] < pos["stop"]):
                        pos["stop"] = pos["be_stop"]
                        pos["be_locked"] = True

        # 4. Mark-to-market at bar close & loss-halt check
        eq_close = cash + (
            exit_pnl(1 if pos["side"] == "long" else -1, pos["qty"], pos["entry"], float(rows_now[pos["symbol"]]["close"]), v1_specs[pos["symbol"]])
            if pos is not None else 0.0
        )
        nav.append((str(ts + pd.Timedelta(hours=1)), eq_close))
        if eq_close <= day_start * (1.0 - daily_halt) or eq_close <= starting_cash * (1.0 - total_halt):
            kill = True
        if kill and pos is not None:
            _close_position(float(rows_now[pos["symbol"]]["close"]), ts + pd.Timedelta(hours=1), "loss_halt")
            nav[-1] = (nav[-1][0], cash)

    eq_s = pd.Series([starting_cash] + [v for _, v in nav], dtype=float)
    final_eq = float(eq_s.iloc[-1])
    pnls = [t["net_pnl"] for t in closed_trades]
    wins = [p for p in pnls if p > 0.01]
    losses = [p for p in pnls if p < -0.01]
    open_unrealized = 0.0
    if pos is not None:
        open_unrealized = final_eq - cash - pos["entry_fee"] - pos["funding"]
    recon_err = abs(sum(pnls) + open_unrealized - (final_eq - starting_cash))
    if recon_err >= 1e-7:
        raise AssertionError(f"Ledger reconciliation failed for {arm}: {recon_err}")

    # Hourly Sharpe & Sortino proxy (annualized sqrt(8760))
    rets_h = eq_s.pct_change().dropna()
    sharpe = None
    if len(rets_h) > 10 and float(rets_h.std()) > 1e-12:
        sharpe = round(float(rets_h.mean() / rets_h.std() * math.sqrt(8760)), 3)

    return {
        "arm": arm,
        "initial_usdt": starting_cash,
        "final_usdt": round(final_eq, 4),
        "return_pct": round((final_eq / starting_cash - 1.0) * 100.0, 4),
        "max_drawdown_pct": round(float((1.0 - eq_s / eq_s.cummax()).max() * 100.0), 4),
        "annualized_sharpe_hourly_proxy": sharpe,
        "closed_trades": len(closed_trades),
        "wins": len(wins),
        "losses": len(losses),
        "winrate_pct": round(100.0 * len(wins) / len(pnls), 2) if pnls else None,
        "fees_paid_usdt": round(fees_paid, 4),
        "funding_net_paid_usdt": round(funding_paid, 4),
        "open_positions_end": int(pos is not None),
        "open_unrealized_usdt": round(open_unrealized, 4),
        "exposure_bar_fraction": round(bars_in_pos / len(eval_bars), 4) if eval_bars else 0.0,
        "loss_halt_triggered": bool(kill),
        "reconciliation_error": recon_err,
        "trades_by_symbol": dict(Counter(t["symbol"] for t in closed_trades)),
        "exit_reasons": dict(Counter(t["exit_reason"] for t in closed_trades)),
        "critic_veto_counts": dict(veto_counts),
        "stand_aside_breakdown": dict(stand_aside_counts),
        "trades": closed_trades,
        "traces_sample": traces_sample,
    }


def run_ablation_tournament(
    frames: dict[str, pd.DataFrame],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    starting_cash: float = 100.0,
    held_out: bool = False,
    market_data_hash: str = "synthetic_or_supplied",
    funding: dict[str, list[dict[str, Any]]] | None = None,
    funding_complete: bool = False,
    macro_observations: dict[str, list[dict[str, Any]]] | None = None,
    specs_override: dict[str, dict[str, Any]] | None = None,
    arms: tuple[str, ...] = ARMS_ORDER,
    include_2x_cost_stress: bool = True,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all tournament arms under identical rules and verify via `audit_comparison`."""
    cfg = cfg or load_v2_config()
    # Strictly slice up to `end` BEFORE preparing indicator tables!
    prefix_frames = {sym: df.loc[df.index <= end].copy() for sym, df in frames.items()}
    prepared = {sym: prepare_market_feature_table(df, cfg=cfg) for sym, df in prefix_frames.items()}

    base_slip = float(cfg["risk_and_costs"]["slippage_each_side_base"])
    stress_mult = float(cfg["risk_and_costs"]["slippage_each_side_stress_multiplier"])

    results_base: dict[str, Any] = {}
    results_stress: dict[str, Any] = {}
    comparison_declarations = []

    for arm in arms:
        res = simulate_single_arm(
            arm,
            prepared,
            start=start,
            end=end,
            starting_cash=starting_cash,
            slippage=base_slip,
            funding=funding,
            macro_observations=macro_observations,
            specs_override=specs_override,
            cfg=cfg,
            record_traces=(arm == "M4_full_piia_v2"),
        )
        res_summary = {k: v for k, v in res.items() if k not in {"trades", "traces_sample"}}
        res_summary["sample_trades"] = res["trades"][:10]
        if arm == "M4_full_piia_v2":
            res_summary["sample_decision_traces"] = res["traces_sample"][:5]
        results_base[arm] = res_summary

        if include_2x_cost_stress:
            st_res = simulate_single_arm(
                arm,
                prepared,
                start=start,
                end=end,
                starting_cash=starting_cash,
                slippage=base_slip * stress_mult,
                funding=funding,
                macro_observations=macro_observations,
                specs_override=specs_override,
                cfg=cfg,
                record_traces=False,
            )
            results_stress[arm] = {
                "final_usdt": st_res["final_usdt"],
                "return_pct": st_res["return_pct"],
                "max_drawdown_pct": st_res["max_drawdown_pct"],
                "closed_trades": st_res["closed_trades"],
                "fees_paid_usdt": st_res["fees_paid_usdt"],
                "loss_halt_triggered": st_res["loss_halt_triggered"],
            }

        comparison_declarations.append({
            "name": arm,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "starting_cash": starting_cash,
            "market_data_hash": market_data_hash,
            "cost_model_id": "bitget_public_taker_plus_5bps_slip",
            "risk_budget_id": "isolated_2x_1pct_stop_20pct_cash_floor_single_pos",
            "execution_model_id": "next_bar_open_intrabar_stop_worst_first",
            "funding_model_id": "verified_public_interval_open_proxy" if funding_complete else "price_only_or_partial_funding",
            "rules_frozen_before_test": True,
            "held_out": held_out,
        })

    comp_audit = audit_comparison(comparison_declarations)
    return {
        "window_start_utc": start.isoformat(),
        "window_end_utc": end.isoformat(),
        "starting_cash_usdt": starting_cash,
        "held_out": held_out,
        "funding_complete": funding_complete,
        "net_performance_claim_permitted": funding_complete,
        "comparison_protocol_audit": comp_audit,
        "base_cost_results": results_base,
        "stress_2x_slippage_results": results_stress if include_2x_cost_stress else None,
    }


def audit_prefix_lookahead(
    frames: dict[str, pd.DataFrame],
    *,
    start: pd.Timestamp,
    cutoffs: list[pd.Timestamp],
    end: pd.Timestamp,
    starting_cash: float = 500.0,
    funding: dict[str, list[dict[str, Any]]] | None = None,
    macro_observations: dict[str, list[dict[str, Any]]] | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove zero lookahead: physically deleting bars after `cutoff` leaves all prior closed trades identical."""
    cfg = cfg or load_v2_config()
    full_prep = {sym: prepare_market_feature_table(df.loc[df.index <= end].copy(), cfg=cfg) for sym, df in frames.items()}
    full_run = simulate_single_arm(
        "M4_full_piia_v2",
        full_prep,
        start=start,
        end=end,
        starting_cash=starting_cash,
        funding=funding,
        macro_observations=macro_observations,
        cfg=cfg,
    )
    full_trades = full_run["trades"]

    checks = []
    all_passed = True
    for cut in cutoffs:
        trunc_frames = {sym: df.loc[df.index <= cut].copy() for sym, df in frames.items()}
        trunc_prep = {sym: prepare_market_feature_table(df, cfg=cfg) for sym, df in trunc_frames.items()}
        trunc_run = simulate_single_arm(
            "M4_full_piia_v2",
            trunc_prep,
            start=start,
            end=cut,
            starting_cash=starting_cash,
            funding=funding,
            macro_observations=macro_observations,
            cfg=cfg,
        )
        # Every trade in `full_trades` whose `exit_time <= cut + 1h` must match `trunc_run["trades"]` exactly!
        limit_ts = cut + pd.Timedelta(hours=1)
        expected_closed = [t for t in full_trades if pd.Timestamp(t["exit_time"]) <= limit_ts]
        actual_closed = trunc_run["trades"]
        matched = expected_closed == actual_closed
        if not matched:
            all_passed = False
        checks.append({
            "cutoff_utc": cut.isoformat(),
            "closed_trades_before_cutoff": len(actual_closed),
            "exact_match_with_full_run": matched,
        })

    return {
        "ok": all_passed,
        "arm_tested": "M4_full_piia_v2",
        "cutoffs_checked": checks,
    }
