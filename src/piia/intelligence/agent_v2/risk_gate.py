"""Programmatic Risk & Cost Gate + Traceable Decision Record Builder (PIIA v2).

Strictly deterministic and completely separate from Researcher/Critic/LLM text.
Enforces:
- Downward-rounded contract sizing (`qty_step`), `min_qty`, and `min_notional`.
- 1% equity stop-loss budget INCLUSIVE of entry + exit taker fees and slippage.
- 20% cash floor, 50% max margin fraction, and 1-interval adverse funding reserve.
- Stop-loss strictly before estimated isolated liquidation price (`>= 1.5%` buffer).
- Explicit Entry, Stop-Loss, Take-Profit / Trailing / Time-Stop, and Invalidation.
"""
from __future__ import annotations

from decimal import Decimal
import math
from typing import Any
import uuid

from piia.intelligence.agent_v2.pillars import canonical_sha256, load_v2_config
from piia.intelligence.agent_v2.roles import (
    run_critic_adversarial_veto,
    run_independent_critic_pass1,
    run_independent_researcher,
    run_synthesizer,
)
from trading_lab.perps.risk import floor_step
from trading_lab.perps.risk_v2 import (
    estimate_isolated_liq_v2,
    fee_adjusted_breakeven,
    resolve_spec,
)


def evaluate_programmatic_risk(
    *,
    symbol: str,
    synth_out: dict[str, Any],
    market_pillar: dict[str, Any],
    equity_usdt: float,
    free_cash_usdt: float,
    open_positions_count: int = 0,
    kill_switch_halted: bool = False,
    planned_entry_price: float | None = None,
    slippage_override: float | None = None,
    spec_override: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run deterministic sizing, liquidation-buffer, and cost checks."""
    cfg = cfg or load_v2_config()
    rc = cfg["risk_and_costs"]
    rg = cfg["roles_and_gates"]

    cand_dir = synth_out.get("candidate_direction") or "stand_aside"
    if cand_dir not in {"long", "short"}:
        return {
            "approved_by_risk_gate": False,
            "final_action": "stand_aside",
            "stand_aside_reason": synth_out.get("decision_stage") or "no_candidate",
            "entry_plan": None,
            "exit_plan": None,
            "position_sizing": {"qty_contracts": 0.0, "notional_usdt": 0.0, "margin_usdt": 0.0},
            "cost_and_risk": None,
        }

    if kill_switch_halted:
        return {
            "approved_by_risk_gate": False,
            "final_action": "stand_aside",
            "stand_aside_reason": "risk_gate:kill_switch_halted",
            "entry_plan": None,
            "exit_plan": None,
            "position_sizing": {"qty_contracts": 0.0, "notional_usdt": 0.0, "margin_usdt": 0.0},
            "cost_and_risk": None,
        }

    if open_positions_count >= int(rc["max_open_positions"]):
        return {
            "approved_by_risk_gate": False,
            "final_action": "stand_aside",
            "stand_aside_reason": "risk_gate:max_open_positions_reached",
            "entry_plan": None,
            "exit_plan": None,
            "position_sizing": {"qty_contracts": 0.0, "notional_usdt": 0.0, "margin_usdt": 0.0},
            "cost_and_risk": None,
        }

    spec = resolve_spec(symbol, spec_override)
    lev = float(rc["leverage"])
    r_frac = float(rc["risk_fraction_per_trade"])
    cash_floor = float(rc["cash_floor_fraction"])
    max_margin_frac = float(rc["max_margin_fraction"])
    slip = float(rc["slippage_each_side_base"] if slippage_override is None else slippage_override)
    min_liq_buf = float(rc["min_stop_to_liquidation_buffer_pct"])

    ref_px = float(planned_entry_price if planned_entry_price is not None else market_pillar["ohlc"]["close"])
    atr14 = float(market_pillar.get("atr14") or 0.0)
    stop_mult = float(synth_out.get("proposed_stop_atr_mult") or 2.0)

    # Check live orderbook VWAP if present
    live_book = market_pillar.get("live_orderbook")
    if isinstance(live_book, dict) and live_book.get("ok") and planned_entry_price is None:
        side_levels = live_book.get("asks" if cand_dir == "long" else "bids") or []
        if side_levels:
            try:
                ref_px = float(side_levels[0]["price"])
            except (KeyError, TypeError, ValueError):
                pass

    if ref_px <= 0 or atr14 <= 0 or equity_usdt <= 0 or free_cash_usdt <= 0:
        return {
            "approved_by_risk_gate": False,
            "final_action": "stand_aside",
            "stand_aside_reason": "risk_gate:invalid_price_atr_or_equity",
            "entry_plan": None,
            "exit_plan": None,
            "position_sizing": {"qty_contracts": 0.0, "notional_usdt": 0.0, "margin_usdt": 0.0},
            "cost_and_risk": None,
        }

    sign = 1.0 if cand_dir == "long" else -1.0
    entry_fill = ref_px * (1.0 + sign * slip)
    stop_dist = stop_mult * atr14
    stop_px = entry_fill - sign * stop_dist
    if stop_px <= 0:
        return {
            "approved_by_risk_gate": False,
            "final_action": "stand_aside",
            "stand_aside_reason": "risk_gate:nonpositive_stop_price",
            "entry_plan": None,
            "exit_plan": None,
            "position_sizing": {"qty_contracts": 0.0, "notional_usdt": 0.0, "margin_usdt": 0.0},
            "cost_and_risk": None,
        }
    stop_fill_px = stop_px * (1.0 - sign * slip)

    loss_per_contract = spec.contract_size * (
        sign * (entry_fill - stop_fill_px)
        + spec.taker * (entry_fill + stop_fill_px)
    )
    if not math.isfinite(loss_per_contract) or loss_per_contract <= 0:
        return {
            "approved_by_risk_gate": False,
            "final_action": "stand_aside",
            "stand_aside_reason": "risk_gate:invalid_loss_per_contract",
            "entry_plan": None,
            "exit_plan": None,
            "position_sizing": {"qty_contracts": 0.0, "notional_usdt": 0.0, "margin_usdt": 0.0},
            "cost_and_risk": None,
        }

    risk_budget_usdt = equity_usdt * r_frac
    qty_risk = risk_budget_usdt / loss_per_contract

    f_rate = market_pillar.get("funding_rate_8h")
    f_reserve_rate = abs(float(f_rate)) if (f_rate is not None and math.isfinite(float(f_rate))) else 0.0
    avail_cash = min(equity_usdt * max_margin_frac, max(0.0, free_cash_usdt - equity_usdt * cash_floor))
    unit_cash = spec.contract_size * entry_fill * ((1.0 / lev) + spec.taker + f_reserve_rate)
    qty_cash = avail_cash / unit_cash if unit_cash > 0 else 0.0

    qty = floor_step(max(0.0, min(qty_risk, qty_cash)), spec.qty_step)
    notional = qty * entry_fill * spec.contract_size

    # Tier check if live tiers present
    mmr = spec.mmr
    live_tiers = market_pillar.get("live_position_tiers")
    if isinstance(live_tiers, dict) and live_tiers.get("ok") and qty > 0:
        from piia.intelligence.public_position_tiers import conservative_tier_for_notional
        picked = conservative_tier_for_notional(live_tiers, Decimal(str(notional)))
        if picked.get("ok"):
            mmr = float(picked["tier"]["keep_margin_rate"])
            if float(picked["tier"]["max_leverage"]) < lev:
                return {
                    "approved_by_risk_gate": False,
                    "final_action": "stand_aside",
                    "stand_aside_reason": "risk_gate:leverage_exceeds_venue_tier",
                    "entry_plan": None,
                    "exit_plan": None,
                    "position_sizing": {"qty_contracts": 0.0, "notional_usdt": 0.0, "margin_usdt": 0.0},
                    "cost_and_risk": None,
                }

    if qty < spec.min_qty or notional < spec.min_notional:
        return {
            "approved_by_risk_gate": False,
            "final_action": "stand_aside",
            "stand_aside_reason": "risk_gate:minimum_order_exceeds_budget",
            "entry_plan": None,
            "exit_plan": None,
            "position_sizing": {
                "qty_contracts": 0.0,
                "min_qty_required": spec.min_qty,
                "min_notional_usdt": spec.min_notional,
                "risk_budget_usdt": round(risk_budget_usdt, 4),
            },
            "cost_and_risk": None,
        }

    # Verify visible orderbook depth if live orderbook is supplied
    if isinstance(live_book, dict) and live_book.get("ok"):
        from piia.intelligence.public_orderbook import walk_book
        levels = live_book.get("asks" if cand_dir == "long" else "bids") or []
        walk = walk_book(levels, Decimal(str(qty)))
        if not walk.get("ok"):
            return {
                "approved_by_risk_gate": False,
                "final_action": "stand_aside",
                "stand_aside_reason": f"risk_gate:orderbook_{walk.get('reason')}",
                "entry_plan": None,
                "exit_plan": None,
                "position_sizing": {"qty_contracts": 0.0, "notional_usdt": 0.0, "margin_usdt": 0.0},
                "cost_and_risk": None,
            }

    est_liq = estimate_isolated_liq_v2(
        side=cand_dir,
        qty=qty,
        entry_fill=entry_fill,
        leverage=lev,
        mmr=mmr,
        taker=spec.taker,
        contract_size=spec.contract_size,
    )
    if est_liq is None:
        return {
            "approved_by_risk_gate": False,
            "final_action": "stand_aside",
            "stand_aside_reason": "risk_gate:liquidation_estimate_unavailable",
            "entry_plan": None,
            "exit_plan": None,
            "position_sizing": {"qty_contracts": 0.0, "notional_usdt": 0.0, "margin_usdt": 0.0},
            "cost_and_risk": None,
        }

    if (cand_dir == "long" and stop_px <= est_liq) or (cand_dir == "short" and stop_px >= est_liq):
        return {
            "approved_by_risk_gate": False,
            "final_action": "stand_aside",
            "stand_aside_reason": "risk_gate:stop_at_or_beyond_estimated_liquidation",
            "entry_plan": None,
            "exit_plan": None,
            "position_sizing": {"qty_contracts": 0.0, "notional_usdt": 0.0, "margin_usdt": 0.0},
            "cost_and_risk": None,
        }

    liq_buf_pct = abs(stop_px - est_liq) / entry_fill
    if liq_buf_pct < min_liq_buf:
        return {
            "approved_by_risk_gate": False,
            "final_action": "stand_aside",
            "stand_aside_reason": "risk_gate:insufficient_liquidation_buffer",
            "entry_plan": None,
            "exit_plan": None,
            "position_sizing": {"qty_contracts": 0.0, "notional_usdt": 0.0, "margin_usdt": 0.0},
            "cost_and_risk": None,
        }

    margin_usdt = notional / lev
    entry_fee = notional * spec.taker
    stop_exit_fee = qty * stop_fill_px * spec.contract_size * spec.taker
    funding_reserve_usdt = notional * f_reserve_rate
    est_stop_loss_usdt = qty * loss_per_contract
    be_px = fee_adjusted_breakeven(entry_fill, cand_dir, entry_fee_rate=spec.taker, exit_fee_rate=spec.taker)

    setup = synth_out.get("setup_type") or "trend_breakout"
    max_hold = int(rg["range_max_hold_hours"] if setup == "range_reversion" else rg["trend_max_hold_hours"])
    be_trigger_px = entry_fill + sign * float(rg["breakeven_ratchet_r"]) * stop_dist

    return {
        "approved_by_risk_gate": True,
        "final_action": cand_dir,
        "stand_aside_reason": None,
        "entry_plan": {
            "reference_price": round(ref_px, 6),
            "assumed_fill_with_slippage": round(entry_fill, 6),
            "execution_assumption": "next_bar_open_or_visible_vwap_with_slippage",
        },
        "exit_plan": {
            "initial_stop_loss": round(stop_px, 6),
            "assumed_stop_fill": round(stop_fill_px, 6),
            "fee_adjusted_breakeven_stop": round(be_px, 6),
            "breakeven_ratchet_trigger_price": round(be_trigger_px, 6),
            "trailing_or_target_rule": synth_out.get("proposed_exit_rule"),
            "max_hold_hours": max_hold,
            "invalidation": synth_out.get("invalidation"),
        },
        "position_sizing": {
            "symbol": spec.symbol,
            "side": cand_dir,
            "qty_contracts": qty,
            "contract_size": spec.contract_size,
            "notional_usdt": round(notional, 4),
            "leverage": lev,
            "margin_mode": "isolated",
            "initial_margin_usdt": round(margin_usdt, 4),
            "free_cash_after_entry_usdt": round(free_cash_usdt - margin_usdt - entry_fee, 4),
        },
        "cost_and_risk": {
            "equity_usdt": round(equity_usdt, 4),
            "risk_budget_usdt": round(risk_budget_usdt, 4),
            "estimated_normal_stop_loss_usdt": round(est_stop_loss_usdt, 4),
            "estimated_normal_stop_loss_pct_equity": round(est_stop_loss_usdt / equity_usdt * 100.0, 4),
            "entry_taker_fee_usdt": round(entry_fee, 4),
            "stop_exit_taker_fee_usdt": round(stop_exit_fee, 4),
            "funding_reserve_1_interval_usdt": round(funding_reserve_usdt, 4),
            "estimated_isolated_liquidation": round(est_liq, 6),
            "stop_to_liquidation_buffer_pct": round(liq_buf_pct * 100.0, 3),
            "exact_liquidation_verified": False,
        },
    }


def run_symbol_decision_trace(
    bundle: dict[str, Any],
    *,
    equity_usdt: float,
    free_cash_usdt: float,
    open_positions_count: int = 0,
    kill_switch_halted: bool = False,
    planned_entry_price: float | None = None,
    slippage_override: float | None = None,
    spec_override: dict[str, Any] | None = None,
    enable_macro: bool = True,
    enable_forecast: bool = True,
    enable_critic_veto: bool = True,
    enable_live_external: bool = True,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute full isolated Researcher -> Critic -> Synthesizer -> Risk Gate for one symbol."""
    cfg = cfg or load_v2_config()
    researcher_out = run_independent_researcher(
        bundle,
        cfg=cfg,
        enable_macro=enable_macro,
        enable_forecast=enable_forecast,
        enable_live_external=enable_live_external,
    )
    critic_pass1 = run_independent_critic_pass1(bundle, cfg=cfg)
    critic_veto = run_critic_adversarial_veto(
        bundle,
        researcher_out,
        critic_pass1,
        cfg=cfg,
        enable_macro=enable_macro,
        enable_forecast=enable_forecast,
        enable_critic_veto=enable_critic_veto,
        enable_live_external=enable_live_external,
    )
    synth_out = run_synthesizer(bundle, researcher_out, critic_pass1, critic_veto)
    risk_out = evaluate_programmatic_risk(
        symbol=bundle["symbol"],
        synth_out=synth_out,
        market_pillar=bundle["pillars"]["market"],
        equity_usdt=equity_usdt,
        free_cash_usdt=free_cash_usdt,
        open_positions_count=open_positions_count,
        kill_switch_halted=kill_switch_halted,
        planned_entry_price=planned_entry_price,
        slippage_override=slippage_override,
        spec_override=spec_override,
        cfg=cfg,
    )

    det_id = "dec_" + canonical_sha256({
        "symbol": bundle["symbol"],
        "decision_utc": bundle["decision_utc"],
        "bundle_hash": bundle["bundle_hash"],
        "equity_usdt": round(equity_usdt, 6),
        "planned_entry": round(planned_entry_price, 6) if planned_entry_price is not None else None,
        "ablation": (enable_macro, enable_forecast, enable_critic_veto, enable_live_external),
    })[:24]

    trace = {
        "decision_id": det_id,
        "protocol_id": cfg["protocol_id"],
        "policy_version": cfg["version"],
        "symbol": bundle["symbol"],
        "decision_utc": bundle["decision_utc"],
        "ablation_flags": {
            "enable_macro": enable_macro,
            "enable_forecast": enable_forecast,
            "enable_critic_veto": enable_critic_veto,
            "enable_live_external": enable_live_external,
        },
        "final_action": risk_out["final_action"],
        "stand_aside_reason": risk_out["stand_aside_reason"],
        "evidence_bundle_hash": bundle["bundle_hash"],
        "pillar_hashes": {k: v["evidence_hash"] for k, v in bundle["pillars"].items()},
        "researcher": researcher_out,
        "critic_independent_pass1": critic_pass1,
        "critic_adversarial_pass2": critic_veto,
        "synthesizer": synth_out,
        "programmatic_risk_gate": risk_out,
        "live_orders_submitted": False,
        "order_permissions": False,
    }
    trace["trace_hash"] = canonical_sha256(trace)
    return trace
