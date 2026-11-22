"""Deterministic isolated-margin paper checks using a public tier snapshot.

Applies only to classic isolated USDT-M paper. Cross, UTA, and multi-asset
modes are explicitly unsupported. The official isolated classic formula is
used with maintenance-margin deduction / pre-calculated offset set to zero
because that term is not numerically specified in public docs. The result is
an estimate, not an exchange-validated liquidation price.

Sources:
- Isolated classic formula: https://www.bitget.com/support/articles/12560603895826
- Estimated isolated formula (same structure, "pre-calculated offset"):
  https://www.bitget.com/support/articles/12560603808759
- Trigger is mark price, not last traded price.

This module never raises configured leverage, never changes the paper kill
switch, and never treats current tiers as historically known.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from typing import Any

from piia.intelligence.public_position_tiers import conservative_tier_for_notional, max_notional_for_leverage

FORMULA_ID = "bitget_classic_isolated_estimated_offset_assumed_zero"
FORMULA = (
    "estimated_liq = (position_margin + maintenance_margin_deduction "
    "- position_size * average_entry * direction) / "
    "(position_size * (MMR + taker - direction))"
)


def _dec(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value))
    except Exception:
        return None
    return number if number.is_finite() else None


def conservative_tick(price: Decimal, *, side: str, price_place: Any) -> tuple[Decimal, str]:
    """Round the estimate against the trader: long up, short down, when tick is known."""
    if side not in {"long", "short"}:
        return price, "wrong_side"
    if price_place in (None, ""):
        return price, "tick_size_unknown_unrounded"
    try:
        places = int(str(price_place))
    except (TypeError, ValueError):
        return price, "tick_size_unknown_unrounded"
    if places < 0 or places > 12:
        return price, "tick_size_unknown_unrounded"
    quant = Decimal("1").scaleb(-places)
    rounding = ROUND_CEILING if side == "long" else ROUND_FLOOR
    return price.quantize(quant, rounding=rounding), "conservative_tick_applied"


def estimated_isolated_liquidation(
    *,
    side: str,
    qty: Decimal,
    entry: Decimal,
    leverage: Decimal,
    mmr: Decimal,
    taker: Decimal,
    contract_size: Decimal = Decimal("1"),
    deduction: Decimal = Decimal("0"),
    price_place: Any = None,
    deduction_known: bool = False,
) -> dict[str, Any]:
    if side not in {"long", "short"}:
        return {"ok": False, "reason": "wrong_side", "exact_liquidation_verified": False}
    if min(qty, entry, leverage, contract_size) <= 0 or mmr <= 0 or mmr >= 1 or taker < 0:
        return {"ok": False, "reason": "invalid_inputs", "exact_liquidation_verified": False}
    if not deduction_known and deduction != 0:
        return {"ok": False, "reason": "undocumented_deduction_value", "exact_liquidation_verified": False}
    direction = Decimal("1") if side == "long" else Decimal("-1")
    notional = qty * entry * contract_size
    margin = notional / leverage
    size = qty * contract_size
    denom = size * (mmr + taker - direction)
    if denom == 0:
        return {"ok": False, "reason": "liquidation_denominator_zero", "exact_liquidation_verified": False}
    raw = (margin + deduction - size * entry * direction) / denom
    if raw <= 0 or not raw.is_finite():
        return {"ok": False, "reason": "non_positive_estimated_liquidation", "exact_liquidation_verified": False}
    rounded, tick_reason = conservative_tick(raw, side=side, price_place=price_place)
    # Conservative means closer to entry (less room).
    if side == "long":
        estimate = max(raw, rounded)
        if not (Decimal("0") < estimate < entry):
            return {"ok": False, "reason": "estimated_liquidation_not_below_entry", "exact_liquidation_verified": False}
    else:
        estimate = min(raw, rounded) if tick_reason == "conservative_tick_applied" else raw
        if not (estimate > entry):
            return {"ok": False, "reason": "estimated_liquidation_not_above_entry", "exact_liquidation_verified": False}
    return {
        "ok": True,
        "reason": None,
        "formula_id": FORMULA_ID,
        "formula": FORMULA,
        "estimated_liquidation_price": str(estimate),
        "unrounded_price": str(raw),
        "tick_reason": tick_reason,
        "position_margin": str(margin),
        "notional": str(notional),
        "mmr": str(mmr),
        "taker": str(taker),
        "maintenance_margin_deduction_assumed": str(deduction),
        "deduction_publicly_specified": False,
        "trigger": "mark_price_not_last_traded_price",
        "exact_liquidation_verified": False,
        "exchange_production_liquidation_validated": False,
        "algebra_matches_published_isolated_classic_formula_with_zero_deduction": True,
        "residual_unknowns": [
            "maintenance_margin_deduction_or_precalculated_offset",
            "insurance_fund_and_liquidation_fees",
            "account_class_classic_vs_uta",
            "accrued_funding_and_fee_deductions_on_a_live_account",
            "partial_deleveraging_process",
        ],
    }


def stop_before_estimated_liq(side: str, stop: Decimal, estimated_liq: Decimal) -> dict[str, Any]:
    if side == "long":
        ok = stop > estimated_liq
    elif side == "short":
        ok = stop < estimated_liq
    else:
        return {"ok": False, "reason": "wrong_side"}
    return {
        "ok": ok,
        "reason": None if ok else "stop_at_or_beyond_estimated_liquidation",
        "gap_past_stop_remains_a_risk": True,
        "buffer_source": "stop_must_be_strictly_before_estimated_liq_no_invented_numeric_buffer",
    }


def apply_isolated_tier_constraints(
    *,
    side: str,
    qty: Decimal,
    entry: Decimal,
    stop: Decimal,
    leverage: Decimal,
    taker: Decimal,
    equity: Decimal,
    cash_floor: Decimal,
    max_margin_fraction: Decimal,
    contract_size: Decimal,
    qty_step: Decimal,
    min_qty: Decimal,
    min_notional: Decimal,
    tiers: dict[str, Any],
    funding_rate: Decimal | None,
    funding_rate_known: bool,
    price_place: Any = None,
    margin_mode: str = "isolated",
    mark: Decimal | None = None,
) -> dict[str, Any]:
    """Size downward to satisfy tiers, IM+fees+funding reserve, and stop-before-liq.

    Never raises leverage. Cross margin is rejected. Current tiers are contemporaneous.
    """
    result: dict[str, Any] = {
        "ok": False,
        "qty": str(qty),
        "exact_liquidation_verified": False,
        "margin_tiers_supported": bool(tiers.get("ok")),
        "margin_mode_supported": margin_mode == "isolated",
        "sized_downward": False,
        "leverage_raised": False,
    }
    if margin_mode != "isolated":
        result["reason"] = "cross_or_unsupported_margin_mode"
        return result
    if side not in {"long", "short"}:
        result["reason"] = "wrong_side"
        return result
    if not tiers.get("ok"):
        result["reason"] = tiers.get("reason") or "margin_tiers_unavailable"
        result["margin_tiers_supported"] = False
        return result
    if mark is not None and entry is not None and mark <= 0:
        result["reason"] = "invalid_mark"
        return result
    # Last traded price is never used as a liquidation trigger.
    result["liquidation_trigger"] = "mark_price_not_last_traded_price"

    cap = max_notional_for_leverage(tiers, leverage)
    if not cap.get("ok"):
        result["reason"] = cap.get("reason")
        return result
    max_notional = Decimal(cap["max_notional"])
    working = qty
    notional = working * entry * contract_size
    from piia.intelligence.public_orderbook import floor_qty
    picked = conservative_tier_for_notional(tiers, notional)
    needs_shrink = (notional > max_notional) or (picked.get("ok") and Decimal(picked["tier"]["max_leverage"]) < leverage) or (not picked.get("ok"))
    if needs_shrink:
        stepped = floor_qty(max_notional / (entry * contract_size), qty_step)
        while stepped >= min_qty:
            candidate_notional = stepped * entry * contract_size
            trial = conservative_tier_for_notional(tiers, candidate_notional)
            if trial.get("ok") and Decimal(trial["tier"]["max_leverage"]) >= leverage:
                break
            stepped = floor_qty(stepped - qty_step, qty_step) if stepped > qty_step else Decimal("0")
        working = stepped
        result["sized_downward"] = working < qty
        notional = working * entry * contract_size
    if working < min_qty or notional < min_notional:
        result.update({"reason": "tier_leverage_cap_reduces_below_minimum", "qty": str(working)})
        return result

    picked = conservative_tier_for_notional(tiers, notional)
    if not picked.get("ok"):
        result["reason"] = picked.get("reason")
        return result
    if Decimal(picked["tier"]["max_leverage"]) < leverage:
        result["reason"] = "leverage_exceeds_tier_cap"
        return result
    mmr = Decimal(picked["tier"]["keep_margin_rate"])
    funding_reserve = Decimal("0")
    if funding_rate_known and funding_rate is not None:
        # One adverse funding interval as a reserve, not a forecast.
        funding_reserve = abs(funding_rate) * notional
    elif not funding_rate_known:
        result["funding_reserve_unknown"] = True
    open_fee = notional * taker
    initial_margin = notional / leverage
    required = initial_margin + open_fee + funding_reserve
    available = equity * (Decimal("1") - cash_floor)
    margin_cap = equity * max_margin_fraction
    if required > min(available, margin_cap):
        # Size downward to fit IM + fees + funding reserve.
        from piia.intelligence.public_orderbook import floor_qty
        # required = notional/L + notional*taker + abs(funding)*notional = notional * coeff
        coeff = (Decimal("1") / leverage) + taker + (abs(funding_rate) if (funding_rate_known and funding_rate is not None) else Decimal("0"))
        if coeff <= 0:
            result["reason"] = "invalid_margin_coefficient"
            return result
        max_fit = min(available, margin_cap) / coeff
        stepped = floor_qty(max_fit / (entry * contract_size), qty_step)
        if stepped < working:
            working = stepped
            result["sized_downward"] = True
            notional = working * entry * contract_size
            funding_reserve = abs(funding_rate) * notional if (funding_rate_known and funding_rate is not None) else Decimal("0")
            open_fee = notional * taker
            initial_margin = notional / leverage
            required = initial_margin + open_fee + funding_reserve
        if working < min_qty or notional < min_notional or required > min(available, margin_cap):
            result.update({"reason": "initial_margin_fee_or_funding_reserve_exceeds_budget", "qty": str(working)})
            return result
        picked = conservative_tier_for_notional(tiers, notional)
        if not picked.get("ok"):
            result["reason"] = picked.get("reason")
            return result
        mmr = Decimal(picked["tier"]["keep_margin_rate"])

    liq = estimated_isolated_liquidation(
        side=side, qty=working, entry=entry, leverage=leverage, mmr=mmr, taker=taker,
        contract_size=contract_size, price_place=price_place,
    )
    if not liq.get("ok"):
        result["reason"] = liq.get("reason")
        result["liquidation"] = liq
        return result
    estimated = Decimal(liq["estimated_liquidation_price"])
    stop_check = stop_before_estimated_liq(side, stop, estimated)
    if not stop_check.get("ok"):
        result["reason"] = stop_check.get("reason")
        result["liquidation"] = liq
        result["stop_check"] = stop_check
        return result
    result.update({
        "ok": True,
        "reason": None,
        "qty": str(working),
        "notional": str(notional),
        "initial_margin": str(initial_margin),
        "open_taker_fee": str(open_fee),
        "funding_reserve": str(funding_reserve),
        "funding_reserve_is_not_a_forecast": True,
        "required_margin_plus_fees_and_reserve": str(required),
        "tier": picked["tier"],
        "tier_selection": picked.get("reason"),
        "liquidation": liq,
        "stop_check": stop_check,
        "margin_tiers_supported": True,
        "exact_liquidation_verified": False,
    })
    return result
