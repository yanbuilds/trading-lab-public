"""Trader-oriented Risk & Trade-Ticket Engine v2 for USDT-M Linear Perpetuals.

Extends v1 `trading_lab.perps.risk` (which remains frozen for audit reproducibility)
with flexible trader parameters (custom equity, leverage 1..10x, risk 0.1%..3%,
limit/maker vs market/taker fees), exact fee-adjusted breakeven prices, net R:R
take-profit ladders, Bitget isolated liquidation buffer checks, and visible
orderbook VWAP slippage verification. Never submits live orders.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math
from pathlib import Path
from typing import Any
import yaml

from trading_lab.perps.risk import Spec, floor_step

POLICY_V2_PATH = Path(__file__).resolve().parents[3] / "config" / "trading_policy_v2.yaml"

DEFAULT_SPECS: dict[str, dict[str, float]] = {
    "BTCUSDT": {
        "min_qty": 0.0001,
        "qty_step": 0.0001,
        "min_notional": 5.0,
        "taker": 0.0006,
        "maker": 0.0002,
        "contract_size": 1.0,
        "mmr": 0.0040,
        "max_leverage_venue": 150.0,
        "price_place": 1,
    },
    "XAUUSDT": {
        "min_qty": 0.01,
        "qty_step": 0.01,
        "min_notional": 5.0,
        "taker": 0.0006,
        "maker": 0.0002,
        "contract_size": 1.0,
        "mmr": 0.0050,
        "max_leverage_venue": 100.0,
        "price_place": 2,
    },
    "NDX100USDT": {
        "min_qty": 0.0001,
        "qty_step": 0.0001,
        "min_notional": 5.0,
        "taker": 0.0006,
        "maker": 0.0002,
        "contract_size": 1.0,
        "mmr": 0.0100,
        "max_leverage_venue": 50.0,
        "price_place": 1,
    },
}


@dataclass(frozen=True)
class ExtendedSpec:
    symbol: str
    min_qty: float
    qty_step: float
    min_notional: float
    taker: float = 0.0006
    maker: float = 0.0002
    contract_size: float = 1.0
    mmr: float = 0.0050
    max_leverage_venue: float = 50.0
    price_place: int | None = None

    def to_v1_spec(self) -> Spec:
        return Spec(
            symbol=self.symbol,
            min_qty=self.min_qty,
            qty_step=self.qty_step,
            min_notional=self.min_notional,
            taker=self.taker,
            contract_size=self.contract_size,
        )


def load_policy_v2(path: Path | None = None) -> dict[str, Any]:
    target = path or POLICY_V2_PATH
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("policy_id") != "regime_adaptive_v2":
        raise ValueError("Invalid trading_policy_v2.yaml")
    return data


def resolve_spec(symbol: str, contract_override: dict[str, Any] | None = None) -> ExtendedSpec:
    clean = symbol.split("/")[0].upper() + ("USDT" if not symbol.upper().endswith("USDT") and "/" in symbol else "")
    if clean not in DEFAULT_SPECS and symbol.upper() in DEFAULT_SPECS:
        clean = symbol.upper()
    base = dict(DEFAULT_SPECS.get(clean, {
        "min_qty": 0.001,
        "qty_step": 0.001,
        "min_notional": 5.0,
        "taker": 0.0006,
        "maker": 0.0002,
        "contract_size": 1.0,
        "mmr": 0.0100,
        "max_leverage_venue": 25.0,
        "price_place": 2,
    }))
    if contract_override:
        for k in ("min_qty", "qty_step", "min_notional", "taker", "maker", "contract_size", "mmr", "max_leverage_venue"):
            val = contract_override.get(k)
            if val is not None and not isinstance(val, bool):
                try:
                    fval = float(val)
                    if math.isfinite(fval) and fval >= 0:
                        base[k] = fval
                except (TypeError, ValueError):
                    pass
        if contract_override.get("price_place") not in (None, ""):
            try:
                base["price_place"] = int(str(contract_override["price_place"]))
            except (TypeError, ValueError):
                pass
    return ExtendedSpec(
        symbol=clean,
        min_qty=float(base["min_qty"]),
        qty_step=float(base["qty_step"]),
        min_notional=float(base["min_notional"]),
        taker=float(base["taker"]),
        maker=float(base["maker"]),
        contract_size=float(base["contract_size"]),
        mmr=float(base["mmr"]),
        max_leverage_venue=float(base["max_leverage_venue"]),
        price_place=base.get("price_place"),  # type: ignore[arg-type]
    )


def fee_adjusted_breakeven(
    entry_fill: float,
    side: str,
    *,
    entry_fee_rate: float,
    exit_fee_rate: float,
) -> float:
    """Exact exit price at which net PnL after entry & exit fees equals zero."""
    if side == "long":
        return entry_fill * (1.0 + entry_fee_rate) / (1.0 - exit_fee_rate)
    if side == "short":
        return entry_fill * (1.0 - entry_fee_rate) / (1.0 + exit_fee_rate)
    raise ValueError("side must be 'long' or 'short'")


def estimate_isolated_liq_v2(
    *,
    side: str,
    qty: float,
    entry_fill: float,
    leverage: float,
    mmr: float,
    taker: float,
    contract_size: float = 1.0,
) -> float | None:
    """Classic isolated USDT-M liquidation estimate with zero deduction offset."""
    if qty <= 0 or entry_fill <= 0 or leverage <= 0 or contract_size <= 0 or not (0 < mmr < 1):
        return None
    direction = 1.0 if side == "long" else -1.0
    size = qty * contract_size
    notional = size * entry_fill
    margin = notional / leverage
    denom = size * (mmr + taker - direction)
    if abs(denom) < 1e-15:
        return None
    raw = (margin - size * entry_fill * direction) / denom
    if not math.isfinite(raw) or raw <= 0:
        return None
    if side == "long" and raw >= entry_fill:
        return None
    if side == "short" and raw <= entry_fill:
        return None
    return float(raw)


def _evaluate_tp_target(
    *,
    label: str,
    target_price: float,
    side: str,
    qty: float,
    entry_fill: float,
    entry_fee_rate: float,
    tp_fee_rate: float,
    contract_size: float,
    stop_risk_usdt: float,
    equity: float,
) -> dict[str, Any]:
    sign = 1.0 if side == "long" else -1.0
    units = qty * contract_size
    gross = sign * (target_price - entry_fill) * units
    fees = (entry_fill * entry_fee_rate + target_price * tp_fee_rate) * units
    net = gross - fees
    net_rr = (net / stop_risk_usdt) if stop_risk_usdt > 0 else None
    req_wr = (100.0 / (1.0 + net_rr)) if (net_rr is not None and net_rr > 0) else None
    return {
        "label": label,
        "target_price": round(target_price, 6),
        "move_pct": round((target_price / entry_fill - 1.0) * 100.0 * sign, 4),
        "gross_pnl_usdt": round(gross, 4),
        "net_pnl_usdt": round(net, 4),
        "return_on_equity_pct": round((net / equity) * 100.0, 4) if equity > 0 else None,
        "net_reward_risk": round(net_rr, 3) if net_rr is not None else None,
        "breakeven_winrate_pct": round(req_wr, 2) if req_wr is not None else None,
    }


def compute_trade_ticket(
    *,
    symbol: str,
    side: str,
    equity_usdt: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float | None = None,
    leverage: float | None = None,
    risk_fraction: float | None = None,
    order_type: str = "market",
    free_cash_usdt: float | None = None,
    funding_rate: float | None = None,
    orderbook: dict[str, Any] | None = None,
    tiers: dict[str, Any] | None = None,
    contract_override: dict[str, Any] | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute a complete pre-trade risk ticket, TP ladder, liquidation buffer & checklist."""
    cfg = policy or load_policy_v2()
    rcfg = cfg["risk"]
    side = side.lower().strip()
    if side not in {"long", "short"}:
        raise ValueError("side must be 'long' or 'short'")
    order_type = order_type.lower().strip()
    if order_type not in {"market", "limit"}:
        raise ValueError("order_type must be 'market' or 'limit'")

    lev = float(rcfg["default_leverage"] if leverage is None else leverage)
    r_frac = float(rcfg["risk_fraction_per_trade"] if risk_fraction is None else risk_fraction)
    free_cash = float(equity_usdt if free_cash_usdt is None else free_cash_usdt)
    slip = float(rcfg["slippage_each_side"]) if order_type == "market" else 0.0
    stop_slip = float(rcfg["slippage_each_side"])
    cash_floor = float(rcfg["cash_floor"])
    max_margin_frac = float(rcfg["max_margin_fraction"])
    max_lev_cap = float(rcfg["max_leverage_cap"])
    max_risk_cap = float(rcfg["max_risk_fraction_cap"])
    min_liq_buf = float(rcfg["min_liquidation_buffer_pct"])
    min_rr = float(rcfg["min_reward_risk_ratio"])

    spec = resolve_spec(symbol, contract_override)
    checklist: list[dict[str, Any]] = []
    warnings: list[str] = []
    reject_reasons: list[str] = []

    nums = (equity_usdt, free_cash, entry_price, stop_loss, lev, r_frac)
    if not all(math.isfinite(x) for x in nums) or min(equity_usdt, entry_price, stop_loss, lev, r_frac) <= 0 or free_cash < 0:
        return {
            "verdict": "REJECTED",
            "actionable_for_manual_review": False,
            "reasons": ["invalid_nonpositive_inputs"],
            "order_permissions": False,
        }

    if side == "long" and stop_loss >= entry_price:
        reject_reasons.append("long_stop_must_be_below_entry")
    if side == "short" and stop_loss <= entry_price:
        reject_reasons.append("short_stop_must_be_above_entry")
    if not (1.0 <= lev <= min(max_lev_cap, spec.max_leverage_venue)):
        reject_reasons.append(f"leverage_outside_cap_1_to_{min(max_lev_cap, spec.max_leverage_venue):g}")
    if r_frac > max_risk_cap + 1e-12:
        reject_reasons.append(f"risk_fraction_exceeds_cap_{max_risk_cap:.2%}")
    if take_profit is not None:
        if not math.isfinite(take_profit) or take_profit <= 0:
            reject_reasons.append("invalid_take_profit")
        elif side == "long" and take_profit <= entry_price:
            reject_reasons.append("long_take_profit_must_be_above_entry")
        elif side == "short" and take_profit >= entry_price:
            reject_reasons.append("short_take_profit_must_be_below_entry")

    if reject_reasons:
        return {
            "symbol": spec.symbol,
            "side": side,
            "verdict": "REJECTED",
            "actionable_for_manual_review": False,
            "reasons": reject_reasons,
            "order_permissions": False,
        }

    sign = 1.0 if side == "long" else -1.0
    entry_fee_rate = spec.maker if order_type == "limit" else spec.taker
    exit_fee_rate = spec.taker
    tp_fee_rate = spec.maker

    entry_fill = entry_price * (1.0 + sign * slip)
    stop_fill_px = stop_loss * (1.0 - sign * stop_slip)
    loss_per_contract = spec.contract_size * (
        sign * (entry_fill - stop_fill_px)
        + entry_fee_rate * entry_fill
        + exit_fee_rate * stop_fill_px
    )
    if not math.isfinite(loss_per_contract) or loss_per_contract <= 0:
        return {
            "symbol": spec.symbol,
            "side": side,
            "verdict": "REJECTED",
            "actionable_for_manual_review": False,
            "reasons": ["invalid_stop_loss_distance"],
            "order_permissions": False,
        }

    risk_budget_usdt = equity_usdt * r_frac
    qty_by_risk = risk_budget_usdt / loss_per_contract

    available_margin_cash = min(
        equity_usdt * max_margin_frac,
        max(0.0, free_cash - equity_usdt * cash_floor),
    )
    funding_reserve_rate = abs(float(funding_rate)) if (funding_rate is not None and math.isfinite(float(funding_rate))) else 0.0
    unit_cash_req = spec.contract_size * entry_fill * ((1.0 / lev) + entry_fee_rate + funding_reserve_rate)
    qty_by_cash = available_margin_cash / unit_cash_req if unit_cash_req > 0 else 0.0

    raw_qty = min(qty_by_risk, qty_by_cash)
    qty = floor_step(max(0.0, raw_qty), spec.qty_step)
    sizing_cap_driver = "stop_risk_budget" if qty_by_risk <= qty_by_cash else "margin_and_cash_floor"

    # Tier check (live tiers if supplied, otherwise fallback tier1 MMR)
    mmr = spec.mmr
    tier_source = "fallback_tier1_mmr"
    if tiers and tiers.get("ok"):
        from piia.intelligence.public_position_tiers import conservative_tier_for_notional, max_notional_for_leverage
        cap_info = max_notional_for_leverage(tiers, Decimal(str(lev)))
        if not cap_info.get("ok"):
            reject_reasons.append(str(cap_info.get("reason") or "leverage_not_permitted_by_tier"))
        else:
            max_tier_notional = float(cap_info["max_notional"])
            if qty * entry_fill * spec.contract_size > max_tier_notional:
                qty = floor_step(max_tier_notional / (entry_fill * spec.contract_size), spec.qty_step)
                sizing_cap_driver = "venue_position_tier_cap"
            picked = conservative_tier_for_notional(tiers, Decimal(str(qty * entry_fill * spec.contract_size)))
            if picked.get("ok"):
                mmr = float(picked["tier"]["keep_margin_rate"])
                tier_source = "live_public_tiers"
                if float(picked["tier"]["max_leverage"]) < lev:
                    reject_reasons.append("leverage_exceeds_selected_tier_cap")

    notional = qty * entry_fill * spec.contract_size
    if qty < spec.min_qty or notional < spec.min_notional:
        # Compute minimum equity needed at this stop distance & risk % so the trader gets actionable guidance!
        min_contracts_needed = max(spec.min_qty, math.ceil((spec.min_notional / (entry_fill * spec.contract_size)) / spec.qty_step) * spec.qty_step)
        min_stop_loss_usdt = min_contracts_needed * loss_per_contract
        min_equity_for_1pct = min_stop_loss_usdt / r_frac if r_frac > 0 else None
        reject_reasons.append("minimum_order_exceeds_budget")
        checklist.append({
            "item": "最小下单量与风险预算匹配",
            "passed": False,
            "detail": (
                f"最小可下数量 {min_contracts_needed:g} 张（名义约 {min_contracts_needed * entry_fill * spec.contract_size:.2f} USDT）"
                f"对应止损亏损约 {min_stop_loss_usdt:.2f} USDT，超出当前单笔风险预算 {risk_budget_usdt:.2f} USDT"
                + (f"（按 {r_frac:.2%} 风险计至少需约 {min_equity_for_1pct:.0f} USDT 权益，或收窄止损距离）。" if min_equity_for_1pct else "。")
            ),
        })
    else:
        checklist.append({
            "item": "最小下单量与风险预算匹配",
            "passed": True,
            "detail": f"建议下单 {qty:g} 张（受限于 {sizing_cap_driver}），名义价值 {notional:.2f} USDT。",
        })

    est_liq = estimate_isolated_liq_v2(
        side=side,
        qty=qty if qty > 0 else spec.min_qty,
        entry_fill=entry_fill,
        leverage=lev,
        mmr=mmr,
        taker=spec.taker,
        contract_size=spec.contract_size,
    )
    liq_buffer_pct = None
    if est_liq is not None:
        if (side == "long" and stop_loss <= est_liq) or (side == "short" and stop_loss >= est_liq):
            reject_reasons.append("stop_at_or_beyond_estimated_liquidation")
            checklist.append({
                "item": "止损严于估算逐仓强平价",
                "passed": False,
                "detail": f"止损价 {stop_loss:.4f} 晚于估算强平价 {est_liq:.4f}，请降低杠杆或收紧止损！",
            })
        else:
            liq_buffer_pct = abs(stop_loss - est_liq) / entry_fill
            buf_ok = liq_buffer_pct >= min_liq_buf
            if not buf_ok:
                warnings.append(f"止损距估算强平价仅 {liq_buffer_pct:.2%}（建议至少 {min_liq_buf:.2%} 缓冲以防插针）")
            checklist.append({
                "item": "止损严于估算逐仓强平价",
                "passed": buf_ok,
                "detail": f"估算逐仓强平价 {est_liq:.4f}，止损缓冲距离 {liq_buffer_pct:.2%}（{tier_source}）。",
            })

    # Orderbook execution check when live book is supplied
    book_execution = None
    if orderbook and orderbook.get("ok") and qty > 0:
        from piia.intelligence.public_orderbook import walk_book
        levels = orderbook["asks"] if side == "long" else orderbook["bids"]
        walk = walk_book(levels, Decimal(str(qty)))
        if not walk.get("ok"):
            warnings.append(f"可见盘口深度不足：{walk.get('reason')}")
            book_execution = {"ok": False, "reason": walk.get("reason"), "filled_qty": walk.get("filled_qty")}
            checklist.append({
                "item": "盘口可见深度核验",
                "passed": False,
                "detail": f"前排可见深度不足以一次性吃单 {qty:g} 张，建议使用限价单或分批执行。",
            })
        else:
            vwap = float(walk["vwap"])
            mid = float(orderbook["mid"]) if orderbook.get("mid") else entry_price
            eff_slip_bps = abs(vwap - mid) / mid * 10_000.0 if mid > 0 else 0.0
            max_bps = float(rcfg.get("max_book_slippage_bps", 15.0))
            slip_ok = eff_slip_bps <= max_bps
            if not slip_ok:
                warnings.append(f"吃单预估滑点 {eff_slip_bps:.1f} bps 偏高（阈值 {max_bps:.1f} bps），建议改用限价单")
            book_execution = {
                "ok": True,
                "vwap": round(vwap, 6),
                "mid": round(mid, 6),
                "spread": orderbook.get("spread"),
                "effective_slippage_bps_vs_mid": round(eff_slip_bps, 2),
                "levels_consumed": len(walk.get("fills") or []),
            }
            checklist.append({
                "item": "盘口可见深度核验",
                "passed": slip_ok,
                "detail": f"可见盘口吃单 VWAP={vwap:.4f}，相对盘口中枢滑点约 {eff_slip_bps:.2f} bps。",
            })

    initial_margin = notional / lev if qty > 0 else 0.0
    entry_fee_usdt = notional * entry_fee_rate if qty > 0 else 0.0
    stop_exit_fee_usdt = (qty * stop_fill_px * spec.contract_size * exit_fee_rate) if qty > 0 else 0.0
    funding_reserve_usdt = notional * funding_reserve_rate if qty > 0 else 0.0
    actual_stop_loss_usdt = qty * loss_per_contract if qty > 0 else 0.0
    breakeven_px = fee_adjusted_breakeven(
        entry_fill, side, entry_fee_rate=entry_fee_rate, exit_fee_rate=tp_fee_rate
    )
    breakeven_market_px = fee_adjusted_breakeven(
        entry_fill, side, entry_fee_rate=entry_fee_rate, exit_fee_rate=exit_fee_rate
    )

    # Price distance per 1R
    stop_dist = abs(entry_price - stop_loss)
    tp_ladder: list[dict[str, Any]] = []
    if qty > 0:
        for r_mult in rcfg.get("tp_ladder_r", [1.5, 2.5, 3.5]):
            tp_px = entry_price + sign * float(r_mult) * stop_dist
            if tp_px > 0:
                tp_ladder.append(
                    _evaluate_tp_target(
                        label=f"TP ({r_mult:g}R)",
                        target_price=tp_px,
                        side=side,
                        qty=qty,
                        entry_fill=entry_fill,
                        entry_fee_rate=entry_fee_rate,
                        tp_fee_rate=tp_fee_rate,
                        contract_size=spec.contract_size,
                        stop_risk_usdt=actual_stop_loss_usdt,
                        equity=equity_usdt,
                    )
                )

    custom_tp = None
    if take_profit is not None and qty > 0:
        custom_tp = _evaluate_tp_target(
            label="Custom TP",
            target_price=take_profit,
            side=side,
            qty=qty,
            entry_fill=entry_fill,
            entry_fee_rate=entry_fee_rate,
            tp_fee_rate=tp_fee_rate,
            contract_size=spec.contract_size,
            stop_risk_usdt=actual_stop_loss_usdt,
            equity=equity_usdt,
        )
        net_rr = custom_tp.get("net_reward_risk")
        rr_ok = net_rr is not None and net_rr >= min_rr
        if not rr_ok:
            warnings.append(f"目标止盈扣除手续费/滑点后的净盈亏比为 {net_rr or 0:.2f}R，低于建议门槛 {min_rr:.2f}R")
        checklist.append({
            "item": f"净盈亏比 ≥ {min_rr:g}R",
            "passed": rr_ok,
            "detail": (
                f"目标价 {take_profit:.4f} 预计净盈利 {custom_tp['net_pnl_usdt']:.2f} USDT "
                f"（净盈亏比 {net_rr:.2f}R，盈亏平衡胜率 {custom_tp['breakeven_winrate_pct']:.1f}%）。"
                if net_rr is not None
                else "目标价净收益为负或不可计算。"
            ),
        })

    # +1.5R breakeven management trigger price
    be_trigger_r = float(rcfg.get("breakeven_trigger_r", 1.5))
    move_to_be_at_price = entry_price + sign * be_trigger_r * stop_dist

    if reject_reasons:
        verdict = "REJECTED"
    elif warnings:
        verdict = "CAUTION"
    else:
        verdict = "READY_FOR_MANUAL_REVIEW"

    return {
        "kind": "pre_trade_decision_ticket_v2",
        "symbol": spec.symbol,
        "side": side,
        "order_type": order_type,
        "verdict": verdict,
        "actionable_for_manual_review": verdict in {"READY_FOR_MANUAL_REVIEW", "CAUTION"} and qty > 0,
        "reasons": reject_reasons,
        "warnings": warnings,
        "sizing": {
            "qty_contracts": qty,
            "qty_step": spec.qty_step,
            "min_qty": spec.min_qty,
            "min_notional_usdt": spec.min_notional,
            "contract_size": spec.contract_size,
            "notional_usdt": round(notional, 4),
            "leverage": lev,
            "effective_account_leverage": round(notional / equity_usdt, 3) if equity_usdt > 0 else 0.0,
            "initial_margin_usdt": round(initial_margin, 4),
            "free_cash_after_entry_usdt": round(free_cash - initial_margin - entry_fee_usdt, 4) if qty > 0 else round(free_cash, 4),
            "sizing_cap_driver": sizing_cap_driver,
        },
        "prices": {
            "planned_entry": entry_price,
            "assumed_entry_fill": round(entry_fill, 6),
            "stop_loss": stop_loss,
            "assumed_stop_fill": round(stop_fill_px, 6),
            "stop_distance_pct": round(stop_dist / entry_price * 100.0, 4),
            "breakeven_limit_exit": round(breakeven_px, 6),
            "breakeven_market_exit": round(breakeven_market_px, 6),
            "move_stop_to_breakeven_trigger_price": round(move_to_be_at_price, 6),
            "estimated_isolated_liquidation": round(est_liq, 6) if est_liq is not None else None,
            "stop_to_liquidation_buffer_pct": round(liq_buffer_pct * 100.0, 3) if liq_buffer_pct is not None else None,
            "tier_mmr_used": mmr,
            "tier_source": tier_source,
        },
        "risk_and_costs": {
            "equity_usdt": equity_usdt,
            "risk_budget_pct": round(r_frac * 100.0, 3),
            "risk_budget_usdt": round(risk_budget_usdt, 4),
            "estimated_stop_loss_usdt": round(actual_stop_loss_usdt, 4),
            "estimated_stop_loss_pct_equity": round(actual_stop_loss_usdt / equity_usdt * 100.0, 4) if equity_usdt > 0 else 0.0,
            "entry_fee_usdt": round(entry_fee_usdt, 4),
            "stop_exit_fee_usdt": round(stop_exit_fee_usdt, 4),
            "funding_reserve_1_interval_usdt": round(funding_reserve_usdt, 4),
        },
        "take_profit_custom": custom_tp,
        "take_profit_ladder": tp_ladder,
        "orderbook_execution": book_execution,
        "checklist": checklist,
        "exact_liquidation_verified": False,
        "order_permissions": False,
        "disclaimer": (
            "本交易工单仅作人工下单前风控核对与本地模拟盘辅助，不自动向交易所发送订单。"
            "逐仓强平价基于公开维持保证金率估算（未含隐藏抵扣项），极端跳空可能越过止损。"
        ),
    }
