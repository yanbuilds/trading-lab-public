"""Read-only arithmetic scenarios for user-supplied linear USDT contracts.

This does not model venue liquidation, margin tiers, funding, or execution.
All prices and assumptions come from the caller; no trade is recommended.
"""
from __future__ import annotations

import math


def portfolio_stress(*, equity: float, positions: list[dict]) -> dict:
    """User-defined simultaneous price scenario for linear contracts, not a margin model."""
    def finite(value, *, positive=False):
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or (value <= 0 if positive else value < 0)):
            raise ValueError("组合输入必须是有效有限数字")
        return float(value)

    equity = finite(equity, positive=True)
    if not isinstance(positions, list) or not 1 <= len(positions) <= 20:
        raise ValueError("请提供 1 至 20 个假设持仓")
    results, notionals, gross, net, total_cost = [], {}, 0.0, 0.0, 0.0
    for position in positions:
        if not isinstance(position, dict):
            raise ValueError("持仓格式无效")
        try:
            symbol, side = position["symbol"], position["side"]
            qty = finite(position["quantity"], positive=True)
            size = finite(position["contract_size"], positive=True)
            mark = finite(position["mark_price"], positive=True)
            stressed = finite(position["stress_price"], positive=True)
            fee = finite(position["exit_fee_rate"])
            slip = finite(position["slippage_rate"])
        except KeyError as exc:
            raise ValueError("持仓缺少必填字段") from exc
        if not isinstance(symbol, str) or not symbol.strip() or side not in {"long", "short"} or fee >= 1 or slip >= 1:
            raise ValueError("标的、方向、手续费或滑点无效")
        symbol = symbol.strip().upper()
        sign = 1 if side == "long" else -1
        units = qty * size
        fill = stressed * (1 - sign * slip)
        notional = units * mark
        mark_pnl = sign * (stressed - mark) * units
        slipped_pnl = sign * (fill - mark) * units
        exit_fee = fill * units * fee
        component = slipped_pnl - exit_fee
        if not all(math.isfinite(x) for x in (units, fill, notional, mark_pnl, slipped_pnl, exit_fee, component)):
            raise ValueError("组合输入数值过大")
        gross += notional
        net += component
        total_cost += mark_pnl - component
        notionals[symbol] = notionals.get(symbol, 0.0) + notional
        results.append({"symbol": symbol, "side": side, "current_notional_usdt": notional,
                        "assumed_stress_price": stressed, "assumed_fill_after_slippage": fill,
                        "gross_pnl_before_slippage_usdt": mark_pnl, "estimated_exit_fee_usdt": exit_fee,
                        "pnl_after_assumed_exit_cost_usdt": component})
    numbers = (gross, net, total_cost, equity + net, gross / equity, *notionals.values())
    if not all(math.isfinite(x) for x in numbers):
        raise ValueError("组合结果数值过大")
    return {"kind": "manual_simultaneous_linear_contract_stress_not_order", "positions": results,
            "entered_equity_usdt": equity, "gross_notional_usdt": gross, "gross_notional_over_equity": gross / equity,
            "gross_notional_by_symbol_usdt": notionals, "largest_symbol_notional_over_equity": max(notionals.values()) / equity,
            "simultaneous_pnl_after_assumed_exit_cost_usdt": net, "assumed_exit_cost_usdt": total_cost,
            "arithmetic_equity_after_pnl": equity + net, "funding": "unknown_excluded",
            "correlation_estimated": False, "liquidation_model_supported": False, "order_permissions": False,
            "limitations": ["价格和费用由用户假设，并非概率、预测或实际报价；多个资产同步发生是情景假设",
                            "不模拟强平、标记价格、保证金、资金费、部分成交或订单簿深度，实际损失可能不同",
                            "现有权益假定已经包含历史持仓盈亏及历史费用；这里仅计算从当前标记到假定退出的变化",
                            "名义敞口为绝对值之和，不以相反方向相互抵消，不代表实际交易所保证金"]}


def position_scenarios(*, side: str, equity: float, entry: float, stop: float,
                       gap_exit: float, quantity: float, contract_size: float,
                       leverage: float, fee_rate: float) -> dict:
    values = (equity, entry, stop, gap_exit, quantity, contract_size, leverage, fee_rate)
    if not all(math.isfinite(x) for x in values):
        raise ValueError("所有输入必须是有限数字")
    if side not in {"long", "short"}:
        raise ValueError("方向必须是多或空")
    if min(equity, entry, stop, gap_exit, quantity, contract_size) <= 0:
        raise ValueError("资金、价格、数量和合约乘数必须大于零")
    if not 1 <= leverage <= 150 or not 0 <= fee_rate < 1:
        raise ValueError("杠杆或手续费率不在有效范围")
    if (side == "long" and not gap_exit <= stop < entry) or (side == "short" and not gap_exit >= stop > entry):
        raise ValueError("多仓需跳空价≤止损价<入场价；空仓需跳空价≥止损价>入场价")
    units = quantity * contract_size
    entry_notional = entry * units
    entry_fee = entry_notional * fee_rate
    if not all(math.isfinite(x) for x in (units, entry_notional, entry_fee, gap_exit * units)):
        raise ValueError("输入数值过大，无法计算")
    sign = 1 if side == "long" else -1

    def scenario(exit_price: float) -> dict:
        gross = sign * (exit_price - entry) * units
        exit_fee = exit_price * units * fee_rate
        net = gross - entry_fee - exit_fee
        loss_pct = max(-net, 0.0) / equity * 100
        post_equity = equity + net
        if not all(math.isfinite(x) for x in (gross, exit_fee, net, loss_pct, post_equity)):
            raise ValueError("输入数值过大，无法计算")
        return {"assumed_exit_price": exit_price, "gross_pnl_usdt": gross,
                "entry_fee_usdt": entry_fee, "exit_fee_usdt": exit_fee,
                "net_pnl_excluding_funding_usdt": net,
                "loss_pct_of_entered_equity": loss_pct,
                "arithmetic_equity_after_pnl": post_equity}

    return {
        "kind": "manual_hypothetical_linear_usdt_not_order",
        "side": side, "quantity_contracts": quantity, "contract_size": contract_size,
        "entry_price": entry, "entered_equity_usdt": equity, "leverage": leverage,
        "fee_rate_each_side": fee_rate, "notional_usdt": entry_notional,
        "notional_divided_by_leverage_usdt": entry_notional / leverage,
        "stop_fill_assumption": scenario(stop), "gap_fill_assumption": scenario(gap_exit),
        "funding": "unknown_excluded", "liquidation_price": None,
        "liquidation_model_supported": False, "order_permissions": False,
        "limitations": ["假定按输入价全额成交，未单独模拟滑点、部分成交和资金费",
                        "名义价值除以杠杆不是交易所实际保证金或可开仓保证",
                        "未计算标记价格、维持保证金档位、强平和其他持仓；可能先于止损被强平",
                        "权益加损益只是算术情景，不代表真实账户结余；亏损可能超出预设情景"],
    }
