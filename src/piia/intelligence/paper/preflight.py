"""On-demand sizing example from a frozen research snapshot; no orders."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

from trading_lab.perps.risk import Spec, floor_step, size_order


def virtual_sizing_example(request: dict, symbol: str, side: str, stop: float, *,
                           current_risk_hash: str, now: datetime | None = None,
                           gap_exit: float | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or side not in {"long", "short"}:
        raise ValueError("方向或当前时间无效")
    try:
        risk = request["paper_risk_context"]
        ticker = request["instruments"][symbol]["ticker"]
        stamp = datetime.fromisoformat(ticker["observed_at_utc"])
        cutoff = datetime.fromisoformat(request["cutoff_at_utc"])
        price = float(ticker["mark"])
        stop = float(stop)
        if (not request["snapshot_ok"] or stamp.tzinfo is None or cutoff.tzinfo is None
                or not timedelta(0) <= now - stamp <= timedelta(minutes=5)
                or not timedelta(0) <= now - cutoff <= timedelta(minutes=5)):
            raise ValueError("报价或研究快照已过期，请重新获取")
        if not risk["available"] or risk["owner_config_hash"] != current_risk_hash:
            raise ValueError("纸面风控不可用或已变化，请重新获取研究包")
        if not math.isfinite(price) or price <= 0 or not math.isfinite(stop) or stop <= 0:
            raise ValueError("价格和止损必须是有效正数")
        if (side == "long" and stop >= price) or (side == "short" and stop <= price):
            raise ValueError("多仓止损须低于参考价，空仓止损须高于参考价")
        if gap_exit is not None:
            gap_exit = float(gap_exit)
            if (not math.isfinite(gap_exit) or gap_exit <= 0
                    or (side == "long" and gap_exit >= stop) or (side == "short" and gap_exit <= stop)):
                raise ValueError("跳空参考价必须比止损价更不利，且为有效正数")
        terms = risk["specs"][symbol]
        spec = Spec(symbol=symbol, **{k: float(terms[k]) for k in
                                    ("min_qty", "qty_step", "min_notional", "taker", "contract_size")})
        equity = float(risk["virtual_equity_usdt"])
        leverage = float(risk["max_leverage"])
        slip = float(risk["slippage"])
        fraction = float(risk["risk_fraction"])
        if (not all(math.isfinite(x) for x in (spec.min_qty, spec.qty_step, spec.min_notional, spec.taker,
                                               spec.contract_size, equity, leverage, slip, fraction))
                or min(spec.min_qty, spec.qty_step, spec.min_notional, spec.contract_size, equity, fraction) <= 0
                or not 0 <= spec.taker < 1 or not 0 <= slip < 1):
            raise ValueError("纸面参数无效")
        qty, reason = size_order(equity, equity, price, abs(price - stop), spec, leverage=leverage,
                                 risk_fraction=fraction, cash_floor=float(risk["cash_floor"]),
                                 max_margin_fraction=float(risk["max_margin_fraction"]), slippage=slip)
        sign = 1 if side == "long" else -1
        entry_fill = price * (1 + sign * slip)
        exit_fill = stop * (1 - sign * slip)
        loss_per_contract = spec.contract_size * (abs(entry_fill - exit_fill) + spec.taker * (entry_fill + exit_fill))
        if not math.isfinite(loss_per_contract) or loss_per_contract <= 0:
            raise ValueError("无法计算有效的风险估算")
        if qty:
            qty = min(qty, floor_step(equity * fraction / loss_per_contract, spec.qty_step))
            if qty < spec.min_qty or qty * price * spec.contract_size < spec.min_notional:
                qty, reason = 0.0, "minimum_order_exceeds_budget"
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("快照缺少有效的价格或纸面参数") from exc
    gap_scenario = None
    if gap_exit is not None:
        gap_fill = gap_exit * (1 - sign * slip)
        gap_loss = qty * spec.contract_size * (sign * (entry_fill - gap_fill) + spec.taker * (entry_fill + gap_fill)) if qty else None
        if not math.isfinite(gap_fill) or (gap_loss is not None and not math.isfinite(gap_loss)):
            raise ValueError("跳空情景超出可计算范围")
        gap_scenario = {"kind": "user_hypothetical_not_forecast", "gap_reference_price": gap_exit,
                        "assumed_exit_after_slippage": gap_fill,
                        "estimated_loss_including_entry_exit_fees_usdt": gap_loss,
                        "exceeds_stop_risk_budget": gap_loss > equity * fraction if gap_loss is not None else None,
                        "exceeds_virtual_equity": gap_loss > equity if gap_loss is not None else None,
                        "liquidation_modelled": False, "is_maximum_loss": False}
    spec_age = None
    spec_freshness = "unknown"
    try:
        spec_stamp = datetime.fromisoformat(risk["spec_snapshot_at_utc"])
        if spec_stamp.tzinfo is None or spec_stamp.utcoffset() is None:
            raise ValueError("timezone required")
        spec_age = (now - spec_stamp).total_seconds()
        if spec_age < 0:
            spec_age = None
        else:
            spec_freshness = "stale" if spec_age > 300 else "recent_but_not_revalidated"
    except (KeyError, TypeError, ValueError, OverflowError):
        pass
    return {"kind": "virtual_paper_example_not_order", "symbol": symbol, "side": side,
            "reference_mark": price, "reference_at_utc": stamp.isoformat(), "assumed_stop": stop,
            "virtual_equity_usdt": equity, "leverage_assumption": leverage,
            "risk_budget_usdt": equity * fraction, "estimated_qty_contracts": qty,
            "estimated_stop_loss_with_assumed_fees_slippage_usdt": qty * loss_per_contract if qty else None,
            "gap_scenario": gap_scenario,
            "minimum_qty": spec.min_qty, "minimum_notional_usdt": spec.min_notional,
            "size_step": spec.qty_step, "reason": reason, "config_hash": current_risk_hash,
            "spec_snapshot_at_utc": risk.get("spec_snapshot_at_utc"),
            "spec_snapshot_age_seconds": spec_age, "spec_snapshot_freshness": spec_freshness,
            "current_exchange_specs_verified": False, "order_eligibility_verified": False,
            "calculated_at_utc": now.isoformat(), "order_permissions": False,
            "limitations": ["仅为虚拟账户示例，假设无其他持仓；实际资金和仓位未知",
                            "规格及费用可能过期；止损价和滑点均是假设，跳空可超过预算",
                            "未计算资金费、精确维持保证金和强平；不代表交易所可接受订单"]}
