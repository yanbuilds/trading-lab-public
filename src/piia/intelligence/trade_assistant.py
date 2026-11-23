"""Practical Trading Decision Assistant (v2) — Radar, Pre-Trade Ticket & Position Monitor.

Unifies the project's public market feeds (`public_data`, `public_orderbook`,
`public_position_tiers`), causal multi-regime v2 strategy (`adaptive_v2`), and
trader-configurable risk calculator (`risk_v2`) into three high-utility workflows:

1. `scan_market_signals`: Multi-contract radar across BTCUSDT / XAUUSDT / NDX100USDT
   (live Bitget public API or offline/synthetic fixture mode), outputting regime,
   1h/4h/1d alignment, active signals, watchlist trigger prices, funding, OI,
   and ready-to-review Trade Tickets.
2. `build_trade_plan`: Interactive pre-trade calculator for manual or strategy
   setups (exact contract sizing, breakeven price, TP1/TP2/TP3 R:R ladder,
   isolated liquidation buffer, orderbook VWAP slippage, and discipline checklist).
3. `advise_open_position`: Live or manual open-position risk monitor (unrealized
   net PnL, current R-multiple, breakeven stop lock trigger, Chandelier trailing
   stop update, distance to stop & estimated liquidation).

Never uses private keys or submits exchange orders.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from typing import Any

import numpy as np
import pandas as pd

from piia.intelligence.paper.specs import CONTRACTS, SAMPLE_MARKS
from trading_lab.perps.adaptive_v2 import latest_bar_signal_summary
from trading_lab.perps.risk_v2 import (
    compute_trade_ticket,
    estimate_isolated_liq_v2,
    fee_adjusted_breakeven,
    load_policy_v2,
    resolve_spec,
)

DEFAULT_SYMBOLS = ("BTCUSDT", "XAUUSDT", "NDX100USDT")


def _candles_to_df(candles: list[dict[str, Any]]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    rows = []
    idx = []
    for c in candles:
        idx.append(pd.Timestamp(c["open_at_utc"]).tz_convert("UTC"))
        rows.append({
            "open": float(c["open"]),
            "high": float(c["high"]),
            "low": float(c["low"]),
            "close": float(c["close"]),
            "volume": float(c.get("volume") or 0.0),
        })
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="date"))
    return df.sort_index()


def generate_demo_snapshot(now: datetime | None = None) -> dict[str, Any]:
    """Generate an explicitly labeled synthetic offline snapshot for UI/CLI demo & tests."""
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    hour_end = now_utc.replace(minute=0, second=0, microsecond=0)
    day_end = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)

    instruments: dict[str, Any] = {}
    configs = {
        "BTCUSDT": {"base": float(SAMPLE_MARKS["BTCUSDT"]["mark"]), "drift": 0.0014, "vol": 0.0045, "seed": 11},
        "XAUUSDT": {"base": float(SAMPLE_MARKS["XAUUSDT"]["mark"]), "drift": 0.0001, "vol": 0.0022, "seed": 22},
        "NDX100USDT": {"base": float(SAMPLE_MARKS["NDX100USDT"]["mark"]), "drift": -0.0011, "vol": 0.0038, "seed": 33},
    }
    for sym, p in configs.items():
        rng = np.random.default_rng(p["seed"])
        n_h1 = 260
        rets = p["drift"] + rng.normal(0.0, p["vol"], size=n_h1)
        # Shape distinct regimes: BTC steady trend_up + breakout on last bar, XAU tight range, NDX downtrend
        if sym == "BTCUSDT":
            rets[-30:] = np.abs(rets[-30:]) * 0.8 + 0.0020
            rets[-1] = 0.012
        elif sym == "XAUUSDT":
            rets[-80:] = np.sin(np.linspace(0, 8 * np.pi, 80)) * 0.0018
        elif sym == "NDX100USDT":
            rets[-30:] = -np.abs(rets[-30:]) * 0.8 - 0.0018
            rets[-1] = -0.010

        closes = np.empty(n_h1, dtype=float)
        closes[-1] = p["base"]
        for i in range(n_h1 - 2, -1, -1):
            closes[i] = closes[i + 1] / (1.0 + rets[i + 1])

        candles_1h = []
        for i in range(n_h1):
            open_at = hour_end - timedelta(hours=n_h1 - i)
            close_at = open_at + timedelta(hours=1)
            o = float(closes[i - 1]) if i > 0 else float(closes[i] * 0.999)
            c = float(closes[i])
            span = abs(c - o) + c * p["vol"] * 0.6
            candles_1h.append({
                "open_at_utc": open_at.isoformat(),
                "close_at_utc": close_at.isoformat(),
                "open": round(o, 4),
                "high": round(max(o, c) + span * 0.5, 4),
                "low": round(min(o, c) - span * 0.5, 4),
                "close": round(c, 4),
                "volume": 100.0,
            })

        candles_1d = []
        for d in range(210):
            d_open = day_end - timedelta(days=210 - d)
            d_close = d_open + timedelta(days=1)
            factor = 1.0 - (210 - d) * (0.001 if sym != "NDX100USDT" else -0.0005)
            px = max(10.0, p["base"] * factor)
            candles_1d.append({
                "open_at_utc": d_open.isoformat(),
                "close_at_utc": d_close.isoformat(),
                "open": round(px * 0.997, 4),
                "high": round(px * 1.008, 4),
                "low": round(px * 0.992, 4),
                "close": round(px, 4),
                "volume": 2400.0,
            })

        mark = float(candles_1h[-1]["close"])
        spread = mark * 0.0002
        instruments[sym] = {
            "symbol": sym,
            "instrument_id": CONTRACTS[sym]["instrument_id"],
            "ticker": {
                "last": mark,
                "mark": mark,
                "index": mark,
                "observed_at_utc": now_utc.isoformat(),
                "provenance": "SYNTHETIC_DEMO_FIXTURE_NOT_LIVE",
            },
            "funding": {
                "funding_rate": "0.00005",
                "retrieved_at_utc": now_utc.isoformat(),
                "provenance": "SYNTHETIC_DEMO_FIXTURE_NOT_LIVE",
            },
            "open_interest": {
                "total_open_interest_base_units": 12500.0,
                "change_24h_pct": 2.4,
                "period_end_utc": hour_end.isoformat(),
            },
            "orderbook": {
                "ok": True,
                "mid": str(round(mark, 4)),
                "spread": str(round(spread, 4)),
                "bids": [{"price": str(round(mark - spread * (k + 0.5), 4)), "size": "50"} for k in range(5)],
                "asks": [{"price": str(round(mark + spread * (k + 0.5), 4)), "size": "50"} for k in range(5)],
            },
            "tiers": None,
            "contract": None,
            "candles_closed_1h": candles_1h,
            "candles_closed_1d": candles_1d,
        }

    return {
        "ok": True,
        "mode": "SYNTHETIC_DEMO_FIXTURE_NOT_LIVE",
        "is_synthetic_demo": True,
        "retrieved_at_utc": now_utc.isoformat(),
        "instruments": instruments,
        "errors": [],
    }


def scan_market_signals(
    symbols: list[str] | tuple[str, ...] | None = None,
    *,
    equity_usdt: float | None = None,
    leverage: float | None = None,
    risk_fraction: float | None = None,
    order_type: str = "limit",
    demo_mode: bool = False,
    snapshot_override: dict[str, Any] | None = None,
    include_orderbook_and_tiers: bool = True,
) -> dict[str, Any]:
    """Scan multi-contract regime, v2 signals, key levels, and auto-sized Trade Tickets."""
    policy = load_policy_v2()
    rcfg = policy["risk"]
    syms = tuple(symbols or DEFAULT_SYMBOLS)
    eq = float(rcfg["default_equity_usdt"] if equity_usdt is None else equity_usdt)
    lev = float(rcfg["default_leverage"] if leverage is None else leverage)
    r_frac = float(rcfg["risk_fraction_per_trade"] if risk_fraction is None else risk_fraction)

    if snapshot_override is not None:
        snap = snapshot_override
        mode = str(snap.get("mode") or "supplied_snapshot")
    elif demo_mode:
        snap = generate_demo_snapshot()
        mode = "SYNTHETIC_DEMO_FIXTURE_NOT_LIVE"
    else:
        from piia.intelligence.public_data import fetch_public_snapshot
        snap = fetch_public_snapshot(list(syms), candle_limit=300)
        mode = "live_public_bitget"
        if include_orderbook_and_tiers:
            from piia.intelligence.public_orderbook import fetch_execution_books
            from piia.intelligence.public_position_tiers import fetch_position_tiers
            books = fetch_execution_books(syms)
            tiers_snap = fetch_position_tiers(syms)
            for sym in syms:
                inst = snap["instruments"].get(sym)
                if inst is not None:
                    b_inst = (books.get("instruments") or {}).get(sym) or {}
                    inst["orderbook"] = b_inst.get("orderbook")
                    inst["contract"] = b_inst.get("contract")
                    inst["tiers"] = (tiers_snap.get("instruments") or {}).get(sym)

    cards: dict[str, Any] = {}
    actionable_candidates: list[dict[str, Any]] = []

    for sym in syms:
        inst = (snap.get("instruments") or {}).get(sym) or {}
        h1_df = _candles_to_df(inst.get("candles_closed_1h") or [])
        d1_df = _candles_to_df(inst.get("candles_closed_1d") or [])
        sig = latest_bar_signal_summary(h1_df, d1_df if not d1_df.empty else None, policy=policy)

        ticker = inst.get("ticker") or {}
        live_mark = float(ticker["mark"]) if ticker.get("mark") is not None else (sig.get("last_close") or 0.0)
        funding_info = inst.get("funding") or {}
        f_rate = None
        if funding_info.get("funding_rate") not in (None, ""):
            try:
                f_rate = float(funding_info["funding_rate"])
            except (TypeError, ValueError):
                f_rate = None
        oi_info = inst.get("open_interest") or {}

        ticket = None
        if sig.get("ok") and sig.get("signal_int") in (-1, 1) and live_mark > 0 and sig.get("suggested_stop_distance"):
            side = "long" if sig["signal_int"] == 1 else "short"
            stop_dist = float(sig["suggested_stop_distance"])
            stop_px = live_mark - sig["signal_int"] * stop_dist
            tp2_px = live_mark + sig["signal_int"] * 2.5 * stop_dist
            if stop_px > 0 and tp2_px > 0:
                ticket = compute_trade_ticket(
                    symbol=sym,
                    side=side,
                    equity_usdt=eq,
                    entry_price=live_mark,
                    stop_loss=stop_px,
                    take_profit=tp2_px,
                    leverage=lev,
                    risk_fraction=r_frac,
                    order_type=order_type,
                    funding_rate=f_rate,
                    orderbook=inst.get("orderbook"),
                    tiers=inst.get("tiers"),
                    contract_override=inst.get("contract"),
                    policy=policy,
                )
                if ticket.get("actionable_for_manual_review"):
                    actionable_candidates.append({
                        "symbol": sym,
                        "side": side,
                        "leg": sig.get("leg"),
                        "signal_strength": sig.get("signal_strength", 0.0),
                        "verdict": ticket["verdict"],
                        "qty_contracts": ticket["sizing"]["qty_contracts"],
                        "entry_price": live_mark,
                        "stop_loss": round(stop_px, 6),
                        "take_profit_2_5r": round(tp2_px, 6),
                    })

        cards[sym] = {
            "symbol": sym,
            "live_mark": live_mark if live_mark > 0 else None,
            "ticker_observed_at_utc": ticker.get("observed_at_utc"),
            "funding_rate_8h": f_rate,
            "funding_annualized_pct": round(f_rate * 3 * 365 * 100.0, 2) if f_rate is not None else None,
            "open_interest_base": oi_info.get("total_open_interest_base_units"),
            "open_interest_change_24h_pct": (
                round(float(oi_info["change_24h_pct"]), 2)
                if oi_info.get("change_24h_pct") is not None else None
            ),
            "analysis": sig,
            "trade_ticket": ticket,
        }

    actionable_candidates.sort(key=lambda x: (-float(x["signal_strength"]), x["symbol"]))
    top_pick = actionable_candidates[0] if actionable_candidates else None

    return {
        "kind": "trading_assistant_market_scan_v2",
        "mode": mode,
        "is_synthetic_demo": bool(snap.get("is_synthetic_demo")),
        "retrieved_at_utc": snap.get("retrieved_at_utc"),
        "snapshot_ok": bool(snap.get("ok")),
        "errors": snap.get("errors") or [],
        "account_assumptions": {
            "equity_usdt": eq,
            "leverage": lev,
            "risk_fraction_pct": round(r_frac * 100.0, 3),
            "order_type": order_type,
        },
        "top_candidate": top_pick,
        "actionable_candidates": actionable_candidates,
        "instruments": cards,
        "order_permissions": False,
    }


def build_trade_plan(
    *,
    symbol: str,
    side: str,
    equity_usdt: float,
    entry_price: float | None = None,
    stop_loss: float | None = None,
    stop_atr_multiple: float = 2.0,
    take_profit: float | None = None,
    target_rr: float | None = 2.5,
    leverage: float = 2.0,
    risk_pct: float = 1.0,
    order_type: str = "limit",
    fetch_live_context: bool = False,
) -> dict[str, Any]:
    """Build a complete pre-trade plan from user inputs or live market context."""
    policy = load_policy_v2()
    orderbook = None
    tiers = None
    contract = None
    f_rate = None
    atr_used = None

    if fetch_live_context or entry_price is None or stop_loss is None:
        from piia.intelligence.public_data import fetch_public_snapshot
        from piia.intelligence.public_orderbook import fetch_execution_books
        from piia.intelligence.public_position_tiers import fetch_position_tiers
        snap = fetch_public_snapshot([symbol], candle_limit=120)
        inst = (snap.get("instruments") or {}).get(symbol) or {}
        books = fetch_execution_books((symbol,))
        tiers_snap = fetch_position_tiers((symbol,))
        b_inst = (books.get("instruments") or {}).get(symbol) or {}
        orderbook = b_inst.get("orderbook")
        contract = b_inst.get("contract")
        tiers = (tiers_snap.get("instruments") or {}).get(symbol)
        if entry_price is None:
            tk = b_inst.get("ticker") or inst.get("ticker") or {}
            if tk.get("mark") is not None:
                entry_price = float(tk["mark"])
        f_obj = inst.get("funding") or {}
        if f_obj.get("funding_rate") not in (None, ""):
            try:
                f_rate = float(f_obj["funding_rate"])
            except (TypeError, ValueError):
                pass
        if stop_loss is None and entry_price is not None:
            h1_df = _candles_to_df(inst.get("candles_closed_1h") or [])
            sig = latest_bar_signal_summary(h1_df, policy=policy)
            atr_val = (sig.get("indicators") or {}).get("atr14")
            if not atr_val or atr_val <= 0:
                raise ValueError(f"无法为 {symbol} 自动计算 ATR14 止损，请显式传入 --stop")
            atr_used = float(atr_val)
            sign = 1.0 if side.lower() == "long" else -1.0
            stop_loss = float(entry_price) - sign * float(stop_atr_multiple) * atr_used

    if entry_price is None or stop_loss is None:
        raise ValueError("entry_price 与 stop_loss 不能为空（离线模式下请显式指定）")

    if take_profit is None and target_rr is not None and target_rr > 0:
        sign = 1.0 if side.lower() == "long" else -1.0
        dist = abs(float(entry_price) - float(stop_loss))
        cand_tp = float(entry_price) + sign * float(target_rr) * dist
        if cand_tp > 0:
            take_profit = cand_tp

    ticket = compute_trade_ticket(
        symbol=symbol,
        side=side,
        equity_usdt=float(equity_usdt),
        entry_price=float(entry_price),
        stop_loss=float(stop_loss),
        take_profit=float(take_profit) if take_profit is not None else None,
        leverage=float(leverage),
        risk_fraction=float(risk_pct) / 100.0,
        order_type=order_type,
        funding_rate=f_rate,
        orderbook=orderbook,
        tiers=tiers,
        contract_override=contract,
        policy=policy,
    )
    ticket["atr14_used_for_stop"] = atr_used
    return ticket


def advise_open_position(
    *,
    symbol: str,
    side: str,
    entry_price: float,
    qty: float,
    stop_loss: float,
    leverage: float = 2.0,
    mark_price: float | None = None,
    candles_1h: list[dict[str, Any]] | None = None,
    fetch_live: bool = False,
) -> dict[str, Any]:
    """Evaluate an open position's live PnL, R-multiple, liquidation buffer & stop updates."""
    side = side.lower().strip()
    if side not in {"long", "short"}:
        raise ValueError("side must be 'long' or 'short'")
    if min(entry_price, qty, stop_loss, leverage) <= 0:
        raise ValueError("entry_price, qty, stop_loss, and leverage must be > 0")

    policy = load_policy_v2()
    spec = resolve_spec(symbol)
    if fetch_live or mark_price is None:
        from piia.intelligence.public_data import fetch_public_snapshot
        snap = fetch_public_snapshot([spec.symbol], candle_limit=120)
        inst = (snap.get("instruments") or {}).get(spec.symbol) or {}
        if mark_price is None and (inst.get("ticker") or {}).get("mark") is not None:
            mark_price = float(inst["ticker"]["mark"])
        if candles_1h is None:
            candles_1h = inst.get("candles_closed_1h")
    if mark_price is None or mark_price <= 0:
        raise ValueError("mark_price unavailable; pass mark_price explicitly or enable fetch_live")

    sign = 1.0 if side == "long" else -1.0
    units = qty * spec.contract_size
    notional_entry = units * entry_price
    notional_mark = units * mark_price
    margin = notional_entry / leverage
    entry_fee = notional_entry * spec.taker
    est_exit_fee = notional_mark * spec.taker
    gross_pnl = sign * (mark_price - entry_price) * units
    net_pnl = gross_pnl - entry_fee - est_exit_fee
    roe_pct = (net_pnl / margin * 100.0) if margin > 0 else 0.0

    initial_stop_dist = abs(entry_price - stop_loss)
    r_multiple = (sign * (mark_price - entry_price) / initial_stop_dist) if initial_stop_dist > 0 else None
    be_px = fee_adjusted_breakeven(entry_price, side, entry_fee_rate=spec.taker, exit_fee_rate=spec.taker)
    est_liq = estimate_isolated_liq_v2(
        side=side, qty=qty, entry_fill=entry_price, leverage=leverage,
        mmr=spec.mmr, taker=spec.taker, contract_size=spec.contract_size,
    )

    dist_to_stop_pct = sign * (mark_price - stop_loss) / mark_price * 100.0
    dist_to_liq_pct = (sign * (mark_price - est_liq) / mark_price * 100.0) if est_liq is not None else None

    recommended_stop = stop_loss
    actions: list[str] = []
    status = "HOLD"

    if dist_to_stop_pct <= 0:
        status = "STOP_BREACHED"
        actions.append(f"⚠️ 当前标记价 {mark_price:.4f} 已触及/越过止损价 {stop_loss:.4f}，请立即核查平仓，切勿扛单！")
    elif est_liq is not None and ((side == "long" and stop_loss <= est_liq) or (side == "short" and stop_loss >= est_liq)):
        status = "LIQUIDATION_RISK"
        actions.append(f"🚨 止损价 {stop_loss:.4f} 晚于估算逐仓强平价 {est_liq:.4f}，存在先于止损被强平风险，请立即上调止损或减仓！")

    be_r = float(policy["risk"].get("breakeven_trigger_r", 1.5))
    if r_multiple is not None and r_multiple >= be_r:
        if (side == "long" and stop_loss < be_px) or (side == "short" and stop_loss > be_px):
            recommended_stop = be_px
            status = "TRAIL_TO_BREAKEVEN"
            actions.append(
                f"✅ 当前浮盈已达 +{r_multiple:.2f}R（≥ {be_r:g}R）：建议将止损推进至含费保本价 {be_px:.4f}，锁定零亏损底线。"
            )
        if r_multiple >= 2.5:
            actions.append(f"💰 浮盈已达 +{r_multiple:.2f}R：可考虑分批止盈 30%~50% 仓位，剩余仓位用跟踪止损让利润奔跑。")

    regime_info = None
    if candles_1h:
        h1_df = _candles_to_df(candles_1h)
        if len(h1_df) >= 65:
            sig = latest_bar_signal_summary(h1_df, policy=policy)
            regime_info = {
                "regime": sig.get("regime"),
                "timeframes": sig.get("timeframes"),
                "chandelier_long_stop": (sig.get("key_levels") or {}).get("chandelier_long_stop"),
                "chandelier_short_stop": (sig.get("key_levels") or {}).get("chandelier_short_stop"),
            }
            ch_long = regime_info["chandelier_long_stop"]
            ch_short = regime_info["chandelier_short_stop"]
            if side == "long" and ch_long and ch_long > recommended_stop and ch_long < mark_price:
                recommended_stop = max(recommended_stop, float(ch_long))
                actions.append(f"📈 Chandelier 吊灯跟踪止损上移至 {recommended_stop:.4f}。")
            elif side == "short" and ch_short and ch_short < recommended_stop and ch_short > mark_price:
                recommended_stop = min(recommended_stop, float(ch_short))
                actions.append(f"📉 Chandelier 吊灯跟踪止损下移至 {recommended_stop:.4f}。")
            if sig.get("regime") == "extreme_vol":
                actions.append("⚡ 盘面进入极端波动状态（extreme_vol），切勿逆势加仓，收紧移动止损保护利润/本金。")
            tf = sig.get("timeframes") or {}
            if (side == "long" and tf.get("1h") == "down" and tf.get("4h") == "down") or (
                side == "short" and tf.get("1h") == "up" and tf.get("4h") == "up"
            ):
                status = "TREND_REVERSED"
                actions.append("🔄 1h 与 4h 均线已同步反向共振，原有趋势逻辑弱化，建议主动减仓或离场。")

    if not actions:
        actions.append("持仓状态正常：继续严格执行预设止损，未达保本推损阈值前避免频繁微调。")

    return {
        "kind": "open_position_risk_advisor_v2",
        "symbol": spec.symbol,
        "side": side,
        "status": status,
        "position": {
            "qty_contracts": qty,
            "entry_price": entry_price,
            "mark_price": mark_price,
            "leverage": leverage,
            "initial_margin_usdt": round(margin, 4),
            "notional_mark_usdt": round(notional_mark, 4),
        },
        "pnl": {
            "gross_unrealized_usdt": round(gross_pnl, 4),
            "net_unrealized_after_fees_usdt": round(net_pnl, 4),
            "roe_on_margin_pct": round(roe_pct, 2),
            "current_r_multiple": round(r_multiple, 3) if r_multiple is not None else None,
        },
        "risk_levels": {
            "current_stop_loss": stop_loss,
            "recommended_stop_loss": round(recommended_stop, 6),
            "fee_adjusted_breakeven": round(be_px, 6),
            "estimated_isolated_liquidation": round(est_liq, 6) if est_liq is not None else None,
            "distance_to_stop_pct": round(dist_to_stop_pct, 3),
            "distance_to_liquidation_pct": round(dist_to_liq_pct, 3) if dist_to_liq_pct is not None else None,
        },
        "market_regime": regime_info,
        "actions": actions,
        "order_permissions": False,
    }


def format_scan_table(scan: dict[str, Any]) -> str:
    """Render a concise terminal summary table from `scan_market_signals`."""
    lines = [
        f"=== Trading Lab v2 盘面雷达 ({scan.get('mode')}) | 截至 {scan.get('retrieved_at_utc')} ===",
        f"账户参数假设: 权益={scan['account_assumptions']['equity_usdt']}U | 杠杆={scan['account_assumptions']['leverage']}x | 单笔风险={scan['account_assumptions']['risk_fraction_pct']}%",
        "-" * 96,
        f"{'标的':<11} {'最新标记价':>11} {'24h涨跌':>8} {'周期(1h/4h/1d)':<15} {'市场状态':<13} {'v2信号':<8} {'ADX':>6} {'RSI':>6} {'资金费(年化)':>12}",
        "-" * 96,
    ]
    for sym, card in (scan.get("instruments") or {}).items():
        an = card.get("analysis") or {}
        tf = an.get("timeframes") or {}
        ind = an.get("indicators") or {}
        tf_str = f"{tf.get('1h', '?')}/{tf.get('4h', '?')}/{tf.get('1d', '?')}"
        mark_s = f"{card['live_mark']:.2f}" if card.get("live_mark") else "N/A"
        ret_s = f"{an['return_24h_pct']:+.2f}%" if an.get("return_24h_pct") is not None else "N/A"
        adx_s = f"{ind['adx14']:.1f}" if ind.get("adx14") is not None else "N/A"
        rsi_s = f"{ind['rsi14']:.1f}" if ind.get("rsi14") is not None else "N/A"
        fund_s = f"{card['funding_annualized_pct']:+.1f}%" if card.get("funding_annualized_pct") is not None else "N/A"
        lines.append(
            f"{sym:<11} {mark_s:>11} {ret_s:>8} {tf_str:<15} {str(an.get('regime', 'N/A')):<13} {str(an.get('signal', 'WAIT')):<8} {adx_s:>6} {rsi_s:>6} {fund_s:>12}"
        )
        if an.get("summary_cn"):
            lines.append(f"  └─ 解读: {an['summary_cn']}")
        tk = card.get("trade_ticket")
        if tk and tk.get("actionable_for_manual_review"):
            lines.append(
                f"  └─ 🎯 工单建议 [{tk['verdict']}]: {tk['side'].upper()} {tk['sizing']['qty_contracts']:g} 张 | "
                f"入场≈{tk['prices']['planned_entry']:.2f} | 止损={tk['prices']['stop_loss']:.2f} | "
                f"保本价={tk['prices']['breakeven_limit_exit']:.2f} | 估算强平={tk['prices']['estimated_isolated_liquidation']}"
            )
    lines.append("-" * 96)
    return "\n".join(lines)
