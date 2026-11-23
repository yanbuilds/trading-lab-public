"""Reproducible 3-layer Market Tide classification engine (PIIA Judgment v3).

Layers:
1. `macro_backdrop`: Joint assessment of Dollar (ICE DXY or lagged DTWEXBGS proxy),
   10Y TIPS real yield (DFII10), Fed net liquidity (WALCL - WDTGAL - RRP), and
   cross-market risk appetite (VIXCLS). Never inferred from a single moving average;
   returns `unknown` when < 2 macro/liquidity pillars are known.
2. `cycle_stage`: Market phase (`expansion`, `ebbing`, `deleveraging`, `repair`, or
   `insufficient_evidence`) from joint return, base-unit OI, realized volume,
   ETF flow, and macro liquidity rules.
3. `instrument_positions`: Per-symbol multi-timeframe alignment (`1d UTC`, `4h`, `1h`)
   distinguishing trend-following, pullbacks, counter-trend bounces, reversal
   candidates, and range consolidations with explicit invalidation levels.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from piia.intelligence.agent_v2.pillars import SYMBOLS, prepare_market_feature_table
from trading_lab.indicators_v2 import ema, sma


def _safe_float(val: Any) -> float | None:
    if val is None or isinstance(val, bool):
        return None
    try:
        v = float(val)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _compute_daily_position(df_1d: pd.DataFrame | None) -> dict[str, Any]:
    if df_1d is None or df_1d.empty or len(df_1d) < 50:
        return {
            "direction": "unknown",
            "bars_count": 0 if df_1d is None else int(len(df_1d)),
            "ema20": None,
            "ema50": None,
            "sma200": None,
            "daily_20d_high": None,
            "daily_20d_low": None,
            "reason": "insufficient_closed_daily_bars_min_50",
        }
    closes = df_1d["close"].astype(float).to_numpy(dtype=float)
    highs = df_1d["high"].astype(float).to_numpy(dtype=float)
    lows = df_1d["low"].astype(float).to_numpy(dtype=float)
    e20 = float(ema(closes, 20)[-1])
    e50 = float(ema(closes, 50)[-1])
    s200 = float(sma(closes, 200)[-1]) if len(closes) >= 200 else None
    last_c = float(closes[-1])
    d20_hi = float(highs[-20:].max()) if len(highs) >= 20 else None
    d20_lo = float(lows[-20:].min()) if len(lows) >= 20 else None

    if e20 > e50 and last_c > e50 and (s200 is None or last_c >= s200):
        direction = "up"
    elif e20 < e50 and last_c < e50 and (s200 is None or last_c <= s200):
        direction = "down"
    else:
        direction = "neutral"

    return {
        "direction": direction,
        "bars_count": int(len(df_1d)),
        "last_close_1d": round(last_c, 4),
        "ema20": round(e20, 4),
        "ema50": round(e50, 4),
        "sma200": round(s200, 4) if s200 is not None and math.isfinite(s200) else None,
        "daily_20d_high": round(d20_hi, 4) if d20_hi is not None else None,
        "daily_20d_low": round(d20_lo, 4) if d20_lo is not None else None,
        "reason": None,
    }


def _compute_instrument_position(
    sym: str,
    df_1h: pd.DataFrame | None,
    df_1d: pd.DataFrame | None,
    *,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    tcfg = cfg["technical_thresholds"]
    daily = _compute_daily_position(df_1d)
    if df_1h is None or df_1h.empty or len(df_1h) < 100:
        return {
            "symbol": sym,
            "daily_dir": daily["direction"],
            "h4_dir": "unknown",
            "h1_dir": "unknown",
            "position_state": "unknown",
            "structure_kind_cn": "数据不足（无法判定多周期结构）",
            "daily": daily,
            "missing_inputs": [f"closed_1h_bars_below_100:{sym}"],
            "invalidation_levels": {},
        }, None

    prep = prepare_market_feature_table(df_1h)
    row = prep.iloc[-1]
    close_px = _safe_float(row.get("close"))
    e20 = _safe_float(row.get("ema20"))
    e50 = _safe_float(row.get("ema50"))
    h4_20 = _safe_float(row.get("htf_ema20"))
    h4_50 = _safe_float(row.get("htf_ema50"))
    adx14 = _safe_float(row.get("adx14"))
    pdi = _safe_float(row.get("plus_di14"))
    mdi = _safe_float(row.get("minus_di14"))
    rsi14 = _safe_float(row.get("rsi14"))
    atr14 = _safe_float(row.get("atr14"))
    atr_m = _safe_float(row.get("atr14_mean50"))
    don_hi = _safe_float(row.get("donchian_high"))
    don_lo = _safe_float(row.get("donchian_low"))
    bb_mid = _safe_float(row.get("bb_mid"))
    bb_up = _safe_float(row.get("bb_upper"))
    bb_lo = _safe_float(row.get("bb_lower"))

    missing: list[str] = []
    if daily["direction"] == "unknown":
        missing.append(f"daily_context_insufficient:{sym}")

    if any(v is None for v in (close_px, e20, e50, h4_20, h4_50, adx14, atr14)):
        missing.append(f"1h_or_4h_warmup_incomplete:{sym}")
        return {
            "symbol": sym,
            "daily_dir": daily["direction"],
            "h4_dir": "unknown",
            "h1_dir": "unknown",
            "position_state": "unknown",
            "structure_kind_cn": "均线或 4H 预热不足（未知）",
            "daily": daily,
            "missing_inputs": missing,
            "invalidation_levels": {},
        }, prep

    assert close_px is not None and e20 is not None and e50 is not None and h4_20 is not None and h4_50 is not None and adx14 is not None and atr14 is not None

    h4_dir = "up" if h4_20 > h4_50 else ("down" if h4_20 < h4_50 else "neutral")
    h1_dir = "up" if e20 > e50 else ("down" if e20 < e50 else "neutral")
    atr_ratio = (atr14 / atr_m) if (atr_m is not None and atr_m > 0) else None
    ret_24h = ((close_px / float(df_1h.iloc[-25]["close"]) - 1.0) * 100.0) if len(df_1h) >= 25 else None

    if adx14 < float(tcfg["adx_range_max"]) and h1_dir != h4_dir:
        pos_state = "range_consolidation"
        struct_cn = "低趋势强度区间震荡"
    elif h4_dir == "up" and h1_dir == "up":
        if daily["direction"] == "down" and adx14 >= float(tcfg["adx_reversal_min"]) and don_hi is not None and close_px > don_hi:
            pos_state = "potential_bullish_reversal"
            struct_cn = "潜在底部向上反转候选（1H/4H多头破48h高点，日线尚未转多）"
        elif daily["direction"] == "down":
            pos_state = "bounce_in_downtrend"
            struct_cn = "日线偏空背景下的 1H/4H 逆势反弹"
        else:
            pos_state = "aligned_uptrend"
            struct_cn = "顺势多头结构（1D/4H/1H 一致向上）" if daily["direction"] == "up" else "4H/1H 顺势上行（日线缺口披露）"
    elif h4_dir == "down" and h1_dir == "down":
        if daily["direction"] == "up" and adx14 >= float(tcfg["adx_reversal_min"]) and don_lo is not None and close_px < don_lo:
            pos_state = "potential_bearish_reversal"
            struct_cn = "潜在高位向下反转候选（1H/4H破48h低点，日线尚未转空）"
        elif daily["direction"] == "up":
            pos_state = "pullback_in_uptrend"
            struct_cn = "日线偏多背景下的 1H/4H 趋势回踩/回调"
        else:
            pos_state = "aligned_downtrend"
            struct_cn = "顺势空头结构（1D/4H/1H 一致向下）" if daily["direction"] == "down" else "4H/1H 顺势下行（日线缺口披露）"
    elif h4_dir == "up" and h1_dir == "down":
        pos_state = "pullback_in_uptrend"
        struct_cn = "4H 上行背景下的 1H 回踩整理"
    elif h4_dir == "down" and h1_dir == "up":
        if adx14 >= float(tcfg["adx_reversal_min"]) and don_hi is not None and close_px > don_hi:
            pos_state = "potential_bullish_reversal"
            struct_cn = "4H 偏空但 1H 放量上破 48h 高点（向上反转观察）"
        else:
            pos_state = "bounce_in_downtrend"
            struct_cn = "4H 下行趋势中的 1H 逆势反弹"
    else:
        pos_state = "range_consolidation" if adx14 < float(tcfg["adx_range_max"]) else "transition_uncertain"
        struct_cn = "周期震荡" if pos_state == "range_consolidation" else "多周期切换待确认"

    return {
        "symbol": sym,
        "close": round(close_px, 4),
        "return_24h_pct": round(ret_24h, 3) if ret_24h is not None else None,
        "daily_dir": daily["direction"],
        "h4_dir": h4_dir,
        "h1_dir": h1_dir,
        "position_state": pos_state,
        "structure_kind_cn": struct_cn,
        "metrics": {
            "adx14": round(adx14, 2),
            "plus_di14": round(pdi, 2) if pdi is not None else None,
            "minus_di14": round(mdi, 2) if mdi is not None else None,
            "rsi14": round(rsi14, 2) if rsi14 is not None else None,
            "atr14": round(atr14, 4),
            "atr_ratio": round(atr_ratio, 3) if atr_ratio is not None else None,
            "h1_ema20": round(e20, 4),
            "h1_ema50": round(e50, 4),
            "h4_ema20": round(h4_20, 4),
            "h4_ema50": round(h4_50, 4),
        },
        "daily": daily,
        "missing_inputs": missing,
        "invalidation_levels": {
            "h1_ema50": round(e50, 4),
            "h4_ema50": round(h4_50, 4),
            "donchian_48h_high": round(don_hi, 4) if don_hi is not None else None,
            "donchian_48h_low": round(don_lo, 4) if don_lo is not None else None,
            "bb_upper": round(bb_up, 4) if bb_up is not None else None,
            "bb_mid": round(bb_mid, 4) if bb_mid is not None else None,
            "bb_lower": round(bb_lo, 4) if bb_lo is not None else None,
            "daily_20d_high": daily.get("daily_20d_high"),
            "daily_20d_low": daily.get("daily_20d_low"),
        },
    }, prep


def compute_market_tide(
    *,
    frames_1h: dict[str, pd.DataFrame],
    frames_1d: dict[str, pd.DataFrame],
    market_snapshot: dict[str, Any],
    macro_summary: dict[str, Any],
    vix_summary: dict[str, Any],
    sent_summary: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Compute reproducible 3-layer Market Tide (macro_backdrop, cycle_stage, instrument_positions)."""
    mcfg = cfg["macro_and_tide_thresholds"]

    # ---------------- Layer 1: Macro Backdrop (never inferred from single MA!) ----------------
    ry_bps = _safe_float(macro_summary.get("real_yield_delta_5obs_bps"))
    usd_pct = _safe_float(macro_summary.get("dollar_pct_5obs"))
    liq_chg = _safe_float(macro_summary.get("net_liquidity_change_pct"))
    vix_val = _safe_float(vix_summary.get("value"))
    vix_elev = bool(vix_summary.get("elevated_volatility_regime"))

    ry_thresh = float(mcfg["real_yield_5obs_bps"])
    usd_thresh = float(mcfg["broad_dollar_5obs_pct"])
    liq_thresh = float(mcfg["net_liquidity_change_pct"])

    dollar_status = "unknown"
    if usd_pct is not None:
        dollar_status = "tightening" if usd_pct >= usd_thresh else ("easing" if usd_pct <= -usd_thresh else "neutral")

    real_yield_status = "unknown"
    if ry_bps is not None:
        real_yield_status = "tightening" if ry_bps >= ry_thresh else ("easing" if ry_bps <= -ry_thresh else "neutral")

    net_liq_status = "unknown"
    if liq_chg is not None:
        net_liq_status = "tightening" if liq_chg <= -liq_thresh else ("easing" if liq_chg >= liq_thresh else "neutral")

    risk_appetite_status = "unknown"
    if vix_val is not None:
        if vix_elev:
            risk_appetite_status = "risk_off"
        elif vix_val <= float(mcfg["vix_low_level"]):
            risk_appetite_status = "risk_on"
        else:
            risk_appetite_status = "balanced"

    known_macro_pillars = sum(1 for x in (dollar_status, real_yield_status, net_liq_status) if x != "unknown")
    macro_missing: list[str] = []
    if not macro_summary.get("ice_dxy_available"):
        macro_missing.append("realtime_ice_dxy_unavailable_using_dtwexbgs_proxy_if_present")
    if dollar_status == "unknown":
        macro_missing.append("broad_dollar_proxy_unavailable")
    if real_yield_status == "unknown":
        macro_missing.append("real_yield_dfii10_unavailable")
    if net_liq_status == "unknown":
        macro_missing.append("fed_net_liquidity_unavailable")
    if risk_appetite_status == "unknown":
        macro_missing.append("vix_volatility_unavailable")

    tight_count = sum(1 for x in (dollar_status, real_yield_status, net_liq_status) if x == "tightening")
    ease_count = sum(1 for x in (dollar_status, real_yield_status, net_liq_status) if x == "easing")

    backdrop_by_symbol: dict[str, str] = {s: "unknown" for s in SYMBOLS}
    if known_macro_pillars < 2:
        macro_state = "unknown"
        macro_cn = "宏观与流动性核心支柱已知项不足 2 项，大环境判定为未知（不从价格均线反推）"
        macro_flip = ["补充完整的 DFII10 实际利率、DTWEXBGS 美元代理与美联储净流动性数据后重新判定"]
    else:
        if tight_count >= 2 and ease_count == 0:
            macro_state = "headwind"
            macro_cn = "紧缩宏观逆风（美元与实际利率/流动性同步偏紧）"
            backdrop_by_symbol = {"BTCUSDT": "headwind", "NDX100USDT": "headwind", "XAUUSDT": "headwind"}
            if risk_appetite_status == "risk_off" and real_yield_status != "tightening":
                backdrop_by_symbol["XAUUSDT"] = "mixed_safe_haven_vs_usd"
        elif ease_count >= 2 and tight_count == 0:
            macro_state = "tailwind"
            if risk_appetite_status == "risk_off":
                macro_state = "mixed_cross_currents"
                macro_cn = "利率/美元放松但 VIX 处于高波动避险状态（跨资产分化）"
                backdrop_by_symbol = {"XAUUSDT": "tailwind", "NDX100USDT": "headwind", "BTCUSDT": "mixed"}
            else:
                macro_cn = "宽松宏观顺风（实际利率与美元压力缓和/流动性改善）"
                backdrop_by_symbol = {"BTCUSDT": "tailwind", "NDX100USDT": "tailwind", "XAUUSDT": "tailwind"}
        elif risk_appetite_status == "risk_off" and real_yield_status == "easing":
            macro_state = "mixed_cross_currents"
            macro_cn = "避险冲击 + 实际利率回落：黄金避险顺风，纳指估值与风险溢价承压"
            backdrop_by_symbol = {"XAUUSDT": "tailwind", "NDX100USDT": "headwind", "BTCUSDT": "mixed"}
        elif tight_count == 1 and ease_count == 0:
            macro_state = "mild_headwind"
            macro_cn = "单维度温和偏紧背景（仅一项收紧，其余中性）"
            backdrop_by_symbol = {s: ("headwind" if risk_appetite_status == "risk_off" and s == "NDX100USDT" else "neutral_mild_headwind") for s in SYMBOLS}
        elif ease_count == 1 and tight_count == 0:
            macro_state = "mild_tailwind"
            macro_cn = "单维度温和偏松背景"
            backdrop_by_symbol = {s: ("headwind" if risk_appetite_status == "risk_off" and s == "NDX100USDT" else "neutral_mild_tailwind") for s in SYMBOLS}
        else:
            macro_state = "mixed_cross_currents"
            macro_cn = "宏观信号中性或互有抵消"
            backdrop_by_symbol = {s: "neutral" for s in SYMBOLS}
        macro_flip = [
            f"若 10Y 实际利率 5 观测变化跨过 ±{ry_thresh:g} bps 且广义美元同向跨过 ±{usd_thresh:g}%，大环境顺逆风将改判",
            f"若 VIX 升破 {float(mcfg['vix_high_level']):g} 或单日暴涨 >= {float(mcfg['vix_spike_1d_pct']):g}%，股票风险偏好切为 risk_off",
        ]

    macro_backdrop = {
        "state": macro_state,
        "state_cn": macro_cn,
        "backdrop_by_symbol": backdrop_by_symbol,
        "pillars": {
            "dollar": {"status": dollar_status, "dtwexbgs_5obs_pct": usd_pct, "ice_dxy_available": macro_summary.get("ice_dxy_available")},
            "real_yield": {"status": real_yield_status, "dfii10_5obs_bps": ry_bps},
            "fed_net_liquidity": {"status": net_liq_status, "change_pct": liq_chg},
            "risk_appetite": {"status": risk_appetite_status, "vix_value": vix_val},
        },
        "known_macro_pillars": known_macro_pillars,
        "dependencies": ["DTWEXBGS", "DFII10", "WALCL/WDTGAL/RRPONTSYD", "VIXCLS"],
        "missing_inputs": macro_missing,
        "flip_conditions": macro_flip,
    }

    # ---------------- Layer 3: Per-Instrument Multi-Timeframe Position ----------------
    positions: dict[str, dict[str, Any]] = {}
    prepared_frames: dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        pos, prep = _compute_instrument_position(sym, frames_1h.get(sym), frames_1d.get(sym), cfg=cfg)
        positions[sym] = pos
        if prep is not None:
            prepared_frames[sym] = prep

    # ---------------- Layer 2: Market Cycle Stage (joint cross-market criteria) ----------------
    btc_ret = _safe_float((positions.get("BTCUSDT") or {}).get("return_24h_pct"))
    ndx_ret = _safe_float((positions.get("NDX100USDT") or {}).get("return_24h_pct"))
    xau_ret = _safe_float((positions.get("XAUUSDT") or {}).get("return_24h_pct"))
    btc_oi = _safe_float((((market_snapshot.get("instruments") or {}).get("BTCUSDT") or {}).get("open_interest") or {}).get("change_24h_pct"))
    etf_3d = _safe_float(sent_summary.get("btc_etf_3d_musd"))

    # Realized buy volume demand over trailing 24 closed 1h BTC bars (closed bars only, NOT momentary orderbook)
    up_down_vol_ratio = None
    held_above_low = None
    btc_df = frames_1h.get("BTCUSDT")
    if btc_df is not None and len(btc_df) >= 25:
        tail24 = btc_df.tail(24)
        up_v = float(tail24.loc[tail24["close"] > tail24["open"], "volume"].sum())
        dn_v = float(tail24.loc[tail24["close"] < tail24["open"], "volume"].sum())
        if dn_v > 0:
            up_down_vol_ratio = round(up_v / dn_v, 3)
        prior_lo = float(btc_df.iloc[-25:-1]["low"].min())
        held_above_low = bool(float(btc_df.iloc[-1]["close"]) > prior_lo)

    stage_missing: list[str] = []
    if btc_ret is None or ndx_ret is None:
        stage_missing.append("btc_or_ndx_24h_closed_return_unavailable")
    if btc_oi is None:
        stage_missing.append("btc_base_oi_24h_change_unavailable")
    if etf_3d is None:
        stage_missing.append("btc_spot_etf_complete_3d_flow_unavailable")

    if btc_ret is None or ndx_ret is None or known_macro_pillars < 2:
        cycle_stage = "insufficient_evidence"
        stage_cn = "证据不足（缺少核心闭盘行情或宏观流动性基准）"
        stage_def = "无法在缺少多资产收益与宏观基底的情况下判定周期阶段"
        stage_flip = ["补充完整的三合约闭合小时线与核心宏观序列后重新计算阶段"]
    elif btc_ret <= -3.0 and btc_oi is not None and btc_oi < 0.0 and (risk_appetite_status == "risk_off" or macro_state == "headwind"):
        cycle_stage = "deleveraging"
        stage_cn = "去杠杆（风险资产下跌 + 合约未平仓量收缩 + 宏观或波动承压）"
        stage_def = "btc_24h <= -3% 且 btc_oi_24h < 0 且伴有高波动/宏观偏紧"
        stage_flip = ["价格不再破 24h 低点、上涨成交量比 >= 1.15 且宏观紧缩消退则转入修复"]
    elif (
        held_above_low is True
        and up_down_vol_ratio is not None
        and up_down_vol_ratio >= 1.15
        and (macro_state in {"tailwind", "mild_tailwind"} or real_yield_status == "easing")
        and (btc_ret >= -0.5 or ndx_ret >= 0.0)
    ):
        cycle_stage = "repair"
        stage_cn = "修复阶段（价格守住前低 + 上涨放量回升 + 实际利率或美元压力缓和）"
        stage_def = "close > 24h_prior_low 且 up/down_vol >= 1.15 且实际利率/宏观缓和"
        stage_flip = ["若实际利率与美元重新急升或再度跌破 24h 前低，将退回退潮/去杠杆"]
    elif (
        btc_ret >= 1.5
        and (ndx_ret >= 0.0 or xau_ret is not None and xau_ret >= 0.0)
        and (btc_oi is not None and btc_oi > 0.0)
        and macro_state in {"tailwind", "mild_tailwind", "mixed_cross_currents"}
        and risk_appetite_status != "risk_off"
    ):
        cycle_stage = "expansion"
        stage_cn = "扩张阶段（风险资产共振走高 + 持仓量扩张 + 未触发宏观逆风/恐慌）"
        stage_def = "btc_24h >= +1.5% 且 权益无恐慌且持仓量扩张且无强宏观逆风"
        stage_flip = ["若实际利率与美元同步走紧或 ETF 连续大幅净流出，则降级为退潮"]
    elif (
        (btc_ret < -0.5 and ndx_ret < -0.2)
        or (macro_state in {"headwind", "mild_headwind"} and etf_3d is not None and etf_3d <= float(mcfg["etf_3d_outflow_musd"]))
        or (risk_appetite_status == "risk_off" and ndx_ret < 0)
    ):
        cycle_stage = "ebbing"
        stage_cn = "退潮阶段（风险偏好走弱 / 资金流降温 / 宏观流动性偏紧）"
        stage_def = "风险资产共振走弱，或宏观逆风叠加机构流入转负/高 VIX"
        stage_flip = ["若价格企稳放量且宏观/ETF 流入同步改善，将转入修复或扩张"]
    elif btc_oi is None and etf_3d is None:
        cycle_stage = "insufficient_evidence"
        stage_cn = "证据不足（缺少持仓量与 ETF 资金流完整确认，不强行归入扩张/退潮）"
        stage_def = "价格小幅波动但衍生品 OI 与现货 ETF 流缺失"
        stage_flip = ["补充完整 OI 24h 变化或 ETF 完整日度数据以确认资金流向阶段"]
    else:
        cycle_stage = "ebbing" if macro_state == "headwind" else "repair" if macro_state == "tailwind" else "insufficient_evidence"
        stage_cn = (
            "温和退潮（价格胶着但宏观大环境偏紧）"
            if cycle_stage == "ebbing"
            else "温和修复震荡（宏观偏松但跨资产动能温和）"
            if cycle_stage == "repair"
            else "盘整中性/证据分歧（等待跨市场突破或资金流明确共振）"
        )
        stage_def = "无单边扩张或强去杠杆信号，受宏观斜率与波动均衡约束"
        stage_flip = ["跨资产 24h 涨跌幅与 OI/宏观出现同向突破时切换阶段"]

    tide = {
        "macro_backdrop": macro_backdrop,
        "cycle_stage": {
            "stage": cycle_stage,
            "stage_cn": stage_cn,
            "definition_used": stage_def,
            "supporting_metrics": {
                "btc_24h_pct": btc_ret,
                "ndx_24h_pct": ndx_ret,
                "xau_24h_pct": xau_ret,
                "btc_oi_24h_pct": btc_oi,
                "btc_up_down_volume_24h_ratio": up_down_vol_ratio,
                "btc_held_above_24h_low": held_above_low,
                "btc_etf_3d_musd": etf_3d,
            },
            "missing_inputs": stage_missing,
            "flip_conditions": stage_flip,
        },
        "instrument_positions": positions,
    }
    return tide, prepared_frames
