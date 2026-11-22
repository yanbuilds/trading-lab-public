"""Phase-1 historical research packets: then-known, anonymized, no outcomes.

Decision dates, costs, holding horizon and stop rule are frozen up front by a
fixed calendar (not by later returns). Parent-facing packets omit calendar
dates and absolute prices. Masking reduces but does not eliminate pretraining
or prior-session contamination. This is a retrospective exploratory test on
already-seen data, not a holdout and not an autonomous multi-agent forecast.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from piia.intelligence.hashing import content_hash
from piia.intelligence.paper.replay import atr, ema
from piia.intelligence.paper.specs import CONTRACTS
from piia.intelligence.paths import ROOT
from trading_lab.perps.compare import load_cache
from trading_lab.perps.risk import Spec, size_order

PROTOCOL = "PIIA_HISTORICAL_RESEARCH_PACKETS_V1"
# Frozen before any outcome is computed. Equally spaced 00:00 UTC instants.
# Last instant leaves a full 72-hour horizon inside the cache (ends 2026-11-16 05:00Z).
DECISION_TIMES = tuple(
    pd.Timestamp(t, tz="UTC")
    for t in (
        "2026-09-01 00:00:00",
        "2026-09-09 00:00:00",
        "2026-09-17 00:00:00",
        "2026-09-25 00:00:00",
        "2026-10-03 00:00:00",
        "2026-10-11 00:00:00",
        "2026-10-19 00:00:00",
        "2026-10-27 00:00:00",
        "2026-11-04 00:00:00",
        "2026-11-12 00:00:00",
    )
)
HOLD_HOURS = 72
STOP_ATR_MULT = 2.0
ATR_PERIOD = 14
LOOKBACK_BARS = 24  # closed 1h bars shown in the packet
CAPITAL = 30.0
LEVERAGE = 2.0
SLIPPAGE = 0.0005
# Backend identity only; never written into the parent-facing packet.
ASSET_MAP = {
    "Asset-A": "BTCUSDT",
    "Asset-B": "XAUUSDT",
    "Asset-C": "NDX100USDT",
}
PAIR = {
    "BTCUSDT": "BTC/USDT:USDT",
    "XAUUSDT": "XAU/USDT:USDT",
    "NDX100USDT": "NDX100/USDT:USDT",
}
DISPLAY_ORDER = ("Asset-A", "Asset-B", "Asset-C")
MISSING = [
    "macro", "fed_policy", "real_rates", "dxy", "tga", "rrp",
    "etf_flows", "onchain", "equities_fundamentals", "sentiment_indices", "news",
]


def _shuffle_ids(n: int) -> list[str]:
    """Reproducible permutation; seed is the protocol name, not performance."""
    digest = hashlib.sha256(PROTOCOL.encode()).digest()
    order = list(range(n))
    # Fisher-Yates with digest bytes
    for i in range(n - 1, 0, -1):
        order[i], order[digest[i] % (i + 1)] = order[digest[i] % (i + 1)], order[i]
        digest = hashlib.sha256(digest).digest()
    return [f"P-{i+1:02d}" for i in order]


def _spec(sym: str) -> Spec:
    c = CONTRACTS[sym]
    return Spec(
        symbol=sym, min_qty=c["min_qty"], qty_step=c["qty_step"],
        min_notional=c["min_notional"], taker=c["taker"], contract_size=c["contract_size"],
    )


def _prefix(df: pd.DataFrame, decision_at: pd.Timestamp) -> pd.DataFrame:
    """Bars whose close time is <= decision_at, i.e. open + 1h <= decision_at, so open < decision_at."""
    return df[df.index < decision_at]


def _funding_at(rows: list[dict], decision_at: pd.Timestamp) -> float | None:
    known = []
    for r in rows:
        ts = pd.Timestamp(int(r["timestamp"]), unit="ms", tz="UTC")
        if ts <= decision_at:
            known.append((ts, float(r["rate"])))
    if not known:
        return None
    known.sort()
    return known[-1][1]


def _features(h1: pd.DataFrame, funding_rate: float | None, spec: Spec) -> dict[str, Any]:
    if len(h1) < 50:
        raise ValueError("insufficient closed bars for EMA50 at a frozen decision time")
    close = h1["close"].to_numpy(dtype=float)
    high = h1["high"].to_numpy(dtype=float)
    low = h1["low"].to_numpy(dtype=float)
    e20 = ema(close, 20)
    e50 = ema(close, 50)
    a14 = atr(high, low, close, ATR_PERIOD)
    last = float(close[-1])
    scale = 100.0 / last
    atr_v = float(a14[-1])
    ema20_v, ema50_v = float(e20[-1]), float(e50[-1])
    cross = "none"
    if np.isfinite([e20[-2], e50[-2], ema20_v, ema50_v]).all():
        if ema20_v > ema50_v and e20[-2] <= e50[-2]:
            cross = "bullish_cross"
        elif ema20_v < ema50_v and e20[-2] >= e50[-2]:
            cross = "bearish_cross"

    def ret(hours: int) -> float | None:
        if len(close) <= hours:
            return None
        prev = float(close[-(hours + 1)])
        if prev <= 0:
            return None
        return float(close[-1] / prev - 1.0)

    # 20-day Donchian location using only completed UTC days before the last bar's day
    daily = h1.resample("1D", label="left", closed="left").agg(
        {"high": "max", "low": "min", "close": "last"}
    ).dropna()
    donchian_pos = None
    if len(daily) >= 21:
        prior = daily.iloc[:-1].tail(20)
        lo, hi = float(prior["low"].min()), float(prior["high"].max())
        if hi > lo:
            donchian_pos = float((float(daily["close"].iloc[-1]) - lo) / (hi - lo))

    atr_pct = atr_v / last
    # vol regime vs the previous 50 closed ATR values (prefix only)
    valid_atr = a14[np.isfinite(a14)]
    vol_regime = "unknown"
    if len(valid_atr) >= 30:
        med = float(np.median(valid_atr[-50:-1])) if len(valid_atr) > 1 else float("nan")
        if np.isfinite(med) and med > 0:
            ratio = atr_v / med
            vol_regime = "high" if ratio >= 1.25 else ("low" if ratio <= 0.75 else "normal")

    qty, reason = size_order(
        CAPITAL, CAPITAL, last, STOP_ATR_MULT * atr_v, spec,
        leverage=LEVERAGE, slippage=SLIPPAGE,
    )
    bars = h1.iloc[-LOOKBACK_BARS:]
    rel_bars = []
    for ts, row in bars.iterrows():
        rel_bars.append({
            "o": round(float(row["open"]) * scale, 4),
            "h": round(float(row["high"]) * scale, 4),
            "l": round(float(row["low"]) * scale, 4),
            "c": round(float(row["close"]) * scale, 4),
        })
    trend = {
        "close_vs_ema50": "above" if last > ema50_v else "below",
        "ema20_vs_ema50": "above" if ema20_v > ema50_v else "below",
        "ema_cross_on_last_closed_bar": cross,
        "donchian_20d_position": None if donchian_pos is None else round(donchian_pos, 3),
    }
    return {
        "last_close_index": 100.0,
        "ema20_index": round(ema20_v * scale, 4),
        "ema50_index": round(ema50_v * scale, 4),
        "atr14_index": round(atr_v * scale, 4),
        "atr14_pct_of_close": round(atr_pct, 6),
        "stop_distance_index": round(STOP_ATR_MULT * atr_v * scale, 4),
        "trend": trend,
        "momentum": {
            "return_24h": None if ret(24) is None else round(ret(24), 6),
            "return_7d": None if ret(24 * 7) is None else round(ret(24 * 7), 6),
            "return_20d": None if ret(24 * 20) is None else round(ret(24 * 20), 6),
        },
        "vol_regime": vol_regime,
        "funding_rate": None if funding_rate is None else round(float(funding_rate), 8),
        "recent_closed_1h_bars_index": rel_bars,
        "sizing_probe_30u_2x_1pct_2atr": {
            "result": "ok" if qty else "rejected",
            "reason": reason,
        },
        "missing_datasets": list(MISSING),
        "notes": [
            "Prices are rebased so the last fully closed 1h close equals 100. Absolute venue prices and calendar dates are omitted.",
            "Indicators use only closed bars whose close time is at or before the (hidden) decision clock.",
            "Macro, news, and on-chain series were not in this cache and are missing, not inferred.",
        ],
    }, {
        "last_close": last,
        "atr14": atr_v,
        "ema20": ema20_v,
        "ema50": ema50_v,
        "qty_probe": qty,
        "qty_reason": reason,
        "bars_used": int(len(h1)),
        "last_bar_open": str(h1.index[-1]),
    }


def build_packets(data_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    frames, specs, funding, prov = load_cache(data_dir)
    display_ids = _shuffle_ids(len(DECISION_TIMES))
    parent_packets = []
    sealed_events = []
    for i, decision_at in enumerate(DECISION_TIMES):
        assets_parent = {}
        assets_sealed = {}
        for alias, sym in ASSET_MAP.items():
            pair = PAIR[sym]
            h1 = _prefix(frames[pair], decision_at)
            rate = _funding_at(funding[pair], decision_at)
            feat, raw = _features(h1, rate, _spec(sym))
            assets_parent[alias] = feat
            assets_sealed[alias] = {"symbol": sym, "instrument_id": CONTRACTS[sym]["instrument_id"], **raw}
        packet_id = display_ids[i]
        parent_packets.append({
            "packet_id": packet_id,
            "decision_clock": "hidden; all observations are then-known closed-bar or earlier funding",
            "assets": assets_parent,
            "decision_required": {
                "universe": list(DISPLAY_ORDER),
                "max_open_positions": 1,
                "allowed_directions": ["long", "short", "flat"],
                "default_stop": f"{STOP_ATR_MULT}*ATR14 of the last closed 1h bar, fixed, in index points",
                "default_horizon_hours": HOLD_HOURS,
            },
        })
        sealed_events.append({
            "packet_id": packet_id,
            "sequence": i + 1,
            "decision_at_utc": str(decision_at),
            "fill_open_at_utc": str(decision_at),
            "horizon_exit_at_utc": str(decision_at + pd.Timedelta(hours=HOLD_HOURS)),
            "assets": assets_sealed,
        })
    parent = {
        "protocol": PROTOCOL,
        "phase": 1,
        "label": "retrospective_exploratory_chat_assisted",
        "held_out": False,
        "already_seen_dataset": True,
        "independent_llm_review": "NOT_WIRED",
        "masking_disclaimer": (
            "Packet IDs are shuffled and prices are rebased to 100. This reduces but does not "
            "eliminate leakage from model pretraining or from earlier work on this same sample."
        ),
        "no_outcomes_in_this_file": True,
        "frozen_rules": {
            "virtual_equity": CAPITAL,
            "max_leverage": LEVERAGE,
            "risk_fraction": 0.01,
            "cash_floor": 0.20,
            "max_margin_fraction": 0.50,
            "max_open_positions": 1,
            "slippage_each_side": SLIPPAGE,
            "taker": 0.0006,
            "funding": "historical rate * contemporaneous open as mark proxy; applied in phase 2 only",
            "stop": f"fixed {STOP_ATR_MULT}*ATR14 on the last closed 1h bar",
            "holding_horizon_hours": HOLD_HOURS,
            "entry": "next 1h open after the decision clock",
            "exit": "horizon open, or earlier stop / 3% daily or 10% total equity halt",
            "comparator": "frozen turtle_s1_donchian candidate, no retuning, evaluated only in phase 2",
            "data_split": "config/piia_data_splits.yaml; this sample sits inside already-seen windows plus prefix warmup; not a pristine holdout",
            "spec_assumption": "current local contract specs applied retrospectively; archival specs unavailable",
            "liquidation": "approximate_stress_not_venue_verified",
        },
        "packet_count": len(parent_packets),
        "missing_datasets_all_packets": list(MISSING),
        "response_format": response_format(),
        "packets": parent_packets,
    }
    sealed = {
        "protocol": PROTOCOL,
        "phase": 1,
        "sealed": True,
        "contains_outcomes": False,
        "asset_map": ASSET_MAP,
        "decision_times_utc": [str(t) for t in DECISION_TIMES],
        "display_order_of_packets": [p["packet_id"] for p in parent_packets],
        "events": sealed_events,
        "provenance_last_common_bar": prov["symbols"][PAIR["BTCUSDT"]]["last"],
        "note": "Sealed mapping for phase-2 execution only. Do not show this file when eliciting decisions.",
    }
    parent["packet_file_hash"] = content_hash(
        json.dumps({k: parent[k] for k in parent if k != "packet_file_hash"}, sort_keys=True, default=str).encode()
    )
    return parent, sealed


def response_format() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "phase": "2_input",
        "instructions": (
            "For every packet_id, write researcher, critic, and synthesizer views using only "
            "that packet. Direction may be flat. If long or short, choose at most one asset. "
            "Do not invent macro or news. Default stop and 72h horizon apply unless you stand aside. "
            "This is chat-assisted manual research, not an independent LLM review."
        ),
        "schema": {
            "protocol": PROTOCOL,
            "actor": "invest-desk_chat_assisted",
            "generation": "manual",
            "independent_llm_review": "NOT_WIRED",
            "decisions": [
                {
                    "packet_id": "P-XX",
                    "researcher": {
                        "direction": "long|short|flat",
                        "asset": "Asset-A|Asset-B|Asset-C|null",
                        "summary": "string",
                        "invalidation": "string",
                        "missing_data": ["..."],
                        "claims": [{"text": "string", "status": "supported|unsupported"}],
                    },
                    "critic": {
                        "direction": "long|short|flat",
                        "asset": "Asset-A|Asset-B|Asset-C|null",
                        "summary": "string",
                        "disagreements": ["string"],
                        "invalidation": "string",
                        "missing_data": ["..."],
                    },
                    "synthesizer": {
                        "direction": "long|short|flat",
                        "asset": "Asset-A|Asset-B|Asset-C|null",
                        "suggested_action": "review_paper_plan|stand_aside|need_more_evidence",
                        "summary": "string",
                        "invalidation": "string",
                        "disagreements": ["string"],
                        "accept_default_2atr_stop_and_72h_horizon": True,
                    },
                }
            ],
        },
    }


def _write_summary_table(path: Path, parent: dict[str, Any]) -> None:
    def pct(v):
        return "n/a" if v is None else f"{v*100:+.2f}%"
    lines = ["指标摘要（价格已折成当时收盘=100；不含日期、绝对价格、事后涨跌）"]
    for p in parent["packets"]:
        lines.append("")
        lines.append(f"=== {p['packet_id']} ===")
        for alias in DISPLAY_ORDER:
            x = p["assets"][alias]
            tr, mo = x["trend"], x["momentum"]
            lines.append(
                f"  {alias}: close/EMA50={tr['close_vs_ema50']}, EMA20/EMA50={tr['ema20_vs_ema50']}"
                f" (EMA20={x['ema20_index']}, EMA50={x['ema50_index']}), cross={tr['ema_cross_on_last_closed_bar']},"
                f" donchian20={tr['donchian_20d_position']}, vol={x['vol_regime']}, ATR%={x['atr14_pct_of_close']*100:.3f}%,"
                f" ret24h={pct(mo['return_24h'])}, ret7d={pct(mo['return_7d'])}, ret20d={pct(mo['return_20d'])},"
                f" funding={x['funding_rate']}, size_probe={x['sizing_probe_30u_2x_1pct_2atr']['reason']}"
            )
    path.write_text("\n".join(lines) + "\n")



def write_packets(data_dir: Path, out_dir: Path) -> dict[str, str]:
    parent, sealed = build_packets(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parent_path = out_dir / "PACKETS_FOR_REVIEW.json"
    sealed_path = out_dir / "BACKEND_SEALED.json"
    template_path = out_dir / "RESPONSE_TEMPLATE.json"
    parent_path.write_text(json.dumps(parent, indent=2, ensure_ascii=False))
    sealed_path.write_text(json.dumps(sealed, indent=2, ensure_ascii=False))
    template_path.write_text(json.dumps({
        "protocol": PROTOCOL,
        "actor": "invest-desk_chat_assisted",
        "generation": "manual",
        "independent_llm_review": "NOT_WIRED",
        "decisions": [
            {
                "packet_id": p["packet_id"],
                "researcher": {"direction": None, "asset": None, "summary": "", "invalidation": "", "missing_data": list(MISSING), "claims": []},
                "critic": {"direction": None, "asset": None, "summary": "", "disagreements": [], "invalidation": "", "missing_data": list(MISSING)},
                "synthesizer": {
                    "direction": None, "asset": None,
                    "suggested_action": None, "summary": "", "invalidation": "",
                    "disagreements": [],
                    "accept_default_2atr_stop_and_72h_horizon": True,
                },
            }
            for p in parent["packets"]
        ],
    }, indent=2, ensure_ascii=False))
    # Parent-facing brief without dates or prices
    brief = out_dir / "README_FOR_REVIEW.txt"
    ids = [p["packet_id"] for p in parent["packets"]]
    brief.write_text(
        "\n".join([
            "PIIA 历史研究决策测试 — 第1阶段（只提供当时可知信息，不含事后结果）",
            f"protocol={PROTOCOL}",
            "性质：回溯、探索性、聊天协助。数据此前已用过，不是样本外，也不是自动多Agent预测。",
            "匿名处理：随机编号，价格按当时最后一根已收盘K收盘价折成100。这只能降低、不能消除模型预训练或先前工作带来的污染。",
            f"包数量：{len(ids)}，编号：{', '.join(ids)}",
            "每个包含 Asset-A / Asset-B / Asset-C 三个同交易所永续合约的当时趋势、动量、波动、近期相对K线、资金费，以及 30U/2x/1%风险/2*ATR 能否满足最小下单量。宏观、新闻、链上均为缺失，勿编造。",
            "冻结规则：30U，最多2倍杠杆，1%风险，单仓，手续费0.06%，单边滑点0.05%，2*ATR14固定止损，持有72小时或先触及止损/停机。入场为决策后下一根1小时开盘。对照策略为已冻结的海龟 System 1，第2阶段才计算收益。",
            "请对每个 packet_id 给出研究、反方、综合意见；综合若做多或做空，最多选一个资产，或全部观望。",
            "请按 RESPONSE_TEMPLATE.json 填写完整 JSON，不要查看 BACKEND_SEALED.json。",
            f"packet_file_hash={parent['packet_file_hash']}",
        ]) + "\n"
    )
    _write_summary_table(out_dir / "SUMMARY_TABLE.txt", parent)
    return {
        "parent": str(parent_path),
        "brief": str(brief),
        "template": str(template_path),
        "summary": str(out_dir / "SUMMARY_TABLE.txt"),
        "sealed": str(sealed_path),
        "packet_file_hash": parent["packet_file_hash"],
        "packet_ids": ",".join(ids),
        "count": str(len(ids)),
    }


def main() -> int:
    data = ROOT / "data" / "perps-extended"
    out = ROOT / "artifacts" / "research_packets" / "historical_v1"
    info = write_packets(data, out)
    print("wrote", info["parent"])
    print("brief", info["brief"])
    print("template", info["template"])
    print("count", info["count"], "ids", info["packet_ids"])
    print("packet_file_hash", info["packet_file_hash"])
    print("sealed mapping written separately; not for elicitation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
