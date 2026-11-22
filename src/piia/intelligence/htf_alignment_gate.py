"""Diagnostic 4h-alignment / no-chase eligibility gate (version v1).

The gate manifest in config/piia_htf_alignment_gate_v1.yaml was frozen before
this experiment's gated P&L. It does not revise historical writeups, stops, or
horizons, and it is not an optimized or held-out strategy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from piia.intelligence.historical_independent import (
    DB_PATH as INDEPENDENT_DB,
    run_independent,
)
from piia.intelligence.historical_packets import ASSET_MAP, PAIR
from piia.intelligence.historical_phase2 import (
    DB_PATH as MANUAL_DB,
    FORBIDDEN_DB,
    PACKET_DIR,
    run_phase2,
    write_report,
)
from piia.intelligence.paper.replay import atr, ema
from piia.intelligence.paths import ROOT
from trading_lab.perps.compare import load_cache

GATE_ID = "piia_htf_alignment_gate_v1"
MANIFEST_PATH = ROOT / "config" / "piia_htf_alignment_gate_v1.yaml"
OUT_DIR = ROOT / "artifacts" / "research_packets" / "htf_alignment_gate_v1"
DATA_DIR = ROOT / "data" / "perps-extended"
MANUAL_GATE_DB = ROOT / "data" / "piia" / "piia.htf_gate_v1_manual.sqlite"
INDEPENDENT_GATE_DB = ROOT / "data" / "piia" / "piia.htf_gate_v1_independent.sqlite"
MANUAL_EV = ROOT / "data" / "piia" / "htf_gate_v1_manual_evidence"
INDEPENDENT_EV = ROOT / "data" / "piia" / "htf_gate_v1_independent_evidence"
EXTENSION_THRESHOLD_ATR = 1.0
SLIPPAGE = 0.0005


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if data.get("id") != GATE_ID:
        raise ValueError("unexpected gate manifest id")
    if float(data["not_overextended"]["threshold_atr"]) != EXTENSION_THRESHOLD_ATR:
        raise ValueError("manifest threshold does not match the frozen code constant")
    return data


def completed_4h(h1: pd.DataFrame, decision_at: pd.Timestamp) -> pd.DataFrame:
    """Resample prefix 1h bars into completed 4h bars only (no partial current bar)."""
    decision_at = pd.Timestamp(decision_at)
    if decision_at.tzinfo is None:
        decision_at = decision_at.tz_localize("UTC")
    else:
        decision_at = decision_at.tz_convert("UTC")
    prefix = h1[h1.index < decision_at]
    if prefix.empty:
        return prefix.copy()
    h4 = prefix.resample("4h", label="left", closed="left", origin="epoch").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "high", "low", "close"])
    complete = h4.index + pd.Timedelta(hours=4) <= decision_at
    return h4.loc[complete]


def _last_indicators(frame: pd.DataFrame) -> dict[str, Any]:
    if frame is None or len(frame) == 0:
        return {"defined": False}
    close = frame["close"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    e20 = ema(close, 20)
    e50 = ema(close, 50)
    a14 = atr(high, low, close, 14)
    if not np.isfinite(e20[-1]) or not np.isfinite(e50[-1]) or not np.isfinite(a14[-1]):
        return {
            "defined": False,
            "bars": int(len(frame)),
            "last_bar_open": str(frame.index[-1]),
            "close": float(close[-1]),
        }
    return {
        "defined": True,
        "bars": int(len(frame)),
        "last_bar_open": str(frame.index[-1]),
        "close": float(close[-1]),
        "ema20": float(e20[-1]),
        "ema50": float(e50[-1]),
        "atr14": float(a14[-1]),
        "close_vs_ema20": "above" if close[-1] > e20[-1] else "below",
        "close_vs_ema50": "above" if close[-1] > e50[-1] else "below",
        "ema20_vs_ema50": "above" if e20[-1] > e50[-1] else "below",
    }


def signed_extension_atr(close: float, ema20: float, atr14: float, side: str) -> float | None:
    if not all(np.isfinite(x) for x in (close, ema20, atr14)) or atr14 <= 0:
        return None
    if side == "long":
        return float(max(0.0, (close - ema20) / atr14))
    if side == "short":
        return float(max(0.0, (ema20 - close) / atr14))
    raise ValueError("side must be long or short")


def volume_quality(h1_prefix: pd.DataFrame, h4: pd.DataFrame) -> dict[str, Any]:
    if "volume" not in h1_prefix.columns:
        return {"available": False, "note": "no volume column in cache"}
    vol = h1_prefix["volume"].astype(float)
    tail = vol.iloc[-24:] if len(vol) else vol
    zeros = int((vol == 0).sum())
    missing = int(vol.isna().sum())
    last = float(vol.iloc[-1]) if len(vol) else None
    med = float(tail.median()) if len(tail) else None
    ratio = None if not med else (last / med if last is not None and med > 0 else None)
    h4_last = float(h4["volume"].iloc[-1]) if h4 is not None and len(h4) and "volume" in h4.columns else None
    return {
        "available": True,
        "source": "bitget_public_1h_cache_summed_to_4h",
        "verified_participation": False,
        "rows": int(len(vol)),
        "zeros": zeros,
        "missing": missing,
        "last_1h_volume": last,
        "median_last_24_1h": med,
        "last_vs_24h_median": None if ratio is None else round(ratio, 4),
        "last_completed_4h_volume": h4_last,
        "note": "Venue contract volume is present but unverified as participation or conviction evidence.",
    }


def evaluate_gate(h1: pd.DataFrame, decision_at: pd.Timestamp, side: str) -> dict[str, Any]:
    decision_at = pd.Timestamp(decision_at)
    prefix = h1[h1.index < decision_at]
    h4 = completed_4h(h1, decision_at)
    i1 = _last_indicators(prefix)
    i4 = _last_indicators(h4)
    ext = None
    if i1.get("defined"):
        ext = signed_extension_atr(i1["close"], i1["ema20"], i1["atr14"], side)
    trend_ok = False
    if i4.get("defined"):
        if side == "long":
            trend_ok = i4["close"] > i4["ema20"] and i4["close"] > i4["ema50"] and i4["ema20"] > i4["ema50"]
        else:
            trend_ok = i4["close"] < i4["ema20"] and i4["close"] < i4["ema50"] and i4["ema20"] < i4["ema50"]
    extension_ok = ext is not None and ext < EXTENSION_THRESHOLD_ATR
    eligible = bool(i1.get("defined") and i4.get("defined") and trend_ok and extension_ok)
    reason = "ok"
    if not i1.get("defined") or not i4.get("defined"):
        reason = "indicators_undefined"
    elif not trend_ok:
        reason = "4h_trend_does_not_agree"
    elif not extension_ok:
        reason = "overextended_vs_1h_ema20"
    prior_1h_high = float(prefix["high"].iloc[-20:].max()) if len(prefix) else None
    prior_1h_low = float(prefix["low"].iloc[-20:].min()) if len(prefix) else None
    loc_high = None
    loc_low = None
    if prior_1h_high is not None and prior_1h_low is not None and prior_1h_high > prior_1h_low and i1.get("close") is not None:
        loc_high = (i1["close"] - prior_1h_low) / (prior_1h_high - prior_1h_low)
        loc_low = 1.0 - loc_high
    return {
        "gate_id": GATE_ID,
        "side": side,
        "eligible": eligible,
        "reason": reason,
        "four_hour_trend_agrees": trend_ok,
        "not_overextended": extension_ok,
        "extension_atr": ext,
        "extension_threshold_atr": EXTENSION_THRESHOLD_ATR,
        "one_hour": i1,
        "four_hour": i4,
        "recent_20_1h_high": prior_1h_high,
        "recent_20_1h_low": prior_1h_low,
        "close_in_recent_20_1h_range": loc_high,
        "last_completed_4h_high": float(h4["high"].iloc[-1]) if len(h4) else None,
        "last_completed_4h_low": float(h4["low"].iloc[-1]) if len(h4) else None,
        "volume_quality": volume_quality(prefix, h4),
        "hypothesis_optimized": False,
    }


def path_diagnostics(
    h1: pd.DataFrame, *, fill: float, stop: float, side: str,
    entry_at: pd.Timestamp, exit_at: pd.Timestamp, horizon_at: pd.Timestamp,
) -> dict[str, Any]:
    """MFE/MAE and 72h hindsight path. Diagnostic only; not a reason to widen the stop."""
    entry_at = pd.Timestamp(entry_at)
    exit_at = pd.Timestamp(exit_at)
    horizon_at = pd.Timestamp(horizon_at)
    # Bars from entry open through the bar preceding official exit time for intra-bar path,
    # plus the exit bar if needed. Use highs/lows of bars with open >= entry and open < exit
    # for the realized stop path, and open < horizon+? through horizon bar open for 72h.
    actual = h1[(h1.index >= entry_at) & (h1.index < exit_at)]
    # include the exit bar for MAE/MFE if stop can occur intra-bar
    exit_bar = h1[h1.index == (exit_at if exit_at in h1.index else exit_at - pd.Timedelta(hours=1))]
    # For atr_stop recorded at close-of-bar timestamp (exit_time often bar close = next hour label
    # or the stop bar open). Our reports used: time-horizon at horizon open; atr_stop at bar close
    # (exit_time = ts+1h). So the stop bar open is exit_at - 1h for atr_stop.
    stop_open = exit_at - pd.Timedelta(hours=1)
    realized = h1[(h1.index >= entry_at) & (h1.index <= stop_open)]
    hyp = h1[(h1.index >= entry_at) & (h1.index < horizon_at)]
    horizon_open = float(h1.at[horizon_at, "open"]) if horizon_at in h1.index else None

    def mfe_mae(bars: pd.DataFrame) -> tuple[float | None, float | None]:
        if bars is None or bars.empty:
            return None, None
        if side == "short":
            mfe = float(fill - bars["low"].min())
            mae = float(bars["high"].max() - fill)
        else:
            mfe = float(bars["high"].max() - fill)
            mae = float(fill - bars["low"].min())
        return mfe, mae

    mfe, mae = mfe_mae(realized)
    mfe_72, mae_72 = mfe_mae(hyp)
    hours_to_exit = (exit_at - entry_at).total_seconds() / 3600.0
    # 72h directional thesis: horizon open vs fill, ignoring the stop (hindsight).
    thesis_ok = None
    if horizon_open is not None:
        thesis_ok = (horizon_open < fill) if side == "short" else (horizon_open > fill)
    favorable_72h = None
    if horizon_open is not None:
        favorable_72h = (fill - horizon_open) if side == "short" else (horizon_open - fill)
    return {
        "hours_to_recorded_exit": hours_to_exit,
        "realized_until_exit": {
            "mfe_price": mfe,
            "mae_price": mae,
            "mfe_over_mae": None if not mae else (None if mfe is None else mfe / mae),
            "bars": int(len(realized)),
        },
        "hindsight_72h_ignoring_stop": {
            "mfe_price": mfe_72,
            "mae_price": mae_72,
            "horizon_open": horizon_open,
            "favorable_price_move": favorable_72h,
            "direction_would_have_been_correct_at_horizon_open": thesis_ok,
            "note": "Hindsight only. A later favorable 72h move is not a reason to widen the frozen 2*ATR stop.",
        },
    }


def make_entry_prepare(frames: dict, sealed_events: list[dict], previous=None):
    by_id = {e["packet_id"]: e for e in sealed_events}

    def entry_prepare(*, raw_open, stop_distance, side, symbol, packet_id, decision, event):
        if previous is not None:
            prior = previous(
                raw_open=raw_open, stop_distance=stop_distance, side=side, symbol=symbol,
                packet_id=packet_id, decision=decision, event=event,
            )
            if not prior.get("proceed"):
                return prior
        ev = event or by_id[packet_id]
        decision_at = pd.Timestamp(ev["decision_at_utc"])
        pair = next(p for p, s in ((PAIR[ASSET_MAP[a]], ASSET_MAP[a]) for a in ASSET_MAP) if s == symbol)
        # symbol is BTCUSDT etc; PAIR maps that already
        pair = PAIR[symbol]
        gate = evaluate_gate(frames[pair], decision_at, side)
        notes = dict(prior.get("notes") or {}) if previous is not None else {}
        notes["htf_gate"] = {
            "eligible": gate["eligible"],
            "reason": gate["reason"],
            "extension_atr": gate["extension_atr"],
            "four_hour_trend_agrees": gate["four_hour_trend_agrees"],
        }
        if not gate["eligible"]:
            return {"proceed": False, "reason": f"htf_gate:{gate['reason']}", "max_qty": None, "notes": notes}
        out = {"proceed": True, "reason": "ok", "max_qty": prior.get("max_qty") if previous is not None else None, "notes": notes}
        if previous is not None and prior.get("max_qty") is not None:
            out["max_qty"] = prior["max_qty"]
        return out

    return entry_prepare


def _taipei(ts) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("Asia/Taipei").strftime("%Y-%m-%d %H:%M") + " (台北时间)"


def audit_trade(frames, ev: dict, side: str, symbol: str, fill: float, stop: float,
                entry_at, exit_at, horizon_at) -> dict[str, Any]:
    decision_at = pd.Timestamp(ev["decision_at_utc"])
    pair = PAIR[symbol]
    gate = evaluate_gate(frames[pair], decision_at, side)
    path = path_diagnostics(
        frames[pair], fill=fill, stop=stop, side=side,
        entry_at=pd.Timestamp(entry_at), exit_at=pd.Timestamp(exit_at),
        horizon_at=pd.Timestamp(horizon_at),
    )
    return {
        "packet_id": ev["packet_id"],
        "symbol": symbol,
        "side": side,
        "decision_at_taipei": _taipei(decision_at),
        "entry_at_taipei": _taipei(entry_at),
        "exit_at_taipei": _taipei(exit_at),
        "gate": gate,
        "path": path,
        "causal_note": (
            "These are observed price-path facts. They do not identify a unique cause "
            "(for example news, squeeze, or random path). Macro and news were missing at entry."
        ),
    }


def summarize_lane(original: dict, gated: dict) -> dict[str, Any]:
    orig_trades = {t["packet_id"]: t for t in original.get("closed_trade_summaries") or [] if not t.get("rejected")}
    gated_blocked = [a for a in gated.get("audits") or [] if str(a.get("rejected") or "").startswith("htf_gate:")]
    gated_filled = {t["packet_id"]: t for t in gated.get("closed_trade_summaries") or [] if not t.get("rejected")}
    avoided_winners, avoided_losers, still, blocked_ids = [], [], [], []
    for pid, t in orig_trades.items():
        net = t.get("net_usd")
        if pid not in gated_filled:
            blocked_ids.append(pid)
            item = {"packet_id": pid, "symbol": t.get("symbol"), "side": t.get("side"), "net_usd": net}
            if net is not None and net > 0.01:
                avoided_winners.append(item)
            elif net is not None and net < -0.01:
                avoided_losers.append(item)
        else:
            still.append({"packet_id": pid, "net_usd_original": net, "net_usd_gated": gated_filled[pid].get("net_usd")})
    winrate = gated.get("winrate_pct")
    if gated.get("closed_trades", 0) == 0:
        winrate = None
    opportunity = sum(float(x["net_usd"]) for x in avoided_winners + avoided_losers if x.get("net_usd") is not None)
    return {
        "original": {
            "final_equity": original.get("final_equity"),
            "return_pct": original.get("return_pct"),
            "max_drawdown_pct": original.get("max_drawdown_pct"),
            "closed_trades": original.get("closed_trades"),
            "wins": original.get("wins"),
            "losses": original.get("losses"),
            "fees": original.get("fees"),
            "funding_net_paid": original.get("funding_net_paid"),
        },
        "gated": {
            "final_equity": gated.get("final_equity"),
            "return_pct": gated.get("return_pct"),
            "max_drawdown_pct": gated.get("max_drawdown_pct"),
            "closed_trades": gated.get("closed_trades"),
            "wins": gated.get("wins"),
            "losses": gated.get("losses"),
            "fees": gated.get("fees"),
            "funding_net_paid": gated.get("funding_net_paid"),
            "winrate_pct": winrate,
            "rejections": gated.get("rejections"),
            "reconciliation_ok": gated.get("reconciliation_ok"),
            "ledger_ok": gated.get("ledger_ok"),
        },
        "blocked_packet_ids": blocked_ids,
        "avoided_winners": avoided_winners,
        "avoided_losers": avoided_losers,
        "still_taken": still,
        "opportunity_cost_net_usd_of_blocked": opportunity,
        "sample_directional_candidates": len(orig_trades),
        "if_all_blocked_interpretation": "0% return and undefined win rate are not proof of profitability",
    }


def eligibility_census(frames, sealed_events) -> dict[str, Any]:
    """Finite diagnostic: at the same 10 clocks, which asset/side pairs the gate would allow.

    Not a traded strategy and not comparable 1:1 with research decisions.
    """
    rows = []
    for ev in sealed_events:
        for alias, sym in ASSET_MAP.items():
            for side in ("long", "short"):
                g = evaluate_gate(frames[PAIR[sym]], pd.Timestamp(ev["decision_at_utc"]), side)
                rows.append({
                    "packet_id": ev["packet_id"], "alias": alias, "symbol": sym, "side": side,
                    "eligible": g["eligible"], "reason": g["reason"],
                    "extension_atr": g["extension_atr"],
                    "four_hour_trend_agrees": g["four_hour_trend_agrees"],
                })
    allowed = [r for r in rows if r["eligible"]]
    return {
        "clocks": 10,
        "evaluated_pairs": len(rows),
        "eligible_pairs": len(allowed),
        "eligible": allowed,
        "note": "Rule-generated eligibility census only; not executed and not a second backtest.",
    }


def run_experiment(*, manual_db: Path = MANUAL_GATE_DB, independent_db: Path = INDEPENDENT_GATE_DB) -> dict[str, Any]:
    if manual_db.resolve() == independent_db.resolve():
        raise ValueError("the two replay databases must be distinct")
    protected = {p.resolve() for p in (FORBIDDEN_DB, MANUAL_DB, INDEPENDENT_DB)}
    if manual_db.resolve() in protected or independent_db.resolve() in protected:
        raise ValueError("refusing protected research database paths")
    if manual_db.exists() or independent_db.exists():
        raise FileExistsError("both replay databases must be new paths")
    manifest = load_manifest()
    frames, *_ = load_cache(DATA_DIR)
    parent = json.loads((PACKET_DIR / "PACKETS_FOR_REVIEW.json").read_text())
    sealed = json.loads((PACKET_DIR / "BACKEND_SEALED.json").read_text())
    independent = json.loads((PACKET_DIR.parent / "historical_v1_independent" / "PHASE2_RESULTS.json").read_text())
    manual = json.loads((PACKET_DIR / "PHASE2_RESULTS.json").read_text())

    # Audits of the two frozen independent fills (and the overlapping manual ones share path).
    ev_by_id = {e["packet_id"]: e for e in sealed["events"]}
    audits = []
    for t in independent["closed_trade_summaries"]:
        ev = ev_by_id[t["packet_id"]]
        audits.append(audit_trade(
            frames, ev, t["side"], t["symbol"], float(t["entry"]), float(t["stop"]),
            t["entry_time"], t["exit_time"], ev["horizon_exit_at_utc"],
        ))

    census = eligibility_census(frames, sealed["events"])
    prepare = make_entry_prepare(frames, sealed["events"])
    from piia.intelligence.historical_independent import (
        funding_inclusive_entry_prepare, LABEL, AUTHOR, REVIEW_MARKER,
        load_independent_roles, PRIOR_DECISIONS, sha256_file,
    )
    # Manual lane: original six candidates + HTF gate, same paper path, new DB.
    manual_gated = run_phase2(
        packet_dir=PACKET_DIR, data_dir=DATA_DIR,
        db_path=manual_db, evidence_store=MANUAL_EV,
        entry_prepare=prepare,
        label="HTF_GATE_V1_MANUAL_LANE",
        author="diagnostic_gate_v1",
        review_marker="NOT_A_NEW_LLM_RUN",
        mode="deterministic_htf_gate_on_frozen_manual_decisions",
        label_quality="exploratory_already_seen_not_holdout",
        integrity_ok=True,
        protected_dbs=(FORBIDDEN_DB, MANUAL_DB, INDEPENDENT_DB),
        report_extra={"gate_id": GATE_ID, "lane": "manual_directional"},
    )
    # Independent lane: compose funding-inclusive size with HTF eligibility.
    ind_prepare = make_entry_prepare(frames, sealed["events"], previous=funding_inclusive_entry_prepare)
    loaded = load_independent_roles()
    sealed_bundle = (loaded["decisions"], loaded["parent"], sealed, loaded["combined_digest"])
    independent_gated = run_phase2(
        packet_dir=PACKET_DIR, data_dir=DATA_DIR,
        db_path=independent_db, evidence_store=INDEPENDENT_EV,
        decisions_bundle=sealed_bundle,
        label="HTF_GATE_V1_INDEPENDENT_LANE",
        author=AUTHOR,
        review_marker=REVIEW_MARKER,
        mode="deterministic_htf_gate_on_frozen_independent_decisions",
        label_quality="exploratory_already_seen_not_holdout",
        entry_prepare=ind_prepare,
        integrity_ok=True,
        protected_dbs=(FORBIDDEN_DB, MANUAL_DB, INDEPENDENT_DB),
        report_extra={"gate_id": GATE_ID, "lane": "independent_directional"},
    )

    report = {
        "gate_id": GATE_ID,
        "manifest": manifest,
        "hypothesis_optimized": False,
        "held_out": False,
        "already_seen_dataset": True,
        "llm_rerun": False,
        "past_research_unchanged": True,
        "stop_and_horizon_unchanged": True,
        "independent_trade_audits": audits,
        "manual_lane": summarize_lane(manual, manual_gated),
        "independent_lane": summarize_lane(independent, independent_gated),
        "eligibility_census_not_traded": census,
        "manual_gated_audits": [
            {k: a.get(k) for k in (
                "packet_id", "synthesizer_direction", "symbol", "action", "rejected",
                "net_usd", "exit_reason", "qty", "entry_gate",
            )}
            for a in manual_gated.get("audits") or []
        ],
        "independent_gated_audits": [
            {k: a.get(k) for k in (
                "packet_id", "synthesizer_direction", "symbol", "action", "rejected",
                "net_usd", "exit_reason", "qty", "entry_gate",
            )}
            for a in independent_gated.get("audits") or []
        ],
        "reconciliation": {
            "manual_ok": bool(manual_gated.get("reconciliation_ok") and manual_gated.get("ledger_ok")),
            "independent_ok": bool(independent_gated.get("reconciliation_ok") and independent_gated.get("ledger_ok")),
        },
        "limitations": manifest.get("limitations") or [],
    }
    # Explicit support judgment is filled after facts exist; keep the structure here.
    report["hypothesis_assessment"] = _assess(report)
    return report


def _assess(report: dict[str, Any]) -> dict[str, Any]:
    """State whether this tiny, already-seen sample can support H1. Not a fitness search."""
    m, i = report["manual_lane"], report["independent_lane"]
    # Support would require a pre-registered, larger, unseen sample. Here we only describe
    # what happened and say the evidence is insufficient either way.
    return {
        "status": "insufficient",
        "reason": (
            "The gate is a one-shot, already-seen, small-sample diagnostic. "
            "Avoided winners or losers on 2–6 prior trades cannot confirm or refute a lasting edge. "
            "No held-out or walk-forward evaluation was performed."
        ),
        "manual_return_change_pct_points": (
            None if m["gated"]["return_pct"] is None or m["original"]["return_pct"] is None
            else m["gated"]["return_pct"] - m["original"]["return_pct"]
        ),
        "independent_return_change_pct_points": (
            None if i["gated"]["return_pct"] is None or i["original"]["return_pct"] is None
            else i["gated"]["return_pct"] - i["original"]["return_pct"]
        ),
        "next_missing_evidence": [
            "A pre-registered application on data not used to design the research decisions or this gate.",
            "Macro, news, and on-chain context at entry, which remain missing.",
            "More than a handful of trades, including periods where 4h trend and 1h extension conflict in both directions.",
            "Verified venue liquidation and deeper-than-1h liquidity.",
        ],
    }


def write_experiment(report: dict[str, Any], out_dir: Path = OUT_DIR) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    js = out_dir / "RESULTS.json"
    js.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    lines = [
        "4h趋势一致且不过度延伸的入场过滤（H1，未优化，已见样本，不是样本外）",
        f"gate_id={GATE_ID}",
        "不修改既有研究写的结论，不改止损或72小时，不重跑模型。",
        "",
        "两笔独立运行已成交空单的入场与路径事实：",
        *(_audit_txt(a) for a in report["independent_trade_audits"]),
        "",
        _lane_txt("单助手6笔方向性候选", report["manual_lane"]),
        _lane_txt("独立角色2笔方向性候选", report["independent_lane"]),
        "",
        f"假设评估：{report['hypothesis_assessment']['status']}。{report['hypothesis_assessment']['reason']}",
        "若过滤后没有成交，0%收益且胜率未定义，不能当成赚钱证明。",
    ]
    txt = out_dir / "RESULTS.txt"
    txt.write_text("\n".join(lines) + "\n")
    return {"json": str(js), "txt": str(txt)}


def _audit_txt(a: dict[str, Any]) -> str:
    g, p, i1, i4 = a["gate"], a["path"], a["gate"]["one_hour"], a["gate"]["four_hour"]
    v = g["volume_quality"]
    h = p["hindsight_72h_ignoring_stop"]
    r = p["realized_until_exit"]
    return "\n".join([
        f"- {a['packet_id']} {a['symbol']} {a['side']}  决策/入场 {a['decision_at_taipei']}  离场 {a['exit_at_taipei']}（{p['hours_to_recorded_exit']:.0f}小时，止损）",
        f"  1h：收盘相对EMA20 {((i1['close']-i1['ema20'])/i1['atr14']):+.3f}ATR，相对EMA50 {((i1['close']-i1['ema50'])/i1['atr14']):+.3f}ATR；方向延伸 {g['extension_atr']:.3f}ATR（阈值1.0）。",
        f"  已完成4h：趋势一致={g['four_hour_trend_agrees']}；收盘相对EMA20 {((i4['close']-i4['ema20'])/i4['atr14']):+.3f}ATR，相对EMA50 {((i4['close']-i4['ema50'])/i4['atr14']):+.3f}ATR。",
        f"  最近20根1h区间位置={g['close_in_recent_20_1h_range']:.3f}；成交量相对24根中位数={v.get('last_vs_24h_median')}（未验证参与度）。",
        f"  实现路径 MFE={r['mfe_price']:.2f}  MAE={r['mae_price']:.2f} 价格点。事后72小时若忽略止损，到期开盘方向正确={h['direction_would_have_been_correct_at_horizon_open']}，但72h MAE={h['mae_price']:.2f}，不能作为放宽止损的理由。",
        f"  过滤结果：eligible={g['eligible']}。{a['causal_note']}",
    ])



def _lane_txt(title: str, lane: dict[str, Any]) -> str:
    o, g = lane["original"], lane["gated"]
    wr = "undefined" if g["closed_trades"] == 0 else (
        "n/a" if g["winrate_pct"] is None else f"{g['winrate_pct']:.1f}% ({g['wins']}/{g['closed_trades']})"
    )
    return "\n".join([
        f"{title}:",
        f"  原：{o['final_equity']:.2f}U  收益 {o['return_pct']:.2f}%  回撤 {o['max_drawdown_pct']:.2f}%  平仓 {o['wins']}/{o['closed_trades']}",
        f"  过滤后：{g['final_equity']:.2f}U  收益 {g['return_pct']:.2f}%  回撤 {g['max_drawdown_pct']:.2f}%  平仓胜率 {wr}  手续费 {g['fees']:.4f}U  资金费净支付 {g['funding_net_paid']:.4f}U",
        f"  拦住：{lane['blocked_packet_ids'] or '无'}",
        f"  避免的盈利单：{lane['avoided_winners'] or '无'}",
        f"  避免的亏损单：{lane['avoided_losers'] or '无'}",
        f"  被拦交易的净盈亏合计（机会成本口径）：{lane['opportunity_cost_net_usd_of_blocked']:.4f}U",
    ])


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Run the exploratory gate using new replay databases")
    parser.add_argument("--manual-db", type=Path, default=MANUAL_GATE_DB)
    parser.add_argument("--independent-db", type=Path, default=INDEPENDENT_GATE_DB)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    report = run_experiment(manual_db=args.manual_db, independent_db=args.independent_db)
    paths = write_experiment(report, out_dir=args.out_dir)
    print("wrote", paths["txt"])
    print("wrote", paths["json"])
    print("manual", report["manual_lane"]["gated"]["return_pct"], "independent", report["independent_lane"]["gated"]["return_pct"])
    print("assess", report["hypothesis_assessment"]["status"])
    print("recon", report["reconciliation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
