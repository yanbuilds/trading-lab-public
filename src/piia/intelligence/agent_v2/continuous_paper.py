"""Append-Only Continuous Prospective Paper Trading Engine (PIIA v2).

Maintains an immutable local SQLite audit ledger (`data/piia/piia_v2_continuous.sqlite`)
with SQLite triggers forbidding `UPDATE` and `DELETE` on `cycles`, `decision_traces`,
and `ledger_events`.

Two strictly separated paper books are tracked prospectively:
1. `shadow_auto_comparison`: Automatically simulates `M4_full_piia_v2` side-by-side
   against `B1_simple_ema_1h`, `B3_donchian_breakout_4h`, and `B0_cash` on newly
   closed hourly candles and public funding rates (strictly local research shadow,
   zero exchange connectivity).
2. `human_approved_paper`: Records unapproved PIIA v2 proposals (`approved=0`)
   that require an explicit human CLI/UI approval action before entering the
   human paper book. Live and exchange-demo orders are permanently disabled.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sqlite3
from typing import Any
import uuid

import pandas as pd

from piia.intelligence.agent_v2.pillars import (
    ROOT,
    SYMBOLS,
    build_bar_evidence_for_symbol,
    canonical_sha256,
    collect_live_six_pillar_snapshot,
    load_v2_config,
    prepare_market_feature_table,
)
from piia.intelligence.agent_v2.risk_gate import run_symbol_decision_trace
from trading_lab.perps.risk import exit_pnl, stop_fill
from trading_lab.perps.risk_v2 import resolve_spec

DEFAULT_LEDGER_PATH = ROOT / "data" / "piia" / "piia_v2_continuous.sqlite"
APP_ID = 0x50495632  # "PIV2"
SHADOW_ARMS = ("M4_full_piia_v2", "B1_simple_ema_1h", "B3_donchian_breakout_4h", "B0_cash")


def _init_db(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    existed = db_path.exists() and db_path.stat().st_size > 0
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    app = conn.execute("PRAGMA application_id").fetchone()[0]
    if existed and app not in (0, APP_ID):
        conn.close()
        raise ValueError(f"Unrelated SQLite database at {db_path}")
    if not existed or app == 0:
        conn.execute(f"PRAGMA application_id={APP_ID}")

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS cycles (
            cycle_id TEXT PRIMARY KEY,
            cutoff_hour_utc TEXT NOT NULL UNIQUE,
            recorded_at_utc TEXT NOT NULL,
            manifest_sha256 TEXT NOT NULL,
            mode TEXT NOT NULL,
            pillars_summary_json TEXT NOT NULL,
            cycle_hash TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS decision_traces (
            decision_id TEXT PRIMARY KEY,
            cycle_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            decision_utc TEXT NOT NULL,
            final_action TEXT NOT NULL,
            stand_aside_reason TEXT,
            trace_json TEXT NOT NULL,
            trace_hash TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS ledger_events (
            event_id TEXT PRIMARY KEY,
            recorded_at_utc TEXT NOT NULL,
            book_name TEXT NOT NULL,
            event_kind TEXT NOT NULL,
            symbol TEXT,
            payload_json TEXT NOT NULL,
            cash_after_usdt REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS human_proposals (
            proposal_id TEXT PRIMARY KEY,
            cycle_id TEXT NOT NULL,
            decision_id TEXT NOT NULL,
            created_at_utc TEXT NOT NULL,
            expires_at_utc TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            proposal_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS human_decisions (
            approval_id TEXT PRIMARY KEY,
            proposal_id TEXT NOT NULL UNIQUE,
            decided_at_utc TEXT NOT NULL,
            actor TEXT NOT NULL,
            choice TEXT NOT NULL,
            note TEXT
        );

        CREATE TRIGGER IF NOT EXISTS immutable_cycles_upd BEFORE UPDATE ON cycles
        BEGIN SELECT RAISE(ABORT, 'append-only cycles'); END;
        CREATE TRIGGER IF NOT EXISTS immutable_cycles_del BEFORE DELETE ON cycles
        BEGIN SELECT RAISE(ABORT, 'append-only cycles'); END;

        CREATE TRIGGER IF NOT EXISTS immutable_traces_upd BEFORE UPDATE ON decision_traces
        BEGIN SELECT RAISE(ABORT, 'append-only decision_traces'); END;
        CREATE TRIGGER IF NOT EXISTS immutable_traces_del BEFORE DELETE ON decision_traces
        BEGIN SELECT RAISE(ABORT, 'append-only decision_traces'); END;

        CREATE TRIGGER IF NOT EXISTS immutable_events_upd BEFORE UPDATE ON ledger_events
        BEGIN SELECT RAISE(ABORT, 'append-only ledger_events'); END;
        CREATE TRIGGER IF NOT EXISTS immutable_events_del BEFORE DELETE ON ledger_events
        BEGIN SELECT RAISE(ABORT, 'append-only ledger_events'); END;
        """
    )
    return conn


def _candles_to_df(candles: list[dict[str, Any]]) -> pd.DataFrame:
    rows, idx = [], []
    for c in candles:
        idx.append(pd.Timestamp(c["open_at_utc"]).tz_convert("UTC"))
        rows.append({
            "open": float(c["open"]),
            "high": float(c["high"]),
            "low": float(c["low"]),
            "close": float(c["close"]),
            "volume": float(c.get("volume") or 0.0),
        })
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="date")).sort_index()


def _reconstruct_book_state(conn: sqlite3.Connection, book_name: str, default_cash: float) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT event_kind, symbol, payload_json, cash_after_usdt FROM ledger_events WHERE book_name = ? ORDER BY rowid ASC",
        (book_name,),
    ).fetchall()
    if not rows:
        return {"cash": default_cash, "open_pos": None, "closed_trades": [], "starting_cash": default_cash}

    cash = float(rows[-1]["cash_after_usdt"])
    starting_cash = float(json.loads(rows[0]["payload_json"]).get("starting_cash", default_cash))
    open_pos = None
    closed_trades = []
    for r in rows:
        kind = r["event_kind"]
        payload = json.loads(r["payload_json"])
        if kind == "open_position":
            open_pos = payload
        elif kind == "update_stop" and open_pos is not None:
            open_pos["stop"] = payload["new_stop"]
            open_pos["be_locked"] = payload.get("be_locked", open_pos.get("be_locked", False))
        elif kind == "funding" and open_pos is not None:
            open_pos["funding"] = open_pos.get("funding", 0.0) + float(payload["amount_paid"])
        elif kind == "close_position":
            closed_trades.append(payload)
            open_pos = None
    return {
        "starting_cash": starting_cash,
        "cash": cash,
        "open_pos": open_pos,
        "closed_trades": closed_trades,
    }


def _append_event(
    conn: sqlite3.Connection,
    book_name: str,
    event_kind: str,
    symbol: str | None,
    payload: dict[str, Any],
    cash_after: float,
    now_iso: str,
) -> None:
    conn.execute(
        "INSERT INTO ledger_events(event_id, recorded_at_utc, book_name, event_kind, symbol, payload_json, cash_after_usdt) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), now_iso, book_name, event_kind, symbol, json.dumps(payload, sort_keys=True, default=str), float(cash_after)),
    )


def _reconcile_open_position_with_bars(
    conn: sqlite3.Connection,
    book_name: str,
    state: dict[str, Any],
    prepared_tables: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
    now_iso: str,
) -> dict[str, Any]:
    """Advance an open paper position across newly closed 1h bars since its entry."""
    pos = state["open_pos"]
    if pos is None:
        return state
    sym = pos["symbol"]
    df = prepared_tables.get(sym)
    if df is None or df.empty:
        return state

    sp = resolve_spec(sym).to_v1_spec()
    slip = float(cfg["risk_and_costs"]["slippage_each_side_base"])
    side_i = 1 if pos["side"] == "long" else -1
    last_checked = pd.Timestamp(pos.get("last_reconciled_bar_open_utc") or pos["entry_bar_open_utc"]).tz_convert("UTC")
    cash = float(state["cash"])

    new_bars = df.loc[df.index > last_checked]
    for ts, row in new_bars.iterrows():
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        # Gap check at bar open
        if (side_i == 1 and o <= pos["stop"]) or (side_i == -1 and o >= pos["stop"]):
            fill_px = o * (1.0 - side_i * slip)
            efee = pos["qty"] * sp.contract_size * fill_px * sp.taker
            gross = exit_pnl(side_i, pos["qty"], pos["entry"], fill_px, sp)
            cash += gross - efee
            closed_rec = {
                "symbol": sym, "side": pos["side"], "qty": pos["qty"],
                "entry": pos["entry"], "exit": fill_px, "exit_time_utc": ts.isoformat(),
                "net_pnl": gross - pos["entry_fee"] - efee - pos.get("funding", 0.0),
                "exit_reason": "gap_stop",
            }
            _append_event(conn, book_name, "close_position", sym, closed_rec, cash, now_iso)
            state["cash"] = cash
            state["open_pos"] = None
            state["closed_trades"].append(closed_rec)
            return state

        # Intrabar stop check
        hit = stop_fill(side_i, pos["stop"], o, l, h)
        if hit is not None:
            fill_px = hit * (1.0 - side_i * slip)
            efee = pos["qty"] * sp.contract_size * fill_px * sp.taker
            gross = exit_pnl(side_i, pos["qty"], pos["entry"], fill_px, sp)
            cash += gross - efee
            closed_rec = {
                "symbol": sym, "side": pos["side"], "qty": pos["qty"],
                "entry": pos["entry"], "exit": fill_px, "exit_time_utc": (ts + pd.Timedelta(hours=1)).isoformat(),
                "net_pnl": gross - pos["entry_fee"] - efee - pos.get("funding", 0.0),
                "exit_reason": "stop_loss",
            }
            _append_event(conn, book_name, "close_position", sym, closed_rec, cash, now_iso)
            state["cash"] = cash
            state["open_pos"] = None
            state["closed_trades"].append(closed_rec)
            return state

        # Time-stop check
        max_h = int(pos.get("max_hold_hours") or 72)
        if (ts + pd.Timedelta(hours=1)) - pd.Timestamp(pos["entry_bar_open_utc"]).tz_convert("UTC") >= pd.Timedelta(hours=max_h):
            fill_px = c * (1.0 - side_i * slip)
            efee = pos["qty"] * sp.contract_size * fill_px * sp.taker
            gross = exit_pnl(side_i, pos["qty"], pos["entry"], fill_px, sp)
            cash += gross - efee
            closed_rec = {
                "symbol": sym, "side": pos["side"], "qty": pos["qty"],
                "entry": pos["entry"], "exit": fill_px, "exit_time_utc": (ts + pd.Timedelta(hours=1)).isoformat(),
                "net_pnl": gross - pos["entry_fee"] - efee - pos.get("funding", 0.0),
                "exit_reason": "time_stop",
            }
            _append_event(conn, book_name, "close_position", sym, closed_rec, cash, now_iso)
            state["cash"] = cash
            state["open_pos"] = None
            state["closed_trades"].append(closed_rec)
            return state

        # Breakeven ratchet check
        if pos.get("be_trigger") and not pos.get("be_locked"):
            reached = (side_i == 1 and h >= float(pos["be_trigger"])) or (side_i == -1 and l <= float(pos["be_trigger"]))
            if reached:
                be_px = float(pos["be_stop"])
                if (side_i == 1 and be_px > pos["stop"]) or (side_i == -1 and be_px < pos["stop"]):
                    pos["stop"] = be_px
                    pos["be_locked"] = True
                    _append_event(conn, book_name, "update_stop", sym, {"new_stop": be_px, "be_locked": True, "at_utc": (ts + pd.Timedelta(hours=1)).isoformat()}, cash, now_iso)

        pos["last_reconciled_bar_open_utc"] = ts.isoformat()
    return state


def run_prospective_cycle(
    *,
    db_path: Path = DEFAULT_LEDGER_PATH,
    starting_cash: float = 500.0,
    demo_mode: bool = False,
    snapshot_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one prospective PIIA v2 cycle: verify freeze -> collect 6 pillars -> reconcile -> trace -> append."""
    from piia.intelligence.agent_v2.manifest import verify_piia_v2_frozen
    manifest_check = verify_piia_v2_frozen()
    if not manifest_check["ok"]:
        raise ValueError(f"Frozen PIIA v2 manifest verification failed: {manifest_check['issues']}")

    cfg = load_v2_config()
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    if snapshot_override is not None:
        collected = snapshot_override
        mode = "supplied_snapshot"
    elif demo_mode:
        from piia.intelligence.trade_assistant import generate_demo_snapshot
        demo_snap = generate_demo_snapshot(now=now)
        collected = {
            "collected_at_utc": now_iso,
            "market_snapshot": demo_snap,
            "books_snapshot": {"ok": True, "instruments": {s: {"orderbook": demo_snap["instruments"][s]["orderbook"], "contract": None} for s in SYMBOLS}},
            "tiers_snapshot": {"ok": False, "instruments": {}},
            "macro_context": {"status": "unavailable", "historical_point_in_time_lagged": False, "macro_tightening_headwind": False, "macro_easing_tailwind": False},
            "event_calendar_context": {"status": "demo_unavailable", "historical_replay_supported": False},
            "flows_positioning_context": {"status": "demo_unavailable", "liquidity_state": "unknown", "historical_replay_supported": False},
            "onchain_context": {"status": "demo_unavailable", "historical_replay_supported": False},
            "polymarket_context": {"status": "demo_unavailable", "historical_replay_supported": False},
            "chronos_forecast": {"weights_ran": False, "forecasts": {}},
        }
        mode = "SYNTHETIC_DEMO_NOT_LIVE"
    else:
        cycle_art_dir = ROOT / "artifacts" / "piia_v2_cycles" / now.strftime("%Y%m%dT%H%M%SZ")
        collected = collect_live_six_pillar_snapshot(SYMBOLS, snapshot_dir=cycle_art_dir)
        mode = "live_prospective_public_6_pillars"

    mkt_snap = collected["market_snapshot"]
    prepared_tables: dict[str, pd.DataFrame] = {}
    latest_closes: list[pd.Timestamp] = []
    for sym in SYMBOLS:
        inst = (mkt_snap.get("instruments") or {}).get(sym) or {}
        df = _candles_to_df(inst.get("candles_closed_1h") or [])
        if not df.empty:
            prepared_tables[sym] = prepare_market_feature_table(df, cfg=cfg)
            latest_closes.append(df.index[-1] + pd.Timedelta(hours=1))

    if len(prepared_tables) != len(SYMBOLS):
        raise ValueError("Incomplete closed 1h candle series across BTCUSDT/XAUUSDT/NDX100USDT")

    cutoff_hour = min(latest_closes)
    cutoff_iso = cutoff_hour.isoformat()

    conn = _init_db(db_path)
    try:
        existing_cycle = conn.execute("SELECT cycle_id FROM cycles WHERE cutoff_hour_utc = ?", (cutoff_iso,)).fetchone()
        if existing_cycle is not None:
            return {
                "status": "already_recorded_for_hour",
                "cycle_id": existing_cycle["cycle_id"],
                "cutoff_hour_utc": cutoff_iso,
                "db_path": str(db_path),
                "order_permissions": False,
            }

        # Initialize books if empty & reconcile existing open positions
        books_to_track = [*SHADOW_ARMS, "human_approved_paper"]
        book_states: dict[str, dict[str, Any]] = {}
        for bname in books_to_track:
            st = _reconstruct_book_state(conn, bname, starting_cash)
            if not conn.execute("SELECT 1 FROM ledger_events WHERE book_name = ? LIMIT 1", (bname,)).fetchone():
                _append_event(conn, bname, "init_book", None, {"starting_cash": starting_cash, "mode": mode}, starting_cash, now_iso)
            st = _reconcile_open_position_with_bars(conn, bname, st, prepared_tables, cfg, now_iso)
            book_states[bname] = st

        cycle_id = str(uuid.uuid4())
        traces_by_symbol: dict[str, dict[str, Any]] = {}
        human_proposals_created: list[dict[str, Any]] = []

        m4_state = book_states["M4_full_piia_v2"]
        m4_open_count = 1 if m4_state["open_pos"] is not None else 0

        for sym in SYMBOLS:
            df_sym = prepared_tables[sym]
            sig_row = df_sym.iloc[-1]
            prev_c = float(df_sym.iloc[-2]["close"]) if len(df_sym) >= 2 else None
            inst_snap = (mkt_snap.get("instruments") or {}).get(sym) or {}
            f_obj = inst_snap.get("funding") or {}
            f_rate = None
            if f_obj.get("funding_rate") not in (None, ""):
                try:
                    f_rate = float(f_obj["funding_rate"])
                except (TypeError, ValueError):
                    pass

            b_inst = ((collected.get("books_snapshot") or {}).get("instruments") or {}).get(sym) or {}
            t_inst = ((collected.get("tiers_snapshot") or {}).get("instruments") or {}).get(sym)

            bundle = build_bar_evidence_for_symbol(
                sym,
                feature_row=sig_row,
                prev_close=prev_c,
                decision_utc=cutoff_hour.to_pydatetime(),
                funding_rate_8h=f_rate,
                macro_ctx=collected.get("macro_context"),
                external_live_pillars={
                    "orderbook": b_inst.get("orderbook"),
                    "tiers": t_inst,
                    "open_interest": inst_snap.get("open_interest"),
                    "flows_positioning": collected.get("flows_positioning_context"),
                    "onchain": collected.get("onchain_context"),
                    "polymarket": collected.get("polymarket_context"),
                    "event_calendar": collected.get("event_calendar_context"),
                    "chronos_forecast": collected.get("chronos_forecast"),
                },
            )

            trace = run_symbol_decision_trace(
                bundle,
                equity_usdt=float(m4_state["cash"]),
                free_cash_usdt=float(m4_state["cash"]),
                open_positions_count=m4_open_count,
                kill_switch_halted=False,
                spec_override=b_inst.get("contract"),
                enable_macro=True,
                enable_forecast=True,
                enable_critic_veto=True,
                enable_live_external=(not demo_mode),
                cfg=cfg,
            )
            traces_by_symbol[sym] = trace

            conn.execute(
                "INSERT INTO decision_traces(decision_id, cycle_id, symbol, decision_utc, final_action, stand_aside_reason, trace_json, trace_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trace["decision_id"],
                    cycle_id,
                    sym,
                    trace["decision_utc"],
                    trace["final_action"],
                    trace["stand_aside_reason"],
                    json.dumps(trace, sort_keys=True, default=str),
                    trace["trace_hash"],
                ),
            )

            if trace["final_action"] in {"long", "short"}:
                prop_id = str(uuid.uuid4())
                expires = (now + timedelta(minutes=15)).isoformat()
                prop_body = {
                    "proposal_id": prop_id,
                    "cycle_id": cycle_id,
                    "decision_id": trace["decision_id"],
                    "symbol": sym,
                    "side": trace["final_action"],
                    "entry_plan": trace["programmatic_risk_gate"]["entry_plan"],
                    "exit_plan": trace["programmatic_risk_gate"]["exit_plan"],
                    "position_sizing": trace["programmatic_risk_gate"]["position_sizing"],
                    "cost_and_risk": trace["programmatic_risk_gate"]["cost_and_risk"],
                    "approved": False,
                    "live_orders": False,
                    "expires_at_utc": expires,
                }
                conn.execute(
                    "INSERT INTO human_proposals(proposal_id, cycle_id, decision_id, created_at_utc, expires_at_utc, symbol, side, proposal_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (prop_id, cycle_id, trace["decision_id"], now_iso, expires, sym, trace["final_action"], json.dumps(prop_body, sort_keys=True, default=str)),
                )
                human_proposals_created.append(prop_body)

        # Open shadow position on `M4_full_piia_v2` if flat and a candidate passed
        if m4_state["open_pos"] is None:
            passed = [
                (float(tr["synthesizer"]["alignment_score"]), sym, tr)
                for sym, tr in traces_by_symbol.items()
                if tr["final_action"] in {"long", "short"}
            ]
            if passed:
                passed.sort(key=lambda x: (-x[0], x[1]))
                _, best_sym, best_tr = passed[0]
                rg = best_tr["programmatic_risk_gate"]
                fee = float(rg["cost_and_risk"]["entry_taker_fee_usdt"])
                new_cash = float(m4_state["cash"]) - fee
                open_payload = {
                    "symbol": best_sym,
                    "side": best_tr["final_action"],
                    "qty": float(rg["position_sizing"]["qty_contracts"]),
                    "entry": float(rg["entry_plan"]["assumed_fill_with_slippage"]),
                    "stop": float(rg["exit_plan"]["initial_stop_loss"]),
                    "be_stop": float(rg["exit_plan"]["fee_adjusted_breakeven_stop"]),
                    "be_trigger": float(rg["exit_plan"]["breakeven_ratchet_trigger_price"]),
                    "be_locked": False,
                    "max_hold_hours": int(rg["exit_plan"]["max_hold_hours"]),
                    "entry_fee": fee,
                    "funding": 0.0,
                    "entry_bar_open_utc": cutoff_iso,
                    "last_reconciled_bar_open_utc": prepared_tables[best_sym].index[-1].isoformat(),
                    "decision_id": best_tr["decision_id"],
                }
                _append_event(conn, "M4_full_piia_v2", "open_position", best_sym, open_payload, new_cash, now_iso)

        # Also evaluate simple shadow baselines (`B1_simple_ema_1h`, `B3_donchian_breakout_4h`) on identical rules
        from piia.intelligence.agent_v2.ablation import _baseline_candidate_from_row
        from piia.intelligence.agent_v2.risk_gate import evaluate_programmatic_risk
        for base_arm in ("B1_simple_ema_1h", "B3_donchian_breakout_4h"):
            b_state = book_states[base_arm]
            if b_state["open_pos"] is not None:
                continue
            for sym in SYMBOLS:
                df_sym = prepared_tables[sym]
                if len(df_sym) < 2:
                    continue
                cand = _baseline_candidate_from_row(base_arm, sym, df_sym.iloc[-1], df_sym.iloc[-2])
                if cand.get("candidate_direction") not in {"long", "short"}:
                    continue
                mkt_pil = {
                    "ohlc": {"close": float(df_sym.iloc[-1]["close"])},
                    "atr14": float(df_sym.iloc[-1]["atr14"]) if pd.notna(df_sym.iloc[-1]["atr14"]) else 0.0,
                    "funding_rate_8h": None,
                }
                rg_b = evaluate_programmatic_risk(
                    symbol=sym,
                    synth_out=cand,
                    market_pillar=mkt_pil,
                    equity_usdt=float(b_state["cash"]),
                    free_cash_usdt=float(b_state["cash"]),
                    open_positions_count=0,
                    kill_switch_halted=False,
                    cfg=cfg,
                )
                if rg_b["approved_by_risk_gate"]:
                    fee_b = float(rg_b["cost_and_risk"]["entry_taker_fee_usdt"])
                    new_cash_b = float(b_state["cash"]) - fee_b
                    open_b = {
                        "symbol": sym,
                        "side": rg_b["final_action"],
                        "qty": float(rg_b["position_sizing"]["qty_contracts"]),
                        "entry": float(rg_b["entry_plan"]["assumed_fill_with_slippage"]),
                        "stop": float(rg_b["exit_plan"]["initial_stop_loss"]),
                        "be_stop": None,
                        "be_trigger": None,
                        "be_locked": False,
                        "max_hold_hours": 168,
                        "entry_fee": fee_b,
                        "funding": 0.0,
                        "entry_bar_open_utc": cutoff_iso,
                        "last_reconciled_bar_open_utc": df_sym.index[-1].isoformat(),
                        "decision_id": f"{base_arm}:{cutoff_iso}:{sym}",
                    }
                    _append_event(conn, base_arm, "open_position", sym, open_b, new_cash_b, now_iso)
                    break

        pillars_summary = {
            "macro_status": (collected.get("macro_context") or {}).get("status"),
            "btc_etf_usable": ((collected.get("flows_positioning_context") or {}).get("btc_etf_flows") or {}).get("usable"),
            "gold_cot_usable": ((collected.get("flows_positioning_context") or {}).get("gold_cftc_cot") or {}).get("usable"),
            "liquidity_state": (collected.get("flows_positioning_context") or {}).get("liquidity_state"),
            "btc_onchain_status": (collected.get("onchain_context") or {}).get("status"),
            "polymarket_status": (collected.get("polymarket_context") or {}).get("status"),
            "chronos_weights_ran": bool((collected.get("chronos_forecast") or {}).get("weights_ran")),
            "books_ok": bool((collected.get("books_snapshot") or {}).get("ok")),
            "tiers_ok": bool((collected.get("tiers_snapshot") or {}).get("ok")),
        }
        cycle_doc = {
            "cycle_id": cycle_id,
            "cutoff_hour_utc": cutoff_iso,
            "recorded_at_utc": now_iso,
            "manifest_sha256": manifest_check["manifest_sha256"],
            "mode": mode,
            "pillars_summary": pillars_summary,
            "decision_hashes": {s: tr["trace_hash"] for s, tr in traces_by_symbol.items()},
        }
        c_hash = canonical_sha256(cycle_doc)
        conn.execute(
            "INSERT INTO cycles(cycle_id, cutoff_hour_utc, recorded_at_utc, manifest_sha256, mode, pillars_summary_json, cycle_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (cycle_id, cutoff_iso, now_iso, manifest_check["manifest_sha256"], mode, json.dumps(pillars_summary, sort_keys=True), c_hash),
        )
        conn.commit()

        return {
            "status": "recorded_prospective_cycle",
            "cycle_id": cycle_id,
            "cutoff_hour_utc": cutoff_iso,
            "mode": mode,
            "manifest_sha256": manifest_check["manifest_sha256"],
            "pillars_summary": pillars_summary,
            "decisions": {
                s: {
                    "decision_id": tr["decision_id"],
                    "final_action": tr["final_action"],
                    "stand_aside_reason": tr["stand_aside_reason"],
                    "regime": tr["researcher"]["regime"],
                    "disagreements": tr["synthesizer"]["disagreements"],
                    "entry_plan": tr["programmatic_risk_gate"]["entry_plan"],
                    "exit_plan": tr["programmatic_risk_gate"]["exit_plan"],
                    "position_sizing": tr["programmatic_risk_gate"]["position_sizing"],
                    "cost_and_risk": tr["programmatic_risk_gate"]["cost_and_risk"],
                }
                for s, tr in traces_by_symbol.items()
            },
            "unapproved_human_proposals": human_proposals_created,
            "db_path": str(db_path),
            "live_orders_submitted": False,
            "order_permissions": False,
        }
    finally:
        conn.close()


def approve_human_proposal(
    proposal_id: str,
    *,
    choice: str,
    actor: str = "local-investor",
    note: str = "",
    db_path: Path = DEFAULT_LEDGER_PATH,
) -> dict[str, Any]:
    """Explicit human approval/rejection for `human_approved_paper` book (never live orders)."""
    if choice not in {"approve", "reject"}:
        raise ValueError("choice must be 'approve' or 'reject'")
    conn = _init_db(db_path)
    try:
        prop_row = conn.execute("SELECT * FROM human_proposals WHERE proposal_id = ?", (proposal_id,)).fetchone()
        if not prop_row:
            raise ValueError(f"Proposal not found: {proposal_id}")
        existing = conn.execute("SELECT * FROM human_decisions WHERE proposal_id = ?", (proposal_id,)).fetchone()
        if existing:
            raise ValueError(f"Proposal {proposal_id} was already decided ({existing['choice']})")

        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        expires = datetime.fromisoformat(prop_row["expires_at_utc"])
        if choice == "approve" and now > expires:
            raise ValueError("Proposal expired; run a fresh agent cycle before approving")

        approval_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO human_decisions(approval_id, proposal_id, decided_at_utc, actor, choice, note) VALUES (?, ?, ?, ?, ?, ?)",
            (approval_id, proposal_id, now_iso, actor, choice, note),
        )

        prop = json.loads(prop_row["proposal_json"])
        st = _reconstruct_book_state(conn, "human_approved_paper", 500.0)
        opened = None
        if choice == "approve":
            if st["open_pos"] is not None:
                raise ValueError("human_approved_paper already has an open position (max_open_positions=1)")
            fee = float(prop["cost_and_risk"]["entry_taker_fee_usdt"])
            new_cash = float(st["cash"]) - fee
            opened = {
                "symbol": prop["symbol"],
                "side": prop["side"],
                "qty": float(prop["position_sizing"]["qty_contracts"]),
                "entry": float(prop["entry_plan"]["assumed_fill_with_slippage"]),
                "stop": float(prop["exit_plan"]["initial_stop_loss"]),
                "be_stop": float(prop["exit_plan"]["fee_adjusted_breakeven_stop"]),
                "be_trigger": float(prop["exit_plan"]["breakeven_ratchet_trigger_price"]),
                "be_locked": False,
                "max_hold_hours": int(prop["exit_plan"]["max_hold_hours"]),
                "entry_fee": fee,
                "funding": 0.0,
                "entry_bar_open_utc": now_iso,
                "last_reconciled_bar_open_utc": now_iso,
                "decision_id": prop["decision_id"],
                "approval_id": approval_id,
            }
            _append_event(conn, "human_approved_paper", "open_position", prop["symbol"], opened, new_cash, now_iso)
        conn.commit()
        return {
            "approval_id": approval_id,
            "proposal_id": proposal_id,
            "choice": choice,
            "paper_position_opened": opened,
            "live_orders_submitted": False,
            "order_permissions": False,
        }
    finally:
        conn.close()


def get_continuous_paper_status(db_path: Path = DEFAULT_LEDGER_PATH) -> dict[str, Any]:
    """Read-only summary of recorded prospective cycles, latest traces, and paper books."""
    db_path = Path(db_path)
    if not db_path.is_file():
        return {
            "initialized": False,
            "cycles_recorded": 0,
            "db_path": str(db_path),
            "books": {},
            "latest_decisions": [],
            "order_permissions": False,
        }
    conn = _init_db(db_path)
    try:
        cycles = conn.execute("SELECT * FROM cycles ORDER BY cutoff_hour_utc DESC").fetchall()
        books_summary = {}
        for bname in (*SHADOW_ARMS, "human_approved_paper"):
            st = _reconstruct_book_state(conn, bname, 500.0)
            books_summary[bname] = {
                "starting_cash_usdt": st["starting_cash"],
                "cash_usdt": round(st["cash"], 4),
                "closed_trades": len(st["closed_trades"]),
                "realized_net_pnl_usdt": round(sum(float(t["net_pnl"]) for t in st["closed_trades"]), 4),
                "open_position": st["open_pos"],
            }

        latest_traces = []
        if cycles:
            last_cid = cycles[0]["cycle_id"]
            t_rows = conn.execute("SELECT symbol, final_action, stand_aside_reason, trace_json, trace_hash FROM decision_traces WHERE cycle_id = ? ORDER BY symbol ASC", (last_cid,)).fetchall()
            for tr in t_rows:
                parsed = json.loads(tr["trace_json"])
                latest_traces.append({
                    "symbol": tr["symbol"],
                    "final_action": tr["final_action"],
                    "stand_aside_reason": tr["stand_aside_reason"],
                    "regime": parsed["researcher"]["regime"],
                    "disagreements": parsed["synthesizer"]["disagreements"],
                    "entry_plan": parsed["programmatic_risk_gate"]["entry_plan"],
                    "exit_plan": parsed["programmatic_risk_gate"]["exit_plan"],
                    "position_sizing": parsed["programmatic_risk_gate"]["position_sizing"],
                    "cost_and_risk": parsed["programmatic_risk_gate"]["cost_and_risk"],
                    "trace_hash": tr["trace_hash"],
                })

        pending = []
        p_rows = conn.execute(
            """
            SELECT p.proposal_id, p.symbol, p.side, p.created_at_utc, p.expires_at_utc, p.proposal_json
            FROM human_proposals p
            LEFT JOIN human_decisions d ON p.proposal_id = d.proposal_id
            WHERE d.proposal_id IS NULL
            ORDER BY p.created_at_utc DESC
            LIMIT 10
            """
        ).fetchall()
        for pr in p_rows:
            pending.append(json.loads(pr["proposal_json"]))

        return {
            "initialized": True,
            "db_path": str(db_path),
            "cycles_recorded": len(cycles),
            "first_cutoff_utc": cycles[-1]["cutoff_hour_utc"] if cycles else None,
            "latest_cutoff_utc": cycles[0]["cutoff_hour_utc"] if cycles else None,
            "latest_mode": cycles[0]["mode"] if cycles else None,
            "latest_pillars_summary": json.loads(cycles[0]["pillars_summary_json"]) if cycles else None,
            "books": books_summary,
            "latest_decisions": latest_traces,
            "pending_human_proposals": pending,
            "order_permissions": False,
        }
    finally:
        conn.close()
