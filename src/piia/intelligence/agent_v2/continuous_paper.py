"""Append-Only Continuous Prospective Paper Trading Engine (PIIA v2.1).

Maintains immutable local SQLite audit ledgers, strictly separated by evidence kind:
`data/piia/piia_v2_2_live_prospective.sqlite` (live public collection only) and
`data/piia/piia_v2_2_demo_synthetic.sqlite` (synthetic demo only),
with SQLite triggers forbidding `UPDATE` and `DELETE` on `cycles`, `decision_traces`,
`ledger_events`, `human_proposals`, `human_decisions`, and `position_settlements`.

Key accounting & risk invariants:
1. Persistent `3%` daily / `10%` total loss-halt state reconstructed from the append-only
   event log on every startup and passed into programmatic risk gates (`kill_switch_halted`).
2. Contiguous settled public funding accounting keyed by settlement timestamp, position side,
   and mark-proxy notional; missing or unverified funding rates are logged explicitly as
   `funding_unknown` and block net PnL/return claims (`net_return_claim_allowed=False`).
3. Persisted closed-bar processing cursor (`last_processed_bar_close_utc`) plus explicit
   `stop_effective_from_utc` so restarts or overlapping historical bars never double-charge
   funding, double-close positions, or retroactively trigger newer ratcheted stops.
4. Pre-approval revalidation of proposal expiry, live/fresh quote age (`<= 5 min`), price
   drift/invalidation, open-position exclusivity, current cash balance, and halt state.
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
    DEFAULT_FUNDING_INTERVAL_HOURS,
    ROOT,
    SYMBOLS,
    build_bar_evidence_for_symbol,
    canonical_sha256,
    collect_live_six_pillar_snapshot,
    fetch_public_settled_funding_recent,
    load_v2_config,
    prepare_market_feature_table,
)
from piia.intelligence.agent_v2.risk_gate import (
    evaluate_programmatic_risk,
    run_symbol_decision_trace,
)
from trading_lab.perps.risk import exit_pnl, stop_fill
from trading_lab.perps.risk_v2 import resolve_spec

# Strictly separated ledgers: live prospective evidence must never share a file with
# synthetic demo cycles or caller-injected snapshots (v2.2 isolation fix).
REAL_LEDGER_PATH = ROOT / "data" / "piia" / "piia_v2_2_live_prospective.sqlite"
DEMO_LEDGER_PATH = ROOT / "data" / "piia" / "piia_v2_2_demo_synthetic.sqlite"
DEFAULT_LEDGER_PATH = REAL_LEDGER_PATH
# Earlier ledgers are preserved untouched and never written or counted by v2.2.
LEGACY_LEDGER_PATHS = (
    ROOT / "data" / "piia" / "piia_v2_continuous.sqlite",
    ROOT / "data" / "piia" / "piia_v2_1_continuous.sqlite",
)
APP_ID = 0x50495632  # "PIV2"
LEDGER_SCHEMA_VERSION = "piia_v2_2_ledger_1"

MODE_LIVE = "live_prospective_public_6_pillars"
MODE_DEMO = "SYNTHETIC_DEMO_NOT_LIVE"
MODE_INJECTED = "caller_injected_snapshot"
KIND_LIVE = "live_prospective"
KIND_DEMO = "demo_synthetic"
KIND_INJECTED = "test_or_injected"
MODE_TO_KIND = {MODE_LIVE: KIND_LIVE, MODE_DEMO: KIND_DEMO, MODE_INJECTED: KIND_INJECTED}


class LedgerKindError(ValueError):
    """Raised when a write would mix live, demo, or injected evidence in one ledger."""


def _utc_now() -> datetime:
    """Single clock source (tests monkeypatch this; production uses the system UTC clock)."""
    return datetime.now(timezone.utc)


def default_ledger_path(demo: bool = False) -> Path:
    """Resolve the default ledger for a mode (read at call time so tests can redirect)."""
    return Path(DEMO_LEDGER_PATH if demo else REAL_LEDGER_PATH)


def _same_file(a: Path, b: Path) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return False
SHADOW_BASE_ARMS = (
    "B0_cash",
    "B1_simple_ema_1h",
    "B2_aligned_ema_4h",
    "B3_donchian_breakout_4h",
    "M4_full_piia_v2",
)
SHADOW_SPECS: tuple[tuple[str, str, float], ...] = tuple(
    [(arm, arm, 1.0) for arm in SHADOW_BASE_ARMS]
    + [(f"{arm}__2x_slip", arm, 2.0) for arm in SHADOW_BASE_ARMS if arm != "B0_cash"]
)
SHADOW_ARMS: tuple[str, ...] = tuple(name for name, _, _ in SHADOW_SPECS)


def _display_path(path: Path) -> str:
    """Avoid leaking local OS home paths when reporting paths inside repo."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _ts_utc(value: str | pd.Timestamp | datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _init_db(db_path: Path, *, ledger_kind: str | None = None) -> sqlite3.Connection:
    """Open (or create) a ledger. Unlabelled legacy ledgers are refused for any write path."""
    db_path = Path(db_path)
    if ledger_kind is not None and ledger_kind not in MODE_TO_KIND.values():
        raise LedgerKindError(f"Unknown ledger kind {ledger_kind!r}")
    if ledger_kind is not None and ledger_kind != KIND_LIVE and _same_file(db_path, REAL_LEDGER_PATH):
        raise LedgerKindError("Demo or injected records can never be written to the real prospective ledger path.")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    existed = db_path.exists() and db_path.stat().st_size > 0
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    app = conn.execute("PRAGMA application_id").fetchone()[0]
    if existed and app not in (0, APP_ID):
        conn.close()
        raise ValueError(f"Unrelated SQLite database at {_display_path(db_path)}")
    had_tables = bool(_sqlite_tables(conn))
    if had_tables and "ledger_meta" not in _sqlite_tables(conn):
        conn.close()
        raise LedgerKindError(
            f"Legacy unlabelled ledger at {_display_path(db_path)} (pre-v2.2, may mix demo and live cycles); "
            "it is preserved read-only and cannot be written. Use `python -m piia agent-inspect-ledger`."
        )
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
            cycle_hash TEXT NOT NULL,
            capture_started_utc TEXT,
            policy_version TEXT
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

        CREATE TABLE IF NOT EXISTS position_settlements (
            book_name TEXT NOT NULL,
            position_id TEXT NOT NULL,
            settlement_utc TEXT NOT NULL,
            status TEXT NOT NULL,
            event_id TEXT NOT NULL,
            PRIMARY KEY (book_name, position_id, settlement_utc)
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

        CREATE TRIGGER IF NOT EXISTS immutable_pos_settle_upd BEFORE UPDATE ON position_settlements
        BEGIN SELECT RAISE(ABORT, 'append-only position_settlements'); END;
        CREATE TRIGGER IF NOT EXISTS immutable_pos_settle_del BEFORE DELETE ON position_settlements
        BEGIN SELECT RAISE(ABORT, 'append-only position_settlements'); END;

        CREATE TRIGGER IF NOT EXISTS immutable_proposals_upd BEFORE UPDATE ON human_proposals
        BEGIN SELECT RAISE(ABORT, 'append-only human_proposals'); END;
        CREATE TRIGGER IF NOT EXISTS immutable_proposals_del BEFORE DELETE ON human_proposals
        BEGIN SELECT RAISE(ABORT, 'append-only human_proposals'); END;

        CREATE TRIGGER IF NOT EXISTS immutable_decisions_upd BEFORE UPDATE ON human_decisions
        BEGIN SELECT RAISE(ABORT, 'append-only human_decisions'); END;
        CREATE TRIGGER IF NOT EXISTS immutable_decisions_del BEFORE DELETE ON human_decisions
        BEGIN SELECT RAISE(ABORT, 'append-only human_decisions'); END;

        CREATE TABLE IF NOT EXISTS ledger_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TRIGGER IF NOT EXISTS immutable_meta_upd BEFORE UPDATE ON ledger_meta
        BEGIN SELECT RAISE(ABORT, 'append-only ledger_meta'); END;
        CREATE TRIGGER IF NOT EXISTS immutable_meta_del BEFORE DELETE ON ledger_meta
        BEGIN SELECT RAISE(ABORT, 'append-only ledger_meta'); END;
        """
    )
    if not had_tables:
        kind = ledger_kind or KIND_INJECTED
        conn.executemany(
            "INSERT INTO ledger_meta(key, value) VALUES (?, ?)",
            [
                ("ledger_kind", kind),
                ("schema_version", LEDGER_SCHEMA_VERSION),
                ("created_at_utc", _utc_now().isoformat()),
            ],
        )
        conn.commit()
    return conn


def _sqlite_tables(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}


def _read_ledger_kind(conn: sqlite3.Connection) -> str | None:
    if "ledger_meta" not in _sqlite_tables(conn):
        return None
    row = conn.execute("SELECT value FROM ledger_meta WHERE key = 'ledger_kind'").fetchone()
    return str(row[0]) if row else None


def _assert_ledger_accepts(db_path: Path, conn: sqlite3.Connection, required_kind: str) -> None:
    """Fail closed unless the opened ledger is labelled with exactly the required evidence kind."""
    actual = _read_ledger_kind(conn)
    if actual != required_kind:
        raise LedgerKindError(
            f"Ledger {_display_path(db_path)} is labelled {actual!r}; refusing to write {required_kind!r} records "
            "(live, demo, and injected evidence must use separate ledgers)."
        )
    if required_kind != KIND_LIVE and _same_file(db_path, REAL_LEDGER_PATH):
        raise LedgerKindError("Demo or injected records can never be written to the real prospective ledger path.")


def _candles_to_df(candles: list[dict[str, Any]]) -> pd.DataFrame:
    rows, idx = [], []
    for c in candles:
        idx.append(_ts_utc(c["open_at_utc"]))
        rows.append(
            {
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "close": float(c["close"]),
                "volume": float(c.get("volume") or 0.0),
            }
        )
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="date")).sort_index()


def _reconstruct_book_state(
    conn: sqlite3.Connection,
    book_name: str,
    default_cash: float,
) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT event_kind, symbol, payload_json, cash_after_usdt FROM ledger_events WHERE book_name = ? ORDER BY rowid ASC",
        (book_name,),
    ).fetchall()
    if not rows:
        return {
            "starting_cash": float(default_cash),
            "cash": float(default_cash),
            "open_pos": None,
            "closed_trades": [],
            "halted": False,
            "halt_reason": None,
            "halted_at_utc": None,
            "day_anchors": {},
            "fees_paid_usdt": 0.0,
            "known_funding_paid_usdt": 0.0,
            "funding_complete": True,
            "unknown_funding_events": 0,
            "realized_pnl_ex_unknown_funding_usdt": 0.0,
            "realized_net_pnl_usdt": 0.0,
            "net_return_claim_allowed": True,
            "reconciliation_error_usdt": 0.0,
            "reconciled_bar_close_utc": None,
        }

    cash = float(rows[-1]["cash_after_usdt"])
    first_payload = json.loads(rows[0]["payload_json"])
    starting_cash = float(first_payload.get("starting_cash", default_cash))
    open_pos: dict[str, Any] | None = None
    closed_trades: list[dict[str, Any]] = []
    halted = False
    halt_reason: str | None = None
    halted_at_utc: str | None = None
    day_anchors: dict[str, float] = {}
    fees_paid = 0.0
    known_funding_paid = 0.0
    unknown_funding_events = 0
    reconciled_bar_close_utc: str | None = None

    for r in rows:
        kind = r["event_kind"]
        payload = json.loads(r["payload_json"])
        if kind == "day_anchor":
            d_key = payload.get("utc_date")
            if d_key and d_key not in day_anchors:
                day_anchors[d_key] = float(payload["day_start_equity_usdt"])
        elif kind == "open_position":
            open_pos = dict(payload)
            open_pos.setdefault("known_funding_paid_usdt", float(open_pos.get("funding", 0.0)))
            open_pos.setdefault("funding_complete", True)
            open_pos.setdefault("unknown_funding_count", 0)
            open_pos.setdefault("settled_timestamps", [])
            open_pos.setdefault("funding_records", [])
            open_pos.setdefault("stop_effective_from_utc", open_pos.get("opened_at_utc") or open_pos.get("entry_bar_open_utc"))
            open_pos.setdefault("last_processed_bar_close_utc", open_pos.get("last_reconciled_bar_open_utc") or open_pos.get("entry_bar_open_utc"))
            fees_paid += float(open_pos["entry_fee"])
            reconciled_bar_close_utc = open_pos["last_processed_bar_close_utc"]
        elif kind == "update_stop" and open_pos is not None:
            open_pos["stop"] = float(payload["new_stop"])
            open_pos["be_locked"] = bool(payload.get("be_locked", open_pos.get("be_locked", False)))
            if payload.get("effective_from_utc"):
                open_pos["stop_effective_from_utc"] = payload["effective_from_utc"]
            if payload.get("last_processed_bar_close_utc"):
                open_pos["last_processed_bar_close_utc"] = payload["last_processed_bar_close_utc"]
                reconciled_bar_close_utc = payload["last_processed_bar_close_utc"]
        elif kind == "funding" and open_pos is not None:
            amt = float(payload["amount_paid_usdt"])
            open_pos["known_funding_paid_usdt"] = float(open_pos.get("known_funding_paid_usdt", 0.0)) + amt
            known_funding_paid += amt
            settle_iso = payload.get("settlement_utc")
            if settle_iso and settle_iso not in open_pos["settled_timestamps"]:
                open_pos["settled_timestamps"].append(settle_iso)
            open_pos["funding_records"].append(payload)
            if payload.get("last_processed_bar_close_utc"):
                open_pos["last_processed_bar_close_utc"] = payload["last_processed_bar_close_utc"]
                reconciled_bar_close_utc = payload["last_processed_bar_close_utc"]
        elif kind == "funding_unknown" and open_pos is not None:
            open_pos["funding_complete"] = False
            open_pos["unknown_funding_count"] = int(open_pos.get("unknown_funding_count", 0)) + 1
            unknown_funding_events += 1
            settle_iso = payload.get("settlement_utc")
            if settle_iso and settle_iso not in open_pos["settled_timestamps"]:
                open_pos["settled_timestamps"].append(settle_iso)
            open_pos["funding_records"].append(payload)
            if payload.get("last_processed_bar_close_utc"):
                open_pos["last_processed_bar_close_utc"] = payload["last_processed_bar_close_utc"]
                reconciled_bar_close_utc = payload["last_processed_bar_close_utc"]
        elif kind == "bar_cursor":
            if payload.get("last_processed_bar_close_utc"):
                reconciled_bar_close_utc = payload["last_processed_bar_close_utc"]
                if open_pos is not None:
                    open_pos["last_processed_bar_close_utc"] = reconciled_bar_close_utc
        elif kind == "close_position":
            fees_paid += float(payload.get("exit_fee", 0.0))
            if payload.get("exit_time_utc"):
                reconciled_bar_close_utc = payload["exit_time_utc"]
            closed_trades.append(payload)
            open_pos = None
        elif kind == "loss_halt":
            halted = True
            halt_reason = payload.get("halt_reason") or "loss_halt"
            halted_at_utc = payload.get("halted_at_utc")

    any_unknown = (
        unknown_funding_events > 0
        or any(not t.get("funding_complete", True) for t in closed_trades)
        or (open_pos is not None and not open_pos.get("funding_complete", True))
    )
    realized_ex_unknown = sum(
        float(t.get("pnl_ex_unknown_funding_usdt", t.get("net_pnl") or 0.0))
        for t in closed_trades
    )
    open_cost_drag = (
        float(open_pos["entry_fee"]) + float(open_pos.get("known_funding_paid_usdt", 0.0))
        if open_pos is not None
        else 0.0
    )
    recon_err = abs((cash - starting_cash) - (realized_ex_unknown - open_cost_drag))

    return {
        "starting_cash": starting_cash,
        "cash": cash,
        "open_pos": open_pos,
        "closed_trades": closed_trades,
        "halted": halted,
        "halt_reason": halt_reason,
        "halted_at_utc": halted_at_utc,
        "day_anchors": day_anchors,
        "fees_paid_usdt": round(fees_paid, 6),
        "known_funding_paid_usdt": round(known_funding_paid, 6),
        "funding_complete": not any_unknown,
        "unknown_funding_events": unknown_funding_events,
        "realized_pnl_ex_unknown_funding_usdt": round(realized_ex_unknown, 6),
        "realized_net_pnl_usdt": None if any_unknown else round(realized_ex_unknown, 4),
        "net_return_claim_allowed": not any_unknown,
        "reconciliation_error_usdt": recon_err,
        "reconciled_bar_close_utc": reconciled_bar_close_utc,
    }


def _append_event(
    conn: sqlite3.Connection,
    book_name: str,
    event_kind: str,
    symbol: str | None,
    payload: dict[str, Any],
    cash_after: float,
    now_iso: str,
) -> str:
    ev_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO ledger_events(event_id, recorded_at_utc, book_name, event_kind, symbol, payload_json, cash_after_usdt) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            ev_id,
            now_iso,
            book_name,
            event_kind,
            symbol,
            json.dumps(payload, sort_keys=True, default=str),
            float(cash_after),
        ),
    )
    return ev_id


def _record_halt_if_breached(
    conn: sqlite3.Connection,
    book_name: str,
    state: dict[str, Any],
    *,
    eq_usdt: float,
    day_key: str,
    day_start_eq: float,
    cfg: dict[str, Any],
    at_utc_iso: str,
    now_iso: str,
) -> str | None:
    if state["halted"]:
        return state["halt_reason"]
    rc = cfg["risk_and_costs"]
    daily_halt = float(rc["daily_loss_halt_fraction"])
    total_halt = float(rc["total_loss_halt_fraction"])
    starting_cash = float(state["starting_cash"])

    reason = None
    if eq_usdt <= starting_cash * (1.0 - total_halt):
        reason = "total_loss_halt_10pct"
    elif eq_usdt <= day_start_eq * (1.0 - daily_halt):
        reason = "daily_loss_halt_3pct"

    if reason is None:
        return None

    _append_event(
        conn,
        book_name,
        "loss_halt",
        None,
        {
            "halt_reason": reason,
            "utc_date": day_key,
            "equity_usdt": round(eq_usdt, 6),
            "day_start_equity_usdt": round(day_start_eq, 6),
            "starting_cash_usdt": starting_cash,
            "daily_threshold_usdt": round(day_start_eq * (1.0 - daily_halt), 6),
            "total_threshold_usdt": round(starting_cash * (1.0 - total_halt), 6),
            "halted_at_utc": at_utc_iso,
        },
        float(state["cash"]),
        now_iso,
    )
    state["halted"] = True
    state["halt_reason"] = reason
    state["halted_at_utc"] = at_utc_iso
    return reason


def _build_settled_funding_lookup(
    settled_funding_ctx: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    insts = (settled_funding_ctx or {}).get("instruments") or {}
    for sym, obj in insts.items():
        if not isinstance(obj, dict):
            continue
        by_ts: dict[int, dict[str, Any]] = {}
        for r in obj.get("records") or []:
            try:
                ms = int(r["timestamp"])
                rate = float(r["rate"])
                if math.isfinite(rate):
                    by_ts[ms] = {"timestamp": ms, "settlement_utc": r.get("settlement_utc"), "rate": rate}
            except (KeyError, TypeError, ValueError):
                continue
        lookup[sym] = {
            "ok": bool(obj.get("ok")),
            "interval_hours": int(obj.get("interval_hours") or DEFAULT_FUNDING_INTERVAL_HOURS.get(sym, 8)),
            "schedule_verified": bool(obj.get("schedule_verified", False)),
            "source_url": obj.get("source_url"),
            "raw_sha256": obj.get("raw_sha256"),
            "retrieved_at_utc": obj.get("retrieved_at_utc"),
            "error": obj.get("error"),
            "by_ts": by_ts,
        }
    return lookup


def _reconcile_open_position_with_bars(
    conn: sqlite3.Connection,
    book_name: str,
    state: dict[str, Any],
    prepared_tables: dict[str, pd.DataFrame],
    settled_funding_ctx: dict[str, Any] | None,
    cfg: dict[str, Any],
    now_iso: str,
    *,
    slippage_each_side: float | None = None,
    current_cutoff_utc: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Advance an open paper position across newly closed 1h bars since its persisted cursor."""
    rg = cfg["roles_and_gates"]
    slip = float(cfg["risk_and_costs"]["slippage_each_side_base"] if slippage_each_side is None else slippage_each_side)
    f_lookup = _build_settled_funding_lookup(settled_funding_ctx)

    pos = state["open_pos"]
    if pos is None:
        if current_cutoff_utc is not None and not state["halted"]:
            d_key = current_cutoff_utc.date().isoformat()
            cash_now = float(state["cash"])
            if d_key not in state["day_anchors"]:
                state["day_anchors"][d_key] = cash_now
                _append_event(
                    conn,
                    book_name,
                    "day_anchor",
                    None,
                    {"utc_date": d_key, "day_start_equity_usdt": round(cash_now, 6), "basis": "flat_cash_at_cycle"},
                    cash_now,
                    now_iso,
                )
            _record_halt_if_breached(
                conn,
                book_name,
                state,
                eq_usdt=cash_now,
                day_key=d_key,
                day_start_eq=state["day_anchors"][d_key],
                cfg=cfg,
                at_utc_iso=current_cutoff_utc.isoformat(),
                now_iso=now_iso,
            )
        return state

    sym = pos["symbol"]
    df = prepared_tables.get(sym)
    if df is None or df.empty:
        return state

    sp = resolve_spec(sym).to_v1_spec()
    side_i = 1 if pos["side"] == "long" else -1
    opened_at = _ts_utc(pos.get("opened_at_utc") or pos["entry_bar_open_utc"])
    cursor_close = _ts_utc(pos["last_processed_bar_close_utc"])
    stop_eff_from = _ts_utc(pos.get("stop_effective_from_utc") or pos.get("opened_at_utc") or pos["entry_bar_open_utc"])

    new_bars = [ts for ts in df.index if (ts + pd.Timedelta(hours=1)) > cursor_close]
    if not new_bars:
        return state

    def _finalize_close(raw_px: float, exit_time_utc: str, reason: str) -> None:
        nonlocal pos
        assert pos is not None
        fill_px = raw_px * (1.0 - side_i * slip)
        exit_fee = float(pos["qty"]) * sp.contract_size * fill_px * sp.taker
        gross = exit_pnl(side_i, float(pos["qty"]), float(pos["entry"]), fill_px, sp)
        new_cash = float(state["cash"]) + gross - exit_fee
        known_f = float(pos.get("known_funding_paid_usdt", 0.0))
        entry_fee = float(pos["entry_fee"])
        pnl_ex_unknown = gross - entry_fee - exit_fee - known_f
        fund_ok = bool(pos.get("funding_complete", True))
        closed_rec = {
            "position_id": pos.get("position_id"),
            "symbol": sym,
            "side": pos["side"],
            "setup_type": pos.get("setup_type"),
            "qty": float(pos["qty"]),
            "contract_size": sp.contract_size,
            "entry": round(float(pos["entry"]), 6),
            "exit": round(fill_px, 6),
            "slippage_each_side": slip,
            "entry_fee": round(entry_fee, 6),
            "exit_fee": round(exit_fee, 6),
            "fee_provenance": {
                "taker_rate": sp.taker,
                "basis": "contract_spec_taker_fee",
            },
            "known_funding_paid_usdt": round(known_f, 6),
            "funding_complete": fund_ok,
            "unknown_funding_count": int(pos.get("unknown_funding_count", 0)),
            "pnl_ex_unknown_funding_usdt": pnl_ex_unknown,
            "net_pnl": pnl_ex_unknown if fund_ok else None,
            "net_pnl_status": "estimated_with_public_funding" if fund_ok else "unknown_missing_funding",
            "exit_time_utc": exit_time_utc,
            "exit_reason": reason,
        }
        _append_event(conn, book_name, "close_position", sym, closed_rec, new_cash, now_iso)
        state["cash"] = new_cash
        state["open_pos"] = None
        state["closed_trades"].append(closed_rec)
        state["fees_paid_usdt"] = round(float(state.get("fees_paid_usdt", 0.0)) + exit_fee, 6)
        state["realized_pnl_ex_unknown_funding_usdt"] = round(
            float(state.get("realized_pnl_ex_unknown_funding_usdt", 0.0)) + pnl_ex_unknown, 6
        )
        if not fund_ok:
            state["funding_complete"] = False
            state["realized_net_pnl_usdt"] = None
            state["net_return_claim_allowed"] = False
        elif state["funding_complete"]:
            state["realized_net_pnl_usdt"] = round(state["realized_pnl_ex_unknown_funding_usdt"], 4)
        pos = None

    last_bar_wrote_stop = False
    for ts in new_bars:
        row = df.loc[ts]
        bar_open = ts
        bar_close = ts + pd.Timedelta(hours=1)
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        last_bar_wrote_stop = False

        # 1. Daily loss anchor on first bar seen for each UTC date
        d_key = bar_open.date().isoformat()
        mark_at_open = o if bar_open >= opened_at else float(pos["entry"])
        if d_key not in state["day_anchors"]:
            eq_anchor = float(state["cash"]) + exit_pnl(side_i, float(pos["qty"]), float(pos["entry"]), mark_at_open, sp)
            state["day_anchors"][d_key] = eq_anchor
            _append_event(
                conn,
                book_name,
                "day_anchor",
                sym,
                {"utc_date": d_key, "day_start_equity_usdt": round(eq_anchor, 6), "basis": "bar_open_mtm"},
                float(state["cash"]),
                now_iso,
            )
        day_start_eq = state["day_anchors"][d_key]

        # 2. Scheduled funding settlement at `bar_open` (strictly if position held from BEFORE bar_open)
        if bar_open > opened_at:
            settle_iso = bar_open.isoformat()
            sym_f = f_lookup.get(sym)
            iv_h = int((sym_f or {}).get("interval_hours") or DEFAULT_FUNDING_INTERVAL_HOURS.get(sym, 8))
            is_boundary = ((int(bar_open.timestamp()) // 3600) % iv_h) == 0
            pos_id = str(pos.get("position_id") or "")
            already_settled = settle_iso in (pos.get("settled_timestamps") or [])
            if not already_settled and pos_id:
                row_db = conn.execute(
                    "SELECT 1 FROM position_settlements WHERE book_name = ? AND position_id = ? AND settlement_utc = ?",
                    (book_name, pos_id, settle_iso),
                ).fetchone()
                already_settled = row_db is not None

            if is_boundary and not already_settled:
                ts_ms = int(bar_open.timestamp() * 1000)
                rec = ((sym_f or {}).get("by_ts") or {}).get(ts_ms)
                verified_ok = bool(sym_f and sym_f.get("ok") and sym_f.get("schedule_verified") and rec is not None)
                notional = float(pos["qty"]) * sp.contract_size * o
                if verified_ok:
                    rate = float(rec["rate"])
                    amt_paid = side_i * notional * rate  # + means long pays when rate>0; short receives
                    new_cash = float(state["cash"]) - amt_paid
                    state["cash"] = new_cash
                    pos["known_funding_paid_usdt"] = float(pos.get("known_funding_paid_usdt", 0.0)) + amt_paid
                    state["known_funding_paid_usdt"] = round(float(state.get("known_funding_paid_usdt", 0.0)) + amt_paid, 6)
                    pos.setdefault("settled_timestamps", []).append(settle_iso)
                    ev_id = _append_event(
                        conn,
                        book_name,
                        "funding",
                        sym,
                        {
                            "position_id": pos_id,
                            "symbol": sym,
                            "side": pos["side"],
                            "qty": float(pos["qty"]),
                            "contract_size": sp.contract_size,
                            "settlement_utc": settle_iso,
                            "interval_hours": iv_h,
                            "rate": rate,
                            "mark_price_proxy": o,
                            "notional_usdt": round(notional, 6),
                            "amount_paid_usdt": amt_paid,
                            "cash_delta_usdt": -amt_paid,
                            "status": "settled_public_estimate",
                            "source_url": sym_f.get("source_url"),
                            "raw_sha256": sym_f.get("raw_sha256"),
                            "retrieved_at_utc": sym_f.get("retrieved_at_utc"),
                            "estimation_basis": "public_settled_rate_times_1h_open_mark_proxy_not_venue_statement",
                            "last_processed_bar_close_utc": pos["last_processed_bar_close_utc"],
                        },
                        new_cash,
                        now_iso,
                    )
                    if pos_id:
                        conn.execute(
                            "INSERT INTO position_settlements(book_name, position_id, settlement_utc, status, event_id) VALUES (?, ?, ?, ?, ?)",
                            (book_name, pos_id, settle_iso, "settled_public_estimate", ev_id),
                        )
                else:
                    why = (
                        sym_f.get("error") if (sym_f and sym_f.get("error"))
                        else ("schedule_unverified" if (sym_f and sym_f.get("ok") and not sym_f.get("schedule_verified")) else "missing_or_unavailable_rate")
                    )
                    pos["funding_complete"] = False
                    pos["unknown_funding_count"] = int(pos.get("unknown_funding_count", 0)) + 1
                    state["funding_complete"] = False
                    state["unknown_funding_events"] = int(state.get("unknown_funding_events", 0)) + 1
                    state["realized_net_pnl_usdt"] = None
                    state["net_return_claim_allowed"] = False
                    pos.setdefault("settled_timestamps", []).append(settle_iso)
                    ev_id = _append_event(
                        conn,
                        book_name,
                        "funding_unknown",
                        sym,
                        {
                            "position_id": pos_id,
                            "symbol": sym,
                            "side": pos["side"],
                            "qty": float(pos["qty"]),
                            "contract_size": sp.contract_size,
                            "settlement_utc": settle_iso,
                            "interval_hours": iv_h,
                            "rate": None,
                            "mark_price_proxy": o,
                            "notional_usdt": round(notional, 6),
                            "amount_paid_usdt": None,
                            "cash_delta_usdt": 0.0,
                            "status": "unknown",
                            "reason": why,
                            "source_url": sym_f.get("source_url") if sym_f else None,
                            "estimation_basis": "missing_or_unverified_funding_not_assumed_zero",
                            "last_processed_bar_close_utc": pos["last_processed_bar_close_utc"],
                        },
                        float(state["cash"]),
                        now_iso,
                    )
                    if pos_id:
                        conn.execute(
                            "INSERT INTO position_settlements(book_name, position_id, settlement_utc, status, event_id) VALUES (?, ?, ?, ?, ?)",
                            (book_name, pos_id, settle_iso, "unknown", ev_id),
                        )

        # 3. Chandelier trailing update at `bar_open` (using `shift_bars=1` column known BEFORE `bar_open`)
        if bar_open > opened_at and pos.get("setup_type") in {"trend_breakout", "trend_pullback", "donchian_breakout"}:
            ch_col = "chandelier_long" if side_i == 1 else "chandelier_short"
            ch_val = row.get(ch_col)
            if ch_val is not None and math.isfinite(float(ch_val)):
                cand_stop = float(ch_val)
                if (side_i == 1 and cand_stop > float(pos["stop"])) or (side_i == -1 and cand_stop < float(pos["stop"])):
                    pos["stop"] = cand_stop
                    stop_eff_from = bar_open
                    pos["stop_effective_from_utc"] = bar_open.isoformat()
                    _append_event(
                        conn,
                        book_name,
                        "update_stop",
                        sym,
                        {
                            "new_stop": cand_stop,
                            "be_locked": bool(pos.get("be_locked", False)),
                            "effective_from_utc": bar_open.isoformat(),
                            "reason": "chandelier_trail",
                            "last_processed_bar_close_utc": pos["last_processed_bar_close_utc"],
                        },
                        float(state["cash"]),
                        now_iso,
                    )

        # Active stop check: stop becomes active only from `stop_effective_from_utc` onward
        stop_active_open = stop_eff_from <= bar_open
        if bar_open >= opened_at and stop_active_open:
            if (side_i == 1 and o <= float(pos["stop"])) or (side_i == -1 and o >= float(pos["stop"])):
                _finalize_close(o, bar_open.isoformat(), "gap_stop")
                _record_halt_if_breached(
                    conn, book_name, state, eq_usdt=float(state["cash"]),
                    day_key=d_key, day_start_eq=day_start_eq, cfg=cfg,
                    at_utc_iso=bar_open.isoformat(), now_iso=now_iso,
                )
                return state

        # 4. Lagged signal / time-stop exits at `bar_open`
        if bar_open > opened_at:
            def _fin(s: Any, col: str) -> float | None:
                if s is None or col not in s.index:
                    return None
                val = s[col]
                if val is None or isinstance(val, bool):
                    return None
                try:
                    fv = float(val)
                    return fv if math.isfinite(fv) else None
                except (TypeError, ValueError):
                    return None

            prev_ts = bar_open - pd.Timedelta(hours=1)
            sig_row = df.loc[prev_ts] if prev_ts in df.index else None
            st_type = pos.get("setup_type") or ""
            exit_reason = None
            if st_type in {"ema_1h", "ema_4h_aligned"}:
                e20, e50 = _fin(sig_row, "ema20"), _fin(sig_row, "ema50")
                if e20 is not None and e50 is not None:
                    rev = (side_i == 1 and e20 < e50) or (side_i == -1 and e20 > e50)
                    if rev:
                        exit_reason = "reverse_signal"
            elif st_type == "range_reversion":
                bb_m, s_close = _fin(sig_row, "bb_mid"), _fin(sig_row, "close")
                mid_hit = bb_m is not None and s_close is not None and (
                    (side_i == 1 and s_close >= bb_m) or (side_i == -1 and s_close <= bb_m)
                )
                if mid_hit:
                    exit_reason = "range_mid_target"
                elif bar_open - opened_at >= pd.Timedelta(hours=int(rg["range_max_hold_hours"])):
                    exit_reason = "range_time_stop"
            else:
                e20, e50 = _fin(sig_row, "ema20"), _fin(sig_row, "ema50")
                h20, h50 = _fin(sig_row, "htf_ema20"), _fin(sig_row, "htf_ema50")
                rev = (
                    e20 is not None and e50 is not None and h20 is not None and h50 is not None
                    and ((side_i == 1 and e20 < e50 and h20 < h50) or (side_i == -1 and e20 > e50 and h20 > h50))
                )
                if rev:
                    exit_reason = "htf_trend_reversal"
                elif st_type != "donchian_breakout":
                    max_h = int(pos.get("max_hold_hours") or rg["trend_max_hold_hours"])
                    if bar_open - opened_at >= pd.Timedelta(hours=max_h):
                        exit_reason = "trend_time_stop"

            if exit_reason is not None:
                _finalize_close(o, bar_open.isoformat(), exit_reason)
                _record_halt_if_breached(
                    conn, book_name, state, eq_usdt=float(state["cash"]),
                    day_key=d_key, day_start_eq=day_start_eq, cfg=cfg,
                    at_utc_iso=bar_open.isoformat(), now_iso=now_iso,
                )
                return state

        # 5. Open-mark loss halt check (chronological: close_position BEFORE loss_halt!)
        rc_cfg = cfg["risk_and_costs"]
        eq_open = float(state["cash"]) + exit_pnl(side_i, float(pos["qty"]), float(pos["entry"]), mark_at_open, sp)
        if (
            eq_open <= float(state["starting_cash"]) * (1.0 - float(rc_cfg["total_loss_halt_fraction"]))
            or eq_open <= day_start_eq * (1.0 - float(rc_cfg["daily_loss_halt_fraction"]))
        ):
            _finalize_close(mark_at_open, bar_open.isoformat(), "loss_halt")
            _record_halt_if_breached(
                conn, book_name, state, eq_usdt=eq_open,
                day_key=d_key, day_start_eq=day_start_eq, cfg=cfg,
                at_utc_iso=bar_open.isoformat(), now_iso=now_iso,
            )
            return state

        # 6. Intrabar stop check (only if stop was effective BEFORE end of bar; strictly not ratcheted at end of THIS bar)
        eff_open = o if bar_open >= opened_at else float(pos["entry"])
        if stop_eff_from < bar_close:
            hit = stop_fill(side_i, float(pos["stop"]), eff_open, l, h)
            if hit is not None:
                _finalize_close(hit, bar_close.isoformat(), "stop_loss")
                _record_halt_if_breached(
                    conn, book_name, state, eq_usdt=float(state["cash"]),
                    day_key=d_key, day_start_eq=day_start_eq, cfg=cfg,
                    at_utc_iso=bar_close.isoformat(), now_iso=now_iso,
                )
                return state

        # 7. Breakeven ratchet check (effective from NEXT bar `bar_close` onward; never retroactive!)
        if pos.get("be_trigger") and not pos.get("be_locked"):
            reached = (side_i == 1 and h >= float(pos["be_trigger"])) or (
                side_i == -1 and l <= float(pos["be_trigger"])
            )
            if reached:
                be_px = float(pos["be_stop"])
                if (side_i == 1 and be_px > float(pos["stop"])) or (
                    side_i == -1 and be_px < float(pos["stop"])
                ):
                    pos["stop"] = be_px
                    pos["be_locked"] = True
                    stop_eff_from = bar_close
                    pos["stop_effective_from_utc"] = bar_close.isoformat()
                    pos["last_processed_bar_close_utc"] = bar_close.isoformat()
                    last_bar_wrote_stop = True
                    _append_event(
                        conn,
                        book_name,
                        "update_stop",
                        sym,
                        {
                            "new_stop": be_px,
                            "be_locked": True,
                            "effective_from_utc": bar_close.isoformat(),
                            "reason": "breakeven_ratchet",
                            "last_processed_bar_close_utc": bar_close.isoformat(),
                        },
                        float(state["cash"]),
                        now_iso,
                    )

        # 8. Close-mark loss halt check (chronological: close_position BEFORE loss_halt!)
        eq_close = float(state["cash"]) + exit_pnl(side_i, float(pos["qty"]), float(pos["entry"]), c, sp)
        if (
            eq_close <= float(state["starting_cash"]) * (1.0 - float(rc_cfg["total_loss_halt_fraction"]))
            or eq_close <= day_start_eq * (1.0 - float(rc_cfg["daily_loss_halt_fraction"]))
        ):
            _finalize_close(c, bar_close.isoformat(), "loss_halt")
            _record_halt_if_breached(
                conn, book_name, state, eq_usdt=eq_close,
                day_key=d_key, day_start_eq=day_start_eq, cfg=cfg,
                at_utc_iso=bar_close.isoformat(), now_iso=now_iso,
            )
            return state

        pos["last_processed_bar_close_utc"] = bar_close.isoformat()
        state["reconciled_bar_close_utc"] = bar_close.isoformat()

    if pos is not None and not last_bar_wrote_stop:
        _append_event(
            conn,
            book_name,
            "bar_cursor",
            sym,
            {
                "position_id": pos.get("position_id"),
                "last_processed_bar_close_utc": pos["last_processed_bar_close_utc"],
                "stop": pos["stop"],
                "stop_effective_from_utc": pos.get("stop_effective_from_utc"),
            },
            float(state["cash"]),
            now_iso,
        )
    return state


def check_three_contract_alignment(
    mkt_snap: dict[str, Any],
    *,
    expected_cutoff: pd.Timestamp,
) -> dict[str, Any]:
    """Require BTCUSDT / XAUUSDT / NDX100USDT to share the same latest closed 1h bar == expected cutoff.

    `expected_cutoff` is the close time of the most recent fully closed hour at capture time.
    Any missing, gapped, lagging (stale) or future bar rejects the whole cycle.
    """
    reasons: list[str] = []
    frames: dict[str, pd.DataFrame] = {}
    last_close_by_symbol: dict[str, str | None] = {}
    one_h = pd.Timedelta(hours=1)
    for sym in SYMBOLS:
        inst = (mkt_snap.get("instruments") or {}).get(sym) or {}
        candles = inst.get("candles_closed_1h") or []
        if not candles:
            reasons.append(f"missing_candles:{sym}")
            last_close_by_symbol[sym] = None
            continue
        try:
            df = _candles_to_df(candles)
        except (KeyError, TypeError, ValueError):
            reasons.append(f"invalid_candles:{sym}")
            last_close_by_symbol[sym] = None
            continue
        if df.index.has_duplicates:
            reasons.append(f"duplicate_candles:{sym}")
        diffs = df.index.to_series().diff().dropna()
        if not diffs.empty and bool((diffs != one_h).any()):
            reasons.append(f"gapped_candles:{sym}")
        last_close = df.index[-1] + one_h
        last_close_by_symbol[sym] = last_close.isoformat()
        if last_close > expected_cutoff:
            reasons.append(f"future_candle:{sym}:last_close={last_close.isoformat()}>expected={expected_cutoff.isoformat()}")
        elif last_close < expected_cutoff:
            reasons.append(f"stale_or_lagging_candle:{sym}:last_close={last_close.isoformat()}<expected={expected_cutoff.isoformat()}")
        frames[sym] = df
    if len({v for v in last_close_by_symbol.values() if v is not None}) > 1:
        reasons.append("misaligned_latest_close_across_contracts")
    ok = not reasons and len(frames) == len(SYMBOLS)
    return {
        "ok": ok,
        "reasons": reasons,
        "expected_cutoff_utc": expected_cutoff.isoformat(),
        "last_close_by_symbol": last_close_by_symbol,
        "frames": frames if ok else {},
    }


def run_prospective_cycle(
    *,
    db_path: Path | None = None,
    starting_cash: float = 500.0,
    demo_mode: bool = False,
    snapshot_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one prospective PIIA cycle: verify freeze -> collect -> align -> reconcile -> trace -> append.

    Evidence kinds are isolated: live public collection may only write to a `live_prospective`
    ledger; demo cycles only to a `demo_synthetic` ledger; caller-injected snapshots only to an
    explicitly supplied `test_or_injected` ledger (never the real ledger path).
    """
    from piia.intelligence.agent_v2.manifest import verify_piia_v2_frozen

    manifest_check = verify_piia_v2_frozen()
    if not manifest_check["ok"]:
        raise ValueError(f"Frozen PIIA v2 manifest verification failed: {manifest_check['issues']}")

    if snapshot_override is not None:
        mode = MODE_INJECTED  # never trust a caller-provided "mode" label
    elif demo_mode:
        mode = MODE_DEMO
    else:
        mode = MODE_LIVE
    required_kind = MODE_TO_KIND[mode]
    if db_path is None:
        if mode == MODE_INJECTED:
            raise LedgerKindError("Caller-injected snapshots require an explicit, non-live db_path.")
        db_path = default_ledger_path(demo=(mode == MODE_DEMO))
    db_path = Path(db_path)
    if mode != MODE_LIVE and _same_file(db_path, REAL_LEDGER_PATH):
        raise LedgerKindError("Demo or injected records can never be written to the real prospective ledger path.")

    cfg = load_v2_config()
    now = _utc_now()
    now_iso = now.isoformat()

    if snapshot_override is not None:
        collected = snapshot_override
    elif demo_mode:
        from piia.intelligence.trade_assistant import generate_demo_snapshot

        demo_snap = generate_demo_snapshot(now=now)
        collected = {
            "collected_at_utc": now_iso,
            "market_snapshot": demo_snap,
            "books_snapshot": {
                "ok": True,
                "instruments": {
                    s: {"orderbook": demo_snap["instruments"][s]["orderbook"], "contract": None}
                    for s in SYMBOLS
                },
            },
            "tiers_snapshot": {"ok": False, "instruments": {}},
            "settled_funding": {"ok": False, "instruments": {}, "error": "demo_mode"},
            "macro_context": {
                "status": "unavailable",
                "historical_point_in_time_lagged": False,
                "macro_tightening_headwind": False,
                "macro_easing_tailwind": False,
            },
            "event_calendar_context": {"status": "demo_unavailable", "historical_replay_supported": False},
            "flows_positioning_context": {
                "status": "demo_unavailable",
                "liquidity_state": "unknown",
                "historical_replay_supported": False,
            },
            "onchain_context": {"status": "demo_unavailable", "historical_replay_supported": False},
            "polymarket_context": {"status": "demo_unavailable", "historical_replay_supported": False},
            "chronos_forecast": {"weights_ran": False, "forecasts": {}},
        }
    else:
        cycle_art_dir = ROOT / "artifacts" / "piia_v2_cycles" / now.strftime("%Y%m%dT%H%M%SZ")
        collected = collect_live_six_pillar_snapshot(SYMBOLS, snapshot_dir=cycle_art_dir)

    # ---- Three-contract closed-bar alignment gate (no ledger I/O before this passes) ----
    one_h = pd.Timedelta(hours=1)
    pre_reasons: list[str] = []
    if mode == MODE_INJECTED:
        raw_ref = collected.get("collected_at_utc")
        if not raw_ref:
            pre_reasons.append("injected_snapshot_missing_collected_at_utc")
            expected_cutoff = _ts_utc(now).floor("h")
        else:
            expected_cutoff = _ts_utc(raw_ref).floor("h")
    else:
        expected_cutoff = _ts_utc(now).floor("h")
        raw_done = collected.get("collected_at_utc")
        if raw_done and _ts_utc(raw_done).floor("h") != expected_cutoff:
            pre_reasons.append("collection_crossed_hour_boundary")

    align = check_three_contract_alignment(collected["market_snapshot"], expected_cutoff=expected_cutoff)
    reasons = pre_reasons + list(align["reasons"])
    if reasons:
        return {
            "status": "rejected_input_alignment",
            "reasons": reasons,
            "expected_cutoff_utc": expected_cutoff.isoformat(),
            "last_close_by_symbol": align["last_close_by_symbol"],
            "mode": mode,
            "db_path": _display_path(db_path),
            "ledger_written": False,
            "live_orders_submitted": False,
            "order_permissions": False,
        }

    mkt_snap = collected["market_snapshot"]
    cutoff_hour = expected_cutoff
    signal_bar_open = cutoff_hour - one_h  # explicit common signal bar (closes exactly at cutoff)
    prev_bar_open = cutoff_hour - 2 * one_h
    prepared_tables: dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        df_aligned = align["frames"][sym].loc[lambda d: d.index <= signal_bar_open]
        prepared_tables[sym] = prepare_market_feature_table(df_aligned, cfg=cfg)

    cutoff_iso = cutoff_hour.isoformat()
    settled_funding_ctx = collected.get("settled_funding")
    base_slip = float(cfg["risk_and_costs"]["slippage_each_side_base"])

    conn = _init_db(db_path, ledger_kind=required_kind)
    try:
        _assert_ledger_accepts(db_path, conn, required_kind)
        existing_cycle = conn.execute(
            "SELECT cycle_id FROM cycles WHERE cutoff_hour_utc = ?", (cutoff_iso,)
        ).fetchone()
        if existing_cycle is not None:
            return {
                "status": "already_recorded_for_hour",
                "cycle_id": existing_cycle["cycle_id"],
                "cutoff_hour_utc": cutoff_iso,
                "db_path": _display_path(db_path),
                "order_permissions": False,
            }

        slip_map = {name: base_slip * mult for name, _, mult in SHADOW_SPECS}
        slip_map["human_approved_paper"] = base_slip

        books_to_track = [*SHADOW_ARMS, "human_approved_paper"]
        book_states: dict[str, dict[str, Any]] = {}
        for bname in books_to_track:
            st = _reconstruct_book_state(conn, bname, starting_cash)
            if not conn.execute("SELECT 1 FROM ledger_events WHERE book_name = ? LIMIT 1", (bname,)).fetchone():
                _append_event(
                    conn,
                    bname,
                    "init_book",
                    None,
                    {
                        "starting_cash": starting_cash,
                        "mode": mode,
                        "book_name": bname,
                        "slippage_each_side": slip_map[bname],
                    },
                    starting_cash,
                    now_iso,
                )
            st = _reconcile_open_position_with_bars(
                conn,
                bname,
                st,
                prepared_tables,
                settled_funding_ctx,
                cfg,
                now_iso,
                slippage_each_side=slip_map[bname],
                current_cutoff_utc=cutoff_hour,
            )
            book_states[bname] = st

        cycle_id = str(uuid.uuid4())
        traces_by_symbol: dict[str, dict[str, Any]] = {}
        bundles_by_symbol: dict[str, dict[str, Any]] = {}
        human_proposals_created: list[dict[str, Any]] = []

        m4_state = book_states["M4_full_piia_v2"]
        m4_open_count = 1 if m4_state["open_pos"] is not None else 0
        m4_halted = bool(m4_state["halted"])
        human_state = book_states["human_approved_paper"]

        for sym in SYMBOLS:
            df_sym = prepared_tables[sym]
            sig_row = df_sym.loc[signal_bar_open]
            prev_c = float(df_sym.loc[prev_bar_open]["close"]) if prev_bar_open in df_sym.index else None
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
            bundles_by_symbol[sym] = bundle

            trace = run_symbol_decision_trace(
                bundle,
                equity_usdt=float(m4_state["cash"]),
                free_cash_usdt=float(m4_state["cash"]),
                open_positions_count=m4_open_count,
                kill_switch_halted=m4_halted,
                spec_override=b_inst.get("contract"),
                enable_cost_gate=True,
                enable_macro_filter=True,
                enable_stat_forecast_filter=True,
                enable_neural_forecast_filter=bool(
                    bundle["pillars"]["forecast"]["neural_chronos"].get("available")
                ),
                enable_live_external=(not demo_mode),
                cfg=cfg,
            )
            traces_by_symbol[sym] = trace

            conn.execute(
                "INSERT INTO decision_traces(decision_id, cycle_id, symbol, decision_utc, final_action, stand_aside_reason, trace_json, trace_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
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

            if (
                trace["final_action"] in {"long", "short"}
                and not m4_halted
                and not human_state["halted"]
                and human_state["open_pos"] is None
            ):
                prop_id = str(uuid.uuid4())
                expires = (now + timedelta(minutes=15)).isoformat()
                rg = trace["programmatic_risk_gate"]
                tkr = inst_snap.get("ticker") or {}
                quote_obs = tkr.get("observed_at_utc") or mkt_snap.get("retrieved_at_utc") or now_iso
                prop_body = {
                    "proposal_id": prop_id,
                    "cycle_id": cycle_id,
                    "decision_id": trace["decision_id"],
                    "cutoff_hour_utc": cutoff_iso,
                    "mode": mode,
                    "symbol": sym,
                    "side": trace["final_action"],
                    "atr14": bundle["pillars"]["market"].get("atr14"),
                    "funding_rate_8h": f_rate,
                    "synth_out": trace["synthesizer"],
                    "quote_snapshot": {
                        "price": rg["entry_plan"]["reference_price"],
                        "observed_at_utc": quote_obs,
                        "source": "cycle_snapshot_reference",
                    },
                    "entry_plan": rg["entry_plan"],
                    "exit_plan": rg["exit_plan"],
                    "position_sizing": rg["position_sizing"],
                    "cost_and_risk": rg["cost_and_risk"],
                    "approved": False,
                    "live_orders": False,
                    "expires_at_utc": expires,
                }
                conn.execute(
                    "INSERT INTO human_proposals(proposal_id, cycle_id, decision_id, created_at_utc, expires_at_utc, symbol, side, proposal_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        prop_id,
                        cycle_id,
                        trace["decision_id"],
                        now_iso,
                        expires,
                        sym,
                        trace["final_action"],
                        json.dumps(prop_body, sort_keys=True, default=str),
                    ),
                )
                human_proposals_created.append(prop_body)

        # Open shadow positions on M4 (`1x` and `2x` slippage) if flat & not halted
        for arm_name, slip_val in (("M4_full_piia_v2", base_slip), ("M4_full_piia_v2__2x_slip", base_slip * 2.0)):
            s_state = book_states[arm_name]
            if s_state["open_pos"] is not None or s_state["halted"]:
                continue
            cand_list: list[tuple[float, str, dict[str, Any], dict[str, Any]]] = []
            for sym in SYMBOLS:
                tr = traces_by_symbol[sym]
                if tr["final_action"] not in {"long", "short"}:
                    continue
                b_inst = ((collected.get("books_snapshot") or {}).get("instruments") or {}).get(sym) or {}
                rg_s = evaluate_programmatic_risk(
                    symbol=sym,
                    synth_out=tr["synthesizer"],
                    market_pillar=bundles_by_symbol[sym]["pillars"]["market"],
                    equity_usdt=float(s_state["cash"]),
                    free_cash_usdt=float(s_state["cash"]),
                    open_positions_count=0,
                    kill_switch_halted=bool(s_state["halted"]),
                    slippage_override=slip_val,
                    spec_override=b_inst.get("contract"),
                    cfg=cfg,
                )
                if rg_s["approved_by_risk_gate"]:
                    cand_list.append((float(tr["synthesizer"]["alignment_score"]), sym, tr, rg_s))
            if cand_list:
                cand_list.sort(key=lambda x: (-x[0], x[1]))
                _, best_sym, best_tr, rg_ok = cand_list[0]
                fee = float(rg_ok["cost_and_risk"]["entry_taker_fee_usdt"])
                new_cash = float(s_state["cash"]) - fee
                pos_id = str(uuid.uuid4())
                sp_sym = resolve_spec(best_sym).to_v1_spec()
                open_payload = {
                    "position_id": pos_id,
                    "symbol": best_sym,
                    "side": best_tr["final_action"],
                    "setup_type": best_tr["synthesizer"]["setup_type"],
                    "qty": float(rg_ok["position_sizing"]["qty_contracts"]),
                    "contract_size": sp_sym.contract_size,
                    "entry": float(rg_ok["entry_plan"]["assumed_fill_with_slippage"]),
                    "entry_reference_px": float(rg_ok["entry_plan"]["reference_price"]),
                    "slippage_each_side": slip_val,
                    "stop": float(rg_ok["exit_plan"]["initial_stop_loss"]),
                    "stop_effective_from_utc": cutoff_iso,
                    "be_stop": float(rg_ok["exit_plan"]["fee_adjusted_breakeven_stop"]),
                    "be_trigger": float(rg_ok["exit_plan"]["breakeven_ratchet_trigger_price"]),
                    "be_locked": False,
                    "max_hold_hours": int(rg_ok["exit_plan"]["max_hold_hours"]),
                    "entry_fee": fee,
                    "entry_fee_provenance": {"taker_rate": sp_sym.taker, "basis": "contract_spec_taker_fee"},
                    "known_funding_paid_usdt": 0.0,
                    "funding_complete": True,
                    "unknown_funding_count": 0,
                    "opened_at_utc": cutoff_iso,
                    "entry_bar_open_utc": cutoff_iso,
                    "last_processed_bar_close_utc": cutoff_iso,
                    "decision_id": best_tr["decision_id"],
                }
                _append_event(conn, arm_name, "open_position", best_sym, open_payload, new_cash, now_iso)

        # Evaluate simple shadow baselines (`B1`, `B2`, `B3`, `1x` & `2x` slippage) under identical halt/cost rules
        from piia.intelligence.agent_v2.ablation import _baseline_candidate_from_row

        for arm_name, base_arm, mult in SHADOW_SPECS:
            if base_arm in {"B0_cash", "M4_full_piia_v2"}:
                continue
            b_state = book_states[arm_name]
            if b_state["open_pos"] is not None or b_state["halted"]:
                continue
            slip_b = base_slip * mult
            for sym in SYMBOLS:
                df_sym = prepared_tables[sym]
                if signal_bar_open not in df_sym.index or prev_bar_open not in df_sym.index:
                    continue
                row_sig = df_sym.loc[signal_bar_open]
                row_prev = df_sym.loc[prev_bar_open]
                cand = _baseline_candidate_from_row(base_arm, sym, row_sig, row_prev)
                if cand.get("candidate_direction") not in {"long", "short"}:
                    continue
                mkt_pil = {
                    "ohlc": {"close": float(row_sig["close"])},
                    "atr14": float(row_sig["atr14"]) if pd.notna(row_sig["atr14"]) else 0.0,
                    "funding_rate_8h": None,
                }
                rg_b = evaluate_programmatic_risk(
                    symbol=sym,
                    synth_out=cand,
                    market_pillar=mkt_pil,
                    equity_usdt=float(b_state["cash"]),
                    free_cash_usdt=float(b_state["cash"]),
                    open_positions_count=0,
                    kill_switch_halted=bool(b_state["halted"]),
                    slippage_override=slip_b,
                    cfg=cfg,
                )
                if rg_b["approved_by_risk_gate"]:
                    fee_b = float(rg_b["cost_and_risk"]["entry_taker_fee_usdt"])
                    new_cash_b = float(b_state["cash"]) - fee_b
                    sp_b = resolve_spec(sym).to_v1_spec()
                    open_b = {
                        "position_id": str(uuid.uuid4()),
                        "symbol": sym,
                        "side": rg_b["final_action"],
                        "setup_type": cand["setup_type"],
                        "qty": float(rg_b["position_sizing"]["qty_contracts"]),
                        "contract_size": sp_b.contract_size,
                        "entry": float(rg_b["entry_plan"]["assumed_fill_with_slippage"]),
                        "entry_reference_px": float(rg_b["entry_plan"]["reference_price"]),
                        "slippage_each_side": slip_b,
                        "stop": float(rg_b["exit_plan"]["initial_stop_loss"]),
                        "stop_effective_from_utc": cutoff_iso,
                        "be_stop": None,
                        "be_trigger": None,
                        "be_locked": False,
                        "max_hold_hours": 168,
                        "entry_fee": fee_b,
                        "entry_fee_provenance": {"taker_rate": sp_b.taker, "basis": "contract_spec_taker_fee"},
                        "known_funding_paid_usdt": 0.0,
                        "funding_complete": True,
                        "unknown_funding_count": 0,
                        "opened_at_utc": cutoff_iso,
                        "entry_bar_open_utc": cutoff_iso,
                        "last_processed_bar_close_utc": cutoff_iso,
                        "decision_id": f"{arm_name}:{cutoff_iso}:{sym}",
                    }
                    _append_event(conn, arm_name, "open_position", sym, open_b, new_cash_b, now_iso)
                    break

        pillars_summary = {
            "macro_status": (collected.get("macro_context") or {}).get("status"),
            "btc_etf_usable": ((collected.get("flows_positioning_context") or {}).get("btc_etf_flows") or {}).get("usable"),
            "gold_cot_usable": ((collected.get("flows_positioning_context") or {}).get("gold_cftc_cot") or {}).get("usable"),
            "liquidity_state": (collected.get("flows_positioning_context") or {}).get("liquidity_state"),
            "btc_onchain_status": (collected.get("onchain_context") or {}).get("status"),
            "polymarket_status": (collected.get("polymarket_context") or {}).get("status"),
            "settled_funding_ok": bool((settled_funding_ctx or {}).get("ok")),
            "chronos_weights_ran": bool((collected.get("chronos_forecast") or {}).get("weights_ran")),
            "books_ok": bool((collected.get("books_snapshot") or {}).get("ok")),
            "tiers_ok": bool((collected.get("tiers_snapshot") or {}).get("ok")),
        }
        recorded_at = _utc_now()
        if mode != MODE_INJECTED and recorded_at >= (cutoff_hour + one_h).to_pydatetime():
            conn.rollback()
            return {
                "status": "rejected_input_alignment",
                "reasons": ["capture_completed_after_cutoff_hour"],
                "expected_cutoff_utc": cutoff_iso,
                "mode": mode,
                "db_path": _display_path(db_path),
                "ledger_written": False,
                "live_orders_submitted": False,
                "order_permissions": False,
            }
        recorded_iso = recorded_at.isoformat()
        cycle_doc = {
            "cycle_id": cycle_id,
            "cutoff_hour_utc": cutoff_iso,
            "capture_started_utc": now_iso,
            "recorded_at_utc": recorded_iso,
            "manifest_sha256": manifest_check["manifest_sha256"],
            "policy_version": cfg["version"],
            "mode": mode,
            "pillars_summary": pillars_summary,
            "decision_hashes": {s: tr["trace_hash"] for s, tr in traces_by_symbol.items()},
        }
        c_hash = canonical_sha256(cycle_doc)
        conn.execute(
            "INSERT INTO cycles(cycle_id, cutoff_hour_utc, recorded_at_utc, manifest_sha256, mode, pillars_summary_json, "
            "cycle_hash, capture_started_utc, policy_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                cycle_id,
                cutoff_iso,
                recorded_iso,
                manifest_check["manifest_sha256"],
                mode,
                json.dumps(pillars_summary, sort_keys=True),
                c_hash,
                now_iso,
                cfg["version"],
            ),
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
                    "module_execution": tr.get("module_execution"),
                    "entry_plan": tr["programmatic_risk_gate"]["entry_plan"],
                    "exit_plan": tr["programmatic_risk_gate"]["exit_plan"],
                    "position_sizing": tr["programmatic_risk_gate"]["position_sizing"],
                    "cost_and_risk": tr["programmatic_risk_gate"]["cost_and_risk"],
                }
                for s, tr in traces_by_symbol.items()
            },
            "unapproved_human_proposals": human_proposals_created,
            "db_path": _display_path(db_path),
            "live_orders_submitted": False,
            "order_permissions": False,
        }
    finally:
        conn.close()


def _fetch_fresh_approval_context(symbol: str) -> dict[str, Any]:
    from piia.intelligence.public_data import fetch_public_snapshot
    from piia.intelligence.public_orderbook import fetch_execution_books
    from piia.intelligence.public_position_tiers import fetch_position_tiers

    mkt = fetch_public_snapshot([symbol], candle_limit=120)
    books = fetch_execution_books((symbol,))
    tiers = fetch_position_tiers((symbol,))
    settled = fetch_public_settled_funding_recent((symbol,))
    inst = (mkt.get("instruments") or {}).get(symbol) or {}
    tkr = inst.get("ticker") or {}
    df = _candles_to_df(inst.get("candles_closed_1h") or [])
    prep = prepare_market_feature_table(df) if not df.empty else pd.DataFrame()
    quote = None
    if tkr and tkr.get("mark") is not None:
        quote = {
            "price": float(tkr["mark"]),
            "observed_at_utc": tkr["observed_at_utc"],
            "source": tkr.get("provenance") or "public_bitget_ticker",
        }
    return {
        "quote": quote,
        "prepared_tables": {symbol: prep} if not prep.empty else {},
        "settled_funding": settled,
        "orderbook": ((books.get("instruments") or {}).get(symbol) or {}).get("orderbook"),
        "tiers": ((tiers.get("instruments") or {}).get(symbol)),
        "contract": ((books.get("instruments") or {}).get(symbol) or {}).get("contract"),
        "funding_rate_8h": (inst.get("funding") or {}).get("funding_rate"),
    }


def approve_human_proposal(
    proposal_id: str,
    *,
    choice: str,
    actor: str = "local-investor",
    note: str = "",
    db_path: Path | None = None,
    current_quote: dict[str, Any] | None = None,
    prepared_tables: dict[str, pd.DataFrame] | None = None,
    settled_funding: dict[str, Any] | None = None,
    now_override: datetime | None = None,
    fetch_live_quote: bool | None = None,
) -> dict[str, Any]:
    """Explicit human approval/rejection with fresh quote, balance, position, and halt revalidation."""
    if choice not in {"approve", "reject"}:
        raise ValueError("choice must be 'approve' or 'reject'")
    cfg = load_v2_config()
    db_path = Path(db_path) if db_path is not None else default_ledger_path(demo=False)
    if not db_path.is_file():
        raise ValueError(f"Ledger not found: {_display_path(db_path)}")
    conn = _init_db(db_path)
    try:
        ledger_kind = _read_ledger_kind(conn)
        if ledger_kind == KIND_LIVE and (
            current_quote is not None
            or prepared_tables is not None
            or settled_funding is not None
            or now_override is not None
            or fetch_live_quote is False
        ):
            raise LedgerKindError(
                "Caller-injected quotes, bars, funding, or clocks are refused on the live prospective ledger; "
                "approval there must revalidate against freshly fetched public data."
            )
        prop_row = conn.execute("SELECT * FROM human_proposals WHERE proposal_id = ?", (proposal_id,)).fetchone()
        if not prop_row:
            raise ValueError(f"Proposal not found: {proposal_id}")
        existing = conn.execute("SELECT * FROM human_decisions WHERE proposal_id = ?", (proposal_id,)).fetchone()
        if existing:
            raise ValueError(f"Proposal {proposal_id} was already decided ({existing['choice']})")

        now = (now_override or _utc_now()).astimezone(timezone.utc)
        now_iso = now.isoformat()
        prop = json.loads(prop_row["proposal_json"])

        if choice == "reject":
            approval_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO human_decisions(approval_id, proposal_id, decided_at_utc, actor, choice, note) VALUES (?, ?, ?, ?, ?, ?)",
                (approval_id, proposal_id, now_iso, actor, choice, note),
            )
            conn.commit()
            return {
                "approval_id": approval_id,
                "proposal_id": proposal_id,
                "choice": choice,
                "paper_position_opened": None,
                "live_orders_submitted": False,
                "order_permissions": False,
            }

        # Reconcile open human position / halt state first if fresh market data available
        live_ctx: dict[str, Any] = {}
        if (
            prop.get("mode") == "live_prospective_public_6_pillars"
            and fetch_live_quote is not False
            and current_quote is None
        ):
            live_ctx = _fetch_fresh_approval_context(prop["symbol"])

        reconcile_tables = prepared_tables if prepared_tables is not None else live_ctx.get("prepared_tables")
        reconcile_funding = settled_funding if settled_funding is not None else live_ctx.get("settled_funding")
        st = _reconstruct_book_state(conn, "human_approved_paper", 500.0)
        if not conn.execute("SELECT 1 FROM ledger_events WHERE book_name = 'human_approved_paper' LIMIT 1").fetchone():
            _append_event(
                conn,
                "human_approved_paper",
                "init_book",
                None,
                {"starting_cash": 500.0, "mode": prop.get("mode") or "approval_init"},
                500.0,
                now_iso,
            )
        if reconcile_tables:
            st = _reconcile_open_position_with_bars(
                conn,
                "human_approved_paper",
                st,
                reconcile_tables,
                reconcile_funding,
                cfg,
                now_iso,
                current_cutoff_utc=pd.Timestamp(now.replace(minute=0, second=0, microsecond=0)),
            )
            conn.commit()
        else:
            d_key = now.date().isoformat()
            day_anchor = st["day_anchors"].get(d_key, st["starting_cash"])
            if st["open_pos"] is None:
                if _record_halt_if_breached(
                    conn,
                    "human_approved_paper",
                    st,
                    eq_usdt=float(st["cash"]),
                    day_key=d_key,
                    day_start_eq=day_anchor,
                    cfg=cfg,
                    at_utc_iso=now_iso,
                    now_iso=now_iso,
                ):
                    conn.commit()

        # 1. Expiry check
        expires = datetime.fromisoformat(prop_row["expires_at_utc"]).astimezone(timezone.utc)
        if now > expires:
            raise ValueError("Proposal rejected / new proposal required [proposal_expired]: proposal expired; run a fresh agent cycle.")

        # 2. Halt check (3% daily / 10% total)
        if st["halted"]:
            raise ValueError(
                f"Proposal rejected / new proposal required [kill_switch_halted:{st['halt_reason']}]: paper book is halted."
            )

        # 3. Single-position exclusivity check
        if st["open_pos"] is not None:
            raise ValueError(
                "Proposal rejected / new proposal required [open_position_exists]: human_approved_paper already has an open position."
            )

        # 4. Quote freshness, invalidation & drift check
        q = current_quote or live_ctx.get("quote")
        if q is None and prop.get("mode") != "live_prospective_public_6_pillars":
            q = prop.get("quote_snapshot")
        if not isinstance(q, dict) or q.get("observed_at_utc") is None:
            raise ValueError(
                "Proposal rejected / new proposal required [quote_missing_or_stale]: current quote with observed_at_utc is required."
            )
        q_obs = datetime.fromisoformat(str(q["observed_at_utc"]))
        if q_obs.tzinfo is None or q_obs.utcoffset() is None:
            raise ValueError("Proposal rejected / new proposal required [quote_naive_timestamp]: timezone-aware quote timestamp required.")
        q_age = now - q_obs.astimezone(timezone.utc)
        if not timedelta(seconds=-5) <= q_age <= timedelta(minutes=5):
            raise ValueError(
                "Proposal rejected / new proposal required [quote_stale]: quote age exceeds 5-minute freshness limit."
            )
        raw_q_px = q.get("price") if q.get("price") is not None else q.get("mark")
        try:
            q_px = float(raw_q_px)
        except (TypeError, ValueError) as exc:
            raise ValueError("Proposal rejected / new proposal required [invalid_quote_price]: invalid price.") from exc
        if not math.isfinite(q_px) or q_px <= 0:
            raise ValueError("Proposal rejected / new proposal required [invalid_quote_price]: nonpositive price.")

        ref_px = float(prop["entry_plan"]["reference_price"])
        stop_px = float(prop["exit_plan"]["initial_stop_loss"])
        atr14 = float(prop.get("atr14") or 0.0)
        side = prop["side"]
        if (side == "long" and q_px <= stop_px) or (side == "short" and q_px >= stop_px):
            raise ValueError(
                "Proposal rejected / new proposal required [quote_crossed_stop]: quote already violated initial stop-loss."
            )
        drift_limit = max(0.5 * atr14, 0.005 * ref_px) if atr14 > 0 else 0.005 * ref_px
        adverse_drift = (q_px - ref_px) if side == "long" else (ref_px - q_px)
        if adverse_drift > drift_limit:
            raise ValueError(
                "Proposal rejected / new proposal required [quote_drifted_adversely]: price moved unfavorably beyond drift budget; request new proposal."
            )

        # 5. Re-evaluate deterministic balance & risk sizing at current quote and current cash
        if atr14 <= 0 and stop_px > 0:
            atr14 = abs(ref_px - stop_px) / float(cfg["roles_and_gates"]["trend_stop_atr_mult"])
        synth_out = prop.get("synth_out") or {
            "candidate_direction": side,
            "setup_type": "approved_proposal_recheck",
            "proposed_stop_atr_mult": abs(ref_px - stop_px) / atr14 if atr14 > 0 else 2.0,
            "proposed_exit_rule": prop["exit_plan"].get("trailing_or_target_rule"),
            "invalidation": prop["exit_plan"].get("invalidation"),
        }
        mkt_pil = {
            "ohlc": {"close": q_px},
            "atr14": atr14,
            "funding_rate_8h": live_ctx.get("funding_rate_8h", prop.get("funding_rate_8h")),
            "live_orderbook": live_ctx.get("orderbook"),
            "live_position_tiers": live_ctx.get("tiers"),
        }
        recheck = evaluate_programmatic_risk(
            symbol=prop["symbol"],
            synth_out=synth_out,
            market_pillar=mkt_pil,
            equity_usdt=float(st["cash"]),
            free_cash_usdt=float(st["cash"]),
            open_positions_count=0,
            kill_switch_halted=bool(st["halted"]),
            planned_entry_price=q_px,
            spec_override=live_ctx.get("contract"),
            cfg=cfg,
        )
        if not recheck["approved_by_risk_gate"]:
            raise ValueError(
                f"Proposal rejected / new proposal required [{recheck['stand_aside_reason']}]: balance or risk gate failed at approval."
            )

        approval_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO human_decisions(approval_id, proposal_id, decided_at_utc, actor, choice, note) VALUES (?, ?, ?, ?, ?, ?)",
            (approval_id, proposal_id, now_iso, actor, choice, note),
        )
        fee = float(recheck["cost_and_risk"]["entry_taker_fee_usdt"])
        new_cash = float(st["cash"]) - fee
        floor_hour = now.replace(minute=0, second=0, microsecond=0).isoformat()
        sp_sym = resolve_spec(prop["symbol"], live_ctx.get("contract")).to_v1_spec()
        opened = {
            "position_id": str(uuid.uuid4()),
            "symbol": prop["symbol"],
            "side": side,
            "setup_type": synth_out.get("setup_type") or "approved_proposal",
            "qty": float(recheck["position_sizing"]["qty_contracts"]),
            "contract_size": sp_sym.contract_size,
            "entry": float(recheck["entry_plan"]["assumed_fill_with_slippage"]),
            "entry_reference_px": q_px,
            "slippage_each_side": float(cfg["risk_and_costs"]["slippage_each_side_base"]),
            "stop": float(recheck["exit_plan"]["initial_stop_loss"]),
            "stop_effective_from_utc": now_iso,
            "be_stop": float(recheck["exit_plan"]["fee_adjusted_breakeven_stop"]),
            "be_trigger": float(recheck["exit_plan"]["breakeven_ratchet_trigger_price"]),
            "be_locked": False,
            "max_hold_hours": int(recheck["exit_plan"]["max_hold_hours"]),
            "entry_fee": fee,
            "entry_fee_provenance": {"taker_rate": sp_sym.taker, "basis": "contract_spec_taker_fee"},
            "known_funding_paid_usdt": 0.0,
            "funding_complete": True,
            "unknown_funding_count": 0,
            "opened_at_utc": now_iso,
            "entry_bar_open_utc": floor_hour,
            "last_processed_bar_close_utc": floor_hour,
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
            "revalidated_with_quote": q,
            "live_orders_submitted": False,
            "order_permissions": False,
        }
    finally:
        conn.close()


def _open_readonly(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _mode_kind(mode: str | None) -> str:
    if mode == MODE_LIVE:
        return KIND_LIVE
    if mode == MODE_DEMO:
        return KIND_DEMO
    return KIND_INJECTED


def inspect_ledger(db_path: Path) -> dict[str, Any]:
    """Read-only compatibility audit of a PIIA v2 ledger file. Never modifies the file."""
    db_path = Path(db_path)
    out: dict[str, Any] = {"db_path": _display_path(db_path), "exists": db_path.is_file()}
    if not out["exists"]:
        out.update({"classification": "absent", "usable_as_live_prospective_ledger": False, "notes": []})
        return out
    conn = _open_readonly(db_path)
    try:
        tables = _sqlite_tables(conn)
        kind = _read_ledger_kind(conn)
        modes: dict[str, int] = {}
        manifests: dict[str, int] = {}
        if "cycles" in tables:
            for r in conn.execute("SELECT mode, manifest_sha256, COUNT(*) AS n FROM cycles GROUP BY mode, manifest_sha256"):
                modes[r["mode"]] = modes.get(r["mode"], 0) + int(r["n"])
                manifests[r["manifest_sha256"]] = manifests.get(r["manifest_sha256"], 0) + int(r["n"])
        book_modes: dict[str, int] = {}
        if "ledger_events" in tables:
            for r in conn.execute("SELECT payload_json FROM ledger_events WHERE event_kind = 'init_book'"):
                m = str(json.loads(r["payload_json"]).get("mode") or "unknown")
                book_modes[m] = book_modes.get(m, 0) + 1
    finally:
        conn.close()

    kinds_seen = {_mode_kind(m) for m in modes} | {_mode_kind(m) for m in book_modes if m != "unknown"}
    if not modes and not book_modes:
        classification = "empty"
    elif not kinds_seen:
        classification = "no_cycles_unlabelled_book_events"
    elif len(kinds_seen) > 1:
        classification = "mixed_demo_live_or_injected"
    else:
        classification = f"{next(iter(kinds_seen))}_only"
    label = kind if kind is not None else "legacy_unlabeled_pre_v2_2"
    usable = bool(
        kind == KIND_LIVE
        and classification in {"empty", f"{KIND_LIVE}_only"}
        and _same_file(db_path, REAL_LEDGER_PATH)
    )
    notes: list[str] = []
    if kind is None:
        notes.append(
            "Pre-v2.2 ledger without an evidence-kind label: preserved untouched, never written, never counted. "
            "Its simulated cash and positions may already reflect demo or injected cycles."
        )
    if classification.startswith("mixed"):
        notes.append("Mixed evidence: demo or injected cycles share simulated balances with live cycles; invalid for validation.")
    out.update(
        {
            "ledger_kind_label": label,
            "classification": classification,
            "cycle_modes": modes,
            "cycle_manifests": manifests,
            "book_init_modes": book_modes,
            "usable_as_live_prospective_ledger": usable,
            "notes": notes,
        }
    )
    return out


def compute_prospective_validation(
    cycle_rows: list[dict[str, Any]],
    *,
    ledger_kind: str | None,
    cfg: dict[str, Any],
    manifest_sha256: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Count only post-freeze, on-time, live observations recorded by the current frozen code."""
    required = int(cfg["evaluation_rules"]["prospective_min_contiguous_hours"])
    freeze_start = _ts_utc(cfg["prospective_freeze_started_utc"])
    base = {
        "required_contiguous_hours": required,
        "freeze_started_utc": freeze_start.isoformat(),
        "counted_observations": 0,
        "longest_contiguous_hours": 0,
        "longest_contiguous_window": None,
        "missing_hours_within_observed_span": 0,
        "missing_hours_since_freeze": None,
        "excluded_cycles": {},
    }
    if ledger_kind != KIND_LIVE:
        base.update(
            {
                "validation_status": "NOT_A_PROSPECTIVE_LEDGER",
                "validation_note": "非真实前瞻账本（演示/注入/旧版未标注），不计入任何验证计数。",
            }
        )
        return base

    one_h = pd.Timedelta(hours=1)
    excluded: dict[str, int] = {}
    counted: set[pd.Timestamp] = set()

    def _excl(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    for row in cycle_rows:
        try:
            cutoff = _ts_utc(row["cutoff_hour_utc"])
            recorded = _ts_utc(row["recorded_at_utc"])
            started = _ts_utc(row.get("capture_started_utc") or row["recorded_at_utc"])
        except (KeyError, TypeError, ValueError):
            _excl("unparseable_timestamps")
            continue
        if row.get("mode") != MODE_LIVE:
            _excl("not_live_mode")
        elif row.get("manifest_sha256") != manifest_sha256:
            _excl("different_frozen_manifest")
        elif row.get("policy_version") != cfg["version"]:
            _excl("different_policy_version")
        elif cutoff < freeze_start:
            _excl("before_freeze_start")
        elif cutoff != cutoff.floor("h"):
            _excl("cutoff_not_hour_aligned")
        elif not (cutoff <= started < cutoff + one_h and cutoff <= recorded < cutoff + one_h):
            _excl("capture_outside_cutoff_hour")
        elif cutoff in counted:
            _excl("duplicate_cutoff")
        else:
            counted.add(cutoff)

    ordered = sorted(counted)
    best_len, best_start, best_end = 0, None, None
    run_len, run_start = 0, None
    prev = None
    for ts in ordered:
        if prev is not None and ts - prev == one_h:
            run_len += 1
        else:
            run_len, run_start = 1, ts
        if run_len > best_len:
            best_len, best_start, best_end = run_len, run_start, ts
        prev = ts

    span_missing = 0
    if ordered:
        span_hours = int((ordered[-1] - ordered[0]) / one_h) + 1
        span_missing = span_hours - len(ordered)
    now_ts = _ts_utc(now or _utc_now())
    latest_possible = now_ts.floor("h")
    since_freeze_missing = None
    if latest_possible >= freeze_start:
        expected = int((latest_possible - freeze_start) / one_h) + 1
        since_freeze_missing = max(0, expected - len(ordered))

    sufficient = best_len >= required
    base.update(
        {
            "counted_observations": len(ordered),
            "longest_contiguous_hours": best_len,
            "longest_contiguous_window": (
                {"first_cutoff_utc": best_start.isoformat(), "last_cutoff_utc": best_end.isoformat()}
                if best_start is not None
                else None
            ),
            "missing_hours_within_observed_span": span_missing,
            "missing_hours_since_freeze": since_freeze_missing,
            "excluded_cycles": excluded,
            "validation_status": (
                "PENDING_AUDIT_CONTIGUOUS_WINDOW_REACHED" if sufficient else "PENDING_INSUFFICIENT_PROSPECTIVE_SAMPLE"
            ),
            "validation_note": (
                f"最长连续真实观测已达 {best_len} 小时（要求 >= {required}），仅进入待审计状态，尚未构成验证结论。"
                if sufficient
                else f"真实前瞻观测 {len(ordered)} 小时，最长连续 {best_len} 小时（要求 >= {required} 连续小时），样本不足，待验证。"
            ),
        }
    )
    return base


def get_continuous_paper_status(db_path: Path | None = None, *, demo: bool = False) -> dict[str, Any]:
    """Read-only summary: ledger kind audit, prospective validation counts, books, latest traces, proposals."""
    from piia.intelligence.agent_v2.manifest import verify_piia_v2_frozen

    cfg = load_v2_config()
    db_path = Path(db_path) if db_path is not None else default_ledger_path(demo=demo)
    manifest_sha = verify_piia_v2_frozen()["manifest_sha256"]
    legacy = []
    if not demo:
        legacy = [inspect_ledger(p) for p in LEGACY_LEDGER_PATHS if Path(p).is_file()]

    audit = inspect_ledger(db_path)
    if not db_path.is_file():
        validation = compute_prospective_validation([], ledger_kind=KIND_DEMO if demo else KIND_LIVE, cfg=cfg, manifest_sha256=manifest_sha)
        return {
            "initialized": False,
            "db_path": _display_path(db_path),
            "ledger_audit": audit,
            "legacy_ledgers": legacy,
            "cycles_recorded": 0,
            "prospective_validation": validation,
            "validation_status": validation["validation_status"],
            "validation_note": validation["validation_note"],
            "books": {},
            "latest_decisions": [],
            "pending_human_proposals": [],
            "order_permissions": False,
        }

    conn = _open_readonly(db_path)
    try:
        kind = _read_ledger_kind(conn)
        tables = _sqlite_tables(conn)
        cycle_cols = {r[1] for r in conn.execute("PRAGMA table_info(cycles)")} if "cycles" in tables else set()
        cycles = (
            [dict(r) for r in conn.execute("SELECT * FROM cycles ORDER BY cutoff_hour_utc DESC")]
            if "cycles" in tables
            else []
        )
        if "capture_started_utc" not in cycle_cols:
            for c in cycles:
                c.setdefault("capture_started_utc", None)
                c.setdefault("policy_version", None)
        effective_kind = kind if not audit.get("classification", "").startswith("mixed") else None
        validation = compute_prospective_validation(
            cycles, ledger_kind=effective_kind, cfg=cfg, manifest_sha256=manifest_sha
        )

        books_summary: dict[str, Any] = {}
        if "ledger_events" in tables:
            for bname in (*SHADOW_ARMS, "human_approved_paper"):
                st = _reconstruct_book_state(conn, bname, 500.0)
                pct = (
                    round((st["realized_net_pnl_usdt"] / st["starting_cash"]) * 100.0, 4)
                    if st["realized_net_pnl_usdt"] is not None and st["starting_cash"] > 0
                    else None
                )
                books_summary[bname] = {
                    "starting_cash_usdt": st["starting_cash"],
                    "cash_usdt": round(st["cash"], 4),
                    "closed_trades": len(st["closed_trades"]),
                    "realized_net_pnl_usdt": st["realized_net_pnl_usdt"],
                    "net_return_pct": pct,
                    "net_return_claim_allowed": st["net_return_claim_allowed"],
                    "realized_pnl_ex_unknown_funding_usdt": st["realized_pnl_ex_unknown_funding_usdt"],
                    "fees_paid_usdt": st["fees_paid_usdt"],
                    "known_funding_paid_usdt": st["known_funding_paid_usdt"],
                    "funding_complete": st["funding_complete"],
                    "unknown_funding_events": st["unknown_funding_events"],
                    "halted": st["halted"],
                    "halt_reason": st["halt_reason"],
                    "halted_at_utc": st["halted_at_utc"],
                    "reconciled_bar_close_utc": st["reconciled_bar_close_utc"],
                    "reconciliation_error_usdt": round(st["reconciliation_error_usdt"], 10),
                    "open_position": st["open_pos"],
                }

        latest_traces = []
        if cycles and "decision_traces" in tables:
            last_cid = cycles[0]["cycle_id"]
            for tr in conn.execute(
                "SELECT symbol, final_action, stand_aside_reason, trace_json, trace_hash FROM decision_traces "
                "WHERE cycle_id = ? ORDER BY symbol ASC",
                (last_cid,),
            ).fetchall():
                parsed = json.loads(tr["trace_json"])
                latest_traces.append(
                    {
                        "symbol": tr["symbol"],
                        "final_action": tr["final_action"],
                        "stand_aside_reason": tr["stand_aside_reason"],
                        "regime": parsed["researcher"]["regime"],
                        "disagreements": parsed["synthesizer"]["disagreements"],
                        "module_execution": parsed.get("module_execution"),
                        "entry_plan": parsed["programmatic_risk_gate"]["entry_plan"],
                        "exit_plan": parsed["programmatic_risk_gate"]["exit_plan"],
                        "position_sizing": parsed["programmatic_risk_gate"]["position_sizing"],
                        "cost_and_risk": parsed["programmatic_risk_gate"]["cost_and_risk"],
                        "trace_hash": tr["trace_hash"],
                    }
                )

        pending = []
        if "human_proposals" in tables and "human_decisions" in tables:
            for pr in conn.execute(
                """
                SELECT p.proposal_json FROM human_proposals p
                LEFT JOIN human_decisions d ON p.proposal_id = d.proposal_id
                WHERE d.proposal_id IS NULL
                ORDER BY p.created_at_utc DESC
                LIMIT 10
                """
            ).fetchall():
                pending.append(json.loads(pr["proposal_json"]))

        return {
            "initialized": True,
            "db_path": _display_path(db_path),
            "ledger_kind": kind,
            "ledger_audit": audit,
            "legacy_ledgers": legacy,
            "cycles_recorded": len(cycles),
            "prospective_validation": validation,
            "validation_status": validation["validation_status"],
            "validation_note": validation["validation_note"],
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
