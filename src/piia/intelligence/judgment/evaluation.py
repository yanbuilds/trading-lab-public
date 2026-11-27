"""Shadow ledger, 4-way AI vs Rules comparison, and 5-domain ablation engine (PIIA Judgment v3).

Strict non-interference & honesty invariants:
- Uses dedicated SQLite files (`piia_judgment_v3_shadow.sqlite` for live,
  `piia_judgment_v3_demo.sqlite` for synthetic demo), enforced by immutable
  `ledger_meta` + `LedgerKindError` + append-only SQLite triggers.
- Never modifies PIIA v2.2.1 (`0c8358c9...`) or Frozen Policy v1 ledgers/files.
- Direct-AI comparisons (`A1`, `A2`) keep `status = "NOT_RUN_NEEDS_MODEL_CONFIG"`
  when unconfigured unless an explicit caller/test provider is passed, and
  never expose future data or project internal conclusions to `A1`.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import os
import sqlite3
from typing import Any, Protocol
import uuid

import httpx
import pandas as pd

from piia.intelligence.agent_v2.pillars import SYMBOLS, canonical_sha256
from piia.intelligence.agent_v2.risk_gate import evaluate_programmatic_risk
from piia.intelligence.judgment.evidence import (
    collect_judgment_inputs,
    generate_demo_judgment_inputs,
    load_judgment_config,
    sanitize_closed_candles,
)
from piia.intelligence.judgment.manifest import verify_judgment_v3_frozen
from piia.intelligence.judgment.news import parse_aware_timestamp
from piia.intelligence.judgment.synthesizer import HORIZONS, synthesize_judgment_packet
from piia.intelligence.settings import (
    model_base_url,
    model_configured,
    model_name,
    model_provider,
)
from trading_lab.perps.risk import exit_pnl, stop_fill
from trading_lab.perps.risk_v2 import resolve_spec

ROOT = Path(__file__).resolve().parents[4]
REAL_JUDGMENT_LEDGER_PATH = ROOT / "data" / "piia" / "piia_judgment_v3_shadow.sqlite"
DEMO_JUDGMENT_LEDGER_PATH = ROOT / "data" / "piia" / "piia_judgment_v3_demo.sqlite"
JUDGMENT_APP_ID = 0x50494A33  # "PIJ3"
JUDGMENT_SCHEMA_VERSION = "piia_judgment_v3_3"

KIND_LIVE_J3 = "live_shadow_judgment_v3"
KIND_DEMO_J3 = "demo_synthetic_judgment_v3"
KIND_INJECTED_J3 = "test_or_injected_judgment_v3"


class JudgmentLedgerKindError(ValueError):
    """Raised when live, demo, and test/injected judgment ledgers would mix."""


class JudgmentAIProvider(Protocol):
    def compare(self, arm_name: str, payload: dict[str, Any]) -> dict[str, Any]: ...


def default_judgment_ledger_path(demo: bool = False) -> Path:
    return Path(DEMO_JUDGMENT_LEDGER_PATH if demo else REAL_JUDGMENT_LEDGER_PATH)


def _same_file(a: Path, b: Path) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return False


def _display_path(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).name


def _init_judgment_db(db_path: Path, *, ledger_kind: str | None = None) -> sqlite3.Connection:
    db_path = Path(db_path)
    valid_kinds = {KIND_LIVE_J3, KIND_DEMO_J3, KIND_INJECTED_J3}
    if ledger_kind is not None and ledger_kind not in valid_kinds:
        raise JudgmentLedgerKindError(f"Unknown judgment ledger kind {ledger_kind!r}")
    if ledger_kind is not None and ledger_kind != KIND_LIVE_J3 and _same_file(db_path, REAL_JUDGMENT_LEDGER_PATH):
        raise JudgmentLedgerKindError("Demo or test judgment records cannot be written to the live shadow ledger.")

    existed = db_path.is_file() and db_path.stat().st_size > 0
    if existed:
        ro = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            app = ro.execute("PRAGMA application_id").fetchone()[0]
            if app not in (0, JUDGMENT_APP_ID):
                raise ValueError(f"Unrelated SQLite database at {_display_path(db_path)}")
            tables = {r[0] for r in ro.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables:
                if "ledger_meta" not in tables:
                    raise JudgmentLedgerKindError("Unlabelled SQLite ledger refused for Judgment v3 writes.")
                row = ro.execute("SELECT value FROM ledger_meta WHERE key='ledger_kind'").fetchone()
                actual = str(row[0]) if row else None
                if ledger_kind is not None and actual != ledger_kind:
                    raise JudgmentLedgerKindError(
                        f"Ledger {_display_path(db_path)} is labelled {actual!r}; refusing {ledger_kind!r}."
                    )
        finally:
            ro.close()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    tables_now = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    had_tables = bool(tables_now)
    if not had_tables:
        conn.execute(f"PRAGMA application_id={JUDGMENT_APP_ID}")

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS ledger_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS judgment_packets (
            packet_id TEXT PRIMARY KEY,
            cutoff_utc TEXT NOT NULL UNIQUE,
            recorded_at_utc TEXT NOT NULL,
            mode TEXT NOT NULL,
            version TEXT NOT NULL,
            judgment_hash TEXT NOT NULL,
            packet_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS arm_judgments (
            row_id TEXT PRIMARY KEY,
            packet_id TEXT NOT NULL,
            cutoff_utc TEXT NOT NULL,
            arm_name TEXT NOT NULL,
            symbol TEXT NOT NULL,
            horizon TEXT NOT NULL,
            target_outcome_utc TEXT NOT NULL,
            predicted_bias TEXT NOT NULL,
            technical_bias TEXT NOT NULL,
            entry_price_ref REAL,
            judgment_hash TEXT NOT NULL,
            UNIQUE(cutoff_utc, arm_name, symbol, horizon)
        );
        CREATE TABLE IF NOT EXISTS outcome_observations (
            obs_id TEXT PRIMARY KEY,
            row_id TEXT NOT NULL UNIQUE,
            resolved_at_utc TEXT NOT NULL,
            cutoff_utc TEXT NOT NULL,
            arm_name TEXT NOT NULL,
            symbol TEXT NOT NULL,
            horizon TEXT NOT NULL,
            predicted_bias TEXT NOT NULL,
            realized_return_pct REAL NOT NULL,
            direction_hit INTEGER,
            outcome_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS shadow_paper_events (
            event_id TEXT PRIMARY KEY,
            recorded_at_utc TEXT NOT NULL,
            book_name TEXT NOT NULL,
            event_kind TEXT NOT NULL,
            symbol TEXT,
            payload_json TEXT NOT NULL,
            cash_after_usdt REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chat_qa_logs (
            qa_id TEXT PRIMARY KEY,
            asked_at_utc TEXT NOT NULL,
            cutoff_utc TEXT NOT NULL,
            judgment_hash TEXT NOT NULL,
            intent TEXT NOT NULL,
            question TEXT NOT NULL,
            answer_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS archived_input_bundles (
            bundle_id TEXT PRIMARY KEY,
            packet_id TEXT NOT NULL,
            cutoff_utc TEXT NOT NULL UNIQUE,
            captured_at_utc TEXT NOT NULL,
            ledger_kind TEXT NOT NULL,
            bundle_sha256 TEXT NOT NULL,
            bundle_json TEXT NOT NULL
        );

        CREATE TRIGGER IF NOT EXISTS j3_meta_upd BEFORE UPDATE ON ledger_meta BEGIN SELECT RAISE(ABORT, 'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS j3_meta_del BEFORE DELETE ON ledger_meta BEGIN SELECT RAISE(ABORT, 'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS j3_pkt_upd BEFORE UPDATE ON judgment_packets BEGIN SELECT RAISE(ABORT, 'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS j3_pkt_del BEFORE DELETE ON judgment_packets BEGIN SELECT RAISE(ABORT, 'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS j3_arm_upd BEFORE UPDATE ON arm_judgments BEGIN SELECT RAISE(ABORT, 'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS j3_arm_del BEFORE DELETE ON arm_judgments BEGIN SELECT RAISE(ABORT, 'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS j3_out_upd BEFORE UPDATE ON outcome_observations BEGIN SELECT RAISE(ABORT, 'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS j3_out_del BEFORE DELETE ON outcome_observations BEGIN SELECT RAISE(ABORT, 'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS j3_evt_upd BEFORE UPDATE ON shadow_paper_events BEGIN SELECT RAISE(ABORT, 'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS j3_evt_del BEFORE DELETE ON shadow_paper_events BEGIN SELECT RAISE(ABORT, 'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS j3_qa_upd BEFORE UPDATE ON chat_qa_logs BEGIN SELECT RAISE(ABORT, 'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS j3_qa_del BEFORE DELETE ON chat_qa_logs BEGIN SELECT RAISE(ABORT, 'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS j3_arch_upd BEFORE UPDATE ON archived_input_bundles BEGIN SELECT RAISE(ABORT, 'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS j3_arch_del BEFORE DELETE ON archived_input_bundles BEGIN SELECT RAISE(ABORT, 'append-only'); END;
        """
    )
    existing_arm_cols = {r[1] for r in conn.execute("PRAGMA table_info(arm_judgments)").fetchall()}
    if "judgment_generated_at_utc" not in existing_arm_cols:
        conn.execute("ALTER TABLE arm_judgments ADD COLUMN judgment_generated_at_utc TEXT")
    if "earliest_executable_at_utc" not in existing_arm_cols:
        conn.execute("ALTER TABLE arm_judgments ADD COLUMN earliest_executable_at_utc TEXT")
    if not had_tables:
        kind = ledger_kind or KIND_INJECTED_J3
        conn.executemany(
            "INSERT INTO ledger_meta(key, value) VALUES (?, ?)",
            [
                ("ledger_kind", kind),
                ("schema_version", JUDGMENT_SCHEMA_VERSION),
                ("created_at_utc", datetime.now(timezone.utc).isoformat()),
            ],
        )
        conn.commit()
    return conn


def _compute_contiguous_stats(cutoffs: list[str]) -> dict[str, Any]:
    parsed = sorted({dt for dt in (parse_aware_timestamp(c) for c in cutoffs) if dt is not None})
    if not parsed:
        return {"max_contiguous_hours": 0, "missing_hours_in_span": 0, "missing_hour_cutoffs": []}
    longest = 1
    run = 1
    one_h = timedelta(hours=1)
    for i in range(1, len(parsed)):
        if parsed[i] - parsed[i - 1] == one_h:
            run += 1
            longest = max(longest, run)
        else:
            run = 1
    expected = int((parsed[-1] - parsed[0]).total_seconds() // 3600) + 1
    present_set = set(parsed)
    missing_list: list[str] = []
    cur = parsed[0]
    while cur <= parsed[-1]:
        if cur not in present_set:
            missing_list.append(cur.isoformat())
        cur += one_h
    return {
        "max_contiguous_hours": longest,
        "missing_hours_in_span": max(0, expected - len(parsed)),
        "missing_hour_cutoffs": missing_list[:48],
    }


def _resolve_pending_outcomes(
    conn: sqlite3.Connection,
    market_snapshot: dict[str, Any],
    *,
    now_cutoff: datetime,
    resolved_at_iso: str,
) -> int:
    """Resolve matured directional predictions strictly post-availability from closed 1h bars."""
    close_lookup: dict[str, dict[str, float]] = {}
    open_lookup: dict[str, dict[str, float]] = {}
    for sym in SYMBOLS:
        inst = (market_snapshot.get("instruments") or {}).get(sym) or {}
        san = sanitize_closed_candles(inst.get("candles_closed_1h"), bar_cutoff_utc=now_cutoff)["all_sanitized"]
        c_mp: dict[str, float] = {}
        o_mp: dict[str, float] = {}
        for c in san:
            c_dt = parse_aware_timestamp(c.get("close_at_utc"))
            o_dt = parse_aware_timestamp(c.get("open_at_utc"))
            if c_dt is not None and c_dt <= now_cutoff:
                c_mp[c_dt.isoformat()] = float(c["close"])
            if o_dt is not None:
                o_mp[o_dt.isoformat()] = float(c["open"])
        close_lookup[sym] = c_mp
        open_lookup[sym] = o_mp

    cfg_h = load_judgment_config()["horizons"]
    pending = conn.execute(
        """
        SELECT a.* FROM arm_judgments a
        LEFT JOIN outcome_observations o ON a.row_id = o.row_id
        WHERE o.row_id IS NULL
        """
    ).fetchall()
    resolved_n = 0
    for r in pending:
        sym = r["symbol"]
        cut_dt = parse_aware_timestamp(r["cutoff_utc"])
        exec_dt = parse_aware_timestamp(r["earliest_executable_at_utc"]) or cut_dt
        tgt_dt = parse_aware_timestamp(r["target_outcome_utc"])
        if exec_dt is None or tgt_dt is None:
            continue
        h_hours = int((cfg_h.get(r["horizon"]) or {}).get("outcome_horizon_hours") or 6)
        min_mature_dt = exec_dt + timedelta(hours=h_hours)
        if tgt_dt < min_mature_dt:
            tgt_dt = min_mature_dt
        if tgt_dt > now_cutoff:
            continue
        exit_px = close_lookup.get(sym, {}).get(tgt_dt.isoformat())
        entry_px = open_lookup.get(sym, {}).get(exec_dt.isoformat())
        if entry_px is None and cut_dt is not None and exec_dt == cut_dt:
            entry_px = r["entry_price_ref"]
        if exit_px is None or entry_px is None or float(entry_px) <= 0:
            continue
        ret_pct = (float(exit_px) / float(entry_px) - 1.0) * 100.0
        bias = r["predicted_bias"]
        hit = None
        if bias == "bullish_bias":
            hit = 1 if ret_pct > 0 else 0
        elif bias == "bearish_bias":
            hit = 1 if ret_pct < 0 else 0
        conn.execute(
            """
            INSERT INTO outcome_observations(
                obs_id, row_id, resolved_at_utc, cutoff_utc, arm_name, symbol, horizon,
                predicted_bias, realized_return_pct, direction_hit, outcome_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                r["row_id"],
                resolved_at_iso,
                r["cutoff_utc"],
                r["arm_name"],
                sym,
                r["horizon"],
                bias,
                round(ret_pct, 4),
                hit,
                json.dumps({"entry_px": entry_px, "exit_px": exit_px, "earliest_executable_at_utc": exec_dt.isoformat(), "target_utc": tgt_dt.isoformat()}),
            ),
        )
        resolved_n += 1
    return resolved_n


def _normalize_bundle_for_archive(inputs_bundle: dict[str, Any]) -> tuple[str, str]:
    cleaned = {k: v for k, v in inputs_bundle.items() if k != "_from_live_collector"}
    cleaned["replay_mode"] = False
    bundle_sha = canonical_sha256(cleaned)
    return json.dumps(cleaned, ensure_ascii=False, default=str), bundle_sha


def record_judgment_cycle(
    inputs_bundle: dict[str, Any] | None = None,
    *,
    db_path: Path | None = None,
    demo_mode: bool = False,
    starting_cash: float = 500.0,
    supplement_note: str | None = None,
    allow_unfrozen: bool = False,
) -> dict[str, Any]:
    """Synthesize a traceable 3-contract JudgmentPacket and append to isolated shadow SQLite ledger."""
    if not allow_unfrozen:
        fz = verify_judgment_v3_frozen()
        if not fz["ok"]:
            raise ValueError(f"Judgment v3 manifest verification failed: {fz['issues']}")

    is_injected = inputs_bundle is not None and not inputs_bundle.get("_from_live_collector")
    if inputs_bundle is None:
        inputs_bundle = generate_demo_judgment_inputs() if demo_mode else collect_judgment_inputs()
        if not demo_mode:
            inputs_bundle["_from_live_collector"] = True
            is_injected = False
    if supplement_note and str(supplement_note).strip():
        inputs_bundle = dict(inputs_bundle)
        inputs_bundle["agent_or_human_supplement"] = str(supplement_note).strip()

    if inputs_bundle.get("is_synthetic_demo"):
        required_kind = KIND_DEMO_J3
        mode = "SYNTHETIC_DEMO_JUDGMENT_V3"
    elif is_injected:
        required_kind = KIND_INJECTED_J3
        mode = "CALLER_INJECTED_JUDGMENT_V3"
    else:
        required_kind = KIND_LIVE_J3
        mode = "live_shadow_judgment_v3"

    if db_path is None:
        if required_kind == KIND_INJECTED_J3:
            raise JudgmentLedgerKindError("Caller-injected judgment bundles require an explicit non-live db_path.")
        db_path = default_judgment_ledger_path(demo=(required_kind == KIND_DEMO_J3))
    db_path = Path(db_path)

    packet = synthesize_judgment_packet(inputs_bundle, starting_cash_usdt=float(starting_cash))
    cutoff_dt = parse_aware_timestamp(packet["cutoff_utc"]) or datetime.now(timezone.utc)
    lc = packet.get("lifecycle_timestamps") or {}
    gen_iso = str(lc.get("judgment_generated_at_utc") or datetime.now(timezone.utc).isoformat())
    exec_iso = str(lc.get("earliest_executable_at_utc") or packet["cutoff_utc"])
    exec_dt = parse_aware_timestamp(exec_iso) or cutoff_dt
    now_iso = datetime.now(timezone.utc).isoformat()
    cfg = load_judgment_config()

    conn = _init_judgment_db(db_path, ledger_kind=required_kind)
    try:
        resolved_prior = _resolve_pending_outcomes(
            conn,
            inputs_bundle["market_snapshot"],
            now_cutoff=cutoff_dt,
            resolved_at_iso=now_iso,
        )
        archive_json, bundle_sha = _normalize_bundle_for_archive(inputs_bundle)
        existing = conn.execute(
            "SELECT packet_id, judgment_hash FROM judgment_packets WHERE cutoff_utc = ?",
            (packet["cutoff_utc"],),
        ).fetchone()
        if existing is not None:
            conn.execute(
                """
                INSERT OR IGNORE INTO archived_input_bundles(bundle_id, packet_id, cutoff_utc, captured_at_utc, ledger_kind, bundle_sha256, bundle_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    existing["packet_id"],
                    packet["cutoff_utc"],
                    str(inputs_bundle.get("collected_at_utc") or now_iso),
                    required_kind,
                    bundle_sha,
                    archive_json,
                ),
            )
            conn.commit()
            return {
                "status": "already_recorded_for_hour",
                "packet_id": existing["packet_id"],
                "cutoff_utc": packet["cutoff_utc"],
                "judgment_hash": existing["judgment_hash"],
                "archived_bundle_sha256": bundle_sha,
                "outcomes_resolved": resolved_prior,
                "packet": packet,
                "db_path": _display_path(db_path),
                "order_permissions": False,
            }

        packet_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO judgment_packets(packet_id, cutoff_utc, recorded_at_utc, mode, version, judgment_hash, packet_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (packet_id, packet["cutoff_utc"], now_iso, mode, cfg["version"], packet["judgment_hash"], json.dumps(packet, ensure_ascii=False, default=str)),
        )
        conn.execute(
            """
            INSERT INTO archived_input_bundles(bundle_id, packet_id, cutoff_utc, captured_at_utc, ledger_kind, bundle_sha256, bundle_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                packet_id,
                packet["cutoff_utc"],
                str(inputs_bundle.get("collected_at_utc") or now_iso),
                required_kind,
                bundle_sha,
                archive_json,
            ),
        )

        # Store pre-outcome directional arms (`A0_technical_only` and `A3_project_full_judgment_v3`)
        for sym in SYMBOLS:
            last_px = packet["judgments"][sym].get("last_close")
            for horizon in HORIZONS:
                h_hours = int(cfg["horizons"][horizon]["outcome_horizon_hours"])
                tgt_iso = (exec_dt + timedelta(hours=h_hours)).isoformat()
                hd = packet["judgments"][sym][horizon]
                for arm_name, bias_val in (
                    ("A0_technical_only", hd["technical_only"]["bias"]),
                    ("A3_project_full_judgment_v3", hd["final_bias"]),
                ):
                    conn.execute(
                        """
                        INSERT INTO arm_judgments(
                            row_id, packet_id, cutoff_utc, arm_name, symbol, horizon,
                            target_outcome_utc, predicted_bias, technical_bias, entry_price_ref, judgment_hash,
                            judgment_generated_at_utc, earliest_executable_at_utc
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(uuid.uuid4()),
                            packet_id,
                            packet["cutoff_utc"],
                            arm_name,
                            sym,
                            horizon,
                            tgt_iso,
                            bias_val,
                            hd["technical_only"]["bias"],
                            last_px,
                            packet["judgment_hash"],
                            gen_iso,
                            exec_iso,
                        ),
                    )

        # Initialize shadow books on first packet (non-interfering with v2.2.1!)
        for book_name in ("B0_cash", "A0_technical_only", "A3_project_full_judgment_v3", "B_frozen_v2_2_1"):
            has_init = conn.execute(
                "SELECT 1 FROM shadow_paper_events WHERE book_name = ? LIMIT 1", (book_name,)
            ).fetchone()
            if not has_init:
                conn.execute(
                    """
                    INSERT INTO shadow_paper_events(event_id, recorded_at_utc, book_name, event_kind, symbol, payload_json, cash_after_usdt)
                    VALUES (?, ?, ?, 'init_book', NULL, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        now_iso,
                        book_name,
                        json.dumps({"starting_cash": float(starting_cash), "mode": mode}),
                        float(starting_cash),
                    ),
                )
        conn.commit()
    finally:
        conn.close()

    return {
        "status": "recorded_judgment_cycle",
        "packet_id": packet_id,
        "cutoff_utc": packet["cutoff_utc"],
        "judgment_hash": packet["judgment_hash"],
        "archived_bundle_sha256": bundle_sha,
        "outcomes_resolved": resolved_prior,
        "packet": packet,
        "db_path": _display_path(db_path),
        "ledger_kind": required_kind,
        "order_permissions": False,
    }


def log_chat_qa(db_path: Path, qa_result: dict[str, Any]) -> None:
    if not Path(db_path).is_file():
        return
    conn = _init_judgment_db(db_path)
    try:
        conn.execute(
            """
            INSERT INTO chat_qa_logs(qa_id, asked_at_utc, cutoff_utc, judgment_hash, intent, question, answer_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                datetime.now(timezone.utc).isoformat(),
                str(qa_result.get("cutoff_utc") or ""),
                str(qa_result.get("judgment_hash") or ""),
                str(qa_result.get("intent") or "overview"),
                str(qa_result.get("question") or ""),
                json.dumps(qa_result, ensure_ascii=False, default=str),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_latest_two_packets(db_path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    db_path = Path(db_path)
    if not db_path.is_file():
        return None, None
    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT packet_json FROM judgment_packets ORDER BY cutoff_utc DESC LIMIT 2"
        ).fetchall()
    finally:
        conn.close()
    cur = json.loads(rows[0]["packet_json"]) if len(rows) >= 1 else None
    prev = json.loads(rows[1]["packet_json"]) if len(rows) >= 2 else None
    return cur, prev


def get_judgment_ledger_status(db_path: Path | None = None, *, demo: bool = False) -> dict[str, Any]:
    db_path = Path(db_path) if db_path is not None else default_judgment_ledger_path(demo=demo)
    cfg = load_judgment_config()
    if not db_path.is_file():
        return {
            "initialized": False,
            "db_path": _display_path(db_path),
            "packets_recorded": 0,
            "current_version_packets": 0,
            "max_contiguous_hours": 0,
            "missing_hours_in_span": 0,
            "missing_hour_cutoffs": [],
            "latest_cutoff_utc": None,
            "previous_cutoff_utc": None,
            "validation_status": "PENDING_INSUFFICIENT_PROSPECTIVE_SAMPLE",
            "validation_note": "尚无 Judgment v3 前瞻影子记录（需满 168 连续小时且结果成熟后审计）。",
            "arm_outcomes": {},
            "latest_packet": None,
            "order_permissions": False,
        }
    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        meta_row = conn.execute("SELECT value FROM ledger_meta WHERE key='ledger_kind'").fetchone()
        kind = str(meta_row[0]) if meta_row else "unknown"
        pkts = conn.execute("SELECT cutoff_utc, version, judgment_hash, packet_json FROM judgment_packets ORDER BY cutoff_utc DESC").fetchall()
        has_arch = bool(conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='archived_input_bundles'").fetchone())
        arch_count = int(conn.execute("SELECT COUNT(*) FROM archived_input_bundles").fetchone()[0]) if has_arch else 0
        out_rows = conn.execute(
            "SELECT arm_name, COUNT(*) AS n, SUM(CASE WHEN direction_hit=1 THEN 1 ELSE 0 END) AS hits, "
            "SUM(CASE WHEN direction_hit IS NOT NULL THEN 1 ELSE 0 END) AS scored, AVG(realized_return_pct) AS avg_ret "
            "FROM outcome_observations GROUP BY arm_name"
        ).fetchall()
    finally:
        conn.close()

    req = int(cfg["shadow_paper"]["prospective_min_contiguous_hours"])
    cur_ver = str(cfg["version"])
    freeze_start = parse_aware_timestamp(cfg.get("prospective_freeze_started_utc"))
    cur_ver_cutoffs = [
        r["cutoff_utc"]
        for r in pkts
        if r["version"] == cur_ver and (freeze_start is None or (parse_aware_timestamp(r["cutoff_utc"]) or freeze_start) >= freeze_start)
    ]
    all_stats = _compute_contiguous_stats([r["cutoff_utc"] for r in pkts])
    cur_stats = _compute_contiguous_stats(cur_ver_cutoffs)
    max_contig = cur_stats["max_contiguous_hours"] if cur_ver_cutoffs else all_stats["max_contiguous_hours"]

    is_live = kind == KIND_LIVE_J3
    v_status = (
        "NOT_A_PROSPECTIVE_LEDGER"
        if not is_live
        else ("PENDING_AUDIT_CONTIGUOUS_WINDOW_REACHED" if max_contig >= req else "PENDING_INSUFFICIENT_PROSPECTIVE_SAMPLE")
    )
    return {
        "initialized": True,
        "db_path": _display_path(db_path),
        "ledger_kind": kind,
        "packets_recorded": len(pkts),
        "archived_bundles_count": arch_count,
        "current_version_packets": len(cur_ver_cutoffs),
        "max_contiguous_hours": max_contig,
        "all_packets_max_contiguous_hours": all_stats["max_contiguous_hours"],
        "missing_hours_in_span": all_stats["missing_hours_in_span"],
        "missing_hour_cutoffs": all_stats["missing_hour_cutoffs"],
        "latest_cutoff_utc": pkts[0]["cutoff_utc"] if pkts else None,
        "previous_cutoff_utc": pkts[1]["cutoff_utc"] if len(pkts) >= 2 else None,
        "validation_status": v_status,
        "validation_note": (
            f"已记录快照 {len(pkts)} 条，当前版本连续 {max_contig} 小时（区间缺失 {all_stats['missing_hours_in_span']} 小时）；达到 {req} 连续小时前保持待验证。"
            if is_live and max_contig < req
            else ("非真实前瞻影子账本。" if not is_live else f"当前版本连续 {max_contig} 小时已达标，进入待审计阶段。")
        ),
        "arm_outcomes": {
            r["arm_name"]: {
                "resolved_count": int(r["n"]),
                "directional_scored": int(r["scored"] or 0),
                "directional_hits": int(r["hits"] or 0),
                "avg_realized_return_pct": round(float(r["avg_ret"]), 4) if r["avg_ret"] is not None else None,
            }
            for r in out_rows
        },
        "latest_packet": json.loads(pkts[0]["packet_json"]) if pkts else None,
        "order_permissions": False,
    }


def _strip_json_fences(raw_text: str) -> str:
    txt = (raw_text or "").strip()
    if txt.startswith("```"):
        lines = txt.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        txt = "\n".join(lines).strip()
    return txt


def _call_official_ai_for_comparison(arm_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Call user-configured official chat API strictly without fabricating or mutating payloads."""
    if not model_configured():
        return {"status": "NOT_RUN_NEEDS_MODEL_CONFIG", "decisions": {}}
    provider = (model_provider() or "").lower()
    base = model_base_url() or ""
    name = model_name() or ""
    from piia.intelligence.agents.providers import parse_official_response
    from piia.intelligence.settings import MODEL_KEY

    key = os.environ.get(MODEL_KEY, "")
    system = (
        "You are evaluating BTCUSDT, XAUUSDT, and NDX100USDT strictly from the supplied snapshot (<= cutoff_utc). "
        "Return ONLY valid JSON mapping each symbol to {\"intraday\": \"bullish_bias|bearish_bias|neutral_stand_aside\", "
        "\"swing_1_3d\": \"bullish_bias|bearish_bias|neutral_stand_aside\", \"reason\": \"...\"}. Never fabricate future facts."
    )
    user_msg = json.dumps(payload, ensure_ascii=False, default=str)
    try:
        if provider in {"openai", "xai"}:
            url = f"{base.rstrip('/')}/v1/chat/completions"
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            body = {
                "model": name,
                "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
            }
        elif provider == "anthropic":
            url = f"{base.rstrip('/')}/v1/messages"
            headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
            body = {
                "model": name,
                "max_tokens": 2048,
                "temperature": 0,
                "system": system,
                "messages": [{"role": "user", "content": user_msg}],
            }
        else:
            return {"status": f"UNSUPPORTED_PROVIDER:{provider}", "decisions": {}}
        with httpx.Client(timeout=25.0) as client:
            resp = client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        text, usage = parse_official_response(provider, resp.json())
        parsed = json.loads(_strip_json_fences(text))
        return {"status": "completed", "arm": arm_name, "provider": provider, "model": name, "usage": usage, "decisions": parsed}
    except Exception as exc:
        return {"status": f"ERROR:{type(exc).__name__}", "error_detail": str(exc)[:200], "decisions": {}}


def _summarize_decisions(decisions: dict[str, dict[str, str]]) -> dict[str, Any]:
    total = 0
    counts = {"bullish_bias": 0, "bearish_bias": 0, "neutral_stand_aside": 0}
    for sym in SYMBOLS:
        for h in HORIZONS:
            total += 1
            b = ((decisions.get(sym) or {}).get(h)) or "neutral_stand_aside"
            if b not in counts:
                b = "neutral_stand_aside"
            counts[b] += 1
    non_stand = counts["bullish_bias"] + counts["bearish_bias"]
    return {
        "counts": counts,
        "directional_coverage_pct": round((non_stand / total) * 100.0, 2) if total else 0.0,
        "stand_aside_pct": round((counts["neutral_stand_aside"] / total) * 100.0, 2) if total else 0.0,
    }


def _bar_open_dt_from_row(row: dict[str, Any]) -> datetime | None:
    raw = row.get("open_at_utc") or row.get("timestamp_utc") or row.get("ts") or row.get("date")
    dt = parse_aware_timestamp(raw)
    if dt is not None:
        return dt.astimezone(timezone.utc)
    if isinstance(raw, (int, float)):
        v = float(raw)
        return datetime.fromtimestamp(v / 1000.0 if v > 1e11 else v, tz=timezone.utc)
    return None


def _bar_close_dt_from_row(row: dict[str, Any]) -> datetime | None:
    c_dt = parse_aware_timestamp(row.get("close_at_utc") or row.get("bar_close_utc"))
    if c_dt is not None:
        return c_dt.astimezone(timezone.utc)
    o_dt = _bar_open_dt_from_row(row)
    return (o_dt + timedelta(hours=1)) if o_dt is not None else None


def run_four_way_comparison(
    inputs_bundle: dict[str, Any],
    *,
    ai_provider: JudgmentAIProvider | None = None,
    call_live_model: bool | None = None,
    starting_cash_usdt: float | None = None,
) -> dict[str, Any]:
    """Compare A0 Technical-Only, A1 Direct AI (Sanitized Candles), A2 AI Freeform (Same Packet), A3 Project Rules."""
    packet = synthesize_judgment_packet(inputs_bundle, starting_cash_usdt=starting_cash_usdt)
    cutoff_utc = packet["cutoff_utc"]
    lc = packet.get("lifecycle_timestamps") or {}
    bar_cutoff_dt = (
        parse_aware_timestamp(lc.get("bar_cutoff_utc"))
        or parse_aware_timestamp(cutoff_utc)
        or datetime.now(timezone.utc)
    )
    decision_cutoff_utc = str(lc.get("decision_cutoff_utc") or cutoff_utc)

    a0_dec = {
        sym: {h: packet["judgments"][sym][h]["technical_only"]["bias"] for h in HORIZONS}
        for sym in SYMBOLS
    }
    a3_dec = {
        sym: {h: packet["judgments"][sym][h]["final_bias"] for h in HORIZONS}
        for sym in SYMBOLS
    }

    # A1 Direct AI receives ONLY sanitized closed bars <= bar_cutoff_utc (deduped, sorted, window 48) + task
    inst_raw = (inputs_bundle.get("market_snapshot") or {}).get("instruments") or {}
    a1_instruments: dict[str, Any] = {}
    a1_sanitization_audit: dict[str, Any] = {}
    for sym in SYMBOLS:
        raw_c = ((inst_raw.get(sym) or {}).get("candles_closed_1h")) or []
        san = sanitize_closed_candles(raw_c, bar_cutoff_utc=bar_cutoff_dt, window=48)
        san_bars = san["candles"]
        san_last_close = float(san_bars[-1]["close"]) if san_bars else None
        a1_instruments[sym] = {
            "last_close": san_last_close,
            "recent_closed_1h_tail": san_bars,
        }
        a1_sanitization_audit[sym] = san["audit"]

    candles_only_payload = {
        "cutoff_utc": bar_cutoff_dt.isoformat(),
        "task": "Assess intraday (4-12h) and 1-3 trading day bias (bullish_bias / bearish_bias / neutral_stand_aside) for BTCUSDT, XAUUSDT, NDX100USDT.",
        "instruments": a1_instruments,
    }
    # A2 Freeform AI receives the EXACT SAME frozen EvidenceAtom list (no future/stale items, no deterministic verdicts)
    same_packet_payload = {
        "cutoff_utc": decision_cutoff_utc,
        "task": candles_only_payload["task"],
        "tide_raw": packet["tide"],
        "evidence_atoms": [
            {k: v for k, v in a.items() if k != "role_by_symbol_horizon"}
            for a in packet["evidence_atoms"]
            if a.get("freshness_status") in {"fresh", "lagged_usable", "missing"}
        ],
    }

    if call_live_model is None:
        auto_call = (
            model_configured()
            and os.environ.get("PIIA_SKIP_NETWORK") != "1"
            and not bool(inputs_bundle.get("is_synthetic_demo"))
        )
    else:
        auto_call = bool(call_live_model)

    def _run_ai(arm: str, pld: dict[str, Any]) -> dict[str, Any]:
        if ai_provider is not None:
            res = ai_provider.compare(arm, pld)
            return {"status": res.get("status", "completed"), "decisions": res.get("decisions") or {}, "payload_keys": sorted(pld)}
        if auto_call:
            out = _call_official_ai_for_comparison(arm, pld)
            out["payload_keys"] = sorted(pld)
            return out
        return {"status": "NOT_RUN_NEEDS_MODEL_CONFIG", "decisions": {}, "payload_keys": sorted(pld)}

    a1_res = _run_ai("A1_direct_ai_candles_only", candles_only_payload)
    a2_res = _run_ai("A2_ai_freeform_same_packet", same_packet_payload)

    def _norm_dec(raw_d: dict[str, Any]) -> dict[str, dict[str, str]]:
        return {
            sym: {
                h: str(((raw_d.get(sym) or {}).get(h)) or "neutral_stand_aside")
                for h in HORIZONS
            }
            for sym in SYMBOLS
        }

    arms: dict[str, Any] = {
        "A0_technical_only": {"status": "executed", "decisions": a0_dec, **_summarize_decisions(a0_dec)},
        "A1_direct_ai_candles_only": (
            {"status": a1_res["status"], "payload_keys": a1_res["payload_keys"], "decisions": _norm_dec(a1_res["decisions"]), **_summarize_decisions(_norm_dec(a1_res["decisions"]))}
            if a1_res["status"] == "completed"
            else {"status": a1_res["status"], "payload_keys": a1_res["payload_keys"], "decisions": {}}
        ),
        "A2_ai_freeform_same_packet": (
            {"status": a2_res["status"], "payload_keys": a2_res["payload_keys"], "decisions": _norm_dec(a2_res["decisions"]), **_summarize_decisions(_norm_dec(a2_res["decisions"]))}
            if a2_res["status"] == "completed"
            else {"status": a2_res["status"], "payload_keys": a2_res["payload_keys"], "decisions": {}}
        ),
        "A3_project_full_judgment_v3": {"status": "executed", "decisions": a3_dec, **_summarize_decisions(a3_dec)},
    }

    diff_a3_vs_a0 = [
        f"{sym}:{h}({a0_dec[sym][h]}->{a3_dec[sym][h]})"
        for sym in SYMBOLS for h in HORIZONS
        if a0_dec[sym][h] != a3_dec[sym][h]
    ]
    shared_model_cfg = {
        "provider": model_provider(),
        "model": model_name(),
        "configured": bool(model_configured()),
    }
    input_audit = {
        "bar_cutoff_utc": bar_cutoff_dt.isoformat(),
        "decision_cutoff_utc": decision_cutoff_utc,
        "starting_cash_usdt": packet.get("starting_cash_usdt"),
        "coherence_sanitization_audit": (packet.get("coherence") or {}).get("sanitization_audit", {}),
        "arms": {
            "A0_technical_only": {
                "source": "sanitized_closed_1h_4h_1d_bars_le_bar_cutoff",
                "sanitization_audit": (packet.get("coherence") or {}).get("sanitization_audit", {}),
            },
            "A1_direct_ai_candles_only": {
                "source": "sanitized_closed_1h_tail_48_le_bar_cutoff_no_project_domains",
                "sanitization_audit": a1_sanitization_audit,
                "payload_sha256": canonical_sha256(candles_only_payload),
                "model_config": shared_model_cfg,
            },
            "A2_ai_freeform_same_packet": {
                "source": "frozen_packet_evidence_atoms_excluding_future_stale_and_verdicts",
                "atoms_passed": len(same_packet_payload["evidence_atoms"]),
                "payload_sha256": canonical_sha256(same_packet_payload),
                "model_config": shared_model_cfg,
            },
            "A3_project_full_judgment_v3": {
                "source": "frozen_packet_deterministic_synthesis_plus_v3_paper_gate",
                "sanitization_audit": (packet.get("coherence") or {}).get("sanitization_audit", {}),
            },
        },
    }
    return {
        "cutoff_utc": cutoff_utc,
        "judgment_hash": packet["judgment_hash"],
        "starting_cash_usdt": packet.get("starting_cash_usdt"),
        "arms": arms,
        "input_audit": input_audit,
        "rule_changes_vs_technical": diff_a3_vs_a0,
        "notes": [
            "A2 vs A1 isolates external data coverage contribution; A3 vs A2 isolates deterministic causal deduplication and hard-gate rules.",
            "Directional hit rates remain PENDING_OUTCOME until future closed horizon bars >= earliest_executable_at_utc + horizon are observed post-cutoff.",
        ],
    }


def _funding_events_between(
    funding_map: dict[int, float] | None,
    start_dt: datetime,
    end_dt: datetime,
) -> tuple[bool, list[tuple[int, float]]]:
    """Enumerate 8h funding boundaries (00:00, 08:00, 16:00 UTC) strictly in (start_dt, end_dt]."""
    s_utc = start_dt.astimezone(timezone.utc)
    e_utc = end_dt.astimezone(timezone.utc)
    if e_utc <= s_utc:
        return True, []
    cur = s_utc.replace(minute=0, second=0, microsecond=0)
    cur = cur.replace(hour=(cur.hour // 8) * 8)
    boundaries: list[int] = []
    while cur <= e_utc:
        if cur > s_utc:
            boundaries.append(int(cur.timestamp() * 1000))
        cur += timedelta(hours=8)
    if not boundaries:
        return True, []
    if not funding_map:
        return False, []
    matched: list[tuple[int, float]] = []
    for ts_ms in boundaries:
        rate = funding_map.get(ts_ms)
        if rate is None:
            return False, matched
        matched.append((ts_ms, float(rate)))
    return True, matched


def _compute_sequence_directional_accuracy(
    packets_and_bars: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    arm_selector: str,
    custom_decisions_seq: list[dict[str, dict[str, str]]] | None = None,
    price_history_bars: dict[str, list[dict[str, Any]]] | None = None,
    horizon_bar_end_utc: datetime | None = None,
) -> dict[str, Any]:
    """Score intraday (6h) and swing_1_3d (48h) bias strictly on mature post-availability bars."""
    cfg = load_judgment_config()
    bar_close_index: dict[str, dict[datetime, float]] = {sym: {} for sym in SYMBOLS}
    bar_open_index: dict[str, dict[datetime, float]] = {sym: {} for sym in SYMBOLS}
    sorted_closes: dict[str, list[datetime]] = {sym: [] for sym in SYMBOLS}
    sorted_opens: dict[str, list[datetime]] = {sym: [] for sym in SYMBOLS}
    if price_history_bars is not None:
        for sym in SYMBOLS:
            for row in price_history_bars.get(sym) or []:
                o_dt = _bar_open_dt_from_row(row)
                c_dt = _bar_close_dt_from_row(row)
                if horizon_bar_end_utc is not None and c_dt is not None and c_dt > horizon_bar_end_utc:
                    continue
                if o_dt is not None and row.get("open") is not None:
                    bar_open_index[sym][o_dt] = float(row["open"])
                if c_dt is not None and row.get("close") is not None:
                    bar_close_index[sym][c_dt] = float(row["close"])
    else:
        for _pkt, mkt in packets_and_bars:
            for sym in SYMBOLS:
                for row in ((mkt.get("instruments") or {}).get(sym) or {}).get("candles_closed_1h") or []:
                    o_dt = _bar_open_dt_from_row(row)
                    c_dt = _bar_close_dt_from_row(row)
                    if o_dt is not None and row.get("open") is not None:
                        bar_open_index[sym][o_dt] = float(row["open"])
                    if c_dt is not None and row.get("close") is not None:
                        bar_close_index[sym][c_dt] = float(row["close"])
    for sym in SYMBOLS:
        sorted_closes[sym] = sorted(bar_close_index[sym])
        sorted_opens[sym] = sorted(bar_open_index[sym])

    out: dict[str, Any] = {}
    for horizon in HORIZONS:
        h_hours = int(cfg["horizons"][horizon]["outcome_horizon_hours"])
        mature = 0
        hits = 0
        directional_mature = 0
        pending_immature = 0
        for idx, (pkt, _mkt) in enumerate(packets_and_bars):
            lc = pkt.get("lifecycle_timestamps") or {}
            exec_dt = (
                parse_aware_timestamp(lc.get("earliest_executable_at_utc"))
                or parse_aware_timestamp(pkt["cutoff_utc"])
            )
            if exec_dt is None:
                continue
            tgt_dt = exec_dt + timedelta(hours=h_hours)
            for sym in SYMBOLS:
                if arm_selector == "custom_decisions" and custom_decisions_seq is not None and idx < len(custom_decisions_seq):
                    bias = ((custom_decisions_seq[idx].get(sym) or {}).get(horizon)) or "neutral_stand_aside"
                elif arm_selector == "tech":
                    bias = pkt["judgments"][sym][horizon]["technical_only"]["bias"]
                else:
                    bias = pkt["judgments"][sym][horizon]["final_bias"]
                e_candidates = [t for t in sorted_opens[sym] if t >= exec_dt]
                x_candidates = [t for t in sorted_closes[sym] if t >= tgt_dt]
                if not e_candidates or not x_candidates:
                    pending_immature += 1
                    continue
                entry_px = bar_open_index[sym].get(exec_dt, bar_open_index[sym][e_candidates[0]])
                exit_px = bar_close_index[sym][x_candidates[0]]
                mature += 1
                if bias not in {"bullish_bias", "bearish_bias"} or entry_px <= 0:
                    continue
                directional_mature += 1
                ret = (exit_px / entry_px) - 1.0
                if (bias == "bullish_bias" and ret > 0) or (bias == "bearish_bias" and ret < 0):
                    hits += 1
        out[horizon] = {
            "horizon_hours": h_hours,
            "mature_pairs": mature,
            "directional_mature_calls": directional_mature,
            "hits": hits,
            "directional_hit_rate_pct": round((hits / directional_mature) * 100.0, 2) if directional_mature else None,
            "pending_immature_pairs": pending_immature,
        }
    return out


def _simulate_shadow_paper_sequence(
    packets_and_bars: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    arm_selector: str,
    starting_cash: float = 100.0,
    slippage_mult: float = 1.0,
    funding_by_symbol: dict[str, list[dict[str, Any]] | dict[int, float]] | None = None,
    respect_pending_review: bool = True,
    custom_decisions_seq: list[dict[str, dict[str, str]]] | None = None,
) -> dict[str, Any]:
    """Run causal post-availability isolated paper accounting on multi-bar replay frames."""
    cfg = load_judgment_config()["shadow_paper"]
    cash = float(starting_cash)
    peak_eq = cash
    max_dd_pct = 0.0
    turnover = 0.0
    closed_trades = 0
    open_bars = 0
    total_bars = len(packets_and_bars)
    pending_review_skips = 0
    min_qty_budget_rejections = 0
    orders_queued = 0
    orders_filled = 0
    open_pos: dict[str, Any] | None = None
    pending_order: dict[str, Any] | None = None
    slip = float(cfg["slippage_base"]) * slippage_mult

    # Normalize funding lookup map per symbol if provided
    funding_maps: dict[str, dict[int, float]] | None = None
    if funding_by_symbol is not None:
        funding_maps = {}
        for sym in SYMBOLS:
            raw_f = funding_by_symbol.get(sym)
            if isinstance(raw_f, dict):
                funding_maps[sym] = {int(k): float(v) for k, v in raw_f.items()}
            elif isinstance(raw_f, list):
                fmap: dict[int, float] = {}
                for item in raw_f:
                    if not isinstance(item, dict):
                        continue
                    ts_raw = item.get("timestamp") if item.get("timestamp") is not None else item.get("timestamp_utc")
                    rate_raw = item.get("rate")
                    if ts_raw is None or rate_raw is None:
                        continue
                    if isinstance(ts_raw, (int, float)):
                        ts_ms = int(ts_raw * 1000) if ts_raw < 1e11 else int(ts_raw)
                    else:
                        dt = parse_aware_timestamp(ts_raw)
                        if dt is None:
                            continue
                        ts_ms = int(dt.timestamp() * 1000)
                    fmap[ts_ms] = float(rate_raw)
                funding_maps[sym] = fmap
            else:
                funding_maps[sym] = {}

    funding_all_verified = True if funding_maps is not None else False
    verified_funding_usdt = 0.0
    funding_events_applied = 0

    def _bias_for(idx: int, pkt: dict[str, Any], sym: str) -> str:
        if arm_selector == "custom_decisions" and custom_decisions_seq is not None and idx < len(custom_decisions_seq):
            b = ((custom_decisions_seq[idx].get(sym) or {}).get("swing_1_3d")) or "neutral_stand_aside"
            return b
        if arm_selector == "v2_official":
            s = pkt["judgments"][sym].get("official_v2_2_1_signal")
            return "bullish_bias" if s == "long" else ("bearish_bias" if s == "short" else "neutral_stand_aside")
        hd = pkt["judgments"][sym]["swing_1_3d"]
        return hd["final_bias"] if arm_selector == "full" else hd["technical_only"]["bias"]

    cur_eq = cash
    for i, (pkt, mkt_snap) in enumerate(packets_and_bars):
        lc = pkt.get("lifecycle_timestamps") or {}
        cutoff_dt = parse_aware_timestamp(lc.get("bar_cutoff_utc")) or parse_aware_timestamp(pkt["cutoff_utc"]) or datetime.now(timezone.utc)
        earliest_exec_dt = parse_aware_timestamp(lc.get("earliest_executable_at_utc")) or cutoff_dt

        # 1. Fill queued post-availability order ONLY on a bar whose bar_open >= earliest_executable_at_utc
        if open_pos is None and pending_order is not None:
            sym = pending_order["symbol"]
            c_rows = ((mkt_snap.get("instruments") or {}).get(sym) or {}).get("candles_closed_1h") or []
            if c_rows:
                bar = c_rows[-1]
                bar_open_dt = _bar_open_dt_from_row(bar)
                bar_close_dt = _bar_close_dt_from_row(bar) or cutoff_dt
                if bar_open_dt is not None and bar_open_dt >= pending_order["earliest_exec_dt"]:
                    side = pending_order["side"]
                    atr_v = pending_order["atr14"]
                    rk = evaluate_programmatic_risk(
                        symbol=sym,
                        synth_out={"candidate_direction": side, "setup_type": "judgment_v3_shadow", "proposed_stop_atr_mult": 2.0},
                        market_pillar={"ohlc": {"close": float(bar["open"])}, "atr14": atr_v, "funding_rate_8h": None},
                        equity_usdt=cash,
                        free_cash_usdt=cash,
                        open_positions_count=0,
                        kill_switch_halted=(cash / starting_cash <= 0.90),
                        planned_entry_price=float(bar["open"]),
                        slippage_override=slip,
                    )
                    pending_order = None
                    if rk["approved_by_risk_gate"]:
                        sz = rk["position_sizing"]
                        ep = rk["entry_plan"]
                        xp = rk["exit_plan"]
                        cr = rk["cost_and_risk"]
                        fee = float(cr["entry_taker_fee_usdt"])
                        cash -= fee
                        turnover += float(sz["notional_usdt"])
                        orders_filled += 1
                        open_pos = {
                            "symbol": sym,
                            "side": side,
                            "qty": float(sz["qty_contracts"]),
                            "notional": float(sz["notional_usdt"]),
                            "entry": float(ep["assumed_fill_with_slippage"]),
                            "stop": float(xp["initial_stop_loss"]),
                            "entry_fee": fee,
                            "opened_dt": bar_open_dt,
                            "last_funding_dt": bar_open_dt,
                        }
                    elif "minimum_order_exceeds_budget" in str(rk.get("stand_aside_reason") or ""):
                        min_qty_budget_rejections += 1

        # 2. Mark-to-market / check stops / accrue funding on open position strictly on bars after fill
        cur_eq = cash
        if open_pos is not None:
            sym = open_pos["symbol"]
            c_rows = ((mkt_snap.get("instruments") or {}).get(sym) or {}).get("candles_closed_1h") or []
            if c_rows:
                bar = c_rows[-1]
                bar_open_dt = _bar_open_dt_from_row(bar)
                bar_close_dt = _bar_close_dt_from_row(bar) or cutoff_dt
                spec = resolve_spec(sym).to_v1_spec()
                side = open_pos["side"]
                # Only count bars at/after opened_dt
                if bar_open_dt is None or bar_open_dt >= open_pos["opened_dt"]:
                    open_bars += 1
                    f_ok, f_evts = _funding_events_between(
                        funding_maps.get(sym) if funding_maps is not None else None,
                        open_pos["last_funding_dt"],
                        bar_close_dt,
                    )
                    if not f_ok:
                        funding_all_verified = False
                    else:
                        sign_f = 1.0 if side == "long" else -1.0
                        for _ts_ms, rate in f_evts:
                            cf = -open_pos["notional"] * rate * sign_f
                            verified_funding_usdt += cf
                            funding_events_applied += 1
                        open_pos["last_funding_dt"] = bar_close_dt

                    hi, lo, cl = float(bar["high"]), float(bar["low"]), float(bar["close"])
                    side_int = 1 if side == "long" else -1
                    hit_sl = (side == "long" and lo <= open_pos["stop"]) or (side == "short" and hi >= open_pos["stop"])
                    if hit_sl:
                        base_stop_px = stop_fill(side_int, open_pos["stop"], float(bar["open"]), lo, hi)
                        raw_exit = float(base_stop_px if base_stop_px is not None else open_pos["stop"]) * (1.0 - slip if side == "long" else 1.0 + slip)
                        gross = exit_pnl(side_int, open_pos["qty"], open_pos["entry"], raw_exit, spec)
                        exit_fee = open_pos["qty"] * raw_exit * spec.contract_size * spec.taker
                        cash += (gross - exit_fee)
                        turnover += open_pos["qty"] * raw_exit * spec.contract_size
                        closed_trades += 1
                        open_pos = None
                        cur_eq = cash
                    else:
                        cur_bias = _bias_for(i, pkt, sym)
                        if cur_bias == ("bearish_bias" if side == "long" else "bullish_bias") and bar_open_dt is not None and bar_open_dt > open_pos["opened_dt"]:
                            raw_exit = cl * (1.0 - slip if side == "long" else 1.0 + slip)
                            gross = exit_pnl(side_int, open_pos["qty"], open_pos["entry"], raw_exit, spec)
                            exit_fee = open_pos["qty"] * raw_exit * spec.contract_size * spec.taker
                            cash += (gross - exit_fee)
                            turnover += open_pos["qty"] * raw_exit * spec.contract_size
                            closed_trades += 1
                            open_pos = None
                            cur_eq = cash
                        else:
                            unreal = (cl - open_pos["entry"]) * open_pos["qty"] * spec.contract_size * (1.0 if side == "long" else -1.0)
                            cur_eq = cash + unreal

        peak_eq = max(peak_eq, cur_eq)
        if peak_eq > 0:
            max_dd_pct = max(max_dd_pct, (peak_eq - cur_eq) / peak_eq * 100.0)

        # 3. Queue new entry order cleanly for future bar >= earliest_executable_at_utc
        if open_pos is None and pending_order is None and i + 1 < total_bars:
            for sym in SYMBOLS:
                bias = _bias_for(i, pkt, sym)
                if bias not in {"bullish_bias", "bearish_bias"}:
                    continue
                if respect_pending_review and arm_selector in {"full", "v2_official"}:
                    pe_st = (pkt["judgments"][sym].get("paper_eligibility") or {}).get("status")
                    if pe_st == "pending_review":
                        pending_review_skips += 1
                        continue
                c_rows = ((mkt_snap.get("instruments") or {}).get(sym) or {}).get("candles_closed_1h") or []
                if len(c_rows) < 20:
                    continue
                atr_v = sum(abs(float(x["high"]) - float(x["low"])) for x in c_rows[-14:]) / 14.0
                pending_order = {
                    "symbol": sym,
                    "side": "long" if bias == "bullish_bias" else "short",
                    "earliest_exec_dt": earliest_exec_dt,
                    "atr14": atr_v,
                }
                orders_queued += 1
                break

    price_fee_ret = round((cur_eq / starting_cash - 1.0) * 100.0, 4) if starting_cash > 0 else 0.0
    realized_ret = round((cash / starting_cash - 1.0) * 100.0, 4) if starting_cash > 0 else 0.0
    if funding_maps is not None and funding_all_verified and (closed_trades > 0 or open_bars > 0):
        net_with_funding = round(((cur_eq + verified_funding_usdt) / starting_cash - 1.0) * 100.0, 4)
        funding_status = "verified_from_historical_schedule"
    elif funding_maps is not None and funding_all_verified and closed_trades == 0 and open_bars == 0:
        net_with_funding = price_fee_ret
        funding_status = "verified_zero_exposure"
    else:
        net_with_funding = None
        funding_status = "unknown_in_shadow_ablation" if funding_maps is None else "partial_or_unknown_funding"

    return {
        "starting_cash_usdt": float(starting_cash),
        "ending_equity_price_and_fee_usdt": round(cur_eq, 4),
        "price_and_fee_return_pct": price_fee_ret,
        "realized_price_and_fee_return_pct": realized_ret,
        "verified_funding_usdt": round(verified_funding_usdt, 4) if funding_maps is not None and funding_all_verified else None,
        "funding_events_applied": funding_events_applied,
        "net_return_pct_with_funding": net_with_funding,
        "funding_status": funding_status,
        "max_drawdown_pct": round(max_dd_pct, 4),
        "exposure_pct": round((open_bars / total_bars) * 100.0, 2) if total_bars else 0.0,
        "turnover_usdt": round(turnover, 2),
        "closed_trades": closed_trades,
        "open_position_at_end": open_pos is not None,
        "orders_queued": orders_queued,
        "orders_filled_post_availability": orders_filled,
        "pending_review_skips": pending_review_skips,
        "min_qty_budget_rejections": min_qty_budget_rejections,
    }


def evaluate_portfolio_risk_audit(
    candles_by_symbol: dict[str, list[dict[str, Any]]],
    proposed_positions: list[dict[str, Any]] | None = None,
    *,
    equity_usdt: float = 100.0,
) -> dict[str, Any]:
    """Audit portfolio-level correlation, concentration, gross leverage, and liquidity separately from isolated simulations."""
    cfg = load_judgment_config().get("portfolio_audit") or {}
    high_corr_min = float(cfg.get("portfolio_high_corr_min", 0.50))
    max_gross_lev = float(cfg.get("portfolio_max_gross_leverage", 2.0))

    ret_series: dict[str, dict[datetime, float]] = {}
    for sym in SYMBOLS:
        rows = candles_by_symbol.get(sym) or []
        prev_c: float | None = None
        s_map: dict[datetime, float] = {}
        for r in rows:
            dt = _bar_close_dt_from_row(r)
            cl = r.get("close")
            if dt is None or cl is None:
                continue
            cf = float(cl)
            if prev_c is not None and prev_c > 0 and cf > 0:
                s_map[dt] = (cf / prev_c) - 1.0
            prev_c = cf
        ret_series[sym] = s_map

    pairwise: dict[str, dict[str, Any]] = {}
    high_corr_pairs: list[str] = []
    pairs = [("BTCUSDT", "NDX100USDT"), ("BTCUSDT", "XAUUSDT"), ("XAUUSDT", "NDX100USDT")]
    for a, b in pairs:
        common = sorted(set(ret_series.get(a, {})) & set(ret_series.get(b, {})))
        key = f"{a}_vs_{b}"
        if len(common) < 24:
            pairwise[key] = {"n_common_bars": len(common), "pearson_r": None, "high_correlation": None}
            continue
        xs = [ret_series[a][t] for t in common]
        ys = [ret_series[b][t] for t in common]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        denom = math.sqrt(vx * vy)
        r_val = (cov / denom) if denom > 0 else 0.0
        is_hi = abs(r_val) >= high_corr_min
        if is_hi:
            high_corr_pairs.append(key)
        pairwise[key] = {
            "n_common_bars": len(common),
            "pearson_r": round(r_val, 4),
            "high_correlation": is_hi,
        }

    props = proposed_positions or []
    gross_notional = sum(float(p.get("notional_usdt") or 0.0) for p in props)
    long_notional = sum(float(p.get("notional_usdt") or 0.0) for p in props if p.get("side") == "long")
    short_notional = sum(float(p.get("notional_usdt") or 0.0) for p in props if p.get("side") == "short")
    gross_lev = (gross_notional / equity_usdt) if equity_usdt > 0 else 0.0
    max_same_dir_lev = max(long_notional, short_notional) / equity_usdt if equity_usdt > 0 else 0.0

    warnings: list[str] = []
    if high_corr_pairs:
        warnings.append(f"high_pairwise_correlation:{','.join(high_corr_pairs)}")
    if gross_lev > max_gross_lev:
        warnings.append(f"gross_leverage_exceeds_cap:{gross_lev:.2f}>{max_gross_lev:.2f}")
    if len(props) >= 2 and max(long_notional, short_notional) == gross_notional:
        warnings.append("same_direction_concentration_all_proposed_positions")

    return {
        "equity_reference_usdt": float(equity_usdt),
        "single_position_sim_proves_portfolio_safety": False,
        "pairwise_return_correlations": pairwise,
        "high_correlation_threshold": high_corr_min,
        "high_correlation_pairs": high_corr_pairs,
        "proposed_positions_count": len(props),
        "gross_notional_usdt": round(gross_notional, 2),
        "gross_leverage": round(gross_lev, 4),
        "same_direction_max_leverage": round(max_same_dir_lev, 4),
        "max_gross_leverage_cap": max_gross_lev,
        "liquidity_and_slippage_note": "Bitget perpetual orderbook depth checked only at single-order entry; portfolio simultaneous liquidation/cascade slippage not assumed away.",
        "warnings": warnings,
        "portfolio_risk_ok": gross_lev <= max_gross_lev,
    }


def run_domain_ablation(
    inputs_bundle: dict[str, Any],
    *,
    replay_sequence: list[dict[str, Any]] | None = None,
    starting_cash: float = 100.0,
    funding_by_symbol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Leave-one-out ablation across Macro, Tide, Sentiment/Flow, News, Forecast + Technical-Only."""
    arm_configs = {
        "Full": {"enable_macro": True, "enable_tide": True, "enable_sentiment_flow": True, "enable_news": True, "enable_forecast": True},
        "No_Macro": {"enable_macro": False, "enable_tide": True, "enable_sentiment_flow": True, "enable_news": True, "enable_forecast": True},
        "No_Tide": {"enable_macro": True, "enable_tide": False, "enable_sentiment_flow": True, "enable_news": True, "enable_forecast": True},
        "No_Sentiment_Flow": {"enable_macro": True, "enable_tide": True, "enable_sentiment_flow": False, "enable_news": True, "enable_forecast": True},
        "No_News": {"enable_macro": True, "enable_tide": True, "enable_sentiment_flow": True, "enable_news": False, "enable_forecast": True},
        "No_Forecast": {"enable_macro": True, "enable_tide": True, "enable_sentiment_flow": True, "enable_news": True, "enable_forecast": False},
        "Technical_Only": {"enable_macro": False, "enable_tide": False, "enable_sentiment_flow": False, "enable_news": False, "enable_forecast": False},
    }

    bundles = replay_sequence if replay_sequence else [inputs_bundle]
    full_packets = [
        synthesize_judgment_packet(b, ablation_switches=arm_configs["Full"], include_v2_control=True, starting_cash_usdt=float(starting_cash))
        for b in bundles
    ]

    results: dict[str, Any] = {}
    for arm_name, sw in arm_configs.items():
        pkts = full_packets if arm_name == "Full" else [
            synthesize_judgment_packet(b, ablation_switches=sw, include_v2_control=False, starting_cash_usdt=float(starting_cash))
            for b in bundles
        ]
        counts = {"bullish_bias": 0, "bearish_bias": 0, "neutral_stand_aside": 0}
        changed_vs_full: list[str] = []
        changed_vs_tech: list[str] = []
        for idx, pkt in enumerate(pkts):
            fp = full_packets[idx]
            for sym in SYMBOLS:
                for h in HORIZONS:
                    b = pkt["judgments"][sym][h]["final_bias"]
                    tb = pkt["judgments"][sym][h]["technical_only"]["bias"]
                    fb = fp["judgments"][sym][h]["final_bias"]
                    counts[b] = counts.get(b, 0) + 1
                    if b != fb:
                        changed_vs_full.append(f"{pkt['cutoff_utc']}:{sym}:{h}({fb}->{b})")
                    if b != tb:
                        changed_vs_tech.append(f"{pkt['cutoff_utc']}:{sym}:{h}({tb}->{b})")
        total = sum(counts.values())
        entry: dict[str, Any] = {
            "switches": sw,
            "counts": counts,
            "directional_coverage_pct": round(((total - counts["neutral_stand_aside"]) / total) * 100.0, 2) if total else 0.0,
            "stand_aside_pct": round((counts["neutral_stand_aside"] / total) * 100.0, 2) if total else 0.0,
            "changed_decisions_vs_full_count": len(changed_vs_full),
            "changed_decisions_vs_full": changed_vs_full[:20],
            "changed_decisions_vs_technical_count": len(changed_vs_tech),
        }
        if replay_sequence:
            pairs = list(zip(pkts, [b["market_snapshot"] for b in bundles]))
            sel = "tech" if arm_name == "Technical_Only" else "full"
            entry["shadow_paper_1x_slip"] = _simulate_shadow_paper_sequence(
                pairs, arm_selector=sel, starting_cash=float(starting_cash), slippage_mult=1.0, funding_by_symbol=funding_by_symbol
            )
            entry["shadow_paper_2x_slip"] = _simulate_shadow_paper_sequence(
                pairs, arm_selector=sel, starting_cash=float(starting_cash), slippage_mult=2.0, funding_by_symbol=funding_by_symbol
            )
            entry["post_maturity_accuracy"] = _compute_sequence_directional_accuracy(pairs, arm_selector=sel)
        results[arm_name] = entry

    last_inst = ((bundles[-1].get("market_snapshot") or {}).get("instruments") or {})
    port_audit = evaluate_portfolio_risk_audit(
        {sym: ((last_inst.get(sym) or {}).get("candles_closed_1h")) or [] for sym in SYMBOLS},
        [
            {"symbol": sym, **full_packets[-1]["judgments"][sym]["paper_eligibility"]["position_sizing"]}
            for sym in SYMBOLS
            if (full_packets[-1]["judgments"][sym]["paper_eligibility"].get("eligible"))
        ],
        equity_usdt=float(starting_cash),
    )

    return {
        "cutoff_utc": full_packets[-1]["cutoff_utc"],
        "starting_cash_usdt": float(starting_cash),
        "snapshots_evaluated": len(bundles),
        "arms": results,
        "portfolio_risk_audit": port_audit,
    }


def _load_piia_v2_eval_dataset(eval_dir: Path) -> dict[str, Any]:
    import csv as _csv

    candles_by_sym: dict[str, list[dict[str, Any]]] = {}
    funding_by_sym: dict[str, list[dict[str, Any]]] = {}
    for sym in SYMBOLS:
        csv_p = eval_dir / f"{sym}_1h.csv"
        if not csv_p.is_file():
            raise FileNotFoundError(f"Missing eval 1h CSV: {csv_p}")
        rows: list[dict[str, Any]] = []
        with csv_p.open("r", encoding="utf-8") as fh:
            reader = _csv.DictReader(fh)
            for row in reader:
                dt = parse_aware_timestamp(row.get("date") or row.get("timestamp_utc"))
                if dt is None:
                    continue
                rows.append(
                    {
                        "open_at_utc": dt.astimezone(timezone.utc).isoformat(),
                        "close_at_utc": (dt.astimezone(timezone.utc) + timedelta(hours=1)).isoformat(),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]),
                    }
                )
        rows.sort(key=lambda r: r["open_at_utc"])
        candles_by_sym[sym] = rows

        fn_p = eval_dir / f"{sym}_funding.json"
        if fn_p.is_file():
            raw_fn = json.loads(fn_p.read_text(encoding="utf-8"))
            funding_by_sym[sym] = raw_fn if isinstance(raw_fn, list) else (raw_fn.get("rows") or [])
        else:
            funding_by_sym[sym] = []

    alf_p = eval_dir / "alfred_macro.json"
    alf_raw = json.loads(alf_p.read_text(encoding="utf-8")) if alf_p.is_file() else {}
    return {
        "candles_by_symbol": candles_by_sym,
        "funding_by_symbol": funding_by_sym,
        "alfred_raw": alf_raw,
    }


def _slice_eval_bundle(dataset: dict[str, Any], cutoff_dt: datetime) -> dict[str, Any]:
    cutoff_utc = cutoff_dt.astimezone(timezone.utc)
    instruments: dict[str, Any] = {}
    for sym in SYMBOLS:
        all_rows = dataset["candles_by_symbol"][sym]
        closed = [
            r for r in all_rows
            if (c_dt := parse_aware_timestamp(r["close_at_utc"])) is not None and c_dt <= cutoff_utc
        ][-720:]
        instruments[sym] = {
            "candles_closed_1h": closed,
            "open_interest": None,
            "funding_latest": None,
        }

    # Causal ALFRED slice: daily macro observations strictly with observation_date < cutoff_utc.date()
    alf_series_out: dict[str, Any] = {}
    raw_series = (dataset.get("alfred_raw") or {}).get("series") or {}
    cutoff_date_str = cutoff_utc.date().isoformat()
    for sid, sdata in raw_series.items():
        if not isinstance(sdata, dict):
            continue
        obs = [
            o for o in (sdata.get("observations") or [])
            if str(o.get("observation_date") or "") < cutoff_date_str
        ]
        alf_series_out[sid] = {
            "series_id": sid,
            "vintage_date": cutoff_date_str,
            "observations": obs,
        }

    return {
        "protocol_version": "piia_judgment_v3_1",
        "collected_at_utc": cutoff_utc.isoformat(),
        "first_retrieved_at_utc": cutoff_utc.isoformat(),
        "bar_cutoff_utc": cutoff_utc.isoformat(),
        "decision_cutoff_utc": cutoff_utc.isoformat(),
        "cutoff_utc": cutoff_utc.isoformat(),
        "is_synthetic_demo": False,
        "replay_mode": True,
        "market_snapshot": {
            "common_cutoff_utc": cutoff_utc.isoformat(),
            "first_retrieved_at_utc": cutoff_utc.isoformat(),
            "instruments": instruments,
        },
        "alfred_series": alf_series_out,
        "event_calendar": {"present_sources": [], "sources": {}},
        "etf_flows": {"usable_for_btc_research": False, "reason": "excluded_in_historical_replay_without_archive"},
        "cot_gold": {"ok": False, "reason": "excluded_in_historical_replay_without_archive"},
        "fear_and_greed": {"ok": False, "reason": "excluded_in_historical_replay_without_archive"},
        "mempool": {"ok": False, "reason": "excluded_in_historical_replay_without_archive"},
        "polymarket": {"ok": False, "reason": "excluded_in_historical_replay_without_archive"},
        "sec_edgar": {"ok": False, "reason": "excluded_in_historical_replay_without_archive"},
        "raw_news_items": [],
        "chronos_forecast": {"weights_ran": False, "status": "NOT_RUN_REPLAY"},
    }


def _evaluate_window_predetermined(
    dataset: dict[str, Any],
    *,
    timestamps: list[datetime],
    bar_horizon_end: datetime,
    starting_cash: float,
    ai_provider: JudgmentAIProvider | None,
    call_live_model: bool,
    run_ablation: bool,
) -> dict[str, Any]:
    bundles = [_slice_eval_bundle(dataset, t) for t in timestamps]
    packets: list[dict[str, Any]] = []
    a1_seq: list[dict[str, dict[str, str]]] = []
    a2_seq: list[dict[str, dict[str, str]]] = []
    a1_by_t: dict[datetime, dict[str, dict[str, str]]] = {}
    a2_by_t: dict[datetime, dict[str, dict[str, str]]] = {}
    ai_statuses_a1: list[str] = []
    ai_statuses_a2: list[str] = []
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for b in bundles:
        pkt = synthesize_judgment_packet(b, include_v2_control=True, starting_cash_usdt=float(starting_cash), replay_mode=True)
        packets.append(pkt)
        pairs.append((pkt, b["market_snapshot"]))
        cut_dt = parse_aware_timestamp(pkt["cutoff_utc"])
        if ai_provider is not None or call_live_model:
            comp = run_four_way_comparison(b, ai_provider=ai_provider, call_live_model=call_live_model, starting_cash_usdt=float(starting_cash))
            a1 = comp["arms"]["A1_direct_ai_candles_only"]
            a2 = comp["arms"]["A2_ai_freeform_same_packet"]
            ai_statuses_a1.append(a1["status"])
            ai_statuses_a2.append(a2["status"])
            if a1.get("decisions"):
                a1_seq.append(a1["decisions"])
                if cut_dt is not None:
                    a1_by_t[cut_dt] = a1["decisions"]
            if a2.get("decisions"):
                a2_seq.append(a2["decisions"])
                if cut_dt is not None:
                    a2_by_t[cut_dt] = a2["decisions"]

    # Stitch ALL intermediate hourly bars up to `bar_horizon_end` so 48h horizons & stops resolve strictly in-window
    first_t = timestamps[0]
    last_t = timestamps[-1]
    all_hourly_times = sorted(
        {
            dt
            for sym in SYMBOLS
            for r in dataset["candles_by_symbol"][sym]
            if (dt := parse_aware_timestamp(r["close_at_utc"])) is not None and first_t <= dt <= bar_horizon_end
        }
    )
    hourly_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    pkt_by_t = {parse_aware_timestamp(p["cutoff_utc"]): p for p in packets}
    cur_pkt = packets[0]
    hourly_a1_seq: list[dict[str, dict[str, str]]] = []
    hourly_a2_seq: list[dict[str, dict[str, str]]] = []
    cur_a1: dict[str, dict[str, str]] = a1_seq[0] if a1_seq else {}
    cur_a2: dict[str, dict[str, str]] = a2_seq[0] if a2_seq else {}
    for h_dt in all_hourly_times:
        if h_dt in pkt_by_t:
            cur_pkt = pkt_by_t[h_dt]
        if h_dt in a1_by_t:
            cur_a1 = a1_by_t[h_dt]
        if h_dt in a2_by_t:
            cur_a2 = a2_by_t[h_dt]
        inst_h = {}
        for sym in SYMBOLS:
            sub = [
                r for r in dataset["candles_by_symbol"][sym]
                if (c_dt := parse_aware_timestamp(r["close_at_utc"])) is not None and c_dt <= h_dt
            ][-60:]
            inst_h[sym] = {"candles_closed_1h": sub}
        hourly_pairs.append((cur_pkt, {"common_cutoff_utc": h_dt.isoformat(), "instruments": inst_h}))
        if len(a1_seq) == len(bundles):
            hourly_a1_seq.append(cur_a1)
        if len(a2_seq) == len(bundles):
            hourly_a2_seq.append(cur_a2)

    fn = dataset["funding_by_symbol"]
    cand_hist = dataset["candles_by_symbol"]
    arms_out: dict[str, Any] = {
        "A0_technical_only": {
            "status": "executed",
            "shadow_paper_1x_slip": _simulate_shadow_paper_sequence(hourly_pairs, arm_selector="tech", starting_cash=starting_cash, slippage_mult=1.0, funding_by_symbol=fn),
            "shadow_paper_2x_slip": _simulate_shadow_paper_sequence(hourly_pairs, arm_selector="tech", starting_cash=starting_cash, slippage_mult=2.0, funding_by_symbol=fn),
            "post_maturity_accuracy": _compute_sequence_directional_accuracy(pairs, arm_selector="tech", price_history_bars=cand_hist, horizon_bar_end_utc=bar_horizon_end),
        },
        "A3_project_full_judgment_v3": {
            "status": "executed",
            "shadow_paper_1x_slip": _simulate_shadow_paper_sequence(hourly_pairs, arm_selector="full", starting_cash=starting_cash, slippage_mult=1.0, funding_by_symbol=fn),
            "shadow_paper_2x_slip": _simulate_shadow_paper_sequence(hourly_pairs, arm_selector="full", starting_cash=starting_cash, slippage_mult=2.0, funding_by_symbol=fn),
            "post_maturity_accuracy": _compute_sequence_directional_accuracy(pairs, arm_selector="full", price_history_bars=cand_hist, horizon_bar_end_utc=bar_horizon_end),
        },
        "v2_2_1_official_control": {
            "status": "executed",
            "shadow_paper_1x_slip": _simulate_shadow_paper_sequence(hourly_pairs, arm_selector="v2_official", starting_cash=starting_cash, slippage_mult=1.0, funding_by_symbol=fn, respect_pending_review=False),
            "shadow_paper_2x_slip": _simulate_shadow_paper_sequence(hourly_pairs, arm_selector="v2_official", starting_cash=starting_cash, slippage_mult=2.0, funding_by_symbol=fn, respect_pending_review=False),
        },
    }
    if len(a1_seq) == len(bundles):
        arms_out["A1_direct_ai_candles_only"] = {
            "status": "completed",
            "shadow_paper_1x_slip": _simulate_shadow_paper_sequence(hourly_pairs, arm_selector="custom_decisions", starting_cash=starting_cash, slippage_mult=1.0, funding_by_symbol=fn, custom_decisions_seq=hourly_a1_seq),
            "post_maturity_accuracy": _compute_sequence_directional_accuracy(pairs, arm_selector="custom_decisions", custom_decisions_seq=a1_seq, price_history_bars=cand_hist, horizon_bar_end_utc=bar_horizon_end),
        }
    else:
        arms_out["A1_direct_ai_candles_only"] = {"status": ai_statuses_a1[0] if ai_statuses_a1 else "NOT_RUN_NEEDS_MODEL_CONFIG"}

    if len(a2_seq) == len(bundles):
        arms_out["A2_ai_freeform_same_packet"] = {
            "status": "completed",
            "shadow_paper_1x_slip": _simulate_shadow_paper_sequence(hourly_pairs, arm_selector="custom_decisions", starting_cash=starting_cash, slippage_mult=1.0, funding_by_symbol=fn, custom_decisions_seq=hourly_a2_seq),
            "post_maturity_accuracy": _compute_sequence_directional_accuracy(pairs, arm_selector="custom_decisions", custom_decisions_seq=a2_seq, price_history_bars=cand_hist, horizon_bar_end_utc=bar_horizon_end),
        }
    else:
        arms_out["A2_ai_freeform_same_packet"] = {"status": ai_statuses_a2[0] if ai_statuses_a2 else "NOT_RUN_NEEDS_MODEL_CONFIG"}

    out: dict[str, Any] = {
        "cutoff_start_utc": first_t.isoformat(),
        "cutoff_end_utc": last_t.isoformat(),
        "bar_horizon_end_utc": bar_horizon_end.isoformat(),
        "predetermined_timestamps_count": len(timestamps),
        "hourly_bars_evaluated": len(hourly_pairs),
        "starting_cash_usdt": float(starting_cash),
        "arms": arms_out,
        "portfolio_risk_audit": evaluate_portfolio_risk_audit(
            {
                sym: [
                    r for r in dataset["candles_by_symbol"][sym]
                    if (c_dt := parse_aware_timestamp(r["close_at_utc"])) is not None and c_dt <= bar_horizon_end
                ]
                for sym in SYMBOLS
            },
            equity_usdt=float(starting_cash),
        ),
    }
    if run_ablation:
        out["domain_ablation"] = run_domain_ablation(
            bundles[0],
            replay_sequence=bundles,
            starting_cash=float(starting_cash),
            funding_by_symbol=fn,
        )
    return out


def run_predetermined_benchmark(
    *,
    eval_dir: Path | None = None,
    starting_cash: float = 100.0,
    ai_provider: JudgmentAIProvider | None = None,
    call_live_model: bool = False,
) -> dict[str, Any]:
    """Run fair 100 USDT multi-arm benchmark on predetermined UTC timestamps with untouched holdout."""
    base_dir = Path(eval_dir) if eval_dir else (ROOT / "data" / "piia_v2_eval")
    dataset = _load_piia_v2_eval_dataset(base_dir)

    # Predetermined fixed 24h UTC schedule (strictly without holdout contamination or tuning):
    # - Development replay: 2026-10-15T12:00Z .. 2026-11-13T12:00Z (30 days; 48h horizons mature by 2026-11-15T23:00Z < holdout)
    # - Untouched holdout: 2026-11-16T00:00Z .. 2026-11-22T23:00Z (cutoffs 2026-11-16T12:00Z .. 2026-11-20T12:00Z; mature by 2026-11-22T23:00Z)
    dev_times = [datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc) + timedelta(days=d) for d in range(30)]
    dev_horizon_end = datetime(2026, 11, 15, 23, 0, tzinfo=timezone.utc)
    holdout_times = [datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc) + timedelta(days=d) for d in range(5)]
    holdout_horizon_end = datetime(2026, 11, 22, 23, 0, tzinfo=timezone.utc)

    dev_res = _evaluate_window_predetermined(
        dataset,
        timestamps=dev_times,
        bar_horizon_end=dev_horizon_end,
        starting_cash=float(starting_cash),
        ai_provider=ai_provider,
        call_live_model=bool(call_live_model),
        run_ablation=True,
    )
    holdout_res = _evaluate_window_predetermined(
        dataset,
        timestamps=holdout_times,
        bar_horizon_end=holdout_horizon_end,
        starting_cash=float(starting_cash),
        ai_provider=ai_provider,
        call_live_model=bool(call_live_model),
        run_ablation=False,
    )
    return {
        "benchmark_id": "piia_judgment_v3_1_fair_100usdt_predetermined",
        "starting_cash_usdt": float(starting_cash),
        "eval_dir_name": base_dir.name,
        "parameter_tuning_performed": False,
        "holdout_policy": {
            "holdout_range_utc": "2026-11-16T00:00:00+00:00 .. 2026-11-22T23:00:00+00:00",
            "dev_cutoffs_end_utc": "2026-11-13T12:00:00+00:00",
            "dev_bar_horizon_end_utc": dev_horizon_end.isoformat(),
            "holdout_cutoffs_end_utc": "2026-11-20T12:00:00+00:00",
            "holdout_bar_horizon_end_utc": holdout_horizon_end.isoformat(),
            "note": "Dev window strictly bounded before 2026-11-16T00:00Z holdout boundary; holdout evaluated with zero rule/threshold tuning.",
        },
        "dev_window": dev_res,
        "holdout_window": holdout_res,
        "order_permissions": False,
    }


_PIT_VERIFIED_ARCHIVE_DOMAINS = (
    "intraday_macro",
    "fear_greed",
    "etf_flows",
    "onchain",
    "exchange_netflow",
    "gold_cot",
    "gold_etf_flows",
    "polymarket",
)


def replay_archived_judgment_ledger(
    db_path: Path | None = None,
    *,
    demo: bool = False,
    starting_cash: float = 100.0,
    run_ablation: bool = True,
) -> dict[str, Any]:
    """Reconstruct and score full 5-domain archived point-in-time judgment snapshots without lookahead."""
    target_db = Path(db_path) if db_path is not None else default_judgment_ledger_path(demo=demo)
    if not target_db.is_file():
        return {
            "status": "NO_ARCHIVED_LEDGER",
            "db_path": _display_path(target_db),
            "archived_bundles_count": 0,
            "starting_cash_usdt": float(starting_cash),
            "order_permissions": False,
        }

    conn = sqlite3.connect(f"{target_db.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        has_arch = bool(conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='archived_input_bundles'").fetchone())
        rows = (
            conn.execute(
                "SELECT cutoff_utc, captured_at_utc, bundle_sha256, bundle_json FROM archived_input_bundles ORDER BY cutoff_utc ASC"
            ).fetchall()
            if has_arch
            else []
        )
    finally:
        conn.close()

    if not rows:
        return {
            "status": "NO_ARCHIVED_BUNDLES",
            "db_path": _display_path(target_db),
            "archived_bundles_count": 0,
            "starting_cash_usdt": float(starting_cash),
            "order_permissions": False,
        }

    bundles: list[dict[str, Any]] = []
    bundle_hashes: list[str] = []
    for row in rows:
        raw_b = json.loads(row["bundle_json"])
        raw_b["replay_mode"] = True
        for k in _PIT_VERIFIED_ARCHIVE_DOMAINS:
            sub = raw_b.get(k)
            if isinstance(sub, dict):
                sub["historical_point_in_time_verified"] = True
        bundles.append(raw_b)
        bundle_hashes.append(str(row["bundle_sha256"]))

    packets = [
        synthesize_judgment_packet(b, include_v2_control=True, starting_cash_usdt=float(starting_cash), replay_mode=True)
        for b in bundles
    ]
    pairs = [(pkt, b["market_snapshot"]) for pkt, b in zip(packets, bundles)]
    cutoffs = [pkt["cutoff_utc"] for pkt in packets]
    contig = _compute_contiguous_stats(cutoffs)

    arms = {
        "A0_technical_only": {
            "shadow_paper_1x_slip": _simulate_shadow_paper_sequence(pairs, arm_selector="tech", starting_cash=float(starting_cash), slippage_mult=1.0),
            "shadow_paper_2x_slip": _simulate_shadow_paper_sequence(pairs, arm_selector="tech", starting_cash=float(starting_cash), slippage_mult=2.0),
            "post_maturity_accuracy": _compute_sequence_directional_accuracy(pairs, arm_selector="tech"),
        },
        "A3_project_full_judgment_v3": {
            "shadow_paper_1x_slip": _simulate_shadow_paper_sequence(pairs, arm_selector="full", starting_cash=float(starting_cash), slippage_mult=1.0),
            "shadow_paper_2x_slip": _simulate_shadow_paper_sequence(pairs, arm_selector="full", starting_cash=float(starting_cash), slippage_mult=2.0),
            "post_maturity_accuracy": _compute_sequence_directional_accuracy(pairs, arm_selector="full"),
        },
        "v2_2_1_official_control": {
            "shadow_paper_1x_slip": _simulate_shadow_paper_sequence(pairs, arm_selector="v2_official", starting_cash=float(starting_cash), slippage_mult=1.0, respect_pending_review=False),
            "shadow_paper_2x_slip": _simulate_shadow_paper_sequence(pairs, arm_selector="v2_official", starting_cash=float(starting_cash), slippage_mult=2.0, respect_pending_review=False),
        },
    }
    last_inst = ((bundles[-1].get("market_snapshot") or {}).get("instruments") or {})
    port_audit = evaluate_portfolio_risk_audit(
        {sym: ((last_inst.get(sym) or {}).get("candles_closed_1h")) or [] for sym in SYMBOLS},
        [
            {"symbol": sym, **packets[-1]["judgments"][sym]["paper_eligibility"]["position_sizing"]}
            for sym in SYMBOLS
            if (packets[-1]["judgments"][sym]["paper_eligibility"].get("eligible"))
        ],
        equity_usdt=float(starting_cash),
    )

    out: dict[str, Any] = {
        "status": "replayed_archived_5_domain_bundles",
        "protocol_version": JUDGMENT_SCHEMA_VERSION,
        "db_path": _display_path(target_db),
        "replay_non_blind_5_domains": True,
        "archived_bundles_count": len(bundles),
        "earliest_cutoff_utc": cutoffs[0],
        "latest_cutoff_utc": cutoffs[-1],
        "contiguous_stats": contig,
        "starting_cash_usdt": float(starting_cash),
        "latest_bundle_sha256": bundle_hashes[-1],
        "arms": arms,
        "portfolio_risk_audit": port_audit,
        "order_permissions": False,
    }
    if run_ablation:
        out["domain_ablation"] = run_domain_ablation(bundles[0], replay_sequence=bundles, starting_cash=float(starting_cash))
    return out
