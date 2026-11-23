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
import sqlite3
from typing import Any, Protocol
import uuid

import httpx

from piia.intelligence.agent_v2.pillars import SYMBOLS, canonical_sha256
from piia.intelligence.agent_v2.risk_gate import evaluate_programmatic_risk
from piia.intelligence.judgment.evidence import (
    collect_judgment_inputs,
    generate_demo_judgment_inputs,
    load_judgment_config,
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
JUDGMENT_SCHEMA_VERSION = "piia_judgment_v3_1"

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
        """
    )
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


def _resolve_pending_outcomes(
    conn: sqlite3.Connection,
    market_snapshot: dict[str, Any],
    *,
    now_cutoff: datetime,
    resolved_at_iso: str,
) -> int:
    """Resolve matured directional predictions forward-only from closed 1h bars."""
    close_lookup: dict[str, dict[str, float]] = {}
    for sym in SYMBOLS:
        inst = (market_snapshot.get("instruments") or {}).get(sym) or {}
        mp: dict[str, float] = {}
        for c in inst.get("candles_closed_1h") or []:
            c_dt = parse_aware_timestamp(c.get("close_at_utc"))
            if c_dt is not None and c_dt <= now_cutoff:
                mp[c_dt.isoformat()] = float(c["close"])
        close_lookup[sym] = mp

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
        tgt_dt = parse_aware_timestamp(r["target_outcome_utc"])
        if tgt_dt is None or tgt_dt > now_cutoff:
            continue
        exit_px = close_lookup.get(sym, {}).get(tgt_dt.isoformat())
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
                json.dumps({"entry_px": entry_px, "exit_px": exit_px, "target_utc": tgt_dt.isoformat()}),
            ),
        )
        resolved_n += 1
    return resolved_n


def record_judgment_cycle(
    inputs_bundle: dict[str, Any] | None = None,
    *,
    db_path: Path | None = None,
    demo_mode: bool = False,
    starting_cash: float = 500.0,
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

    packet = synthesize_judgment_packet(inputs_bundle)
    cutoff_dt = parse_aware_timestamp(packet["cutoff_utc"]) or datetime.now(timezone.utc)
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
        existing = conn.execute(
            "SELECT packet_id, judgment_hash FROM judgment_packets WHERE cutoff_utc = ?",
            (packet["cutoff_utc"],),
        ).fetchone()
        if existing is not None:
            conn.commit()
            return {
                "status": "already_recorded_for_hour",
                "packet_id": existing["packet_id"],
                "cutoff_utc": packet["cutoff_utc"],
                "judgment_hash": existing["judgment_hash"],
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

        # Store pre-outcome directional arms (`A0_technical_only` and `A3_project_full_judgment_v3`)
        for sym in SYMBOLS:
            last_px = packet["judgments"][sym].get("last_close")
            for horizon in HORIZONS:
                h_hours = int(cfg["horizons"][horizon]["outcome_horizon_hours"])
                tgt_iso = (cutoff_dt + timedelta(hours=h_hours)).isoformat()
                hd = packet["judgments"][sym][horizon]
                for arm_name, bias_val in (
                    ("A0_technical_only", hd["technical_only"]["bias"]),
                    ("A3_project_full_judgment_v3", hd["final_bias"]),
                ):
                    conn.execute(
                        """
                        INSERT INTO arm_judgments(
                            row_id, packet_id, cutoff_utc, arm_name, symbol, horizon,
                            target_outcome_utc, predicted_bias, technical_bias, entry_price_ref, judgment_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        ),
                    )

        # Initialize shadow books on first packet (500 USDT, non-interfering with v2.2.1!)
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
        pkts = conn.execute("SELECT cutoff_utc, judgment_hash, packet_json FROM judgment_packets ORDER BY cutoff_utc DESC").fetchall()
        out_rows = conn.execute(
            "SELECT arm_name, COUNT(*) AS n, SUM(CASE WHEN direction_hit=1 THEN 1 ELSE 0 END) AS hits, "
            "SUM(CASE WHEN direction_hit IS NOT NULL THEN 1 ELSE 0 END) AS scored, AVG(realized_return_pct) AS avg_ret "
            "FROM outcome_observations GROUP BY arm_name"
        ).fetchall()
    finally:
        conn.close()

    req = int(cfg["shadow_paper"]["prospective_min_contiguous_hours"])
    is_live = kind == KIND_LIVE_J3
    v_status = (
        "NOT_A_PROSPECTIVE_LEDGER"
        if not is_live
        else ("PENDING_AUDIT_CONTIGUOUS_WINDOW_REACHED" if len(pkts) >= req else "PENDING_INSUFFICIENT_PROSPECTIVE_SAMPLE")
    )
    return {
        "initialized": True,
        "db_path": _display_path(db_path),
        "ledger_kind": kind,
        "packets_recorded": len(pkts),
        "latest_cutoff_utc": pkts[0]["cutoff_utc"] if pkts else None,
        "previous_cutoff_utc": pkts[1]["cutoff_utc"] if len(pkts) >= 2 else None,
        "validation_status": v_status,
        "validation_note": (
            f"已记录 {len(pkts)} 小时；达到 {req} 连续小时前保持待验证。"
            if is_live and len(pkts) < req
            else ("非真实前瞻影子账本。" if not is_live else f"已记录 {len(pkts)} 条，进入待审计阶段。")
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


def _call_official_ai_for_comparison(arm_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Call user-configured official chat API strictly without fabricating or mutating payloads."""
    if not model_configured():
        return {"status": "NOT_RUN_NEEDS_MODEL_CONFIG", "decisions": {}}
    provider = (model_provider() or "").lower()
    base = model_base_url() or ""
    name = model_name() or ""
    import os
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
            body = {"model": name, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]}
        elif provider == "anthropic":
            url = f"{base.rstrip('/')}/v1/messages"
            headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
            body = {"model": name, "max_tokens": 2048, "system": system, "messages": [{"role": "user", "content": user_msg}]}
        else:
            return {"status": f"UNSUPPORTED_PROVIDER:{provider}", "decisions": {}}
        with httpx.Client(timeout=25.0) as client:
            resp = client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        text, usage = parse_official_response(provider, resp.json())
        parsed = json.loads(text)
        return {"status": "completed", "arm": arm_name, "provider": provider, "model": name, "usage": usage, "decisions": parsed}
    except Exception as exc:
        return {"status": f"ERROR:{type(exc).__name__}", "decisions": {}}


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


def run_four_way_comparison(
    inputs_bundle: dict[str, Any],
    *,
    ai_provider: JudgmentAIProvider | None = None,
    call_live_model: bool = False,
) -> dict[str, Any]:
    """Compare A0 Technical-Only, A1 Direct AI (Candles), A2 AI Freeform (Same Packet), A3 Project Rules."""
    packet = synthesize_judgment_packet(inputs_bundle)
    cutoff_utc = packet["cutoff_utc"]

    a0_dec = {
        sym: {h: packet["judgments"][sym][h]["technical_only"]["bias"] for h in HORIZONS}
        for sym in SYMBOLS
    }
    a3_dec = {
        sym: {h: packet["judgments"][sym][h]["final_bias"] for h in HORIZONS}
        for sym in SYMBOLS
    }

    # A1 Direct AI receives ONLY closed bars <= cutoff_utc + task instructions (ZERO project curated domains!)
    candles_only_payload = {
        "cutoff_utc": cutoff_utc,
        "task": "Assess intraday (4-12h) and 1-3 trading day bias (bullish_bias / bearish_bias / neutral_stand_aside) for BTCUSDT, XAUUSDT, NDX100USDT.",
        "instruments": {
            sym: {
                "last_close": packet["judgments"][sym].get("last_close"),
                "recent_closed_1h_tail": ((inputs_bundle["market_snapshot"].get("instruments") or {}).get(sym) or {}).get("candles_closed_1h", [])[-48:],
            }
            for sym in SYMBOLS
        },
    }
    # A2 Freeform AI receives the EXACT SAME frozen EvidenceAtom list (no future/stale items, no deterministic verdicts)
    same_packet_payload = {
        "cutoff_utc": cutoff_utc,
        "task": candles_only_payload["task"],
        "tide_raw": packet["tide"],
        "evidence_atoms": [
            {k: v for k, v in a.items() if k != "role_by_symbol_horizon"}
            for a in packet["evidence_atoms"]
            if a["freshness_status"] in {"fresh", "lagged_usable", "missing"}
        ],
    }

    def _run_ai(arm: str, pld: dict[str, Any]) -> dict[str, Any]:
        if ai_provider is not None:
            res = ai_provider.compare(arm, pld)
            return {"status": res.get("status", "completed"), "decisions": res.get("decisions") or {}, "payload_keys": sorted(pld)}
        if call_live_model:
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
    return {
        "cutoff_utc": cutoff_utc,
        "judgment_hash": packet["judgment_hash"],
        "arms": arms,
        "rule_changes_vs_technical": diff_a3_vs_a0,
        "notes": [
            "A2 vs A1 isolates external data coverage contribution; A3 vs A2 isolates deterministic causal deduplication and hard-gate rules.",
            "Directional hit rates remain PENDING_OUTCOME until future closed horizon bars are observed post-cutoff.",
        ],
    }


def _simulate_shadow_paper_sequence(
    packets_and_bars: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    arm_selector: str,
    starting_cash: float = 500.0,
    slippage_mult: float = 1.0,
) -> dict[str, Any]:
    """Run identical 500 USDT single-position isolated cost/stop accounting on multi-hour replay frames."""
    cfg = load_judgment_config()["shadow_paper"]
    cash = float(starting_cash)
    peak_eq = cash
    max_dd_pct = 0.0
    turnover = 0.0
    closed_trades = 0
    open_bars = 0
    total_bars = len(packets_and_bars)
    open_pos: dict[str, Any] | None = None
    slip = float(cfg["slippage_base"]) * slippage_mult

    for i, (pkt, mkt_snap) in enumerate(packets_and_bars):
        # Mark-to-market / close on opposite or expired horizon or stop
        cur_eq = cash
        if open_pos is not None:
            open_bars += 1
            sym = open_pos["symbol"]
            c_rows = ((mkt_snap.get("instruments") or {}).get(sym) or {}).get("candles_closed_1h") or []
            if c_rows:
                bar = c_rows[-1]
                hi, lo, cl = float(bar["high"]), float(bar["low"]), float(bar["close"])
                spec = resolve_spec(sym).to_v1_spec()
                side = open_pos["side"]
                hit_sl = (side == "long" and lo <= open_pos["stop"]) or (side == "short" and hi >= open_pos["stop"])
                exit_reason = None
                raw_exit = cl
                if hit_sl:
                    raw_exit = stop_fill(side, float(bar["open"]), open_pos["stop"], slip)
                    exit_reason = "stop_loss"
                else:
                    cur_hd = pkt["judgments"][sym]["swing_1_3d"]
                    cur_bias = cur_hd["final_bias"] if arm_selector == "full" else cur_hd["technical_only"]["bias"]
                    if cur_bias == ("bearish_bias" if side == "long" else "bullish_bias"):
                        raw_exit = cl * (1.0 - slip if side == "long" else 1.0 + slip)
                        exit_reason = "direction_flip"
                if exit_reason:
                    gross, exit_fee, net = exit_pnl(side, open_pos["qty"], open_pos["entry"], raw_exit, open_pos["entry_fee"], spec)
                    cash += (net + open_pos["entry_fee"])
                    turnover += open_pos["qty"] * raw_exit * spec.contract_size
                    closed_trades += 1
                    open_pos = None
                else:
                    unreal = (cl - open_pos["entry"]) * open_pos["qty"] * spec.contract_size * (1.0 if side == "long" else -1.0)
                    cur_eq = cash + unreal
        cur_eq = cash if open_pos is None else cur_eq
        peak_eq = max(peak_eq, cur_eq)
        if peak_eq > 0:
            max_dd_pct = max(max_dd_pct, (peak_eq - cur_eq) / peak_eq * 100.0)

        if open_pos is None and i + 1 < total_bars:
            for sym in SYMBOLS:
                hd = pkt["judgments"][sym]["swing_1_3d"]
                bias = hd["final_bias"] if arm_selector == "full" else hd["technical_only"]["bias"]
                if bias not in {"bullish_bias", "bearish_bias"}:
                    continue
                side = "long" if bias == "bullish_bias" else "short"
                c_rows = ((mkt_snap.get("instruments") or {}).get(sym) or {}).get("candles_closed_1h") or []
                if len(c_rows) < 20:
                    continue
                last_c = float(c_rows[-1]["close"])
                atr_v = sum(abs(float(x["high"]) - float(x["low"])) for x in c_rows[-14:]) / 14.0
                rk = evaluate_programmatic_risk(
                    symbol=sym,
                    synth_out={"candidate_direction": side, "setup_type": "judgment_v3_shadow", "proposed_stop_atr_mult": 2.0},
                    market_pillar={"ohlc": {"close": last_c}, "atr14": atr_v, "funding_rate_8h": None},
                    equity_usdt=cash,
                    free_cash_usdt=cash,
                    open_positions_count=0,
                    kill_switch_halted=(cash / starting_cash <= 0.90),
                    slippage_override=slip,
                )
                if rk["approved_by_risk_gate"]:
                    sz = rk["position_sizing"]
                    ep = rk["entry_plan"]
                    xp = rk["exit_plan"]
                    cr = rk["cost_and_risk"]
                    fee = float(cr["entry_taker_fee_usdt"])
                    cash -= fee
                    turnover += float(sz["notional_usdt"])
                    open_pos = {
                        "symbol": sym,
                        "side": side,
                        "qty": float(sz["qty_contracts"]),
                        "entry": float(ep["assumed_fill_with_slippage"]),
                        "stop": float(xp["initial_stop_loss"]),
                        "entry_fee": fee,
                    }
                    break

    return {
        "price_and_fee_return_pct": round((cash / starting_cash - 1.0) * 100.0, 4),
        "net_return_pct_with_funding": None,  # unknown unless settled funding verified
        "funding_status": "unknown_in_shadow_ablation",
        "max_drawdown_pct": round(max_dd_pct, 4),
        "exposure_pct": round((open_bars / total_bars) * 100.0, 2) if total_bars else 0.0,
        "turnover_usdt": round(turnover, 2),
        "closed_trades": closed_trades,
    }


def run_domain_ablation(
    inputs_bundle: dict[str, Any],
    *,
    replay_sequence: list[dict[str, Any]] | None = None,
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
    full_packets = [synthesize_judgment_packet(b, ablation_switches=arm_configs["Full"], include_v2_control=False) for b in bundles]

    results: dict[str, Any] = {}
    for arm_name, sw in arm_configs.items():
        pkts = full_packets if arm_name == "Full" else [
            synthesize_judgment_packet(b, ablation_switches=sw, include_v2_control=False) for b in bundles
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
            entry["shadow_paper_1x_slip"] = _simulate_shadow_paper_sequence(pairs, arm_selector=sel, slippage_mult=1.0)
            entry["shadow_paper_2x_slip"] = _simulate_shadow_paper_sequence(pairs, arm_selector=sel, slippage_mult=2.0)
        results[arm_name] = entry

    return {
        "cutoff_utc": full_packets[-1]["cutoff_utc"],
        "snapshots_evaluated": len(bundles),
        "arms": results,
    }
