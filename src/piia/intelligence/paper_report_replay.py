"""On-demand local price-only report recomputation; no model or exchange calls."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile

from piia.intelligence.paths import ROOT
from piia.intelligence.manual_observation import interpreter
from piia.intelligence.report_ledger_check import APP_ID, check_report_ledger
from piia.intelligence.shadow_claim_audit import audit_claims

MAX_REPORT_BYTES = 2_000_000


def ledger_digest(path: Path) -> str:
    """Digest inputs used by the replay within one read transaction."""
    digest = hashlib.sha256()
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as db:
        db.execute("BEGIN")
        if db.execute("PRAGMA application_id").fetchone()[0] != APP_ID:
            raise ValueError("unrelated ledger")
        tables = {x[0] for x in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "observations" not in tables:
            raise ValueError("missing observations")
        for table in ("observations", "ai_shadow_claims", "ai_shadow_events", "chronos_shadow_claims", "chronos_shadow_events"):
            digest.update((table + "\n").encode())
            if table not in tables:
                digest.update(b"missing\n")
                continue
            for row in db.execute(f"SELECT * FROM {table} ORDER BY rowid"):
                digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())
                digest.update(b"\n")
    return digest.hexdigest()


def verify_saved_report(*, root: Path = ROOT) -> dict:
    """Compare saved report to a fresh replay; report remains a research approximation."""
    root = Path(root)
    report_path = root / "artifacts" / "prospective_paper_report.json"
    ledger = root / "data" / "piia" / "frozen_policy_v1.sqlite"
    try:
        python = interpreter()
        if python is None:
            raise ValueError("replay_environment_unavailable")
        raw = report_path.read_bytes()
        if len(raw) > MAX_REPORT_BYTES:
            raise ValueError("report_too_large")
        saved = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))
        if not isinstance(saved, dict) or not check_report_ledger(saved, ledger=ledger)["ok"]:
            raise ValueError("report_ledger_check_failed")
        before = ledger_digest(ledger)
        for arm in ("ai_shadow", "chronos_shadow"):
            entry = saved.get(arm)
            if isinstance(entry, dict) and entry.get("status") == "complete" and not audit_claims(ledger, arm)["ok"]:
                raise ValueError("shadow_claim_audit_failed")
        env = os.environ.copy()
        env.pop("PIIA_MODEL_API_KEY", None)
        env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        with tempfile.TemporaryDirectory(prefix="paper-replay-") as temp:
            output = Path(temp) / "replay.json"
            completed = subprocess.run([str(python), "-m", "experiments.prospective_paper_report",
                                        "--ledger", str(ledger), "--out", str(output)], cwd=str(ROOT), env=env,
                                       capture_output=True, timeout=180, check=False)
            if completed.returncode != 0 or not output.is_file() or output.stat().st_size > MAX_REPORT_BYTES:
                raise ValueError("replay_failed")
            replay = json.loads(output.read_bytes(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))
        if not isinstance(replay, dict) or not check_report_ledger(replay, ledger=ledger)["ok"]:
            raise ValueError("replay_ledger_check_failed")
        if ledger_digest(ledger) != before or report_path.read_bytes() != raw:
            raise ValueError("inputs_changed_during_replay")
        saved = {k: v for k, v in saved.items() if k != "generated_at_utc"}
        replay = {k: v for k, v in replay.items() if k != "generated_at_utc"}
        if saved != replay:
            raise ValueError("report_values_differ_from_replay")
        return {"status": "matched_local_replay", "order_permissions": False,
                "ledger_sha256": before, "report_sha256": hashlib.sha256(raw).hexdigest(),
                "note": "Local price-only replay match; not independent audit, real execution or verified net performance."}
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error, subprocess.SubprocessError) as exc:
        return {"status": "replay_unavailable_or_mismatch", "reason": str(exc) if isinstance(exc, ValueError) else type(exc).__name__,
                "order_permissions": False}
