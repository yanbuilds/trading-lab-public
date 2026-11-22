"""Manual on-demand Chronos-2 shadow; independent paper research, no orders."""
from __future__ import annotations
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile

from experiments import frozen_policy_capture as capture
from piia.intelligence import ai_shadow
from piia.intelligence.chronos_forecast import default_forecast_python

ROOT = capture.ROOT
LEDGER = capture.LEDGER
MODEL = ROOT.parent / "forecast-model" / "chronos2-95a9710"
REVISION = "95a9710e2596287d08352589f42634fa5abdf0a7"
WEIGHTS = "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42"
CONFIG = "ef1143bfdc9c0376d9a056eefca46cb4b1ec3d0ffacd541ff56feb40fb708031"
SOURCES = tuple(dict.fromkeys((*capture.PINNED, "CHRONOS2_SHADOW_V1.md", "tools/run_chronos2_shadow.py",
    "experiments/chronos2_shadow.py", "experiments/chronos2_shadow_report.py",
    "experiments/prospective_paper_report.py", "experiments/prospective_funding_stress.py",
    "CHRONOS_SHADOW_V1.md", "experiments/chronos_shadow.py", "src/trading_lab/perps/research.py",
    "src/piia/intelligence/chronos_forecast.py", "src/piia/intelligence/evidence_gates.py",
    "FUNDING_OBSERVER_V1.md", "FUNDING_STRESS_V1.md", "experiments/funding_history.py",
    "experiments/funding_observer.py", "experiments/prospective_funding_realization.py",
    "src/piia/intelligence/ai_shadow.py", "src/piia/intelligence/paths.py", "src/piia/intelligence/paper/specs.py")))


def _sha(raw):
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def source_pins_before(observed):
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise ValueError("aware observation time required")
    pins = {}
    try:
        for name in SOURCES:
            raw = (ROOT / name).read_bytes()
            tracked = subprocess.run(["git", "show", f"HEAD:{name}"], cwd=ROOT, capture_output=True,
                                     timeout=15, check=True).stdout
            if _sha(raw) != _sha(tracked):
                raise ValueError("shadow source differs from HEAD")
            pins[name] = _sha(raw)
        stamp = subprocess.run(["git", "log", "-1", "--format=%ct", "HEAD", "--", *SOURCES],
                               cwd=ROOT, capture_output=True, timeout=15, check=True).stdout.strip()
        if int(stamp) >= int(observed.timestamp()):
            raise ValueError("shadow source not committed before observation")
    except (OSError, subprocess.SubprocessError, TypeError, OverflowError) as exc:
        raise ValueError("shadow source commit unavailable") from exc
    return pins


def _claim(path, ident, source_hash):
    with sqlite3.connect(path) as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("PRAGMA application_id").fetchone()[0] != ai_shadow.APP_ID:
            raise ValueError("unrelated database")
        row = db.execute("SELECT payload_json FROM observations WHERE id=?", (ident,)).fetchone()
        if not row or hashlib.sha256(row[0].encode()).hexdigest() != source_hash:
            raise ValueError("source observation changed")
        db.execute("CREATE TABLE IF NOT EXISTS chronos2_shadow_claims (observation_id INTEGER PRIMARY KEY, started_at TEXT NOT NULL, observation_sha256 TEXT NOT NULL)")
        for action in ("UPDATE", "DELETE"):
            db.execute(f"CREATE TRIGGER IF NOT EXISTS no_chronos2_claim_{action.lower()} BEFORE {action} ON chronos2_shadow_claims BEGIN SELECT RAISE(ABORT, 'append-only claim'); END")
        db.execute("INSERT INTO chronos2_shadow_claims VALUES (?, ?, ?)", (ident, ai_shadow.now_utc().isoformat(), source_hash))


def _save(path, ident, observation, source_hash, event):
    cutoff, created = ai_shadow._timestamp(observation["cutoff_utc"]), ai_shadow._timestamp(event["created_at_utc"])
    if not ai_shadow._timestamp(observation["created_at_utc"]) <= created < cutoff + timedelta(hours=1):
        raise ValueError("late shadow result")
    if abs((ai_shadow.now_utc() - created).total_seconds()) > 120:
        raise ValueError("shadow result not current")
    with sqlite3.connect(path) as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("PRAGMA application_id").fetchone()[0] != ai_shadow.APP_ID:
            raise ValueError("unrelated database")
        row = db.execute("SELECT payload_json FROM observations WHERE id=?", (ident,)).fetchone()
        claim = db.execute("SELECT observation_sha256 FROM chronos2_shadow_claims WHERE observation_id=?", (ident,)).fetchone()
        if not row or hashlib.sha256(row[0].encode()).hexdigest() != source_hash or not claim or claim[0] != source_hash:
            raise ValueError("source observation or claim changed")
        db.execute("CREATE TABLE IF NOT EXISTS chronos2_shadow_events (observation_id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, payload_json TEXT NOT NULL)")
        for action in ("UPDATE", "DELETE"):
            db.execute(f"CREATE TRIGGER IF NOT EXISTS no_chronos2_event_{action.lower()} BEFORE {action} ON chronos2_shadow_events BEGIN SELECT RAISE(ABORT, 'append-only event'); END")
        db.execute("INSERT INTO chronos2_shadow_events VALUES (?, ?, ?)", (ident, created.isoformat(), json.dumps(event, sort_keys=True, allow_nan=False)))


def valid_output(output, symbols):
    if not isinstance(output, dict) or output.get("model_revision") != REVISION or output.get("weights_sha256") != WEIGHTS or output.get("config_sha256") != CONFIG:
        return False
    if (output.get("chronos_version") != "2.3.2" or output.get("torch_version") != "2.10.0" or output.get("device") != "cpu"
            or output.get("horizon_hours") != 24 or output.get("context_bars") != 300 or output.get("seed") != 0
            or output.get("order_permissions") is not False or set(output.get("outputs") or {}) != set(symbols)):
        return False
    try:
        for result in output["outputs"].values():
            vals = [result[k] for k in ("q10", "median", "q90")]
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in vals) or not vals[0] <= vals[1] <= vals[2]:
                return False
    except (KeyError, TypeError):
        return False
    return True


def run(path=LEDGER):
    path = Path(path)
    capture.verify_policy()
    ident, observation, source_hash = ai_shadow.latest_observation(path, ai_shadow.now_utc())
    pins = source_pins_before(ai_shadow._timestamp(observation["created_at_utc"]))
    if observation.get("capture_script_sha256") != hashlib.sha256((ROOT / "experiments/frozen_policy_capture.py").read_bytes()).hexdigest():
        raise ValueError("capture source mismatch")
    signals = {s: observation["states"][s]["candidate_signal"] for s in ai_shadow.SYMBOLS}
    _claim(path, ident, source_hash)
    active = [s for s in ai_shadow.SYMBOLS if signals[s]]
    result, failure, model_called = None, None, False
    if active:
        try:
            python = default_forecast_python()
            if python is None or ai_shadow.now_utc() >= ai_shadow._timestamp(observation["cutoff_utc"]) + timedelta(minutes=55):
                raise ValueError("model environment or time unavailable")
            series = {s: [float(b["close"]) for b in observation["evidence"][s]["bars"]] for s in active}
            with tempfile.TemporaryDirectory(prefix="chronos2-shadow-") as temp:
                inp, out = Path(temp) / "input.json", Path(temp) / "output.json"
                inp.write_text(json.dumps(series, allow_nan=False), encoding="utf-8")
                env = os.environ.copy()
                env.pop("PIIA_MODEL_API_KEY", None)
                env["HF_HUB_OFFLINE"] = env["TRANSFORMERS_OFFLINE"] = "1"
                model_called = True
                done = subprocess.run([str(python), str(ROOT / "tools/run_chronos2_shadow.py"), "--input", str(inp),
                                       "--output", str(out), "--model", str(MODEL)], capture_output=True,
                                      env=env, timeout=900, check=False)
                if done.returncode or not out.is_file() or out.stat().st_size > 100_000:
                    raise ValueError("model inference failed")
                result = json.loads(out.read_text(encoding="utf-8"))
            if not valid_output(result, active):
                raise ValueError("invalid model output")
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            failure = type(exc).__name__
            result = None
    decisions = {}
    for symbol in ai_shadow.SYMBOLS:
        signal = signals[symbol]
        last = float(observation["evidence"][symbol]["bars"][-1]["close"])
        median = result["outputs"][symbol]["median"] if result and signal else None
        direction = (1 if median > last else -1 if median < last else 0) if median is not None else 0
        decisions[symbol] = {"mechanical_signal": signal, "model_pass": bool(signal and direction == signal)}
    created = ai_shadow.now_utc()
    if source_pins_before(ai_shadow._timestamp(observation["created_at_utc"])) != pins:
        raise ValueError("shadow source changed during run")
    event = {"status": "model_failed" if failure else "completed" if active else "no_candidate",
             "created_at_utc": created.isoformat(), "observation_id": ident, "observation_sha256": source_hash,
             "source_pins": pins, "decisions": decisions, "model_output": result,
             "error_type": failure, "model_process_attempted": model_called,
             "weights_ran_verified": bool(result), "order_permissions": False}
    _save(path, ident, observation, source_hash, event)
    return {"status": event["status"], "observation_id": ident, "model_process_attempted": model_called,
            "weights_ran_verified": bool(result), "order_permissions": False}


if __name__ == "__main__":
    try:
        print(json.dumps(run(), ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
        raise SystemExit(f"Chronos-2 shadow unavailable: {type(exc).__name__}") from exc
