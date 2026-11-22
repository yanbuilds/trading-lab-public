"""Manual prospective Chronos-veto observation; no historical import or orders."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile

from piia.intelligence import ai_shadow
from piia.intelligence.chronos_forecast import MODEL_ID, REVISION, SAFETENSORS_SHA256, run_chronos_tiny

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data/piia/frozen_policy_v1.sqlite"
PINS = {
    "CHRONOS_SHADOW_V1.md": "9a18c85d57d1eab9ef496a788aaad9357a4c05fbba6435149dae0356a805ff28",
    "src/piia/intelligence/chronos_forecast.py": "e40ca61f3dc61ec48a0984a0f755f160e074a133ac432b15e4e66c3d9ec780b5",
    "tools/run_chronos_tiny.py": "2abc59a051820bb2ebb47a45a9a3adb279829f0c1671a696efc1d2d561cb71c3",
    "src/piia/intelligence/ai_shadow.py": "d722ccbf05aa52e91824f3c3d45e466c67b861dd4c99d3360fea90cb5cacc354",
    "src/piia/intelligence/evidence_gates.py": "f692810c982662a3a75e9a40f0b2db95152991a38e4c1feb97d5460cd2458204",
    "experiments/frozen_policy_capture.py": "8c3be28f3f9e75caa47a6762bb7fd1f7fd0106addcd5e6b99a23d1e4eb9075e6",
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_pins():
    for name, digest in PINS.items():
        if _sha(ROOT / name) != digest:
            raise ValueError(f"Chronos shadow source changed: {name}")


def verify_committed_before(observed_at):
    """Require current pinned files to match HEAD before the prospective observation.

    Git metadata and local clocks can be altered; this is not independent proof.
    """
    names = [*PINS, "experiments/chronos_shadow.py", "experiments/prospective_paper_report.py"]
    try:
        for name in names:
            result = subprocess.run(["git", "show", f"HEAD:{name}"], cwd=ROOT, capture_output=True, timeout=15, check=True)
            if hashlib.sha256(result.stdout.replace(b"\r\n", b"\n")).hexdigest() != _sha(ROOT / name):
                raise ValueError(f"uncommitted Chronos shadow source: {name}")
        log = subprocess.run(["git", "log", "-1", "--format=%ct", "HEAD", "--", *names],
                             cwd=ROOT, capture_output=True, timeout=15, check=True)
        committed = datetime.fromtimestamp(int(log.stdout.strip()), timezone.utc)
    except (OSError, subprocess.SubprocessError, TypeError, ValueError) as exc:
        raise ValueError("Chronos shadow commit cannot be verified") from exc
    if observed_at.tzinfo is None or observed_at.utcoffset() is None or committed > observed_at:
        raise ValueError("Chronos shadow version was not committed before observation")


def _write_claim(path, ident, observation_sha):
    with sqlite3.connect(path) as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("PRAGMA application_id").fetchone()[0] != ai_shadow.APP_ID:
            raise ValueError("unrelated database")
        row = db.execute("SELECT payload_json FROM observations WHERE id=?", (ident,)).fetchone()
        if not row or hashlib.sha256(row[0].encode("utf-8")).hexdigest() != observation_sha:
            raise ValueError("source observation changed")
        db.execute("CREATE TABLE IF NOT EXISTS chronos_shadow_claims (observation_id INTEGER PRIMARY KEY, "
                   "started_at TEXT NOT NULL, observation_sha256 TEXT NOT NULL)")
        for action in ("UPDATE", "DELETE"):
            db.execute(f"CREATE TRIGGER IF NOT EXISTS no_chronos_claim_{action.lower()} BEFORE {action} ON chronos_shadow_claims "
                       "BEGIN SELECT RAISE(ABORT, 'append-only Chronos claim'); END")
        db.execute("INSERT INTO chronos_shadow_claims VALUES (?, ?, ?)", (ident, ai_shadow.now_utc().isoformat(), observation_sha))


def _save(path, ident, observation, observation_sha, result):
    cutoff = ai_shadow._timestamp(observation["cutoff_utc"])
    created = ai_shadow._timestamp(result["created_at_utc"])
    if not ai_shadow._timestamp(observation["created_at_utc"]) <= created < cutoff + timedelta(hours=1):
        raise ValueError("Chronos result arrived too late")
    if abs((ai_shadow.now_utc() - created).total_seconds()) > 120:
        raise ValueError("Chronos result is not current")
    with sqlite3.connect(path) as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("PRAGMA application_id").fetchone()[0] != ai_shadow.APP_ID:
            raise ValueError("unrelated database")
        row = db.execute("SELECT payload_json FROM observations WHERE id=?", (ident,)).fetchone()
        claim = db.execute("SELECT observation_sha256 FROM chronos_shadow_claims WHERE observation_id=?", (ident,)).fetchone()
        if not row or hashlib.sha256(row[0].encode("utf-8")).hexdigest() != observation_sha or not claim or claim[0] != observation_sha:
            raise ValueError("source observation or claim changed")
        db.execute("CREATE TABLE IF NOT EXISTS chronos_shadow_events (observation_id INTEGER PRIMARY KEY, "
                   "created_at TEXT NOT NULL, payload_json TEXT NOT NULL)")
        for action in ("UPDATE", "DELETE"):
            db.execute(f"CREATE TRIGGER IF NOT EXISTS no_chronos_event_{action.lower()} BEFORE {action} ON chronos_shadow_events "
                       "BEGIN SELECT RAISE(ABORT, 'append-only Chronos event'); END")
        db.execute("INSERT INTO chronos_shadow_events VALUES (?, ?, ?)",
                   (ident, created.isoformat(), json.dumps(result, sort_keys=True, allow_nan=False)))


def validate_forecast(output, symbol, cutoff, last_close):
    """Validate exact model, data endpoint and 24h median before a decision."""
    from piia.intelligence.evidence_gates import _valid_forecast
    item = (output.get("forecasts") or {}).get(symbol) or {}
    instrument = {"last_closed_close_at_utc": cutoff,
                  "indicators_prefix_only": {"last_close": last_close}}
    return bool(output.get("ok") is True and output.get("weights_ran") is True
                and output.get("model_id") == MODEL_ID and output.get("revision") == REVISION
                and (output.get("snapshot") or {}).get("safetensors_sha256") == SAFETENSORS_SHA256
                and output.get("library_version") == "2.3.2" and output.get("torch_version") == "2.10.0"
                and output.get("device") == "cpu" and output.get("seed") == 0
                and output.get("prediction_length") == 24 and output.get("num_samples") == 20 and output.get("context_limit") == 300
                and item.get("bars_used") == 300 and item.get("prediction_length") == 24 and item.get("num_samples") == 20
                and _valid_forecast(item, instrument))


def run(path=LEDGER):
    from experiments import frozen_policy_capture
    frozen_policy_capture.verify_policy()
    verify_pins()
    own_sha = _sha(__file__)
    ident, observation, observation_sha = ai_shadow.latest_observation(Path(path), ai_shadow.now_utc())
    verify_committed_before(ai_shadow._timestamp(observation["created_at_utc"]))
    if observation.get("capture_script_sha256") != hashlib.sha256((ROOT / "experiments/frozen_policy_capture.py").read_bytes()).hexdigest():
        raise ValueError("observation capture source mismatch")
    signals = {s: observation["states"][s]["candidate_signal"] for s in ai_shadow.SYMBOLS}
    _write_claim(path, ident, observation_sha)
    outputs, decisions = {}, {}
    with tempfile.TemporaryDirectory(prefix="chronos-shadow-") as temp:
        for symbol in ai_shadow.SYMBOLS:
            signal = signals[symbol]
            if not signal:
                decisions[symbol] = {"mechanical_signal": 0, "model_pass": False}
                continue
            try:
                if ai_shadow.now_utc() >= ai_shadow._timestamp(observation["cutoff_utc"]) + timedelta(minutes=55):
                    raise ValueError("insufficient time for model")
                closes = [float(b["close"]) for b in observation["evidence"][symbol]["bars"]]
                series = {symbol: {"closes": closes, "last_close_at_utc": observation["cutoff_utc"], "timeframe": "1H_closed"}}
                output = run_chronos_tiny(series, Path(temp) / f"{symbol}.json", prediction_length=24,
                                          num_samples=20, context_limit=300, seed=0, allow_download=False)
                if not validate_forecast(output, symbol, observation["cutoff_utc"], closes[-1]):
                    raise ValueError("invalid model output")
                median = output["forecasts"][symbol]["quantile_50"][-1]
                direction = 1 if median > closes[-1] else -1 if median < closes[-1] else 0
                decisions[symbol] = {"mechanical_signal": signal, "model_pass": direction == signal}
                outputs[symbol] = output
            except Exception as exc:
                decisions[symbol] = {"mechanical_signal": signal, "model_pass": False}
                outputs[symbol] = {"ok": False, "error_type": type(exc).__name__}
    created = ai_shadow.now_utc()
    verify_pins()
    frozen_policy_capture.verify_policy()
    if _sha(__file__) != own_sha:
        raise ValueError("Chronos shadow code changed during run")
    failed = any(not output.get("ok") for output in outputs.values())
    result = {"status": "model_failed" if failed else "completed" if outputs else "no_candidate",
              "created_at_utc": created.isoformat(), "observation_id": ident, "observation_sha256": observation_sha,
              "protocol_sha256": PINS["CHRONOS_SHADOW_V1.md"], "shadow_code_sha256": own_sha,
              "source_pins": PINS, "decisions": decisions, "model_outputs": outputs, "order_permissions": False}
    _save(path, ident, observation, observation_sha, result)
    return {"status": result["status"], "observation_id": ident, "model_called": bool(outputs), "order_permissions": False}


if __name__ == "__main__":
    try:
        print(json.dumps(run(), indent=2))
    except (ValueError, KeyError, TypeError, OSError, sqlite3.Error) as exc:
        raise SystemExit(f"Chronos shadow unavailable: {type(exc).__name__}: {exc}") from exc
