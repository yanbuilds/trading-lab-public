"""Run one owner-requested local research observation, without a scheduler or orders."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile

from piia.intelligence.paths import ROOT


def interpreter() -> Path | None:
    candidates = (ROOT.parent / "freqtrade-env" / "Scripts" / "python.exe",
                  ROOT.parent / "freqtrade-env" / "bin" / "python")
    return next((p for p in candidates if p.is_file()), None)


def observation_id_for_cutoff(cutoff: str) -> int:
    """Resolve the just-created observation, without creating or changing a ledger."""
    ledger = ROOT / "data" / "piia" / "frozen_policy_v1.sqlite"
    if not ledger.is_file() or not isinstance(cutoff, str) or not cutoff:
        raise ValueError("observation unavailable")
    with sqlite3.connect(f"{ledger.resolve().as_uri()}?mode=ro", uri=True) as db:
        if db.execute("PRAGMA application_id").fetchone()[0] != 0x46505631:
            raise ValueError("unrelated database")
        row = db.execute("SELECT id FROM observations WHERE cutoff = ?", (cutoff,)).fetchone()
    if not row or type(row[0]) is not int:
        raise ValueError("observation not found")
    return row[0]


def capture_once() -> dict:
    """Explicit UI action only. Never retries, creates schedules or calls a model API."""
    python = interpreter()
    result = {"status": "unavailable", "order_permissions": False, "scheduled": False,
              "model_api_called": False, "steps": {}}
    if python is None:
        result["reason"] = "freqtrade_environment_missing"
        return result
    env = os.environ.copy()
    env.pop("PIIA_MODEL_API_KEY", None)
    env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    observation_id = None
    for name, module, accepted in (
        ("policy", "experiments.frozen_policy_capture", {"captured_research_only"}),
        ("funding", "experiments.funding_observer", {"complete", "incomplete"}),
        ("chronos", "experiments.chronos_shadow", {"completed", "no_candidate", "model_failed"}),
    ):
        try:
            completed = subprocess.run([str(python), "-m", module], cwd=str(ROOT), env=env,
                                       capture_output=True, text=True, timeout=180, check=False)
        except (OSError, subprocess.TimeoutExpired):
            result["steps"][name] = {"status": "outcome_unknown", "note": "Check the local ledger before any later manual attempt."}
            if name == "policy":
                break
            continue
        if completed.returncode != 0:
            result["steps"][name] = {"status": "unavailable", "exit_code": completed.returncode}
            if name == "policy":
                break
            continue
        try:
            payload = json.loads(completed.stdout)
            if not isinstance(payload, dict) or payload.get("status") not in accepted or payload.get("order_permissions") is not False:
                raise ValueError("unexpected step response")
            if name == "policy":
                observation_id = observation_id_for_cutoff(payload.get("cutoff_utc"))
            elif type(payload.get("observation_id")) is not int or payload["observation_id"] != observation_id:
                result["steps"][name] = {"status": "observation_mismatch", "note": "Check the local ledger; steps referred to different observations."}
                break
            result["steps"][name] = {k: payload[k] for k in ("status", "cutoff_utc", "observation_id", "model_called", "errors") if k in payload}
        except (TypeError, ValueError, OSError, sqlite3.Error):
            result["steps"][name] = {"status": "outcome_unknown", "note": "Check the local ledger before any later manual attempt."}
            if name == "policy":
                break
    if result["steps"].get("policy", {}).get("status") == "captured_research_only":
        result["status"] = "captured_research_only"
    return result


def capture_chronos2_once() -> dict:
    """Explicit separate action; never retries or invokes a paid model API."""
    python = interpreter()
    if python is None:
        return {"status": "unavailable", "reason": "freqtrade_environment_missing", "order_permissions": False}
    env = os.environ.copy()
    env.pop("PIIA_MODEL_API_KEY", None)
    env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    try:
        done = subprocess.run([str(python), "-m", "experiments.chronos2_shadow"], cwd=str(ROOT), env=env,
                              capture_output=True, text=True, timeout=900, check=False)
        if done.returncode:
            return {"status": "unavailable", "exit_code": done.returncode,
                    "note": "Check the local claim and event ledger before attempting again.", "order_permissions": False}
        payload = json.loads(done.stdout)
        if (not isinstance(payload, dict) or payload.get("status") not in {"completed", "no_candidate", "model_failed"}
                or payload.get("order_permissions") is not False):
            raise ValueError("unexpected response")
        return {k: payload.get(k) for k in ("status", "observation_id", "model_process_attempted", "weights_ran_verified", "order_permissions")}
    except (OSError, ValueError, TypeError, subprocess.TimeoutExpired):
        return {"status": "outcome_unknown", "note": "Check the local claim and event ledger before attempting again.",
                "order_permissions": False}


def run_segmented_report() -> dict:
    """Explicit local replay from a consistent ledger snapshot; never calls a model or exchange."""
    python = interpreter()
    if python is None:
        return {"status": "unavailable", "reason": "freqtrade_environment_missing", "order_permissions": False}
    env = os.environ.copy()
    for key in ("PIIA_MODEL_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "XAI_API_KEY"):
        env.pop(key, None)
    env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    try:
        with tempfile.TemporaryDirectory(prefix="segmented-ui-") as temp:
            output = Path(temp) / "report.json"
            done = subprocess.run([str(python), "-m", "experiments.guarded_forecast_report", "--out", str(output)],
                                  cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                  timeout=180, check=False)
            if done.returncode or not output.is_file() or output.stat().st_size > 4_000_000:
                raise ValueError("local report unavailable")
            result = json.loads(output.read_bytes(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError("invalid number")))
        if (not isinstance(result, dict) or result.get("status") != "all_segments_no_aggregate"
                or result.get("aggregate_return") is not None or result.get("order_permissions") is not False
                or not isinstance(result.get("segments"), list)
                or result.get("forecast_baseline_protocol") != "FORECAST_BASELINES_V1"
                or result.get("post_hoc_display_guard", {}).get("status") != "applied"):
            raise ValueError("unexpected report")
        return result
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        return {"status": "unavailable", "reason": "report_not_active_or_replay_failed", "order_permissions": False}


def run_funding_segment_report() -> dict:
    """Explicit comparison using public funding history; never uses a model or an account."""
    python = interpreter()
    if python is None:
        return {"status": "unavailable", "reason": "freqtrade_environment_missing", "order_permissions": False}
    env = os.environ.copy()
    for key in ("PIIA_MODEL_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "XAI_API_KEY"):
        env.pop(key, None)
    env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    try:
        with tempfile.TemporaryDirectory(prefix="funding-segment-ui-") as temp:
            output = Path(temp) / "report.json"
            done = subprocess.run([str(python), "-m", "experiments.funding_segment_comparison", "--out", str(output)],
                                  cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                  timeout=300, check=False)
            if done.returncode or not output.is_file() or output.stat().st_size > 8_000_000:
                raise ValueError("local funding comparison unavailable")
            result = json.loads(output.read_bytes(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError("invalid number")))
        if (not isinstance(result, dict) or result.get("status") != "all_segments_no_aggregate"
                or result.get("aggregate_return") is not None or result.get("order_permissions") is not False
                or not isinstance(result.get("segments"), list)):
            raise ValueError("unexpected report")
        from piia.intelligence.ai_shadow_source_audit import check_sources
        for section in result["segments"]:
            if not isinstance(section, dict):
                raise ValueError("invalid segment")
            value = section.get("result")
            accounts = value.get("accounts") if isinstance(value, dict) else None
            if isinstance(accounts, dict) and "ai_shadow" in accounts:
                if not check_sources()["ok"]:
                    accounts.pop("ai_shadow")
                    value.setdefault("unavailable_arms", {})["ai_shadow"] = "source_audit_failed"
        return result
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        return {"status": "unavailable", "reason": "report_not_active_or_comparison_failed", "order_permissions": False}
