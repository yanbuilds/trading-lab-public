"""Invoke official Chronos-t5-tiny through the isolated forecast environment.

PIIA itself does not import torch. This wrapper only records a forecast when
the runner writes a real output with weights_ran=true. Missing environment,
failed hash checks, or a skipped run are reported as exact blockers.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from piia.intelligence.paths import ROOT

MODEL_ID = "amazon/chronos-t5-tiny"
REVISION = "29d808298f1a62493e7b9a5e08529d0d930fa189"
LICENSE = "apache-2.0"
SAFETENSORS_SHA256 = "eed458758bd8165d80f496a90bcd2cfed9f1bf7d7b08677acd64e0f7d72bdcf2"
RUNNER = ROOT / "tools" / "run_chronos_tiny.py"


def default_forecast_python() -> Path | None:
    override = os.environ.get("PIIA_FORECAST_PYTHON")
    if override:
        path = Path(override)
        return path if path.is_file() else None
    candidates = (ROOT.parent / "forecast-env" / "Scripts" / "python.exe",
                  ROOT.parent / "forecast-env" / "bin" / "python")
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def default_hf_cache() -> Path:
    override = os.environ.get("HF_HOME") or os.environ.get("PIIA_HF_CACHE")
    if override:
        return Path(override)
    return ROOT.parent / "hf-cache"


def env_blocker() -> dict[str, Any] | None:
    python = default_forecast_python()
    if python is None:
        return {
            "ok": False,
            "weights_ran": False,
            "reason": "forecast_env_python_not_found",
            "blocker": (
                "Isolated forecast environment not found. Create "
                f"{ROOT.parent / 'forecast-env'} with CPU torch and "
                "chronos-forecasting==2.3.2, or set PIIA_FORECAST_PYTHON. "
                "No random-walk substitute is used as the neural forecast."
            ),
            "model_id": MODEL_ID,
        }
    if not RUNNER.is_file():
        return {"ok": False, "weights_ran": False, "reason": "chronos_runner_missing",
                "blocker": f"Missing {RUNNER}", "model_id": MODEL_ID}
    return None


def run_chronos_tiny(
    series: dict[str, dict[str, Any]],
    out_path: Path,
    *,
    prediction_length: int = 24,
    num_samples: int = 20,
    context_limit: int = 512,
    seed: int | None = None,
    allow_download: bool = False,
    timeout: int = 180,
) -> dict[str, Any]:
    """Run official Chronos-t5-tiny. Return the runner payload or an exact blocker."""
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError("seed must be a nonnegative 32-bit integer")
    blocked = env_blocker()
    if blocked is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(blocked, indent=2), encoding="utf-8")
        return blocked
    python = default_forecast_python()
    assert python is not None
    work = out_path.parent
    work.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(default_hf_cache()))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "2")
    env["TRANSFORMERS_OFFLINE"] = "0" if allow_download else "1"
    env["HF_HUB_OFFLINE"] = "0" if allow_download else "1"
    # The output must come from this invocation. A preexisting destination
    # could otherwise be mistaken for a successful run after a subprocess error.
    with tempfile.TemporaryDirectory(prefix="chronos-run-", dir=work) as temp_dir:
        request_path, result_path = Path(temp_dir) / "input.json", Path(temp_dir) / "output.json"
        request_path.write_text(json.dumps({
            "prediction_length": prediction_length, "num_samples": num_samples,
            "context_limit": context_limit, "seed": seed, "cache_dir": str(default_hf_cache()), "series": series,
        }, indent=2), encoding="utf-8")
        cmd = [str(python), str(RUNNER), "--series-json", str(request_path), "--out", str(result_path),
               "--cache-dir", str(default_hf_cache())]
        if allow_download:
            cmd.append("--allow-download")
        try:
            completed = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True,
                                       timeout=timeout, check=False)
        except (subprocess.TimeoutExpired, OSError) as exc:
            payload = {"ok": False, "weights_ran": False,
                       "reason": "chronos_inference_timeout" if isinstance(exc, subprocess.TimeoutExpired) else "chronos_runner_start_failed",
                       "blocker": type(exc).__name__, "model_id": MODEL_ID}
        else:
            if completed.returncode != 0:
                payload = {"ok": False, "weights_ran": False, "reason": "chronos_runner_failed",
                           "blocker": "Forecast runner returned a nonzero exit status.",
                           "returncode": completed.returncode, "model_id": MODEL_ID}
                # Preserve only known diagnostic codes; never copy arbitrary stderr or secrets.
                try:
                    failed = json.loads(result_path.read_text(encoding="utf-8"))
                    if isinstance(failed, dict) and failed.get("reason") in {
                        "forecast_env_missing_dependency", "chronos_inference_failed",
                        "refusing_non_cpu_device", "no_symbol_produced_a_real_forecast",
                    }:
                        payload["runner_reason"] = failed["reason"]
                        if failed.get("error") in {"FileNotFoundError", "ImportError", "ValueError", "OSError"}:
                            payload["runner_error_type"] = failed["error"]
                except (OSError, ValueError, TypeError):
                    pass
            elif not result_path.is_file():
                payload = {"ok": False, "weights_ran": False, "reason": "chronos_runner_wrote_no_output",
                           "blocker": "Forecast runner exited without writing a payload",
                           "returncode": completed.returncode, "model_id": MODEL_ID}
            else:
                try:
                    payload = json.loads(result_path.read_text(encoding="utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("invalid runner output")
                except (OSError, UnicodeError, ValueError):
                    payload = {"ok": False, "weights_ran": False, "reason": "chronos_runner_invalid_output",
                               "blocker": "Forecast runner output is not a valid JSON object.", "model_id": MODEL_ID}
    if payload.get("weights_ran") is not True:
        payload["weights_ran"] = False
        payload.pop("forecasts", None)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    payload.setdefault("returncode", completed.returncode)
    if payload.get("ok") is not True:
        payload.update(ok=False, weights_ran=False, reason="chronos_runner_inconsistent_status",
                       blocker="Runner did not report a successful inference.")
    if payload.get("model_id") != MODEL_ID:
        payload.update(ok=False, weights_ran=False, reason="unexpected_model_id",
                       blocker="Runner reported a model other than the pinned official model.")
    if payload.get("weights_ran") and payload.get("revision") != REVISION:
        payload["ok"] = False
        payload["weights_ran"] = False
        payload["reason"] = "unexpected_revision"
        payload["blocker"] = "Runner reported a revision other than the pinned official snapshot."
    if payload.get("weights_ran") and seed is not None and payload.get("seed") != seed:
        payload.update(ok=False, weights_ran=False, reason="unexpected_seed",
                       blocker="Runner did not confirm the requested random seed.")
    if payload.get("weights_ran"):
        digest = (payload.get("snapshot") or {}).get("safetensors_sha256")
        if digest != SAFETENSORS_SHA256:
            payload["ok"] = False
            payload["weights_ran"] = False
            payload["reason"] = "safetensors_sha256_mismatch"
            payload["blocker"] = "Local weights did not match the pinned official sha256."
    if not payload["weights_ran"]:
        payload.pop("forecasts", None)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def describe_capability() -> dict[str, Any]:
    python = default_forecast_python()
    return {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "license": LICENSE,
        "safetensors_sha256": SAFETENSORS_SHA256,
        "forecast_python": str(python) if python else None,
        "hf_cache": str(default_hf_cache()),
        "runner": str(RUNNER),
        "trust_remote_code": False,
        "device_policy": "cpu_only",
        "env_blocker": None if python else env_blocker(),
        "note": (
            "Smallest official permissive Chronos variant (t5-tiny, 8M, Apache-2.0). "
            "Training leakage unknown; forecast calibration unverified; not an alpha proof."
        ),
    }
