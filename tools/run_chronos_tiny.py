#!/usr/bin/env python3
"""Run official amazon/chronos-t5-tiny on CPU from a local pinned snapshot.

This script is intended to run inside the isolated forecast-env. It does not
import PIIA, does not use GPU, and refuses trust_remote_code. It never claims
a forecast was produced unless the model actually returned values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

MODEL_ID = "amazon/chronos-t5-tiny"
REVISION = "29d808298f1a62493e7b9a5e08529d0d930fa189"
LICENSE = "apache-2.0"
SAFETENSORS_SHA256 = "eed458758bd8165d80f496a90bcd2cfed9f1bf7d7b08677acd64e0f7d72bdcf2"
MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024
ALLOW_PATTERNS = ["config.json", "generation_config.json", "model.safetensors", "README.md"]


def _fail(reason: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "weights_ran": False, "reason": reason, "model_id": MODEL_ID,
            "revision": REVISION, "license": LICENSE, **extra}


def _snapshot_dir(cache_dir: Path) -> Path:
    return cache_dir / "models--amazon--chronos-t5-tiny" / "snapshots" / REVISION


def _verify_snapshot(path: Path) -> dict[str, Any]:
    weights = path / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError("model.safetensors missing")
    size = weights.stat().st_size
    if size > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"weights exceed {MAX_DOWNLOAD_BYTES} bytes")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != SAFETENSORS_SHA256:
        raise ValueError("safetensors sha256 mismatch")
    license_ok = False
    readme = path / "README.md"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8", errors="replace")
        license_ok = "apache-2.0" in text.lower() or "Apache-2.0" in text
    return {"local_path": str(path), "safetensors_sha256": digest, "safetensors_bytes": size,
            "license_file_mentions_apache_2": license_ok}


def ensure_snapshot(cache_dir: Path, allow_download: bool) -> tuple[Path, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = _snapshot_dir(cache_dir)
    if local.is_dir() and (local / "model.safetensors").is_file():
        return local, {"downloaded": False, **_verify_snapshot(local)}
    if not allow_download:
        raise FileNotFoundError("local snapshot missing and download disabled")
    from huggingface_hub import snapshot_download, model_info
    info = model_info(MODEL_ID, revision=REVISION)
    card = info.cardData or {}
    license_id = (card.get("license") or "").lower()
    if license_id not in {"apache-2.0", "apache-2", "apache2"}:
        raise ValueError(f"unexpected or missing license: {license_id or 'absent'}")
    if info.sha and info.sha != REVISION:
        raise ValueError("remote revision sha mismatch")
    path = Path(snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        cache_dir=str(cache_dir),
        allow_patterns=ALLOW_PATTERNS,
    ))
    verified = _verify_snapshot(path)
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    if total > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"snapshot total {total} exceeds download bound")
    return path, {"downloaded": True, "hub_license": license_id, **verified}


def predict_series(pipeline, closes: list[float], prediction_length: int, num_samples: int) -> dict[str, Any]:
    import torch
    context = torch.tensor([float(x) for x in closes], dtype=torch.float32)
    with torch.inference_mode():
        forecast = pipeline.predict(
            context, prediction_length=prediction_length, num_samples=num_samples,
        )
    # Expected shape: (num_series, num_samples, prediction_length)
    arr = forecast.detach().cpu().float()
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"unexpected forecast shape {tuple(arr.shape)}")
    samples, horizon = int(arr.shape[0]), int(arr.shape[1])
    q10 = arr.quantile(0.1, dim=0).tolist()
    q50 = arr.quantile(0.5, dim=0).tolist()
    q90 = arr.quantile(0.9, dim=0).tolist()
    mean = arr.mean(dim=0).tolist()
    last = float(closes[-1])
    def _ret(values: list[float]) -> list[float] | None:
        if last <= 0:
            return None
        return [float(v / last - 1.0) for v in values]
    return {
        "num_samples": samples,
        "prediction_length": horizon,
        "quantile_10": q10,
        "quantile_50": q50,
        "quantile_90": q90,
        "mean": mean,
        "return_from_last_close_q50": _ret(q50),
        "return_from_last_close_q10": _ret(q10),
        "return_from_last_close_q90": _ret(q90),
        "samples": [[float(x) for x in row] for row in arr.tolist()],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU inference for official Chronos-t5-tiny")
    parser.add_argument("--series-json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    raw = json.loads(Path(args.series_json).read_text(encoding="utf-8"))
    cache = Path(args.cache_dir or raw.get("cache_dir") or "/workspace/invest-desk/hf-cache")
    prediction_length = int(raw.get("prediction_length") or 24)
    num_samples = int(raw.get("num_samples") or 20)
    context_limit = int(raw.get("context_limit") or 512)
    out_path = Path(args.out)
    try:
        import torch
        from chronos import ChronosPipeline
        from importlib.metadata import version
    except ImportError as exc:
        payload = _fail("forecast_env_missing_dependency", error=type(exc).__name__, detail=str(exc))
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 2

    seed = raw.get("seed")
    if seed is not None:
        if type(seed) is not int or not 0 <= seed < 2**32:
            out_path.write_text(json.dumps(_fail("invalid_seed"), indent=2), encoding="utf-8")
            return 2
        torch.manual_seed(seed)

    try:
        local, snap_meta = ensure_snapshot(cache, allow_download=args.allow_download)
        load_t0 = time.time()
        pipeline = ChronosPipeline.from_pretrained(
            str(local),
            device_map="cpu",
            dtype=torch.float32,
            trust_remote_code=False,
            local_files_only=True,
        )
        load_s = time.time() - load_t0
        device = str(next(pipeline.model.model.parameters()).device)
        if not device.startswith("cpu"):
            payload = _fail("refusing_non_cpu_device", device=device)
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return 2
        forecasts: dict[str, Any] = {}
        infer_t0 = time.time()
        for symbol, item in (raw.get("series") or {}).items():
            closes = list(item.get("closes") or [])
            if len(closes) < 32:
                forecasts[symbol] = {"ok": False, "weights_ran": False, "reason": "insufficient_closed_history",
                                     "bars_used": len(closes)}
                continue
            window = closes[-context_limit:]
            if any(not isinstance(x, (int, float)) or x != x for x in window):
                forecasts[symbol] = {"ok": False, "weights_ran": False, "reason": "non_finite_close"}
                continue
            pred = predict_series(pipeline, window, prediction_length, num_samples)
            forecasts[symbol] = {
                "ok": True,
                "weights_ran": True,
                "bars_used": len(window),
                "last_close": float(window[-1]),
                "last_close_at_utc": item.get("last_close_at_utc"),
                "timeframe": item.get("timeframe") or "1H_closed",
                **pred,
            }
        infer_s = time.time() - infer_t0
        payload = {
            "ok": any(v.get("ok") for v in forecasts.values()),
            "weights_ran": any(v.get("weights_ran") for v in forecasts.values()),
            "model_id": MODEL_ID,
            "revision": REVISION,
            "license": LICENSE,
            "library": "chronos-forecasting",
            "library_version": version("chronos-forecasting"),
            "torch_version": version("torch"),
            "device": "cpu",
            "gpu_used": False,
            "trust_remote_code": False,
            "training_leakage": "unknown",
            "calibration_verified": False,
            "alpha_claim": False,
            "seed": seed,
            "snapshot": snap_meta,
            "prediction_length": prediction_length,
            "num_samples": num_samples,
            "context_limit": context_limit,
            "load_seconds": round(load_s, 3),
            "infer_seconds": round(infer_s, 3),
            "elapsed_seconds": round(time.time() - started, 3),
            "forecasts": forecasts,
        }
        if not payload["weights_ran"]:
            payload["reason"] = "no_symbol_produced_a_real_forecast"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 0 if payload["weights_ran"] else 2
    except Exception as exc:
        payload = _fail("chronos_inference_failed", error=type(exc).__name__, detail=str(exc),
                        elapsed_seconds=round(time.time() - started, 3))
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
