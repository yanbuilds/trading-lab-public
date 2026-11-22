"""Reparse saved mempool.space snapshots; local integrity is not source authentication."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from piia.intelligence.onchain_btc import REQUIRED_PARTS, _unix_utc


def verify_onchain_snapshot(context: dict, snapshot_dir: Path) -> None:
    if context.get("usable_for_btc_research") is not True or context.get("complete") is not True:
        raise ValueError("complete on-chain snapshot required")
    hashes, metrics = context.get("raw_sha256"), context.get("metrics")
    if not isinstance(hashes, dict) or set(hashes) != set(REQUIRED_PARTS) or not isinstance(metrics, dict):
        raise ValueError("incomplete on-chain evidence")
    raw = {}
    for name in REQUIRED_PARTS:
        path = Path(snapshot_dir) / f"{name}.bin"
        if path.stat().st_size > 1_000_000:
            raise ValueError("on-chain response too large")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != hashes[name]:
            raise ValueError("on-chain raw hash mismatch")
        raw[name] = content
    def parsed(name):
        return json.loads(raw[name], parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    height = int(raw["tip_height"].decode().strip())
    tip = raw["tip_hash"].decode().strip()
    block, mem, fees, adj, hr = (parsed(name) for name in ("last_block", "mempool", "recommended_fees", "difficulty_adjustment", "hashrate"))
    if len(tip) != 64 or block["id"] != tip or int(block["height"]) != height or height < 0:
        raise ValueError("on-chain tip mismatch")
    received = datetime.fromisoformat(context["retrieved_at_utc"].replace("Z", "+00:00"))
    if received.tzinfo is None or received.utcoffset() is None:
        raise ValueError("timezone required")
    received = received.astimezone(timezone.utc)
    block_time = _unix_utc(block["timestamp"])
    expected = {
        "tip_height": {"value": height}, "tip_hash": {"value": tip},
        "last_block": {"height": height, "id": tip, "timestamp_unix": int(block["timestamp"]),
                       "observation_at_utc": block_time.isoformat(), "tx_count": int(block["tx_count"]),
                       "weight": int(block["weight"]), "age_seconds": (received - block_time).total_seconds()},
        "mempool": {"count": int(mem["count"]), "vsize": int(mem["vsize"]), "total_fee": int(mem["total_fee"])},
        "recommended_fees": {k: int(fees[k]) for k in ("fastestFee", "halfHourFee", "hourFee", "economyFee", "minimumFee")},
        "difficulty_adjustment": {"progress_percent": float(adj["progressPercent"]),
            "difficulty_change_percent": float(adj["difficultyChange"]), "remaining_blocks": int(adj["remainingBlocks"]),
            "next_retarget_height": int(adj["nextRetargetHeight"]),
            "estimated_retarget_at_utc": _unix_utc(adj["estimatedRetargetDate"]).isoformat()},
        "hashrate": {"current_hashrate_hashes_per_second": float(hr["currentHashrate"]),
            "current_hashrate_exahash_per_second_derived": float(hr["currentHashrate"]) / 1e18,
            "current_difficulty": float(hr["currentDifficulty"])},
    }
    for name, values in expected.items():
        actual = metrics.get(name)
        if not isinstance(actual, dict):
            raise ValueError("on-chain metric missing")
        for key, value in values.items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError("nonfinite on-chain metric")
            if actual.get(key) != value or isinstance(actual.get(key), bool):
                raise ValueError("on-chain metric does not match raw response")
    if metrics["mempool"].get("unit_vsize") != "vbytes" or metrics["recommended_fees"].get("unit") != "sat_per_vbyte":
        raise ValueError("on-chain units invalid")
