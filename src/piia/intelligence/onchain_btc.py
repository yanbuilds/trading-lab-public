"""Public Bitcoin on-chain context from the documented mempool.space REST API.

This is current-research evidence only: no historical backfill, no trading signal,
and no claim of exchange-flow or realized-cap analytics. Completeness requires a
fixed set of documented metrics, not a single field. The project uses the public
HTTP API as a client and does not vendor mempool software.
"""
from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import httpx

BASE = "https://mempool.space"
DOCS = "https://mempool.space/docs/api/rest"
ALLOWED_HOSTS = {"mempool.space"}
BLOCK_STALE_SECONDS = 3 * 3600
REQUIRED_PARTS = (
    "tip_height",
    "tip_hash",
    "last_block",
    "mempool",
    "recommended_fees",
    "difficulty_adjustment",
    "hashrate",
)


def _aware(now: datetime | None) -> datetime:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("timezone-aware time required")
    return now.astimezone(timezone.utc)


def _get(client: httpx.Client, path: str) -> httpx.Response:
    url = BASE + path
    host = urlparse(url).hostname
    if host not in ALLOWED_HOSTS:
        raise ValueError(f"refusing non-allowlisted host {host}")
    response = client.get(url)
    response.raise_for_status()
    return response


def _store(raw_store: dict[str, bytes], name: str, response: httpx.Response) -> httpx.Response:
    raw_store[name] = response.content
    return response


def _unix_utc(value: Any) -> datetime:
    ts = float(value)
    if ts > 10**12:
        ts = ts / 1000.0
    return datetime.fromtimestamp(ts, timezone.utc)


def fetch_btc_onchain(timeout: float = 12.0, *, now: datetime | None = None, client: httpx.Client | None = None, snapshot_dir: Any = None) -> dict[str, Any]:
    """Fetch a conservative, current Bitcoin on-chain snapshot."""
    fixed_time = now is not None
    retrieved = _aware(now)
    out: dict[str, Any] = {
        "kind": "mempool_space_public_rest_current_onchain",
        "asset": "BTC",
        "applies_to": ["BTCUSDT"],
        "not_applicable_to": ["XAUUSDT", "NDX100USDT"],
        "source_host": "mempool.space",
        "source_docs": DOCS,
        "retrieved_at_utc": retrieved.isoformat(),
        "historical_replay_supported": False,
        "alpha_claim": False,
        "metrics": {},
        "errors": {},
        "raw_sha256": {},
        "complete": False,
        "partial": False,
        "stale": False,
    }
    raw_store: dict[str, bytes] = {}
    close_client = False
    if client is None:
        client = httpx.Client(timeout=timeout, follow_redirects=False, headers={"User-Agent": "PIIA-research-lab/0.1"})
        close_client = True
    try:
        height_res = _store(raw_store, "tip_height", _get(client, "/api/blocks/tip/height"))
        out["raw_sha256"]["tip_height"] = hashlib.sha256(height_res.content).hexdigest()
        height = int(height_res.text.strip())
        out["metrics"]["tip_height"] = {
            "value": height, "unit": "blocks", "source_url": BASE + "/api/blocks/tip/height",
            "observation_at_utc": retrieved.isoformat(),
        }
        time.sleep(0.15)
        hash_res = _store(raw_store, "tip_hash", _get(client, "/api/blocks/tip/hash"))
        out["raw_sha256"]["tip_hash"] = hashlib.sha256(hash_res.content).hexdigest()
        tip_hash = hash_res.text.strip()
        if len(tip_hash) != 64:
            raise ValueError("unexpected tip hash")
        out["metrics"]["tip_hash"] = {
            "value": tip_hash, "unit": "block_hash", "source_url": BASE + "/api/blocks/tip/hash",
            "observation_at_utc": retrieved.isoformat(),
        }
        time.sleep(0.15)
        block_res = _store(raw_store, "last_block", _get(client, f"/api/block/{tip_hash}"))
        out["raw_sha256"]["last_block"] = hashlib.sha256(block_res.content).hexdigest()
        block = block_res.json()
        if int(block.get("height")) != height or str(block.get("id")) != tip_hash:
            raise ValueError("tip block does not match tip height/hash")
        block_time = _unix_utc(block["timestamp"])
        out["metrics"]["last_block"] = {
            "height": int(block["height"]),
            "id": str(block["id"]),
            "timestamp_unix": int(block["timestamp"]),
            "observation_at_utc": block_time.isoformat(),
            "tx_count": int(block["tx_count"]),
            "weight": int(block["weight"]),
            "unit": "confirmed_bitcoin_block",
            "source_url": BASE + f"/api/block/{tip_hash}",
            "age_seconds": (retrieved - block_time).total_seconds(),
        }
        time.sleep(0.15)
        mem_res = _store(raw_store, "mempool", _get(client, "/api/mempool"))
        out["raw_sha256"]["mempool"] = hashlib.sha256(mem_res.content).hexdigest()
        mem = mem_res.json()
        out["metrics"]["mempool"] = {
            "count": int(mem["count"]),
            "vsize": int(mem["vsize"]),
            "total_fee": int(mem["total_fee"]),
            "unit_count": "transactions",
            "unit_vsize": "vbytes",
            "unit_total_fee": "satoshis",
            "source_url": BASE + "/api/mempool",
            "observation_at_utc": retrieved.isoformat(),
            "note": "Unconfirmed mempool snapshot at retrieval, not a confirmed-chain flow.",
        }
        time.sleep(0.15)
        fee_res = _store(raw_store, "recommended_fees", _get(client, "/api/v1/fees/recommended"))
        out["raw_sha256"]["recommended_fees"] = hashlib.sha256(fee_res.content).hexdigest()
        fees = fee_res.json()
        out["metrics"]["recommended_fees"] = {
            "fastestFee": int(fees["fastestFee"]),
            "halfHourFee": int(fees["halfHourFee"]),
            "hourFee": int(fees["hourFee"]),
            "economyFee": int(fees["economyFee"]),
            "minimumFee": int(fees["minimumFee"]),
            "unit": "sat_per_vbyte",
            "source_url": BASE + "/api/v1/fees/recommended",
            "observation_at_utc": retrieved.isoformat(),
            "note": "Explorer fee recommendations, not an execution guarantee.",
        }
        time.sleep(0.15)
        adj_res = _store(raw_store, "difficulty_adjustment", _get(client, "/api/v1/difficulty-adjustment"))
        out["raw_sha256"]["difficulty_adjustment"] = hashlib.sha256(adj_res.content).hexdigest()
        adj = adj_res.json()
        out["metrics"]["difficulty_adjustment"] = {
            "progress_percent": float(adj["progressPercent"]),
            "difficulty_change_percent": float(adj["difficultyChange"]),
            "remaining_blocks": int(adj["remainingBlocks"]),
            "next_retarget_height": int(adj["nextRetargetHeight"]),
            "estimated_retarget_at_utc": _unix_utc(adj["estimatedRetargetDate"]).isoformat(),
            "source_url": BASE + "/api/v1/difficulty-adjustment",
            "observation_at_utc": retrieved.isoformat(),
        }
        time.sleep(0.15)
        hr_res = _store(raw_store, "hashrate", _get(client, "/api/v1/mining/hashrate/3d"))
        out["raw_sha256"]["hashrate"] = hashlib.sha256(hr_res.content).hexdigest()
        hr = hr_res.json()
        current = float(hr["currentHashrate"])
        difficulty = float(hr["currentDifficulty"])
        out["metrics"]["hashrate"] = {
            "current_hashrate_hashes_per_second": current,
            "current_hashrate_exahash_per_second_derived": current / 1e18,
            "current_difficulty": difficulty,
            "unit_hashrate": "hashes_per_second",
            "unit_difficulty": "dimensionless_difficulty",
            "source_url": BASE + "/api/v1/mining/hashrate/3d",
            "observation_at_utc": retrieved.isoformat(),
            "note": "Current network hashrate and difficulty from the documented 3-day hashrate endpoint. EH/s is derived for readability.",
        }
    except (httpx.HTTPError, UnicodeError, ValueError, KeyError, TypeError, OverflowError) as exc:
        out["errors"]["fetch"] = type(exc).__name__
        out["errors"]["detail"] = str(exc)
    finally:
        if close_client:
            client.close()

    present = [part for part in REQUIRED_PARTS if part in out["metrics"]]
    out["present_parts"] = present
    out["missing_parts"] = [part for part in REQUIRED_PARTS if part not in out["metrics"]]
    last = out["metrics"].get("last_block") or {}
    age = last.get("age_seconds")
    out["stale"] = bool(last) and age is not None and age > BLOCK_STALE_SECONDS
    out["partial"] = bool(out["missing_parts"]) or bool(out["errors"])
    out["complete"] = (not out["partial"]) and (not out["stale"]) and set(present) == set(REQUIRED_PARTS)
    out["coverage_note"] = (
        "Documented mempool.space public REST metrics only: chain tip, latest confirmed block time, "
        "mempool size, recommended fees, difficulty adjustment, and current hashrate/difficulty. "
        "This is not exchange inflow/outflow, realized cap, UTXO age, or miner-reserve data. "
        "One field is not treated as a complete on-chain picture."
    )
    out["usable_for_btc_research"] = out["complete"]
    out["completed_at_utc"] = (retrieved if fixed_time else datetime.now(timezone.utc)).isoformat()
    if snapshot_dir is not None:
        from pathlib import Path
        dest = Path(snapshot_dir)
        dest.mkdir(parents=True, exist_ok=True)
        for name, body in raw_store.items():
            (dest / f"{name}.bin").write_bytes(body)
        (dest / "context.json").write_text(__import__("json").dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    return out
