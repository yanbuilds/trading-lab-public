"""Current Bitcoin sentiment only; never a historical replay or trade signal."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import httpx

URL = "https://api.alternative.me/fng/?limit=2"


def _observations(raw, now):
    if len(raw) > 100_000:
        raise ValueError("sentiment response too large")
    data = json.loads(raw)["data"]
    if not isinstance(data, list) or not data:
        raise ValueError("missing observations")
    observations = []
    for row in data[:2]:
        raw_value = row["value"]
        if isinstance(raw_value, bool) or not str(raw_value).isdigit():
            raise ValueError("invalid value")
        value = int(raw_value)
        stamp = datetime.fromtimestamp(int(row["timestamp"]), timezone.utc)
        if not 0 <= value <= 100 or stamp > now:
            raise ValueError("out-of-range or future observation")
        if observations and stamp >= datetime.fromisoformat(observations[-1]["observed_at_utc"]):
            raise ValueError("observations not strictly newest first")
        observations.append({"value": value, "classification": str(row["value_classification"]),
                             "observed_at_utc": stamp.isoformat()})
    return observations


def verify_fear_greed_snapshot(item, snapshot_dir, now):
    if item.get("raw_snapshot_file") != "fear_greed.json":
        raise ValueError("sentiment snapshot name mismatch")
    path = Path(snapshot_dir) / "fear_greed.json"
    if path.stat().st_size > 100_000:
        raise ValueError("sentiment snapshot too large")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != item.get("raw_sha256"):
        raise ValueError("sentiment snapshot hash mismatch")
    received = datetime.fromisoformat(item["retrieved_at_utc"])
    if received.tzinfo is None or received.utcoffset() is None or received > now:
        raise ValueError("sentiment retrieval time invalid")
    observations = _observations(raw, received)
    if observations != item.get("observations"):
        raise ValueError("sentiment snapshot content mismatch")
    age = (now - datetime.fromisoformat(observations[0]["observed_at_utc"])).total_seconds()
    if not 0 <= age <= 172800:
        raise ValueError("sentiment observation stale")
    return {"ok": True, "independent_timing_verified": False}


def fetch_fear_greed(*, client=None, now=None, snapshot_dir=None):
    fixed_time = now is not None
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("timezone-aware time required")
    now = now.astimezone(timezone.utc)
    result = {"kind": "alternative_me_bitcoin_fear_greed", "source_url": URL,
              "attribution": "Alternative.me", "retrieved_at_utc": now.isoformat(),
              "usable_for_research": False, "historical_replay_supported": False,
              "note": "Daily Bitcoin sentiment; not independent of price, a calibrated forecast, or a trading signal."}
    own = client is None
    client = client or httpx.Client(timeout=12, follow_redirects=False)
    try:
        response = client.get(URL)
        response.raise_for_status()
        raw = response.content
        observations = _observations(raw, now)
        if snapshot_dir is not None:
            directory = Path(snapshot_dir)
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "fear_greed.json").open("xb") as handle:
                handle.write(raw)
            result.update(raw_snapshot_file="fear_greed.json", raw_sha256=hashlib.sha256(raw).hexdigest())
        age = (now - datetime.fromisoformat(observations[0]["observed_at_utc"])).total_seconds()
        result.update(observations=observations, stale=age > 172800,
                      usable_for_research=age <= 172800)
    except (httpx.HTTPError, ValueError, KeyError, TypeError, OverflowError, OSError) as exc:
        result.update(error=type(exc).__name__, stale=True)
    finally:
        if own:
            client.close()
    result["completed_at_utc"] = (now if fixed_time else datetime.now(timezone.utc)).isoformat()
    return result
