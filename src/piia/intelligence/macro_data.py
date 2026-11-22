"""On-demand, date-vintage ALFRED context for current research only.

This does not establish intraday release times and is not a historical replay
data source. Only fixed public series and the ALFRED host are permitted.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import math
from pathlib import Path
from urllib.parse import urlencode

import httpx

SERIES = {
    "DGS10": "10年期美国国债收益率",
    "DGS2": "2年期美国国债收益率",
    "DFII10": "10年期通胀指数国债实际收益率",
    "DFF": "联邦基金有效利率",
}
# Official FRED/ALFRED public series for dollar and liquidity context.
# DTWEXBGS is the Fed broad trade-weighted dollar index, not the ICE DXY.
EXTENDED_SERIES = {
    "DTWEXBGS": {
        "label": "美联储广义贸易加权美元指数",
        "unit": "index",
        "topic": "dollar",
        "note": "Official Fed broad dollar index (DTWEXBGS). This is not the ICE U.S. Dollar Index (DXY).",
    },
    "RRPONTSYD": {
        "label": "隔夜逆回购协议余额",
        "unit": "billions_of_dollars",
        "topic": "liquidity",
        "note": "Overnight reverse repurchase agreements; official FRED series RRPONTSYD.",
    },
    "WDTGAL": {
        "label": "美国财政部一般账户（TGA）",
        "unit": "millions_of_dollars",
        "topic": "liquidity",
        "note": "Treasury General Account, weekly official series WDTGAL.",
    },
    "WALCL": {
        "label": "美联储总资产",
        "unit": "millions_of_dollars",
        "topic": "liquidity",
        "note": "Federal Reserve total assets, weekly official series WALCL.",
    },
}
SERIES_UNITS = {name: "percent_per_annum" for name in SERIES}
SERIES_UNITS.update({name: meta["unit"] for name, meta in EXTENDED_SERIES.items()})
SERIES_LABELS = dict(SERIES)
SERIES_LABELS.update({name: meta["label"] for name, meta in EXTENDED_SERIES.items()})
BASE = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"


def _record_error(out: dict, series: str, exc: Exception) -> None:
    """Record safe diagnostics without provider bodies, headers or exception text."""
    out["errors"][series] = type(exc).__name__
    detail = {"type": type(exc).__name__}
    if isinstance(exc, httpx.HTTPStatusError):
        detail["http_status"] = exc.response.status_code
    out.setdefault("error_details", {})[series] = detail


def parse_vintage_observations(raw: bytes, series: str, vintage_date: str, retrieved_at: datetime) -> list[tuple]:
    """Return dated observations under the same conservative vintage rules."""
    if series not in SERIES_LABELS or retrieved_at.tzinfo is None:
        raise ValueError("invalid series or retrieval timestamp")
    vintage = datetime.strptime(vintage_date, "%Y-%m-%d").date()
    retrieved_date = retrieved_at.astimezone(timezone.utc).date()
    if vintage > retrieved_date - timedelta(days=2):
        raise ValueError("vintage must be at least two UTC calendar days old")
    if len(raw) > 1_000_000:
        raise ValueError("ALFRED response too large")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    value_col = f"{series}_{vintage_date.replace('-', '')}"
    if reader.fieldnames != ["observation_date", value_col]:
        raise ValueError("unexpected ALFRED vintage columns")
    observations = []
    seen = set()
    for row in reader:
        d = datetime.strptime(row["observation_date"], "%Y-%m-%d").date()
        if d > vintage or d in seen:
            raise ValueError("future or duplicate observation")
        seen.add(d)
        value = row[value_col]
        if value in (None, "", "."):
            continue
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("non-finite macro observation")
        observations.append((d, value))
    if not observations:
        raise ValueError("no available observation")
    observations.sort(key=lambda x: x[0])
    return observations


def parse_vintage_csv(raw: bytes, series: str, vintage_date: str, retrieved_at: datetime) -> dict:
    observations = parse_vintage_observations(raw, series, vintage_date, retrieved_at)
    retrieved_date = retrieved_at.astimezone(timezone.utc).date()
    latest_date, latest = observations[-1]
    age = (retrieved_date - latest_date).days
    meta = EXTENDED_SERIES.get(series, {})
    return {"series_id": series, "label": SERIES_LABELS[series], "unit": SERIES_UNITS[series],
            "topic": meta.get("topic") or "rates", "series_note": meta.get("note"),
            "observation_date": latest_date.isoformat(), "value": latest,
            "vintage_date": vintage_date, "retrieved_at_utc": retrieved_at.astimezone(timezone.utc).isoformat(),
            "age_days": age, "stale": age > 14,
            "source_page": f"https://fred.stlouisfed.org/series/{series}",
            "source_csv_sha256": hashlib.sha256(raw).hexdigest(),
            "time_limit": "date-vintage only; intraday publication time not verified; not for historical replay"}


def _save_raw_snapshot(parsed: dict, raw: bytes, snapshot_dir: Path | None) -> None:
    if snapshot_dir is None:
        return
    directory = Path(snapshot_dir)
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{parsed['series_id']}_{parsed['vintage_date']}.csv"
    with (directory / name).open("xb") as handle:
        handle.write(raw)
    parsed["raw_snapshot_file"] = name


def verify_vintage_snapshot(item: dict, snapshot_dir: Path) -> dict:
    """Reparse locally saved bytes; integrity is not independent timing proof."""
    series, vintage = item["series_id"], item["vintage_date"]
    if series not in SERIES_LABELS:
        raise ValueError("unknown series")
    if not isinstance(vintage, str) or datetime.strptime(vintage, "%Y-%m-%d").date().isoformat() != vintage:
        raise ValueError("invalid vintage date")
    name = f"{series}_{vintage}.csv"
    if item.get("raw_snapshot_file") != name:
        raise ValueError("snapshot name mismatch")
    path = Path(snapshot_dir) / name
    if path.stat().st_size > 1_000_000:
        raise ValueError("macro snapshot too large")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != item.get("source_csv_sha256"):
        raise ValueError("macro snapshot hash mismatch")
    actual = parse_vintage_csv(raw, series, vintage, datetime.fromisoformat(item["retrieved_at_utc"]))
    for field in ("series_id", "observation_date", "value", "vintage_date", "age_days", "stale", "unit"):
        if item.get(field) != actual[field]:
            raise ValueError(f"macro snapshot metadata mismatch: {field}")
    return {"ok": True, "independent_timing_verified": False, "historical_replay_supported": False}


def fetch_alfred_context(timeout: float = 12.0, *, now: datetime | None = None,
                         snapshot_dir: Path | None = None) -> dict:
    fixed_time = now is not None
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("timezone-aware time required")
    now = now.astimezone(timezone.utc)
    vintage = (now.date() - timedelta(days=2)).isoformat()
    first = (now.date() - timedelta(days=100)).isoformat()
    out = {"kind": "alfred_conservative_date_vintage", "retrieved_at_utc": now.isoformat(),
           "vintage_date": vintage, "intraday_release_verified": False,
           "historical_replay_supported": False, "series": {}, "errors": {}}
    for series in SERIES:
        params = {"id": series, "cosd": first, "vintage_date": vintage}
        try:
            with httpx.Client(timeout=timeout, follow_redirects=False) as client:
                response = client.get(BASE, params=params)
            response.raise_for_status()
            received_at = now if fixed_time else datetime.now(timezone.utc)
            parsed = parse_vintage_csv(response.content, series, vintage, received_at)
            _save_raw_snapshot(parsed, response.content, snapshot_dir)
            parsed["source_url"] = BASE + "?" + urlencode(params)
            out["series"][series] = parsed
        except (httpx.HTTPError, UnicodeError, ValueError, KeyError, TypeError, OSError) as exc:
            _record_error(out, series, exc)
    out["retrieved_at_utc"] = (now if fixed_time else datetime.now(timezone.utc)).isoformat()
    out["coverage_note"] = "Four series only; other macro data, policy targets and intraday release times remain missing."
    out["complete"] = len(out["series"]) == len(SERIES) and not any(x["stale"] for x in out["series"].values())
    return out


def fetch_alfred_extended_context(timeout: float = 12.0, *, now: datetime | None = None,
                                  snapshot_dir: Path | None = None) -> dict:
    """Current official ALFRED vintages for rates, broad dollar, and liquidity.

    Still date-vintage only. Intraday release times are unverified. Historical
    replay / point-in-time backfill is not supported. Missing series are listed,
    never substituted.
    """
    core = fetch_alfred_context(timeout=timeout, now=now, snapshot_dir=snapshot_dir)
    fixed_time = now is not None
    now = now or datetime.now(timezone.utc)
    now = now.astimezone(timezone.utc)
    vintage = core["vintage_date"]
    first = (now.date() - timedelta(days=120)).isoformat()
    out = dict(core)
    out["kind"] = "alfred_conservative_date_vintage_extended"
    out["dollar_note"] = "DTWEXBGS is the Fed broad dollar index, not ICE DXY."
    out["coverage_note"] = (
        "Includes conservative-date ALFRED vintages for Treasury yields, the "
        "10-year real yield, the effective federal funds rate, the Fed broad "
        "dollar index, overnight reverse repo, the Treasury General Account, "
        "and Fed total assets. Intraday release times are unverified. ICE DXY, "
        "policy-target details, and other macro series remain missing. Not a "
        "historical replay source."
    )
    for series, meta in EXTENDED_SERIES.items():
        params = {"id": series, "cosd": first, "vintage_date": vintage}
        try:
            with httpx.Client(timeout=timeout, follow_redirects=False) as client:
                response = client.get(BASE, params=params)
            response.raise_for_status()
            received_at = now if fixed_time else datetime.now(timezone.utc)
            parsed = parse_vintage_csv(response.content, series, vintage, received_at)
            _save_raw_snapshot(parsed, response.content, snapshot_dir)
            parsed["source_url"] = BASE + "?" + urlencode(params)
            out["series"][series] = parsed
        except (httpx.HTTPError, UnicodeError, ValueError, KeyError, TypeError, OSError) as exc:
            _record_error(out, series, exc)
    out["retrieved_at_utc"] = (now if fixed_time else datetime.now(timezone.utc)).isoformat()
    wanted = set(SERIES) | set(EXTENDED_SERIES)
    out["complete"] = set(out["series"]) == wanted and not any(x["stale"] for x in out["series"].values())
    out["historical_replay_supported"] = False
    out["intraday_release_verified"] = False
    return out


def fetch_alfred_history(series_ids: list[str], timeout: float = 12.0, *, now: datetime | None = None,
                         lookback_days: int = 80) -> dict:
    """Conservative-date ALFRED histories for research features. Not a replay archive."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("timezone-aware time required")
    now = now.astimezone(timezone.utc)
    vintage = (now.date() - timedelta(days=2)).isoformat()
    first = (now.date() - timedelta(days=lookback_days)).isoformat()
    out = {"kind": "alfred_conservative_date_vintage_history",
           "retrieved_at_utc": now.isoformat(), "vintage_date": vintage,
           "intraday_release_verified": False, "historical_replay_supported": False,
           "dollar_note": "DTWEXBGS is the Fed broad dollar index, not ICE DXY.",
           "series": {}, "errors": {}}
    for series in series_ids:
        if series not in SERIES_LABELS:
            out["errors"][series] = "unknown_series"
            continue
        params = {"id": series, "cosd": first, "vintage_date": vintage}
        try:
            with httpx.Client(timeout=timeout, follow_redirects=False) as client:
                response = client.get(BASE, params=params)
            response.raise_for_status()
            observations = parse_vintage_observations(response.content, series, vintage, now)
            out["series"][series] = {
                "series_id": series, "label": SERIES_LABELS[series], "unit": SERIES_UNITS[series],
                "vintage_date": vintage,
                "source_page": f"https://fred.stlouisfed.org/series/{series}",
                "source_url": BASE + "?" + urlencode(params),
                "source_csv_sha256": hashlib.sha256(response.content).hexdigest(),
                "observations": [{"observation_date": d.isoformat(), "value": v} for d, v in observations],
                "time_limit": "date-vintage only; intraday publication time not verified; not for historical replay",
            }
        except (httpx.HTTPError, UnicodeError, ValueError, KeyError, TypeError) as exc:
            _record_error(out, series, exc)
    return out
