"""Five-domain public evidence assembler & temporal integrity guard (PIIA Judgment v3).

Reuses the repository's existing public adapters (`public_data`, `macro_data`,
`event_calendar`, `etf_flows`, `fear_greed`, `gold_cftc`, `onchain_btc`,
`polymarket`, `chronos_forecast`, `liquidity_positioning`, `agent_v2.pillars`)
and adds strict source verification, data-tier separation, anti-lookahead
filtering, DXY vs DTWEXBGS distinction, VIX & SEC EDGAR coverage, verified news
ingestion, and forecast-vs-baseline alignment.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
import pandas as pd
import yaml

from piia.intelligence.agent_v2.pillars import SYMBOLS, canonical_sha256
from piia.intelligence.judgment.news import (
    classify_and_filter_news_items,
    fetch_ndx_sec_edgar_filings,
    fetch_official_news_feeds,
    parse_aware_timestamp,
)
from piia.intelligence.settings import model_configured, model_name, model_provider

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_JUDGMENT_CONFIG_PATH = ROOT / "config" / "piia_judgment_v3.yaml"
ALFRED_BASE = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"
STOOQ_DX_URL = "https://stooq.com/q/l/?s=dx.f&f=sd2t2ohlcv&h&e=csv"

DATA_TIERS = (
    "realtime_verified",
    "intraday_delayed_public",
    "latest_daily_vintage",
    "lagged_official_report",
    "proxy_series",
    "scheduled_only",
    "unavailable",
)
FRESHNESS_STATES = (
    "fresh",
    "lagged_usable",
    "stale_excluded",
    "future_excluded",
    "missing",
)


def load_judgment_config(path: Path | None = None) -> dict[str, Any]:
    target = Path(path) if path else DEFAULT_JUDGMENT_CONFIG_PATH
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("protocol_id") != "PIIA_TRACEABLE_JUDGMENT_V3":
        raise ValueError("Invalid piia_judgment_v3.yaml")
    return data


def _ts_utc(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            return None
        return value.astimezone(timezone.utc)
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            return None
        return value.tz_convert("UTC").to_pydatetime()
    return parse_aware_timestamp(value)


def make_atom(
    *,
    evidence_id: str,
    domain: str,
    factor_name: str,
    source_name: str,
    source_url: str | None,
    raw_sha256: str | None,
    data_tier: str,
    observed_at_utc: str | None,
    published_at_utc: str | None,
    available_at_utc: str | None,
    available_at_known: bool,
    retrieved_at_utc: str,
    freshness_status: str,
    applies_to_symbols: list[str],
    applies_to_horizons: list[str],
    value_summary: dict[str, Any],
    direction_by_symbol: dict[str, str],
    causal_driver_group: str,
    is_hard_gate_candidate: bool = False,
    can_decide_direction_alone: bool = False,
    limitation_note: str = "",
) -> dict[str, Any]:
    if data_tier not in DATA_TIERS:
        raise ValueError(f"Unsupported data_tier: {data_tier}")
    if freshness_status not in FRESHNESS_STATES:
        raise ValueError(f"Unsupported freshness_status: {freshness_status}")
    return {
        "evidence_id": evidence_id,
        "domain": domain,
        "factor_name": factor_name,
        "source_name": source_name,
        "source_url": source_url,
        "raw_sha256": raw_sha256,
        "data_tier": data_tier,
        "observed_at_utc": observed_at_utc,
        "published_at_utc": published_at_utc,
        "available_at_utc": available_at_utc,
        "available_at_known": bool(available_at_known),
        "retrieved_at_utc": retrieved_at_utc,
        "freshness_status": freshness_status,
        "applies_to_symbols": list(applies_to_symbols),
        "applies_to_horizons": list(applies_to_horizons),
        "value_summary": value_summary,
        "direction_by_symbol": {s: direction_by_symbol.get(s, "unknown") for s in SYMBOLS},
        "causal_driver_group": causal_driver_group,
        "is_hard_gate_candidate": bool(is_hard_gate_candidate),
        "can_decide_direction_alone": bool(can_decide_direction_alone),
        "role_by_symbol_horizon": {},
        "limitation_note": limitation_note,
    }


def validate_atom_temporal_bounds(
    atom: dict[str, Any],
    *,
    cutoff_utc: datetime,
    max_age_hours: float | None = None,
) -> dict[str, Any]:
    """Enforce zero-lookahead and freshness bounds on an existing EvidenceAtom."""
    out = dict(atom)
    cutoff_utc = cutoff_utc.astimezone(timezone.utc)
    if out["freshness_status"] in {"missing", "future_excluded", "stale_excluded"}:
        return out
    obs_dt = _ts_utc(out.get("observed_at_utc"))
    pub_dt = _ts_utc(out.get("published_at_utc"))
    avail_dt = _ts_utc(out.get("available_at_utc"))
    for stamp in (obs_dt, pub_dt):
        if stamp is not None and stamp > cutoff_utc:
            out["freshness_status"] = "future_excluded"
            out["direction_by_symbol"] = {s: "unknown" for s in SYMBOLS}
            out["limitation_note"] = (out.get("limitation_note") or "") + " [已剔除：观测/发布时间晚于决策截止]"
            return out
    if out.get("available_at_known") and avail_dt is not None and avail_dt > cutoff_utc:
        out["freshness_status"] = "future_excluded"
        out["direction_by_symbol"] = {s: "unknown" for s in SYMBOLS}
        out["limitation_note"] = (out.get("limitation_note") or "") + " [已剔除：可用时间晚于决策截止]"
        return out
    ref = pub_dt or obs_dt
    if max_age_hours is not None and ref is not None:
        if (cutoff_utc - ref).total_seconds() > max_age_hours * 3600.0:
            out["freshness_status"] = "stale_excluded"
            out["direction_by_symbol"] = {s: "unknown" for s in SYMBOLS}
            out["limitation_note"] = (out.get("limitation_note") or "") + f" [已剔除：超过 {max_age_hours:g} 小时有效期]"
    return out


def validate_market_snapshot_coherence(
    market_snapshot: dict[str, Any],
    *,
    cutoff_utc: datetime,
) -> dict[str, Any]:
    """Verify all three contracts share closed 1h cutoff == `cutoff_utc` with no future/gapped bars."""
    cutoff_utc = cutoff_utc.astimezone(timezone.utc)
    one_h = timedelta(hours=1)
    reasons: list[str] = []
    last_close_by_sym: dict[str, str | None] = {}
    frames_1h: dict[str, pd.DataFrame] = {}
    frames_1d: dict[str, pd.DataFrame] = {}

    for sym in SYMBOLS:
        inst = (market_snapshot.get("instruments") or {}).get(sym) or {}
        candles = inst.get("candles_closed_1h") or []
        if not candles:
            reasons.append(f"missing_1h_candles:{sym}")
            last_close_by_sym[sym] = None
            continue
        rows, idx = [], []
        bad = False
        for c in candles:
            try:
                open_dt = _ts_utc(c["open_at_utc"])
                close_dt = _ts_utc(c["close_at_utc"])
                if open_dt is None or close_dt is None or close_dt - open_dt != one_h:
                    bad = True
                    break
                if close_dt > cutoff_utc:
                    reasons.append(f"future_1h_candle:{sym}:{close_dt.isoformat()}>{cutoff_utc.isoformat()}")
                    bad = True
                    break
                idx.append(pd.Timestamp(open_dt))
                rows.append(
                    {
                        "open": float(c["open"]),
                        "high": float(c["high"]),
                        "low": float(c["low"]),
                        "close": float(c["close"]),
                        "volume": float(c.get("volume") or 0.0),
                    }
                )
            except (KeyError, TypeError, ValueError):
                bad = True
                break
        if bad or not rows:
            if f"future_1h_candle" not in ",".join(reasons):
                reasons.append(f"invalid_1h_candles:{sym}")
            last_close_by_sym[sym] = None
            continue
        df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="date")).sort_index()
        if df.index.has_duplicates:
            reasons.append(f"duplicate_1h_candles:{sym}")
        diffs = df.index.to_series().diff().dropna()
        if not diffs.empty and bool((diffs != one_h).any()):
            reasons.append(f"gapped_1h_candles:{sym}")
        last_close = (df.index[-1] + pd.Timedelta(hours=1)).to_pydatetime().astimezone(timezone.utc)
        last_close_by_sym[sym] = last_close.isoformat()
        if last_close < cutoff_utc:
            reasons.append(f"lagging_1h_candle:{sym}:{last_close.isoformat()}<{cutoff_utc.isoformat()}")
        frames_1h[sym] = df

        # Parse closed UTC daily candles <= cutoff_utc when provided
        d_rows, d_idx = [], []
        for dc in inst.get("candles_closed_1d") or []:
            try:
                do = _ts_utc(dc["open_at_utc"])
                dc_close = _ts_utc(dc["close_at_utc"])
                if do is None or dc_close is None or dc_close > cutoff_utc:
                    continue
                d_idx.append(pd.Timestamp(do))
                d_rows.append(
                    {
                        "open": float(dc["open"]),
                        "high": float(dc["high"]),
                        "low": float(dc["low"]),
                        "close": float(dc["close"]),
                        "volume": float(dc.get("volume") or 0.0),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        if d_rows:
            frames_1d[sym] = pd.DataFrame(d_rows, index=pd.DatetimeIndex(d_idx, name="date")).sort_index()

    if len({v for v in last_close_by_sym.values() if v is not None}) > 1:
        reasons.append("misaligned_contract_close_timestamps")

    return {
        "aligned": not reasons and len(frames_1h) == len(SYMBOLS),
        "reasons": reasons,
        "cutoff_utc": cutoff_utc.isoformat(),
        "last_close_by_symbol": last_close_by_sym,
        "frames_1h": frames_1h,
        "frames_1d": frames_1d,
    }


def fetch_alfred_public_series(
    series_ids: list[str],
    *,
    now: datetime,
    lookback_days: int = 120,
    timeout: float = 12.0,
    client: httpx.Client | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Fetch conservative date-vintage ALFRED CSV series (including VIXCLS, NASDAQ100, SP500)."""
    now = now.astimezone(timezone.utc)
    vintage = (now.date() - timedelta(days=2)).isoformat()
    first = (now.date() - timedelta(days=lookback_days)).isoformat()
    series_out: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    own = client is None
    client = client or httpx.Client(timeout=timeout, follow_redirects=False)
    try:
        for sid in series_ids:
            params = {"id": sid, "cosd": first, "vintage_date": vintage}
            url = f"{ALFRED_BASE}?{urlencode(params)}"
            try:
                resp = client.get(ALFRED_BASE, params=params)
                resp.raise_for_status()
                raw = resp.content
                if len(raw) > 1_000_000:
                    raise ValueError("ALFRED response too large")
                reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
                val_col = f"{sid}_{vintage.replace('-', '')}"
                if reader.fieldnames != ["observation_date", val_col]:
                    raise ValueError("unexpected ALFRED columns")
                obs: list[dict[str, Any]] = []
                for row in reader:
                    d_str = row["observation_date"]
                    d_obj = datetime.strptime(d_str, "%Y-%m-%d").date()
                    if d_obj > datetime.strptime(vintage, "%Y-%m-%d").date():
                        raise ValueError("future ALFRED observation")
                    raw_val = row[val_col]
                    if raw_val in (None, "", "."):
                        continue
                    v = float(raw_val)
                    if not math.isfinite(v):
                        continue
                    obs.append({"observation_date": d_str, "value": v})
                obs.sort(key=lambda r: r["observation_date"])
                if not obs:
                    raise ValueError("empty ALFRED series")
                series_out[sid] = {
                    "series_id": sid,
                    "vintage_date": vintage,
                    "source_url": url,
                    "raw_sha256": hashlib.sha256(raw).hexdigest(),
                    "retrieved_at_utc": now.isoformat(),
                    "observations": obs,
                }
            except Exception as exc:
                errors[sid] = type(exc).__name__
    finally:
        if own:
            client.close()
    return series_out, errors


def fetch_public_dxy_proxy(
    *,
    cutoff_utc: datetime,
    max_age_minutes: float = 60.0,
    timeout: float = 6.0,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Optionally query Stooq delayed public DX.F dollar futures; never mislabel as ICE DXY spot."""
    own = client is None
    client = client or httpx.Client(timeout=timeout, follow_redirects=True)
    try:
        resp = client.get(STOOQ_DX_URL)
        resp.raise_for_status()
        raw = resp.content
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        if not rows or rows[0].get("Date") in (None, "N/D") or rows[0].get("Close") in (None, "N/D"):
            return {"ok": False, "reason": "stooq_nd_or_empty"}
        r0 = rows[0]
        date_s = str(r0["Date"]).strip()
        time_s = str(r0.get("Time") or "").strip()
        if not time_s or time_s == "N/D":
            return {"ok": False, "reason": "stooq_missing_intraday_timestamp"}
        # Stooq CET/CEST times cannot be assumed UTC without verified offset -> treat as unverified unless explicit
        px = float(r0["Close"])
        op = float(r0["Open"]) if r0.get("Open") not in (None, "N/D", "") else px
        if not math.isfinite(px) or px <= 0:
            return {"ok": False, "reason": "invalid_price"}
        return {
            "ok": False,
            "reason": "stooq_timezone_unverified_not_ice_spot",
            "raw_close": px,
            "raw_open": op,
            "date": date_s,
            "source_url": STOOQ_DX_URL,
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
        }
    except Exception as exc:
        return {"ok": False, "reason": type(exc).__name__}
    finally:
        if own:
            client.close()


def _series_stats(
    series_entry: dict[str, Any] | None,
    *,
    cutoff_utc: datetime,
    lag_days: int = 2,
    max_age_days: int = 14,
    lookback_obs: int = 5,
) -> dict[str, Any]:
    if not isinstance(series_entry, dict) or not series_entry.get("observations"):
        return {"status": "missing"}
    usable: list[tuple[datetime, float, datetime]] = []
    for row in series_entry["observations"]:
        try:
            d = datetime.strptime(str(row["observation_date"])[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            val = float(row["value"])
            if not math.isfinite(val):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        avail = d + timedelta(days=lag_days)
        if d <= cutoff_utc and avail <= cutoff_utc:
            usable.append((d, val, avail))
    usable.sort(key=lambda x: x[0])
    if not usable:
        return {"status": "missing"}
    last_d, last_v, last_avail = usable[-1]
    age_days = (cutoff_utc.date() - last_d.date()).days
    stale = age_days > max_age_days
    d1 = last_v - usable[-2][1] if len(usable) >= 2 else None
    p1 = ((last_v / usable[-2][1] - 1.0) * 100.0) if len(usable) >= 2 and usable[-2][1] != 0 else None
    d5 = last_v - usable[-(lookback_obs + 1)][1] if len(usable) > lookback_obs else None
    p5 = ((last_v / usable[-(lookback_obs + 1)][1] - 1.0) * 100.0) if len(usable) > lookback_obs and usable[-(lookback_obs + 1)][1] != 0 else None
    return {
        "status": "stale" if stale else "ok",
        "stale": stale,
        "observation_date": last_d.date().isoformat(),
        "published_or_observed_utc": last_d.isoformat(),
        "available_at_utc": last_avail.isoformat(),
        "age_days": age_days,
        "value": round(last_v, 6),
        "delta_1_obs": round(d1, 6) if d1 is not None else None,
        "pct_change_1_obs": round(p1, 4) if p1 is not None else None,
        "delta_5_obs": round(d5, 6) if d5 is not None else None,
        "pct_change_5_obs": round(p5, 4) if p5 is not None else None,
        "source_url": series_entry.get("source_url"),
        "raw_sha256": series_entry.get("raw_sha256"),
        "retrieved_at_utc": series_entry.get("retrieved_at_utc") or cutoff_utc.isoformat(),
    }


def explain_macro_divergence_gold_vs_ndx(macro_summary: dict[str, Any], vix_summary: dict[str, Any]) -> str:
    """Explain why the same macro regime can drive XAUUSDT and NDX100USDT differently."""
    ry_bps = macro_summary.get("real_yield_delta_5obs_bps")
    usd_pct = macro_summary.get("dollar_pct_5obs")
    vix_lvl = vix_summary.get("value")
    parts: list[str] = []
    if ry_bps is not None and usd_pct is not None:
        parts.append(
            f"当前 10Y 美债实际利率 5 观测变化为 {ry_bps:+.1f} bps，美元代理 5 观测变化为 {usd_pct:+.2f}%。"
        )
    parts.append(
        "传导差异机制：(1) 实际利率上行时，对黄金直接抬升无息资产持有机会成本，对纳指 100 则通过提高长久期科技股贴现率承压——二者同向受抑；"
        "(2) 但若发生增长担忧或地缘避险升温（伴随 VIX 走高"
        + (f"，当前 {vix_lvl:.1f}" if isinstance(vix_lvl, (int, float)) else "")
        + " 与美债收益率回落/通胀避险需求上升），黄金会因主权避险与信用对冲获得支撑（偏多），"
        "而纳指 100 却因风险溢价上升与估值压缩承压（偏空），从而出现同源宏观下的多空分化。"
    )
    return " ".join(parts)


def _match_calendar_and_releases(
    event_calendar: dict[str, Any] | None,
    verified_news: list[dict[str, Any]],
    *,
    cutoff_utc: datetime,
    imminent_hours: float,
) -> tuple[list[dict[str, Any]], bool, str | None]:
    events_structured: list[dict[str, Any]] = []
    imminent_veto = False
    imminent_title: str | None = None
    sources = (event_calendar or {}).get("sources") or {}

    for src_name, src_obj in sources.items():
        if not isinstance(src_obj, dict):
            continue
        raw_list = src_obj.get("window_events") or src_obj.get("upcoming_events") or src_obj.get("releases") or []
        for ev in raw_list:
            if not isinstance(ev, dict):
                continue
            title = str(ev.get("title") or "").strip()
            raw_sched = ev.get("scheduled_at_utc") or ev.get("scheduled_at_iso") or ev.get("scheduled_at_eastern_iso")
            sched_dt = _ts_utc(raw_sched) if raw_sched else None
            if sched_dt is None and ev.get("scheduled_local_date"):
                try:
                    sched_dt = datetime.fromisoformat(f"{ev['scheduled_local_date']}T13:30:00+00:00")
                except ValueError:
                    sched_dt = None
            if sched_dt is None:
                continue
            hours_ahead = (sched_dt - cutoff_utc).total_seconds() / 3600.0
            if not -24.0 <= hours_ahead <= 72.0:
                continue
            is_tier1 = bool(
                any(k in title.lower() for k in ("fomc", "cpi", "consumer price", "pce", "employment situation", "gdp"))
            )
            if is_tier1 and 0.0 <= hours_ahead <= imminent_hours:
                imminent_veto = True
                imminent_title = f"{title} ({sched_dt.isoformat()})"

            # Match verified actual release strictly from verified_news <= cutoff_utc
            matched_actual = None
            tok_title = {w for w in title.lower().split() if len(w) >= 3}
            for item in verified_news:
                if item.get("verification_status") != "verified" or item.get("freshness_status") == "future_excluded":
                    continue
                if item.get("news_kind") != "primary_release":
                    continue
                if tok_title and len(tok_title & set(item["title"].lower().split())) >= 1:
                    matched_actual = {
                        "status": "released_verified",
                        "title": item["title"],
                        "published_at_utc": item["published_at_utc"],
                        "source_url": item["url"],
                    }
                    break
            if matched_actual is None:
                matched_actual = {
                    "status": "not_yet_released" if hours_ahead > 0 else "release_unverified",
                    "title": None,
                    "published_at_utc": None,
                    "source_url": None,
                }

            events_structured.append(
                {
                    "agency": ev.get("agency") or src_name,
                    "title": title,
                    "tier1_macro": is_tier1,
                    "schedule": {
                        "scheduled_at_utc": sched_dt.isoformat(),
                        "hours_from_cutoff": round(hours_ahead, 2),
                        "source_url": ev.get("source_url") or src_obj.get("source_url"),
                    },
                    "consensus": {
                        "value": ev.get("consensus"),
                        "status": "supplied" if ev.get("consensus") is not None else "unavailable_in_official_public_feed",
                    },
                    "actual": matched_actual,
                }
            )
    events_structured.sort(key=lambda e: abs(e["schedule"]["hours_from_cutoff"]))
    return events_structured[:10], imminent_veto, imminent_title


def build_macro_evidence(
    *,
    alfred_series: dict[str, dict[str, Any]],
    dxy_live: dict[str, Any] | None,
    event_calendar: dict[str, Any] | None,
    verified_news: list[dict[str, Any]],
    vix_summary: dict[str, Any],
    cutoff_utc: datetime,
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build Domain 1 (Macro) EvidenceAtoms separating ICE DXY, proxies, daily rates, and events."""
    cutoff_iso = cutoff_utc.isoformat()
    fcfg = cfg["freshness_limits"]
    mcfg = cfg["macro_and_tide_thresholds"]
    max_days = int(fcfg["alfred_daily_max_age_days"])
    ry_thresh = float(mcfg["real_yield_5obs_bps"])
    usd_thresh = float(mcfg["broad_dollar_5obs_pct"])
    liq_thresh = float(mcfg["net_liquidity_change_pct"])

    atoms: list[dict[str, Any]] = []

    # 1a. ICE DXY realtime check (never conflate with DTWEXBGS!)
    dxy_dir = {s: "unknown" for s in SYMBOLS}
    if isinstance(dxy_live, dict) and dxy_live.get("ok") and dxy_live.get("is_ice_dxy") is True:
        obs_dt = _ts_utc(dxy_live.get("observed_at_utc"))
        chg = dxy_live.get("change_24h_pct")
        if obs_dt is not None and obs_dt <= cutoff_utc and (cutoff_utc - obs_dt).total_seconds() <= float(fcfg["dxy_intraday_max_age_minutes"]) * 60.0:
            if chg is not None:
                if float(chg) >= float(mcfg["dxy_24h_pct"]):
                    dxy_dir = {s: "bearish" for s in SYMBOLS}
                elif float(chg) <= -float(mcfg["dxy_24h_pct"]):
                    dxy_dir = {s: "bullish" for s in SYMBOLS}
                else:
                    dxy_dir = {s: "neutral" for s in SYMBOLS}
            atoms.append(
                make_atom(
                    evidence_id="macro:ice_dxy",
                    domain="macro",
                    factor_name="ICE DXY 美元指数（实时/延迟公开报价）",
                    source_name=str(dxy_live.get("source_name") or "public_dxy"),
                    source_url=dxy_live.get("source_url"),
                    raw_sha256=dxy_live.get("raw_sha256"),
                    data_tier=str(dxy_live.get("data_tier") or "intraday_delayed_public"),
                    observed_at_utc=obs_dt.isoformat(),
                    published_at_utc=obs_dt.isoformat(),
                    available_at_utc=obs_dt.isoformat(),
                    available_at_known=True,
                    retrieved_at_utc=dxy_live.get("retrieved_at_utc") or cutoff_iso,
                    freshness_status="fresh",
                    applies_to_symbols=list(SYMBOLS),
                    applies_to_horizons=["intraday", "swing_1_3d"],
                    value_summary={"value": dxy_live.get("value"), "change_24h_pct": chg, "is_ice_dxy": True},
                    direction_by_symbol=dxy_dir,
                    causal_driver_group="usd_real_rate_tightening",
                    limitation_note="公开带时间戳 DXY 报价，仍受数据源交易所延迟条款约束",
                )
            )
    if not any(a["evidence_id"] == "macro:ice_dxy" for a in atoms):
        atoms.append(
            make_atom(
                evidence_id="macro:ice_dxy",
                domain="macro",
                factor_name="ICE DXY 美元指数",
                source_name="unverified_or_unavailable",
                source_url=(dxy_live or {}).get("source_url"),
                raw_sha256=None,
                data_tier="unavailable",
                observed_at_utc=None,
                published_at_utc=None,
                available_at_utc=None,
                available_at_known=False,
                retrieved_at_utc=cutoff_iso,
                freshness_status="missing",
                applies_to_symbols=list(SYMBOLS),
                applies_to_horizons=["intraday", "swing_1_3d"],
                value_summary={"status": "unavailable", "reason": (dxy_live or {}).get("reason") or "no_verified_realtime_ice_dxy", "is_ice_dxy": False},
                direction_by_symbol={s: "unknown" for s in SYMBOLS},
                causal_driver_group="usd_real_rate_tightening",
                limitation_note="未获取到带可核验盘中时间戳的 ICE DXY，绝不以广义美元指数冒充实时 DXY",
            )
        )

    # 1b. Fed Broad Dollar Proxy (DTWEXBGS)
    dtwex = _series_stats(alfred_series.get("DTWEXBGS"), cutoff_utc=cutoff_utc, max_age_days=max_days)
    if dtwex["status"] == "ok":
        usd_pct = dtwex.get("pct_change_5_obs")
        usd_dir = {s: "neutral" for s in SYMBOLS}
        if usd_pct is not None:
            if usd_pct >= usd_thresh:
                usd_dir = {s: "bearish" for s in SYMBOLS}
            elif usd_pct <= -usd_thresh:
                usd_dir = {s: "bullish" for s in SYMBOLS}
        atoms.append(
            make_atom(
                evidence_id="macro:broad_dollar_proxy:DTWEXBGS",
                domain="macro",
                factor_name="美联储广义贸易加权美元指数 (DTWEXBGS 代理序列)",
                source_name="ALFRED (St. Louis Fed)",
                source_url=dtwex.get("source_url"),
                raw_sha256=dtwex.get("raw_sha256"),
                data_tier="proxy_series",
                observed_at_utc=dtwex["published_or_observed_utc"],
                published_at_utc=None,
                available_at_utc=dtwex["available_at_utc"],
                available_at_known=True,
                retrieved_at_utc=dtwex["retrieved_at_utc"],
                freshness_status="lagged_usable",
                applies_to_symbols=list(SYMBOLS),
                applies_to_horizons=["swing_1_3d"],
                value_summary={"value": dtwex["value"], "pct_change_5_obs": usd_pct, "observation_date": dtwex["observation_date"], "is_ice_dxy": False},
                direction_by_symbol=usd_dir,
                causal_driver_group="usd_real_rate_tightening",
                limitation_note="DTWEXBGS 为保守+2日滞后贸易加权代理序列，仅作 1-3 日美元背景，非实时 ICE DXY",
            )
        )
    else:
        atoms.append(
            make_atom(
                evidence_id="macro:broad_dollar_proxy:DTWEXBGS",
                domain="macro",
                factor_name="美联储广义美元代理 (DTWEXBGS)",
                source_name="ALFRED (St. Louis Fed)",
                source_url=None,
                raw_sha256=None,
                data_tier="proxy_series",
                observed_at_utc=None,
                published_at_utc=None,
                available_at_utc=None,
                available_at_known=False,
                retrieved_at_utc=cutoff_iso,
                freshness_status="missing" if dtwex["status"] == "missing" else "stale_excluded",
                applies_to_symbols=list(SYMBOLS),
                applies_to_horizons=["swing_1_3d"],
                value_summary={"status": dtwex["status"]},
                direction_by_symbol={s: "unknown" for s in SYMBOLS},
                causal_driver_group="usd_real_rate_tightening",
                limitation_note="广义美元代理序列缺失或过期",
            )
        )

    # 1c. Daily Real Yield (DFII10) & Nominal Rates (DGS10, DGS2, DFF)
    dfii = _series_stats(alfred_series.get("DFII10"), cutoff_utc=cutoff_utc, max_age_days=max_days)
    dgs10 = _series_stats(alfred_series.get("DGS10"), cutoff_utc=cutoff_utc, max_age_days=max_days)
    dgs2 = _series_stats(alfred_series.get("DGS2"), cutoff_utc=cutoff_utc, max_age_days=max_days)
    dff = _series_stats(alfred_series.get("DFF"), cutoff_utc=cutoff_utc, max_age_days=max_days)

    ry_bps = round(dfii["delta_5_obs"] * 100.0, 2) if (dfii["status"] == "ok" and dfii.get("delta_5_obs") is not None) else None
    if dfii["status"] == "ok":
        ry_dir = {s: "neutral" for s in SYMBOLS}
        if ry_bps is not None:
            if ry_bps >= ry_thresh:
                ry_dir = {s: "bearish" for s in SYMBOLS}
            elif ry_bps <= -ry_thresh:
                ry_dir = {s: "bullish" for s in SYMBOLS}
        atoms.append(
            make_atom(
                evidence_id="macro:real_yield_daily:DFII10",
                domain="macro",
                factor_name="10年期 TIPS 美国实际利率 (DFII10 日频版本)",
                source_name="ALFRED (St. Louis Fed)",
                source_url=dfii.get("source_url"),
                raw_sha256=dfii.get("raw_sha256"),
                data_tier="latest_daily_vintage",
                observed_at_utc=dfii["published_or_observed_utc"],
                published_at_utc=None,
                available_at_utc=dfii["available_at_utc"],
                available_at_known=True,
                retrieved_at_utc=dfii["retrieved_at_utc"],
                freshness_status="lagged_usable",
                applies_to_symbols=list(SYMBOLS),
                applies_to_horizons=["swing_1_3d"],
                value_summary={"value_pct": dfii["value"], "delta_5_obs_bps": ry_bps, "observation_date": dfii["observation_date"], "intraday_realtime": False},
                direction_by_symbol=ry_dir,
                causal_driver_group="usd_real_rate_tightening",
                limitation_note="日频最近已定版 TIPS 实际利率（非盘中实时报价，盘中发布具体时刻未验证）",
            )
        )
    else:
        atoms.append(
            make_atom(
                evidence_id="macro:real_yield_daily:DFII10",
                domain="macro",
                factor_name="10年期 TIPS 美国实际利率 (DFII10)",
                source_name="ALFRED (St. Louis Fed)",
                source_url=None,
                raw_sha256=None,
                data_tier="latest_daily_vintage",
                observed_at_utc=None,
                published_at_utc=None,
                available_at_utc=None,
                available_at_known=False,
                retrieved_at_utc=cutoff_iso,
                freshness_status="missing" if dfii["status"] == "missing" else "stale_excluded",
                applies_to_symbols=list(SYMBOLS),
                applies_to_horizons=["swing_1_3d"],
                value_summary={"status": dfii["status"], "intraday_realtime": False},
                direction_by_symbol={s: "unknown" for s in SYMBOLS},
                causal_driver_group="usd_real_rate_tightening",
                limitation_note="实际利率日频序列缺失或过期",
            )
        )

    nominal_ok = dgs10["status"] == "ok" and dgs2["status"] == "ok"
    curve_2s10s = round(dgs10["value"] - dgs2["value"], 4) if nominal_ok else None
    breakeven_10y = round(dgs10["value"] - dfii["value"], 4) if (dgs10["status"] == "ok" and dfii["status"] == "ok") else None
    dgs10_bps = round(dgs10["delta_5_obs"] * 100.0, 2) if (dgs10["status"] == "ok" and dgs10.get("delta_5_obs") is not None) else None
    nom_dir = {s: "neutral" for s in SYMBOLS}
    if dgs10_bps is not None:
        if dgs10_bps >= ry_thresh:
            nom_dir = {"NDX100USDT": "bearish", "XAUUSDT": "bearish", "BTCUSDT": "bearish"}
        elif dgs10_bps <= -ry_thresh:
            nom_dir = {"NDX100USDT": "bullish", "XAUUSDT": "bullish", "BTCUSDT": "bullish"}
    atoms.append(
        make_atom(
            evidence_id="macro:nominal_rates:DGS10_DGS2",
            domain="macro",
            factor_name="美债名义收益率 (DGS10 / DGS2 / 10Y 盈亏平衡通胀)",
            source_name="ALFRED (St. Louis Fed)",
            source_url=dgs10.get("source_url") or dgs2.get("source_url"),
            raw_sha256=dgs10.get("raw_sha256"),
            data_tier="latest_daily_vintage",
            observed_at_utc=dgs10.get("published_or_observed_utc"),
            published_at_utc=None,
            available_at_utc=dgs10.get("available_at_utc"),
            available_at_known=dgs10["status"] == "ok",
            retrieved_at_utc=dgs10.get("retrieved_at_utc") or cutoff_iso,
            freshness_status="lagged_usable" if dgs10["status"] == "ok" else "missing",
            applies_to_symbols=list(SYMBOLS),
            applies_to_horizons=["swing_1_3d"],
            value_summary={
                "dgs10_pct": dgs10.get("value"),
                "dgs2_pct": dgs2.get("value"),
                "dgs10_delta_5obs_bps": dgs10_bps,
                "curve_2s10s_pct": curve_2s10s,
                "breakeven_10y_pct": breakeven_10y,
                "intraday_realtime": False,
            },
            direction_by_symbol=nom_dir,
            causal_driver_group="usd_real_rate_tightening",
            limitation_note="日频美债收益率曲线（同属美元与利率紧缩因果组，与 DFII10/DTWEXBGS 去重不叠加权重）",
        )
    )

    # 1d. Rate Expectations (DGS2 - DFF policy-path spread; CME FedWatch noted unavailable)
    policy_spread = round(dgs2["value"] - dff["value"], 4) if (dgs2["status"] == "ok" and dff["status"] == "ok") else None
    exp_dir = {s: "neutral" for s in SYMBOLS}
    if policy_spread is not None:
        if policy_spread <= -0.35:
            exp_dir = {s: "bullish" for s in SYMBOLS}
        elif policy_spread >= 0.25:
            exp_dir = {s: "bearish" for s in SYMBOLS}
    atoms.append(
        make_atom(
            evidence_id="macro:rate_expectations:DGS2_minus_DFF",
            domain="macro",
            factor_name="利率预期代理 (2Y 美债减联邦基金有效利率 + CME 缺口披露)",
            source_name="ALFRED (St. Louis Fed)",
            source_url=dff.get("source_url") or dgs2.get("source_url"),
            raw_sha256=dff.get("raw_sha256"),
            data_tier="latest_daily_vintage",
            observed_at_utc=dgs2.get("published_or_observed_utc"),
            published_at_utc=None,
            available_at_utc=dgs2.get("available_at_utc"),
            available_at_known=policy_spread is not None,
            retrieved_at_utc=dff.get("retrieved_at_utc") or cutoff_iso,
            freshness_status="lagged_usable" if policy_spread is not None else "missing",
            applies_to_symbols=list(SYMBOLS),
            applies_to_horizons=["swing_1_3d"],
            value_summary={
                "dff_pct": dff.get("value"),
                "dgs2_minus_dff_pct": policy_spread,
                "cme_fedwatch": "unavailable_no_free_verified_api",
            },
            direction_by_symbol=exp_dir,
            causal_driver_group="policy_rate_expectations",
            limitation_note="2Y-DFF 利差反映债券市场定价的降息/加息路径，CME FedWatch 官方工具未公开接入",
        )
    )

    # 1e. Fed Net Liquidity (WALCL - WDTGAL - RRPONTSYD * 1000)
    walcl = _series_stats(alfred_series.get("WALCL"), cutoff_utc=cutoff_utc, max_age_days=21)
    tga = _series_stats(alfred_series.get("WDTGAL"), cutoff_utc=cutoff_utc, max_age_days=21)
    rrp = _series_stats(alfred_series.get("RRPONTSYD"), cutoff_utc=cutoff_utc, max_age_days=max_days)
    net_liq_musd = None
    net_liq_chg_pct = None
    if walcl["status"] == "ok" and tga["status"] == "ok" and rrp["status"] == "ok":
        net_liq_musd = round(walcl["value"] - tga["value"] - rrp["value"] * 1000.0, 2)
        if walcl.get("delta_1_obs") is not None and tga.get("delta_1_obs") is not None and rrp.get("delta_1_obs") is not None:
            prev_liq = net_liq_musd - (walcl["delta_1_obs"] - tga["delta_1_obs"] - rrp["delta_1_obs"] * 1000.0)
            if prev_liq > 0:
                net_liq_chg_pct = round((net_liq_musd / prev_liq - 1.0) * 100.0, 3)
    liq_dir = {s: "neutral" for s in SYMBOLS}
    if net_liq_chg_pct is not None:
        if net_liq_chg_pct >= liq_thresh:
            liq_dir = {s: "bullish" for s in SYMBOLS}
        elif net_liq_chg_pct <= -liq_thresh:
            liq_dir = {s: "bearish" for s in SYMBOLS}
    atoms.append(
        make_atom(
            evidence_id="macro:fed_net_liquidity",
            domain="macro",
            factor_name="美联储美元净流动性 (WALCL - TGA - RRP)",
            source_name="ALFRED (St. Louis Fed)",
            source_url=walcl.get("source_url"),
            raw_sha256=walcl.get("raw_sha256"),
            data_tier="latest_daily_vintage",
            observed_at_utc=walcl.get("published_or_observed_utc"),
            published_at_utc=None,
            available_at_utc=walcl.get("available_at_utc"),
            available_at_known=net_liq_musd is not None,
            retrieved_at_utc=walcl.get("retrieved_at_utc") or cutoff_iso,
            freshness_status="lagged_usable" if net_liq_musd is not None else "missing",
            applies_to_symbols=list(SYMBOLS),
            applies_to_horizons=["swing_1_3d"],
            value_summary={
                "net_liquidity_musd": net_liq_musd,
                "change_pct_recent": net_liq_chg_pct,
                "walcl_musd": walcl.get("value"),
                "tga_musd": tga.get("value"),
                "rrp_busd": rrp.get("value"),
            },
            direction_by_symbol=liq_dir,
            causal_driver_group="fed_dollar_liquidity",
            limitation_note="WALCL 与 TGA 为周度发布、RRP 为日频，保守滞后仅作宏观流动性背景",
        )
    )

    # 1f. Key Economic Events (schedule / consensus / actual separation)
    events_list, imminent_veto, imminent_title = _match_calendar_and_releases(
        event_calendar, verified_news, cutoff_utc=cutoff_utc, imminent_hours=float(fcfg["event_imminent_veto_hours_ahead"])
    )
    cal_avail = isinstance(event_calendar, dict) and bool(event_calendar.get("present_sources"))
    ev_dir = {s: ("two_sided_risk" if imminent_veto else "neutral") for s in SYMBOLS}
    atoms.append(
        make_atom(
            evidence_id="macro:economic_events_calendar",
            domain="macro",
            factor_name="美经济与货币政策日程 (日程 / 预期 / 已公布实际值)",
            source_name="Fed / BLS / BEA Official Calendars",
            source_url="https://www.federalreserve.gov/json/calendar.json",
            raw_sha256=canonical_sha256(events_list) if events_list else None,
            data_tier="scheduled_only",
            observed_at_utc=(event_calendar or {}).get("retrieved_at_utc") or cutoff_iso,
            published_at_utc=(event_calendar or {}).get("retrieved_at_utc") or cutoff_iso,
            available_at_utc=(event_calendar or {}).get("retrieved_at_utc") or cutoff_iso,
            available_at_known=cal_avail,
            retrieved_at_utc=(event_calendar or {}).get("retrieved_at_utc") or cutoff_iso,
            freshness_status="fresh" if cal_avail else "missing",
            applies_to_symbols=list(SYMBOLS),
            applies_to_horizons=["intraday", "swing_1_3d"],
            value_summary={
                "imminent_tier1_veto": imminent_veto,
                "imminent_event_title": imminent_title,
                "window_events": events_list,
                "consensus_coverage": "unavailable_in_official_public_feed",
            },
            direction_by_symbol=ev_dir,
            causal_driver_group="scheduled_macro_event_risk",
            is_hard_gate_candidate=imminent_veto,
            limitation_note="官方日程表仅含预计发布时间，不含市场一致预期值；已公布结果仅绑定截止前官方发布 RSS",
        )
    )

    macro_summary = {
        "ice_dxy_available": any(a["evidence_id"] == "macro:ice_dxy" and a["freshness_status"] == "fresh" for a in atoms),
        "dxy_value": (dxy_live or {}).get("value") if isinstance(dxy_live, dict) and dxy_live.get("is_ice_dxy") else None,
        "broad_dollar_dtwexbgs": dtwex.get("value") if dtwex["status"] == "ok" else None,
        "dollar_pct_5obs": dtwex.get("pct_change_5_obs") if dtwex["status"] == "ok" else None,
        "real_yield_dfii10_pct": dfii.get("value") if dfii["status"] == "ok" else None,
        "real_yield_delta_5obs_bps": ry_bps,
        "dgs10_pct": dgs10.get("value") if dgs10["status"] == "ok" else None,
        "dgs2_pct": dgs2.get("value") if dgs2["status"] == "ok" else None,
        "dff_pct": dff.get("value") if dff["status"] == "ok" else None,
        "curve_2s10s_pct": curve_2s10s,
        "breakeven_10y_pct": breakeven_10y,
        "policy_spread_2y_minus_dff": policy_spread,
        "net_liquidity_musd": net_liq_musd,
        "net_liquidity_change_pct": net_liq_chg_pct,
        "imminent_tier1_event": imminent_title if imminent_veto else None,
        "events_count": len(events_list),
        "events": events_list,
        "gold_vs_ndx_divergence_note": explain_macro_divergence_gold_vs_ndx(
            {"real_yield_delta_5obs_bps": ry_bps, "dollar_pct_5obs": dtwex.get("pct_change_5_obs")},
            vix_summary,
        ),
    }
    return atoms, macro_summary


def build_sentiment_flow_evidence(
    *,
    market_snapshot: dict[str, Any],
    alfred_series: dict[str, dict[str, Any]],
    fear_greed_ctx: dict[str, Any] | None,
    etf_ctx: dict[str, Any] | None,
    gold_cot_ctx: dict[str, Any] | None,
    onchain_ctx: dict[str, Any] | None,
    exchange_netflow_ctx: dict[str, Any] | None,
    gold_etf_ctx: dict[str, Any] | None,
    ndx_edgar_filings: list[dict[str, Any]],
    polymarket_ctx: dict[str, Any] | None,
    cutoff_utc: datetime,
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Build Domain 3 (Sentiment & Capital Flows) atoms across BTC, NDX100, Gold, and Polymarket."""
    cutoff_iso = cutoff_utc.isoformat()
    fcfg = cfg["freshness_limits"]
    mcfg = cfg["macro_and_tide_thresholds"]
    atoms: list[dict[str, Any]] = []

    # 3A. BTC Fear & Greed
    fg_obs = (fear_greed_ctx or {}).get("observations") or []
    fg_ok = bool((fear_greed_ctx or {}).get("usable_for_research") and fg_obs)
    fg_val = fg_obs[0].get("value") if fg_obs else None
    fg_ts = fg_obs[0].get("observed_at_utc") if fg_obs else None
    fg_dir = {s: "unknown" for s in SYMBOLS}
    if fg_ok and fg_val is not None:
        if int(fg_val) <= int(mcfg["fear_greed_low"]):
            fg_dir["BTCUSDT"] = "two_sided_risk"  # extreme fear: capitulation caution, not blind directional vote
        elif int(fg_val) >= int(mcfg["fear_greed_high"]):
            fg_dir["BTCUSDT"] = "bearish"  # crowding caution
        else:
            fg_dir["BTCUSDT"] = "neutral"
    atoms.append(
        validate_atom_temporal_bounds(
            make_atom(
                evidence_id="sentiment:btc:fear_greed",
                domain="sentiment_flow",
                factor_name="加密恐惧贪婪指数 (Alternative.me)",
                source_name="Alternative.me",
                source_url=(fear_greed_ctx or {}).get("source_url"),
                raw_sha256=(fear_greed_ctx or {}).get("raw_sha256"),
                data_tier="lagged_official_report",
                observed_at_utc=fg_ts,
                published_at_utc=fg_ts,
                available_at_utc=fg_ts,
                available_at_known=fg_ok,
                retrieved_at_utc=(fear_greed_ctx or {}).get("retrieved_at_utc") or cutoff_iso,
                freshness_status="lagged_usable" if fg_ok else "missing",
                applies_to_symbols=["BTCUSDT"],
                applies_to_horizons=["intraday", "swing_1_3d"],
                value_summary={"value": fg_val, "classification": fg_obs[0].get("classification") if fg_obs else None},
                direction_by_symbol=fg_dir,
                causal_driver_group="btc_sentiment_crowding",
                limitation_note="日频综合情绪指标，非独立于价格的交易信号",
            ),
            cutoff_utc=cutoff_utc,
            max_age_hours=float(fcfg["fear_greed_max_age_hours"]),
        )
    )

    # 3B. US Spot BTC ETF Flows (dashes '-' NEVER converted to zero!)
    etf_obj = etf_ctx or {}
    latest_row = etf_obj.get("latest_row") or {}
    latest_complete = etf_obj.get("latest_complete_row") or {}
    complete_days = [d for d in (etf_obj.get("complete_daily_flows") or []) if d.get("complete")]
    complete_days.sort(key=lambda d: str(d.get("flow_date") or ""))
    etf_3d_sum = None
    etf_fresh = "missing"
    etf_dir = {s: "unknown" for s in SYMBOLS}
    partial_unreported = bool(etf_obj.get("latest_row_is_partial") or (latest_row and not latest_row.get("complete")))
    if etf_obj.get("usable_for_btc_research") and latest_complete:
        asof_d = str(latest_complete.get("flow_date") or etf_obj.get("asof_complete_flow_date") or "")
        recent_3 = [d for d in complete_days if str(d.get("flow_date") or "") <= cutoff_utc.date().isoformat()][-3:]
        vals_3 = [d.get("sum_of_reported_funds_millions_usd") for d in recent_3 if d.get("sum_of_reported_funds_millions_usd") is not None]
        if len(vals_3) == 3:
            etf_3d_sum = round(sum(float(v) for v in vals_3), 3)
            etf_fresh = "lagged_usable"
            if etf_3d_sum <= float(mcfg["etf_3d_outflow_musd"]):
                etf_dir["BTCUSDT"] = "bearish"
            elif etf_3d_sum >= float(mcfg["etf_3d_inflow_musd"]):
                etf_dir["BTCUSDT"] = "bullish"
            else:
                etf_dir["BTCUSDT"] = "neutral"
    latest_flow_date = latest_complete.get("flow_date") or latest_row.get("flow_date")
    atoms.append(
        make_atom(
            evidence_id="sentiment:btc:spot_etf_flows",
            domain="sentiment_flow",
            factor_name="美国现货 BTC ETF 日净流量 (Farside/TFTC)",
            source_name="Farside Investors / TFTC",
            source_url=etf_obj.get("source_url"),
            raw_sha256=etf_obj.get("raw_sha256"),
            data_tier="lagged_official_report",
            observed_at_utc=f"{latest_flow_date}T21:00:00+00:00" if latest_flow_date else None,
            published_at_utc=f"{latest_flow_date}T21:00:00+00:00" if latest_flow_date else None,
            available_at_utc=f"{latest_flow_date}T21:00:00+00:00" if latest_flow_date else None,
            available_at_known=etf_fresh == "lagged_usable",
            retrieved_at_utc=etf_obj.get("retrieved_at_utc") or cutoff_iso,
            freshness_status=etf_fresh,
            applies_to_symbols=["BTCUSDT"],
            applies_to_horizons=["swing_1_3d"],
            value_summary={
                "usable": etf_fresh == "lagged_usable",
                "latest_row_is_partial": partial_unreported,
                "latest_row_unreported_funds": latest_row.get("unreported_fund_count"),
                "latest_complete_date": latest_complete.get("flow_date"),
                "latest_complete_musd": latest_complete.get("sum_of_reported_funds_millions_usd"),
                "sum_3_complete_days_musd": etf_3d_sum,
                "dashes_treated_as_zero": False,
            },
            direction_by_symbol=etf_dir,
            causal_driver_group="btc_spot_etf_flow",
            limitation_note=(
                "最新交易日存在横线/缺报基金（strictly unknown，绝不填零）；"
                if partial_unreported
                else ""
            ) + "日度披露汇总，非盘中实时申赎流水",
        )
    )

    # 3C. Bitget Perps Funding + Open Interest (all 3 contracts, single-venue scope)
    for sym in SYMBOLS:
        inst = (market_snapshot.get("instruments") or {}).get(sym) or {}
        f_obj = inst.get("funding") or {}
        oi_obj = inst.get("open_interest") or {}
        f_rate = None
        if isinstance(f_obj, dict) and f_obj.get("funding_rate") is not None:
            try:
                f_rate = float(f_obj["funding_rate"])
            except (TypeError, ValueError):
                f_rate = None
        oi_chg = oi_obj.get("change_24h_pct") if isinstance(oi_obj, dict) else None
        oi_ok = oi_chg is not None and math.isfinite(float(oi_chg))
        has_any = f_rate is not None or oi_ok
        sym_dir = {s: "unknown" for s in SYMBOLS}
        if has_any:
            if f_rate is not None and f_rate > float(mcfg["adverse_funding_8h"]) and oi_ok and float(oi_chg) >= float(mcfg["oi_crowding_24h_pct"]):
                sym_dir[sym] = "two_sided_risk"  # crowded long positioning risk
            elif f_rate is not None and f_rate < -float(mcfg["adverse_funding_8h"]) and oi_ok and float(oi_chg) >= float(mcfg["oi_crowding_24h_pct"]):
                sym_dir[sym] = "two_sided_risk"  # crowded short squeeze risk
            else:
                sym_dir[sym] = "neutral"
        atoms.append(
            make_atom(
                evidence_id=f"sentiment:{sym.lower()}:funding_and_oi",
                domain="sentiment_flow",
                factor_name=f"{sym} Bitget 当期资金费率与 24h 持仓量变化",
                source_name="Bitget USDT-FUTURES Public API",
                source_url=oi_obj.get("source_url") or f_obj.get("source_url"),
                raw_sha256=canonical_sha256({"funding": f_obj, "oi": oi_obj}) if has_any else None,
                data_tier="realtime_verified" if has_any else "unavailable",
                observed_at_utc=(oi_obj or {}).get("period_end_utc") or (f_obj or {}).get("retrieved_at_utc"),
                published_at_utc=(oi_obj or {}).get("period_end_utc") or (f_obj or {}).get("retrieved_at_utc"),
                available_at_utc=(oi_obj or {}).get("period_end_utc") or (f_obj or {}).get("retrieved_at_utc"),
                available_at_known=has_any,
                retrieved_at_utc=(f_obj or {}).get("retrieved_at_utc") or cutoff_iso,
                freshness_status="fresh" if has_any else "missing",
                applies_to_symbols=[sym],
                applies_to_horizons=["intraday", "swing_1_3d"],
                value_summary={
                    "funding_rate": f_rate,
                    "oi_24h_change_pct": round(float(oi_chg), 3) if oi_ok else None,
                    "oi_base_units": oi_obj.get("total_open_interest_base_units") if isinstance(oi_obj, dict) else None,
                    "venue_scope": "bitget_single_venue_only",
                },
                direction_by_symbol=sym_dir,
                causal_driver_group=f"perp_positioning_{sym.lower()}",
                limitation_note="仅反映 Bitget 单所永续仓位与当期费率，非全市场合约总持仓",
            )
        )

    # 3D. BTC On-Chain Network Activity vs Exchange Netflow (explicitly separate!)
    oc = onchain_ctx or {}
    oc_metrics = oc.get("metrics") or {}
    last_blk = oc_metrics.get("last_block") or {}
    oc_usable = bool(oc.get("usable_for_btc_research") and last_blk.get("observation_at_utc"))
    fastest_fee = (oc_metrics.get("recommended_fees") or {}).get("fastestFee")
    oc_dir = {s: "unknown" for s in SYMBOLS}
    if oc_usable:
        oc_dir["BTCUSDT"] = "neutral"
    atoms.append(
        validate_atom_temporal_bounds(
            make_atom(
                evidence_id="sentiment:btc:onchain_network_activity",
                domain="sentiment_flow",
                factor_name="BTC 链上网络活跃度与手续费拥堵 (mempool.space，非交易所资金流)",
                source_name="mempool.space Public REST",
                source_url=oc.get("source_docs"),
                raw_sha256=canonical_sha256(oc.get("raw_sha256")) if oc.get("raw_sha256") else None,
                data_tier="realtime_verified" if oc_usable else "unavailable",
                observed_at_utc=last_blk.get("observation_at_utc"),
                published_at_utc=last_blk.get("observation_at_utc"),
                available_at_utc=last_blk.get("observation_at_utc"),
                available_at_known=oc_usable,
                retrieved_at_utc=oc.get("retrieved_at_utc") or cutoff_iso,
                freshness_status="fresh" if oc_usable else "missing",
                applies_to_symbols=["BTCUSDT"],
                applies_to_horizons=["intraday", "swing_1_3d"],
                value_summary={
                    "tip_height": (oc_metrics.get("tip_height") or {}).get("value"),
                    "mempool_tx_count": (oc_metrics.get("mempool") or {}).get("count"),
                    "fastest_fee_sat_vb": fastest_fee,
                    "measure": "network_activity_and_fee_congestion_not_exchange_flow",
                },
                direction_by_symbol=oc_dir,
                causal_driver_group="btc_onchain_network",
                limitation_note="仅体现比特币全网区块高度与内存池拥堵费用，不等于交易所净流入/流出",
            ),
            cutoff_utc=cutoff_utc,
            max_age_hours=float(fcfg["onchain_tip_max_age_hours"]),
        )
    )

    ex_net = exchange_netflow_ctx or {}
    ex_avail = bool(ex_net.get("usable") and ex_net.get("netflow_btc") is not None)
    ex_dir = {s: "unknown" for s in SYMBOLS}
    if ex_avail:
        nf = float(ex_net["netflow_btc"])
        ex_dir["BTCUSDT"] = "bearish" if nf > 0 else ("bullish" if nf < 0 else "neutral")
    atoms.append(
        make_atom(
            evidence_id="sentiment:btc:onchain_exchange_netflow",
            domain="sentiment_flow",
            factor_name="BTC 交易所链上净流入/流出",
            source_name=str(ex_net.get("source_name") or "unavailable_in_free_public_feeds"),
            source_url=ex_net.get("source_url"),
            raw_sha256=ex_net.get("raw_sha256"),
            data_tier="realtime_verified" if ex_avail else "unavailable",
            observed_at_utc=ex_net.get("observed_at_utc"),
            published_at_utc=ex_net.get("observed_at_utc"),
            available_at_utc=ex_net.get("observed_at_utc"),
            available_at_known=ex_avail,
            retrieved_at_utc=cutoff_iso,
            freshness_status="fresh" if ex_avail else "missing",
            applies_to_symbols=["BTCUSDT"],
            applies_to_horizons=["intraday", "swing_1_3d"],
            value_summary={"netflow_btc": ex_net.get("netflow_btc"), "status": "available" if ex_avail else "unavailable"},
            direction_by_symbol=ex_dir,
            causal_driver_group="btc_exchange_onchain_flow",
            limitation_note="当前公开免费数据源不含经核验的交易所钱包地址标签净流数据（保持缺失/未知，不与 mempool 混为一谈）",
        )
    )

    # 3E. Nasdaq-100: VIX (can_decide_direction_alone=False!) + Relative Breadth + SEC EDGAR Events
    vix = _series_stats(alfred_series.get("VIXCLS"), cutoff_utc=cutoff_utc, max_age_days=int(fcfg["alfred_daily_max_age_days"]))
    vix_elevated = False
    vix_dir = {s: "unknown" for s in SYMBOLS}
    if vix["status"] == "ok":
        vix_val = vix["value"]
        vix_pct1 = vix.get("pct_change_1_obs")
        vix_elevated = vix_val >= float(mcfg["vix_high_level"]) or (vix_pct1 is not None and vix_pct1 >= float(mcfg["vix_spike_1d_pct"]))
        vix_dir = {
            "NDX100USDT": "two_sided_risk" if vix_elevated else "neutral",
            "XAUUSDT": "neutral",
            "BTCUSDT": "neutral",
        }
    atoms.append(
        make_atom(
            evidence_id="sentiment:ndx:vix_volatility_backdrop",
            domain="sentiment_flow",
            factor_name="CBOE VIX 波动率背景 (ALFRED VIXCLS，仅作波动背景，不可单定方向)",
            source_name="ALFRED (St. Louis Fed / CBOE)",
            source_url=vix.get("source_url"),
            raw_sha256=vix.get("raw_sha256"),
            data_tier="latest_daily_vintage",
            observed_at_utc=vix.get("published_or_observed_utc"),
            published_at_utc=None,
            available_at_utc=vix.get("available_at_utc"),
            available_at_known=vix["status"] == "ok",
            retrieved_at_utc=vix.get("retrieved_at_utc") or cutoff_iso,
            freshness_status="lagged_usable" if vix["status"] == "ok" else "missing",
            applies_to_symbols=["NDX100USDT", "XAUUSDT"],
            applies_to_horizons=["swing_1_3d"],
            value_summary={
                "value": vix.get("value"),
                "pct_change_1_obs": vix.get("pct_change_1_obs"),
                "elevated_volatility_regime": vix_elevated,
                "observation_date": vix.get("observation_date"),
            },
            direction_by_symbol=vix_dir,
            causal_driver_group="equity_volatility_backdrop",
            can_decide_direction_alone=False,
            limitation_note="VIX 为全市场波动率风险背景，硬规则禁止仅凭 VIX 单独给出纳指做多或做空方向",
        )
    )

    ndx_s = _series_stats(alfred_series.get("NASDAQ100"), cutoff_utc=cutoff_utc, max_age_days=int(fcfg["alfred_daily_max_age_days"]))
    spx_s = _series_stats(alfred_series.get("SP500"), cutoff_utc=cutoff_utc, max_age_days=int(fcfg["alfred_daily_max_age_days"]))
    rel_spread = None
    rel_dir = {s: "unknown" for s in SYMBOLS}
    if ndx_s["status"] == "ok" and spx_s["status"] == "ok" and ndx_s.get("pct_change_5_obs") is not None and spx_s.get("pct_change_5_obs") is not None:
        rel_spread = round(ndx_s["pct_change_5_obs"] - spx_s["pct_change_5_obs"], 3)
        if rel_spread >= float(mcfg["ndx_spx_rel_5obs_pct"]):
            rel_dir["NDX100USDT"] = "bullish"
        elif rel_spread <= -float(mcfg["ndx_spx_rel_5obs_pct"]):
            rel_dir["NDX100USDT"] = "bearish"
        else:
            rel_dir["NDX100USDT"] = "neutral"
    atoms.append(
        make_atom(
            evidence_id="sentiment:ndx:tech_relative_strength",
            domain="sentiment_flow",
            factor_name="科技相对大盘强度 (NASDAQ100 vs SP500 5日差值) 及广度缺口",
            source_name="ALFRED (St. Louis Fed)",
            source_url=ndx_s.get("source_url") or spx_s.get("source_url"),
            raw_sha256=ndx_s.get("raw_sha256"),
            data_tier="latest_daily_vintage",
            observed_at_utc=ndx_s.get("published_or_observed_utc"),
            published_at_utc=None,
            available_at_utc=ndx_s.get("available_at_utc"),
            available_at_known=rel_spread is not None,
            retrieved_at_utc=ndx_s.get("retrieved_at_utc") or cutoff_iso,
            freshness_status="lagged_usable" if rel_spread is not None else "missing",
            applies_to_symbols=["NDX100USDT"],
            applies_to_horizons=["swing_1_3d"],
            value_summary={
                "ndx_5obs_pct": ndx_s.get("pct_change_5_obs"),
                "spx_5obs_pct": spx_s.get("pct_change_5_obs"),
                "ndx_minus_spx_pct": rel_spread,
                "exchange_intraday_advance_decline": "unavailable_no_free_verified_feed",
            },
            direction_by_symbol=rel_dir,
            causal_driver_group="ndx_tech_breadth",
            limitation_note="使用日频 NASDAQ100/SP500 相对表现代理；纳指成分股盘中涨跌家数与成交广度缺失",
        )
    )

    edgar_recent = [f for f in ndx_edgar_filings if f.get("ticker")]
    atoms.append(
        make_atom(
            evidence_id="sentiment:ndx:sec_edgar_constituents",
            domain="sentiment_flow",
            factor_name="纳指核心权重股 SEC EDGAR 重大公告 (8-K / 10-Q / 10-K)",
            source_name="SEC EDGAR Official API",
            source_url=edgar_recent[0]["url"] if edgar_recent else "https://www.sec.gov/edgar/searchedgar/companysearch",
            raw_sha256=canonical_sha256(edgar_recent) if edgar_recent else None,
            data_tier="realtime_verified" if edgar_recent else "unavailable",
            observed_at_utc=edgar_recent[0]["published_at_raw"] if edgar_recent else None,
            published_at_utc=edgar_recent[0]["published_at_raw"] if edgar_recent else None,
            available_at_utc=edgar_recent[0]["published_at_raw"] if edgar_recent else None,
            available_at_known=bool(edgar_recent),
            retrieved_at_utc=cutoff_iso,
            freshness_status="fresh" if edgar_recent else "lagged_usable",
            applies_to_symbols=["NDX100USDT"],
            applies_to_horizons=["intraday", "swing_1_3d"],
            value_summary={"recent_filings_72h": edgar_recent[:6], "count": len(edgar_recent)},
            direction_by_symbol={s: ("neutral" if s == "NDX100USDT" else "unknown") for s in SYMBOLS},
            causal_driver_group="ndx_constituent_filings",
            limitation_note="仅涵盖 NVDA/AAPL/MSFT/AMZN/META/GOOGL 近 72 小时官方 SEC 申报文件（无新文件即说明近 72h 无重大 8-K/季报公告）",
        )
    )

    # 3F. Gold CFTC COT (explicitly lagged futures) + Gold ETF Flows (missing unless verified)
    cot = gold_cot_ctx or {}
    cot_ok = bool(cot.get("usable_for_gold_research") and cot.get("commitments_contracts"))
    cot_comm = cot.get("commitments_contracts") or {}
    net_nc = (int(cot_comm["noncommercial_long"]) - int(cot_comm["noncommercial_short"])) if ("noncommercial_long" in cot_comm and "noncommercial_short" in cot_comm) else None
    cot_dir = {s: "unknown" for s in SYMBOLS}
    if cot_ok and net_nc is not None:
        cot_dir["XAUUSDT"] = "neutral"  # weekly futures positioning background only
    atoms.append(
        make_atom(
            evidence_id="sentiment:xau:cftc_cot_lagged",
            domain="sentiment_flow",
            factor_name="CFTC COMEX 黄金期货 COT 持仓 (周度滞后报告)",
            source_name="U.S. CFTC Public Report",
            source_url=cot.get("source_url"),
            raw_sha256=cot.get("raw_sha256"),
            data_tier="lagged_official_report",
            observed_at_utc=f"{cot['asof_date']}T20:30:00+00:00" if cot.get("asof_date") else None,
            published_at_utc=f"{cot['asof_date']}T20:30:00+00:00" if cot.get("asof_date") else None,
            available_at_utc=None,
            available_at_known=False,
            retrieved_at_utc=cot.get("retrieved_at_utc") or cutoff_iso,
            freshness_status="lagged_usable" if cot_ok else "missing",
            applies_to_symbols=["XAUUSDT"],
            applies_to_horizons=["swing_1_3d"],
            value_summary={
                "asof_date": cot.get("asof_date"),
                "noncommercial_net_contracts": net_nc,
                "open_interest_contracts": cot.get("open_interest_contracts"),
                "is_lagged_weekly_futures": True,
            },
            direction_by_symbol=cot_dir,
            causal_driver_group="gold_cot_positioning",
            can_decide_direction_alone=False,
            limitation_note="COT 为滞后周度 COMEX 期货持仓披露（截至周二、周五发布），不能当作即时买盘或黄金 ETF 资金流",
        )
    )

    getf = gold_etf_ctx or {}
    getf_avail = bool(getf.get("usable") and getf.get("net_tonnage_or_flow") is not None)
    getf_dir = {s: "unknown" for s in SYMBOLS}
    if getf_avail:
        fv = float(getf["net_tonnage_or_flow"])
        getf_dir["XAUUSDT"] = "bullish" if fv > 0 else ("bearish" if fv < 0 else "neutral")
    atoms.append(
        make_atom(
            evidence_id="sentiment:xau:gold_etf_flows",
            domain="sentiment_flow",
            factor_name="可核验黄金 ETF 持仓/资金流向",
            source_name=str(getf.get("source_name") or "unavailable_in_free_public_feeds"),
            source_url=getf.get("source_url"),
            raw_sha256=getf.get("raw_sha256"),
            data_tier="lagged_official_report" if getf_avail else "unavailable",
            observed_at_utc=getf.get("observed_at_utc"),
            published_at_utc=getf.get("observed_at_utc"),
            available_at_utc=getf.get("observed_at_utc"),
            available_at_known=getf_avail,
            retrieved_at_utc=cutoff_iso,
            freshness_status="lagged_usable" if getf_avail else "missing",
            applies_to_symbols=["XAUUSDT"],
            applies_to_horizons=["swing_1_3d"],
            value_summary={"net_flow": getf.get("net_tonnage_or_flow"), "status": "available" if getf_avail else "unavailable"},
            direction_by_symbol=getf_dir,
            causal_driver_group="gold_etf_flow",
            limitation_note="公开免费源未提供逐日带时间戳的实物黄金 ETF（GLD/IAU）份额申赎吨位表，保持未知",
        )
    )

    # 3G. Polymarket (strict topical, spread<=0.05, book age<=5m filter)
    pm = polymarket_ctx or {}
    pm_max_spread = float(fcfg["polymarket_max_spread"])
    pm_max_age_s = float(fcfg["polymarket_book_max_age_minutes"]) * 60.0
    pm_selected: list[dict[str, Any]] = []
    for ev in pm.get("events") or []:
        if not isinstance(ev, dict):
            continue
        rel = [s for s in (ev.get("relevant_instruments") or []) if s in SYMBOLS]
        if not rel:
            continue
        deadline = _ts_utc(ev.get("deadline_utc") or ev.get("end_date"))
        if deadline is not None and deadline < cutoff_utc:
            continue
        for m in ev.get("markets") or []:
            if not isinstance(m, dict):
                continue
            for o in m.get("outcomes") or []:
                if not isinstance(o, dict):
                    continue
                book = o.get("book") or {}
                age_s = book.get("book_age_seconds")
                spread = book.get("spread")
                usable = book.get("usable_for_implied_probability") is True
                if not usable:
                    continue
                if age_s is not None and float(age_s) > pm_max_age_s:
                    continue
                if spread is not None and float(spread) > pm_max_spread:
                    continue
                pm_selected.append(
                    {
                        "event_title": ev.get("title") or ev.get("question"),
                        "question": m.get("question") or ev.get("question"),
                        "outcome_name": o.get("name"),
                        "implied_odds_mid": book.get("midpoint") or o.get("platform_quoted_price"),
                        "change_24h_pp": (o.get("change_24h") or {}).get("change_24h_pp"),
                        "observed_at_utc": book.get("asof_utc") or pm.get("asof_utc"),
                        "spread": spread,
                        "relevant_symbols": rel,
                        "url": ev.get("url") or m.get("source_url"),
                    }
                )
    pm_has = bool(pm_selected)
    atoms.append(
        make_atom(
            evidence_id="sentiment:polymarket:filtered_events",
            domain="sentiment_flow",
            factor_name="Polymarket 直接相关事件赔率 (严格点差<=0.05且5分钟内有效)",
            source_name="Polymarket Gamma + CLOB",
            source_url="https://gamma-api.polymarket.com",
            raw_sha256=canonical_sha256(pm_selected) if pm_has else None,
            data_tier="realtime_verified" if pm_has else "unavailable",
            observed_at_utc=pm_selected[0]["observed_at_utc"] if pm_has else pm.get("asof_utc"),
            published_at_utc=pm_selected[0]["observed_at_utc"] if pm_has else pm.get("asof_utc"),
            available_at_utc=pm_selected[0]["observed_at_utc"] if pm_has else pm.get("asof_utc"),
            available_at_known=pm_has,
            retrieved_at_utc=pm.get("retrieved_at_utc") or cutoff_iso,
            freshness_status="fresh" if pm_has else "missing",
            applies_to_symbols=list(SYMBOLS),
            applies_to_horizons=["intraday", "swing_1_3d"],
            value_summary={
                "selected_count": len(pm_selected),
                "events": pm_selected[:5],
                "is_market_pricing_not_fact": True,
            },
            direction_by_symbol={s: ("neutral" if pm_has else "unknown") for s in SYMBOLS},
            causal_driver_group="polymarket_odds_context",
            can_decide_direction_alone=False,
            limitation_note="赔率属于实时流动性盘口市场定价，不是既定事实或校准概率，仅作背景参考",
        )
    )

    vix_summary = {
        "status": vix["status"],
        "value": vix.get("value") if vix["status"] == "ok" else None,
        "pct_change_1_obs": vix.get("pct_change_1_obs") if vix["status"] == "ok" else None,
        "elevated_volatility_regime": vix_elevated,
        "observation_date": vix.get("observation_date") if vix["status"] == "ok" else None,
    }
    sent_summary = {
        "btc_fear_greed": fg_val if fg_ok else None,
        "btc_etf_3d_musd": etf_3d_sum,
        "btc_etf_partial_unreported": partial_unreported,
        "vix": vix_summary,
        "ndx_minus_spx_5obs_pct": rel_spread,
        "gold_cot_noncomm_net": net_nc if cot_ok else None,
        "polymarket_selected": pm_selected[:5],
    }
    return atoms, vix_summary, sent_summary


def build_forecast_and_model_evidence(
    *,
    frames_1h: dict[str, pd.DataFrame],
    chronos_payload: dict[str, Any] | None,
    llm_opinion: dict[str, Any] | None,
    cutoff_utc: datetime,
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build Domain 5 (Forecasts & Model Provenance): Pinned Neural + Causal Baselines + AI flags."""
    cutoff_iso = cutoff_utc.isoformat()
    fcfg = cfg["macro_and_tide_thresholds"]
    atoms: list[dict[str, Any]] = []
    forecasts_by_symbol: dict[str, Any] = {}
    chronos_raw = chronos_payload or {}
    c_model_id = chronos_raw.get("model_id") or "amazon/chronos-t5-tiny"
    c_rev = chronos_raw.get("revision") or "29d808298f1a62493e7b9a5e08529d0d930fa189"
    c_run_at = _ts_utc(chronos_raw.get("run_at_utc") or chronos_raw.get("recorded_at_utc"))

    for sym in SYMBOLS:
        df = frames_1h.get(sym)
        if df is None or df.empty:
            continue
        last_close = float(df.iloc[-1]["close"])
        closes = df["close"].astype(float)
        atr_approx = float((df["high"].astype(float) - df["low"].astype(float)).tail(14).mean()) or (last_close * 0.005)
        log_ret = (closes / closes.shift(1)).apply(lambda x: math.log(x) if x and x > 0 else float("nan")).dropna()

        # Simple causal baselines side by side
        mom_24 = (last_close / float(closes.iloc[-25]) - 1.0) if len(closes) >= 25 and float(closes.iloc[-25]) > 0 else None
        mom_pred_6h = last_close * (1.0 + mom_24 * (6.0 / 24.0)) if mom_24 is not None else None
        mom_pred_24h = last_close * (1.0 + mom_24) if mom_24 is not None else None

        stat_avail = len(log_ret) >= 168
        stat_6h = stat_24h = stat_q10_24 = stat_q90_24 = None
        if stat_avail:
            tail = log_ret.tail(168)
            mu = float(tail.mean())
            sd = float(tail.std(ddof=1))
            stat_6h = round(last_close * math.exp(mu * 6.0), 4)
            stat_24h = round(last_close * math.exp(mu * 24.0), 4)
            band_24 = 1.96 * sd * math.sqrt(24.0)
            stat_q10_24 = round(last_close * math.exp(mu * 24.0 - band_24), 4)
            stat_q90_24 = round(last_close * math.exp(mu * 24.0 + band_24), 4)

        stat_dir = "neutral" if stat_avail else "unknown"
        if stat_24h is not None:
            delta_atr = (stat_24h - last_close) / atr_approx
            if delta_atr >= float(fcfg["forecast_support_atr_mult"]):
                stat_dir = "bullish"
            elif delta_atr <= -float(fcfg["forecast_opposing_atr_mult"]):
                stat_dir = "bearish"

        atoms.append(
            make_atom(
                evidence_id=f"forecast:{sym.lower()}:statistical_drift_168h",
                domain="forecast",
                factor_name=f"{sym} 168h 因果统计漂移基准 (非神经网络)",
                source_name="Causal Historical Log-Return Baseline",
                source_url=None,
                raw_sha256=canonical_sha256({"sym": sym, "cutoff": cutoff_iso, "stat_24h": stat_24h}),
                data_tier="realtime_verified" if stat_avail else "unavailable",
                observed_at_utc=cutoff_iso,
                published_at_utc=cutoff_iso,
                available_at_utc=cutoff_iso,
                available_at_known=stat_avail,
                retrieved_at_utc=cutoff_iso,
                freshness_status="fresh" if stat_avail else "missing",
                applies_to_symbols=[sym],
                applies_to_horizons=["intraday", "swing_1_3d"],
                value_summary={
                    "model_type": "statistical_drift_168h_not_neural",
                    "last_close": last_close,
                    "horizon_6h_point": stat_6h,
                    "horizon_24h_point": stat_24h,
                    "horizon_24h_q10": stat_q10_24,
                    "horizon_24h_q90": stat_q90_24,
                    "momentum_24h_baseline": round(mom_pred_24h, 4) if mom_pred_24h is not None else None,
                    "naive_flat_baseline": last_close,
                },
                direction_by_symbol={s: (stat_dir if s == sym else "unknown") for s in SYMBOLS},
                causal_driver_group=f"forecast_{sym.lower()}",
                limitation_note="无前视统计随机游走漂移基准（非深度学习模型），未校准置信度",
            )
        )

        # Neural Chronos: MUST match current cutoff & current hour, never borrow an earlier run!
        c_item = (chronos_raw.get("forecasts") or {}).get(sym) if chronos_raw.get("weights_ran") is True else None
        neural_block: dict[str, Any] = {
            "model_id": c_model_id,
            "revision": c_rev,
            "weights_ran": False,
            "status": "not_run",
            "reason": chronos_raw.get("reason") or "forecast_env_or_weights_unavailable",
            "run_at_utc": c_run_at.isoformat() if c_run_at else None,
            "input_cutoff_utc": None,
            "q10_24h": None,
            "q50_24h": None,
            "q90_24h": None,
        }
        neural_dir = "unknown"
        neural_fresh = "missing"
        if isinstance(c_item, dict) and c_item.get("ok") and c_item.get("quantile_50"):
            in_cut = _ts_utc(c_item.get("last_close_at_utc"))
            neural_block["input_cutoff_utc"] = in_cut.isoformat() if in_cut else None
            if in_cut is None or in_cut != cutoff_utc:
                neural_block["status"] = "expired_excluded"
                neural_block["reason"] = "input_cutoff_mismatch_cannot_borrow_prior_forecast"
                neural_fresh = "stale_excluded"
            elif c_run_at is not None and not (cutoff_utc <= c_run_at < cutoff_utc + timedelta(hours=1)):
                neural_block["status"] = "expired_excluded"
                neural_block["reason"] = "run_timestamp_outside_cutoff_hour"
                neural_fresh = "stale_excluded"
            else:
                q10 = float(c_item["quantile_10"][-1])
                q50 = float(c_item["quantile_50"][-1])
                q90 = float(c_item["quantile_90"][-1])
                q50_6h = float(c_item["quantile_50"][5]) if len(c_item["quantile_50"]) >= 6 else None
                neural_block.update(
                    {
                        "weights_ran": True,
                        "status": "executed_verified",
                        "reason": None,
                        "q10_24h": round(q10, 4),
                        "q50_24h": round(q50, 4),
                        "q90_24h": round(q90, 4),
                        "q50_6h": round(q50_6h, 4) if q50_6h is not None else None,
                    }
                )
                neural_fresh = "fresh"
                delta_n = (q50 - last_close) / atr_approx
                if delta_n >= float(fcfg["forecast_support_atr_mult"]):
                    neural_dir = "bullish"
                elif delta_n <= -float(fcfg["forecast_opposing_atr_mult"]):
                    neural_dir = "bearish"
                else:
                    neural_dir = "neutral"

        atoms.append(
            make_atom(
                evidence_id=f"forecast:{sym.lower()}:chronos_neural",
                domain="forecast",
                factor_name=f"{sym} 神经网络预测 ({c_model_id})",
                source_name=f"Chronos ({c_model_id}@{c_rev[:8]})",
                source_url=None,
                raw_sha256=canonical_sha256(neural_block),
                data_tier="realtime_verified" if neural_block["weights_ran"] else "unavailable",
                observed_at_utc=neural_block["input_cutoff_utc"],
                published_at_utc=neural_block["run_at_utc"] or neural_block["input_cutoff_utc"],
                available_at_utc=neural_block["run_at_utc"] or neural_block["input_cutoff_utc"],
                available_at_known=bool(neural_block["weights_ran"]),
                retrieved_at_utc=cutoff_iso,
                freshness_status=neural_fresh,
                applies_to_symbols=[sym],
                applies_to_horizons=["intraday", "swing_1_3d"],
                value_summary=neural_block,
                direction_by_symbol={s: (neural_dir if s == sym else "unknown") for s in SYMBOLS},
                causal_driver_group=f"forecast_{sym.lower()}",
                limitation_note=(
                    "已按当前截止时间实际运行固定版本 Chronos（非校准交易概率）"
                    if neural_block["weights_ran"]
                    else f"模型未用于本次判断（status={neural_block['status']}, reason={neural_block['reason']}），绝不借用较早一次预测"
                ),
            )
        )

        forecasts_by_symbol[sym] = {
            "last_close": last_close,
            "input_cutoff_utc": cutoff_iso,
            "chronos_neural": neural_block,
            "baselines": {
                "naive_flat_q50": last_close,
                "momentum_24h_6h_proxy": round(mom_pred_6h, 4) if mom_pred_6h is not None else None,
                "momentum_24h_24h_proxy": round(mom_pred_24h, 4) if mom_pred_24h is not None else None,
                "statistical_drift_168h_6h": stat_6h,
                "statistical_drift_168h_24h_q50": stat_24h,
                "statistical_drift_168h_24h_q10": stat_q10_24,
                "statistical_drift_168h_24h_q90": stat_q90_24,
            },
        }

    llm_ran = bool(isinstance(llm_opinion, dict) and llm_opinion.get("executed") is True)
    ai_provenance = {
        "llm_research_opinion": (
            llm_opinion
            if llm_ran
            else {
                "executed": False,
                "status": "NOT_RUN" if not model_configured() else "NOT_REQUESTED_IN_DETERMINISTIC_RUN",
                "provider": model_provider() if model_configured() else None,
                "model_name": model_name() if model_configured() else None,
            }
        ),
        "independent_ai_critic": {
            "executed": bool(isinstance(llm_opinion, dict) and llm_opinion.get("independent_critic_executed")),
            "status": "NOT_WIRED",
            "note": "本项目内置反方门控属于确定性规则引擎（deterministic_rule），不是独立大模型审查",
        },
    }
    return atoms, {"by_symbol": forecasts_by_symbol, "ai_provenance": ai_provenance}


def collect_judgment_inputs(
    *,
    symbols: tuple[str, ...] = SYMBOLS,
    include_live_alfred_extras: bool = True,
    now_override: datetime | None = None,
) -> dict[str, Any]:
    """Collect real public evidence across all 5 domains + market bars (never fills missing data)."""
    from piia.intelligence.chronos_forecast import run_chronos_tiny
    from piia.intelligence.etf_flows import fetch_us_spot_btc_etf_flows
    from piia.intelligence.etf_tftc import fetch_tftc_etf
    from piia.intelligence.event_calendar import fetch_official_event_calendar
    from piia.intelligence.fear_greed import fetch_fear_greed
    from piia.intelligence.gold_cftc import fetch_comex_gold_positioning
    from piia.intelligence.onchain_btc import fetch_btc_onchain
    from piia.intelligence.polymarket import fetch_prediction_markets
    from piia.intelligence.public_data import fetch_public_snapshot
    from piia.intelligence.public_orderbook import fetch_execution_books
    from piia.intelligence.public_position_tiers import fetch_position_tiers

    now = (now_override or datetime.now(timezone.utc)).astimezone(timezone.utc)
    cutoff_utc = now.replace(minute=0, second=0, microsecond=0)
    collection_errors: dict[str, str] = {}

    market_snap = fetch_public_snapshot(list(symbols), candle_limit=300)
    try:
        books_snap = fetch_execution_books(symbols)
    except Exception as exc:
        books_snap = {"ok": False, "instruments": {}}
        collection_errors["orderbook"] = type(exc).__name__
    try:
        tiers_snap = fetch_position_tiers(symbols)
    except Exception as exc:
        tiers_snap = {"ok": False, "instruments": {}}
        collection_errors["position_tiers"] = type(exc).__name__

    # ALFRED series for Macro + VIX + NDX/SPX relative strength
    series_wanted = ["DGS10", "DGS2", "DFII10", "DFF", "DTWEXBGS", "RRPONTSYD", "WDTGAL", "WALCL"]
    if include_live_alfred_extras:
        series_wanted.extend(["VIXCLS", "NASDAQ100", "SP500"])
    alfred_series, alfred_errors = fetch_alfred_public_series(series_wanted, now=now)
    collection_errors.update({f"alfred:{k}": v for k, v in alfred_errors.items()})

    dxy_live = fetch_public_dxy_proxy(cutoff_utc=cutoff_utc)

    try:
        event_calendar = fetch_official_event_calendar(now=now)
    except Exception as exc:
        event_calendar = {"present_sources": [], "sources": {}, "errors": {"calendar": type(exc).__name__}}
        collection_errors["event_calendar"] = type(exc).__name__

    try:
        etf_ctx = fetch_us_spot_btc_etf_flows(now=now)
        if not etf_ctx.get("usable_for_btc_research"):
            etf_ctx = fetch_tftc_etf(now=now)
    except Exception as exc:
        etf_ctx = {"usable_for_btc_research": False, "error": type(exc).__name__}
        collection_errors["btc_etf"] = type(exc).__name__

    try:
        fear_greed_ctx = fetch_fear_greed(now=now)
    except Exception as exc:
        fear_greed_ctx = {"usable_for_research": False, "error": type(exc).__name__}
        collection_errors["fear_greed"] = type(exc).__name__

    try:
        gold_cot_ctx = fetch_comex_gold_positioning(now=now)
    except Exception as exc:
        gold_cot_ctx = {"usable_for_gold_research": False, "error": type(exc).__name__}
        collection_errors["gold_cot"] = type(exc).__name__

    try:
        onchain_ctx = fetch_btc_onchain(now=now)
    except Exception as exc:
        onchain_ctx = {"usable_for_btc_research": False, "error": type(exc).__name__}
        collection_errors["btc_onchain"] = type(exc).__name__

    try:
        polymarket_ctx = fetch_prediction_markets(now=now)
    except Exception as exc:
        polymarket_ctx = {"usable_for_research": False, "events": [], "error": type(exc).__name__}
        collection_errors["polymarket"] = type(exc).__name__

    rss_items, rss_err = fetch_official_news_feeds(now=now)
    edgar_items, edgar_err = fetch_ndx_sec_edgar_filings(cutoff_utc=cutoff_utc)
    collection_errors.update({f"news_rss:{k}": v for k, v in rss_err.items()})
    collection_errors.update({f"sec_edgar:{k}": v for k, v in edgar_err.items()})

    series_for_fc = {}
    for sym in symbols:
        inst = (market_snap.get("instruments") or {}).get(sym) or {}
        c1h = [c for c in (inst.get("candles_closed_1h") or []) if (_ts_utc(c.get("close_at_utc")) or cutoff_utc) <= cutoff_utc]
        if c1h:
            series_for_fc[sym] = {"closes": [float(c["close"]) for c in c1h], "last_close_at_utc": c1h[-1]["close_at_utc"], "timeframe": "1H_closed"}
    try:
        fc_path = ROOT / "artifacts" / "judgment_v3_chronos_tmp.json"
        chronos_payload = run_chronos_tiny(series_for_fc, fc_path, allow_download=False)
        chronos_payload["run_at_utc"] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        chronos_payload = {"weights_ran": False, "reason": type(exc).__name__, "forecasts": {}}

    return {
        "is_synthetic_demo": False,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "cutoff_utc": cutoff_utc.isoformat(),
        "market_snapshot": market_snap,
        "books_snapshot": books_snap,
        "tiers_snapshot": tiers_snap,
        "alfred_series": alfred_series,
        "dxy_live": dxy_live,
        "event_calendar": event_calendar,
        "fear_greed": fear_greed_ctx,
        "etf_flows": etf_ctx,
        "gold_cot": gold_cot_ctx,
        "onchain": onchain_ctx,
        "exchange_netflow": None,
        "gold_etf_flows": None,
        "polymarket": polymarket_ctx,
        "raw_news_items": [*rss_items, *edgar_items],
        "ndx_edgar_filings": edgar_items,
        "chronos_forecast": chronos_payload,
        "llm_opinion": None,
        "collection_errors": collection_errors,
    }


def generate_demo_judgment_inputs(*, now: datetime | None = None, step_shift_hours: int = 0) -> dict[str, Any]:
    """Generate explicitly labelled synthetic multi-domain inputs for offline UI/CLI demo & tests."""
    from piia.intelligence.trade_assistant import generate_demo_snapshot

    base_now = (now or datetime(2026, 11, 24, 10, 15, tzinfo=timezone.utc)).astimezone(timezone.utc) + timedelta(hours=step_shift_hours)
    cutoff_utc = base_now.replace(minute=0, second=0, microsecond=0)
    demo_mkt = generate_demo_snapshot(now=base_now)
    # Trim any candles beyond cutoff just to mirror closed-bar invariant
    dates_obs = [(cutoff_utc - timedelta(days=i)).date().isoformat() for i in range(10, 1, -1)]
    alfred_series = {
        "DFII10": {"series_id": "DFII10", "source_url": "https://fred.stlouisfed.org/series/DFII10", "observations": [{"observation_date": d, "value": 2.02 + 0.03 * i} for i, d in enumerate(dates_obs)]},
        "DGS10": {"series_id": "DGS10", "source_url": "https://fred.stlouisfed.org/series/DGS10", "observations": [{"observation_date": d, "value": 4.35 + 0.03 * i} for i, d in enumerate(dates_obs)]},
        "DGS2": {"series_id": "DGS2", "source_url": "https://fred.stlouisfed.org/series/DGS2", "observations": [{"observation_date": d, "value": 4.28 + 0.02 * i} for i, d in enumerate(dates_obs)]},
        "DFF": {"series_id": "DFF", "source_url": "https://fred.stlouisfed.org/series/DFF", "observations": [{"observation_date": d, "value": 4.58} for d in dates_obs]},
        "DTWEXBGS": {"series_id": "DTWEXBGS", "source_url": "https://fred.stlouisfed.org/series/DTWEXBGS", "observations": [{"observation_date": d, "value": 124.0 + 0.25 * i} for i, d in enumerate(dates_obs)]},
        "WALCL": {"series_id": "WALCL", "source_url": "https://fred.stlouisfed.org/series/WALCL", "observations": [{"observation_date": d, "value": 6980000.0 - 5000.0 * i} for i, d in enumerate(dates_obs)]},
        "WDTGAL": {"series_id": "WDTGAL", "source_url": "https://fred.stlouisfed.org/series/WDTGAL", "observations": [{"observation_date": d, "value": 810000.0 + 4000.0 * i} for i, d in enumerate(dates_obs)]},
        "RRPONTSYD": {"series_id": "RRPONTSYD", "source_url": "https://fred.stlouisfed.org/series/RRPONTSYD", "observations": [{"observation_date": d, "value": 190.0} for d in dates_obs]},
        "VIXCLS": {"series_id": "VIXCLS", "source_url": "https://fred.stlouisfed.org/series/VIXCLS", "observations": [{"observation_date": d, "value": 18.0 + (7.5 if i == len(dates_obs) - 1 and step_shift_hours else 0.0)} for i, d in enumerate(dates_obs)]},
        "NASDAQ100": {"series_id": "NASDAQ100", "source_url": "https://fred.stlouisfed.org/series/NASDAQ100", "observations": [{"observation_date": d, "value": 20500.0 - 40.0 * i} for i, d in enumerate(dates_obs)]},
        "SP500": {"series_id": "SP500", "source_url": "https://fred.stlouisfed.org/series/SP500", "observations": [{"observation_date": d, "value": 5920.0 - 2.0 * i} for i, d in enumerate(dates_obs)]},
    }
    return {
        "is_synthetic_demo": True,
        "collected_at_utc": base_now.isoformat(),
        "cutoff_utc": cutoff_utc.isoformat(),
        "market_snapshot": demo_mkt,
        "books_snapshot": {
            "ok": True,
            "instruments": {s: {"orderbook": demo_mkt["instruments"][s]["orderbook"], "contract": None} for s in SYMBOLS},
        },
        "tiers_snapshot": {"ok": False, "instruments": {}},
        "alfred_series": alfred_series,
        "dxy_live": {"ok": False, "reason": "synthetic_demo_no_live_ice_dxy"},
        "event_calendar": {
            "present_sources": ["fed", "bls", "bea"],
            "retrieved_at_utc": cutoff_utc.isoformat(),
            "sources": {
                "bls": {
                    "source_url": "https://www.bls.gov/schedule/news_release/",
                    "window_events": [
                        {
                            "agency": "bls",
                            "title": "Consumer Price Index (Synthetic Demo Schedule)",
                            "scheduled_at_utc": (cutoff_utc + timedelta(hours=30)).isoformat(),
                            "source_url": "https://www.bls.gov/schedule/news_release/",
                        }
                    ],
                }
            },
        },
        "fear_greed": {
            "usable_for_research": True,
            "source_url": "https://api.alternative.me/fng/?limit=2",
            "retrieved_at_utc": base_now.isoformat(),
            "observations": [{"value": 68, "classification": "Greed", "observed_at_utc": (cutoff_utc - timedelta(hours=2)).isoformat()}],
        },
        "etf_flows": {
            "usable_for_btc_research": True,
            "source_url": "https://farside.co.uk/btc/",
            "retrieved_at_utc": base_now.isoformat(),
            "asof_complete_flow_date": (cutoff_utc - timedelta(days=2)).date().isoformat(),
            "latest_row_is_partial": True,
            "latest_row": {"flow_date": (cutoff_utc - timedelta(days=1)).date().isoformat(), "unreported_fund_count": 2, "complete": False},
            "latest_complete_row": {"flow_date": (cutoff_utc - timedelta(days=2)).date().isoformat(), "sum_of_reported_funds_millions_usd": 210.0, "complete": True},
            "complete_daily_flows": [
                {"flow_date": (cutoff_utc - timedelta(days=4)).date().isoformat(), "sum_of_reported_funds_millions_usd": 120.0, "complete": True},
                {"flow_date": (cutoff_utc - timedelta(days=3)).date().isoformat(), "sum_of_reported_funds_millions_usd": 165.0, "complete": True},
                {"flow_date": (cutoff_utc - timedelta(days=2)).date().isoformat(), "sum_of_reported_funds_millions_usd": 210.0, "complete": True},
            ],
        },
        "gold_cot": {
            "usable_for_gold_research": True,
            "source_url": "https://www.cftc.gov/dea/futures/deacmxsf.htm",
            "retrieved_at_utc": base_now.isoformat(),
            "asof_date": (cutoff_utc - timedelta(days=5)).date().isoformat(),
            "open_interest_contracts": 512000,
            "commitments_contracts": {"noncommercial_long": 295000, "noncommercial_short": 88000},
        },
        "onchain": {
            "usable_for_btc_research": True,
            "source_docs": "https://mempool.space/docs/api/rest",
            "retrieved_at_utc": base_now.isoformat(),
            "raw_sha256": {"tip": "demo"},
            "metrics": {
                "tip_height": {"value": 872000},
                "last_block": {"observation_at_utc": (cutoff_utc - timedelta(minutes=12)).isoformat()},
                "mempool": {"count": 14200},
                "recommended_fees": {"fastestFee": 18},
            },
        },
        "exchange_netflow": None,
        "gold_etf_flows": None,
        "polymarket": {"usable_for_research": False, "events": []},
        "raw_news_items": [
            {
                "news_id": "demo_news_geo_tariff",
                "title": "Synthetic Demo Official Statement: Tariff and geopolitical risk discussion raises safe-haven caution",
                "url": "https://www.federalreserve.gov/newsevents/pressreleases.htm",
                "published_at_utc": (cutoff_utc - timedelta(hours=3)).isoformat(),
                "source_name": "Synthetic Demo Primary Notice",
                "source_kind": "primary_official",
            }
        ],
        "ndx_edgar_filings": [],
        "chronos_forecast": {"weights_ran": False, "reason": "SYNTHETIC_DEMO_WEIGHTS_NOT_RUN", "forecasts": {}},
        "llm_opinion": None,
        "collection_errors": {},
    }
