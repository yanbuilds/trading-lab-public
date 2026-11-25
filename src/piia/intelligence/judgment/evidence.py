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
import tempfile
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
YAHOO_CHART_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
COINMETRICS_EXFLOW_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
BITGET_MIX_BASE = "https://api.bitget.com/api/v2/mix/market"

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


def sanitize_closed_candles(
    raw_candles: list[Any] | None,
    *,
    bar_cutoff_utc: datetime,
    bar_duration: timedelta = timedelta(hours=1),
    window: int | None = None,
) -> dict[str, Any]:
    """Filter invalid/unclosed/future candles first, sort chronologically, deduplicate, then slice `window`."""
    bar_cutoff_utc = bar_cutoff_utc.astimezone(timezone.utc)
    raw_list = list(raw_candles or [])
    audit: dict[str, Any] = {
        "raw_count": len(raw_list),
        "dropped_invalid": 0,
        "dropped_unclosed": 0,
        "dropped_future": 0,
        "dropped_duplicate": 0,
        "was_out_of_order": False,
        "valid_count": 0,
        "window_count": 0,
        "source_bar_cutoff_utc": bar_cutoff_utc.isoformat(),
    }
    valid: list[tuple[datetime, datetime, dict[str, Any]]] = []
    for c in raw_list:
        if not isinstance(c, dict):
            audit["dropped_invalid"] += 1
            continue
        open_dt = _ts_utc(c.get("open_at_utc"))
        close_dt = _ts_utc(c.get("close_at_utc"))
        if open_dt is None or close_dt is None:
            audit["dropped_invalid"] += 1
            continue
        try:
            op = float(c["open"])
            hi = float(c["high"])
            lo = float(c["low"])
            cl = float(c["close"])
            vol = float(c.get("volume") or 0.0)
        except (KeyError, TypeError, ValueError):
            audit["dropped_invalid"] += 1
            continue
        if not all(math.isfinite(v) for v in (op, hi, lo, cl, vol)) or op <= 0 or hi < lo or hi < max(op, cl) or lo > min(op, cl) or vol < 0:
            audit["dropped_invalid"] += 1
            continue
        if close_dt - open_dt != bar_duration or c.get("is_closed") is False or c.get("closed") is False or str(c.get("confirm", "1")) == "0":
            audit["dropped_unclosed"] += 1
            continue
        if close_dt > bar_cutoff_utc:
            audit["dropped_future"] += 1
            continue
        valid.append(
            (
                open_dt,
                close_dt,
                {
                    "open_at_utc": open_dt.isoformat(),
                    "close_at_utc": close_dt.isoformat(),
                    "open": op,
                    "high": hi,
                    "low": lo,
                    "close": cl,
                    "volume": vol,
                },
            )
        )

    if any(valid[i][0] > valid[i + 1][0] for i in range(len(valid) - 1)):
        audit["was_out_of_order"] = True
    valid.sort(key=lambda item: item[0])

    deduped: list[dict[str, Any]] = []
    seen_opens: set[datetime] = set()
    for open_dt, _close_dt, row in valid:
        if open_dt in seen_opens:
            audit["dropped_duplicate"] += 1
            continue
        seen_opens.add(open_dt)
        deduped.append(row)

    audit["valid_count"] = len(deduped)
    sliced = deduped[-int(window) :] if (window is not None and window > 0) else deduped
    audit["window_count"] = len(sliced)
    return {"candles": sliced, "all_sanitized": deduped, "audit": audit}


def validate_market_snapshot_coherence(
    market_snapshot: dict[str, Any],
    *,
    cutoff_utc: datetime,
) -> dict[str, Any]:
    """Sanitize and verify all three contracts share closed 1h cutoff == `cutoff_utc` without gaps."""
    cutoff_utc = cutoff_utc.astimezone(timezone.utc)
    one_h = timedelta(hours=1)
    reasons: list[str] = []
    last_close_by_sym: dict[str, str | None] = {}
    frames_1h: dict[str, pd.DataFrame] = {}
    frames_1d: dict[str, pd.DataFrame] = {}
    sanitization_audit: dict[str, Any] = {}

    for sym in SYMBOLS:
        inst = (market_snapshot.get("instruments") or {}).get(sym) or {}
        clean_1h = sanitize_closed_candles(inst.get("candles_closed_1h"), bar_cutoff_utc=cutoff_utc, bar_duration=one_h)
        sanitized = clean_1h["all_sanitized"]
        sanitization_audit[sym] = clean_1h["audit"]
        if not sanitized:
            reasons.append(f"missing_1h_candles:{sym}")
            last_close_by_sym[sym] = None
            continue

        idx = [pd.Timestamp(_ts_utc(c["open_at_utc"])) for c in sanitized]
        rows = [{"open": c["open"], "high": c["high"], "low": c["low"], "close": c["close"], "volume": c["volume"]} for c in sanitized]
        df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="date")).sort_index()
        diffs = df.index.to_series().diff().dropna()
        if not diffs.empty and bool((diffs != one_h).any()):
            reasons.append(f"gapped_1h_candles:{sym}")
        last_close = (df.index[-1] + pd.Timedelta(hours=1)).to_pydatetime().astimezone(timezone.utc)
        last_close_by_sym[sym] = last_close.isoformat()
        if last_close < cutoff_utc:
            reasons.append(f"lagging_1h_candle:{sym}:{last_close.isoformat()}<{cutoff_utc.isoformat()}")
        frames_1h[sym] = df

        clean_1d = sanitize_closed_candles(inst.get("candles_closed_1d"), bar_cutoff_utc=cutoff_utc, bar_duration=timedelta(days=1))
        if clean_1d["all_sanitized"]:
            d_idx = [pd.Timestamp(_ts_utc(c["open_at_utc"])) for c in clean_1d["all_sanitized"]]
            d_rows = [{"open": c["open"], "high": c["high"], "low": c["low"], "close": c["close"], "volume": c["volume"]} for c in clean_1d["all_sanitized"]]
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
        "sanitization_audit": sanitization_audit,
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
    raw_vintage = series_entry.get("vintage_date")
    if raw_vintage:
        try:
            v_date = datetime.strptime(str(raw_vintage)[:10], "%Y-%m-%d").date()
            if v_date > cutoff_utc.date():
                return {"status": "future_vintage_excluded", "vintage_date": str(raw_vintage)}
        except ValueError:
            pass
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


def fetch_yahoo_intraday_macro_and_breadth(
    *,
    bar_cutoff_utc: datetime,
    timeout: float = 8.0,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Fetch public 1h delayed/intraday macro & breadth proxies strictly sliced <= bar_cutoff_utc."""
    from urllib.parse import quote as _url_quote

    cutoff_utc = bar_cutoff_utc.astimezone(timezone.utc)
    tickers = ("DX-Y.NYB", "^TNX", "TIP", "QQQ", "QQEW")
    own = client is None
    client = client or httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 TradingLabResearch/3.2"},
    )
    series_out: dict[str, Any] = {}
    errors: dict[str, str] = {}
    try:
        for sym in tickers:
            url = f"{YAHOO_CHART_BASE}/{_url_quote(sym)}?interval=1h&range=5d"
            try:
                resp = client.get(url)
                resp.raise_for_status()
                raw = resp.content
                payload = resp.json()
                res = ((payload.get("chart") or {}).get("result") or [None])[0]
                if not isinstance(res, dict):
                    errors[sym] = "empty_chart_result"
                    continue
                ts_list = res.get("timestamp") or []
                closes = (((res.get("indicators") or {}).get("quote") or [{}])[0] or {}).get("close") or []
                pts: list[tuple[datetime, float]] = []
                for ts_val, cl_val in zip(ts_list, closes):
                    if ts_val is None or cl_val is None:
                        continue
                    try:
                        c_f = float(cl_val)
                        t_dt = datetime.fromtimestamp(int(ts_val), tz=timezone.utc)
                    except (TypeError, ValueError, OSError):
                        continue
                    if math.isfinite(c_f) and t_dt <= cutoff_utc:
                        pts.append((t_dt, c_f))
                pts.sort(key=lambda x: x[0])
                if not pts:
                    errors[sym] = "no_bars_le_cutoff"
                    continue
                last_dt, last_px = pts[-1]
                target_24h = last_dt - timedelta(hours=24)
                ref_24h = min(pts, key=lambda x: abs((x[0] - target_24h).total_seconds()))
                px_24h = ref_24h[1] if len(pts) >= 2 and ref_24h[0] < last_dt - timedelta(hours=6) else None
                first_dt, first_px = pts[0]
                pct_24h = ((last_px / px_24h - 1.0) * 100.0) if (px_24h is not None and px_24h > 0) else None
                pct_5d = ((last_px / first_px - 1.0) * 100.0) if (first_px > 0 and (last_dt - first_dt).total_seconds() >= 24 * 3600) else None
                delta_24h = (last_px - px_24h) if px_24h is not None else None
                series_out[sym] = {
                    "ok": True,
                    "symbol": sym,
                    "close": round(last_px, 6),
                    "obs_at_utc": last_dt.isoformat(),
                    "bars_count": len(pts),
                    "pct_24h": round(pct_24h, 4) if pct_24h is not None else None,
                    "pct_5d": round(pct_5d, 4) if pct_5d is not None else None,
                    "delta_24h": round(delta_24h, 6) if delta_24h is not None else None,
                    "source_url": url,
                    "raw_sha256": hashlib.sha256(raw).hexdigest(),
                }
            except Exception as exc:
                errors[sym] = type(exc).__name__
    finally:
        if own:
            client.close()

    qqew = series_out.get("QQEW") or {}
    qqq = series_out.get("QQQ") or {}
    qqew_minus_qqq_5d = (
        round(float(qqew["pct_5d"]) - float(qqq["pct_5d"]), 4)
        if (qqew.get("ok") and qqq.get("ok") and qqew.get("pct_5d") is not None and qqq.get("pct_5d") is not None)
        else None
    )
    qqew_minus_qqq_24h = (
        round(float(qqew["pct_24h"]) - float(qqq["pct_24h"]), 4)
        if (qqew.get("ok") and qqq.get("ok") and qqew.get("pct_24h") is not None and qqq.get("pct_24h") is not None)
        else None
    )
    return {
        "ok": bool(series_out),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "bar_cutoff_utc": cutoff_utc.isoformat(),
        "series": series_out,
        "qqew_minus_qqq_5d_pct": qqew_minus_qqq_5d,
        "qqew_minus_qqq_24h_pct": qqew_minus_qqq_24h,
        "errors": errors,
    }


def fetch_coinmetrics_btc_exchange_flows(
    *,
    decision_cutoff_utc: datetime,
    timeout: float = 8.0,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Fetch public CoinMetrics BTC exchange inflows/outflows strictly gated by status-time <= cutoff."""
    cutoff_utc = decision_cutoff_utc.astimezone(timezone.utc)
    params = "assets=btc&metrics=FlowInExNtv,FlowOutExNtv,FlowInExUSD,FlowOutExUSD&page_size=10"
    url = f"{COINMETRICS_EXFLOW_URL}?{params}"
    own = client is None
    client = client or httpx.Client(timeout=timeout, follow_redirects=True, headers={"User-Agent": "TradingLabResearch/3.2"})
    try:
        resp = client.get(url)
        resp.raise_for_status()
        raw = resp.content
        raw_sha = hashlib.sha256(raw).hexdigest()
        payload = resp.json()
        eligible: list[tuple[datetime, datetime, dict[str, Any]]] = []
        for row in payload.get("data") or []:
            if not isinstance(row, dict):
                continue
            t_dt = _ts_utc(row.get("time"))
            in_st = _ts_utc(row.get("FlowInExNtv-status-time"))
            out_st = _ts_utc(row.get("FlowOutExNtv-status-time"))
            if t_dt is None or in_st is None or out_st is None:
                continue
            avail_dt = max(in_st, out_st)
            if t_dt <= cutoff_utc and avail_dt <= cutoff_utc:
                try:
                    float(row["FlowInExNtv"])
                    float(row["FlowOutExNtv"])
                except (KeyError, TypeError, ValueError):
                    continue
                eligible.append((t_dt, avail_dt, row))
        if not eligible:
            return {"usable": False, "reason": "no_point_in_time_verified_row", "source_url": url, "raw_sha256": raw_sha}
        eligible.sort(key=lambda x: x[0])
        t_dt, avail_dt, row = eligible[-1]
        in_btc = float(row["FlowInExNtv"])
        out_btc = float(row["FlowOutExNtv"])
        net_btc = in_btc - out_btc
        net_usd_musd = None
        if row.get("FlowInExUSD") is not None and row.get("FlowOutExUSD") is not None:
            try:
                net_usd_musd = round((float(row["FlowInExUSD"]) - float(row["FlowOutExUSD"])) / 1e6, 3)
            except (TypeError, ValueError):
                pass
        return {
            "usable": True,
            "source_name": "CoinMetrics Community API",
            "source_url": url,
            "raw_sha256": raw_sha,
            "observed_at_utc": t_dt.isoformat(),
            "available_at_utc": avail_dt.isoformat(),
            "flow_in_btc": round(in_btc, 4),
            "flow_out_btc": round(out_btc, 4),
            "netflow_btc": round(net_btc, 4),
            "netflow_usd_musd": net_usd_musd,
            "metric_status": str(row.get("FlowInExNtv-status") or "flash"),
        }
    except Exception as exc:
        return {"usable": False, "reason": type(exc).__name__, "source_url": url}
    finally:
        if own:
            client.close()


def fetch_bitget_venue_microstructure(
    symbols: tuple[str, ...] = SYMBOLS,
    *,
    decision_cutoff_utc: datetime,
    timeout: float = 8.0,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Fetch live Bitget mark/last/index prices and current funding interval schedule."""
    own = client is None
    client = client or httpx.Client(timeout=timeout, follow_redirects=True, headers={"User-Agent": "TradingLabResearch/3.2"})
    out_syms: dict[str, Any] = {}
    errors: dict[str, str] = {}
    try:
        ticker_map: dict[str, dict[str, Any]] = {}
        try:
            r_all = client.get(f"{BITGET_MIX_BASE}/tickers?productType=USDT-FUTURES")
            r_all.raise_for_status()
            for row in (r_all.json().get("data") or []):
                if row.get("symbol") in symbols:
                    ticker_map[str(row["symbol"])] = row
        except Exception as exc:
            errors["tickers"] = type(exc).__name__

        for sym in symbols:
            frow: dict[str, Any] = {}
            try:
                r_fr = client.get(f"{BITGET_MIX_BASE}/current-fund-rate?productType=USDT-FUTURES&symbol={sym}")
                r_fr.raise_for_status()
                d_list = r_fr.json().get("data") or []
                if d_list:
                    frow = d_list[0]
            except Exception as exc:
                errors[f"fund_rate:{sym}"] = type(exc).__name__

            trow = ticker_map.get(sym) or {}
            try:
                last_p = float(trow["lastPr"]) if trow.get("lastPr") is not None else None
                mark_p = float(trow["markPrice"]) if trow.get("markPrice") is not None else None
                idx_p = float(trow["indexPrice"]) if trow.get("indexPrice") is not None else None
            except (TypeError, ValueError):
                last_p = mark_p = idx_p = None
            mark_vs_last_bps = ((mark_p - last_p) / last_p * 10000.0) if (mark_p and last_p and last_p > 0) else None
            mark_vs_idx_bps = ((mark_p - idx_p) / idx_p * 10000.0) if (mark_p and idx_p and idx_p > 0) else None
            interval_h = None
            if frow.get("fundingRateInterval") is not None:
                try:
                    interval_h = int(frow["fundingRateInterval"])
                except (TypeError, ValueError):
                    interval_h = None
            next_upd_iso = None
            if frow.get("nextUpdate"):
                try:
                    next_upd_iso = datetime.fromtimestamp(int(frow["nextUpdate"]) / 1000.0, tz=timezone.utc).isoformat()
                except (TypeError, ValueError, OSError):
                    next_upd_iso = None
            out_syms[sym] = {
                "last_price": last_p,
                "mark_price": mark_p,
                "index_price": idx_p,
                "mark_vs_last_bps": round(mark_vs_last_bps, 2) if mark_vs_last_bps is not None else None,
                "mark_vs_index_bps": round(mark_vs_idx_bps, 2) if mark_vs_idx_bps is not None else None,
                "funding_rate": float(frow["fundingRate"]) if frow.get("fundingRate") is not None else None,
                "funding_interval_hours": interval_h,
                "next_funding_utc": next_upd_iso,
                "min_funding_rate": frow.get("minFundingRate"),
                "max_funding_rate": frow.get("maxFundingRate"),
            }
    finally:
        if own:
            client.close()
    return {
        "ok": any((v.get("mark_price") is not None or v.get("funding_interval_hours") is not None) for v in out_syms.values()),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_cutoff_utc": decision_cutoff_utc.isoformat(),
        "symbols": out_syms,
        "errors": errors,
    }


_CALENDAR_STOPWORDS = frozenset({
    "and", "the", "for", "from", "with", "of", "in", "on", "to", "by", "at",
    "release", "releases", "report", "second", "third", "first", "advance",
    "preliminary", "estimate", "quarter", "annual", "monthly", "weekly",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "economic", "projections",
    "press", "board", "bureau", "department", "news", "schedule",
})


def _agency_matches_news_item(agency: str, item: dict[str, Any]) -> bool:
    ag = agency.lower().strip()
    url = str(item.get("url") or "").lower()
    src = str(item.get("source_name") or "").lower()
    if "bea" in ag:
        return ("bea.gov" in url) or ("bea" in src) or ("bureau of economic analysis" in src)
    if "bls" in ag:
        return ("bls.gov" in url) or ("bls" in src) or ("labor statistics" in src)
    if "fed" in ag:
        return ("federalreserve.gov" in url) or ("federal reserve" in src)
    return True


def _post_release_reaction_proxy(
    sched_dt: datetime,
    *,
    cutoff_utc: datetime,
    frames_1h: dict[str, pd.DataFrame] | None,
) -> dict[str, Any] | None:
    """Measure observed 0-6h post-release price impulse when official feed lacks consensus."""
    if not frames_1h:
        return None
    hours_since = (cutoff_utc - sched_dt).total_seconds() / 3600.0
    if not (0.0 <= hours_since <= 6.0):
        return None
    per_sym: dict[str, Any] = {}
    win_start = sched_dt - timedelta(hours=1)
    for sym, df in frames_1h.items():
        if df is None or df.empty:
            continue
        post_df = df.loc[(df.index >= win_start) & (df.index <= cutoff_utc)]
        if post_df.empty:
            continue
        base_px = float(post_df.iloc[0]["open"])
        last_px = float(post_df.iloc[-1]["close"])
        atr_approx = float((df["high"].tail(14) - df["low"].tail(14)).mean()) if len(df) >= 14 else 0.0
        if base_px > 0:
            ret_pct = (last_px / base_px - 1.0) * 100.0
            atr_mult = ((last_px - base_px) / atr_approx) if atr_approx > 0 else None
            per_sym[sym] = {
                "ret_since_release_pct": round(ret_pct, 4),
                "atr_multiple": round(atr_mult, 3) if atr_mult is not None else None,
            }
    if not per_sym:
        return None
    return {
        "status": "observed_post_release_price_impulse",
        "hours_since_release": round(hours_since, 2),
        "by_symbol": per_sym,
        "note": "实测发布后价格冲击代理（绝非华尔街一致预期值）",
    }


def _normalize_tnx_yield(tnx_series: dict[str, Any]) -> tuple[float | None, float | None]:
    """Return (yield_pct, delta_24h_bps) handling both CBOE 10x index (42.5) and percent (4.25) Yahoo quotes."""
    if not tnx_series.get("ok") or tnx_series.get("close") is None:
        return None, None
    c_val = float(tnx_series["close"])
    d24 = tnx_series.get("delta_24h")
    # Yahoo ^TNX may quote either 10x yield (e.g. 42.5 = 4.25%) or direct yield % (e.g. 4.904 = 4.904%)
    if abs(c_val) > 20.0:
        y_pct = round(c_val / 10.0, 4)
        bps = round(float(d24) * 10.0, 2) if d24 is not None else None
    else:
        y_pct = round(c_val, 4)
        bps = round(float(d24) * 100.0, 2) if d24 is not None else None
    return y_pct, bps


def _match_calendar_and_releases(
    event_calendar: dict[str, Any] | None,
    verified_news: list[dict[str, Any]],
    *,
    cutoff_utc: datetime,
    imminent_hours: float,
    frames_1h: dict[str, pd.DataFrame] | None = None,
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

            # Match verified actual release strictly requiring time window + agency match + specific topic tokens
            matched_actual = None
            agency_str = str(ev.get("agency") or src_name)
            import re as _re
            sched_tokens = {
                w for w in _re.findall(r"[a-z0-9]+", title.lower())
                if len(w) >= 3 and not w.isdigit() and w not in _CALENDAR_STOPWORDS
            }
            if cutoff_utc >= sched_dt - timedelta(minutes=15) and sched_tokens:
                win_lo = sched_dt - timedelta(minutes=15)
                win_hi = min(cutoff_utc, sched_dt + timedelta(hours=12))
                for item in verified_news:
                    if item.get("verification_status") != "verified" or item.get("freshness_status") == "future_excluded":
                        continue
                    if item.get("news_kind") != "primary_release":
                        continue
                    pub_dt = _ts_utc(item.get("published_at_utc"))
                    if pub_dt is None or not (win_lo <= pub_dt <= win_hi):
                        continue
                    if not _agency_matches_news_item(agency_str, item):
                        continue
                    item_tokens = {
                        w for w in _re.findall(r"[a-z0-9]+", str(item.get("title") or "").lower())
                        if len(w) >= 3 and not w.isdigit() and w not in _CALENDAR_STOPWORDS
                    }
                    if sched_tokens & item_tokens:
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

            reaction_proxy = _post_release_reaction_proxy(sched_dt, cutoff_utc=cutoff_utc, frames_1h=frames_1h)
            events_structured.append(
                {
                    "agency": agency_str,
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
                    "post_release_market_reaction": reaction_proxy,
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
    intraday_macro: dict[str, Any] | None = None,
    frames_1h: dict[str, pd.DataFrame] | None = None,
    replay_mode: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build Domain 1 (Macro) EvidenceAtoms separating ICE DXY, intraday proxies, daily rates, and events."""
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
                value_summary={"status": "unavailable", "reason": (dxy_live or {}).get("reason") or "ice_proprietary_index_no_free_verified_realtime_api", "is_ice_dxy": False},
                direction_by_symbol={s: "unknown" for s in SYMBOLS},
                causal_driver_group="usd_real_rate_tightening",
                limitation_note="未获取到带可核验盘中时间戳的 ICE DXY（属商业专有指数且免费源无带可信UTC时钟实时值），绝不以广义美元指数冒充实时 DXY",
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
        dtwex_fresh = "future_excluded" if dtwex["status"] == "future_vintage_excluded" else ("missing" if dtwex["status"] == "missing" else "stale_excluded")
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
                freshness_status=dtwex_fresh,
                applies_to_symbols=list(SYMBOLS),
                applies_to_horizons=["swing_1_3d"],
                value_summary={"status": dtwex["status"]},
                direction_by_symbol={s: "unknown" for s in SYMBOLS},
                causal_driver_group="usd_real_rate_tightening",
                limitation_note="广义美元代理序列缺失、过期或含未来修订版本号",
            )
        )

    # 1c. Daily Real Yield (DFII10), Intraday Real Yield Gap, & Nominal Rates (DGS10, DGS2, DFF)
    atoms.append(
        make_atom(
            evidence_id="macro:real_yield_intraday",
            domain="macro",
            factor_name="盘中实时美国实际利率",
            source_name="unverified_or_unavailable",
            source_url=None,
            raw_sha256=None,
            data_tier="unavailable",
            observed_at_utc=None,
            published_at_utc=None,
            available_at_utc=None,
            available_at_known=False,
            retrieved_at_utc=cutoff_iso,
            freshness_status="missing",
            applies_to_symbols=list(SYMBOLS),
            applies_to_horizons=["intraday"],
            value_summary={"status": "unavailable", "reason": "tips_intraday_real_yield_requires_licensed_dealer_feed", "intraday_realtime": False},
            direction_by_symbol={s: "unknown" for s in SYMBOLS},
            causal_driver_group="usd_real_rate_tightening",
            limitation_note="公开免费数据源无盘中实时 TIPS 实际利率报价，日频 DFII10 绝不冒充盘中实时值",
        )
    )
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
        dfii_fresh = "future_excluded" if dfii["status"] == "future_vintage_excluded" else ("missing" if dfii["status"] == "missing" else "stale_excluded")
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
                freshness_status=dfii_fresh,
                applies_to_symbols=list(SYMBOLS),
                applies_to_horizons=["swing_1_3d"],
                value_summary={"status": dfii["status"], "intraday_realtime": False},
                direction_by_symbol={s: "unknown" for s in SYMBOLS},
                causal_driver_group="usd_real_rate_tightening",
                limitation_note="实际利率日频序列缺失、过期或含未来修订版本号",
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

    # 1b-extra & 1c-extra. Verified Public 1H Intraday Dollar (DX-Y.NYB) & Rates/TIPS (^TNX, TIP) Proxies
    im_raw = intraday_macro or {}
    im_pit_ok = (not replay_mode) or bool(im_raw.get("historical_point_in_time_verified"))
    im_series = (im_raw.get("series") or {}) if im_pit_ok else {}

    dx_y = im_series.get("DX-Y.NYB") or {}
    dx_obs_dt = _ts_utc(dx_y.get("obs_at_utc"))
    dx_fresh = "missing"
    if dx_y.get("ok") and dx_obs_dt is not None:
        if dx_obs_dt > cutoff_utc:
            dx_fresh = "future_excluded"
        else:
            age_h = (cutoff_utc - dx_obs_dt).total_seconds() / 3600.0
            dx_fresh = "fresh" if age_h <= 6.0 else ("lagged_usable" if age_h <= 72.0 else "stale_excluded")
    if dx_fresh in {"fresh", "lagged_usable"}:
        dx_p5 = dx_y.get("pct_5d")
        dx_p24 = dx_y.get("pct_24h")
        dx_thresh = float(mcfg.get("intraday_dxy_5d_pct", 0.75))
        dx_ref = dx_p24 if (dx_p24 is not None and abs(float(dx_p24)) >= float(mcfg["dxy_24h_pct"])) else dx_p5
        dx_lim = float(mcfg["dxy_24h_pct"]) if dx_ref is dx_p24 else dx_thresh
        dx_dir = {s: "neutral" for s in SYMBOLS}
        if dx_ref is not None:
            if float(dx_ref) >= dx_lim:
                dx_dir = {s: "bearish" for s in SYMBOLS}
            elif float(dx_ref) <= -dx_lim:
                dx_dir = {s: "bullish" for s in SYMBOLS}
        atoms.append(
            make_atom(
                evidence_id="macro:intraday_dollar_proxy:DX_Y_NYB",
                domain="macro",
                factor_name="盘中美元期货/指数公开代理 (DX-Y.NYB 1H)",
                source_name="Yahoo Finance Public Chart (DX-Y.NYB)",
                source_url=dx_y.get("source_url"),
                raw_sha256=dx_y.get("raw_sha256"),
                data_tier="intraday_delayed_public",
                observed_at_utc=dx_obs_dt.isoformat() if dx_obs_dt else None,
                published_at_utc=dx_obs_dt.isoformat() if dx_obs_dt else None,
                available_at_utc=dx_obs_dt.isoformat() if dx_obs_dt else None,
                available_at_known=True,
                retrieved_at_utc=im_raw.get("retrieved_at_utc") or cutoff_iso,
                freshness_status=dx_fresh,
                applies_to_symbols=list(SYMBOLS),
                applies_to_horizons=["intraday", "swing_1_3d"],
                value_summary={"close": dx_y.get("close"), "pct_24h": dx_p24, "pct_5d": dx_p5, "is_ice_dxy": False, "proxy_ticker": "DX-Y.NYB"},
                direction_by_symbol=dx_dir,
                causal_driver_group="usd_real_rate_tightening",
                limitation_note="Yahoo DX-Y.NYB 1H 美元指数代理（受交易所延迟与交易时段约束，绝不冒充实时直连 ICE DXY）",
            )
        )

    tnx = im_series.get("^TNX") or {}
    tip_etf = im_series.get("TIP") or {}
    tnx_dt = _ts_utc(tnx.get("obs_at_utc"))
    tip_dt = _ts_utc(tip_etf.get("obs_at_utc"))
    rates_dt = tnx_dt or tip_dt
    rates_fresh = "missing"
    if (tnx.get("ok") or tip_etf.get("ok")) and rates_dt is not None:
        if rates_dt > cutoff_utc:
            rates_fresh = "future_excluded"
        else:
            age_h = (cutoff_utc - rates_dt).total_seconds() / 3600.0
            rates_fresh = "fresh" if age_h <= 24.0 else ("lagged_usable" if age_h <= 72.0 else "stale_excluded")
    tnx_pct, tnx_bps_24h = _normalize_tnx_yield(tnx)
    if rates_fresh in {"fresh", "lagged_usable"}:
        tip_p24 = tip_etf.get("pct_24h") if tip_etf.get("ok") else None
        bps_thr = float(mcfg.get("intraday_tnx_24h_bps", 6.0))
        rt_dir = {s: "neutral" for s in SYMBOLS}
        if tnx_bps_24h is not None:
            if tnx_bps_24h >= bps_thr and (tip_p24 is None or float(tip_p24) <= -0.10):
                rt_dir = {s: "bearish" for s in SYMBOLS}
            elif tnx_bps_24h <= -bps_thr and (tip_p24 is None or float(tip_p24) >= 0.10):
                rt_dir = {s: "bullish" for s in SYMBOLS}
        atoms.append(
            make_atom(
                evidence_id="macro:intraday_rates_tips_proxy:TNX_TIP",
                domain="macro",
                factor_name="盘中 10Y 名义美债收益率与 TIPS 代理 (^TNX / TIP 1H)",
                source_name="Yahoo Finance Public Chart (^TNX & TIP)",
                source_url=tnx.get("source_url") or tip_etf.get("source_url"),
                raw_sha256=tnx.get("raw_sha256") or tip_etf.get("raw_sha256"),
                data_tier="proxy_series",
                observed_at_utc=rates_dt.isoformat() if rates_dt else None,
                published_at_utc=rates_dt.isoformat() if rates_dt else None,
                available_at_utc=rates_dt.isoformat() if rates_dt else None,
                available_at_known=True,
                retrieved_at_utc=im_raw.get("retrieved_at_utc") or cutoff_iso,
                freshness_status=rates_fresh,
                applies_to_symbols=list(SYMBOLS),
                applies_to_horizons=["intraday"],
                value_summary={"tnx_yield_pct": tnx_pct, "tnx_delta_24h_bps": tnx_bps_24h, "tip_pct_24h": tip_p24, "intraday_realtime": False},
                direction_by_symbol=rt_dir,
                causal_driver_group="usd_real_rate_tightening",
                limitation_note="^TNX 与 TIP ETF 属于盘中公开延迟市场代理（非商业实时 TIPS 实际利率报价），且同属利率紧缩因果组去重",
            )
        )

    # 1f. Key Economic Events (schedule / consensus / actual separation + post-release reaction proxy)
    events_list, imminent_veto, imminent_title = _match_calendar_and_releases(
        event_calendar,
        verified_news,
        cutoff_utc=cutoff_utc,
        imminent_hours=float(fcfg["event_imminent_veto_hours_ahead"]),
        frames_1h=frames_1h,
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
            limitation_note="官方日程表仅含预计发布时间，不含市场一致预期值；已公布结果仅绑定同机构、发布时点之后的官方公报并附盘后实测脉冲代理",
        )
    )

    macro_summary = {
        "ice_dxy_available": any(a["evidence_id"] == "macro:ice_dxy" and a["freshness_status"] == "fresh" for a in atoms),
        "dxy_value": (dxy_live or {}).get("value") if isinstance(dxy_live, dict) and dxy_live.get("is_ice_dxy") else None,
        "intraday_dollar_dx_y_nyb": dx_y.get("close") if dx_fresh in {"fresh", "lagged_usable"} else None,
        "intraday_dollar_dx_y_nyb_pct_24h": dx_y.get("pct_24h") if dx_fresh in {"fresh", "lagged_usable"} else None,
        "intraday_tnx_pct": tnx_pct if rates_fresh in {"fresh", "lagged_usable"} else None,
        "intraday_tnx_delta_24h_bps": tnx_bps_24h if rates_fresh in {"fresh", "lagged_usable"} else None,
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
    intraday_macro: dict[str, Any] | None = None,
    replay_mode: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Build Domain 3 (Sentiment & Capital Flows) atoms across BTC, NDX100, Gold, and Polymarket."""
    cutoff_iso = cutoff_utc.isoformat()
    fcfg = cfg["freshness_limits"]
    mcfg = cfg["macro_and_tide_thresholds"]
    atoms: list[dict[str, Any]] = []

    # 3A. BTC Fear & Greed
    fg_raw = fear_greed_ctx or {}
    fg_pit_ok = (not replay_mode) or bool(fg_raw.get("historical_point_in_time_verified"))
    fg_obs = (fg_raw.get("observations") or []) if fg_pit_ok else []
    fg_ok = bool(fg_pit_ok and fg_raw.get("usable_for_research") and fg_obs)
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
                source_url=fg_raw.get("source_url"),
                raw_sha256=fg_raw.get("raw_sha256"),
                data_tier="lagged_official_report" if fg_pit_ok else "unavailable",
                observed_at_utc=fg_ts,
                published_at_utc=fg_ts,
                available_at_utc=fg_ts,
                available_at_known=fg_ok,
                retrieved_at_utc=fg_raw.get("retrieved_at_utc") or cutoff_iso,
                freshness_status="lagged_usable" if fg_ok else "missing",
                applies_to_symbols=["BTCUSDT"],
                applies_to_horizons=["intraday", "swing_1_3d"],
                value_summary={
                    "value": fg_val,
                    "classification": fg_obs[0].get("classification") if fg_obs else None,
                    "api_responded_ok": bool(fg_raw.get("usable_for_research")),
                    "reason": None if fg_ok else ("not_available_point_in_time_in_replay" if not fg_pit_ok else "unavailable"),
                },
                direction_by_symbol=fg_dir,
                causal_driver_group="btc_sentiment_crowding",
                limitation_note="日频综合情绪指标，非独立于价格的交易信号；历史回放不回填",
            ),
            cutoff_utc=cutoff_utc,
            max_age_hours=float(fcfg["fear_greed_max_age_hours"]),
        )
    )

    # 3B. US Spot BTC ETF Flows (dashes '-' NEVER converted to zero; supports both Farside 3d and TFTC 1d!)
    etf_obj = etf_ctx or {}
    etf_pit_ok = (not replay_mode) or bool(etf_obj.get("historical_point_in_time_verified"))
    latest_row = etf_obj.get("latest_row") or {}
    latest_complete = etf_obj.get("latest_complete_row") or {}
    complete_days = [d for d in (etf_obj.get("complete_daily_flows") or []) if d.get("complete")]
    complete_days.sort(key=lambda d: str(d.get("flow_date") or ""))
    etf_3d_sum = None
    etf_latest_single = None
    etf_fresh = "missing"
    etf_dir = {s: "unknown" for s in SYMBOLS}
    etf_mode = "unavailable"
    partial_unreported = bool(etf_obj.get("latest_row_is_partial") or (latest_row and not latest_row.get("complete")))
    etf_max_days = int(fcfg.get("etf_complete_max_age_days", 7))
    etf_mod_dt = _ts_utc(etf_obj.get("date_modified"))

    if not etf_pit_ok:
        etf_mode = "not_available_point_in_time_in_replay"
    elif etf_obj.get("usable_for_btc_research") and latest_complete:
        if etf_mod_dt is not None and etf_mod_dt > cutoff_utc:
            etf_fresh = "future_excluded"
            etf_mode = "date_modified_after_cutoff"
        else:
            recent_3 = [d for d in complete_days if str(d.get("flow_date") or "") <= cutoff_utc.date().isoformat()][-3:]
            vals_3 = [d.get("sum_of_reported_funds_millions_usd") for d in recent_3 if d.get("sum_of_reported_funds_millions_usd") is not None]
            flow_d_str = str(latest_complete.get("flow_date") or "")
            flow_d_ok = False
            if flow_d_str and flow_d_str <= cutoff_utc.date().isoformat():
                try:
                    fd = datetime.strptime(flow_d_str[:10], "%Y-%m-%d").date()
                    flow_d_ok = 0 <= (cutoff_utc.date() - fd).days <= etf_max_days
                except ValueError:
                    flow_d_ok = False
            if len(vals_3) == 3 and flow_d_ok:
                etf_3d_sum = round(sum(float(v) for v in vals_3), 3)
                etf_fresh = "lagged_usable"
                etf_mode = "three_day_complete_farside"
                if etf_3d_sum <= float(mcfg["etf_3d_outflow_musd"]):
                    etf_dir["BTCUSDT"] = "bearish"
                elif etf_3d_sum >= float(mcfg["etf_3d_inflow_musd"]):
                    etf_dir["BTCUSDT"] = "bullish"
                else:
                    etf_dir["BTCUSDT"] = "neutral"
            elif flow_d_ok and latest_complete.get("complete") is True:
                single_v = latest_complete.get("table_total_millions_usd")
                if single_v is None:
                    single_v = latest_complete.get("sum_of_reported_funds_millions_usd")
                if single_v is not None and math.isfinite(float(single_v)):
                    etf_latest_single = round(float(single_v), 3)
                    etf_fresh = "lagged_usable"
                    etf_mode = "single_complete_session_tftc" if "tftc" in str(etf_obj.get("kind") or "") else "single_complete_session"
                    if etf_latest_single <= float(mcfg.get("etf_1d_outflow_musd", -100.0)):
                        etf_dir["BTCUSDT"] = "bearish"
                    elif etf_latest_single >= float(mcfg.get("etf_1d_inflow_musd", 100.0)):
                        etf_dir["BTCUSDT"] = "bullish"
                    else:
                        etf_dir["BTCUSDT"] = "neutral"

    latest_flow_date = latest_complete.get("flow_date") or latest_row.get("flow_date")
    latest_complete_val = (
        latest_complete.get("sum_of_reported_funds_millions_usd")
        if latest_complete.get("sum_of_reported_funds_millions_usd") is not None
        else latest_complete.get("table_total_millions_usd")
    )
    etf_pub_iso = etf_mod_dt.isoformat() if etf_mod_dt is not None else (f"{latest_flow_date}T21:00:00+00:00" if latest_flow_date else None)
    atoms.append(
        make_atom(
            evidence_id="sentiment:btc:spot_etf_flows",
            domain="sentiment_flow",
            factor_name="美国现货 BTC ETF 日净流量 (Farside/TFTC)",
            source_name="TFTC Bitcoin ETF Tracker" if "tftc" in str(etf_obj.get("kind") or "") else "Farside Investors / TFTC",
            source_url=etf_obj.get("source_url"),
            raw_sha256=etf_obj.get("raw_sha256"),
            data_tier="lagged_official_report" if etf_pit_ok else "unavailable",
            observed_at_utc=f"{latest_flow_date}T21:00:00+00:00" if (latest_flow_date and etf_pit_ok) else None,
            published_at_utc=etf_pub_iso if etf_pit_ok else None,
            available_at_utc=etf_pub_iso if etf_pit_ok else None,
            available_at_known=etf_fresh == "lagged_usable",
            retrieved_at_utc=etf_obj.get("retrieved_at_utc") or cutoff_iso,
            freshness_status=etf_fresh,
            applies_to_symbols=["BTCUSDT"],
            applies_to_horizons=["swing_1_3d"],
            value_summary={
                "usable": etf_fresh == "lagged_usable",
                "compiler_mode": etf_mode,
                "api_responded_ok": bool(etf_obj.get("usable_for_btc_research")),
                "latest_row_is_partial": partial_unreported,
                "latest_row_unreported_funds": latest_row.get("unreported_fund_count"),
                "latest_complete_date": latest_complete.get("flow_date") if etf_pit_ok else None,
                "latest_complete_musd": round(float(latest_complete_val), 3) if (latest_complete_val is not None and etf_pit_ok) else None,
                "sum_3_complete_days_musd": etf_3d_sum,
                "dashes_treated_as_zero": False,
            },
            direction_by_symbol=etf_dir,
            causal_driver_group="btc_spot_etf_flow",
            limitation_note=(
                "最新交易日存在横线/缺报基金（strictly unknown，绝不填零）；"
                if partial_unreported
                else ""
            ) + ("历史回放不回填现货 ETF 数据；" if not etf_pit_ok else "第三方汇编日度已公布完结交易日净流，非盘中实时申赎流水"),
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
    oc_pit_ok = (not replay_mode) or bool(oc.get("historical_point_in_time_verified"))
    oc_metrics = oc.get("metrics") or {}
    last_blk = oc_metrics.get("last_block") or {}
    oc_usable = bool(oc_pit_ok and oc.get("usable_for_btc_research") and last_blk.get("observation_at_utc"))
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
                observed_at_utc=last_blk.get("observation_at_utc") if oc_pit_ok else None,
                published_at_utc=last_blk.get("observation_at_utc") if oc_pit_ok else None,
                available_at_utc=last_blk.get("observation_at_utc") if oc_pit_ok else None,
                available_at_known=oc_usable,
                retrieved_at_utc=oc.get("retrieved_at_utc") or cutoff_iso,
                freshness_status="fresh" if oc_usable else "missing",
                applies_to_symbols=["BTCUSDT"],
                applies_to_horizons=["intraday", "swing_1_3d"],
                value_summary={
                    "tip_height": (oc_metrics.get("tip_height") or {}).get("value") if oc_pit_ok else None,
                    "mempool_tx_count": (oc_metrics.get("mempool") or {}).get("count") if oc_pit_ok else None,
                    "fastest_fee_sat_vb": fastest_fee if oc_pit_ok else None,
                    "measure": "network_activity_and_fee_congestion_not_exchange_flow",
                    "api_responded_ok": bool(oc.get("usable_for_btc_research")),
                    "reason": None if oc_usable else ("not_available_point_in_time_in_replay" if not oc_pit_ok else (oc.get("error") or "unavailable")),
                },
                direction_by_symbol=oc_dir,
                causal_driver_group="btc_onchain_network",
                limitation_note="仅体现比特币全网区块高度与内存池拥堵费用，不等于交易所净流入/流出；历史回放不回填",
            ),
            cutoff_utc=cutoff_utc,
            max_age_hours=float(fcfg["onchain_tip_max_age_hours"]),
        )
    )

    ex_net = exchange_netflow_ctx or {}
    ex_pit_ok = (not replay_mode) or bool(ex_net.get("historical_point_in_time_verified")) or bool(ex_net.get("available_at_utc"))
    ex_avail_dt = _ts_utc(ex_net.get("available_at_utc") or ex_net.get("observed_at_utc"))
    ex_obs_dt = _ts_utc(ex_net.get("observed_at_utc"))
    ex_time_ok = (ex_avail_dt is not None and ex_avail_dt <= cutoff_utc and (cutoff_utc - ex_avail_dt).total_seconds() <= 72 * 3600)
    ex_avail = bool(ex_pit_ok and ex_net.get("usable") and ex_net.get("netflow_btc") is not None and ex_time_ok)
    ex_fresh = "future_excluded" if (ex_avail_dt is not None and ex_avail_dt > cutoff_utc) else ("fresh" if (ex_avail and (cutoff_utc - ex_avail_dt).total_seconds() <= 48 * 3600) else ("lagged_usable" if ex_avail else "missing"))
    ex_dir = {s: "unknown" for s in SYMBOLS}
    if ex_avail:
        nf = float(ex_net["netflow_btc"])
        nf_thr = float(mcfg.get("btc_exchange_netflow_btc_min", 1500.0))
        ex_dir["BTCUSDT"] = "bearish" if nf >= nf_thr else ("bullish" if nf <= -nf_thr else "neutral")
    atoms.append(
        make_atom(
            evidence_id="sentiment:btc:onchain_exchange_netflow",
            domain="sentiment_flow",
            factor_name="BTC 交易所链上净流入/流出",
            source_name=str(ex_net.get("source_name") or "unavailable_in_free_public_feeds"),
            source_url=ex_net.get("source_url"),
            raw_sha256=ex_net.get("raw_sha256"),
            data_tier="lagged_official_report" if ex_avail else "unavailable",
            observed_at_utc=ex_obs_dt.isoformat() if ex_obs_dt else ex_net.get("observed_at_utc"),
            published_at_utc=ex_avail_dt.isoformat() if ex_avail_dt else ex_net.get("available_at_utc"),
            available_at_utc=ex_avail_dt.isoformat() if ex_avail_dt else ex_net.get("available_at_utc"),
            available_at_known=ex_avail,
            retrieved_at_utc=cutoff_iso,
            freshness_status=ex_fresh,
            applies_to_symbols=["BTCUSDT"],
            applies_to_horizons=["intraday", "swing_1_3d"] if ex_avail else ["swing_1_3d"],
            value_summary={
                "netflow_btc": ex_net.get("netflow_btc"),
                "netflow_usd_musd": ex_net.get("netflow_usd_musd"),
                "flow_in_btc": ex_net.get("flow_in_btc"),
                "flow_out_btc": ex_net.get("flow_out_btc"),
                "metric_status": ex_net.get("metric_status"),
                "status": "available" if ex_avail else (ex_net.get("reason") or "unavailable"),
            },
            direction_by_symbol=ex_dir,
            causal_driver_group="btc_exchange_onchain_flow",
            limitation_note=(
                "CoinMetrics 社区 API 按 status-time 严格门控的交易所链上净流入/流出（日频/flash；绝不与 mempool 混为一谈）"
                if ex_avail
                else "当前未获取到经 status-time 核验的交易所钱包净流数据（保持缺失/未知，不与 mempool 混为一谈）"
            ),
        )
    )

    # 3E. Nasdaq-100: VIX (can_decide_direction_alone=False!) + Relative Breadth (ALFRED + QQEW/QQQ 1H) + SEC EDGAR Events
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

    im_raw_b = intraday_macro or {}
    im_pit_b = (not replay_mode) or bool(im_raw_b.get("historical_point_in_time_verified"))
    qqew_qqq_5d = im_raw_b.get("qqew_minus_qqq_5d_pct") if im_pit_b else None
    qqew_qqq_24h = im_raw_b.get("qqew_minus_qqq_24h_pct") if im_pit_b else None
    qqew_obs_dt = _ts_utc((((im_raw_b.get("series") or {}).get("QQEW")) or {}).get("obs_at_utc"))
    qqew_ok = bool(im_pit_b and qqew_qqq_5d is not None and qqew_obs_dt is not None and qqew_obs_dt <= cutoff_utc and (cutoff_utc - qqew_obs_dt).total_seconds() <= 72 * 3600)
    if qqew_ok and rel_spread is None:
        eq_thr = float(mcfg.get("qqew_qqq_5d_spread_pct", 0.75))
        if float(qqew_qqq_5d) >= eq_thr:
            rel_dir["NDX100USDT"] = "bullish"  # broad equal-weight participation
        elif float(qqew_qqq_5d) <= -eq_thr:
            rel_dir["NDX100USDT"] = "bearish"  # narrow cap-weight divergence
        else:
            rel_dir["NDX100USDT"] = "neutral"

    rs_has = (rel_spread is not None) or qqew_ok
    rs_fresh = "fresh" if (qqew_ok and (cutoff_utc - qqew_obs_dt).total_seconds() <= 24 * 3600) else ("lagged_usable" if rs_has else "missing")
    atoms.append(
        make_atom(
            evidence_id="sentiment:ndx:tech_relative_strength",
            domain="sentiment_flow",
            factor_name="科技相对大盘强度 (NASDAQ100/SP500 + QQEW/QQQ 广度代理)",
            source_name=(
                "ALFRED + Yahoo QQEW/QQQ 1H"
                if (rel_spread is not None and qqew_ok)
                else ("Yahoo Public Chart (QQEW vs QQQ 1H)" if qqew_ok else "ALFRED (St. Louis Fed)")
            ),
            source_url=((((im_raw_b.get("series") or {}).get("QQEW")) or {}).get("source_url") if qqew_ok else (ndx_s.get("source_url") or spx_s.get("source_url"))),
            raw_sha256=((((im_raw_b.get("series") or {}).get("QQEW")) or {}).get("raw_sha256") if qqew_ok else ndx_s.get("raw_sha256")),
            data_tier="intraday_delayed_public" if qqew_ok else "latest_daily_vintage",
            observed_at_utc=qqew_obs_dt.isoformat() if qqew_ok and qqew_obs_dt else ndx_s.get("published_or_observed_utc"),
            published_at_utc=qqew_obs_dt.isoformat() if qqew_ok and qqew_obs_dt else None,
            available_at_utc=qqew_obs_dt.isoformat() if qqew_ok and qqew_obs_dt else ndx_s.get("available_at_utc"),
            available_at_known=rs_has,
            retrieved_at_utc=(im_raw_b.get("retrieved_at_utc") if qqew_ok else ndx_s.get("retrieved_at_utc")) or cutoff_iso,
            freshness_status=rs_fresh,
            applies_to_symbols=["NDX100USDT"],
            applies_to_horizons=["intraday", "swing_1_3d"] if qqew_ok else ["swing_1_3d"],
            value_summary={
                "ndx_5obs_pct": ndx_s.get("pct_change_5_obs"),
                "spx_5obs_pct": spx_s.get("pct_change_5_obs"),
                "ndx_minus_spx_pct": rel_spread,
                "qqew_minus_qqq_5d_pct": qqew_qqq_5d,
                "qqew_minus_qqq_24h_pct": qqew_qqq_24h,
                "breadth_measure": "equal_vs_cap_weight_qqew_qqq_proxy" if qqew_ok else "unavailable_no_free_verified_feed",
            },
            direction_by_symbol=rel_dir,
            causal_driver_group="ndx_tech_breadth",
            limitation_note=(
                "已接入 QQEW(等权)/QQQ(市值加权) 1H 相对广度代理及 ALFRED 日频相对强弱（受美股交易时段约束，仍非盘中全成分涨跌家数）"
                if qqew_ok
                else "使用日频 NASDAQ100/SP500 相对表现代理；纳指成分股盘中涨跌家数与成交广度缺失"
            ),
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
    cot_pit_ok = (not replay_mode) or bool(cot.get("historical_point_in_time_verified"))
    cot_ok = bool(cot_pit_ok and cot.get("usable_for_gold_research") and cot.get("commitments_contracts"))
    cot_comm = cot.get("commitments_contracts") or {}
    net_nc = (int(cot_comm["noncommercial_long"]) - int(cot_comm["noncommercial_short"])) if (cot_pit_ok and "noncommercial_long" in cot_comm and "noncommercial_short" in cot_comm) else None
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
            data_tier="lagged_official_report" if cot_pit_ok else "unavailable",
            observed_at_utc=f"{cot['asof_date']}T20:30:00+00:00" if (cot_pit_ok and cot.get("asof_date")) else None,
            published_at_utc=f"{cot['asof_date']}T20:30:00+00:00" if (cot_pit_ok and cot.get("asof_date")) else None,
            available_at_utc=None,
            available_at_known=False,
            retrieved_at_utc=cot.get("retrieved_at_utc") or cutoff_iso,
            freshness_status="lagged_usable" if cot_ok else "missing",
            applies_to_symbols=["XAUUSDT"],
            applies_to_horizons=["swing_1_3d"],
            value_summary={
                "asof_date": cot.get("asof_date") if cot_pit_ok else None,
                "noncommercial_net_contracts": net_nc,
                "open_interest_contracts": cot.get("open_interest_contracts") if cot_pit_ok else None,
                "is_lagged_weekly_futures": True,
                "reason": None if cot_ok else ("not_available_point_in_time_in_replay" if not cot_pit_ok else "unavailable"),
            },
            direction_by_symbol=cot_dir,
            causal_driver_group="gold_cot_positioning",
            can_decide_direction_alone=False,
            limitation_note="COT 为滞后周度 COMEX 期货持仓披露（截至周二、周五发布），不能当作即时买盘或黄金 ETF 资金流；历史回放不回填",
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

    # 3G. Polymarket (strict topical, spread<=0.05, book age<=5m, zero future-timestamp filter)
    pm = polymarket_ctx or {}
    pm_pit_ok = (not replay_mode) or bool(pm.get("historical_point_in_time_verified"))
    pm_max_spread = float(fcfg["polymarket_max_spread"])
    pm_max_age_s = float(fcfg["polymarket_book_max_age_minutes"]) * 60.0
    pm_selected: list[dict[str, Any]] = []
    pm_audit_counts = {"dropped_future_ts": 0, "dropped_stale_ts": 0, "dropped_missing_ts": 0, "dropped_spread_or_illiquid": 0}
    raw_pm_events = (pm.get("events") or []) if pm_pit_ok else []
    pm_api_responded_ok = bool(pm_pit_ok and pm.get("usable_for_research") and len(raw_pm_events) > 0)

    for ev in raw_pm_events:
        if not isinstance(ev, dict):
            continue
        raw_rel = (ev.get("asset_mapping") or {}).get("relevant_instruments") or ev.get("relevant_instruments") or []
        rel = [s for s in raw_rel if s in SYMBOLS]
        if not rel:
            continue
        deadline = _ts_utc(ev.get("deadline_utc") or ev.get("end_date"))
        if deadline is not None and deadline < cutoff_utc:
            continue
        for m in ev.get("markets") or []:
            if not isinstance(m, dict) or m.get("closed"):
                continue
            for o in m.get("outcomes") or []:
                if not isinstance(o, dict):
                    continue
                book = o.get("book") or {}
                book_asof = _ts_utc(book.get("asof_utc") or pm.get("asof_utc"))
                if book_asof is None:
                    pm_audit_counts["dropped_missing_ts"] += 1
                    continue
                if book_asof > cutoff_utc:
                    pm_audit_counts["dropped_future_ts"] += 1
                    continue
                age_s = book.get("book_age_seconds")
                computed_age = (cutoff_utc - book_asof).total_seconds()
                eff_age = max(float(age_s), computed_age) if age_s is not None else computed_age
                if eff_age > pm_max_age_s:
                    pm_audit_counts["dropped_stale_ts"] += 1
                    continue
                spread = book.get("official_spread")
                if spread is None:
                    spread = book.get("spread_from_book") if book.get("spread_from_book") is not None else book.get("spread")
                usable = book.get("usable_for_implied_probability") is True
                if not usable or (spread is not None and float(spread) > pm_max_spread):
                    pm_audit_counts["dropped_spread_or_illiquid"] += 1
                    continue
                mid = book.get("official_midpoint")
                if mid is None:
                    mid = book.get("midpoint_from_book") if book.get("midpoint_from_book") is not None else book.get("midpoint")
                chg24_obj = book.get("change_24h") or o.get("change_24h") or {}
                chg_to_dt = _ts_utc(chg24_obj.get("to_timestamp_utc"))
                chg24_pp = chg24_obj.get("change_24h_pp") if (chg_to_dt is None or chg_to_dt <= cutoff_utc) else None
                pm_selected.append(
                    {
                        "event_title": ev.get("title") or ev.get("question"),
                        "question": m.get("question") or ev.get("question"),
                        "category": ev.get("category"),
                        "mapping_kind": (ev.get("asset_mapping") or {}).get("mapping_kind"),
                        "outcome_name": o.get("name"),
                        "implied_odds_mid": round(float(mid), 4) if mid is not None else o.get("platform_quoted_price"),
                        "change_24h_pp": round(float(chg24_pp), 3) if chg24_pp is not None else None,
                        "observed_at_utc": book_asof.isoformat(),
                        "spread": round(float(spread), 5) if spread is not None else None,
                        "relevant_symbols": rel,
                        "url": ev.get("url") or m.get("url") or m.get("source_url"),
                    }
                )

    pm_has = bool(pm_selected)
    if pm_has:
        pm_fresh = "fresh"
        pm_reason = None
    elif not pm_pit_ok:
        pm_fresh = "missing"
        pm_reason = "not_available_point_in_time_in_replay"
    elif pm_audit_counts["dropped_future_ts"] > 0 and pm_audit_counts["dropped_stale_ts"] == 0 and pm_audit_counts["dropped_spread_or_illiquid"] == 0:
        pm_fresh = "future_excluded"
        pm_reason = "all_candidate_books_after_cutoff_future_excluded"
    elif pm_audit_counts["dropped_stale_ts"] > 0 and pm_audit_counts["dropped_spread_or_illiquid"] == 0:
        pm_fresh = "stale_excluded"
        pm_reason = "all_candidate_books_exceeded_5m_age"
    elif pm_api_responded_ok:
        pm_fresh = "missing"
        pm_reason = "api_ok_no_liquid_tight_spread_market_passed_filters"
    else:
        pm_fresh = "missing"
        pm_reason = pm.get("error") or "unavailable"

    atoms.append(
        make_atom(
            evidence_id="sentiment:polymarket:filtered_events",
            domain="sentiment_flow",
            factor_name="Polymarket 相关预测市场赔率 (点差<=0.05且5分钟内有效)",
            source_name="Polymarket Gamma + CLOB",
            source_url="https://gamma-api.polymarket.com",
            raw_sha256=canonical_sha256(pm_selected) if pm_has else None,
            data_tier="realtime_verified" if pm_has else "unavailable",
            observed_at_utc=pm_selected[0]["observed_at_utc"] if pm_has else (pm.get("asof_utc") if pm_pit_ok else None),
            published_at_utc=pm_selected[0]["observed_at_utc"] if pm_has else (pm.get("asof_utc") if pm_pit_ok else None),
            available_at_utc=pm_selected[0]["observed_at_utc"] if pm_has else (pm.get("asof_utc") if pm_pit_ok else None),
            available_at_known=pm_has,
            retrieved_at_utc=pm.get("retrieved_at_utc") or cutoff_iso,
            freshness_status=pm_fresh,
            applies_to_symbols=list(SYMBOLS),
            applies_to_horizons=["intraday", "swing_1_3d"],
            value_summary={
                "selected_count": len(pm_selected),
                "events": pm_selected[:5],
                "api_responded_ok": pm_api_responded_ok,
                "audit_counts": pm_audit_counts,
                "missing_or_excluded_reason": pm_reason,
                "is_market_pricing_not_fact": True,
            },
            direction_by_symbol={s: ("neutral" if pm_has else "unknown") for s in SYMBOLS},
            causal_driver_group="polymarket_odds_context",
            can_decide_direction_alone=False,
            limitation_note="赔率属于实时流动性盘口市场定价，不是既定事实或校准概率，仅作背景参考；API 成功不等于因子参与判断",
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
        "btc_etf_latest_musd": etf_latest_single,
        "btc_etf_compiler_mode": etf_mode,
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
                published_at_utc=neural_block["input_cutoff_utc"],
                available_at_utc=neural_block["input_cutoff_utc"],
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

    started_now = (now_override or datetime.now(timezone.utc)).astimezone(timezone.utc)
    bar_cutoff_utc = started_now.replace(minute=0, second=0, microsecond=0)
    decision_cutoff_utc = started_now
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
    alfred_series, alfred_errors = fetch_alfred_public_series(series_wanted, now=decision_cutoff_utc)
    collection_errors.update({f"alfred:{k}": v for k, v in alfred_errors.items()})

    dxy_live = fetch_public_dxy_proxy(cutoff_utc=bar_cutoff_utc)
    try:
        intraday_macro = fetch_yahoo_intraday_macro_and_breadth(bar_cutoff_utc=bar_cutoff_utc)
        collection_errors.update({f"yahoo_intraday:{k}": v for k, v in (intraday_macro.get("errors") or {}).items()})
    except Exception as exc:
        intraday_macro = {"ok": False, "series": {}, "errors": {"yahoo": type(exc).__name__}}
        collection_errors["yahoo_intraday"] = type(exc).__name__

    try:
        event_calendar = fetch_official_event_calendar(now=decision_cutoff_utc)
    except Exception as exc:
        event_calendar = {"present_sources": [], "sources": {}, "errors": {"calendar": type(exc).__name__}}
        collection_errors["event_calendar"] = type(exc).__name__

    try:
        etf_ctx = fetch_us_spot_btc_etf_flows(now=decision_cutoff_utc)
        if not etf_ctx.get("usable_for_btc_research"):
            etf_ctx = fetch_tftc_etf(now=decision_cutoff_utc)
    except Exception as exc:
        etf_ctx = {"usable_for_btc_research": False, "error": type(exc).__name__}
        collection_errors["btc_etf"] = type(exc).__name__

    try:
        fear_greed_ctx = fetch_fear_greed(now=decision_cutoff_utc)
    except Exception as exc:
        fear_greed_ctx = {"usable_for_research": False, "error": type(exc).__name__}
        collection_errors["fear_greed"] = type(exc).__name__

    try:
        gold_cot_ctx = fetch_comex_gold_positioning(now=decision_cutoff_utc)
    except Exception as exc:
        gold_cot_ctx = {"usable_for_gold_research": False, "error": type(exc).__name__}
        collection_errors["gold_cot"] = type(exc).__name__

    try:
        onchain_ctx = fetch_btc_onchain(now=decision_cutoff_utc)
    except Exception as exc:
        onchain_ctx = {"usable_for_btc_research": False, "error": type(exc).__name__}
        collection_errors["btc_onchain"] = type(exc).__name__

    try:
        exchange_netflow_ctx = fetch_coinmetrics_btc_exchange_flows(decision_cutoff_utc=decision_cutoff_utc)
        if not exchange_netflow_ctx.get("usable") and exchange_netflow_ctx.get("reason"):
            collection_errors["coinmetrics_netflow"] = str(exchange_netflow_ctx["reason"])
    except Exception as exc:
        exchange_netflow_ctx = {"usable": False, "reason": type(exc).__name__}
        collection_errors["coinmetrics_netflow"] = type(exc).__name__

    try:
        venue_micro = fetch_bitget_venue_microstructure(symbols, decision_cutoff_utc=decision_cutoff_utc)
        collection_errors.update({f"venue_micro:{k}": v for k, v in (venue_micro.get("errors") or {}).items()})
    except Exception as exc:
        venue_micro = {"ok": False, "symbols": {}, "errors": {"venue": type(exc).__name__}}
        collection_errors["venue_micro"] = type(exc).__name__

    try:
        polymarket_ctx = fetch_prediction_markets(now=decision_cutoff_utc)
    except Exception as exc:
        polymarket_ctx = {"usable_for_research": False, "events": [], "error": type(exc).__name__}
        collection_errors["polymarket"] = type(exc).__name__

    rss_items, rss_err = fetch_official_news_feeds(now=decision_cutoff_utc, include_corporate_ir=True)
    edgar_items, edgar_err = fetch_ndx_sec_edgar_filings(cutoff_utc=decision_cutoff_utc, retrieved_at=started_now)
    collection_errors.update({f"news_rss:{k}": v for k, v in rss_err.items()})
    collection_errors.update({f"sec_edgar:{k}": v for k, v in edgar_err.items()})

    series_for_fc = {}
    for sym in symbols:
        inst = (market_snap.get("instruments") or {}).get(sym) or {}
        c1h = sanitize_closed_candles(inst.get("candles_closed_1h"), bar_cutoff_utc=bar_cutoff_utc)["all_sanitized"]
        if c1h:
            series_for_fc[sym] = {"closes": [float(c["close"]) for c in c1h], "last_close_at_utc": c1h[-1]["close_at_utc"], "timeframe": "1H_closed"}
    try:
        with tempfile.TemporaryDirectory(prefix="piia_chronos_") as tmpdir:
            fc_path = Path(tmpdir) / "chronos_out.json"
            chronos_payload = run_chronos_tiny(series_for_fc, fc_path, allow_download=False)
        chronos_payload["run_at_utc"] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        chronos_payload = {"weights_ran": False, "reason": type(exc).__name__, "forecasts": {}}

    collected_now = datetime.now(timezone.utc)
    if now_override is None:
        decision_cutoff_utc = max(decision_cutoff_utc, collected_now)
    collected_at_utc = collected_now.isoformat()
    return {
        "is_synthetic_demo": False,
        "replay_mode": False,
        "first_retrieved_at_utc": started_now.isoformat(),
        "collected_at_utc": collected_at_utc,
        "bar_cutoff_utc": bar_cutoff_utc.isoformat(),
        "decision_cutoff_utc": decision_cutoff_utc.isoformat(),
        "cutoff_utc": bar_cutoff_utc.isoformat(),
        "market_snapshot": market_snap,
        "books_snapshot": books_snap,
        "tiers_snapshot": tiers_snap,
        "alfred_series": alfred_series,
        "dxy_live": dxy_live,
        "intraday_macro": intraday_macro,
        "event_calendar": event_calendar,
        "fear_greed": fear_greed_ctx,
        "etf_flows": etf_ctx,
        "gold_cot": gold_cot_ctx,
        "onchain": onchain_ctx,
        "exchange_netflow": exchange_netflow_ctx,
        "venue_microstructure": venue_micro,
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
