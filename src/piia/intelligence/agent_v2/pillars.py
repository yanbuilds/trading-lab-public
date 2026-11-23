"""Six-Pillar Evidence Assembler with Strict Point-in-Time Provenance (PIIA v2).

Pillars:
1. `market`: Closed 1h/4h/1d candles, causal indicators, funding rate, OI,
   and (in live mode) visible v2 orderbook depth + isolated position tiers.
2. `macro`: Official ALFRED yields, 10Y TIPS real rate (DFII10), broad dollar
   (DTWEXBGS), RRP/TGA liquidity, and official U.S. Fed/BLS/BEA event calendar.
   In historical mode, only observations whose conservative `available_at_utc`
   (`observation_date + 2 UTC days`) is `<= decision_utc` may enter; intraday
   release timestamps remain explicitly marked unverified.
3. `flows_positioning`: US Spot BTC ETF net flows (Farside / TFTC fallback,
   BTCUSDT only), CFTC COMEX Gold COT (XAUUSDT context), and deterministic
   `liquidity_positioning` joint state. Marked `not_available_point_in_time`
   in historical bar replay unless archived with verified `available_at_utc`.
4. `onchain`: `mempool.space` BTC chain tip, fee pressure, and mining state
   (BTCUSDT only; prospective/live only).
5. `polymarket`: Official Gamma + CLOB orderbook implied probabilities & 24h
   delta sample (uncalibrated market context; prospective/live only).
6. `forecast`: Local pinned Chronos (`t5-tiny` / `Chronos-2`) 24h quantiles when
   available, paired with the causal 168h log-return `statistical_drift_forecast`
   benchmark.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from piia.intelligence.forecast_benchmark import random_walk_drift_forecast
from piia.intelligence.hashing import content_hash
from trading_lab.indicators_v2 import (
    adx_di,
    atr,
    bollinger_bands,
    chandelier_stops,
    donchian_channels,
    ema,
    rsi,
    sma,
)

ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = ROOT / "config" / "piia_agent_v2.yaml"
SYMBOLS = ("BTCUSDT", "XAUUSDT", "NDX100USDT")
DEFAULT_FUNDING_INTERVAL_HOURS = {"BTCUSDT": 8, "NDX100USDT": 8, "XAUUSDT": 4}
BITGET_HISTORY_FUND_URL = "https://api.bitget.com/api/v2/mix/market/history-fund-rate"
BITGET_CURRENT_FUND_URL = "https://api.bitget.com/api/v2/mix/market/current-fund-rate"


def load_v2_config(path: Path | None = None) -> dict[str, Any]:
    target = Path(path) if path else CONFIG_PATH
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("protocol_id") != "PIIA_PERSONAL_INVESTMENT_INTELLIGENCE_AGENT_V2":
        raise ValueError("Invalid piia_agent_v2.yaml")
    return data


def canonical_sha256(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _completed_4h_emas(h1: pd.DataFrame, fast_p: int = 20, slow_p: int = 50) -> tuple[np.ndarray, np.ndarray]:
    if h1.empty:
        return np.array([], dtype=float), np.array([], dtype=float)
    grouped = h1["close"].resample("4h", label="right", closed="left", origin="epoch")
    c4 = grouped.last().where(grouped.count() == 4)
    vals = c4.to_numpy(dtype=float)
    valid = np.isfinite(vals)
    f_arr = np.full(len(vals), np.nan, dtype=float)
    s_arr = np.full(len(vals), np.nan, dtype=float)
    if valid.sum() >= slow_p:
        comp = vals[valid]
        f_arr[valid] = ema(comp, fast_p)
        s_arr[valid] = ema(comp, slow_p)
    close_times = h1.index + pd.Timedelta(hours=1)
    f_out = pd.Series(f_arr, index=c4.index).reindex(close_times, method="ffill").to_numpy(dtype=float)
    s_out = pd.Series(s_arr, index=c4.index).reindex(close_times, method="ffill").to_numpy(dtype=float)
    return f_out, s_out


def prepare_market_feature_table(h1: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Compute strictly causal closed-bar market features for fast replay & live use."""
    cfg = cfg or load_v2_config()
    mcfg = cfg["evidence_pillars"]["market"]
    fcfg = cfg["evidence_pillars"]["forecast"]

    out = h1.copy()
    high = out["high"].to_numpy(dtype=float)
    low = out["low"].to_numpy(dtype=float)
    close = out["close"].to_numpy(dtype=float)

    htf_f, htf_s = _completed_4h_emas(out, int(mcfg["ema_fast"]), int(mcfg["ema_slow"]))
    out["htf_ema20"] = htf_f
    out["htf_ema50"] = htf_s
    out["ema20"] = ema(close, int(mcfg["ema_fast"]))
    out["ema50"] = ema(close, int(mcfg["ema_slow"]))
    out["atr14"] = atr(high, low, close, int(mcfg["atr_period"]))
    out["atr14_mean50"] = sma(out["atr14"].to_numpy(dtype=float), int(mcfg["atr_mean_period"]))
    adx_v, pdi_v, mdi_v = adx_di(high, low, close, int(mcfg["adx_period"]))
    out["adx14"] = adx_v
    out["plus_di14"] = pdi_v
    out["minus_di14"] = mdi_v
    out["rsi14"] = rsi(close, int(mcfg["rsi_period"]))

    bb_mid, bb_up, bb_lo, bb_bw, bb_pct = bollinger_bands(
        close, int(mcfg["bollinger_period"]), float(mcfg["bollinger_std"])
    )
    out["bb_mid"] = bb_mid
    out["bb_upper"] = bb_up
    out["bb_lower"] = bb_lo
    out["bb_bandwidth"] = bb_bw
    out["bb_pct_b"] = bb_pct

    don_hi, don_lo = donchian_channels(high, low, int(mcfg["donchian_breakout_bars"]), shift_bars=1)
    out["donchian_high"] = don_hi
    out["donchian_low"] = don_lo

    ch_long, ch_short = chandelier_stops(
        high, low, out["atr14"].to_numpy(dtype=float),
        int(mcfg["chandelier_period"]), float(mcfg["chandelier_atr_mult"]), shift_bars=1,
    )
    out["chandelier_long"] = ch_long
    out["chandelier_short"] = ch_short

    # Vectorized causal 168h log-return drift forecast (matches `random_walk_drift_forecast`)
    lookback = int(fcfg["benchmark_lookback_hours"])
    horizon = int(fcfg["horizon_hours"])
    log_c = np.log(np.where(close > 0, close, np.nan))
    log_ret = pd.Series(log_c, index=out.index).diff()
    drift_h = log_ret.rolling(lookback, min_periods=lookback).mean()
    sigma_h = log_ret.rolling(lookback, min_periods=lookback).std(ddof=1)
    half_band = 1.96 * sigma_h * math.sqrt(horizon)
    out["fc_point_24h"] = (close * np.exp(drift_h.to_numpy(dtype=float) * horizon))
    out["fc_low_24h"] = (close * np.exp(drift_h.to_numpy(dtype=float) * horizon - half_band.to_numpy(dtype=float)))
    out["fc_high_24h"] = (close * np.exp(drift_h.to_numpy(dtype=float) * horizon + half_band.to_numpy(dtype=float)))
    return out


def point_in_time_macro_context(
    macro_series_observations: dict[str, list[dict[str, Any]]] | None,
    decision_utc: datetime,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract strictly lagged macro changes (`available_at_utc <= decision_utc`).

    Never uses observations whose conservative `observation_date + lag_days`
    falls after `decision_utc`.
    """
    cfg = cfg or load_v2_config()
    mcfg = cfg["evidence_pillars"]["macro"]
    lag_days = int(mcfg["point_in_time_daily_lag_utc_days"])
    lookback_obs = int(mcfg["lookback_trading_obs"])
    decision_utc = decision_utc.astimezone(timezone.utc)

    if not macro_series_observations:
        return {
            "status": "unavailable",
            "historical_point_in_time_lagged": False,
            "intraday_release_verified": False,
            "reason": "no_macro_observations_supplied",
            "series": {},
        }

    extracted: dict[str, Any] = {}
    for sid, obs_list in macro_series_observations.items():
        usable: list[tuple[date, float, datetime]] = []
        for row in obs_list or []:
            try:
                obs_d = date.fromisoformat(str(row["observation_date"])[:10])
                val = float(row["value"])
                if not math.isfinite(val):
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            avail_utc = datetime(obs_d.year, obs_d.month, obs_d.day, tzinfo=timezone.utc) + timedelta(days=lag_days)
            if avail_utc <= decision_utc:
                usable.append((obs_d, val, avail_utc))
        usable.sort(key=lambda x: x[0])
        if not usable:
            continue
        latest_d, latest_v, latest_avail = usable[-1]
        age_days = (decision_utc.date() - latest_d).days
        delta_5 = None
        pct_change_5 = None
        if len(usable) > lookback_obs:
            prev_v = usable[-(lookback_obs + 1)][1]
            delta_5 = latest_v - prev_v
            if prev_v != 0:
                pct_change_5 = (latest_v / prev_v - 1.0) * 100.0
        extracted[sid] = {
            "observation_date": latest_d.isoformat(),
            "available_at_utc": latest_avail.isoformat(),
            "value": latest_v,
            "delta_5_obs": round(delta_5, 6) if delta_5 is not None else None,
            "pct_change_5_obs": round(pct_change_5, 6) if pct_change_5 is not None else None,
            "stale": age_days > 16,
        }

    dfii = extracted.get("DFII10") or {}
    dtwex = extracted.get("DTWEXBGS") or {}
    dgs10 = extracted.get("DGS10") or {}
    dgs2 = extracted.get("DGS2") or {}

    ry_delta = dfii.get("delta_5_obs")
    usd_chg = dtwex.get("pct_change_5_obs")
    ry_thresh = float(mcfg["real_yield_headwind_delta_pct"])
    usd_thresh = float(mcfg["broad_dollar_headwind_change_pct"])

    tightening_headwind = bool(
        ry_delta is not None and usd_chg is not None
        and not dfii.get("stale") and not dtwex.get("stale")
        and ry_delta >= ry_thresh and usd_chg >= usd_thresh
    )
    easing_tailwind = bool(
        ry_delta is not None and usd_chg is not None
        and not dfii.get("stale") and not dtwex.get("stale")
        and ry_delta <= -ry_thresh and usd_chg <= -usd_thresh
    )
    curve_spread = (
        round(float(dgs10["value"]) - float(dgs2["value"]), 4)
        if (dgs10.get("value") is not None and dgs2.get("value") is not None)
        else None
    )

    body = {
        "status": "available" if extracted else "unavailable",
        "decision_utc": decision_utc.isoformat(),
        "lag_days_applied": lag_days,
        "intraday_release_verified": False,
        "historical_point_in_time_lagged": True,
        "note": (
            "Conservative +2 UTC calendar day lag applied to daily ALFRED observations; "
            "intraday exact release ticks and historical intra-week revisions are not claimed."
        ),
        "curve_10y_minus_2y": curve_spread,
        "macro_tightening_headwind": tightening_headwind,
        "macro_easing_tailwind": easing_tailwind,
        "series": extracted,
    }
    body["evidence_hash"] = canonical_sha256(body)
    return body


def build_bar_evidence_for_symbol(
    symbol: str,
    *,
    feature_row: pd.Series,
    prev_close: float | None,
    decision_utc: datetime,
    funding_rate_8h: float | None,
    macro_ctx: dict[str, Any] | None = None,
    external_live_pillars: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical, hash-stamped 6-pillar evidence bundle at `decision_utc`."""
    ext = external_live_pillars or {}
    close_px = float(feature_row["close"])
    open_px = float(feature_row["open"])
    high_px = float(feature_row["high"])
    low_px = float(feature_row["low"])

    def _f(col: str) -> float | None:
        v = feature_row.get(col)
        if v is None or isinstance(v, bool):
            return None
        try:
            fv = float(v)
            return fv if math.isfinite(fv) else None
        except (TypeError, ValueError):
            return None

    market_pillar = {
        "symbol": symbol,
        "bar_close_utc": decision_utc.isoformat(),
        "historical_replay_supported": True,
        "ohlc": {"open": open_px, "high": high_px, "low": low_px, "close": close_px},
        "prev_close": prev_close,
        "ema20_1h": _f("ema20"),
        "ema50_1h": _f("ema50"),
        "htf_ema20_4h": _f("htf_ema20"),
        "htf_ema50_4h": _f("htf_ema50"),
        "atr14": _f("atr14"),
        "atr14_mean50": _f("atr14_mean50"),
        "adx14": _f("adx14"),
        "plus_di14": _f("plus_di14"),
        "minus_di14": _f("minus_di14"),
        "rsi14": _f("rsi14"),
        "bb_mid": _f("bb_mid"),
        "bb_upper": _f("bb_upper"),
        "bb_lower": _f("bb_lower"),
        "donchian_high": _f("donchian_high"),
        "donchian_low": _f("donchian_low"),
        "chandelier_long": _f("chandelier_long"),
        "chandelier_short": _f("chandelier_short"),
        "funding_rate_8h": funding_rate_8h,
        "live_orderbook": ext.get("orderbook"),
        "live_position_tiers": ext.get("tiers"),
        "open_interest": ext.get("open_interest"),
    }
    market_pillar["evidence_hash"] = canonical_sha256(market_pillar)

    # Forecast pillar: prefer bound Chronos quantile forecast if supplied & valid, plus causal statistical drift
    chronos_item = ((ext.get("chronos_forecast") or {}).get("forecasts") or {}).get(symbol)
    fc_point = _f("fc_point_24h")
    fc_low = _f("fc_low_24h")
    fc_high = _f("fc_high_24h")
    forecast_pillar: dict[str, Any] = {
        "symbol": symbol,
        "horizon_hours": 24,
        "calibration_status": "not_calibrated",
        "historical_replay_supported": True,
        "statistical_drift": {
            "available": fc_point is not None,
            "last_close": close_px,
            "point_24h": fc_point,
            "interval_low": fc_low,
            "interval_high": fc_high,
        },
        "neural_chronos": (
            {
                "available": True,
                "model_id": (ext.get("chronos_forecast") or {}).get("model_id"),
                "q10_24h": chronos_item["quantile_10"][-1],
                "q50_24h": chronos_item["quantile_50"][-1],
                "q90_24h": chronos_item["quantile_90"][-1],
            }
            if isinstance(chronos_item, dict) and chronos_item.get("ok") and chronos_item.get("quantile_50")
            else {"available": False, "reason": "weights_not_run_or_historical_bar"}
        ),
    }
    forecast_pillar["evidence_hash"] = canonical_sha256(forecast_pillar)

    macro_pillar = macro_ctx or {
        "status": "unavailable",
        "historical_point_in_time_lagged": False,
        "macro_tightening_headwind": False,
        "macro_easing_tailwind": False,
    }
    if "evidence_hash" not in macro_pillar:
        macro_pillar = {**macro_pillar, "evidence_hash": canonical_sha256(macro_pillar)}

    # Contemporaneous-only pillars: explicitly marked unavailable on historical replay!
    flows_pillar = ext.get("flows_positioning") or {
        "status": "not_available_point_in_time",
        "historical_replay_supported": False,
        "note": "Live ETF flows / COT / single-venue liquidation state not backfilled onto historical bars.",
    }
    if "evidence_hash" not in flows_pillar:
        flows_pillar = {**flows_pillar, "evidence_hash": canonical_sha256(flows_pillar)}

    onchain_pillar = (
        ext.get("onchain")
        if symbol == "BTCUSDT" and ext.get("onchain")
        else {
            "status": "not_applicable" if symbol != "BTCUSDT" else "not_available_point_in_time",
            "historical_replay_supported": False,
        }
    )
    if "evidence_hash" not in onchain_pillar:
        onchain_pillar = {**onchain_pillar, "evidence_hash": canonical_sha256(onchain_pillar)}

    polymarket_pillar = ext.get("polymarket") or {
        "status": "not_available_point_in_time",
        "historical_replay_supported": False,
        "odds_are_uncalibrated_context": True,
    }
    if "evidence_hash" not in polymarket_pillar:
        polymarket_pillar = {**polymarket_pillar, "evidence_hash": canonical_sha256(polymarket_pillar)}

    calendar_pillar = ext.get("event_calendar") or {
        "status": "not_available_point_in_time",
        "historical_replay_supported": False,
    }
    if "evidence_hash" not in calendar_pillar:
        calendar_pillar = {**calendar_pillar, "evidence_hash": canonical_sha256(calendar_pillar)}

    bundle = {
        "symbol": symbol,
        "decision_utc": decision_utc.isoformat(),
        "pillars": {
            "market": market_pillar,
            "macro": macro_pillar,
            "flows_positioning": flows_pillar,
            "onchain": onchain_pillar,
            "polymarket": polymarket_pillar,
            "forecast": forecast_pillar,
            "event_calendar": calendar_pillar,
        },
    }
    bundle["bundle_hash"] = canonical_sha256(bundle)
    return bundle


def fetch_public_settled_funding_recent(
    symbols: tuple[str, ...] = SYMBOLS,
    *,
    timeout: float = 12.0,
) -> dict[str, Any]:
    """Fetch and validate recent public Bitget settled funding history per symbol (never invent rates)."""
    import httpx

    retrieved_at = datetime.now(timezone.utc)
    instruments: dict[str, Any] = {}
    all_ok = True

    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        for sym in symbols:
            default_iv = int(DEFAULT_FUNDING_INTERVAL_HOURS.get(sym, 8))
            interval_hours: int | None = None
            interval_verified = False
            try:
                cur_resp = client.get(
                    BITGET_CURRENT_FUND_URL,
                    params={"productType": "USDT-FUTURES", "symbol": sym},
                )
                cur_resp.raise_for_status()
                cur_payload = cur_resp.json()
                if cur_payload.get("code") == "00000":
                    cdata = cur_payload.get("data")
                    crow = cdata[0] if isinstance(cdata, list) and cdata else (cdata if isinstance(cdata, dict) else {})
                    raw_iv = crow.get("fundingRateInterval") if isinstance(crow, dict) else None
                    if raw_iv is not None and not isinstance(raw_iv, bool) and str(raw_iv).isdigit():
                        parsed_iv = int(raw_iv)
                        if parsed_iv > 0 and 24 % parsed_iv == 0:
                            interval_hours = parsed_iv
                            interval_verified = True
            except Exception:
                pass

            eff_iv = interval_hours if interval_hours is not None else default_iv
            source_url = (
                f"{BITGET_HISTORY_FUND_URL}?productType=USDT-FUTURES&symbol={sym}&pageSize=100&pageNo=1"
            )
            try:
                resp = client.get(
                    BITGET_HISTORY_FUND_URL,
                    params={
                        "productType": "USDT-FUTURES",
                        "symbol": sym,
                        "pageSize": "100",
                        "pageNo": "1",
                    },
                )
                resp.raise_for_status()
                raw_bytes = resp.content
                raw_sha = hashlib.sha256(raw_bytes).hexdigest()
                payload = resp.json()
                if payload.get("code") != "00000" or not isinstance(payload.get("data"), list):
                    raise ValueError("invalid_history_response")
                req_time_raw = payload.get("requestTime")
                if isinstance(req_time_raw, bool) or not str(req_time_raw).isdigit():
                    raise ValueError("invalid_request_time")
                server_ms = int(req_time_raw)
                parsed_rows: list[dict[str, Any]] = []
                prev_ms: int | None = None
                for row in payload["data"]:
                    if not isinstance(row, dict) or row.get("symbol") != sym:
                        raise ValueError("funding_symbol_mismatch")
                    ft_raw = row.get("fundingTime")
                    fr_raw = row.get("fundingRate")
                    if isinstance(ft_raw, bool) or not str(ft_raw).isdigit():
                        raise ValueError("invalid_funding_time")
                    if isinstance(fr_raw, bool) or fr_raw is None:
                        raise ValueError("invalid_funding_rate")
                    ts_ms = int(ft_raw)
                    rate = float(fr_raw)
                    if ts_ms <= 0 or ts_ms > server_ms or ts_ms % 3_600_000 != 0:
                        raise ValueError("non_hour_aligned_or_future_funding")
                    if not math.isfinite(rate) or abs(rate) > 1.0:
                        raise ValueError("funding_rate_out_of_bounds")
                    if prev_ms is not None and ts_ms >= prev_ms:
                        raise ValueError("funding_history_not_strictly_descending")
                    prev_ms = ts_ms
                    parsed_rows.append(
                        {
                            "timestamp": ts_ms,
                            "settlement_utc": datetime.fromtimestamp(ts_ms / 1000.0, timezone.utc).isoformat(),
                            "rate": rate,
                        }
                    )
                if not parsed_rows:
                    raise ValueError("empty_funding_history")
                parsed_rows.reverse()  # oldest first
                contiguous = all(
                    (b["timestamp"] - a["timestamp"]) == eff_iv * 3_600_000
                    for a, b in zip(parsed_rows, parsed_rows[1:])
                )
                grid_aligned = all(((r["timestamp"] // 3_600_000) % eff_iv) == 0 for r in parsed_rows)
                schedule_verified = bool(interval_verified and contiguous and grid_aligned)
                instruments[sym] = {
                    "ok": True,
                    "symbol": sym,
                    "source_url": source_url,
                    "raw_sha256": raw_sha,
                    "retrieved_at_utc": retrieved_at.isoformat(),
                    "server_at_utc": datetime.fromtimestamp(server_ms / 1000.0, timezone.utc).isoformat(),
                    "interval_hours": eff_iv,
                    "interval_verified": interval_verified,
                    "schedule_verified": schedule_verified,
                    "records": parsed_rows,
                    "limitations": (
                        "Bitget public history-fund-rate pageNo=1 only; public settled rate proxy, "
                        "not an authenticated account funding statement."
                    ),
                }
            except Exception as exc:
                all_ok = False
                instruments[sym] = {
                    "ok": False,
                    "symbol": sym,
                    "source_url": source_url,
                    "raw_sha256": None,
                    "retrieved_at_utc": retrieved_at.isoformat(),
                    "server_at_utc": None,
                    "interval_hours": eff_iv,
                    "interval_verified": interval_verified,
                    "schedule_verified": False,
                    "error": f"{type(exc).__name__}:{exc}",
                    "records": [],
                }

    return {
        "ok": all_ok,
        "retrieved_at_utc": retrieved_at.isoformat(),
        "instruments": instruments,
    }


def collect_live_six_pillar_snapshot(
    symbols: tuple[str, ...] = SYMBOLS,
    *,
    snapshot_dir: Path | None = None,
    allow_forecast_download: bool = False,
) -> dict[str, Any]:
    """Collect all 6 live public data pillars fail-honestly (never inventing missing feeds)."""
    from piia.intelligence.public_data import fetch_public_snapshot
    from piia.intelligence.public_orderbook import fetch_execution_books
    from piia.intelligence.public_position_tiers import fetch_position_tiers

    now = datetime.now(timezone.utc)
    if snapshot_dir is not None:
        snapshot_dir = Path(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    market_snap = fetch_public_snapshot(list(symbols), candle_limit=300)
    try:
        books_snap = fetch_execution_books(symbols, snapshot_dir=(snapshot_dir / "books") if snapshot_dir else None)
    except Exception as exc:
        books_snap = {"ok": False, "errors": {"fetch": type(exc).__name__}, "instruments": {}}
    try:
        tiers_snap = fetch_position_tiers(symbols, snapshot_dir=(snapshot_dir / "tiers") if snapshot_dir else None)
    except Exception as exc:
        tiers_snap = {"ok": False, "errors": {"fetch": type(exc).__name__}, "instruments": {}}

    # Pillar 2: Macro (ALFRED history + current extended) & Official Event Calendar
    macro_hist = None
    try:
        from piia.intelligence.macro_data import fetch_alfred_history
        raw_hist = fetch_alfred_history(["DGS10", "DGS2", "DFII10", "DFF", "DTWEXBGS", "RRPONTSYD", "WDTGAL"])
        macro_hist = {
            sid: item.get("observations") or []
            for sid, item in (raw_hist.get("series") or {}).items()
        }
    except Exception:
        macro_hist = None
    macro_ctx = point_in_time_macro_context(macro_hist, now)

    try:
        from piia.intelligence.event_calendar import fetch_official_event_calendar
        calendar_ctx = fetch_official_event_calendar(snapshot_dir=(snapshot_dir / "calendar") if snapshot_dir else None)
    except Exception as exc:
        calendar_ctx = {"status": "unavailable", "error": type(exc).__name__, "historical_replay_supported": False}

    # Pillar 3: ETF flows + Gold COT + Liquidity Positioning
    try:
        from piia.intelligence.etf_flows import fetch_us_spot_btc_etf_flows
        from piia.intelligence.etf_tftc import fetch_tftc_etf
        etf_ctx = fetch_us_spot_btc_etf_flows(snapshot_dir=(snapshot_dir / "etf") if snapshot_dir else None)
        if not etf_ctx.get("usable_for_btc_research"):
            etf_ctx = fetch_tftc_etf(snapshot_dir=(snapshot_dir / "etf_fallback") if snapshot_dir else None)
    except Exception as exc:
        etf_ctx = {"usable_for_btc_research": False, "error": type(exc).__name__}

    try:
        from piia.intelligence.gold_cftc import fetch_comex_gold_positioning
        gold_cot = fetch_comex_gold_positioning(snapshot_dir=(snapshot_dir / "gold_cot") if snapshot_dir else None)
    except Exception as exc:
        gold_cot = {"usable_for_research": False, "error": type(exc).__name__}

    try:
        from piia.intelligence.liquidity_positioning import build_liquidity_context
        liq_ctx = build_liquidity_context(market_snap, execution_books=books_snap, mode="current_snapshot", now=now)
    except Exception as exc:
        liq_ctx = {"official_state": "unknown", "error": type(exc).__name__}

    flows_positioning_ctx = {
        "status": "collected_partial_or_complete",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_replay_supported": False,
        "btc_etf_flows": {
            "usable": bool(etf_ctx.get("usable_for_btc_research")),
            "asof_complete_flow_date": etf_ctx.get("asof_complete_flow_date"),
            "latest_complete_total_musd": (etf_ctx.get("latest_complete_day") or {}).get("total_musd")
            if isinstance(etf_ctx.get("latest_complete_day"), dict) else etf_ctx.get("latest_total_musd"),
        },
        "gold_cftc_cot": {
            "usable": bool(gold_cot.get("usable_for_research")),
            "report_date": gold_cot.get("report_date"),
            "noncommercial_net_contracts": gold_cot.get("noncommercial_net_contracts"),
        },
        "liquidity_state": liq_ctx.get("official_state", "unknown"),
    }
    flows_positioning_ctx["evidence_hash"] = canonical_sha256(flows_positioning_ctx)

    # Pillar 4: BTC On-Chain (`mempool.space`)
    try:
        from piia.intelligence.onchain_btc import fetch_btc_onchain
        raw_onchain = fetch_btc_onchain(snapshot_dir=(snapshot_dir / "onchain") if snapshot_dir else None)
        onchain_ctx = {
            "status": "available" if raw_onchain.get("usable_for_btc_research") else "incomplete_or_stale",
            "usable_for_btc_research": bool(raw_onchain.get("usable_for_btc_research")),
            "retrieved_at_utc": raw_onchain.get("retrieved_at_utc"),
            "tip_height": ((raw_onchain.get("metrics") or {}).get("tip_height") or {}).get("value"),
            "fastest_fee_sat_vb": (((raw_onchain.get("metrics") or {}).get("recommended_fees") or {}).get("value") or {}).get("fastestFee"),
            "mempool_count": (((raw_onchain.get("metrics") or {}).get("mempool") or {}).get("value") or {}).get("count"),
            "historical_replay_supported": False,
        }
    except Exception as exc:
        onchain_ctx = {"status": "unavailable", "usable_for_btc_research": False, "error": type(exc).__name__, "historical_replay_supported": False}
    onchain_ctx["evidence_hash"] = canonical_sha256(onchain_ctx)

    # Pillar 5: Polymarket
    try:
        from piia.intelligence.polymarket import fetch_prediction_markets
        raw_pm = fetch_prediction_markets(snapshot_dir=(snapshot_dir / "polymarket") if snapshot_dir else None)
        events_sample = []
        for ev in (raw_pm.get("events") or [])[:5]:
            events_sample.append({
                "title": ev.get("title"),
                "topics": ev.get("topics"),
                "markets_count": len(ev.get("markets") or []),
            })
        polymarket_ctx = {
            "status": "available" if raw_pm.get("usable_for_research") else "unavailable_or_empty",
            "usable_for_research": bool(raw_pm.get("usable_for_research")),
            "asof_utc": raw_pm.get("asof_utc"),
            "events_sampled": len(raw_pm.get("events") or []),
            "sample": events_sample,
            "odds_are_uncalibrated_context": True,
            "historical_replay_supported": False,
        }
    except Exception as exc:
        polymarket_ctx = {"status": "unavailable", "usable_for_research": False, "error": type(exc).__name__, "historical_replay_supported": False}
    polymarket_ctx["evidence_hash"] = canonical_sha256(polymarket_ctx)

    # Pillar 6: Optional local Chronos-t5-tiny + Statistical Drift Benchmark
    series_for_fc = {}
    for sym in symbols:
        inst = (market_snap.get("instruments") or {}).get(sym) or {}
        candles = inst.get("candles_closed_1h") or []
        if candles:
            series_for_fc[sym] = {
                "closes": [float(c["close"]) for c in candles],
                "last_close_at_utc": candles[-1]["close_at_utc"],
                "timeframe": "1H_closed",
            }
    try:
        from piia.intelligence.chronos_forecast import run_chronos_tiny
        fc_out_path = (snapshot_dir / "chronos_t5_tiny.json") if snapshot_dir else (ROOT / "artifacts" / "piia_v2_tmp_forecast.json")
        chronos_payload = run_chronos_tiny(series_for_fc, fc_out_path, allow_download=allow_forecast_download)
    except Exception as exc:
        chronos_payload = {"weights_ran": False, "reason": type(exc).__name__, "forecasts": {}}

    settled_funding = fetch_public_settled_funding_recent(symbols)

    return {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "market_snapshot": market_snap,
        "books_snapshot": books_snap,
        "tiers_snapshot": tiers_snap,
        "settled_funding": settled_funding,
        "macro_context": macro_ctx,
        "event_calendar_context": calendar_ctx,
        "flows_positioning_context": flows_positioning_ctx,
        "onchain_context": onchain_ctx,
        "polymarket_context": polymarket_ctx,
        "chronos_forecast": chronos_payload,
    }
