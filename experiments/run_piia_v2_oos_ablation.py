"""One-Shot Frozen Out-of-Sample & Module Ablation Runner for PIIA v2.

1. Verifies `verify_piia_v2_frozen()` before touching any market data.
2. Fetches (or reads cached) official public Bitget 1H candles, historical
   funding rates, contract specs, and ALFRED macro series into gitignored
   `data/piia_v2_eval/` (or runs `--demo-synthetic` offline).
3. Evaluates all 9 tournament arms (5 simple baselines + 4 progressive PIIA v2
   module ablations) on:
   - `unseen_post_2026_11_16_holdout` (`held_out=True`)
   - `pre_2026_11_16_exploratory` (`held_out=False`)
   - `rolling_14d_walk_forward_slices`
   for separate `100 USDT` and `500 USDT` books under `1x` base and `2x`
   slippage stress, plus automated prefix-truncation lookahead verification.
4. Writes full unedited results (including negative/null/rejected outcomes) to
   `artifacts/piia_v2_oos_ablation.json`. Never submits orders.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import httpx
import numpy as np
import pandas as pd

from piia.intelligence.agent_v2.ablation import (
    ARMS_ORDER,
    audit_prefix_lookahead,
    run_ablation_tournament,
)
from piia.intelligence.agent_v2.manifest import verify_piia_v2_frozen
from piia.intelligence.agent_v2.pillars import ROOT, SYMBOLS, load_v2_config
from piia.intelligence.public_orderbook import parse_contracts

DATA_DIR_DEFAULT = ROOT / "data" / "piia_v2_eval"
OUT_JSON_DEFAULT = ROOT / "artifacts" / "piia_v2_oos_ablation.json"
BITGET_BASE = "https://api.bitget.com"


def _fetch_bitget_1h_history(
    client: httpx.Client,
    symbol: str,
    start_utc: datetime,
    end_utc: datetime,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch closed 1H candles from Bitget public v2 recent + history-candles endpoints."""
    end_ms = int(end_utc.timestamp() * 1000)
    start_ms = int(start_utc.timestamp() * 1000)
    by_ts: dict[int, list[float]] = {}

    # 1. Recent candles endpoint (up to 1000 recent 1H bars)
    r_recent = client.get(
        f"{BITGET_BASE}/api/v2/mix/market/candles",
        params={"productType": "USDT-FUTURES", "symbol": symbol, "granularity": "1H", "limit": "1000"},
    )
    r_recent.raise_for_status()
    p_recent = r_recent.json()
    if p_recent.get("code") == "00000":
        for row in p_recent.get("data") or []:
            ts_ms = int(row[0])
            if start_ms <= ts_ms < end_ms:
                by_ts[ts_ms] = [float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])]

    # 2. Paginate backwards with `history-candles` (`endTime` cursor)
    cursor_end = min(by_ts.keys()) if by_ts else end_ms
    for _page in range(30):
        if cursor_end <= start_ms:
            break
        time.sleep(0.25)
        resp = client.get(
            f"{BITGET_BASE}/api/v2/mix/market/history-candles",
            params={
                "productType": "USDT-FUTURES",
                "symbol": symbol,
                "granularity": "1H",
                "endTime": str(cursor_end),
                "limit": "200",
            },
        )
        resp.raise_for_status()
        payload = resp.json()
        rows = payload.get("data") or []
        if payload.get("code") != "00000" or not rows:
            break
        oldest_in_page = None
        for row in rows:
            ts_ms = int(row[0])
            if oldest_in_page is None or ts_ms < oldest_in_page:
                oldest_in_page = ts_ms
            if start_ms <= ts_ms < end_ms:
                by_ts[ts_ms] = [float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])]
        if oldest_in_page is None or oldest_in_page >= cursor_end:
            break
        cursor_end = oldest_in_page

    if not by_ts:
        raise ValueError(f"No public 1H bars returned for {symbol}")

    ordered_ms = sorted(by_ts.keys())
    idx = pd.to_datetime(ordered_ms, unit="ms", utc=True)
    df = pd.DataFrame([by_ts[t] for t in ordered_ms], index=idx, columns=["open", "high", "low", "close", "volume"])
    df.index.name = "date"
    raw_bars = len(df)
    diffs = df.index.to_series().diff()
    gaps = [str(t) for t in df.index[diffs > pd.Timedelta(hours=1)]]
    if gaps:
        last_gap_idx = df.index[diffs > pd.Timedelta(hours=1)][-1]
        df = df.loc[last_gap_idx:]
    return df, {
        "raw_bars": raw_bars,
        "contiguous_bars": len(df),
        "gaps_found": gaps,
        "first_bar_open_utc": str(df.index[0]),
        "last_bar_open_utc": str(df.index[-1]),
    }


def _fetch_bitget_funding_history(
    client: httpx.Client,
    symbol: str,
    earliest_ms: int,
    end_ms: int,
) -> tuple[list[dict[str, Any]], int]:
    """Fetch public historical funding settlements from Bitget v2."""
    records: dict[int, float] = {}
    interval_hours = 8
    try:
        r_curr = client.get(
            f"{BITGET_BASE}/api/v2/mix/market/current-fund-rate",
            params={"productType": "USDT-FUTURES", "symbol": symbol},
        )
        if r_curr.status_code == 200:
            cdata = (r_curr.json().get("data") or [{}])
            crow = cdata[0] if isinstance(cdata, list) and cdata else cdata
            if isinstance(crow, dict) and crow.get("fundingRateInterval"):
                interval_hours = int(crow["fundingRateInterval"])
    except Exception:
        pass

    for page_no in range(1, 20):
        time.sleep(0.2)
        resp = client.get(
            f"{BITGET_BASE}/api/v2/mix/market/history-fund-rate",
            params={
                "productType": "USDT-FUTURES",
                "symbol": symbol,
                "pageSize": "100",
                "pageNo": str(page_no),
            },
        )
        resp.raise_for_status()
        payload = resp.json()
        rows = payload.get("data") or []
        if payload.get("code") != "00000" or not rows:
            break
        page_times = []
        for item in rows:
            raw_t = item.get("fundingTime") or item.get("settleTime") or item.get("timestamp")
            raw_r = item.get("fundingRate")
            if raw_t is None or raw_r is None:
                continue
            t_ms = int(raw_t)
            r_val = float(raw_r)
            page_times.append(t_ms)
            if earliest_ms <= t_ms <= end_ms and math.isfinite(r_val):
                records[t_ms] = r_val
        if not page_times or min(page_times) <= earliest_ms:
            break

    out = [{"timestamp": t, "rate": records[t]} for t in sorted(records)]
    return out, interval_hours


def _check_funding_coverage(
    funding: dict[str, list[dict[str, Any]]],
    intervals: dict[str, int],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    missing: dict[str, Any] = {}
    complete = True
    for sym, rows in funding.items():
        iv = int(intervals.get(sym, 8))
        expected = pd.date_range(start.ceil(f"{iv}h"), end, freq=f"{iv}h")
        avail = {pd.Timestamp(int(r["timestamp"]), unit="ms", tz="UTC") for r in rows}
        miss = [str(t) for t in expected if t not in avail]
        if miss:
            complete = False
            missing[sym] = {"count": len(miss), "first": miss[0], "last": miss[-1]}
    return {"complete": complete, "missing": missing}


def fetch_eval_dataset(data_dir: Path, *, days_back: int = 92) -> dict[str, Any]:
    """Download public Bitget 1H candles, funding, contract specs & ALFRED macro."""
    data_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    end_closed_hour = now.replace(minute=0, second=0, microsecond=0)
    start_fetch = end_closed_hour - timedelta(days=days_back)

    prov: dict[str, Any] = {
        "fetched_at_utc": now.isoformat(),
        "start_requested_utc": start_fetch.isoformat(),
        "end_exclusive_utc": end_closed_hour.isoformat(),
        "symbols": {},
        "files_sha256": {},
    }

    with httpx.Client(timeout=20.0, follow_redirects=False) as client:
        c_resp = client.get(f"{BITGET_BASE}/api/v2/mix/market/contracts?productType=USDT-FUTURES")
        c_resp.raise_for_status()
        contracts_parsed = parse_contracts(
            c_resp.json(), SYMBOLS, source_url=str(c_resp.url), retrieved=now, raw=c_resp.content
        )
        c_path = data_dir / "contracts.json"
        c_path.write_text(json.dumps(contracts_parsed, indent=2, sort_keys=True), encoding="utf-8")
        prov["files_sha256"]["contracts.json"] = hashlib.sha256(c_path.read_bytes()).hexdigest()

        for sym in SYMBOLS:
            df, bar_meta = _fetch_bitget_1h_history(client, sym, start_fetch, end_closed_hour)
            bar_file = data_dir / f"{sym}_1h.csv"
            df.to_csv(bar_file)
            prov["files_sha256"][bar_file.name] = hashlib.sha256(bar_file.read_bytes()).hexdigest()

            f_rows, iv_h = _fetch_bitget_funding_history(
                client, sym, int(df.index[0].timestamp() * 1000), int(end_closed_hour.timestamp() * 1000)
            )
            f_file = data_dir / f"{sym}_funding.json"
            f_file.write_text(json.dumps(f_rows, indent=2), encoding="utf-8")
            prov["files_sha256"][f_file.name] = hashlib.sha256(f_file.read_bytes()).hexdigest()

            prov["symbols"][sym] = {
                **bar_meta,
                "funding_records": len(f_rows),
                "funding_interval_hours": iv_h,
                "funding_first_utc": pd.Timestamp(f_rows[0]["timestamp"], unit="ms", tz="UTC").isoformat() if f_rows else None,
                "funding_last_utc": pd.Timestamp(f_rows[-1]["timestamp"], unit="ms", tz="UTC").isoformat() if f_rows else None,
            }

    # Fetch ALFRED history
    try:
        from piia.intelligence.macro_data import fetch_alfred_history
        alfred = fetch_alfred_history(["DGS10", "DGS2", "DFII10", "DFF", "DTWEXBGS", "RRPONTSYD", "WDTGAL"], lookback_days=days_back + 40)
    except Exception as exc:
        alfred = {"status": "unavailable", "error": type(exc).__name__, "series": {}}
    m_file = data_dir / "alfred_macro.json"
    m_file.write_text(json.dumps(alfred, indent=2), encoding="utf-8")
    prov["files_sha256"][m_file.name] = hashlib.sha256(m_file.read_bytes()).hexdigest()

    p_file = data_dir / "provenance_v2.json"
    p_file.write_text(json.dumps(prov, indent=2, sort_keys=True), encoding="utf-8")
    return prov


def load_eval_dataset(data_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, list[dict[str, Any]]], dict[str, int], dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    prov = json.loads((data_dir / "provenance_v2.json").read_text(encoding="utf-8"))
    contracts_doc = json.loads((data_dir / "contracts.json").read_text(encoding="utf-8"))
    specs_override = contracts_doc.get("contracts") or {}

    frames: dict[str, pd.DataFrame] = {}
    funding: dict[str, list[dict[str, Any]]] = {}
    intervals: dict[str, int] = {}
    for sym in SYMBOLS:
        df = pd.read_csv(data_dir / f"{sym}_1h.csv", parse_dates=["date"]).set_index("date")
        df.index = pd.to_datetime(df.index, utc=True)
        frames[sym] = df.sort_index()
        funding[sym] = json.loads((data_dir / f"{sym}_funding.json").read_text(encoding="utf-8"))
        intervals[sym] = int(prov["symbols"][sym].get("funding_interval_hours", 8))

    alfred_doc = json.loads((data_dir / "alfred_macro.json").read_text(encoding="utf-8"))
    macro_obs = {
        sid: item.get("observations") or []
        for sid, item in (alfred_doc.get("series") or {}).items()
    }
    return frames, funding, intervals, macro_obs, specs_override, prov


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR_DEFAULT)
    ap.add_argument("--fetch-public", action="store_true", help="fetch fresh public Bitget + ALFRED data into --data-dir")
    ap.add_argument("--demo-synthetic", action="store_true", help="run on labeled synthetic series when offline")
    ap.add_argument("--out", type=Path, default=OUT_JSON_DEFAULT)
    args = ap.parse_args(argv)

    # STEP 1: Cryptographic freeze check BEFORE touching data!
    freeze_check = verify_piia_v2_frozen()
    if not freeze_check["ok"]:
        raise SystemExit(f"REFUSED: Frozen PIIA v2 manifest check failed: {freeze_check['issues']}")

    cfg = load_v2_config()
    holdout_boundary = pd.Timestamp(cfg["evaluation_splits"]["unseen_holdout_start_utc"]).tz_convert("UTC")
    warmup = int(cfg["evaluation_splits"]["warmup_bars_required"])

    if args.demo_synthetic:
        from tests.test_piia_agent_v2 import _synthetic_3contract_frames
        frames = _synthetic_3contract_frames(1500)
        # Re-index so boundary falls cleanly inside synthetic span
        end_syn = pd.Timestamp("2026-11-23T00:00:00+00:00")
        idx_syn = pd.date_range(end=end_syn - pd.Timedelta(hours=1), periods=1500, freq="1h", tz="UTC")
        for s in frames:
            frames[s].index = idx_syn
        funding = {s: [{"timestamp": int(t.timestamp() * 1000), "rate": 0.00005} for t in idx_syn if t.hour % 8 == 0] for s in SYMBOLS}
        intervals = {s: 8 for s in SYMBOLS}
        macro_obs = {}
        specs_override = {}
        prov = {"kind": "SYNTHETIC_OFFLINE_DEMO_NOT_REAL_MARKET", "files_sha256": {"synthetic": "demo"}}
    else:
        if args.fetch_public:
            fetch_eval_dataset(args.data_dir)
        elif not (args.data_dir / "provenance_v2.json").is_file():
            raise SystemExit(
                "Archived local evaluation dataset not present (gitignored per AGENTS.md); "
                "original v2.0.0 slice is not reproduced from Git alone. "
                "Pass --fetch-public for current public window or --demo-synthetic for engineering test only."
            )
        frames, funding, intervals, macro_obs, specs_override, prov = load_eval_dataset(args.data_dir)

    dataset_hash = hashlib.sha256(json.dumps(prov.get("files_sha256", {}), sort_keys=True).encode("utf-8")).hexdigest()
    common = sorted(set.intersection(*(set(d.index) for d in frames.values())))
    if len(common) < warmup + 48:
        raise SystemExit(f"Insufficient common 1H bars across 3 contracts: {len(common)}")

    earliest_eval = common[warmup]
    last_common = common[-1]

    # Determine earliest funding-covered start
    earliest_funding_ts = earliest_eval
    f_starts = []
    for s, rows in funding.items():
        if rows:
            f_starts.append(pd.Timestamp(int(rows[0]["timestamp"]), unit="ms", tz="UTC"))
    if len(f_starts) == len(SYMBOLS):
        earliest_funding_ts = max(earliest_eval, max(f_starts))
        # Snap to common index
        valid_c = [t for t in common if t >= earliest_funding_ts]
        if valid_c:
            earliest_funding_ts = valid_c[0]

    report: dict[str, Any] = {
        "protocol_id": cfg["protocol_id"],
        "version": cfg["version"],
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_manifest_verification": freeze_check,
        "dataset_provenance": prov,
        "dataset_sha256": dataset_hash,
        "common_bars_total": len(common),
        "common_first_utc": str(common[0]),
        "common_last_utc": str(last_common),
        "tuning_performed_after_freeze": False,
        "live_orders_submitted": False,
        "is_synthetic_engineering_test": bool(args.demo_synthetic),
        "module_execution_summary": {
            "deterministic_research_rule": {"executed": True, "basis": "causal 1h/4h OHLCV indicators"},
            "deterministic_critic_cost_gate": {"executed": True, "basis": "ATR/friction ratio, volatility regime, funding crowding, 4h conflict"},
            "macro_alfred_lagged_filter": {"executed": bool(macro_obs), "basis": "ALFRED daily series with conservative +2 UTC day lag"},
            "statistical_drift_forecast_168h": {"executed": True, "basis": "causal 168h log-return drift & sigma*sqrt(24) band"},
            "neural_chronos_model": {"executed": False, "basis": "Chronos neural weights not executed in historical ablation"},
            "official_llm_review": {"executed": False, "status": "NOT_WIRED"},
            "contemporaneous_live_pillars": {"executed": False, "basis": "ETF flows / COT / mempool / Polymarket unavailable point-in-time in historical replay"},
        },
        "windows": {},
    }

    # Window 1: Post-2026-11-16 Replay Window (pre-freeze timing NOT independently verified)
    holdout_bars = [t for t in common if t >= holdout_boundary]
    if len(holdout_bars) >= 24:
        h_start, h_end = holdout_bars[0], holdout_bars[-1]
        h_fund_cov = _check_funding_coverage(funding, intervals, h_start, h_end)
        report["windows"]["unseen_post_2026_11_16_holdout"] = {
            "label": "SYNTHETIC_ENGINEERING_TEST" if args.demo_synthetic else "HISTORICAL_REPLAY_PRE_FREEZE_TIMING_UNVERIFIED",
            "timing_independently_verified": False,
            "note": (
                "Synthetic fixture for software verification only; never empirical evidence."
                if args.demo_synthetic
                else "历史回放，事前冻结未获独立验证：同提交哈希仅证明文件一致，非独立事前时间戳证明。"
            ),
            "bars_count": len(holdout_bars),
            "funding_coverage": h_fund_cov,
            "book_100u": run_ablation_tournament(
                frames, start=h_start, end=h_end, starting_cash=100.0, held_out=False,
                market_data_hash=dataset_hash, funding=funding, funding_complete=h_fund_cov["complete"],
                macro_observations=macro_obs, specs_override=specs_override, cfg=cfg,
            ),
            "book_500u": run_ablation_tournament(
                frames, start=h_start, end=h_end, starting_cash=500.0, held_out=False,
                market_data_hash=dataset_hash, funding=funding, funding_complete=h_fund_cov["complete"],
                macro_observations=macro_obs, specs_override=specs_override, cfg=cfg,
            ),
        }

    # Window 2: PRE-2026-11-16 EXPLORATORY COMMON SPAN
    pre_bars = [t for t in common if earliest_funding_ts <= t < holdout_boundary]
    if len(pre_bars) >= 48:
        p_start, p_end = pre_bars[0], pre_bars[-1]
        p_fund_cov = _check_funding_coverage(funding, intervals, p_start, p_end)
        # Note: `audit_comparison` checks `held_out=True` as a protocol gate for holdouts;
        # on exploratory pre-freeze data we pass `held_out=False` honestly so `audit_comparison`
        # records `not_held_out` as an explicit reminder!
        report["windows"]["pre_2026_11_16_exploratory"] = {
            "label": "PRE_FREEZE_EXPLORATORY_ALREADY_SEEN_NOT_HOLDOUT",
            "bars_count": len(pre_bars),
            "funding_coverage": p_fund_cov,
            "book_100u": run_ablation_tournament(
                frames, start=p_start, end=p_end, starting_cash=100.0, held_out=False,
                market_data_hash=dataset_hash, funding=funding, funding_complete=p_fund_cov["complete"],
                macro_observations=macro_obs, specs_override=specs_override, cfg=cfg,
            ),
            "book_500u": run_ablation_tournament(
                frames, start=p_start, end=p_end, starting_cash=500.0, held_out=False,
                market_data_hash=dataset_hash, funding=funding, funding_complete=p_fund_cov["complete"],
                macro_observations=macro_obs, specs_override=specs_override, cfg=cfg,
            ),
        }

    # Window 3: Non-overlapping 14-day (336h) Rolling Walk-Forward Slices across full funding-verified span
    full_eval_bars = [t for t in common if t >= earliest_funding_ts]
    wf_slices = []
    slice_len = 336
    pos_idx = 0
    slice_num = 1
    while pos_idx + slice_len <= len(full_eval_bars):
        s_start = full_eval_bars[pos_idx]
        s_end = full_eval_bars[pos_idx + slice_len - 1]
        s_cov = _check_funding_coverage(funding, intervals, s_start, s_end)
        s_res = run_ablation_tournament(
            frames, start=s_start, end=s_end, starting_cash=500.0, held_out=False,
            market_data_hash=dataset_hash, funding=funding, funding_complete=s_cov["complete"],
            macro_observations=macro_obs, specs_override=specs_override,
            arms=("B0_cash", "B1_simple_ema_1h", "B2_aligned_ema_4h", "B3_donchian_breakout_4h", "M1_researcher_market_only", "M2a_plus_cost_gate_only", "M4_full_piia_v2"),
            include_2x_cost_stress=False, cfg=cfg,
        )
        wf_slices.append({
            "slice": slice_num,
            "start_utc": str(s_start),
            "end_utc": str(s_end),
            "funding_complete": s_cov["complete"],
            "returns_pct_500u": {k: v["return_pct"] for k, v in s_res["base_cost_results"].items()},
            "max_dd_pct_500u": {k: v["max_drawdown_pct"] for k, v in s_res["base_cost_results"].items()},
            "trades_500u": {k: v["closed_trades"] for k, v in s_res["base_cost_results"].items()},
            "turnover_multiple_500u": {k: v["turnover_multiple"] for k, v in s_res["base_cost_results"].items()},
            "exposure_bar_fraction_500u": {k: v["exposure_bar_fraction"] for k, v in s_res["base_cost_results"].items()},
            "loss_halt_500u": {k: v["loss_halt_reason"] for k, v in s_res["base_cost_results"].items()},
        })
        pos_idx += slice_len
        slice_num += 1
    report["rolling_14d_walk_forward_500u"] = wf_slices

    # Automated Prefix-Truncation Lookahead Audit across 3 real timestamps
    if len(full_eval_bars) >= 120:
        n_b = len(full_eval_bars)
        cutoffs = [full_eval_bars[n_b // 3], full_eval_bars[(2 * n_b) // 3], full_eval_bars[-24]]
        report["prefix_truncation_lookahead_audit"] = audit_prefix_lookahead(
            frames,
            start=full_eval_bars[0],
            cutoffs=cutoffs,
            end=full_eval_bars[-1],
            starting_cash=500.0,
            funding=funding,
            macro_observations=macro_obs,
            cfg=cfg,
        )

    # Re-verify freeze after evaluation completes!
    post_check = verify_piia_v2_frozen()
    if not post_check["ok"]:
        raise SystemExit("Frozen files mutated during evaluation!")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"Saved evaluation report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
