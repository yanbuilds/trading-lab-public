"""CLI runner for Causal Rolling Walk-Forward v2 (`regime_adaptive_v2` vs `ema_1h_v2`).

Supports running on a local CSV cache (`--data-dir data/perps-extended`) or on
explicitly labeled synthetic multi-regime data (`--demo-synthetic`) when local
market archives are absent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading_lab.perps.walk_forward import run_walk_forward


def _synthetic_frames(bars: int = 1200) -> dict[str, pd.DataFrame]:
    idx = pd.date_range("2026-01-01", periods=bars, freq="1h", tz="UTC")
    out = {}
    for seed, sym, base in ((7, "BTCUSDT", 65000.0), (13, "XAUUSDT", 3800.0), (19, "NDX100USDT", 27000.0)):
        rng = np.random.default_rng(seed)
        # Alternating trend and range phases
        phases = np.sin(np.linspace(0, 6 * np.pi, bars)) * 0.0018 + rng.normal(0.0, 0.003, size=bars)
        closes = base * np.cumprod(1.0 + phases)
        opens = np.roll(closes, 1)
        opens[0] = base
        span = np.abs(closes - opens) + closes * 0.002
        out[sym] = pd.DataFrame(
            {
                "open": opens,
                "high": np.maximum(opens, closes) + 0.5 * span,
                "low": np.minimum(opens, closes) - 0.5 * span,
                "close": closes,
                "volume": 100.0,
            },
            index=idx,
        )
    return out


def _load_csv_dir(data_dir: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for sym in ("BTCUSDT", "XAUUSDT", "NDX100USDT"):
        p = data_dir / f"{sym}.csv"
        if p.exists():
            df = pd.read_csv(p, parse_dates=["date"]).set_index("date")
            df.index = pd.to_datetime(df.index, utc=True)
            frames[sym] = df.sort_index()
    if not frames:
        raise FileNotFoundError(f"No <SYMBOL>.csv files found in {data_dir}")
    return frames


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/perps-extended"))
    ap.add_argument("--demo-synthetic", action="store_true", help="run on labeled synthetic multi-regime series")
    ap.add_argument("--warmup-bars", type=int, default=240)
    ap.add_argument("--fold-bars", type=int, default=300)
    ap.add_argument("--step-bars", type=int, default=300)
    ap.add_argument("--capital", type=float, default=500.0)
    ap.add_argument("--leverage", type=float, default=2.0)
    ap.add_argument("--risk-pct", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.demo_synthetic:
        frames = _synthetic_frames(1200)
        dataset_label = "SYNTHETIC_DEMO_MULTI_REGIME_NOT_REAL_MARKET"
    else:
        frames = _load_csv_dir(args.data_dir)
        dataset_label = f"local_csv:{args.data_dir}"

    report = run_walk_forward(
        frames,
        warmup_bars=args.warmup_bars,
        fold_bars=args.fold_bars,
        step_bars=args.step_bars,
        capital=args.capital,
        leverage=args.leverage,
        risk_fraction=args.risk_pct / 100.0,
    )
    report["dataset_label"] = dataset_label
    text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
