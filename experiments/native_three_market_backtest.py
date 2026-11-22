"""One fixed native Freqtrade backtest, limited to three public tier requests.

The only framework override narrows leverage-tier retrieval to the fixed
whitelist. It does not alter fills, strategies or enable an exchange account.
Input prices/funding are validated before the framework can run. This is a
previously viewed exploratory interval, not a held-out or live result.
"""
from __future__ import annotations

import asyncio
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from trading_lab.perps.risk import assert_simulation

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = {"BTC": 8, "XAU": 4, "NDX100": 8}
PAIRS = [f"{s}/USDT:USDT" for s in SYMBOLS]
START = pd.Timestamp("2026-11-02T00:00:00Z")
END = pd.Timestamp("2026-11-16T00:00:00Z")
WARMUP = 60


def write_prefix_data(data_dir: Path, out_dir: Path, end: pd.Timestamp) -> dict:
    """Copy only pre-cutoff rows; never alter the original provider files."""
    end = pd.Timestamp(end)
    if end.tzinfo is None:
        raise ValueError("aware cutoff required")
    end = end.tz_convert("UTC")
    manifest = {}
    for symbol in SYMBOLS:
        for kind in ("futures", "mark", "funding_rate"):
            matches = list(data_dir.rglob(f"{symbol}_USDT_USDT-1h-{kind}.feather"))
            if len(matches) != 1:
                raise ValueError(f"missing or ambiguous {symbol} {kind}")
            frame = pd.read_feather(matches[0])
            dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
            prefix = frame.loc[dates < end].copy()
            if prefix.empty:
                raise ValueError(f"empty prefix for {symbol} {kind}")
            target = out_dir / "futures" / matches[0].name
            target.parent.mkdir(parents=True, exist_ok=True)
            prefix.reset_index(drop=True).to_feather(target)
            written = pd.read_feather(target)
            if (pd.to_datetime(written.date, utc=True) >= end).any():
                raise ValueError("post-cutoff row in prefix")
            manifest[f"{symbol}:{kind}"] = {"rows": len(written),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    return manifest


def audit_data(data_dir: Path, reference_dir: Path, start=START, end=END, warmup=WARMUP) -> dict:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start.tzinfo is None or end.tzinfo is None or start >= end or warmup < 0:
        raise ValueError("ordered aware boundaries and nonnegative warmup required")
    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    if start != start.floor("h") or end != end.floor("h"):
        raise ValueError("hour-aligned boundaries required")
    summary = {"start_utc": start.isoformat(), "end_exclusive_utc": end.isoformat(),
               "warmup_hours": warmup, "assumed_intervals_hours": SYMBOLS,
               "funding_reference": "separate previous Bitget API download, not independent source",
               "historical_interval_changes_verified": False, "files": {}}
    for symbol, interval in SYMBOLS.items():
        for kind in ("futures", "mark", "funding_rate"):
            matches = list(data_dir.rglob(f"{symbol}_USDT_USDT-1h-{kind}.feather"))
            if len(matches) != 1:
                raise ValueError(f"{symbol} {kind}: missing or ambiguous file")
            path = matches[0]
            frame = pd.read_feather(path)
            frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
            index = pd.DatetimeIndex(frame["date"])
            if index.has_duplicates or not index.is_monotonic_increasing:
                raise ValueError(f"{symbol} {kind}: duplicate or unsorted timestamps")
            left = start if kind == "funding_rate" else start - pd.Timedelta(hours=warmup)
            expected = pd.date_range(left, end - pd.Timedelta(hours=1), freq=f"{interval if kind == 'funding_rate' else 1}h")
            relevant = frame[(frame.date >= left) & (frame.date < end)]
            if len(expected.difference(pd.DatetimeIndex(relevant.date))) or len(relevant) != len(expected):
                raise ValueError(f"{symbol} {kind}: missing or unexpected observations")
            fields = ["funding_rate"] if kind == "funding_rate" else ["open", "high", "low", "close"]
            values = relevant[fields].apply(pd.to_numeric, errors="raise")
            if not all(math.isfinite(float(v)) for v in values.to_numpy().flat):
                raise ValueError(f"{symbol} {kind}: nonfinite values")
            if kind != "funding_rate":
                if ((values <= 0).any().any() or (values.high < values[["open", "low", "close"]].max(axis=1)).any()
                        or (values.low > values[["open", "high", "close"]].min(axis=1)).any()):
                    raise ValueError(f"{symbol} {kind}: invalid OHLC")
            else:
                ref_path = reference_dir / f"{symbol}USDT-funding.csv"
                ref = pd.read_csv(ref_path)
                ref["date"] = pd.to_datetime(ref["timestamp"], unit="ms", utc=True, errors="raise")
                if ref.date.duplicated().any():
                    raise ValueError(f"{symbol}: duplicate reference funding")
                merged = relevant.merge(ref[["date", "rate"]], on="date", how="left", validate="one_to_one")
                reference = pd.to_numeric(merged.rate, errors="coerce")
                if reference.isna().any() or any(not math.isfinite(float(v)) for v in reference):
                    raise ValueError(f"{symbol}: missing/invalid reference funding")
                if ((merged.funding_rate.astype(float) - reference).abs() > 1e-12).any():
                    raise ValueError(f"{symbol}: funding mismatch")
                summary["files"][f"{symbol}:funding_reference"] = {
                    "path": str(ref_path), "sha256": hashlib.sha256(ref_path.read_bytes()).hexdigest()}
            summary["files"][f"{symbol}:{kind}"] = {"path": str(path), "rows_in_window": len(relevant),
                                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return summary


def main() -> None:
    import freqtrade
    from freqtrade.exchange.exchange import Exchange
    from freqtrade.main import main as freqtrade_main

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-end", choices=["20261106", "20261109", "20261112"],
                        help="run a fixed earlier-end causality diagnostic")
    parser.add_argument("--tier-snapshot", type=Path, help="reuse a prior audited public tier snapshot")
    args = parser.parse_args()
    if bool(args.prefix_end) != bool(args.tier_snapshot):
        raise ValueError("prefix diagnostic requires an audited tier snapshot, and vice versa")
    end = pd.to_datetime(args.prefix_end, format="%Y%m%d", utc=True) if args.prefix_end else END
    frozen_tiers = None
    original_snapshot = None
    if args.tier_snapshot:
        snapshot_path = args.tier_snapshot.resolve()
        prior_audit = json.loads((snapshot_path.parent / "input_audit.json").read_text(encoding="utf-8"))
        for key, source in (("config_sha256", ROOT / "freqtrade/config.simulation.json"),
                            ("strategy_sha256", ROOT / "freqtrade/strategies/BitgetBaseline.py"),
                            ("risk_sha256", ROOT / "src/trading_lab/perps/risk.py")):
            if prior_audit.get(key) != hashlib.sha256(source.read_bytes()).hexdigest():
                raise ValueError(f"prior run differs: {key}")
        digest = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
        if prior_audit.get("public_tier_snapshot_sha256") != digest:
            raise ValueError("prior tier snapshot hash mismatch")
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        original_snapshot = snapshot
        frozen_tiers = snapshot.get("pairs")
        if not isinstance(frozen_tiers, dict) or set(frozen_tiers) != set(PAIRS) or any(not frozen_tiers[p] for p in PAIRS):
            raise ValueError("unexpected tier snapshot pairs")

    if freqtrade.__version__ != "2026.10":
        raise ValueError("expected the fixed Freqtrade 2026.10 version")
    config_path = ROOT / "freqtrade" / "config.simulation.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert_simulation(config)
    if (config.get("trading_mode") != "futures" or config.get("exchange", {}).get("name") != "bitget"
            or config.get("exchange", {}).get("pair_whitelist") != PAIRS):
        raise ValueError("unexpected research configuration")
    data_dir = ROOT / "data" / "perps-native-100u"
    audit = audit_data(data_dir, ROOT / "data" / "perps-extended", end=end)
    if frozen_tiers is not None:
        for name, entry in audit["files"].items():
            if prior_audit.get("files", {}).get(name, {}).get("sha256") != entry["sha256"]:
                raise ValueError(f"prior input data changed: {name}")
    output_dir = ROOT / "artifacts" / ("ft-native-three-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output_dir.mkdir(parents=True, exist_ok=False)
    if args.prefix_end:
        prefix_dir = output_dir / "prefix_data"
        audit["prefix_files"] = write_prefix_data(data_dir, prefix_dir, end)
        audit["prefix_note"] = "Only rows strictly before the cutoff; cutoff funding itself is excluded."
        data_dir = prefix_dir
    audit.update({"freqtrade_version": freqtrade.__version__, "virtual_capital_usdt": 100,
                  "fee_each_side": .0006, "retrieval_override": "sequential public tier calls for exactly the three configured pairs",
                  "market_tier_and_specs_historical_validation": False,
                  "held_out": False, "already_viewed": True, "live_orders": False,
                  "prefix_diagnostic": bool(args.prefix_end),
                  "tier_snapshot_reused": bool(frozen_tiers),
                  "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  "risk_sha256": hashlib.sha256((ROOT / "src/trading_lab/perps/risk.py").read_bytes()).hexdigest(),
                  "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "strategy_sha256": hashlib.sha256((ROOT / "freqtrade/strategies/BitgetBaseline.py").read_bytes()).hexdigest()})
    (output_dir / "input_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    def three_tiers(self):
        assert_simulation(self._config)
        mode = self._config.get("runmode")
        if getattr(mode, "value", mode) != "backtest":
            raise ValueError("tier override is restricted to backtesting")
        async def fetch():
            found = {}
            for pair in PAIRS:
                returned, tiers = await self.get_market_leverage_tiers(pair)
                if returned != pair or not tiers:
                    raise ValueError(f"missing public tier data for {pair}")
                found[pair] = tiers
                await asyncio.sleep(0.5)
            return found
        with self._loop_lock:
            tiers = frozen_tiers if frozen_tiers is not None else self.loop.run_until_complete(fetch())
        snapshot = {"retrieved_at_utc": (original_snapshot["retrieved_at_utc"] if original_snapshot else datetime.now(timezone.utc).isoformat()),
                    "used_at_utc": datetime.now(timezone.utc).isoformat(),
                    "reused": original_snapshot is not None,
                    "historical_tiers_verified": False, "pairs": tiers}
        snapshot_path = output_dir / "public_tier_snapshot.json"
        snapshot_path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
        audit["public_tier_snapshot_sha256"] = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
        (output_dir / "input_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        return tiers

    Exchange.load_leverage_tiers = three_tiers
    print(f"EXPLORATORY ONLY: tier loader restricted to {PAIRS}; audit={output_dir / 'input_audit.json'}", flush=True)
    freqtrade_main(["backtesting", "-c", str(config_path), "--userdir", str(ROOT / "artifacts/ft-native-probe"),
                    "--strategy-path", str(ROOT / "freqtrade/strategies"), "--strategy", "BitgetBaseline",
                    "--datadir", str(data_dir), "--timerange", "20261102-" + (args.prefix_end or "20261116"), "--dry-run-wallet", "100",
                    "--fee", "0.0006", "--enable-protections", "--cache", "none", "--export", "trades",
                    "--backtest-directory", str(output_dir)])


if __name__ == "__main__":
    main()
