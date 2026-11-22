"""Compare a native full run with a physically truncated diagnostic run.

Forced exits in the truncated run have no comparable full-run exit. Their
entries still must match. This does not establish complete freedom from bias.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

ENTRY = ("pair", "is_short", "open_date", "amount", "open_rate", "leverage",
         "initial_stop_loss_abs", "initial_stop_loss_ratio")
EXIT = ENTRY + ("close_date", "close_rate", "profit_abs", "funding_fees", "exit_reason")


def compare(full: list[dict], prefix: list[dict], cutoff: pd.Timestamp) -> dict:
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("cutoff must be timezone aware")
    def before(trade, field):
        stamp = pd.Timestamp(trade[field])
        if stamp.tzinfo is None:
            raise ValueError("trade timestamp must be timezone aware")
        return stamp < cutoff
    def canonical(rows, keys):
        return sorted((tuple(row[k] for k in keys) for row in rows), key=lambda x: (x[2], x[0]))
    entries_full = [t for t in full if before(t, "open_date")]
    entries_prefix = [t for t in prefix if before(t, "open_date")]
    if canonical(entries_full, ENTRY) != canonical(entries_prefix, ENTRY):
        raise ValueError("pre-cutoff entry mismatch")
    exits_full = [t for t in full if before(t, "close_date") and t["exit_reason"] != "force_exit"]
    exits_prefix = [t for t in prefix if before(t, "close_date") and t["exit_reason"] != "force_exit"]
    if canonical(exits_full, EXIT) != canonical(exits_prefix, EXIT):
        raise ValueError("pre-cutoff natural exit mismatch")
    forced = [t for t in prefix if t["exit_reason"] == "force_exit"]
    return {"cutoff_utc": cutoff.tz_convert("UTC").isoformat(),
            "matching_entries": len(entries_full), "matching_natural_exits": len(exits_full),
            "prefix_forced_exits_excluded": len(forced),
            "limitation": "One cutoff, small sample; forced exits and post-cutoff funding not comparable; market metadata not frozen."}


def read_trades(path: Path) -> list[dict]:
    with ZipFile(path) as z:
        names = [name for name in z.namelist() if name.endswith(".json") and not name.endswith("_config.json")]
        if len(names) != 1:
            raise ValueError("ambiguous result archive")
        return json.loads(z.read(names[0]))["strategy"]["BitgetBaseline"]["trades"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full", type=Path)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--cutoff", required=True)
    args = parser.parse_args()
    print(json.dumps(compare(read_trades(args.full), read_trades(args.prefix), pd.Timestamp(args.cutoff)), indent=2))


if __name__ == "__main__":
    main()
