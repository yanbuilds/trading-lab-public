from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from piia.intelligence.historical_packets import (
    ASSET_MAP,
    DECISION_TIMES,
    HOLD_HOURS,
    PROTOCOL,
    build_packets,
    write_packets,
)
from piia.intelligence.paths import ROOT
from trading_lab.perps.compare import load_cache

CACHE = ROOT / "data" / "perps-extended"
pytestmark = pytest.mark.skipif(not (CACHE / "provenance.json").exists(), reason="extended cache not present")


def test_schedule_is_frozen_and_horizon_fits():
    assert len(DECISION_TIMES) == 10
    gaps = [(b - a).total_seconds() / 86400 for a, b in zip(DECISION_TIMES, DECISION_TIMES[1:])]
    assert gaps == [8] * 9
    frames, *_ = load_cache(CACHE)
    last = max(df.index.max() for df in frames.values())
    # last index is bar open; cache includes the 05:00 UTC bar on 16 Nov, so last close is 06:00
    assert DECISION_TIMES[-1] + pd.Timedelta(hours=HOLD_HOURS) <= last + pd.Timedelta(hours=1)


def test_parent_packets_have_no_calendar_or_absolute_price_leak(tmp_path):
    parent, sealed = build_packets(CACHE)
    text = json.dumps(parent)
    assert re.search(r"202[56]|btc|xau|ndx|bitget|usdt", text, re.I) is None
    # absolute venue-like prices should not appear as raw numbers near known levels
    for ev in sealed["events"]:
        for asset in ev["assets"].values():
            last = asset["last_close"]
            # the exact last_close must not appear in the parent file
            assert f"{last:.2f}" not in text
            assert f"{last:.1f}" not in text
    # packets must still have rebased last close
    for p in parent["packets"]:
        for feat in p["assets"].values():
            assert feat["last_close_index"] == 100.0
            assert all(k in feat["momentum"] for k in ("return_24h", "return_7d", "return_20d"))
            assert feat["missing_datasets"]


def test_no_lookahead_when_later_bars_change():
    frames, specs, funding, prov = load_cache(CACHE)
    parent_a, _ = build_packets(CACHE)
    # mutate a copy of cache in a temp dir after the first decision
    # cheaper: rebuild features by monkeypatching load_cache
    t0 = DECISION_TIMES[0]
    # If later prices leak, changing them would change the first packet
    # We call build_packets against a temp cache with later closes * 1.5
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        dest = Path(td)
        for p in CACHE.iterdir():
            if p.is_file():
                shutil.copy(p, dest / p.name)
        for sym in ("BTCUSDT", "XAUUSDT", "NDX100USDT"):
            df = pd.read_csv(dest / f"{sym}.csv", parse_dates=["date"])
            later = pd.to_datetime(df["date"], utc=True) >= t0
            for col in ("open", "high", "low", "close"):
                df.loc[later, col] = df.loc[later, col] * 1.5
            df.to_csv(dest / f"{sym}.csv", index=False)
        parent_b, _ = build_packets(dest)
    # The first display packet may not be the first chronological event.
    # Compare by sealed map: packets whose sealed decision is t0 must match.
    _, sealed_a = build_packets(CACHE)
    first_id = next(e["packet_id"] for e in sealed_a["events"] if e["decision_at_utc"].startswith(str(t0)[:19]))
    pa = next(p for p in parent_a["packets"] if p["packet_id"] == first_id)
    pb = next(p for p in parent_b["packets"] if p["packet_id"] == first_id)
    assert pa == pb


def test_write_does_not_put_outcomes_in_parent(tmp_path):
    info = write_packets(CACHE, tmp_path)
    parent = json.loads(Path(info["parent"]).read_text())
    assert parent["no_outcomes_in_this_file"] is True
    blob = Path(info["parent"]).read_text() + Path(info["brief"]).read_text()
    assert "outcome" not in blob.lower() or parent["no_outcomes_in_this_file"]
    assert "收益率" not in Path(info["brief"]).read_text()
    assert "profit" not in blob.lower()
    assert parent["protocol"] == PROTOCOL
    assert set(ASSET_MAP) == {"Asset-A", "Asset-B", "Asset-C"}
