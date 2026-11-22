from __future__ import annotations

import pandas as pd
import pytest

from piia.intelligence.htf_alignment_gate import (
    EXTENSION_THRESHOLD_ATR,
    GATE_ID,
    MANIFEST_PATH,
    completed_4h,
    evaluate_gate,
    load_manifest,
    signed_extension_atr,
)


def test_manifest_is_frozen_and_not_marked_optimized():
    m = load_manifest()
    assert m["id"] == GATE_ID
    assert m["hypothesis_optimized"] is False
    assert m["held_out"] is False
    assert m["changes_past_research"] is False
    assert m["changes_stop_or_horizon"] is False
    assert float(m["not_overextended"]["threshold_atr"]) == EXTENSION_THRESHOLD_ATR
    assert MANIFEST_PATH.exists()


def test_completed_4h_excludes_partial_current_bar():
    idx = pd.date_range("2026-10-26 16:00", periods=16, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "open": range(16), "high": range(1, 17), "low": range(16),
        "close": range(16), "volume": [1] * 16,
    }, index=idx)
    # Decision at 2026-10-27 00:00: last completed 4h is 20:00-00:00.
    h4 = completed_4h(df, pd.Timestamp("2026-10-27 00:00", tz="UTC"))
    assert h4.index.max() == pd.Timestamp("2026-10-26 20:00", tz="UTC")
    # Decision at 02:00 still must not include the 00:00-04:00 partial bar.
    h4b = completed_4h(df, pd.Timestamp("2026-10-27 02:00", tz="UTC"))
    assert h4b.index.max() == pd.Timestamp("2026-10-26 20:00", tz="UTC")
    assert pd.Timestamp("2026-10-27 00:00", tz="UTC") not in h4b.index


def test_4h_features_ignore_future_prices():
    idx = pd.date_range("2026-09-01 00:00", periods=400, freq="1h", tz="UTC")
    close = pd.Series(range(400), index=idx, dtype=float)
    df = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1.0,
    }, index=idx)
    t = pd.Timestamp("2026-09-10 00:00", tz="UTC")
    a = evaluate_gate(df, t, "long")
    df2 = df.copy()
    df2.loc[df2.index >= t, ["open", "high", "low", "close"]] *= 3
    b = evaluate_gate(df2, t, "long")
    assert a["four_hour"] == b["four_hour"]
    assert a["one_hour"] == b["one_hour"]
    assert a["extension_atr"] == b["extension_atr"]
    assert a["eligible"] == b["eligible"]


def test_signed_extension_is_directional():
    assert signed_extension_atr(110, 100, 10, "long") == 1.0
    assert signed_extension_atr(110, 100, 10, "short") == 0.0
    assert signed_extension_atr(90, 100, 10, "short") == 1.0
    assert signed_extension_atr(90, 100, 10, "long") == 0.0


def test_results_do_not_claim_holdout_or_optimization():
    from piia.intelligence.htf_alignment_gate import OUT_DIR
    import json
    path = OUT_DIR / "RESULTS.json"
    pytest.skip("results missing") if not path.exists() else None
    r = json.loads(path.read_text())
    assert r["hypothesis_optimized"] is False
    assert r["held_out"] is False
    assert r["llm_rerun"] is False
    assert r["past_research_unchanged"] is True
    assert r["hypothesis_assessment"]["status"] == "insufficient"
    assert r["reconciliation"]["manual_ok"] is True
    assert r["reconciliation"]["independent_ok"] is True
    assert r["manual_lane"]["gated"]["closed_trades"] + len(r["manual_lane"]["blocked_packet_ids"]) == r["manual_lane"]["sample_directional_candidates"]


def test_formal_databases_untouched_by_gate_experiment():
    from piia.intelligence.historical_phase2 import FORBIDDEN_DB, PACKET_DIR
    from piia.intelligence.historical_independent import DB_PATH as IND_DB
    from piia.intelligence.historical_phase2 import DB_PATH as MAN_DB
    from piia.intelligence.htf_alignment_gate import OUT_DIR
    result = OUT_DIR / "RESULTS.json"
    pytest.skip("results missing") if not result.exists() else None
    mtime = result.stat().st_mtime
    for path in (FORBIDDEN_DB, IND_DB, MAN_DB, PACKET_DIR / "CHAT_DECISIONS_FROZEN.json"):
        if path.exists():
            assert path.stat().st_mtime < mtime
