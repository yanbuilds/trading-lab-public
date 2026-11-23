"""Regression tests for PIIA v2.2 ledger isolation, prospective counting, and three-contract alignment.

All inputs are labelled synthetic fixtures; the live collector and clock are monkeypatched, so no
network access occurs and nothing is written outside pytest's tmp_path.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pytest

import piia.intelligence.agent_v2.continuous_paper as cp
from piia.intelligence.agent_v2.manifest import verify_piia_v2_frozen
from piia.intelligence.agent_v2.pillars import SYMBOLS, load_v2_config

ONE_H = pd.Timedelta(hours=1)
# Comfortably after the v2.2 prospective freeze start registered in config/piia_agent_v2.yaml.
FAKE_NOW = datetime(2026, 11, 30, 10, 20, tzinfo=timezone.utc)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _candles(sym: str, last_close: pd.Timestamp, bars: int = 300, seed: int = 7) -> list[dict]:
    """Synthetic closed 1h candles whose final bar closes exactly at `last_close`."""
    base = {"BTCUSDT": 65000.0, "XAUUSDT": 3850.0, "NDX100USDT": 27500.0}[sym]
    rng = np.random.default_rng(seed + len(sym))
    rets = rng.normal(0.0, 0.002, size=bars)
    if sym == "BTCUSDT":
        rets[-60:] = np.abs(rets[-60:]) * 0.5 + 0.0022
        rets[-1] = 0.011
    closes = base * np.cumprod(1.0 + rets)
    first_open = last_close - bars * ONE_H
    out = []
    for i in range(bars):
        t_open = (first_open + i * ONE_H).to_pydatetime()
        o = float(closes[i - 1]) if i else float(closes[0] * 0.999)
        c = float(closes[i])
        span = abs(c - o) + c * 0.0015
        out.append({
            "open_at_utc": t_open.isoformat(),
            "close_at_utc": (t_open + timedelta(hours=1)).isoformat(),
            "open": o,
            "high": max(o, c) + 0.4 * span,
            "low": min(o, c) - 0.4 * span,
            "close": c,
            "volume": 100.0,
        })
    return out


def _snapshot(last_close_by_sym: dict[str, pd.Timestamp | None], collected_at: datetime) -> dict:
    instruments = {}
    for sym in SYMBOLS:
        lc = last_close_by_sym.get(sym)
        candles = _candles(sym, lc) if lc is not None else []
        last_px = candles[-1]["close"] if candles else None
        instruments[sym] = {
            "candles_closed_1h": candles,
            "funding": {"funding_rate": "0.0001"},
            "ticker": {"mark": last_px, "last": last_px, "observed_at_utc": collected_at.isoformat()} if candles else None,
            "open_interest": None,
        }
    return {
        "collected_at_utc": collected_at.isoformat(),
        "market_snapshot": {"retrieved_at_utc": collected_at.isoformat(), "instruments": instruments},
        "books_snapshot": {"ok": False, "instruments": {}},
        "tiers_snapshot": {"ok": False, "instruments": {}},
        "settled_funding": {"ok": False, "instruments": {}},
        "macro_context": {"status": "unavailable", "historical_point_in_time_lagged": False,
                          "macro_tightening_headwind": False, "macro_easing_tailwind": False},
        "event_calendar_context": {"status": "unavailable", "historical_replay_supported": False},
        "flows_positioning_context": {"status": "unavailable", "liquidity_state": "unknown", "historical_replay_supported": False},
        "onchain_context": {"status": "unavailable", "historical_replay_supported": False},
        "polymarket_context": {"status": "unavailable", "historical_replay_supported": False},
        "chronos_forecast": {"weights_ran": False, "forecasts": {}},
    }


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Redirect default ledgers to tmp, freeze the clock, and stub the live public collector."""
    real = tmp_path / "live_prospective.sqlite"
    demo = tmp_path / "demo_synthetic.sqlite"
    monkeypatch.setattr(cp, "REAL_LEDGER_PATH", real)
    monkeypatch.setattr(cp, "DEMO_LEDGER_PATH", demo)
    monkeypatch.setattr(cp, "LEGACY_LEDGER_PATHS", ())
    state = {"now": FAKE_NOW, "snapshot": None}
    monkeypatch.setattr(cp, "_utc_now", lambda: state["now"])

    def _fake_collect(*_a, **_k):
        assert state["snapshot"] is not None, "test must provide a live snapshot"
        return state["snapshot"]

    monkeypatch.setattr(cp, "collect_live_six_pillar_snapshot", _fake_collect)
    return {"real": real, "demo": demo, "state": state, "tmp": tmp_path}


def _aligned(now: datetime) -> dict:
    cutoff = pd.Timestamp(now).floor("h")
    return _snapshot({s: cutoff for s in SYMBOLS}, now)


# ---------------------------------------------------------------- 1. ledger isolation


def test_default_ledgers_are_distinct_and_demo_then_live_same_hour_both_record(env):
    assert not cp._same_file(cp.default_ledger_path(False), cp.default_ledger_path(True))

    demo_res = cp.run_prospective_cycle(demo_mode=True)
    assert demo_res["status"] == "recorded_prospective_cycle"
    assert env["demo"].is_file() and not env["real"].exists()

    env["state"]["snapshot"] = _aligned(FAKE_NOW)
    live_res = cp.run_prospective_cycle()
    assert live_res["status"] == "recorded_prospective_cycle"
    assert live_res["cutoff_hour_utc"] == demo_res["cutoff_hour_utc"]  # same hour, not blocked

    live_status = cp.get_continuous_paper_status()
    assert live_status["ledger_kind"] == cp.KIND_LIVE
    assert live_status["prospective_validation"]["counted_observations"] == 1
    demo_status = cp.get_continuous_paper_status(demo=True)
    assert demo_status["ledger_kind"] == cp.KIND_DEMO
    assert demo_status["prospective_validation"]["counted_observations"] == 0
    assert demo_status["validation_status"] == "NOT_A_PROSPECTIVE_LEDGER"


def test_demo_and_injected_writes_are_refused_by_live_ledger(env):
    env["state"]["snapshot"] = _aligned(FAKE_NOW)
    assert cp.run_prospective_cycle()["status"] == "recorded_prospective_cycle"
    before = _sha(env["real"])

    with pytest.raises(cp.LedgerKindError):
        cp.run_prospective_cycle(db_path=env["real"], demo_mode=True)
    with pytest.raises(cp.LedgerKindError):
        cp.run_prospective_cycle(db_path=env["real"], snapshot_override=_aligned(FAKE_NOW))
    with pytest.raises(cp.LedgerKindError):
        cp.run_prospective_cycle(snapshot_override=_aligned(FAKE_NOW))  # injected needs explicit path

    # A copy of a live-labelled ledger at another path still refuses demo/injected writes.
    other_live = env["tmp"] / "other_live.sqlite"
    other_live.write_bytes(env["real"].read_bytes())
    with pytest.raises(cp.LedgerKindError):
        cp.run_prospective_cycle(db_path=other_live, demo_mode=True)
    # And a demo ledger refuses live collection.
    cp.run_prospective_cycle(demo_mode=True)
    with pytest.raises(cp.LedgerKindError):
        cp.run_prospective_cycle(db_path=env["demo"])

    assert _sha(env["real"]) == before
    assert cp.get_continuous_paper_status()["prospective_validation"]["counted_observations"] == 1


def test_injected_samples_labelled_live_never_count(env):
    snap = _aligned(FAKE_NOW)
    snap["mode"] = cp.MODE_LIVE  # a caller cannot relabel injected data as live
    inj = env["tmp"] / "injected.sqlite"
    res = cp.run_prospective_cycle(db_path=inj, snapshot_override=snap)
    assert res["status"] == "recorded_prospective_cycle"
    assert res["mode"] == cp.MODE_INJECTED
    st = cp.get_continuous_paper_status(inj)
    assert st["ledger_kind"] == cp.KIND_INJECTED
    assert st["prospective_validation"]["counted_observations"] == 0

    # Live-ledger approvals refuse caller-injected quotes / clocks.
    env["state"]["snapshot"] = _aligned(FAKE_NOW)
    cp.run_prospective_cycle()
    with pytest.raises(cp.LedgerKindError):
        cp.approve_human_proposal(
            "any", choice="approve", db_path=env["real"],
            current_quote={"price": 1.0, "observed_at_utc": FAKE_NOW.isoformat()},
        )


# ---------------------------------------------------------------- 2. legacy / mixed ledgers


def _make_legacy_mixed_ledger(path: Path) -> None:
    """Emulate a pre-v2.2 unlabelled ledger that mixes demo and live cycles in one file."""
    with sqlite3.connect(path) as conn:
        conn.execute(f"PRAGMA application_id={cp.APP_ID}")
        conn.executescript(
            """
            CREATE TABLE cycles (cycle_id TEXT PRIMARY KEY, cutoff_hour_utc TEXT NOT NULL UNIQUE,
                recorded_at_utc TEXT NOT NULL, manifest_sha256 TEXT NOT NULL, mode TEXT NOT NULL,
                pillars_summary_json TEXT NOT NULL, cycle_hash TEXT NOT NULL);
            CREATE TABLE ledger_events (event_id TEXT PRIMARY KEY, recorded_at_utc TEXT NOT NULL,
                book_name TEXT NOT NULL, event_kind TEXT NOT NULL, symbol TEXT,
                payload_json TEXT NOT NULL, cash_after_usdt REAL NOT NULL);
            """
        )
        conn.execute("INSERT INTO cycles VALUES ('c1','2026-11-23T00:00:00+00:00','2026-11-23T00:10:00+00:00','m','SYNTHETIC_DEMO_NOT_LIVE','{}','h1')")
        conn.execute("INSERT INTO cycles VALUES ('c2','2026-11-23T01:00:00+00:00','2026-11-23T01:10:00+00:00','m','live_prospective_public_6_pillars','{}','h2')")
        conn.execute("INSERT INTO ledger_events VALUES ('e1','2026-11-23T00:10:00+00:00','M4_full_piia_v2','init_book',NULL,?,500.0)",
                     (json.dumps({"starting_cash": 500.0, "mode": "SYNTHETIC_DEMO_NOT_LIVE"}),))


def test_mixed_legacy_ledger_is_reported_preserved_and_never_valid(env, monkeypatch):
    legacy = env["tmp"] / "legacy_v2_1.sqlite"
    _make_legacy_mixed_ledger(legacy)
    before = _sha(legacy)

    audit = cp.inspect_ledger(legacy)
    assert audit["ledger_kind_label"] == "legacy_unlabeled_pre_v2_2"
    assert audit["classification"] == "mixed_demo_live_or_injected"
    assert audit["usable_as_live_prospective_ledger"] is False

    with pytest.raises(cp.LedgerKindError):
        cp._init_db(legacy)
    env["state"]["snapshot"] = _aligned(FAKE_NOW)
    with pytest.raises(cp.LedgerKindError):
        cp.run_prospective_cycle(db_path=legacy)
    with pytest.raises(cp.LedgerKindError):
        cp.run_prospective_cycle(db_path=legacy, demo_mode=True)

    st = cp.get_continuous_paper_status(legacy)
    assert st["prospective_validation"]["counted_observations"] == 0
    assert st["validation_status"] == "NOT_A_PROSPECTIVE_LEDGER"

    # Status of the fresh live ledger reports legacy ledgers without counting them.
    monkeypatch.setattr(cp, "LEGACY_LEDGER_PATHS", (legacy,))
    live_st = cp.get_continuous_paper_status()
    assert live_st["legacy_ledgers"][0]["classification"] == "mixed_demo_live_or_injected"
    assert live_st["prospective_validation"]["counted_observations"] == 0
    assert _sha(legacy) == before  # preserved byte-for-byte


def test_live_labelled_ledger_with_foreign_rows_is_flagged_mixed(env):
    env["state"]["snapshot"] = _aligned(FAKE_NOW)
    cp.run_prospective_cycle()
    with sqlite3.connect(env["real"]) as conn:  # simulate tampering: an injected row appended
        conn.execute(
            "INSERT INTO cycles(cycle_id, cutoff_hour_utc, recorded_at_utc, manifest_sha256, mode, pillars_summary_json, cycle_hash) "
            "VALUES ('x','2026-11-30T11:00:00+00:00','2026-11-30T11:05:00+00:00','m',?, '{}','h')",
            (cp.MODE_DEMO,),
        )
    st = cp.get_continuous_paper_status()
    assert st["ledger_audit"]["classification"] == "mixed_demo_live_or_injected"
    assert st["prospective_validation"]["counted_observations"] == 0


# ---------------------------------------------------------------- 3. contiguous prospective counting


def _row(cutoff: pd.Timestamp, *, mode=None, manifest=None, version=None, late=False) -> dict:
    cfg = load_v2_config()
    rec = cutoff + (pd.Timedelta(minutes=70) if late else pd.Timedelta(minutes=5))
    return {
        "cutoff_hour_utc": cutoff.isoformat(),
        "capture_started_utc": (cutoff + pd.Timedelta(minutes=2)).isoformat(),
        "recorded_at_utc": rec.isoformat(),
        "mode": mode or cp.MODE_LIVE,
        "manifest_sha256": manifest or verify_piia_v2_frozen()["manifest_sha256"],
        "policy_version": version or cfg["version"],
    }


def test_gapped_records_are_not_one_contiguous_window():
    cfg = load_v2_config()
    man = verify_piia_v2_frozen()["manifest_sha256"]
    start = pd.Timestamp(cfg["prospective_freeze_started_utc"])
    hours = [0, 1, 2, 3, 4, 6, 7, 8]
    rows = [_row(start + h * ONE_H) for h in hours]
    rows += [
        _row(start - ONE_H),                                  # before freeze
        _row(start + 20 * ONE_H, mode=cp.MODE_DEMO),          # demo row
        _row(start + 21 * ONE_H, manifest="other"),           # different frozen code
        _row(start + 22 * ONE_H, late=True),                  # captured after its hour
    ]
    v = cp.compute_prospective_validation(
        rows, ledger_kind=cp.KIND_LIVE, cfg=cfg, manifest_sha256=man, now=(start + 9 * ONE_H).to_pydatetime()
    )
    assert v["counted_observations"] == 8
    assert v["longest_contiguous_hours"] == 5
    assert v["missing_hours_within_observed_span"] == 1
    assert v["missing_hours_since_freeze"] == 2  # hours 5 and 9
    assert v["excluded_cycles"] == {
        "before_freeze_start": 1, "not_live_mode": 1, "different_frozen_manifest": 1, "capture_outside_cutoff_hour": 1,
    }
    assert v["validation_status"] == "PENDING_INSUFFICIENT_PROSPECTIVE_SAMPLE"


def test_scattered_records_cannot_reach_sufficiency_but_contiguous_window_only_enters_audit():
    cfg = load_v2_config()
    man = verify_piia_v2_frozen()["manifest_sha256"]
    start = pd.Timestamp(cfg["prospective_freeze_started_utc"])
    req = int(cfg["evaluation_rules"]["prospective_min_contiguous_hours"])

    scattered = [_row(start + 2 * i * ONE_H) for i in range(req + 40)]
    v1 = cp.compute_prospective_validation(scattered, ledger_kind=cp.KIND_LIVE, cfg=cfg, manifest_sha256=man)
    assert v1["counted_observations"] == req + 40
    assert v1["longest_contiguous_hours"] == 1
    assert v1["validation_status"] == "PENDING_INSUFFICIENT_PROSPECTIVE_SAMPLE"

    contiguous = [_row(start + i * ONE_H) for i in range(req)]
    v2 = cp.compute_prospective_validation(contiguous, ledger_kind=cp.KIND_LIVE, cfg=cfg, manifest_sha256=man)
    assert v2["longest_contiguous_hours"] == req
    assert v2["validation_status"] == "PENDING_AUDIT_CONTIGUOUS_WINDOW_REACHED"

    v3 = cp.compute_prospective_validation(contiguous, ledger_kind=cp.KIND_DEMO, cfg=cfg, manifest_sha256=man)
    assert v3["counted_observations"] == 0


# ---------------------------------------------------------------- 4. three-contract alignment


def test_lagging_contract_rejects_cycle_without_any_ledger_change_then_recovers_once(env):
    h1 = FAKE_NOW
    env["state"]["now"] = h1
    env["state"]["snapshot"] = _aligned(h1)
    first = cp.run_prospective_cycle()
    assert first["status"] == "recorded_prospective_cycle"
    before = _sha(env["real"])
    status_before = cp.get_continuous_paper_status()

    h2 = h1 + timedelta(hours=1)
    cutoff2 = pd.Timestamp(h2).floor("h")
    env["state"]["now"] = h2
    env["state"]["snapshot"] = _snapshot(
        {"BTCUSDT": cutoff2, "XAUUSDT": cutoff2 - ONE_H, "NDX100USDT": cutoff2}, h2
    )
    rej = cp.run_prospective_cycle()
    assert rej["status"] == "rejected_input_alignment"
    assert rej["ledger_written"] is False
    assert any(r.startswith("stale_or_lagging_candle:XAUUSDT") for r in rej["reasons"])
    assert "misaligned_latest_close_across_contracts" in rej["reasons"]
    assert "unapproved_human_proposals" not in rej
    assert _sha(env["real"]) == before  # no cycle, proposal, cursor, funding, or position change
    status_mid = cp.get_continuous_paper_status()
    assert status_mid["cycles_recorded"] == status_before["cycles_recorded"]
    assert status_mid["books"] == status_before["books"]

    env["state"]["snapshot"] = _aligned(h2)  # data recovers within the same hour
    ok = cp.run_prospective_cycle()
    assert ok["status"] == "recorded_prospective_cycle"
    assert ok["cutoff_hour_utc"] == cutoff2.isoformat()
    again = cp.run_prospective_cycle()
    assert again["status"] == "already_recorded_for_hour"
    final = cp.get_continuous_paper_status()
    assert final["cycles_recorded"] == 2
    assert final["prospective_validation"]["counted_observations"] == 2
    assert final["prospective_validation"]["longest_contiguous_hours"] == 2


@pytest.mark.parametrize(
    "shape,reason_prefix",
    [
        ({"BTCUSDT": 1, "XAUUSDT": 0, "NDX100USDT": 0}, "future_candle:BTCUSDT"),
        ({"BTCUSDT": 0, "XAUUSDT": None, "NDX100USDT": 0}, "missing_candles:XAUUSDT"),
        ({"BTCUSDT": -1, "XAUUSDT": -1, "NDX100USDT": -1}, "stale_or_lagging_candle:BTCUSDT"),
    ],
)
def test_future_missing_or_stale_bars_are_rejected_before_any_ledger_io(env, shape, reason_prefix):
    cutoff = pd.Timestamp(FAKE_NOW).floor("h")
    env["state"]["snapshot"] = _snapshot(
        {s: (None if off is None else cutoff + off * ONE_H) for s, off in shape.items()}, FAKE_NOW
    )
    res = cp.run_prospective_cycle()
    assert res["status"] == "rejected_input_alignment"
    assert any(r.startswith(reason_prefix) for r in res["reasons"])
    assert not env["real"].exists()


def test_collection_crossing_hour_boundary_is_rejected(env):
    snap = _aligned(FAKE_NOW)
    snap["collected_at_utc"] = (FAKE_NOW + timedelta(hours=1)).isoformat()
    env["state"]["snapshot"] = snap
    res = cp.run_prospective_cycle()
    assert res["status"] == "rejected_input_alignment"
    assert "collection_crossed_hour_boundary" in res["reasons"]
    assert not env["real"].exists()


def test_signal_inputs_are_selected_at_common_cutoff():
    cutoff = pd.Timestamp(FAKE_NOW).floor("h")
    snap = _snapshot({s: cutoff for s in SYMBOLS}, FAKE_NOW)
    align = cp.check_three_contract_alignment(snap["market_snapshot"], expected_cutoff=cutoff)
    assert align["ok"] is True
    for sym in SYMBOLS:
        assert align["frames"][sym].index[-1] + ONE_H == cutoff
        assert align["last_close_by_symbol"][sym] == cutoff.isoformat()


# ---------------------------------------------------------------- 5. pre-freeze write gate


def _build_pre_freeze_live_ledger(
    path: Path,
    *,
    freeze_start: datetime,
    starting_cash: float = 100.0,
    also_add_post_freeze_row: bool = False,
) -> None:
    """Create a schema-valid live_prospective SQLite ledger that contains a pre-freeze cycle (tmp_path only)."""
    pre_cutoff = (pd.Timestamp(freeze_start) - ONE_H).isoformat()
    pre_rec = (pd.Timestamp(freeze_start) - pd.Timedelta(minutes=45)).isoformat()
    man = verify_piia_v2_frozen()["manifest_sha256"]
    ver = load_v2_config()["version"]
    with sqlite3.connect(path) as conn:
        conn.execute(f"PRAGMA application_id={cp.APP_ID}")
        conn.executescript(
            """
            CREATE TABLE cycles (
                cycle_id TEXT PRIMARY KEY,
                cutoff_hour_utc TEXT NOT NULL UNIQUE,
                recorded_at_utc TEXT NOT NULL,
                manifest_sha256 TEXT NOT NULL,
                mode TEXT NOT NULL,
                pillars_summary_json TEXT NOT NULL,
                cycle_hash TEXT NOT NULL,
                capture_started_utc TEXT,
                policy_version TEXT
            );
            CREATE TABLE ledger_events (
                event_id TEXT PRIMARY KEY,
                recorded_at_utc TEXT NOT NULL,
                book_name TEXT NOT NULL,
                event_kind TEXT NOT NULL,
                symbol TEXT,
                payload_json TEXT NOT NULL,
                cash_after_usdt REAL NOT NULL
            );
            CREATE TABLE ledger_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TRIGGER immutable_cycles_upd BEFORE UPDATE ON cycles
            BEGIN SELECT RAISE(ABORT, 'append-only cycles'); END;
            CREATE TRIGGER immutable_cycles_del BEFORE DELETE ON cycles
            BEGIN SELECT RAISE(ABORT, 'append-only cycles'); END;
            """
        )
        conn.executemany(
            "INSERT INTO ledger_meta(key, value) VALUES (?, ?)",
            [
                ("ledger_kind", cp.KIND_LIVE),
                ("schema_version", cp.LEDGER_SCHEMA_VERSION),
                ("created_at_utc", pre_rec),
            ],
        )
        conn.execute(
            "INSERT INTO cycles(cycle_id, cutoff_hour_utc, recorded_at_utc, manifest_sha256, mode, "
            "pillars_summary_json, cycle_hash, capture_started_utc, policy_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("c_pre", pre_cutoff, pre_rec, man, cp.MODE_LIVE, "{}", "hash_pre", pre_rec, ver),
        )
        if also_add_post_freeze_row:
            post_cutoff = pd.Timestamp(freeze_start).isoformat()
            post_rec = (pd.Timestamp(freeze_start) + pd.Timedelta(minutes=5)).isoformat()
            conn.execute(
                "INSERT INTO cycles(cycle_id, cutoff_hour_utc, recorded_at_utc, manifest_sha256, mode, "
                "pillars_summary_json, cycle_hash, capture_started_utc, policy_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("c_post", post_cutoff, post_rec, man, cp.MODE_LIVE, "{}", "hash_post", post_rec, ver),
            )
        init_payload = json.dumps({"starting_cash": starting_cash, "mode": cp.MODE_LIVE, "book_name": "M4_full_piia_v2"})
        conn.execute(
            "INSERT INTO ledger_events(event_id, recorded_at_utc, book_name, event_kind, symbol, payload_json, cash_after_usdt) "
            "VALUES (?, ?, ?, ?, NULL, ?, ?)",
            ("ev_pre", pre_rec, "M4_full_piia_v2", "init_book", init_payload, starting_cash),
        )


def test_pre_freeze_live_collection_rejected_without_creating_ledger_then_freeze_hour_starts_clean_500u(env, monkeypatch):
    from fastapi.testclient import TestClient
    from piia.intelligence.api import app
    from piia.intelligence.cli import main as piia_main

    freeze_start = pd.Timestamp(load_v2_config()["prospective_freeze_started_utc"]).to_pydatetime()

    # 1. One hour before freeze: attempt live collection with 100 USDT via direct call, CLI, and FastAPI.
    t_pre = freeze_start - timedelta(hours=1) + timedelta(minutes=15)
    env["state"]["now"] = t_pre
    env["state"]["snapshot"] = None  # rejected before collector or SQLite I/O

    res_pre = cp.run_prospective_cycle(starting_cash=100.0)
    assert res_pre["status"] == "rejected_before_freeze_start"
    assert res_pre["ledger_written"] is False
    assert not env["real"].exists()

    # Also test 1 second before freeze_start (still falls into freeze_start - 1h cutoff).
    env["state"]["now"] = freeze_start - timedelta(seconds=1)
    env["state"]["snapshot"] = _aligned(t_pre)
    assert cp.run_prospective_cycle(starting_cash=100.0)["status"] == "rejected_before_freeze_start"
    assert not env["real"].exists()

    # CLI route shares the same underlying gate and creates no ledger file.
    cli_live = env["tmp"] / "cli_pre_freeze_live.sqlite"
    assert piia_main(["agent-run", "--starting-cash", "100", "--db", str(cli_live)]) == 3
    assert not cli_live.exists()

    # FastAPI route shares the same underlying gate and creates no ledger file.
    api_live = env["tmp"] / "api_pre_freeze_live.sqlite"
    monkeypatch.setenv("PIIA_V2_LEDGER_PATH", str(api_live))
    with TestClient(app) as client:
        resp = client.post("/agent-v2/run-cycle", json={"demo_mode": False, "starting_cash": 100.0})
        assert resp.status_code == 200
        assert resp.json()["status"] == "rejected_before_freeze_start"
        assert resp.json()["ledger_written"] is False
    assert not api_live.exists()

    # 2. Exact freeze start hour: first live collection with 500 USDT succeeds with clean 500 starting_cash.
    t_start = freeze_start + timedelta(minutes=5)
    env["state"]["now"] = t_start
    env["state"]["snapshot"] = _aligned(t_start)
    res_ok = cp.run_prospective_cycle(starting_cash=500.0)
    assert res_ok["status"] == "recorded_prospective_cycle"
    assert res_ok["cutoff_hour_utc"] == pd.Timestamp(freeze_start).isoformat()
    assert env["real"].is_file()

    st = cp.get_continuous_paper_status()
    assert st["initialized"] is True
    assert st["usable_for_prospective_validation"] is True
    assert st["cycles_recorded"] == 1
    assert st["prospective_validation"]["counted_observations"] == 1
    assert st["prospective_validation"]["longest_contiguous_hours"] == 1
    for book_name, book in st["books"].items():
        assert book["starting_cash_usdt"] == 500.0, f"{book_name} carried wrong starting cash"


def test_existing_live_ledger_with_pre_freeze_cycle_is_flagged_invalid_and_refuses_writes_byte_for_byte(env, monkeypatch):
    from fastapi.testclient import TestClient
    from piia.intelligence.api import app

    freeze_start = pd.Timestamp(load_v2_config()["prospective_freeze_started_utc"]).to_pydatetime()
    contam = env["tmp"] / "pre_freeze_live.sqlite"
    contam_both = env["tmp"] / "pre_and_post_live.sqlite"
    _build_pre_freeze_live_ledger(contam, freeze_start=freeze_start, starting_cash=100.0, also_add_post_freeze_row=False)
    _build_pre_freeze_live_ledger(contam_both, freeze_start=freeze_start, starting_cash=100.0, also_add_post_freeze_row=True)
    sha_before = _sha(contam)
    sha2_before = _sha(contam_both)

    audit = cp.inspect_ledger(contam)
    assert audit["ledger_kind_label"] == cp.KIND_LIVE
    assert audit["contaminated_pre_freeze"] is True
    assert audit["pre_freeze_cycles"] == 1
    assert audit["pre_freeze_events"] >= 1
    assert audit["classification"] == "contaminated_pre_freeze_cycles"
    assert audit["usable_as_live_prospective_ledger"] is False

    # Move clock to post-freeze hour with valid aligned snapshot: continuation writes MUST fail closed.
    t_post = freeze_start + timedelta(hours=1, minutes=5)
    env["state"]["now"] = t_post
    env["state"]["snapshot"] = _aligned(t_post)

    with pytest.raises(cp.LedgerKindError, match="pre-freeze"):
        cp.run_prospective_cycle(db_path=contam, starting_cash=500.0)
    with pytest.raises(cp.LedgerKindError, match="pre-freeze"):
        cp._init_db(contam, ledger_kind=cp.KIND_LIVE)
    with pytest.raises(cp.LedgerKindError, match="pre-freeze"):
        cp.approve_human_proposal("any", choice="reject", db_path=contam)

    monkeypatch.setenv("PIIA_V2_LEDGER_PATH", str(contam))
    with TestClient(app) as client:
        resp = client.post("/agent-v2/run-cycle", json={"demo_mode": False, "starting_cash": 500.0})
        assert resp.status_code == 400
        assert "pre-freeze" in resp.json()["detail"].lower()

    # Even if a contaminated ledger already contains a post-freeze row, validation stays invalid with 0 counted.
    for path, expected_cycles in ((contam, 1), (contam_both, 2)):
        st = cp.get_continuous_paper_status(path)
        assert st["usable_for_prospective_validation"] is False
        assert st["cycles_recorded"] == expected_cycles
        assert st["books"]["M4_full_piia_v2"]["starting_cash_usdt"] == 100.0
        assert st["validation_status"] == "INVALID_PRE_FREEZE_CONTAMINATED_LEDGER"
        assert st["prospective_validation"]["counted_observations"] == 0

    assert _sha(contam) == sha_before
    assert _sha(contam_both) == sha2_before

