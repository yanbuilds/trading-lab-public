"""Synthetic observations test segmentation only; no historical market results."""
from datetime import timedelta
import json
import sqlite3

import pytest

from experiments import segmented_forward_report as segment


def row(hour):
    return {"cutoff_utc": (segment.ACTIVATION + timedelta(hours=hour)).isoformat()}


def test_all_segments_are_retained_and_never_aggregated(monkeypatch, tmp_path):
    monkeypatch.setattr(segment, "audit_claims", lambda *a, **kw: {"ok": False})
    monkeypatch.setattr(segment.report, "read_chronos_events", lambda *_: {})
    monkeypatch.setattr(segment.report, "read_shadow_events", lambda *_: {})
    monkeypatch.setattr(segment.chronos2_shadow_report, "read_events", lambda *_: {})
    monkeypatch.setattr(segment.report, "recorded_specs", lambda records: {})
    monkeypatch.setattr(segment.report, "evaluate", lambda records, specs, **kw: {"observations": len(records)})
    monkeypatch.setattr(segment.chronos2_shadow_report, "evaluate", lambda *a, **kw: (_ for _ in ()).throw(ValueError("missing")))
    result = segment.build_report([row(-1), row(0), row(1), row(3), row(4), row(5)], tmp_path / "fixture")
    assert result["excluded_pre_activation_observations"] == 1
    assert result["missing_hours_between_segments"] == 1
    assert [s["observations"] for s in result["segments"]] == [2, 3]
    assert result["segments"][0]["status"] == "insufficient_contiguous_observations"
    assert result["segments"][1]["status"] == "price_only_local_diagnostic"
    assert result["segments"][1]["result"]["chronos2_shadow"]["result"] is None
    assert result["aggregate_return"] is None and result["order_permissions"] is False


def test_unsorted_or_duplicate_observations_fail():
    with pytest.raises(ValueError):
        segment.partition([row(0), row(0)])
    with pytest.raises(ValueError):
        segment.partition([row(1), row(0)])


def test_real_replay_on_synthetic_discontinuous_ledger(tmp_path):
    from experiments import frozen_policy_capture as capture
    from tests.test_prospective_paper_report import synthetic_record
    records = [synthetic_record(segment.ACTIVATION + timedelta(hours=n)) for n in (0, 1, 2, 4)]
    ledger = tmp_path / "synthetic.sqlite"
    with sqlite3.connect(ledger) as db:
        db.execute(f"PRAGMA application_id={capture.APP_ID}")
        db.execute("CREATE TABLE observations (id INTEGER PRIMARY KEY, cutoff TEXT, created_at TEXT, payload_json TEXT)")
        for i, item in enumerate(records, 1):
            db.execute("INSERT INTO observations VALUES (?,?,?,?)", (i, item["cutoff_utc"], item["created_at_utc"], json.dumps(item)))
    output = segment.build_report(segment.report.read_observations(ledger), ledger)
    assert output["missing_hours_between_segments"] == 1
    assert [item["status"] for item in output["segments"]] == ["price_only_local_diagnostic", "insufficient_contiguous_observations"]
    first = output["segments"][0]["result"]
    assert first["candidate"]["initial"] == first["ema"]["initial"] == 100
    assert first["candidate"]["funding_net_paid"] is None
    assert first["ai_shadow"]["result"] is None and first["chronos2_shadow"]["result"] is None
    assert output["aggregate_return"] is None
