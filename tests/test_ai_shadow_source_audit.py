from datetime import datetime, timezone
from types import SimpleNamespace
from piia.intelligence import ai_shadow_source_audit as audit


def test_source_check_rejects_uncommitted_and_late_files(monkeypatch, tmp_path):
    path = tmp_path / "source.py"
    path.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "SOURCES", ("source.py",))
    def fake_run(cmd, **kwargs):
        return SimpleNamespace(stdout=b"fixture" if cmd[1] == "show" else b"1794873600")
    monkeypatch.setattr(audit.subprocess, "run", fake_run)
    assert audit.check_sources()["ok"]
    path.write_text("changed", encoding="utf-8")
    assert "uncommitted_or_changed:source.py" in audit.check_sources()["issues"]
    earlier = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert "committed_at_or_after_cutoff:source.py" in audit.check_sources(before=earlier)["issues"]


def test_failed_audit_hides_ai_results_without_mutating_original(monkeypatch):
    monkeypatch.setattr(audit, "check_sources", lambda **_: {"ok": False, "issues": ["fixture"]})
    original = {"first_decision_utc": "2026-11-18T00:00:00+00:00", "ai_shadow_status": "complete",
                "accounts": {"ai_shadow": {"final": 120}, "candidate": {"final": 100}},
                "scenarios": {"stress": {"accounts": {"ai_shadow": {"final": 90}}}}}
    result, check = audit.gate_report(original)
    assert not check["ok"] and result["ai_shadow_status"] == "source_audit_failed"
    assert "ai_shadow" not in result["accounts"] and "candidate" in result["accounts"]
    assert "ai_shadow" not in result["scenarios"]["stress"]["accounts"]
    assert "ai_shadow" in original["accounts"]
