"""Local pre-observation source check, not independent timestamp or model verification."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import subprocess
from copy import deepcopy

from piia.intelligence.paths import ROOT

SOURCES = (
    "AI_SHADOW_V1.md", "src/piia/intelligence/ai_shadow.py",
    "src/piia/intelligence/agents/workflow.py", "src/piia/intelligence/agents/providers.py",
    "src/piia/intelligence/agents/io.py", "src/piia/intelligence/settings.py",
    "src/piia/intelligence/hashing.py", "src/piia/intelligence/ai_shadow_source_audit.py",
)
POLICY_START = datetime(2026, 11, 18, tzinfo=timezone.utc)


def check_sources(*, before: datetime = POLICY_START) -> dict:
    """Check disk/HEAD equality and each source's latest commit against the cutoff."""
    if before.tzinfo is None or before.utcoffset() is None:
        raise ValueError("aware cutoff required")
    cutoff = min(before.astimezone(timezone.utc), POLICY_START)
    failures, files = [], {}
    for name in SOURCES:
        try:
            disk = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
            tracked = subprocess.run(["git", "show", f"HEAD:{name}"], cwd=ROOT, capture_output=True,
                                     timeout=15, check=True).stdout.replace(b"\r\n", b"\n")
            log = subprocess.run(["git", "log", "-1", "--format=%ct", "HEAD", "--", name], cwd=ROOT,
                                 capture_output=True, timeout=15, check=True).stdout.strip()
            committed = datetime.fromtimestamp(int(log), timezone.utc)
            if disk != tracked:
                failures.append(f"uncommitted_or_changed:{name}")
            if committed >= cutoff:
                failures.append(f"committed_at_or_after_cutoff:{name}")
            files[name] = {"sha256": hashlib.sha256(disk).hexdigest(), "last_commit_at_utc": committed.isoformat()}
        except (OSError, subprocess.SubprocessError, ValueError, OverflowError):
            failures.append(f"unavailable:{name}")
    return {"ok": not failures, "issues": failures, "cutoff_utc": cutoff.isoformat(), "files": files,
            "independent_timestamp_verified": False, "remote_model_version_verified": False,
            "limitations": ["Local Git metadata and clock are not independent proof.",
                            "Remote model aliases, dependencies, and runtime can change without source changes.",
                            "Passing does not establish an AI edge or approve trading."]}


def gate_report(report: dict) -> tuple[dict, dict | None]:
    """Hide AI results in local UI reports if workflow source history is unverified."""
    out = deepcopy(report)
    ai = out.get("ai_shadow")
    if not ((isinstance(ai, dict) and ai.get("status") == "complete")
            or out.get("ai_shadow_status") == "complete"):
        return out, None
    try:
        before = datetime.fromisoformat(out["first_decision_utc"])
        result = check_sources(before=before)
    except (KeyError, TypeError, ValueError):
        result = {"ok": False, "issues": ["invalid_or_missing_first_decision_time"]}
    if not result["ok"]:
        if isinstance(ai, dict):
            out["ai_shadow"] = {"status": "source_audit_failed", "result": None}
        if "ai_shadow_status" in out:
            out["ai_shadow_status"] = "source_audit_failed"
        if isinstance(out.get("accounts"), dict):
            out["accounts"].pop("ai_shadow", None)
        for scenario in (out.get("scenarios") or {}).values():
            if isinstance(scenario, dict) and isinstance(scenario.get("accounts"), dict):
                scenario["accounts"].pop("ai_shadow", None)
    return out, result
