"""Cryptographic SHA-256 Freeze Verifier for PIIA v2 Protocol & Modules.

Pinned BEFORE fetching or evaluating post-2026-11-16 out-of-sample market data.
Any modification to the protocol, thresholds, roles, risk gate, ablation harness,
or continuous paper engine fails closed until a new policy version is declared.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]

PINNED_SHA256 = {
    "PIIA_V2_PROTOCOL.md": "48bb560d684764295484d849f00f5bbcda1c6fb92fabdcad172277c68773c33c",
    "config/piia_agent_v2.yaml": "57a0a768a062065d4ec3b3c7992726744175db91a2a823487205cbe00da611c7",
    "src/piia/intelligence/agent_v2/pillars.py": "bac65ec297e587d761c8bc92cd312d0cf9ec2373682450718b46f937f8eb549e",
    "src/piia/intelligence/agent_v2/roles.py": "befdcbfb9a20f81ca1ff11ef5b7734a0c39d4e3b90781f98d53b6d392acf4567",
    "src/piia/intelligence/agent_v2/risk_gate.py": "63d666592b6ddcaa5fc416ba6f027a54fac591017244d1df25dc44911ec4d20a",
    "src/piia/intelligence/agent_v2/ablation.py": "9b8f77d5a320524c94df61151c4e3596863a2e1506aee161d756326afd0a41b5",
    "src/piia/intelligence/agent_v2/continuous_paper.py": "db9754a70ae448c718fd4fdbb011ca380ee82745315ab6bc20a700bfe4f99ed5",
}


def verify_piia_v2_frozen(root: Path | None = None) -> dict[str, Any]:
    """Verify LF-normalized SHA-256 hashes of all frozen PIIA v2 files."""
    base = Path(root) if root else ROOT
    issues: list[str] = []
    observed: dict[str, str] = {}
    for rel, expected in PINNED_SHA256.items():
        target = base / rel
        if not target.is_file():
            issues.append(f"missing:{rel}")
            continue
        raw = target.read_bytes().replace(b"\r\n", b"\n")
        digest = hashlib.sha256(raw).hexdigest()
        observed[rel] = digest
        if digest != expected:
            issues.append(f"modified:{rel}")
    manifest_sha = hashlib.sha256(
        json.dumps(PINNED_SHA256, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "ok": not issues,
        "issues": issues,
        "manifest_sha256": manifest_sha,
        "files": observed,
        "limitations": [
            "Local SHA-256 verifies source consistency against the frozen pre-registration, not independent third-party timestamp notarization.",
        ],
    }
