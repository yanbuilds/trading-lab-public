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
    "PIIA_V2_PROTOCOL.md": "8e651cae821f2bcbac241b8666ddac5d647bab8d3fedf29dd76c79f09703d3f4",
    "config/piia_agent_v2.yaml": "40a8e6aa6c9782f58c355f85af9a9339b4d939d4c52f6a5154f597e7f3d512ba",
    "src/piia/intelligence/agent_v2/pillars.py": "2984e74a7816f1ca4ce46180ed2027c2589c99a7f8a9ca3c1b2d2711b02f3988",
    "src/piia/intelligence/agent_v2/roles.py": "f25d2999577ec5243ef9e75218380261c5eab0219ce7550be52b137652d71804",
    "src/piia/intelligence/agent_v2/risk_gate.py": "e15bea45dc60088e177a3ce49a7281b90ffb26202c0d08ab27c58507f77bfead",
    "src/piia/intelligence/agent_v2/ablation.py": "4876a358f9a4522b8bd54a3fc8fec2d112fd83740d3c3e7ac316fcb632a30921",
    "src/piia/intelligence/agent_v2/continuous_paper.py": "f3fca3ed7fbfbb71039f6d90ca58ad95ad7db148c5bd07bc19a2d4608617582d",
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
