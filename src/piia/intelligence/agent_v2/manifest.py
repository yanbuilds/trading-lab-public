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
    "PIIA_V2_PROTOCOL.md": "45f702ec4b53d62377d2c0da94b7568d218dfc547248eb4f7cf946ccffa96e32",
    "config/piia_agent_v2.yaml": "967ac9efa76bc74be0865fc2f68857557f3b0e814e6ce928c6401357a438e25f",
    "src/piia/intelligence/agent_v2/pillars.py": "2984e74a7816f1ca4ce46180ed2027c2589c99a7f8a9ca3c1b2d2711b02f3988",
    "src/piia/intelligence/agent_v2/roles.py": "f25d2999577ec5243ef9e75218380261c5eab0219ce7550be52b137652d71804",
    "src/piia/intelligence/agent_v2/risk_gate.py": "e15bea45dc60088e177a3ce49a7281b90ffb26202c0d08ab27c58507f77bfead",
    "src/piia/intelligence/agent_v2/ablation.py": "4876a358f9a4522b8bd54a3fc8fec2d112fd83740d3c3e7ac316fcb632a30921",
    "src/piia/intelligence/agent_v2/continuous_paper.py": "6b40ae73e60d36a20268dcdc82f562f7cf1dbc6bc1794e200422693a86695f08",
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
