"""SHA-256 integrity manifest for PIIA Judgment v3 modules & config."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]

PINNED_SHA256 = {
    "PIIA_JUDGMENT_V3_PROTOCOL.md": "d66bd9102c233fbe710c1b4f84b7c64c62c154e0eb6b0eab0af586d4c6f7cd8f",
    "config/piia_judgment_v3.yaml": "e44fd493b99bbe403868ea54f595f23318a9b555cb2f78673d6592d3a0d30a2c",
    "src/piia/intelligence/judgment/news.py": "4bb8ecd0d46e3016cdd502517ac42a458a0841065ab84e963d092a32f828c0c3",
    "src/piia/intelligence/judgment/evidence.py": "d6ee73515980e20f820934800be35e987ea081b04dbcf476fa70792f7b562a2a",
    "src/piia/intelligence/judgment/tide.py": "88fda335b845915eebad16db4f4841c6fcdf8b0aae004f60c149776755a059fb",
    "src/piia/intelligence/judgment/synthesizer.py": "57e2251e53ff4a0ace52447e9b37a3740b5616cf0e16279581813598d2b41fa9",
    "src/piia/intelligence/judgment/chat_workbench.py": "cbe6d9dd14cd91231ead9331c53b565874f6488a616ae1141ed49a4491bc42b9",
    "src/piia/intelligence/judgment/evaluation.py": "ad2e6c8a37f720c9b67d2c108a86650baa51046421c95d582af64e6896a8c7f4",
    "experiments/replay_nvidia_20261123_case.py": "1fe93603fbe88742eb96a95f553ef550917a78d3d99972b4cff2d2f0d36a0a85",
}


def verify_judgment_v3_frozen(root: Path | None = None) -> dict[str, Any]:
    """Verify LF-normalized SHA-256 hashes of all PIIA Judgment v3 files."""
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
            "Local SHA-256 checks repository consistency only; prospective validation status requires >= 168 contiguous post-freeze hours.",
        ],
    }
