"""SHA-256 integrity manifest for PIIA Judgment v3 modules & config."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]

PINNED_SHA256 = {
    "PIIA_JUDGMENT_V3_PROTOCOL.md": "679a6bb09b40c21b988d2927106560809d3b4bebf1f1bd5c684d32336b369dad",
    "config/piia_judgment_v3.yaml": "2249f32273f09de677f5b78e4552fbba18e670644363fb28fc4c316282ca3123",
    "src/piia/intelligence/judgment/news.py": "4bb8ecd0d46e3016cdd502517ac42a458a0841065ab84e963d092a32f828c0c3",
    "src/piia/intelligence/judgment/evidence.py": "0fb573743d677517ab2b7e6f39b35c7530a4abd2c7026c355cd4a483859593c6",
    "src/piia/intelligence/judgment/tide.py": "88fda335b845915eebad16db4f4841c6fcdf8b0aae004f60c149776755a059fb",
    "src/piia/intelligence/judgment/synthesizer.py": "eec3b3e378d5f381ba8c86e432ea6505967c324d081ec9ebe144284384fb0ba4",
    "src/piia/intelligence/judgment/chat_workbench.py": "70df54bada077938f8ec9c076e704d2cd1d709a8da06ef8e70663b06111c661c",
    "src/piia/intelligence/judgment/evaluation.py": "0a511066fb6c29b256bf1eedd9ff6e5d97903840b71352d9bd5b839396117cf2",
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
