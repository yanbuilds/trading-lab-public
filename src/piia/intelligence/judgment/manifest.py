"""SHA-256 integrity manifest for PIIA Judgment v3 modules & config."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]

PINNED_SHA256 = {
    "PIIA_JUDGMENT_V3_PROTOCOL.md": "d66bd9102c233fbe710c1b4f84b7c64c62c154e0eb6b0eab0af586d4c6f7cd8f",
    "config/piia_judgment_v3.yaml": "a709198eb5ec43ea89b80ec3d887a27194c867d03b2aae7df2f68ffc5f10263a",
    "src/piia/intelligence/judgment/news.py": "4bb8ecd0d46e3016cdd502517ac42a458a0841065ab84e963d092a32f828c0c3",
    "src/piia/intelligence/judgment/evidence.py": "7b438676b023e57299d1a8ccae97fb4381c37b5577e0cdcd8e7efdc7eb2f4ef2",
    "src/piia/intelligence/judgment/tide.py": "88fda335b845915eebad16db4f4841c6fcdf8b0aae004f60c149776755a059fb",
    "src/piia/intelligence/judgment/synthesizer.py": "57e2251e53ff4a0ace52447e9b37a3740b5616cf0e16279581813598d2b41fa9",
    "src/piia/intelligence/judgment/chat_workbench.py": "9946ab0dfea7dd53b7794ce6235757c1aebf21a02218612e7f99f301d10bf133",
    "src/piia/intelligence/judgment/evaluation.py": "0763db831f92a652307dd4909e46b521e5297ab6a52b79f403f05608bffbaefd",
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
