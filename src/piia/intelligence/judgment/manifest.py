"""SHA-256 integrity manifest for PIIA Judgment v3 modules & config."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]

PINNED_SHA256 = {
    "PIIA_JUDGMENT_V3_PROTOCOL.md": "4ce921613aa29036f76277ec5a6464c9700f59ab7cc58933fe67baac50002add",
    "config/piia_judgment_v3.yaml": "09b9ae5dae300e6e1e655a04017de16a75293241f5f263597ae4578dcec33bcc",
    "src/piia/intelligence/judgment/news.py": "bc71a395d11a4f950f5aabf397440b69a1bfaf88cffae9d68710c5a7f943707f",
    "src/piia/intelligence/judgment/evidence.py": "33e2cc249cebfbaf80901c5c36e01d9e073d8f6300e5a85cc7bd4913d2bdf90a",
    "src/piia/intelligence/judgment/tide.py": "d1fdbca87698756c0a00d4445dc94822fa071fad741e1d1bb0ab28ad568e5b37",
    "src/piia/intelligence/judgment/synthesizer.py": "43b57bdc1d74b7c7afa1845af1263f6388d0b48d72b2bc1b1899ad8bbbab98ab",
    "src/piia/intelligence/judgment/chat_workbench.py": "65a7c844d7e4489f99b68a5e705e108447494e4063e48ddad4360e8cfc2bc896",
    "src/piia/intelligence/judgment/evaluation.py": "599c72a5e62a8a72e7c36d0f3013db22bc768e5df8be4160273fcf4a3941d2f2",
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
