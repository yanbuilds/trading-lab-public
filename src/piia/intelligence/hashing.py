from __future__ import annotations
import hashlib, re
from pathlib import Path
HASH_RE = re.compile(r"^[a-f0-9]{64}$")

def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def blob_path(store: Path, digest: str) -> Path:
    if not HASH_RE.match(digest):
        raise ValueError("content hash must be 64 lowercase hex characters")
    return store / digest[:2] / digest[2:4] / digest

def write_blob(store: Path, data: bytes) -> str:
    digest = content_hash(data)
    path = blob_path(store, digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)
    elif path.read_bytes() != data:
        raise ValueError("content-address collision")
    return digest

def read_blob(store: Path, digest: str) -> bytes:
    path = blob_path(store, digest)
    data = path.read_bytes()
    actual = content_hash(data)
    if actual != digest:
        raise ValueError(f"evidence blob hash mismatch: expected {digest}, got {actual}")
    return data
