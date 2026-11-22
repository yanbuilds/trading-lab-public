from __future__ import annotations
from pathlib import Path
import yaml
from piia.intelligence.hashing import content_hash
from piia.intelligence.paths import ROOT

DEFAULT_PATH = ROOT / "config" / "piia_paper_risk.yaml"

def load_owner_risk(path: Path | None = None) -> tuple[dict, str]:
    p = Path(path or DEFAULT_PATH)
    raw = p.read_bytes()
    data = yaml.safe_load(raw) or {}
    if data.get("exchange_live_verified") is True:
        raise ValueError("owner risk file cannot enable exchange_live_verified in this build")
    data["exchange_live_verified"] = False
    data["engine"] = data.get("engine") or "local_simplified"
    return data, content_hash(raw)
