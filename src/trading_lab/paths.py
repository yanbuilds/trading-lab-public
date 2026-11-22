from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
CONFIG_DEFAULT = ROOT / "config" / "default.yaml"
DATA = ROOT / "data"
CACHE = DATA / "cache"
CSV_DIR = DATA / "csv"
FIXTURES = DATA / "fixtures"
ARTIFACTS = ROOT / "artifacts"
APP = ROOT / "app"
