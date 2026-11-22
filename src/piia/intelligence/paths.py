from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "piia"
EVIDENCE = DATA / "evidence"
DEFAULT_SQLITE = DATA / "piia.dev.sqlite"
def default_database_url() -> str:
    return "sqlite:///" + DEFAULT_SQLITE.resolve().as_posix()
