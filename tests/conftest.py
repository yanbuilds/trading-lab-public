from __future__ import annotations

import sys
from pathlib import Path

# 让 `import helpers` 在 pytest 收集时可用
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from helpers import make_cfg, ohlc_from_close, panel_from_frames  # noqa: F401
except ModuleNotFoundError:
    # Isolated PIIA env does not install the legacy strategy-lab dependencies.
    make_cfg = ohlc_from_close = panel_from_frames = None  # type: ignore


def tmp_cfg(tmp_path):
    if make_cfg is None:
        raise RuntimeError("legacy helpers unavailable in this environment")
    return make_cfg(tmp_path)
