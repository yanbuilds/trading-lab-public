"""Run pinned Chronos-2 locally on synthetic or captured closes; no network."""
import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path

WEIGHTS = "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42"
CONFIG = "ef1143bfdc9c0376d9a056eefca46cb4b1ec3d0ffacd541ff56feb40fb708031"
REVISION = "95a9710e2596287d08352589f42634fa5abdf0a7"


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()
    if version("chronos-forecasting") != "2.3.2" or version("torch").split("+")[0] != "2.10.0":
        raise ValueError("unsupported model runtime")
    if digest(args.model / "model.safetensors") != WEIGHTS or digest(args.model / "config.json") != CONFIG:
        raise ValueError("model files do not match pinned hashes")
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data or set(data) - {"BTCUSDT", "XAUUSDT", "NDX100USDT"}:
        raise ValueError("invalid input symbols")
    for values in data.values():
        if not isinstance(values, list) or len(values) != 300 or any(
                isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("invalid closed-bar inputs")
    import numpy as np
    import torch
    from chronos import BaseChronosPipeline
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = BaseChronosPipeline.from_pretrained(str(args.model.resolve()), device_map="cpu", local_files_only=True)
    symbols = sorted(data)
    quantiles, _ = model.predict_quantiles([np.array(data[s], dtype=np.float32) for s in symbols],
                                          prediction_length=24, quantile_levels=[.1, .5, .9], batch_size=3)
    outputs = {}
    for symbol, tensor in zip(symbols, quantiles, strict=True):
        values = tensor.detach().cpu().numpy().squeeze()
        if (values.shape != (24, 3) or not np.isfinite(values).all() or not (values > 0).all()
                or not ((values[:, 0] <= values[:, 1]) & (values[:, 1] <= values[:, 2])).all()):
            raise ValueError("invalid forecast result")
        outputs[symbol] = {"q10": float(values[-1, 0]), "median": float(values[-1, 1]), "q90": float(values[-1, 2])}
    if digest(args.model / "model.safetensors") != WEIGHTS or digest(args.model / "config.json") != CONFIG:
        raise ValueError("model changed during inference")
    args.output.write_text(json.dumps({"model_revision": REVISION, "weights_sha256": WEIGHTS,
        "config_sha256": CONFIG, "chronos_version": "2.3.2", "torch_version": "2.10.0",
        "device": "cpu", "horizon_hours": 24, "context_bars": 300,
        "seed": 0, "outputs": outputs, "order_permissions": False}, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
