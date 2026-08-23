#!/usr/bin/env python3
"""Compare deployment-format TinyLiDARNet weights on an extracted sequence."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CONTROLLER_DIR = SCRIPT_DIR.parents[1] / "workspace/src/aichallenge_submit/tiny_lidar_net_controller"
sys.path.insert(0, str(CONTROLLER_DIR / "tiny_lidar_net_controller"))

from model.tinylidarnet import TinyLidarNetNp  # noqa: E402


def load_model(path: Path, input_dim: int) -> TinyLidarNetNp:
    model = TinyLidarNetNp(input_dim=input_dim, output_dim=2)
    loaded = np.load(path, allow_pickle=True)
    params = dict(loaded.items()) if isinstance(loaded, np.lib.npyio.NpzFile) else loaded.item()
    for name, value in params.items():
        if name in model.params:
            model.params[name] = value
    return model


def metrics(model: TinyLidarNetNp, scans: np.ndarray, steers: np.ndarray, batch_size: int) -> dict:
    predictions = []
    normalized = np.clip(scans, 0.0, 30.0).astype(np.float32) / 30.0
    for start in range(0, len(normalized), batch_size):
        predictions.append(model(normalized[start:start + batch_size, None, :])[:, 1])
    predicted = np.concatenate(predictions)
    error = predicted - steers
    sharp = np.abs(steers) >= 0.25
    return {
        "samples": len(steers),
        "steering_mae": float(np.mean(np.abs(error))),
        "steering_rmse": float(np.sqrt(np.mean(error ** 2))),
        "steering_correlation": float(np.corrcoef(predicted, steers)[0, 1]),
        "sharp_turn_samples": int(np.sum(sharp)),
        "sharp_turn_mae": float(np.mean(np.abs(error[sharp]))) if np.any(sharp) else None,
        "prediction_abs_p95": float(np.percentile(np.abs(predicted), 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--weights", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    scans = np.load(args.sequence / "scans.npy")
    steers = np.load(args.sequence / "steers.npy")
    report = {
        str(path): metrics(load_model(path, scans.shape[1]), scans, steers, args.batch_size)
        for path in args.weights
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
