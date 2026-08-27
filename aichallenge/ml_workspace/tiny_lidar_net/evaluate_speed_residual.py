#!/usr/bin/env python3
"""Evaluate a learned speed residual on an entirely independent run."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_PACKAGE = (
    Path(__file__).resolve().parents[2]
    / "workspace/src/aichallenge_submit/tiny_lidar_net_controller"
)
sys.path.insert(0, str(REPO_PACKAGE))
from tiny_lidar_net_controller.learned_speed_residual import (  # noqa: E402
    LearnedSpeedResidual,
)


def summarize(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = target - prediction
    meaningful = np.abs(target) > 0.01
    return {
        "samples": int(len(target)),
        "mae_rad": float(np.mean(np.abs(error))) if len(error) else 0.0,
        "rmse_rad": float(np.sqrt(np.mean(error**2))) if len(error) else 0.0,
        "prediction_p95_rad": (
            float(np.percentile(np.abs(prediction), 95)) if len(prediction) else 0.0
        ),
        "direction_accuracy": (
            float(np.mean(np.sign(target[meaningful]) == np.sign(prediction[meaningful])))
            if np.any(meaningful)
            else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scans = np.load(args.dataset / "scans.npy")
    student = np.load(args.dataset / "student_steers.npy")
    teacher = np.load(args.dataset / "teacher_steers.npy")
    speeds = np.abs(np.load(args.dataset / "speeds.npy"))
    model = LearnedSpeedResidual(str(args.weights))
    corrections = np.asarray(
        [
            model.compute(scan, steer, speed) - steer
            for scan, steer, speed in zip(scans, student, speeds)
        ]
    )
    gates = np.asarray([model.speed_gate(speed) for speed in speeds])
    raw_target = np.clip(
        teacher - student, -model.maximum_correction, model.maximum_correction
    )
    target = gates * raw_target

    report = {}
    for label, mask in {
        "active": speeds > model.activation_speed_mps,
        "over_2_5_mps": speeds > 2.5,
        "over_3_0_mps": speeds > 3.0,
    }.items():
        report[label] = {
            "zero_baseline": summarize(target[mask], np.zeros(np.sum(mask))),
            "model": summarize(target[mask], corrections[mask]),
        }
        baseline = report[label]["zero_baseline"]["mae_rad"]
        learned = report[label]["model"]["mae_rad"]
        report[label]["mae_improvement_percent"] = (
            100.0 * (baseline - learned) / baseline if baseline > 0.0 else 0.0
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
