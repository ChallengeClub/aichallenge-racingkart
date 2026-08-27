#!/usr/bin/env python3
"""Train and time-split-evaluate a compact speed-conditioned residual model."""

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
    extract_residual_features,
)


def speed_gate(speeds: np.ndarray, activation: float, full: float) -> np.ndarray:
    return np.clip((np.abs(speeds) - activation) / (full - activation), 0.0, 1.0)


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = target - prediction
    nonzero = np.abs(target) > 0.01
    return {
        "samples": int(len(target)),
        "mae_rad": float(np.mean(np.abs(error))) if len(error) else 0.0,
        "rmse_rad": float(np.sqrt(np.mean(error**2))) if len(error) else 0.0,
        "p95_abs_prediction_rad": (
            float(np.percentile(np.abs(prediction), 95)) if len(prediction) else 0.0
        ),
        "direction_accuracy": (
            float(np.mean(np.sign(target[nonzero]) == np.sign(prediction[nonzero])))
            if np.any(nonzero)
            else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--activation-speed", type=float, default=2.3)
    parser.add_argument("--full-effect-speed", type=float, default=3.3)
    parser.add_argument("--maximum-correction", type=float, default=0.12)
    parser.add_argument("--minimum-training-speed", type=float, default=2.0)
    parser.add_argument("--sector-count", type=int, default=18)
    args = parser.parse_args()

    scans = np.load(args.dataset / "scans.npy")
    student = np.load(args.dataset / "student_steers.npy")
    teacher = np.load(args.dataset / "teacher_steers.npy")
    speeds = np.load(args.dataset / "speeds.npy")
    features = np.asarray(
        [
            extract_residual_features(
                scan, steer, speed, sector_count=args.sector_count
            )
            for scan, steer, speed in zip(scans, student, speeds)
        ]
    )
    raw_target = np.clip(
        teacher - student, -args.maximum_correction, args.maximum_correction
    )
    gates = speed_gate(speeds, args.activation_speed, args.full_effect_speed)
    safe_target = gates * raw_target

    count = len(scans)
    train_end = int(count * 0.7)
    validation_start = int(count * 0.8)
    train_mask = (np.arange(count) < train_end) & (
        np.abs(speeds) >= args.minimum_training_speed
    )
    validation_mask = (np.arange(count) >= validation_start) & (
        np.abs(speeds) >= args.minimum_training_speed
    )
    if np.sum(train_mask) < features.shape[1] + 1:
        raise RuntimeError("not enough active training samples")
    if np.sum(validation_mask) == 0:
        raise RuntimeError("no active validation samples in held-out time segment")

    feature_mean = np.mean(features[train_mask], axis=0)
    feature_scale = np.std(features[train_mask], axis=0)
    feature_scale[feature_scale < 1e-6] = 1.0
    normalized = (features - feature_mean) / feature_scale
    train_x = np.column_stack(
        (normalized[train_mask], np.ones(np.sum(train_mask), dtype=np.float64))
    )
    train_y = raw_target[train_mask]

    candidates = []
    best = None
    for regularization in (0.01, 0.1, 1.0, 10.0, 100.0):
        penalty = np.eye(train_x.shape[1]) * regularization
        penalty[-1, -1] = 0.0
        weights = np.linalg.solve(train_x.T @ train_x + penalty, train_x.T @ train_y)
        raw_prediction = normalized @ weights[:-1] + weights[-1]
        correction = gates * np.clip(
            raw_prediction, -args.maximum_correction, args.maximum_correction
        )
        score = metrics(safe_target[validation_mask], correction[validation_mask])
        candidates.append({"regularization": regularization, **score})
        if best is None or score["mae_rad"] < best[0]:
            best = (score["mae_rad"], regularization, weights, correction)

    _, regularization, weights, correction = best
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coefficients=weights[:-1],
        intercept=np.asarray(weights[-1]),
        activation_speed_mps=np.asarray(args.activation_speed),
        full_effect_speed_mps=np.asarray(args.full_effect_speed),
        maximum_correction=np.asarray(args.maximum_correction),
        sector_count=np.asarray(args.sector_count),
        maximum_range_m=np.asarray(30.0),
    )

    high_validation = validation_mask & (np.abs(speeds) > 2.5)
    report = {
        "total_samples": count,
        "training_samples": int(np.sum(train_mask)),
        "validation_samples": int(np.sum(validation_mask)),
        "validation_samples_over_2_5_mps": int(np.sum(high_validation)),
        "chosen_regularization": regularization,
        "zero_baseline_validation": metrics(
            safe_target[validation_mask], np.zeros(np.sum(validation_mask))
        ),
        "model_validation": metrics(
            safe_target[validation_mask], correction[validation_mask]
        ),
        "zero_baseline_high_speed": metrics(
            safe_target[high_validation], np.zeros(np.sum(high_validation))
        ),
        "model_high_speed": metrics(
            safe_target[high_validation], correction[high_validation]
        ),
        "regularization_candidates": candidates,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
