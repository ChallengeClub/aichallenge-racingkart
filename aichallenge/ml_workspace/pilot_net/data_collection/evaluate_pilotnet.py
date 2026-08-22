#!/usr/bin/env python3
"""Evaluate a PilotNet baseline repeatedly in AWSIM with zero SAC residual."""

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--rl-source",
        type=Path,
        default=Path("/aichallenge/ml_workspace/reinforcement_learning/src"),
    )
    return parser.parse_args()


def create_env(config_path: Path, rl_source: Path):
    sys.path.insert(0, str(rl_source))

    from config.load_config import load_config
    from environment.awsim_env import AWSIMEnv
    from select_parts import (
        select_action_adapter,
        select_context_manager,
        select_observation_builder,
        select_reward_function,
        select_termination_function,
    )

    cfg = load_config(str(config_path))
    action_cfg = dict(cfg["action_adapter"])
    if float(action_cfg.get("steering_residual_scale", 0.0)) != 0.0:
        raise ValueError("steering_residual_scale must be 0 for baseline evaluation")
    if float(action_cfg.get("acceleration_residual_scale", 0.0)) != 0.0:
        raise ValueError("acceleration_residual_scale must be 0 for baseline evaluation")

    return AWSIMEnv(
        context_manager=select_context_manager(cfg["context_manager"]),
        action_adapter=select_action_adapter(action_cfg),
        observation_builder=select_observation_builder(cfg["observation_builder"]),
        reward_function=select_reward_function(cfg["reward"]),
        termination_function=select_termination_function(cfg["termination"]),
    )


def evaluate_episode(env, episode_index: int, max_steps: int):
    import numpy as np

    _, reset_info = env.reset()
    speeds = []
    total_reward = 0.0
    max_lap = int(reset_info.get("lap_count", 0))
    max_section = int(reset_info.get("section", 0))
    first_lap_step = None
    last_info = reset_info
    terminated = False

    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    for step in range(1, max_steps + 1):
        _, reward, terminated, _, info = env.step(zero_action)
        last_info = info
        speed = float(info["speed"])
        lap = int(info["lap_count"])
        section = int(info["section"])
        speeds.append(speed)
        total_reward += float(reward)
        max_lap = max(max_lap, lap)
        max_section = max(max_section, section)
        if first_lap_step is None and lap >= 2:
            first_lap_step = step
        if terminated:
            break

    completed_laps = max(0, max_lap - 1)
    result = {
        "episode": episode_index,
        "steps": len(speeds),
        "terminated": terminated,
        "reached_step_limit": not terminated and len(speeds) >= max_steps,
        "completed_at_least_one_lap": completed_laps >= 1,
        "completed_laps": completed_laps,
        "first_lap_step": first_lap_step,
        "max_lap": max_lap,
        "max_section": max_section,
        "final_lap": int(last_info.get("lap_count", 0)),
        "final_section": int(last_info.get("section", 0)),
        "mean_speed_mps": statistics.fmean(speeds) if speeds else 0.0,
        "max_speed_mps": max(speeds, default=0.0),
        "final_speed_mps": float(last_info.get("speed", 0.0)),
        "session_time_s": float(last_info.get("session_time", 0.0)),
        "reported_lap_time_s": float(last_info.get("lap_time", 0.0)),
        "total_reward": total_reward,
    }
    print("EPISODE_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    args = parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        raise ValueError("episodes and max-steps must be positive")

    env = create_env(args.config, args.rl_source)
    results = []
    try:
        for episode_index in range(1, args.episodes + 1):
            results.append(evaluate_episode(env, episode_index, args.max_steps))
    finally:
        env.close()

    lap_successes = sum(r["completed_at_least_one_lap"] for r in results)
    uninterrupted = sum(not r["terminated"] for r in results)
    first_lap_steps = [r["first_lap_step"] for r in results if r["first_lap_step"]]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "episodes": args.episodes,
        "max_steps_per_episode": args.max_steps,
        "lap_successes": lap_successes,
        "lap_success_rate": lap_successes / args.episodes,
        "uninterrupted_episodes": uninterrupted,
        "uninterrupted_rate": uninterrupted / args.episodes,
        "mean_first_lap_step": (
            statistics.fmean(first_lap_steps) if first_lap_steps else None
        ),
        "mean_speed_mps": statistics.fmean(r["mean_speed_mps"] for r in results),
        "max_speed_mps": max(r["max_speed_mps"] for r in results),
        "episodes_detail": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("EVALUATION_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
