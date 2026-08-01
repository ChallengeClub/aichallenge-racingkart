import numpy as np

from context.context_types import EnvState, StepContext
from reward.default_reward import DefaultAWSIMReward


def _context(**overrides):
    values = {
        "env_state": EnvState({"vehicle_speed_mps": 2.0}),
        "prev_env_state": EnvState({"vehicle_speed_mps": 1.5}),
        "agent_action": np.array([0.2, 0.5], dtype=np.float32),
        "sim_action": np.array([0.2, 0.5], dtype=np.float32),
        "step_count": 30,
        "collision_count": 0,
        "section_changed": False,
        "lap_completed": False,
        "collision": False,
        "info": {"previous_sim_action": np.array([0.1, 0.4], dtype=np.float32)},
    }
    values.update(overrides)
    return StepContext(**values)


def test_progress_bonuses_make_lap_progress_preferable():
    reward_fn = DefaultAWSIMReward(
        speed_reward_scale=1.0,
        step_time_penalty=0.05,
        collision_penalty=100.0,
        section_progress_bonus=20.0,
        lap_completion_bonus=250.0,
    )

    normal, _ = reward_fn.compute(_context())
    progressed, _ = reward_fn.compute(_context(section_changed=True))
    completed, _ = reward_fn.compute(_context(section_changed=True, lap_completed=True))

    assert progressed - normal == 20.0
    assert completed - progressed == 250.0


def test_collision_dominates_speed_and_progress_reward():
    reward_fn = DefaultAWSIMReward(
        speed_reward_scale=1.0,
        step_time_penalty=0.05,
        collision_penalty=100.0,
        section_progress_bonus=20.0,
    )

    reward, context = reward_fn.compute(_context(section_changed=True, collision=True))

    assert reward < 0.0
    assert context.info["reward_breakdown"]["collision_penalty"] == -100.0


def test_jerky_controls_are_penalized():
    reward_fn = DefaultAWSIMReward(
        speed_reward_scale=0.0,
        step_time_penalty=0.0,
        collision_penalty=0.0,
        steering_penalty_scale=0.05,
        steering_change_penalty_scale=0.15,
        acceleration_change_penalty_scale=0.02,
    )

    smooth, _ = reward_fn.compute(
        _context(
            sim_action=np.array([0.1, 0.4], dtype=np.float32),
            info={"previous_sim_action": np.array([0.1, 0.4], dtype=np.float32)},
        )
    )
    jerky, _ = reward_fn.compute(
        _context(
            sim_action=np.array([0.8, 1.0], dtype=np.float32),
            info={"previous_sim_action": np.array([-0.8, 0.0], dtype=np.float32)},
        )
    )

    assert jerky < smooth
