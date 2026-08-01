from __future__ import annotations

from context.context_types import StepContext
from reward.interfaces import RewardFunction


class DefaultAWSIMReward(RewardFunction):
    def __init__(
        self,
        speed_reward_scale: float = 0.1,
        step_time_penalty: float = 0.05,
        collision_penalty: float = 50.0,
        section_progress_bonus: float = 0.0,
        lap_completion_bonus: float = 0.0,
        steering_penalty_scale: float = 0.0,
        steering_change_penalty_scale: float = 0.0,
        acceleration_change_penalty_scale: float = 0.0,
        raceline_progress_reward_scale: float = 0.0,
        cross_track_penalty_scale: float = 0.0,
    ) -> None:
        self.speed_reward_scale = speed_reward_scale
        self.step_time_penalty = step_time_penalty
        self.collision_penalty = collision_penalty
        self.section_progress_bonus = section_progress_bonus
        self.lap_completion_bonus = lap_completion_bonus
        self.steering_penalty_scale = steering_penalty_scale
        self.steering_change_penalty_scale = steering_change_penalty_scale
        self.acceleration_change_penalty_scale = acceleration_change_penalty_scale
        self.raceline_progress_reward_scale = raceline_progress_reward_scale
        self.cross_track_penalty_scale = cross_track_penalty_scale

    def compute(self, context: StepContext,) -> tuple[float, StepContext]:
        speed = float(context.env_state.get_value("vehicle_speed_mps", 0.0))
        speed_reward = self.speed_reward_scale * max(0.0, speed)
        collision_penalty = self.collision_penalty if context.collision else 0.0
        section_bonus = self.section_progress_bonus if context.section_changed else 0.0
        lap_bonus = self.lap_completion_bonus if context.lap_completed else 0.0

        steering_penalty = 0.0
        steering_change_penalty = 0.0
        acceleration_change_penalty = 0.0
        raceline_valid = bool(context.info.get("raceline_valid", False))
        progress_reward = 0.0
        cross_track_penalty = 0.0
        if raceline_valid:
            progress_reward = self.raceline_progress_reward_scale * max(
                0.0, float(context.info.get("raceline_progress_delta", 0.0))
            )
            cross_track_penalty = self.cross_track_penalty_scale * float(
                context.info.get("cross_track_error_m", 0.0)
            )
        if context.sim_action is not None:
            steering = float(context.sim_action[0])
            steering_penalty = self.steering_penalty_scale * abs(steering)
            previous_action = context.info.get("previous_sim_action")
            if previous_action is not None:
                steering_change_penalty = self.steering_change_penalty_scale * abs(
                    steering - float(previous_action[0])
                )
                acceleration_change_penalty = self.acceleration_change_penalty_scale * abs(
                    float(context.sim_action[1]) - float(previous_action[1])
                )

        reward = (
            speed_reward
            + section_bonus
            + lap_bonus
            + progress_reward
            - collision_penalty
            - self.step_time_penalty
            - steering_penalty
            - steering_change_penalty
            - acceleration_change_penalty
            - cross_track_penalty
        )

        context.info["reward_breakdown"] = {
            "speed_reward": speed_reward,
            "time_penalty": -self.step_time_penalty,
            "collision_penalty": -collision_penalty,
            "section_bonus": section_bonus,
            "lap_bonus": lap_bonus,
            "steering_penalty": -steering_penalty,
            "steering_change_penalty": -steering_change_penalty,
            "acceleration_change_penalty": -acceleration_change_penalty,
            "raceline_progress_reward": progress_reward,
            "cross_track_penalty": -cross_track_penalty,
            "total_reward": reward,
        }
        return reward, context
