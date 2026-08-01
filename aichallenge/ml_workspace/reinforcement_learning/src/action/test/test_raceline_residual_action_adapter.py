import csv

import numpy as np

from action.raceline_residual_action_adapter import RacelineResidualActionAdapter
from context.context_types import EnvState, StepContext


def _context(x=0.0, y=0.0, speed=0.0):
    return StepContext(
        env_state=EnvState(
            {
                "kinematic_pose_x_m": x,
                "kinematic_pose_y_m": y,
                "kinematic_orientation_x": 0.0,
                "kinematic_orientation_y": 0.0,
                "kinematic_orientation_z": 0.0,
                "kinematic_orientation_w": 1.0,
                "vehicle_speed_mps": speed,
            }
        ),
        prev_env_state=None,
        agent_action=None,
        sim_action=None,
        step_count=0,
        collision_count=0,
        section_changed=False,
        lap_completed=False,
        collision=False,
    )


def test_baseline_steers_toward_raceline_and_controls_speed(tmp_path):
    path = tmp_path / "raceline.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["x", "y"])
        writer.writeheader()
        writer.writerow({"x": 0, "y": 0})
        writer.writerow({"x": 5, "y": 2})
        writer.writerow({"x": 10, "y": 2})

    adapter = RacelineResidualActionAdapter(str(path), lookahead_index=1, target_speed_mps=4.0)
    command = adapter.adapt(np.zeros(2, dtype=np.float32), _context())
    at_speed = adapter.adapt(np.zeros(2, dtype=np.float32), _context(speed=4.0))

    assert command["steering"] > 0.0
    assert command["acceleration"] == 1.0
    assert at_speed["acceleration"] == 0.0
