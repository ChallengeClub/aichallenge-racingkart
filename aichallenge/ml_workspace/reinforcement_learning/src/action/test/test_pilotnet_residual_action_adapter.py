import numpy as np

from action.pilotnet_residual_action_adapter import PilotNetResidualActionAdapter
from context.context_types import EnvState, StepContext


class FakePilotCore:
    def process(self, image):
        assert image.shape == (8, 8, 3)
        return 0.6, -0.1


def _context(speed=1.0):
    return StepContext(
        env_state=EnvState(
            {
                "camera_image": np.zeros((8, 8, 3), dtype=np.uint8),
                "vehicle_speed_mps": speed,
            }
        ),
        prev_env_state=None,
        agent_action=None,
        sim_action=None,
        step_count=1,
        collision_count=0,
        section_changed=False,
        lap_completed=False,
        collision=False,
    )


def test_pilotnet_baseline_and_sac_residual_are_combined():
    adapter = PilotNetResidualActionAdapter(
        package_path="unused",
        checkpoint_path="unused",
        steering_residual_scale=0.02,
        acceleration_residual_scale=0.02,
        pilot_core=FakePilotCore(),
    )

    command = adapter.adapt(np.array([1.0, -1.0], dtype=np.float32), _context())

    assert np.isclose(command["steering"], -0.08)
    assert np.isclose(command["acceleration"], 0.58)


def test_allowed_wheel_speed_caps_acceleration():
    adapter = PilotNetResidualActionAdapter(
        package_path="unused",
        checkpoint_path="unused",
        max_speed_mps=2.0,
        pilot_core=FakePilotCore(),
    )

    command = adapter.adapt(np.zeros(2, dtype=np.float32), _context(speed=2.1))

    assert command["acceleration"] == 0.0
