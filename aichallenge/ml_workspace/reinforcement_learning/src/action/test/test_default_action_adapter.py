import numpy as np

from action.default_action_adapter import DefaultAWSIMActionAdapter


def test_symmetric_policy_acceleration_maps_to_simulator_throttle():
    adapter = DefaultAWSIMActionAdapter(
        min_accel=-1.0,
        max_accel=1.0,
        sim_min_accel=0.0,
        sim_max_accel=1.0,
    )

    assert adapter.adapt(np.array([0.0, -1.0], dtype=np.float32))["acceleration"] == 0.0
    assert adapter.adapt(np.array([0.0, 0.0], dtype=np.float32))["acceleration"] == 0.5
    assert adapter.adapt(np.array([0.0, 1.0], dtype=np.float32))["acceleration"] == 1.0
