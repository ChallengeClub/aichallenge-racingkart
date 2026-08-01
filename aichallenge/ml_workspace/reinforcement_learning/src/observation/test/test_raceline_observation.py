import csv
from types import SimpleNamespace

import numpy as np

from context.context_manager import AWSIMContextManager
from observation.default_observation import RacelineImageSpeedObservationBuilder


def test_raceline_observation_reports_forward_progress(tmp_path):
    raceline = tmp_path / "raceline.csv"
    with raceline.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["x", "y"])
        writer.writeheader()
        for x in range(20):
            writer.writerow({"x": x, "y": 0})

    extract_map = {
        "pose": {
            "x": "kinematic_pose_x_m",
            "y": "kinematic_pose_y_m",
            "qx": "kinematic_orientation_x",
            "qy": "kinematic_orientation_y",
            "qz": "kinematic_orientation_z",
            "qw": "kinematic_orientation_w",
        }
    }
    manager = AWSIMContextManager(extract_map=extract_map)
    builder = RacelineImageSpeedObservationBuilder(str(raceline), [0, 2], 10.0)
    node = SimpleNamespace(pose={"x": 1.0, "y": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0})

    context = manager.update(node=node, step_count=1, agent_action=None, sim_action=None)
    observation, context = builder.build(context)
    node.pose["x"] = 2.0
    context = manager.update(node=node, step_count=2, agent_action=None, sim_action=None)
    observation, context = builder.build(context)

    assert observation["raceline"].shape == (4,)
    assert observation["raceline"].dtype == np.float32
    assert context.info["raceline_progress_delta"] == 1
    assert context.info["cross_track_error_m"] == 1.0
    assert context.info["raceline_valid"] is True


def test_missing_pose_does_not_create_huge_cross_track_error(tmp_path):
    raceline = tmp_path / "raceline.csv"
    with raceline.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["x", "y"])
        writer.writeheader()
        writer.writerow({"x": 89630, "y": 43130})
        writer.writerow({"x": 89631, "y": 43130})

    manager = AWSIMContextManager(extract_map={})
    builder = RacelineImageSpeedObservationBuilder(str(raceline), [0, 1], 10.0)
    context = manager.update(
        node=SimpleNamespace(), step_count=1, agent_action=None, sim_action=None
    )

    observation, context = builder.build(context)

    assert np.all(observation["raceline"] == 0.0)
    assert context.info["raceline_valid"] is False
    assert context.info["cross_track_error_m"] == 0.0
