from types import SimpleNamespace

from context.context_manager import AWSIMContextManager


def test_section_wrap_is_counted_as_progress():
    manager = AWSIMContextManager(
        extract_map={
            "awsim_status": {
                "section": "awsim_section",
                "lap_count": "awsim_lap_count",
            }
        }
    )
    node = SimpleNamespace(awsim_status={"section": 5, "lap_count": 0})
    manager.update(node=node, step_count=1, agent_action=None, sim_action=None)

    node.awsim_status = {"section": 0, "lap_count": 1}
    context = manager.update(node=node, step_count=2, agent_action=None, sim_action=None)

    assert context.section_changed is True
    assert context.lap_completed is True
