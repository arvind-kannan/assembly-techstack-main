"""The `straight_line_checked` motion planner, verified in pybullet: the Comau NJ290 arm of the cell (its URDF
collision shapes) follows each share and transit in a straight line, placed by IK, and any contact with a body
of the scene is an infeasible operation (ADR-0008 of the techstack: never a silent fallback)."""
from pathlib import Path

import pytest

p = pytest.importorskip("pybullet")

from cell_simulation import CellSimulationService
from sim_server_protocol import InvalidRequest
from straight_line_planner import Arm, register
from test_cell_simulation import CELL, TRAVEL_TIMES, op, request, share

COMAU = str(Path(__file__).resolve().parents[1] / "robot_descriptions" / "comau_nj290_robotNC.urdf")
TCP = "tcp_frame"
DOWN = [3.141592653589793, 0.0, 0.0]  # tool pointing down: x, y, z, rx, ry, rz poses below use it
HOME = [0.0, 0.0, -1.8, 0.0, -1.4, 0.0]  # joints q1..q6: TCP at about (2.13, 0, 2.42) m, tool down and tilted
FREE_PATH = [[1.8, 0.0, 2.2, *DOWN], [1.8, 0.0, 2.0, *DOWN]]


def link_index(body, name, client):
    return next(i for i in range(p.getNumJoints(body, physicsClientId=client))
                if p.getJointInfo(body, i, physicsClientId=client)[12].decode() == name)


def comau(client, position=(0, 0, 0), yaw=0.0):
    """A Comau NJ290 arm in its home configuration."""
    body = p.loadURDF(COMAU, position, p.getQuaternionFromEuler([0, 0, yaw]), useFixedBase=True, physicsClientId=client)
    movable = [j for j in range(p.getNumJoints(body, physicsClientId=client))
               if p.getJointInfo(body, j, physicsClientId=client)[2] != p.JOINT_FIXED]
    for joint, value in zip(movable, HOME):
        p.resetJointState(body, joint, value, physicsClientId=client)
    return Arm(body, link_index(body, TCP, client))


@pytest.fixture
def scene():
    """A pybullet DIRECT scene with robot_1 at the origin; returns (service, client, arms)."""
    client = p.connect(p.DIRECT)
    arms = {"robot_1": comau(client)}
    service = CellSimulationService()
    register(service, arms, client)
    yield service, client, arms
    p.disconnect(client)


def box(client, centre, half):
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=client)
    return p.createMultiBody(0, shape, basePosition=centre, physicsClientId=client)


def unscrewing(path, robot="robot_1", op_id="op_1", parameters=None):
    return op(op_id, "unscrewing", "screw", ("base", "lid"), share(robot, "T6", path), parameters=parameters)


def test_the_planner_is_configured_on_the_server_by_name(scene):
    service, _, _ = scene
    answer = service.handle({"request": "configure_planner", "planner": "straight_line_checked"})
    assert answer == {"ok": True, "planner": "straight_line_checked"}


def test_a_free_straight_share_is_feasible_and_takes_its_length_over_the_feed(scene):
    service, _, _ = scene
    service.handle({"request": "configure_planner", "planner": "straight_line_checked", "options": {"feed": 0.1}})
    report = service.handle(request(unscrewing(FREE_PATH)))
    assert report["infeasible"] is None, report["infeasible"]
    share_segment = next(s for s in report["timeline"] if s.get("operation_id") == "op_1")
    assert share_segment["end"] - share_segment["start"] == pytest.approx(2.0)  # 0.2 m at 0.1 m/s


def test_a_share_through_an_obstacle_is_an_infeasible_operation_with_reason_collision(scene):
    service, client, _ = scene
    box(client, [1.8, 0.0, 2.1], [0.05, 0.05, 0.02])  # across the middle of the path
    service.handle({"request": "configure_planner", "planner": "straight_line_checked"})
    report = service.handle(request(unscrewing(FREE_PATH)))
    assert report["feasible_operations"] == []
    infeasible = report["infeasible"]
    assert (infeasible["operation_id"], infeasible["robot"], infeasible["reason_code"]) == ("op_1", "robot_1", "collision")
    assert "op_1" in infeasible["text"] and "touches" in infeasible["text"]


def test_check_feasibility_reports_the_same_collision(scene):
    service, client, _ = scene
    box(client, [1.8, 0.0, 2.1], [0.05, 0.05, 0.02])
    service.handle({"request": "configure_planner", "planner": "straight_line_checked"})
    group = request(unscrewing(FREE_PATH))
    answer = service.handle({"request": "check_feasibility", "concurrent_group": group["operation_plan"]["operations"],
                             "snapshot": group["start_snapshot"], "cell": CELL,
                             "travel_times": TRAVEL_TIMES})
    assert answer["feasible"] is False and answer["infeasible"]["reason_code"] == "collision"


def test_a_transit_through_an_obstacle_is_infeasible_before_the_share(scene):
    service, client, _ = scene
    box(client, [1.95, 0.0, 2.3], [0.05, 0.05, 0.02])  # between home (2.13, 0, 2.42) and the path start (1.8, 0, 2.2)
    service.handle({"request": "configure_planner", "planner": "straight_line_checked"})
    infeasible = service.handle(request(unscrewing(FREE_PATH)))["infeasible"]
    assert infeasible["reason_code"] == "collision" and infeasible["text"].startswith("transit to op_1")


def test_a_pose_the_arm_cannot_reach_is_out_of_reach(scene):
    service, _, _ = scene
    service.handle({"request": "configure_planner", "planner": "straight_line_checked"})
    far = [[6.0, 0.0, 2.2, *DOWN], [6.0, 0.0, 2.0, *DOWN]]
    infeasible = service.handle(request(unscrewing(far)))["infeasible"]
    assert infeasible["reason_code"] == "out_of_reach"


def test_planning_leaves_the_scene_as_it_was(scene):
    service, client, arms = scene
    body = arms["robot_1"].body
    before = [p.getJointState(body, j, physicsClientId=client)[0] for j in range(p.getNumJoints(body, physicsClientId=client))]
    service.handle({"request": "configure_planner", "planner": "straight_line_checked"})
    service.handle(request(unscrewing(FREE_PATH)))
    after = [p.getJointState(body, j, physicsClientId=client)[0] for j in range(p.getNumJoints(body, physicsClientId=client))]
    assert after == pytest.approx(before)


def test_poses_relative_to_the_reference_part_are_placed_by_its_reference_pose(scene):
    service, _, _ = scene
    base_at = [1.8, 0.0, 2.0, 0.0, 0.0, 0.0]  # the base part's frame in the cell
    service.handle({"request": "configure_planner", "planner": "straight_line_checked",
                    "options": {"reference_poses": {"base": base_at}}})
    relative = [[0.0, 0.0, 0.2, *DOWN], [0.0, 0.0, 0.0, *DOWN]]
    report = service.handle(request(unscrewing(relative, parameters={"reference_part": "base"}), target="cell"))
    assert report["infeasible"] is None, report["infeasible"]
    nc = report["nc_programs"]["1"]["text"]
    assert "X1800.000 Y0.000 Z2200.000" in nc and "X1800.000 Y0.000 Z2000.000" in nc


def test_an_operation_relative_to_a_part_without_reference_pose_is_an_invalid_request(scene):
    service, _, _ = scene
    service.handle({"request": "configure_planner", "planner": "straight_line_checked"})
    with pytest.raises(InvalidRequest, match="reference_poses"):
        service.handle(request(unscrewing(FREE_PATH, parameters={"reference_part": "base"})))


def test_a_carried_tool_moves_with_the_arm_and_is_checked_too(scene):
    _, client, arms = scene
    arm = arms["robot_1"]
    tool = box(client, [5.0, 5.0, 0.0], [0.02, 0.3, 0.02])  # a bar, lying anywhere until the arm carries it
    mounted = ((0.0, 0.45, 0.0), (0.0, 0.0, 0.0, 1.0))  # its centre 0.45 m to the side of the TCP link
    service = CellSimulationService()
    register(service, {"robot_1": Arm(arm.body, arm.tcp_link, tools=lambda: [(tool, mounted)])}, client)
    service.handle({"request": "configure_planner", "planner": "straight_line_checked"})
    assert service.handle(request(unscrewing(FREE_PATH)))["infeasible"] is None
    box(client, [1.8, -0.65, 2.1], [0.05, 0.05, 0.02])  # the tool turns with the TCP (home yaw is pi off DOWN)
    infeasible = service.handle(request(unscrewing(FREE_PATH)))["infeasible"]
    assert infeasible["reason_code"] == "collision" and infeasible["text"].endswith(")") and "robot_1 at" in infeasible["text"]
    assert p.getBodyInfo(tool, physicsClientId=client)[1].decode() in infeasible["text"].split(" touches ")[0].split(": ")[-1]


def two_robot_scene(client, gap):
    """robot_1 at the origin and robot_2 facing it at x = `gap`; both reach towards each other."""
    arms = {"robot_1": comau(client), "robot_2": comau(client, (gap, 0, 0), 3.141592653589793)}
    service = CellSimulationService()
    register(service, arms, client)
    service.handle({"request": "configure_planner", "planner": "straight_line_checked"})
    return service


def both_unscrewing(path_2):
    return (op("op_1", "unscrewing", "screw", ("base", "lid"), share("robot_1", "T6", FREE_PATH)),
            op("op_2", "unscrewing", "lid", ("base", "lid"), share("robot_2", "T6", path_2)))


def test_two_robots_whose_paths_come_too_close_get_a_motion_level_sync_point():
    client = p.connect(p.DIRECT)
    try:
        service = two_robot_scene(client, 3.8)
        mirrored = [[2.0, 0.0, 2.2, *DOWN], [2.0, 0.0, 2.0, *DOWN]]  # robot_2's TCP 1.8 m in front of it, 0.2 m from robot_1's
        report = service.handle(request(*both_unscrewing(mirrored)))
        assert report["infeasible"] is None, report["infeasible"]
        assert [s["layer"] for s in report["sync_points_added"]] and all(
            s["layer"] == "motion" for s in report["sync_points_added"])
    finally:
        p.disconnect(client)


def test_two_robots_far_apart_run_at_the_same_time_without_sync_points():
    client = p.connect(p.DIRECT)
    try:
        service = two_robot_scene(client, 8.0)
        report = service.handle(request(*both_unscrewing([[6.2, 0.0, 2.2, *DOWN], [6.2, 0.0, 2.0, *DOWN]])))
        assert report["infeasible"] is None, report["infeasible"]
        assert report["sync_points_added"] == []
    finally:
        p.disconnect(client)


def test_share_poses_command_the_tool_tip_when_the_tcp_is_offset_from_the_arm_link(scene):
    service, client, arms = scene
    arm = arms["robot_1"]
    path = [[2.0, 0.0, 2.0, *DOWN], [2.0, 0.0, 1.8, *DOWN]]  # too low for the arm link itself (q5 limit)
    service.handle({"request": "configure_planner", "planner": "straight_line_checked"})
    assert service.handle(request(unscrewing(path)))["infeasible"]["reason_code"] == "out_of_reach"
    tip = ((0.0, 0.0, 0.3), (0.0, 0.0, 0.0, 1.0))  # a 0.3 m tool along the link's z axis (down, for DOWN poses)
    with_tool = CellSimulationService()
    register(with_tool, {"robot_1": Arm(arm.body, arm.tcp_link, tcp=lambda: tip)}, client)
    with_tool.handle({"request": "configure_planner", "planner": "straight_line_checked"})
    report = with_tool.handle(request(unscrewing(path), target="cell"))
    assert report["infeasible"] is None, report["infeasible"]
    assert "X2000.000 Y0.000 Z1800.000" in report["nc_programs"]["1"]["text"]  # the NC commands the tip, as given


@pytest.mark.parametrize("options", [
    {"conflict_samples": 2.5}, {"feed": 0}, {"clearance": -0.1}, {"conflict_margin": "far"},
    {"orientation": [0.0, 0.0]}, {"reference_poses": {"base": [0.0, 0.0]}}, {"speed": 1.0},
])
def test_bad_planner_options_are_refused_when_the_planner_is_configured(scene, options):
    service, _, _ = scene
    with pytest.raises(InvalidRequest):
        service.handle({"request": "configure_planner", "planner": "straight_line_checked", "options": options})
