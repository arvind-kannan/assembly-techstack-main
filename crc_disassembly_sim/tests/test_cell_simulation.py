"""The cell simulation for operation plans (simulate, check_feasibility, configure_planner), without pybullet."""
import copy
import socket
import threading
import pickle

import pytest

from cell_simulation import (
    INFEASIBLE_CODES, CellSimulationService, MotionFailure, MotionPlanner, ReasonCode, Segment, ShareMotion,
)
from sim_server_protocol import InvalidRequest, serve_connection

XML = """<assembly name="lid_screwed_on_base">
	<part name="base" mesh_path="base.stl" density="8000.0" material="Generic" />
	<part name="lid" mesh_path="lid.stl" density="8000.0" material="Generic" />
	<part name="screw" mesh_path="screw.stl" density="8000.0" material="Generic" />
	<connection type="screw" part_name1="screw" part_name2="lid" screw_head="T6" />
	<connection type="generic" part_name1="base" part_name2="lid" />
</assembly>"""
CELL = {
    "robots": [
        {"id": "robot_1", "nc_channel": 1, "tools": ["H5", "T6"],
         "capabilities": ["gripping_and_removing", "unscrewing"]},
        {"id": "robot_2", "nc_channel": 2, "tools": ["T6"], "capabilities": ["unscrewing", "inspecting"]},
    ],
    "tools": [],
    # the techstack's cell shape (AT 174): stations with kind, pose and supply slots; the simulation reads none of it
    "stations": [{"name": "supply", "kind": "supply", "pose": [-1.2, -2.6, 0.8, 0.0, 0.0, 0.0], "processes": [],
                  "coupling_places": [], "slots": [{"name": "lid_slot", "pose": [0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                    "part": "lid", "screw_type": None, "robots": []}]}],
    "pallets": [], "tooling": [],
}
# the cell's travel-time table, a field of the request next to the cell (techstack ADR-0012, interim until #235)
TRAVEL_TIMES = {"times": {"toolchange_time_assembly_to_storage": 10.0, "toolchange_time_at_toolstorage": 60.0,
                          "toolchange_time_storage_to_assembly": 10.0}}
SNAPSHOT = {"assembly_xml": XML, "cell_state": {"holds": {}, "mounted_tools": {"robot_1": "H5", "robot_2": None}}}
PATH = [[0.0, 0.0, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.1]]


class StraightLine(MotionPlanner):
    """A test planner: polyline length over a feed (metres per second); transits cost a fixed time.
    `fail_transit_to`: a first pose it finds no transit to; `conflicting`: pairs of segment ids that clash."""

    name = "straight_line"

    def __init__(self, options):
        self.feed = options.get("feed", 0.1)
        self.fail_transit_to = options.get("fail_transit_to")
        self.conflicting = [tuple(p) for p in options.get("conflicting", ())]
        self.calls = []

    def plan_share(self, robot, operation, share):
        self.calls.append(("share", robot["id"], operation["id"]))
        length = sum(sum((b - a) ** 2 for a, b in zip(p, q)) ** 0.5 for p, q in zip(share["path"], share["path"][1:]))
        return ShareMotion(length / self.feed, [{"G": 1, "X": p[0], "Y": p[1], "Z": p[2], "F": 1000.0} for p in share["path"]],
                           tuple(share["path"][-1]))

    def plan_transit(self, robot, start_pose, goal_pose):
        self.calls.append(("transit", robot["id"], tuple(goal_pose)))
        if self.fail_transit_to is not None and list(goal_pose) == list(self.fail_transit_to):
            return MotionFailure(ReasonCode.NO_PATH_TRANSIT, "boxed in by the fixture")
        return ShareMotion(2.0, [{"G": 0, "X": goal_pose[0], "Y": goal_pose[1], "Z": goal_pose[2]}], tuple(goal_pose))

    def conflicts(self, first, second):
        return (first.id, second.id) in self.conflicting or (second.id, first.id) in self.conflicting


@pytest.fixture
def service():
    service = CellSimulationService()
    service.register_planner("straight_line", StraightLine)
    return service


def configured(service, **options):
    service.handle({"request": "configure_planner", "planner": "straight_line", "options": options})
    return service


def share(robot="robot_1", tool="H5", path=(), duration=5.0):
    return {"robot": robot, "tool": tool, "path": [list(p) for p in path], "path_duration": duration, "skill_parameters": {}}


def op(op_id, process, target, state, *shares, parameters=None):
    return {"id": op_id, "process": process, "target": target, "shares": list(shares), "direction": None,
            "expected_product_state": list(state), "parameters": parameters or {}}


def inspection(op_id, state):
    return op(op_id, "inspecting", None, state, share("robot_1", None, duration=10.0), share("robot_2", None, duration=10.0))


def unscrew(op_id="op_1", path=()):
    return op(op_id, "unscrewing", "screw", ("base", "lid"), share("robot_1", "T6", path))


def remove_lid(op_id="op_2", path=()):
    return op(op_id, "gripping_and_removing", "lid", ("base",), share("robot_1", "H5", path))


def request(*operations, sync_points=(), target="simulation", mode="dry", **extra):
    plan = {"operations": list(operations), "sequences": {}, "sync_points": list(sync_points),
            "metrics": {"makespan": 0.0, "busy_time": {}, "tool_changes": 0, "inspections": 0}, "start_snapshot": SNAPSHOT}
    return {"request": "simulate", "operation_plan": plan, "start_snapshot": SNAPSHOT, "cell": CELL,
            "travel_times": TRAVEL_TIMES, "target": target, "execution_mode": mode, **extra}


def sync(robot, before, until, reason="order", layer="planner"):
    return {"robot": robot, "before": before, "until": until, "layer": layer, "reason": reason}


# --- planner configuration ---------------------------------------------------------------------------------

def test_the_planner_is_configured_once_by_name(service):
    assert service.handle({"request": "configure_planner", "planner": "straight_line"}) == {"ok": True, "planner": "straight_line"}


def test_an_unknown_planner_is_an_invalid_request_that_names_the_registered_ones(service):
    with pytest.raises(InvalidRequest, match="straight_line"):
        service.handle({"request": "configure_planner", "planner": "cumotion"})
    assert service.planner is None


# --- no silent fallback --------------------------------------------------------------------------------------

def test_without_a_configured_planner_an_operation_with_a_path_is_infeasible_and_never_direct_motion(service):
    report = service.handle(request(unscrew(path=PATH), inspection("inspection_1", ("base", "lid")), target="cell"))

    assert report["infeasible"]["operation_id"] == "op_1"
    assert report["infeasible"]["reason_code"] == "no_path_share" and "configure_planner" in report["infeasible"]["text"]
    assert report["feasible_operations"] == [] and report["nc_programs"] is None


def test_operations_without_a_path_need_no_planner_and_take_their_declared_time(service):
    report = service.handle(request(unscrew(), inspection("inspection_1", ("base", "lid"))))

    assert report["infeasible"] is None and report["feasible_operations"] == ["op_1", "inspection_1"]
    assert report["metrics"]["makespan"] == 5.0 + 10.0 + 80.0 - 80.0 + 80.0  # tool change H5 -> T6 first, then 5 s, then 10 s
    assert report["metrics"]["tool_changes"] == 1 and report["metrics"]["inspections"] == 1


def test_an_inspection_takes_the_tables_inspection_time_over_its_declared_duration(service):
    table = copy.deepcopy(TRAVEL_TIMES)
    table["times"]["inspection_time"] = 7.0
    report = service.handle({**request(inspection("inspection_1", ("base", "lid", "screw"))), "travel_times": table})

    spans = [t["end"] - t["start"] for t in report["timeline"] if t["operation_id"] == "inspection_1"]
    assert spans == [7.0, 7.0]  # declared 10 s per share; the table's key is inspection_time, as in the techstack


def test_a_planner_that_reports_a_non_infeasible_code_is_a_planner_bug_not_a_reason(service):
    class Broken(StraightLine):
        def plan_share(self, robot, operation, share):
            return MotionFailure(ReasonCode.CONTACT_ABORT, "wrong kind")

    service.register_planner("broken", Broken)
    service.handle({"request": "configure_planner", "planner": "broken"})

    with pytest.raises(RuntimeError, match="contact_abort"):  # the server turns it into internal_error
        service.handle(request(unscrew(path=PATH)))


# --- planned motion -----------------------------------------------------------------------------------------

def test_cell_time_comes_from_the_planned_motion_transit_included(service):
    configured(service, feed=0.1)

    report = service.handle(request(unscrew(path=PATH)))

    # tool change 80 s, transit 2 s, path 0.2 m at 0.1 m/s = 2 s
    assert report["metrics"]["makespan"] == 80.0 + 2.0 + 2.0
    kinds = [(t["kind"], t["operation_id"]) for t in report["timeline"] if t["kind"] != "tool_change"]
    assert kinds == [("transit", "op_1:transit"), ("operation", "op_1")]
    assert report["metrics"]["busy_time"]["robot_1"] == 84.0


def test_a_transit_without_a_path_stops_the_simulation_with_the_planners_reason(service):
    configured(service, fail_transit_to=PATH[0])

    report = service.handle(request(unscrew(path=PATH), remove_lid(), target="cell"))

    assert report["infeasible"]["reason_code"] == "no_path_transit" and "boxed in" in report["infeasible"]["text"]
    assert report["feasible_operations"] == [] and report["nc_programs"] is None
    assert report["infeasible"]["reason_code"] in {c.value for c in INFEASIBLE_CODES}


def test_the_simulation_stops_at_the_first_infeasible_operation(service):
    ghost = op("op_2", "gripping_and_removing", "ghost", ("base",), share("robot_1", "H5"))

    report = service.handle(request(unscrew(), ghost, remove_lid("op_3"), inspection("inspection_1", ("base",))))

    assert report["infeasible"]["operation_id"] == "op_2" and report["infeasible"]["reason_code"] == "cell_state_mismatch"
    assert report["feasible_operations"] == ["op_1"] and report["observed_snapshots"] == {}
    assert [o["operation_id"] for o in report["outcomes"]] == ["op_1"]


@pytest.mark.parametrize("bad, code", [
    (op("op_1", "milling", "lid", ("base", "screw"), share("robot_2", "T6")), "unsupported_process"),
    (op("op_1", "unscrewing", "screw", ("base", "lid"), share("robot_1", "T6"), parameters={"plan_failure": "no free direction"}),
     "no_path_share"),
])
def test_model_level_reasons_come_from_the_closed_list(service, bad, code):
    report = service.handle(request(bad))

    assert report["infeasible"]["reason_code"] == code and report["infeasible"]["text"]


def two_share_failure(**parameters):
    return op("op_1", "unscrewing", "screw", ("base", "lid"), share("robot_1", "T6"), share("robot_2", "T6"),
              parameters={"plan_failure": "no free direction", **parameters})


@pytest.mark.parametrize("parameters, robot", [({"plan_failure_robot": "robot_2"}, "robot_2"), ({}, "robot_1")])
def test_a_plan_failure_names_the_failing_shares_robot_else_the_first_share(service, parameters, robot):
    marked = two_share_failure(**parameters)

    simulated = service.handle(request(marked))["infeasible"]
    queried = query(service, marked)["infeasible"]

    assert (simulated["reason_code"], simulated["robot"]) == ("no_path_share", robot)
    assert (queried["reason_code"], queried["robot"]) == ("no_path_share", robot)


# --- observation ---------------------------------------------------------------------------------------------

def test_the_observed_snapshot_at_each_inspection_reached(service):
    report = service.handle(request(unscrew(), inspection("inspection_1", ("base", "lid")), remove_lid("op_2"),
                                    inspection("inspection_2", ("base",))))

    assert set(report["observed_snapshots"]) == {"inspection_1", "inspection_2"}
    assert 'name="screw"' not in report["observed_snapshots"]["inspection_1"]["assembly_xml"]
    assert 'name="lid"' not in report["observed_snapshots"]["inspection_2"]["assembly_xml"]
    assert report["observed_snapshots"]["inspection_2"]["cell_state"]["mounted_tools"]["robot_1"] == "H5"
    assert all(o["succeeded"] for o in report["outcomes"])


# --- sync points and NC -------------------------------------------------------------------------------------------

XML2 = """<assembly name="two_screws">
	<part name="base" mesh_path="base.stl" density="8000.0" material="Generic" />
	<part name="lid" mesh_path="lid.stl" density="8000.0" material="Generic" />
	<part name="screw_1" mesh_path="screw.stl" density="8000.0" material="Generic" />
	<part name="screw_2" mesh_path="screw.stl" density="8000.0" material="Generic" />
</assembly>"""


def two_screw_request(*extra, sync_points=(), target="cell"):
    """Both robots unscrew one screw each (T6 on both); `extra` operations follow."""
    snapshot = {**SNAPSHOT, "assembly_xml": XML2}
    first = op("op_1", "unscrewing", "screw_1", ("base", "lid", "screw_2"), share("robot_1", "T6", PATH))
    second = op("op_2", "unscrewing", "screw_2", ("base", "lid", "screw_1"), share("robot_2", "T6", PATH))
    req = request(first, second, *extra, sync_points=sync_points, target=target)
    req["start_snapshot"] = req["operation_plan"]["start_snapshot"] = snapshot
    return req


def nc(report, channel):
    return report["nc_programs"][str(channel)]["text"]


def test_a_motion_conflict_adds_a_motion_level_sync_point_and_a_waitm_in_both_programs(service):
    configured(service, conflicting=[("op_1", "op_2")])

    report = service.handle(two_screw_request(inspection("inspection_1", ("base", "lid"))))

    assert report["infeasible"] is None
    [added] = report["sync_points_added"]
    assert added == {"robot": "robot_2", "before": "op_2", "until": "op_1", "layer": "motion", "reason": "paths_too_close"}
    spans = {t["operation_id"]: (t["start"], t["end"]) for t in report["timeline"] if t["kind"] == "operation"}
    assert spans["op_2"][0] >= spans["op_1"][1]  # robot 2 waited until robot 1's segment was over
    for channel in (1, 2):
        assert "WAITM(1,1,2) ; sync robot_2 waits before op_2 for op_1 (paths_too_close)" in nc(report, channel)
    lines = nc(report, 1).splitlines()
    assert next(i for i, l in enumerate(lines) if l.startswith("WAITM")) > max(i for i, l in enumerate(lines) if l.startswith("G1"))


def test_segments_that_do_not_conflict_run_at_the_same_time_without_a_sync_point(service):
    configured(service)

    report = service.handle(two_screw_request(inspection("inspection_1", ("base", "lid"))))

    assert report["sync_points_added"] == []
    spans = {t["operation_id"]: (t["start"], t["end"]) for t in report["timeline"] if t["kind"] == "operation"}
    assert spans["op_1"] == spans["op_2"] and "WAITM" not in nc(report, 1)


def test_a_task_level_sync_point_becomes_a_waitm_in_both_programs_without_a_motion_level_one(service):
    configured(service)

    report = service.handle(two_screw_request(sync_points=[sync("robot_2", "op_2", "op_1")]))

    assert report["sync_points_added"] == []
    spans = {t["operation_id"]: (t["start"], t["end"]) for t in report["timeline"] if t["kind"] == "operation"}
    assert spans["op_2"][0] >= spans["op_1"][1]
    for channel in (1, 2):
        assert "WAITM(1,1,2) ; sync robot_2 waits before op_2 for op_1 (order)" in nc(report, channel)


def test_nc_programs_end_at_the_first_inspection_but_the_whole_plan_is_rehearsed(service):
    configured(service)
    plan = (unscrew(path=PATH), inspection("inspection_1", ("base", "lid")), remove_lid(path=PATH),
            inspection("inspection_2", ("base",)))

    report = service.handle(request(*plan, target="cell"))

    assert report["infeasible"] is None and report["feasible_operations"] == ["op_1", "inspection_1", "op_2", "inspection_2"]
    assert set(report["nc_programs"]) == {"1", "2"} and report["nc_programs"]["1"]["file_name"] == "MRTAMP_CH1.MPF"
    text = nc(report, 1)
    assert "op_1: unscrewing screw" not in text and "end of segment (inspection)" in text
    assert text.count("G1") == len(PATH) + 1 and "op_2" not in text  # unscrewing's path and transit only
    assert text.rstrip().endswith("M30")
    assert "TOOLCHANGE H5 -> T6" in text


def test_a_late_infeasible_operation_means_no_nc_at_all(service):
    configured(service)
    ghost = op("op_2", "gripping_and_removing", "ghost", ("base",), share("robot_1", "H5"))

    report = service.handle(request(unscrew(path=PATH), inspection("inspection_1", ("base", "lid")), ghost, target="cell"))

    assert report["infeasible"]["operation_id"] == "op_2" and report["nc_programs"] is None
    assert "inspection_1" in report["observed_snapshots"]


def test_a_simulation_target_produces_no_nc(service):
    assert service.handle(request(unscrew(), inspection("inspection_1", ("base", "lid"))))["nc_programs"] is None


# --- execution modes ---------------------------------------------------------------------------------------

def test_physics_is_opt_in_and_refused_when_the_simulation_cannot_run_it(service):
    with pytest.raises(InvalidRequest, match="physics"):
        service.handle(request(unscrew(), mode="physics"))


def test_physics_failure_is_a_failed_operation_that_the_next_inspection_observes(service):
    service.physics_executor = lambda plan, segments: {"op_2": {"succeeded": False, "text": "lid jammed"}}

    report = service.handle(request(unscrew(), remove_lid(), inspection("inspection_1", ("base",)), mode="physics"))

    assert [(o["operation_id"], o["succeeded"]) for o in report["outcomes"]] == [("op_1", True), ("op_2", False)]
    assert report["outcomes"][1]["reason_code"] == "contact_abort"
    assert 'name="lid"' in report["observed_snapshots"]["inspection_1"]["assembly_xml"]  # the lid is still on
    assert 'name="screw"' not in report["observed_snapshots"]["inspection_1"]["assembly_xml"]


# --- check_feasibility ---------------------------------------------------------------------------------------

def query(service, *group, cell=CELL):
    return service.handle({"request": "check_feasibility", "concurrent_group": list(group), "snapshot": SNAPSHOT, "cell": cell,
                           "travel_times": TRAVEL_TIMES})


def test_a_feasible_group_answers_with_durations_and_executes_nothing(service):
    configured(service)
    group = [unscrew(path=PATH), op("op_2", "unscrewing", "screw", ("base", "lid"), share("robot_2", "T6", PATH))]
    group[1]["target"] = "lid"  # a different part
    group[1]["process"] = "unscrewing"

    answer = query(service, *group, cell={**CELL, "robots": [CELL["robots"][0], {**CELL["robots"][1]}]})

    assert answer["feasible"] is True and answer["infeasible"] is None
    assert answer["durations"] == {"op_1": 84.0, "op_2": 84.0}  # a tool change (robot 2 has none mounted), transit, path
    assert service.planner.calls  # it planned; but nothing about the plan was kept
    assert query(service, *group)["durations"] == answer["durations"]


def test_a_conflicting_group_is_infeasible_with_a_collision(service):
    configured(service, conflicting=[("op_1:transit", "op_2:transit")])
    one, two = unscrew(path=PATH), op("op_2", "unscrewing", "lid", ("base", "screw"), share("robot_2", "T6", PATH))

    answer = query(service, one, two)

    assert answer["feasible"] is False
    assert (answer["infeasible"]["operation_id"], answer["infeasible"]["reason_code"]) == ("op_2", "collision")


def test_two_operations_on_one_part_at_once_are_infeasible(service):
    one = unscrew()
    two = op("op_2", "unscrewing", "screw", ("base", "lid"), share("robot_2", "T6"))

    answer = query(service, one, two)

    assert answer["infeasible"]["operation_id"] == "op_2" and answer["infeasible"]["reason_code"] == "cell_state_mismatch"


def test_a_group_with_two_operations_on_one_robot_is_an_invalid_request(service):
    with pytest.raises(InvalidRequest, match="one operation per robot"):
        query(service, unscrew("op_1"), remove_lid("op_2"))


def test_a_feasibility_query_without_a_planner_never_assumes_free_motion(service):
    answer = query(service, unscrew(path=PATH))

    assert answer["feasible"] is False and answer["infeasible"]["reason_code"] == "no_path_share"


# --- malformed requests ------------------------------------------------------------------------------------------

MALFORMED = [
    pytest.param(lambda: {"request": "simulate"}, id="no-fields"),
    pytest.param(lambda: {**request(unscrew()), "operation_plan": None}, id="no-plan"),
    pytest.param(lambda: {**request(unscrew()), "target": "moon"}, id="target"),
    pytest.param(lambda: {**request(unscrew()), "execution_mode": "turbo"}, id="mode"),
    pytest.param(lambda: {**request(unscrew()), "cell": {"robots": []}}, id="no-robots"),
    pytest.param(lambda: {k: v for k, v in request(unscrew()).items() if k != "travel_times"}, id="no-travel-times"),
    pytest.param(lambda: request(unscrew(), unscrew()), id="duplicate-ids"),
    pytest.param(lambda: request(unscrew(), remove_lid(), sync_points=[sync("robot_1", "op_1", "op_2")]), id="sync-waits-for-later-op"),
    pytest.param(lambda: request(unscrew(), sync_points=[sync("robot_1", "op_1", "ghost")]), id="sync-unknown-op"),
    pytest.param(lambda: {**request(unscrew()), "start_snapshot": {"assembly_xml": "<broken", "cell_state": {"holds": {}, "mounted_tools": {}}}}, id="xml"),
    pytest.param(lambda: {"request": "check_feasibility", "concurrent_group": [], "snapshot": SNAPSHOT, "cell": CELL,
                          "travel_times": TRAVEL_TIMES}, id="empty-group"),
    pytest.param(lambda: {"request": "check_feasibility", "concurrent_group": ["x"], "snapshot": SNAPSHOT, "cell": CELL,
                          "travel_times": TRAVEL_TIMES}, id="group-not-ops"),
]


@pytest.mark.parametrize("make", MALFORMED)
def test_a_malformed_request_is_an_invalid_request(service, make):
    with pytest.raises((InvalidRequest, KeyError)):
        service.handle(make())


def test_malformed_and_failing_requests_do_not_end_the_server(service):
    configured(service)
    a, b = socket.socketpair()
    threading.Thread(target=serve_connection, args=(b, service.handle), daemon=True).start()

    def call(req):
        payload = pickle.dumps(req)
        a.sendall(len(payload).to_bytes(16, "big") + payload)
        size = int.from_bytes(_recv(a, 16), "big")
        return pickle.loads(_recv(a, size))

    def _recv(conn, n):
        data = b""
        while len(data) < n:
            data += conn.recv(n - len(data))
        return data

    assert call({"request": "simulate", "operation_plan": None})["error"] == "invalid_request"
    good = call(request(unscrew(), inspection("inspection_1", ("base", "lid"))))
    assert good["ok"] is True and good["infeasible"] is None
    assert call({"request": "nonsense"})["error"] == "invalid_request"
    a.close()


def test_the_request_is_not_changed_by_simulating_it(service):
    configured(service)
    req = request(unscrew(path=PATH), inspection("inspection_1", ("base", "lid")), target="cell")
    before = copy.deepcopy(req)

    service.handle(req)

    assert req == before


# --- regressions found by review ---------------------------------------------------------------------------------

def test_a_sync_across_the_inspection_cut_leaves_no_unpaired_waitm(service):
    configured(service)
    o1 = op("o1", "unscrewing", "screw", ("base", "lid"), share("robot_1", "T6", PATH))
    o3 = op("o3", "unscrewing", "lid", ("base",), share("robot_2", "T6"))

    report = service.handle(request(o1, inspection("i2", ("base", "lid")), o3, sync_points=[sync("robot_2", "o3", "o1")],
                                    target="cell"))

    assert report["infeasible"] is None
    for channel in (1, 2):  # the waiter's mark lies beyond the cut, so the blocker's is dropped too
        assert "WAITM" not in nc(report, channel)
    assert "falls beyond the end of this segment" in nc(report, 1)


def test_a_sync_before_the_cut_keeps_its_waitm_pair(service):
    configured(service)

    report = service.handle(two_screw_request(inspection("inspection_1", ("base", "lid")), sync_points=[sync("robot_2", "op_2", "op_1")]))

    assert all("WAITM(1,1,2)" in nc(report, channel) for channel in (1, 2))


def test_every_robots_program_ends_at_the_first_inspection_even_one_that_took_no_part(service):
    configured(service)
    only_robot_2 = op("ins", "inspecting", None, ("base", "lid", "screw"), share("robot_2", None, duration=3.0))
    later = op("op_2", "gripping_and_removing", "lid", ("base", "screw"), share("robot_1", "H5", PATH))

    report = service.handle(request(only_robot_2, later, target="cell"))

    assert "op_2" not in nc(report, 1) and "end of segment (inspection)" in nc(report, 1)


def test_two_sync_points_before_one_operation_are_both_honoured(service):
    configured(service)
    a = op("a", "unscrewing", "screw", ("base", "lid"), share("robot_2", "T6", duration=3.0))
    c = op("c", "unscrewing", "lid", ("base",), share("robot_2", "T6", duration=500.0))
    d = op("d", "gripping_and_removing", "base", (), share("robot_1", "H5", duration=1.0))
    report = service.handle(request(a, c, d, sync_points=[sync("robot_1", "d", "c"), sync("robot_1", "d", "a")], target="cell"))

    spans = {t["operation_id"]: (t["start"], t["end"]) for t in report["timeline"] if t["kind"] == "operation"}
    assert spans["d"][0] >= spans["c"][1]
    assert nc(report, 1).count("WAITM") == 2 and nc(report, 2).count("WAITM") == 2


def test_a_group_of_operations_without_paths_is_feasible_without_a_planner(service):
    one = unscrew("op_1")
    two = op("op_2", "unscrewing", "lid", ("base", "screw"), share("robot_2", "T6"))

    assert query(service, one, two)["feasible"] is True


def test_an_inspection_with_a_robot_that_is_not_in_the_cell_is_unsupported_not_a_crash(service):
    ghost = op("i", "inspecting", None, ("base", "lid", "screw"), share("robot_2", None, PATH), share("ghost", None, PATH))

    for report in (service.handle(request(ghost)), query(service, ghost)):
        found = report.get("infeasible")
        assert found["reason_code"] == "unsupported_process" and "ghost" in found["text"]


def test_physics_outcomes_start_from_the_request_snapshot_not_the_plan_snapshot(service):
    service.physics_executor = lambda plan, segments: {"op_1": {"succeeded": False}}
    req = request(unscrew(), mode="physics")
    req["operation_plan"]["start_snapshot"] = {**SNAPSHOT, "assembly_xml": XML.replace("screw", "other")}

    report = service.handle(req)

    assert report["outcomes"][0]["observed_product_state"] == ["base", "lid", "screw"]


@pytest.mark.parametrize("mutate", [
    pytest.param(lambda r: r["operation_plan"]["operations"][0]["shares"][0].update(path=[[0, 0, 0.1], [0, 0]]), id="ragged-path"),
    pytest.param(lambda r: r["operation_plan"]["operations"][0]["shares"][0].update(path=[5]), id="pose-not-a-list"),
    pytest.param(lambda r: r["operation_plan"]["operations"][0]["shares"][0].update(path=[["a", "b", "c"]]), id="pose-not-numbers"),
    pytest.param(lambda r: r["operation_plan"]["operations"][0].update(expected_product_state=[["a"]]), id="state-not-names"),
    pytest.param(lambda r: r["operation_plan"]["operations"][0]["shares"][0].update(path_duration=float("nan")), id="nan-duration"),
    pytest.param(lambda r: r["operation_plan"]["operations"][0]["shares"][0].update(path_duration=-5.0), id="negative-duration"),
    pytest.param(lambda r: r["operation_plan"]["operations"][0]["shares"][0].update(path_duration=float("inf")), id="inf-duration"),
    pytest.param(lambda r: r["operation_plan"]["operations"][0]["shares"][0].update(path_duration=True), id="bool-duration"),
    pytest.param(lambda r: r["travel_times"]["times"].update(toolchange_time_at_toolstorage="x"), id="time-not-a-number"),
    pytest.param(lambda r: r["travel_times"]["times"].update(toolchange_time_at_toolstorage=None), id="time-none"),
    pytest.param(lambda r: r.update(travel_times={"toolchange_time_at_toolstorage": 60.0}), id="times-not-in-a-table"),
    pytest.param(lambda r: r.update(product_model_xml="<a"), id="model-xml-broken"),
    pytest.param(lambda r: r["start_snapshot"].update(assembly_xml='<assembly><part mesh_path="x"/></assembly>'), id="part-without-name"),
    pytest.param(lambda r: r["operation_plan"]["operations"][0]["shares"].append(share("robot_1", "T6")), id="two-shares-one-robot"),
    pytest.param(lambda r: r["cell"]["robots"].append(dict(r["cell"]["robots"][0])), id="duplicate-robot"),
])
def test_malformed_values_are_an_invalid_request_not_an_internal_error(service, mutate):
    configured(service)
    req = copy.deepcopy(request(unscrew(path=PATH), inspection("inspection_1", ("base", "lid"))))
    mutate(req)

    with pytest.raises(InvalidRequest):
        service.handle(req)


def test_a_group_with_duplicate_operation_ids_is_an_invalid_request(service):
    with pytest.raises(InvalidRequest, match="unique"):
        query(service, unscrew("op_1"), op("op_1", "unscrewing", "lid", ("base",), share("robot_2", "T6")))


def test_numpy_numbers_from_the_techstack_are_accepted(service):
    import numpy as np

    configured(service)
    req = request(unscrew(path=[[np.float32(0.0), 0.0, np.float64(0.1)], [0.0, 0.0, 0.0]]), inspection("inspection_1", ("base", "lid")))
    req["operation_plan"]["operations"][0]["shares"][0]["path_duration"] = np.float32(5.0)
    req["cell"]["robots"][0]["nc_channel"] = np.int64(1)

    assert service.handle(req)["infeasible"] is None


# --- one operation per call (ADR-0010 of the techstack) ------------------------------------------------------

class WholeOperation(StraightLine):
    """A test planner that plans a whole operation in one call and remembers what it got; every motion ends in
    the joint configuration `(number of calls so far,)`. `fail_transit_to` here names an operation id."""

    name = "whole_operation"

    def __init__(self, options):
        super().__init__(options)
        self.given = []

    def plan_operation(self, robots, operation, starts, snapshot):
        self.given.append((operation["id"], sorted(robots), copy.deepcopy(starts), copy.deepcopy(snapshot)))
        if operation["id"] == self.fail_transit_to:
            return MotionFailure(ReasonCode.NO_PATH_TRANSIT, "boxed in", robot="robot_1", during="transit")
        motions, end = {}, (len(self.given),)
        for s in operation["shares"]:
            if s["path"]:
                motions[s["robot"]] = (ShareMotion(1.0, [{"comment": "transit"}], tuple(s["path"][0]), end),
                                       ShareMotion(2.0, [{"comment": "share"}], tuple(s["path"][-1]), end))
        return motions


@pytest.fixture
def whole(service):
    service.register_planner("whole_operation", WholeOperation)
    service.handle({"request": "configure_planner", "planner": "whole_operation"})
    return service


def test_a_planner_plans_one_operation_per_call_with_all_its_shares(whole):
    handover = op("op_h", "unscrewing", "screw", ("base", "lid"), share("robot_1", "T6", PATH), share("robot_2", "T6", PATH))

    report = whole.handle(request(handover))

    assert report["infeasible"] is None
    assert [(g[0], g[1]) for g in whole.planner.given] == [("op_h", ["robot_1", "robot_2"])]
    assert whole.planner.calls == []  # neither plan_share nor plan_transit
    kinds = sorted((t["kind"], t["robot"]) for t in report["timeline"] if t["kind"] != "tool_change")
    assert kinds == [("operation", "robot_1"), ("operation", "robot_2"), ("transit", "robot_1"), ("transit", "robot_2")]


def test_the_planner_gets_the_snapshot_as_it_is_before_the_operation_holds_included(whole):
    without_screw = XML.replace('<part name="screw" mesh_path="screw.stl" density="8000.0" material="Generic" />', "")
    grip = op("op_g", "gripping", "screw", ("base", "lid"), share("robot_1", "H5", PATH), parameters={"supply_slot": "s1"})
    insert = op("op_i", "inserting", "screw", ("base", "lid", "screw"), share("robot_1", "H5", PATH))
    cell = copy.deepcopy(CELL)
    cell["robots"][0]["capabilities"] += ["gripping", "inserting"]
    req = request(grip, insert, product_model_xml=XML, cell=cell)
    req["start_snapshot"] = req["operation_plan"]["start_snapshot"] = {**SNAPSHOT, "assembly_xml": without_screw}

    assert whole.handle(req)["infeasible"] is None

    first, second = whole.planner.given[0][3], whole.planner.given[1][3]
    assert first["product_state"] == ["base", "lid"] and first["holds"] == {}
    assert second["holds"] == {"robot_1": "screw"} and 'name="screw"' not in second["assembly_xml"]
    assert second["mounted_tools"] == {"robot_1": "H5", "robot_2": None} and 'name="screw"' in second["product_model_xml"]


def test_each_robot_starts_where_its_last_motion_ended(whole):
    whole.handle(request(unscrew("op_1", PATH), remove_lid("op_2", PATH)))

    starts_1, starts_2 = whole.planner.given[0][2], whole.planner.given[1][2]
    assert starts_1["robot_1"] == {"pose": None, "configuration": None}  # home
    assert starts_2["robot_1"] == {"pose": tuple(PATH[-1]), "configuration": (1,)}
    assert starts_2["robot_2"] == {"pose": None, "configuration": None}


def test_a_failed_transit_of_a_whole_operation_keeps_its_reason_and_robot(whole):
    whole.planner.fail_transit_to = "op_1"

    report = whole.handle(request(unscrew("op_1", PATH), target="cell"))

    assert report["infeasible"] == {"kind": "infeasible_operation", "operation_id": "op_1", "robot": "robot_1",
                                    "reason_code": "no_path_transit", "text": "transit to op_1: boxed in"}
    assert report["nc_programs"] is None


def test_a_planner_that_leaves_out_a_share_is_a_planner_bug(whole):
    class Forgetful(WholeOperation):
        def plan_operation(self, robots, operation, starts, snapshot):
            return {}

    whole.register_planner("forgetful", Forgetful)
    whole.handle({"request": "configure_planner", "planner": "forgetful"})

    with pytest.raises(RuntimeError, match="robot_1"):  # the server turns it into internal_error
        whole.handle(request(unscrew(path=PATH)))


def test_check_feasibility_plans_whole_operations_from_home_and_the_given_snapshot(whole):
    assert query(whole, unscrew("op_1", PATH))["feasible"] is True

    _, _, starts, snapshot = whole.planner.given[0]
    assert starts["robot_1"] == {"pose": None, "configuration": None}
    assert snapshot["product_state"] == ["base", "lid", "screw"] and snapshot["mounted_tools"]["robot_1"] == "H5"
    assert snapshot["product_model_xml"] == XML  # without a product model in the request, the snapshot's


def test_check_feasibility_gives_the_planner_the_full_product_model_held_parts_included(whole):
    held = {"assembly_xml": XML.replace('<part name="screw" mesh_path="screw.stl" density="8000.0" material="Generic" />', ""),
            "cell_state": {"holds": {"robot_1": "screw"}, "mounted_tools": {"robot_1": "H5", "robot_2": None}}}
    remove = op("op_1", "gripping_and_removing", "lid", ("base",), share("robot_1", "H5", PATH))

    whole.handle({"request": "check_feasibility", "concurrent_group": [remove], "snapshot": held, "cell": CELL,
                  "travel_times": TRAVEL_TIMES, "product_model_xml": XML})

    snapshot = whole.planner.given[0][3]
    assert 'name="screw"' in snapshot["product_model_xml"] and 'name="screw"' not in snapshot["assembly_xml"]
