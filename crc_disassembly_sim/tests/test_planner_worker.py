"""The planner worker client and protocol (techstack #212), against a fake worker on a localhost socket."""
import copy
import socket
import threading

import pytest

import planner_worker
import planner_worker_client
from cell_simulation import CellSimulationService
from planner_worker import PlannerWorker
from sim_server_protocol import InvalidRequest
from test_cell_simulation import CELL, PATH, SNAPSHOT, XML, inspection, op, query, request, share, unscrew


class FakeWorker(PlannerWorker):
    """Moves every share with joint moves: a transit of 1 s and a share of 2 s, ending in the configuration
    `(number of operations planned so far,) * 6`. `infeasible`: the answer it gives instead; `answer`: a raw
    answer it gives instead (to test malformed ones); `clash`: pairs of segment ids that conflict."""

    name = "fake"

    def __init__(self):
        self.scenes, self.conflict_tests = [], []
        self.infeasible = self.answer = None
        self.clash = set()

    def plan_operation(self, scene):
        self.scenes.append(copy.deepcopy(scene))
        if self.answer is not None:
            return self.answer
        if self.infeasible is not None:
            return {"infeasible": self.infeasible}
        end = [float(len(self.scenes))] * 6
        motions = {}
        for s in scene["operation"]["shares"]:
            if s["path"]:
                joints = {f"RA{i + 1}": v for i, v in enumerate(end)}
                motions[s["robot"]] = {
                    "transit": {"duration": 1.0, "commands": [{"G": 1, **joints, "F": 5000.0}], "end_pose": s["path"][0],
                                "end_configuration": end, "conflict_data": {"segment": "transit", "robot": s["robot"]}},
                    "share": {"duration": 2.0, "commands": [{"G": 1, "X": 0.1, "Y": 0.2, "Z": 0.3, "F": 600.0}],
                              "end_pose": s["path"][-1], "end_configuration": end,
                              "conflict_data": {"segment": "share", "robot": s["robot"]}},
                }
        return {"motions": motions}

    def conflicts(self, first, second):
        self.conflict_tests.append((first, second))
        return (first["id"], second["id"]) in self.clash or (second["id"], first["id"]) in self.clash


@pytest.fixture
def worker():
    fake, ready = FakeWorker(), threading.Event()
    address = []

    def bound(where):
        address.append(where)
        ready.set()

    threading.Thread(target=planner_worker.serve, args=(fake,), kwargs={"ready": bound}, daemon=True).start()
    assert ready.wait(5)
    fake.address = "%s:%d" % tuple(address[0])
    return fake


def client_service(address, **options):
    service = CellSimulationService()
    planner_worker_client.register(service)
    service.handle({"request": "configure_planner", "planner": "planner_worker",
                    "options": {"address": address, "timeout": 5.0, **options}})
    return service


def free_port_nobody_listens_on():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def nc(report, channel):
    return report["nc_programs"][str(channel)]["text"]


# --- one operation per call ------------------------------------------------------------------------------------

def test_the_worker_gets_one_request_per_operation_and_its_nc_commands_reach_the_programs(worker):
    service = client_service(worker.address)
    handover = op("op_h", "unscrewing", "screw", ("base", "lid"), share("robot_1", "T6", PATH), share("robot_2", "T6", PATH))

    report = service.handle(request(handover, inspection("inspection_1", ("base", "lid")), target="cell"))

    assert report["infeasible"] is None
    assert [s["operation"]["id"] for s in worker.scenes] == ["op_h"]
    for channel in (1, 2):
        assert "RA1=57.296" in nc(report, channel) and "X100.000 Y200.000 Z300.000" in nc(report, channel)
    times = {(t["kind"], t["robot"]): t["end"] - t["start"] for t in report["timeline"] if t.get("operation_id", "").startswith("op_h")}
    assert times[("transit", "robot_1")] == 1.0 and times[("operation", "robot_1")] == 2.0


def test_the_scene_holds_the_snapshot_held_parts_and_each_robots_start_configuration(worker):
    service = client_service(worker.address, obstacles=[{"name": "fixture", "box": [0.2, 0.2, 0.1]}],
                             reference_poses={"base": [0.5, 0.0, 0.8, 0.0, 0.0, 1.57]})
    without_screw = XML.replace('<part name="screw" mesh_path="screw.stl" density="8000.0" material="Generic" />', "")
    cell = copy.deepcopy(CELL)
    cell["robots"][0]["capabilities"] += ["gripping", "inserting"]
    grip = op("op_g", "gripping", "screw", ("base", "lid"), share("robot_1", "H5", PATH), parameters={"supply_slot": "s1"})
    insert = op("op_i", "inserting", "screw", ("base", "lid", "screw"), share("robot_1", "H5", PATH),
                parameters={"reference_part": "base"})
    req = request(grip, insert, product_model_xml=XML, cell=cell)
    req["start_snapshot"] = req["operation_plan"]["start_snapshot"] = {**SNAPSHOT, "assembly_xml": without_screw}

    assert service.handle(req)["infeasible"] is None

    first, second = worker.scenes
    assert first["robots"]["robot_1"]["start_configuration"] is None and first["held_parts"] == {}
    assert second["robots"]["robot_1"] == {
        "start_pose": PATH[-1], "start_configuration": [1.0] * 6, "mounted_tool": "H5", "holds": "screw",
        "in_operation": True, "nc_channel": 1, "tools": ["H5", "T6"], "capabilities": cell["robots"][0]["capabilities"]}
    assert second["robots"]["robot_2"] == {"start_pose": None, "start_configuration": None, "mounted_tool": None,
                                           "holds": None, "in_operation": False}
    assert second["held_parts"] == {"robot_1": "screw"}
    assert 'name="screw"' not in second["product"]["assembly_xml"] and 'name="screw"' in second["product"]["product_model_xml"]
    assert second["product"]["reference_part"] == "base" and second["product"]["reference_pose"] == [0.5, 0.0, 0.8, 0.0, 0.0, 1.57]
    assert second["obstacles"] == [{"name": "fixture", "box": [0.2, 0.2, 0.1]}]


def test_numpy_numbers_reach_the_worker_as_plain_floats(worker):
    import numpy as np

    service = client_service(worker.address)
    req = request(unscrew(path=[[np.float32(0.0), 0.0, np.float64(0.1)], [0.0, 0.0, 0.0]]))

    assert service.handle(req)["infeasible"] is None
    path = worker.scenes[0]["operation"]["shares"][0]["path"]
    assert all(type(v) is float for pose in path for v in pose)


# --- no silent fallback ------------------------------------------------------------------------------------------

def test_an_unreachable_worker_makes_the_operation_infeasible_and_produces_no_nc():
    service = client_service(f"127.0.0.1:{free_port_nobody_listens_on()}")

    report = service.handle(request(unscrew("op_1", PATH), target="cell"))

    assert report["infeasible"]["operation_id"] == "op_1" and report["infeasible"]["robot"] == "robot_1"
    assert report["infeasible"]["reason_code"] == "no_path_share"
    assert "did not answer" in report["infeasible"]["text"] and "never falls back" in report["infeasible"]["text"]
    assert report["feasible_operations"] == [] and report["nc_programs"] is None


def test_an_unreachable_worker_makes_a_group_infeasible_in_check_feasibility():
    service = client_service(f"127.0.0.1:{free_port_nobody_listens_on()}")

    answer = query(service, unscrew("op_1", PATH))

    assert answer["feasible"] is False and answer["infeasible"]["reason_code"] == "no_path_share"


def test_the_workers_infeasible_operation_comes_back_with_its_reason_robot_and_text(worker):
    worker.infeasible = {"reason_code": "no_path_transit", "robot": "robot_1", "during": "transit", "text": "no IK"}
    service = client_service(worker.address)

    report = service.handle(request(unscrew("op_1", PATH), target="cell"))

    assert report["infeasible"] == {"kind": "infeasible_operation", "operation_id": "op_1", "robot": "robot_1",
                                    "reason_code": "no_path_transit", "text": "transit to op_1: no IK"}
    assert report["nc_programs"] is None


@pytest.mark.parametrize("answer, match", [
    pytest.param({"infeasible": {"reason_code": "contact_abort", "robot": "robot_1", "during": "share", "text": "x"}},
                 "contact_abort", id="not-an-infeasible-code"),
    pytest.param({"infeasible": {"reason_code": "made_up", "robot": "robot_1", "during": "share", "text": "x"}},
                 "made_up", id="unknown-code"),
    pytest.param({"infeasible": {"reason_code": "collision", "text": "x"}}, "without its robot", id="no-robot-no-during"),
    pytest.param({"motions": {}}, "robot_1", id="share-left-out"),
    pytest.param({"motions": {"robot_1": {"transit": {"duration": 1.0, "commands": []}}}}, "no share motion",
                 id="no-share-motion"),
    pytest.param({"motions": {"robot_1": {"transit": {"duration": -1.0, "commands": []},
                                          "share": {"duration": 1.0, "commands": []}}}}, "duration", id="negative-duration"),
    pytest.param({"motions": {"robot_1": {"transit": {"duration": 1.0, "commands": "G1"},
                                          "share": {"duration": 1.0, "commands": []}}}}, "commands", id="commands-not-a-list"),
    pytest.param({"motions": {"robot_1": {"transit": {"duration": 1.0, "commands": []}, "share": {"duration": 1.0}}}},
                 "commands", id="commands-left-out"),
    pytest.param({}, "neither motions nor infeasible", id="empty"),
])
def test_a_malformed_worker_answer_is_a_worker_bug_not_an_infeasible_operation(worker, answer, match):
    worker.answer = answer
    service = client_service(worker.address)

    with pytest.raises(RuntimeError, match=match):  # the sim server turns it into internal_error
        service.handle(request(unscrew("op_1", PATH)))


def test_a_worker_that_fails_on_a_request_is_an_internal_error_with_its_message(worker):
    def broken(scene):
        raise ValueError("cuRobo ran out of memory")

    worker.plan_operation = broken
    service = client_service(worker.address)

    with pytest.raises(RuntimeError, match="internal_error: ValueError: cuRobo ran out of memory"):
        service.handle(request(unscrew("op_1", PATH)))


# --- conflicts ---------------------------------------------------------------------------------------------------

def test_the_worker_decides_conflicts_with_the_data_it_returned(worker):
    worker.clash = {("op_1", "op_2")}
    service = client_service(worker.address)
    two = op("op_2", "unscrewing", "lid", ("base",), share("robot_2", "T6", PATH))

    report = service.handle(request(unscrew("op_1", PATH), two, inspection("inspection_1", ("base", "lid")), target="cell"))

    assert report["infeasible"] is None
    assert any(s["until"] == "op_1" and s["before"] == "op_2" for s in report["sync_points_added"])
    first, second = next((a, b) for a, b in worker.conflict_tests if {a["id"], b["id"]} == {"op_1", "op_2"})
    assert {first["motion"]["segment"], second["motion"]["segment"]} == {"share"}
    assert "WAITM" in nc(report, 1) and "WAITM" in nc(report, 2)


def test_a_conflict_test_the_worker_fails_on_is_an_error_never_a_guess(worker):
    service = client_service(worker.address)
    two = op("op_2", "unscrewing", "lid", ("base",), share("robot_2", "T6", PATH))
    worker.conflicts = lambda first, second: (_ for _ in ()).throw(ValueError("no geometry"))

    with pytest.raises(RuntimeError, match="failed on conflicts"):
        service.handle(request(unscrew("op_1", PATH), two))


def test_a_conflict_test_the_worker_does_not_answer_is_an_error_never_a_guess(worker):
    service = client_service(worker.address)
    two = op("op_2", "unscrewing", "lid", ("base",), share("robot_2", "T6", PATH))
    plan = worker.plan_operation

    def plan_then_vanish(scene):  # the worker goes away after planning op_2, before its conflict test
        if scene["operation"]["id"] == "op_2":
            service.planner.address = ("127.0.0.1", free_port_nobody_listens_on())
        return plan(scene)

    worker.plan_operation = plan_then_vanish

    with pytest.raises(RuntimeError, match="did not answer a conflict test"):
        service.handle(request(unscrew("op_1", PATH), two))


# --- configuration -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("options, match", [
    pytest.param({}, "address", id="no-address"),
    pytest.param({"address": "localhost"}, "address", id="no-port"),
    pytest.param({"address": "localhost:99999"}, "address", id="port-out-of-range"),
    pytest.param({"address": "localhost:5800", "timeout": 0}, "timeout", id="zero-timeout"),
    pytest.param({"address": "localhost:5800", "obstacles": {"a": 1}}, "obstacles", id="obstacles-not-a-list"),
    pytest.param({"address": "localhost:5800", "reference_poses": {"base": [0, 0, 0]}}, "reference_poses", id="short-pose"),
    pytest.param({"address": "localhost:5800", "planner": "curobo"}, "unknown", id="unknown-option"),
])
def test_bad_options_are_refused_at_configure_planner(options, match):
    service = CellSimulationService()
    planner_worker_client.register(service)

    with pytest.raises(InvalidRequest, match=match):
        service.handle({"request": "configure_planner", "planner": "planner_worker", "options": options})
    assert service.planner is None


def test_a_worker_registered_with_an_address_is_configured_by_name_alone(worker):
    service = CellSimulationService()
    planner_worker_client.register(service, "curobo", worker.address)

    assert service.handle({"request": "configure_planner", "planner": "curobo"}) == {"ok": True, "planner": "curobo"}
    assert service.handle(request(unscrew("op_1", PATH)))["infeasible"] is None


def test_an_operation_relative_to_a_part_without_a_reference_pose_is_an_invalid_request(worker):
    service = client_service(worker.address)
    relative = op("op_1", "unscrewing", "screw", ("base", "lid"), share("robot_1", "T6", PATH),
                  parameters={"reference_part": "base"})

    with pytest.raises(InvalidRequest, match="reference_poses"):
        service.handle(request(relative))


def test_a_request_in_another_protocol_version_is_refused_by_the_worker(worker):
    host, port = worker.address.rsplit(":", 1)

    answer = planner_worker.call((host, int(port)), {"request": "plan_operation", "protocol": 0, "scene": {}}, 5.0)

    assert answer["ok"] is False and answer["error"] == "invalid_request" and "protocol" in answer["message"]
