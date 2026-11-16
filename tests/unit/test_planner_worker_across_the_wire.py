"""A planner worker behind the real cell simulation, seen from the techstack's client (ADR-0011): one scene
description per operation, the worker's NC commands in the report, and no fallback when the worker is gone."""

import os
import socket
import threading

import pytest
from diss_assembly_dataclasses.feedback import ReasonCode

from tests.sim_loopback import _sim_modules, planner_worker_simulation, requires_sim
from tests.unit.test_mrtamp_sim_client import two_screws_request


def fake_worker():
    """A planner worker in a thread on a free localhost port: every share is one joint move of 1.5 s after a
    transit of 0.5 s; returns the worker and its `"host:port"`."""
    _sim_modules()
    import planner_worker

    class Fake(planner_worker.PlannerWorker):
        def __init__(self):
            self.scenes = []

        def plan_operation(self, scene):
            self.scenes.append(scene)
            motions = {}
            for share in scene["operation"]["shares"]:
                end = [0.1 * len(self.scenes)] * 6
                move = {f"RA{i + 1}": v for i, v in enumerate(end)}
                motions[share["robot"]] = {
                    "transit": {
                        "duration": 0.5,
                        "commands": [],
                        "end_configuration": end,
                    },
                    "share": {
                        "duration": 1.5,
                        "commands": [{"G": 1, **move, "F": 100.0}],
                        "end_configuration": end,
                    },
                }
            return {"motions": motions}

        def conflicts(self, first, second):
            return False

    worker, bound = Fake(), []
    ready = threading.Event()
    threading.Thread(
        target=planner_worker.serve,
        args=(worker,),
        kwargs={"ready": lambda where: (bound.append(where), ready.set())},
        daemon=True,
    ).start()
    assert ready.wait(5)
    return worker, "%s:%d" % tuple(bound[0])


@requires_sim
def test_each_operation_reaches_the_worker_as_a_scene_and_its_nc_commands_come_back():
    worker, address = fake_worker()
    simulation = planner_worker_simulation(address)

    report = simulation.simulate(two_screws_request(target="cell"))

    assert report.ok and report.infeasible is None
    assert [s["operation"]["id"] for s in worker.scenes] == ["op_1", "op_2"]
    assert worker.scenes[0]["robots"]["robot_1"]["mounted_tool"] == "T6"
    spans = {
        t["operation_id"]: (t["start"], t["end"])
        for t in report.timeline
        if t["kind"] == "operation"
    }
    assert spans["op_1"] == (0.5, 2.0)
    assert all("RA1=" in program["text"] for program in report.nc_programs.values())


@requires_sim
def test_an_unreachable_worker_is_an_infeasible_operation_and_no_nc():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        address = "127.0.0.1:%d" % s.getsockname()[1]
    simulation = planner_worker_simulation(address, options={"timeout": 2.0})

    report = simulation.simulate(two_screws_request(target="cell"))

    assert report.ok and report.nc_programs is None
    assert (
        report.infeasible.reason_code is ReasonCode.NO_PATH_SHARE
        and address in report.infeasible.text
    )


@pytest.mark.gpu
@requires_sim
def test_a_real_planner_worker_answers_with_motion_or_a_reason_from_the_closed_list():
    """On the GPU workstation: start a planner worker, then `PLANNER_WORKER_ADDRESS=host:port pytest -m gpu`."""
    address = os.environ.get("PLANNER_WORKER_ADDRESS")
    if not address:
        pytest.skip(
            "PLANNER_WORKER_ADDRESS is not set (host:port of a running planner worker)"
        )
    simulation = planner_worker_simulation(address)

    report = simulation.simulate(two_screws_request(target="cell"))

    assert report.ok
    if report.infeasible is None:
        assert report.nc_programs
    else:
        assert (
            "did not answer" not in report.infeasible.text
        )  # the worker is up: a planning answer, not silence
