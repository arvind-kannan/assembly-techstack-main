"""The client of the real cell simulation across the wire: planned motion, sync points, NC, errors, reconnecting."""

import pickle
import socket
import threading
from dataclasses import replace

import pytest
from diss_assembly_dataclasses.feedback import ReasonCode
from diss_assembly_dataclasses.operation import Operation, Share
from diss_assembly_dataclasses.operation_plan import (
    OperationPlan,
    PlanMetrics,
    SyncPoint,
)
from diss_assembly_dataclasses.process import Process

from mrtamp.cell_simulation import SimulationRequest
from mrtamp.nominal_schedule import nominal_schedule
from mrtamp.sim_client import LENGTH_BYTES, RealCellSimulation
from tests.mrtamp_scenarios import (
    TWO_SCREWS_XML,
    product_model,
    travel_times,
    two_robot_cell,
    two_screws_snapshot_both_on_t6,
)
from tests.sim_loopback import _sim_modules, loopback_simulation, requires_sim

PATH = (
    (0.0, 0.0, 10.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 10.0),
)  # 20 units at the test planner's 100 units/s


def unscrew(op_id, target, robot, state, path=PATH):
    return Operation(
        op_id,
        Process.UNSCREWING,
        target,
        (Share(robot, "T6", path, 1.0),),
        expected_product_state=state,
    )


def two_screws_request(*, sync_points=(), target="simulation"):
    operations = (
        unscrew("op_1", "screw_1", "robot_1", ("base", "lid", "screw_2")),
        unscrew("op_2", "screw_2", "robot_2", ("base", "lid", "screw_1")),
    )
    snapshot = two_screws_snapshot_both_on_t6()
    plan = OperationPlan(operations, {}, tuple(sync_points), PlanMetrics(0.0), snapshot)
    return SimulationRequest(
        plan,
        snapshot,
        two_robot_cell(),
        target,
        product_model=product_model(TWO_SCREWS_XML),
        travel_times=travel_times(),
    )


@requires_sim
def test_cell_times_come_from_the_planned_motion_across_the_wire():
    simulation, _ = loopback_simulation()

    report = simulation.simulate(two_screws_request())

    assert report.ok and report.infeasible is None
    kinds = {(t["kind"], t["operation_id"]) for t in report.timeline}
    assert {
        ("transit", "op_1:transit"),
        ("operation", "op_1"),
        ("transit", "op_2:transit"),
        ("operation", "op_2"),
    } <= kinds
    spans = {
        t["operation_id"]: (t["start"], t["end"])
        for t in report.timeline
        if t["kind"] == "operation"
    }
    assert (
        spans["op_1"] == spans["op_2"] == (1.0, 1.2)
    )  # transit 1 s, then 20 units at 100 units/s; nothing in the way
    assert report.metrics.makespan == 1.2 and report.sync_points_added == ()


@requires_sim
def test_a_motion_conflict_comes_back_as_a_motion_level_sync_point():
    simulation, _ = loopback_simulation(
        planner_options={"conflicting": [("op_1", "op_2")]}
    )

    report = simulation.simulate(two_screws_request(target="cell"))

    assert report.sync_points_added == (
        SyncPoint("robot_2", "op_2", "op_1", "motion", "paths_too_close"),
    )
    assert all(
        "WAITM(1,1,2)" in program["text"] for program in report.nc_programs.values()
    )
    nominal = nominal_schedule(
        two_screws_request().operation_plan.operations,
        two_robot_cell(),
        {"robot_1": "T6", "robot_2": "T6"},
        travel_times=travel_times(),
    )[1]
    assert (
        report.metrics.makespan > nominal.makespan - 1.0
        and report.metrics.makespan >= 1.4
    )  # robot 2 waited for robot 1


@requires_sim
def test_without_a_configured_planner_a_plan_with_paths_is_infeasible_never_moved_directly():
    simulation, _ = loopback_simulation(configure=False)

    report = simulation.simulate(two_screws_request(target="cell"))

    assert report.ok and report.nc_programs is None
    assert (
        report.infeasible.reason_code is ReasonCode.NO_PATH_SHARE
        and "configure_planner" in report.infeasible.text
    )


@requires_sim
def test_the_planner_is_chosen_once_by_name_and_an_unknown_one_is_refused():
    simulation, service = loopback_simulation(configure=False)

    simulation.configure_planner("straight_line", {"feed": 50.0})
    with pytest.raises(ValueError, match="cumotion"):
        simulation.configure_planner("cumotion")

    assert service.planner.feed == 50.0  # the refused request left the setting alone


@requires_sim
def test_the_server_rejecting_a_request_comes_back_as_a_rejection():
    simulation, _ = loopback_simulation()
    request = two_screws_request()
    waits_for_a_later_operation = SyncPoint(
        "robot_1", "op_1", "op_2", "planner", "order"
    )
    bad = replace(
        request,
        operation_plan=replace(
            request.operation_plan, sync_points=(waits_for_a_later_operation,)
        ),
    )

    report = simulation.simulate(bad)

    assert not report.ok and report.rejection.reason_code is ReasonCode.INVALID_REQUEST


@requires_sim
def test_a_kept_open_connection_is_dropped_after_a_failure_and_the_next_call_reconnects():
    _, service = loopback_simulation()
    attempts = []

    def connect():
        client, server = socket.socketpair()
        attempts.append(client)
        if len(attempts) == 1:
            server.close()  # the first connection is dead on arrival
        else:
            threading.Thread(
                target=_sim_modules()[1].serve_connection,
                args=(server, service.handle),
                daemon=True,
            ).start()
        return client

    simulation = RealCellSimulation(connect, keep_open=True)

    with pytest.raises(OSError):
        simulation.simulate(two_screws_request())
    assert simulation.simulate(two_screws_request()).ok  # reconnected
    simulation.close()


def test_a_robot_may_wait_for_several_operations_before_one():
    cell = two_robot_cell()
    ops = (
        Operation(
            "a", Process.UNSCREWING, "screw_1", (Share("robot_2", "T6", (), 3.0),)
        ),
        Operation(
            "c", Process.UNSCREWING, "screw_2", (Share("robot_2", "T6", (), 500.0),)
        ),
        Operation(
            "d",
            Process.GRIPPING_AND_REMOVING,
            "lid",
            (Share("robot_1", "H5", (), 1.0),),
        ),
    )
    syncs = (
        SyncPoint("robot_1", "d", "c", "planner", "order"),
        SyncPoint("robot_1", "d", "a", "planner", "order"),
    )

    timeline, _, _ = nominal_schedule(
        ops,
        cell,
        {"robot_1": "H5", "robot_2": "T6"},
        syncs,
        travel_times=travel_times(),
    )

    d = next(
        t for t in timeline if t["kind"] == "operation" and t["operation_id"] == "d"
    )
    assert d["start"] == 503.0


def test_both_calls_send_the_xml_of_the_product_model_they_get():
    sent = []

    def connect():
        client, server = socket.socketpair()

        def answer():
            size = int.from_bytes(server.recv(LENGTH_BYTES, socket.MSG_WAITALL), "big")
            sent.append(pickle.loads(server.recv(size, socket.MSG_WAITALL)))
            reply = pickle.dumps({"ok": False, "error": "internal_error"})
            server.sendall(len(reply).to_bytes(LENGTH_BYTES, "big") + reply)
            server.close()

        threading.Thread(target=answer, daemon=True).start()
        return client

    simulation = RealCellSimulation(connect)
    request = two_screws_request()

    simulation.simulate(request)
    simulation.check_feasibility(
        request.operation_plan.operations[:1],
        request.start_snapshot,
        request.cell,
        product_model=product_model(TWO_SCREWS_XML),
        travel_times=travel_times(),
    )

    assert [wire["product_model_xml"] for wire in sent] == [TWO_SCREWS_XML] * 2
