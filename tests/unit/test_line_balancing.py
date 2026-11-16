"""Line balancing on tiny instances with a hand-computed optimum."""

import json

import pytest

pytest.importorskip(
    "ortools",
    reason="line balancing needs ortools (not in uv.lock; uv add --dev ortools)",
)

from disassembly_pomdp.graph_precedence import Task
from disassembly_pomdp.line_balancing import (
    InfeasibleLineError,
    allocate_stations,
    choose_k_salbp_e,
)

UNSCREW = "unscrewing"
UNCLIP = "unclipping"
GRIP = "gripping and removing"


def tasks(*specs):
    """Tasks from (part, process, action_time) triples."""
    return {
        part: Task(part, process, None, time, None) for part, process, time in specs
    }


def assert_precedence_holds(plan, precedence):
    station_of = {
        part: station.station_id for station in plan.stations for part in station.parts
    }
    for p, q in precedence:
        assert station_of[p] <= station_of[q], f"{p} must not come after {q}"
    for station in plan.stations:
        order = station.parts
        for p, q in precedence:
            if p in order and q in order:
                assert order.index(p) < order.index(q)


def test_two_processes_in_a_chain_get_one_station_each():
    ts = tasks(("screw", UNSCREW, 2.0), ("lid", GRIP, 3.0))
    precedence = {("screw", "lid")}

    plan = allocate_stations(ts, precedence, 2)

    assert_precedence_holds(plan, precedence)
    assert plan.cycle_time == 3.0
    assert plan.n_stations == 2
    assert plan.n_stations_used == 2
    assert [(s.station_id, s.process, s.parts) for s in plan.stations] == [
        (1, UNSCREW, ["screw"]),
        (2, GRIP, ["lid"]),
    ]
    assert plan.cycle_time_optimal and plan.stations_used_optimal


def test_parallel_screws_are_balanced_and_a_spare_station_stays_empty():
    # 4 screws of 1 s before a 2 s lid: two screw stations of 2 s and the lid give cycle time 2;
    # the fourth station cannot lower it and stays empty.
    ts = tasks(*((f"s{i}", UNSCREW, 1.0) for i in range(4)), ("lid", GRIP, 2.0))
    precedence = {(f"s{i}", "lid") for i in range(4)}

    plan = allocate_stations(ts, precedence, 4)

    assert_precedence_holds(plan, precedence)
    assert plan.cycle_time == 2.0
    assert plan.n_stations == 4
    assert plan.n_stations_used == 3
    assert sorted(s.total_time for s in plan.stations) == [2.0, 2.0, 2.0]


SCREW_UNCLIP_SCREW = (
    tasks(
        ("outer_screw", UNSCREW, 1.0),
        ("clip", UNCLIP, 1.0),
        ("inner_screw", UNSCREW, 1.0),
    ),
    {("outer_screw", "clip"), ("clip", "inner_screw"), ("outer_screw", "inner_screw")},
)


def test_a_process_comes_back_at_a_later_non_adjacent_station():
    ts, precedence = SCREW_UNCLIP_SCREW

    plan = allocate_stations(ts, precedence, 3)

    assert_precedence_holds(plan, precedence)
    assert [(s.station_id, s.process, s.parts) for s in plan.stations] == [
        (1, UNSCREW, ["outer_screw"]),
        (2, UNCLIP, ["clip"]),
        (3, UNSCREW, ["inner_screw"]),
    ]
    assert plan.cycle_time == 1.0


def test_too_few_stations_for_the_process_order_is_infeasible():
    ts, precedence = (
        SCREW_UNCLIP_SCREW  # two processes, but the order needs three stations
    )

    with pytest.raises(InfeasibleLineError):
        allocate_stations(ts, precedence, 2)


def test_fewer_stations_than_processes_raises():
    ts, precedence = SCREW_UNCLIP_SCREW

    with pytest.raises(ValueError, match="at least 2 stations"):
        allocate_stations(ts, precedence, 1)


def test_a_precedence_cycle_gives_no_feasible_assignment():
    ts = tasks(("a", UNSCREW, 1.0), ("b", GRIP, 1.0))

    with pytest.raises(InfeasibleLineError):
        allocate_stations(ts, {("a", "b"), ("b", "a")}, 2)


SCREWS_BEFORE_LID = (
    tasks(*((f"s{i}", UNSCREW, 1.0) for i in range(4)), ("lid", GRIP, 2.0)),
    {(f"s{i}", "lid") for i in range(4)},
)


def test_the_same_seed_gives_the_same_plan():
    ts, precedence = SCREWS_BEFORE_LID

    assert allocate_stations(ts, precedence, 4, random_seed=7) == allocate_stations(
        ts, precedence, 4, random_seed=7
    )


def test_salbp_e_chooses_the_k_with_least_k_times_cycle_time():
    # K=2: 2 x 4 s = 8, K=3: 3 x 2 s = 6, K=4: 8, K=5: 10.
    ts, precedence = SCREWS_BEFORE_LID

    result = choose_k_salbp_e(ts, precedence)

    assert_precedence_holds(result.plan, precedence)
    assert result.plan.n_stations == 3
    assert result.plan.cycle_time == 2.0
    assert [(e.n_stations, e.status, e.cycle_time) for e in result.sweep] == [
        (2, "optimal", 4.0),
        (3, "optimal", 2.0),
        (4, "optimal", 2.0),
        (5, "optimal", 2.0),
    ]
    assert result.complete


def test_salbp_e_keeps_the_smaller_k_on_a_tie():
    # One process, two 1 s screws: K=1 gives 1 x 2 s, K=2 gives 2 x 1 s.
    ts = tasks(("a", UNSCREW, 1.0), ("b", UNSCREW, 1.0))

    result = choose_k_salbp_e(ts, set())

    assert result.plan.n_stations == 1


def test_salbp_e_records_infeasible_k_and_goes_on():
    ts, precedence = SCREW_UNCLIP_SCREW

    result = choose_k_salbp_e(ts, precedence)

    assert_precedence_holds(result.plan, precedence)
    assert [(e.n_stations, e.status) for e in result.sweep] == [
        (2, "infeasible"),
        (3, "optimal"),
    ]
    assert result.plan.n_stations == 3


def test_a_used_up_budget_stops_the_sweep_after_the_first_k():
    ts, precedence = SCREWS_BEFORE_LID

    result = choose_k_salbp_e(ts, precedence, total_time_s=0.0)

    assert [e.n_stations for e in result.sweep] == [2]
    assert result.plan.n_stations == 2
    assert not result.complete


def test_plan_and_sweep_have_a_json_form():
    ts, precedence = SCREWS_BEFORE_LID

    data = json.loads(json.dumps(choose_k_salbp_e(ts, precedence).to_dict()))

    assert data["complete"] is True
    assert data["plan"]["n_stations"] == 3
    assert data["plan"]["n_stations_used"] == 3
    assert data["plan"]["cycle_time"] == 2.0
    assert data["plan"]["total_work"] == 6.0
    assert data["plan"]["utilization"] == 1.0  # 6 s of work / (3 stations x 2 s)
    assert {"station_id", "process", "parts", "total_time"} <= set(
        data["plan"]["stations"][0]
    )
    assert data["sweep"][1] == {
        "n_stations": 3,
        "status": "optimal",
        "cycle_time": 2.0,
        "efficiency": 1.0,
        "score": 6.0,
    }
