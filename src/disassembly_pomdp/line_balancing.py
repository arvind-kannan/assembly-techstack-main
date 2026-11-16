"""Line balancing: assign the steps of a disassembly to the stations of a line (CP-SAT).

Ported from Jan's `allocate_stations` and `choose_k_salbp_e` (#133, #136). Input is the output of
`graph_precedence.extract_precedence`: the tasks and the precedence relation. K stations stand in line
order; every station does one process; for each precedence pair (p, q), p's station is not later in the
line than q's. Phase 1 minimises cycle time, phase 2 fixes it and minimises the stations used.

Algorithm by Jan (`line_balance_disassembly.py`, received as a zip in 2026-10); the licence is open, ask Jan
before this code is made public outside the KIT GitLab.

Not modeled (as in Jan's file): tool change time between steps of the same process, and blocking
between stations (cycle time = slowest station, enough buffering assumed).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import networkx as nx
from ortools.sat.python import cp_model

from disassembly_pomdp.graph_precedence import Task

_TIME_SCALE = 1000  # seconds to milliseconds; CP-SAT is integer-only


class InfeasibleLineError(ValueError):
    """No assignment of the steps to the stations exists (proven by the solver)."""


@dataclass(frozen=True)
class Station:
    """A used station: its position in the line (1-based, gaps where stations stay empty)."""

    station_id: int
    process: str
    parts: list[
        str
    ]  # the target parts of its steps, in an order the precedence relation allows
    total_time: float


@dataclass(frozen=True)
class LinePlan:
    n_stations: int  # K offered
    stations: list[Station]  # used stations only, in line order
    cycle_time: float
    cycle_time_optimal: bool
    stations_used_optimal: bool

    @property
    def n_stations_used(self) -> int:
        return len(self.stations)

    @property
    def total_work(self) -> float:
        return sum(s.total_time for s in self.stations)

    @property
    def utilization(self) -> float:
        """Total work / (K offered x cycle time), as in Jan's resource utilization."""
        return (
            self.total_work / (self.n_stations * self.cycle_time)
            if self.cycle_time > 0
            else 0.0
        )

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "n_stations_used": self.n_stations_used,
            "total_work": self.total_work,
            "utilization": self.utilization,
        }


def allocate_stations(
    tasks: dict[str, Task],
    precedence: set[tuple[str, str]],
    n_stations: int,
    *,
    time_limit_s: float = 10.0,
    num_workers: int = 1,
    random_seed: int = 0,
) -> LinePlan:
    """The plan with minimal cycle time on `n_stations` stations and, subject to that, fewest stations used.

    `time_limit_s` applies to each of the two phases. With one worker (the default) and a fixed
    `random_seed`, a solve that finishes within the limit gives the same plan on every run; more workers
    are faster on hard cases but may return a different plan of equal cycle time. A solve cut short by the
    limit returns the best plan found so far, which depends on the machine's speed.

    Raises `ValueError` for fewer stations than processes, `InfeasibleLineError` if no assignment exists
    (too few stations for the process order, or a cycle in the precedence relation), `TimeoutError` if
    no plan is found within the limit. If only phase 2 is cut short, the phase-1 plan is kept and
    `stations_used_optimal` is False.
    """
    task_list = list(tasks.values())
    n = len(task_list)
    idx_of_part = {t.target_part: i for i, t in enumerate(task_list)}
    processes = sorted({t.process for t in task_list})
    if n_stations < len(processes):
        raise ValueError(
            f"need at least {len(processes)} stations (one per process {processes}), got {n_stations}"
        )
    process_idx = {p: i for i, p in enumerate(processes)}
    durations = [round(t.action_time * _TIME_SCALE) for t in task_list]
    total_time = sum(durations)
    K = n_stations

    model = cp_model.CpModel()
    assign = {
        (i, k): model.NewBoolVar(f"assign_{i}_{k}") for i in range(n) for k in range(K)
    }
    for i in range(n):
        model.AddExactlyOne(assign[i, k] for k in range(K))

    station_process = [
        model.NewIntVar(0, max(len(processes) - 1, 0), f"process_{k}") for k in range(K)
    ]
    for i, t in enumerate(task_list):
        for k in range(K):
            model.Add(station_process[k] == process_idx[t.process]).OnlyEnforceIf(
                assign[i, k]
            )

    pos = [model.NewIntVar(0, K - 1, f"pos_{i}") for i in range(n)]
    for i in range(n):
        model.Add(pos[i] == sum(k * assign[i, k] for k in range(K)))
    for p, q in precedence:
        model.Add(pos[idx_of_part[p]] <= pos[idx_of_part[q]])

    loads = [sum(assign[i, k] * durations[i] for i in range(n)) for k in range(K)]
    cycle_time = model.NewIntVar(0, total_time, "cycle_time")
    for k in range(K):
        model.Add(cycle_time >= loads[k])

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = num_workers
    solver.parameters.random_seed = random_seed

    # Phase 1: minimal cycle time.
    model.Minimize(cycle_time)
    status = solver.Solve(model)
    if status == cp_model.INFEASIBLE:
        raise InfeasibleLineError(
            f"no assignment of {n} steps to {K} stations satisfies the precedence relation and one process "
            f"per station; offer more stations or check the precedence relation for a cycle"
        )
    if status == cp_model.UNKNOWN:
        raise TimeoutError(
            f"no station assignment found within {time_limit_s} s; raise time_limit_s"
        )
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(
            f"CP-SAT failed on the line-balancing model (status {solver.StatusName(status)})"
        )
    cycle_time_optimal = status == cp_model.OPTIMAL
    best_cycle_time = solver.Value(cycle_time)
    plan_values = _assignment(solver, assign, n, K)

    # Phase 2: cycle time fixed, fewest stations used.
    model.Add(cycle_time <= best_cycle_time)
    used = [model.NewBoolVar(f"used_{k}") for k in range(K)]
    for k in range(K):
        model.Add(loads[k] <= total_time * used[k])
    model.Minimize(sum(used))
    status2 = solver.Solve(model)
    stations_used_optimal = status2 == cp_model.OPTIMAL
    if status2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        plan_values = _assignment(solver, assign, n, K)
    # Extension point: Jan's phase 3 broke ties by the distance between toolpaths of steps sharing a
    # station; our steps carry no toolpath, so it is not ported. A further tie-break goes here.

    stations = []
    for k in range(K):
        indices = [i for i in range(n) if plan_values[i] == k]
        if not indices:
            continue
        order = nx.DiGraph()
        order.add_nodes_from(task_list[i].target_part for i in indices)
        order.add_edges_from((p, q) for p, q in precedence if p in order and q in order)
        stations.append(
            Station(
                station_id=k + 1,
                process=task_list[indices[0]].process,
                parts=list(nx.lexicographical_topological_sort(order)),
                total_time=sum(durations[i] for i in indices) / _TIME_SCALE,
            )
        )
    return LinePlan(
        K,
        stations,
        best_cycle_time / _TIME_SCALE,
        cycle_time_optimal,
        stations_used_optimal,
    )


def _assignment(solver: cp_model.CpSolver, assign, n: int, K: int) -> list[int]:
    """The station index of every task in the solver's current solution."""
    return [next(k for k in range(K) if solver.Value(assign[i, k])) for i in range(n)]


@dataclass(frozen=True)
class SweepEntry:
    """One K tried by the SALBP-E sweep; status is optimal, feasible, infeasible or timeout."""

    n_stations: int
    status: str
    cycle_time: float | None = None
    efficiency: float | None = None  # total work / (K x cycle time)
    score: float | None = None  # K x cycle time


@dataclass(frozen=True)
class SalbpEResult:
    plan: LinePlan  # the plan of the chosen K
    sweep: list[SweepEntry]
    complete: bool  # False if the time budget stopped the sweep or some K timed out

    def to_dict(self) -> dict:
        return {
            "plan": self.plan.to_dict(),
            "sweep": [asdict(e) for e in self.sweep],
            "complete": self.complete,
        }


def choose_k_salbp_e(
    tasks: dict[str, Task],
    precedence: set[tuple[str, str]],
    *,
    max_stations: int | None = None,
    time_limit_s: float = 10.0,
    total_time_s: float | None = None,
    num_workers: int = 1,
    random_seed: int = 0,
) -> SalbpEResult:
    """Choose the number of stations K by SALBP-E: minimise K x cycle time, the smaller K on ties.

    Sweeps K from the number of processes up to `max_stations` (default: the number of steps; beyond it
    some station stays empty, so K x cycle time only grows). Each K is solved with `allocate_stations`;
    an infeasible K is recorded and skipped. Worst case without a budget: (number of K) x 2 phases x
    `time_limit_s`. With `total_time_s`, the first K always runs and later K start only while the budget
    lasts, each with min(`time_limit_s`, budget left); the result is then marked incomplete.
    """
    lo = max(len({t.process for t in tasks.values()}), 1)
    hi = max_stations if max_stations is not None else len(tasks)
    deadline = None if total_time_s is None else time.monotonic() + total_time_s

    sweep: list[SweepEntry] = []
    best: LinePlan | None = None
    complete = True
    for k in range(lo, hi + 1):
        limit = time_limit_s
        if deadline is not None and k > lo:
            left = deadline - time.monotonic()
            if left <= 0:
                complete = False
                break
            limit = min(limit, left)
        try:
            plan = allocate_stations(
                tasks,
                precedence,
                k,
                time_limit_s=limit,
                num_workers=num_workers,
                random_seed=random_seed,
            )
        except InfeasibleLineError:
            sweep.append(SweepEntry(k, "infeasible"))
            continue
        except TimeoutError:
            sweep.append(SweepEntry(k, "timeout"))
            complete = False
            continue
        score = k * plan.cycle_time
        sweep.append(
            SweepEntry(
                k,
                "optimal" if plan.cycle_time_optimal else "feasible",
                plan.cycle_time,
                plan.utilization,
                score,
            )
        )
        if best is None or score < best.n_stations * best.cycle_time - 1e-9:
            best = plan

    if best is None:
        if not complete:
            raise TimeoutError(
                f"SALBP-E found no plan within the time limits (sweep: {sweep})"
            )
        raise InfeasibleLineError(
            f"SALBP-E found no plan for any K in [{lo}, {hi}] (sweep: {sweep})"
        )
    return SalbpEResult(best, sweep, complete)
