"""Temporal Plan Graph (TPG) execution layer for the two-arm cell.

Offline, time-aware coordination in the spirit of APEX-MR (Huang et al. 2025)
built on Pellegrinelli et al. (2017)'s pairwise trajectory-safety filter:

  1. every arm's motion is fixed in advance as a sequence of SEGMENTS
     (transit, engage, pull-out, to-box, drop, retreat, ...), each with the
     collision spheres it sweeps (recorded during a solo rehearsal in the sim);
  2. pairs of segments of different arms are classified: a pair that cannot
     interfere whatever the timing needs nothing; a conflicting pair gets a
     DEPENDENCY EDGE  "arm B may not START segment j before arm A has FINISHED
     segment i";
  3. execution is asynchronous: each arm runs its own program at its own
     speed and only waits at edges.  Every edge is one rendezvous, i.e. one
     WAITM in the exported Sinumerik programs — the sim and the real cell run
     the same coordination.

Edge generators (rules) are pluggable.  The first one is the ZONE RULE: a
1.2 x 1.2 m "danger box" above the assembly (the clamping table's footprint,
or the same box centred on a floor-mounted assembly) may hold only one arm at
a time; the arms take turns, and a pair of in-zone segments that the sphere
test proves SAFE is exempt from the turn-taking.

This module is pure Python/numpy (no PyBullet) so the graph logic is testable
on its own; the simulator supplies segments, spheres and a driver.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

SIDES = ('C', 'D')


def other(side: str) -> str:
    return 'D' if side == 'C' else 'C'


@dataclass
class DangerZone:
    """Axis-aligned box: `half` metres around centre_xy in x/y, from z_min up."""
    centre_xy: Tuple[float, float]
    half: float = 0.6
    z_min: float = 0.0

    def contains_spheres(self, spheres: Optional[np.ndarray]) -> bool:
        """True if any sphere (x, y, z, r) of the sample overlaps the box."""
        if spheres is None or len(spheres) == 0:
            return False
        s = np.asarray(spheres, dtype=float)
        cx, cy = self.centre_xy
        r = s[:, 3]
        in_x = np.abs(s[:, 0] - cx) <= self.half + r
        in_y = np.abs(s[:, 1] - cy) <= self.half + r
        in_z = s[:, 2] + r >= self.z_min
        return bool(np.any(in_x & in_y & in_z))

    def contains_points(self, xyz) -> bool:
        pts = np.asarray(xyz, dtype=float).reshape(-1, 3)
        if len(pts) == 0:
            return False
        return self.contains_spheres(np.hstack([pts, np.zeros((len(pts), 1))]))


def sphere_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Minimum surface-to-surface distance between two sphere sets (x, y, z, r)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float('inf')
    d_min = float('inf')
    for i in range(0, len(a), 512):
        blk = a[i:i + 512]
        centres = np.linalg.norm(blk[:, None, :3] - b[None, :, :3], axis=2)
        gap = centres - blk[:, None, 3] - b[None, :, 3]
        d_min = min(d_min, float(gap.min()))
    return d_min


@dataclass
class Edge:
    """`wait_side` may not start its segment `wait_idx` before `until_side`
    has finished its segment `until_idx`."""
    wait_side: str
    wait_idx: int
    until_side: str
    until_idx: int
    why: str = ''

    def key(self):
        return (self.wait_side, self.wait_idx, self.until_side, self.until_idx)


class TemporalPlanGraph:
    """Two per-arm segment lists plus dependency edges between them.

    A segment is a dict with at least 'name'; 'spheres' (an (N, 4) array of
    the spheres it sweeps, world frame) enables the geometric rules, and the
    zone rule falls back to 'points' (an (N, 3) array of tool positions) or
    nothing (=> treated as inside, the conservative choice).
    """

    def __init__(self, segs_C: List[dict], segs_D: List[dict]):
        self.segs: Dict[str, List[dict]] = {'C': list(segs_C), 'D': list(segs_D)}
        self.edges: List[Edge] = []
        self.safe_pairs: set = set()        # ((C idx), (D idx)) proven SAFE by spheres
        self.log: List[str] = []

    # ------------------------------------------------------------------ rules
    def mark_zone(self, zone: DangerZone) -> None:
        for side in SIDES:
            for seg in self.segs[side]:
                if 'spheres' in seg and seg['spheres'] is not None:
                    seg['inside'] = zone.contains_spheres(seg['spheres'])
                elif 'points' in seg and seg['points'] is not None:
                    seg['inside'] = zone.contains_points(seg['points'])
                else:
                    seg['inside'] = True          # unknown geometry: assume it enters

    def inside_runs(self, side: str) -> List[Tuple[int, int]]:
        """Maximal runs [first_idx, last_idx] of consecutive in-zone segments."""
        runs, start = [], None
        for k, seg in enumerate(self.segs[side]):
            if seg.get('inside', True):
                if start is None:
                    start = k
            elif start is not None:
                runs.append((start, k - 1))
                start = None
        if start is not None:
            runs.append((start, len(self.segs[side]) - 1))
        return runs

    def classify_pairs(self, margin: float) -> None:
        """Pellegrinelli filter: a pair of segments whose swept spheres never
        come within `margin` is SAFE regardless of timing."""
        self.safe_pairs.clear()

        def bounds(sp):
            sp = np.asarray(sp, dtype=float)
            return sp[:, :3].min(axis=0) - sp[:, 3:4].max(), sp[:, :3].max(axis=0) + sp[:, 3:4].max()

        for i, sC in enumerate(self.segs['C']):
            if sC.get('spheres') is None:
                continue
            loC, hiC = bounds(sC['spheres'])
            for j, sD in enumerate(self.segs['D']):
                if sD.get('spheres') is None:
                    continue
                # Cheap bound first: if the two swept volumes' boxes are more
                # than `margin` apart, so is every sphere pair (a full physical
                # rehearsal holds ~1e5 spheres per arm: the exact product over
                # all 96x96 segment pairs would be 1e10 distances).
                loD, hiD = bounds(sD['spheres'])
                box_gap = float(np.max(np.maximum(loC - hiD, loD - hiC)))
                if box_gap > margin:
                    sC.setdefault('gaps', {})[j] = box_gap
                    self.safe_pairs.add((i, j))
                    continue
                gap = sphere_gap(sC['spheres'], sD['spheres'])
                sC.setdefault('gaps', {})[j] = gap
                if gap > margin:
                    self.safe_pairs.add((i, j))

    def ranks(self, first: str = 'C') -> Dict[Tuple[str, int], int]:
        """Turn order.  In-zone runs alternate C-run0, D-run0, C-run1, ...
        (starting with `first`); every segment gets the position of the last
        run of its own arm that starts at or before it (-1 before any run).
        Ranks are non-decreasing along each arm, so ordering every conflict
        from lower to higher rank can never create a cycle."""
        runs = {s: self.inside_runs(s) for s in SIDES}
        a, b = first, other(first)
        order = []
        for k in range(max(len(runs[a]), len(runs[b]))):
            if k < len(runs[a]):
                order.append((a, runs[a][k]))
            if k < len(runs[b]):
                order.append((b, runs[b][k]))
        rank = {}
        for s in SIDES:
            starts = [(r0, pos) for pos, (side, (r0, _)) in enumerate(order) if side == s]
            for k in range(len(self.segs[s])):
                rank[(s, k)] = max([pos for r0, pos in starts if r0 <= k], default=-1)
        self._order = order
        return rank

    def add_zone_rule(self, zone: DangerZone, first: str = 'C',
                      margin: Optional[float] = None) -> None:
        """Zone turn-taking + pairwise safety, as dependency edges.

        Conflicting pair = two segments of different arms that are NOT proven
        SAFE by their swept spheres (gap > margin); without spheres a pair
        conflicts only if both segments enter the box.  Every conflict is
        ordered by rank (the arm whose in-zone run comes first goes first):
        the later segment may not start before the earlier one has finished.
        """
        self.mark_zone(zone)
        if margin is not None:
            self.classify_pairs(margin)
        self._build_edges(self.ranks(first), first, need_spheres=False)

    def cycle_ranks(self, first: str = 'C') -> Dict[Tuple[str, int], int]:
        """Segment-level order from the balanced schedule: segment k of an arm
        belongs to cycle seg['cycle'] (its k-th screw); cycle c of `first`
        precedes cycle c of the other arm, which precedes cycle c+1.  The
        rank is non-decreasing along each arm => acyclic by construction."""
        rank = {}
        for s in SIDES:
            for k, seg in enumerate(self.segs[s]):
                rank[(s, k)] = 2 * int(seg.get('cycle', 0)) + (0 if s == first else 1)
        return rank

    def add_segment_rule(self, first: str = 'C', margin: float = 0.15) -> None:
        """APEX-MR style: no zone.  Every pair of segments of different arms
        whose swept spheres come within `margin` (or that has no spheres to
        judge by) gets a dependency edge, ordered by the schedule's cycle:
        C's cycle c, then D's cycle c, then C's c+1 ...  Segments that never
        come near each other — wherever they are — run in parallel."""
        self.classify_pairs(margin)
        for s in SIDES:
            for seg in self.segs[s]:
                seg.setdefault('inside', True)
        self._build_edges(self.cycle_ranks(first), first, need_spheres=True)

    def _build_edges(self, rank, first: str, need_spheres: bool) -> None:
        best: Dict[Tuple[str, int], Edge] = {}
        for i, sC in enumerate(self.segs['C']):
            for j, sD in enumerate(self.segs['D']):
                if (i, j) in self.safe_pairs:
                    continue
                have = sC.get('spheres') is not None and sD.get('spheres') is not None
                if not have and not need_spheres and                         not (sC.get('inside', True) and sD.get('inside', True)):
                    continue
                rc, rd = rank[('C', i)], rank[('D', j)]
                if rc == rd:                       # untied by the priority arm
                    rc, rd = (0, 1) if first == 'C' else (1, 0)
                w, wi, u, ui = ('D', j, 'C', i) if rc < rd else ('C', i, 'D', j)
                gap = sC.get('gaps', {}).get(j)
                why = (f'robot_{w} "{self.segs[w][wi]["name"]}" waits for robot_{u} '
                       f'"{self.segs[u][ui]["name"]}"'
                       + (f' (closest {gap*100:.0f} cm)' if gap is not None
                          else (' (no spheres)' if need_spheres else ' (both in the box)')))
                cur = best.get((w, wi))
                if cur is None or ui > cur.until_idx:
                    best[(w, wi)] = Edge(w, wi, u, ui, why)
        # An edge is implied when an EARLIER segment of the same arm already
        # waits for the same or a later segment of the other arm.
        for e in sorted(best.values(), key=lambda e: (e.wait_side, e.wait_idx)):
            implied = any(x.wait_side == e.wait_side and x.until_side == e.until_side
                          and x.wait_idx < e.wait_idx and x.until_idx >= e.until_idx
                          for x in self.edges)
            if not implied:
                self.add_edge(e)
        self.edges.sort(key=lambda e: (e.wait_side, e.wait_idx))

    def insert_segment(self, side: str, idx: int, seg: dict) -> None:
        """Insert a segment at `idx` of one arm (e.g. a live 'wait-out' or
        'return' move) and shift every edge index of that arm accordingly.
        Only ever done at the arm's CURRENT index, so finished indices are
        unaffected."""
        self.segs[side].insert(idx, seg)
        for e in self.edges:
            if e.wait_side == side and e.wait_idx >= idx:
                e.wait_idx += 1
            if e.until_side == side and e.until_idx >= idx:
                e.until_idx += 1

    def add_edge(self, e: Edge) -> None:
        if all(e.key() != x.key() for x in self.edges):
            self.edges.append(e)

    # -------------------------------------------------------------- queries
    def blocked_by(self, side: str, idx: int, finished: set) -> Optional[Edge]:
        for e in self.edges:
            if e.wait_side == side and e.wait_idx == idx and (e.until_side, e.until_idx) not in finished:
                return e
        return None

    def check_acyclic(self) -> bool:
        """Edges + each arm's own order must form a DAG, else execution deadlocks."""
        nodes = [(s, k) for s in SIDES for k in range(len(self.segs[s]))]
        succ = {n: [] for n in nodes}
        for s in SIDES:
            for k in range(len(self.segs[s]) - 1):
                succ[(s, k)].append((s, k + 1))
        for e in self.edges:
            succ[(e.until_side, e.until_idx)].append((e.wait_side, e.wait_idx))
        state = {n: 0 for n in nodes}

        def visit(n):
            state[n] = 1
            for m in succ[n]:
                if state[m] == 1 or (state[m] == 0 and not visit(m)):
                    return False
            state[n] = 2
            return True
        return all(state[n] or visit(n) for n in nodes)

    def describe(self) -> str:
        lines = [f'TPG: C {len(self.segs["C"])} segs, D {len(self.segs["D"])} segs, '
                 f'{len(self.edges)} edges, {len(self.safe_pairs)} SAFE pairs']
        for s in SIDES:
            lines.append(f'  robot_{s} in-zone runs: {self.inside_runs(s)}')
        lines += [f'  edge: {e.why}' for e in self.edges]
        lines += [f'  {l}' for l in self.log]
        return '\n'.join(lines)

    # ------------------------------------------------------------ execution
    def run(self, driver) -> bool:
        """Asynchronous execution against a driver:
            driver.start(side, idx)          load segment idx of that arm
            driver.step(side) -> bool        advance one step; True when the segment is finished
            driver.finished(side, idx)       segment done (on_done hooks)
            driver.wait(edge) -> bool        arm is blocked at an edge (called once); True if the
                                             driver inserted a move-out segment at the arm's index
            driver.release(edge)             edge satisfied, arm resumes (sync marker)
            driver.tick() -> bool            physics step(s); False => abort (contact)
        Returns True when both arms completed."""
        idx = {s: 0 for s in SIDES}
        n = lambda s: len(self.segs[s])          # segments may be inserted live (Plan B)
        started = {s: False for s in SIDES}
        waiting = {s: None for s in SIDES}
        finished = set()
        self.idx = idx
        while any(idx[s] < n(s) for s in SIDES):
            for s in SIDES:
                if idx[s] >= n(s):
                    continue
                if not started[s]:
                    e = self.blocked_by(s, idx[s], finished)
                    if e is not None:
                        if waiting[s] is None:
                            waiting[s] = e
                            if driver.wait(e):
                                # the driver inserted a move-out segment at idx:
                                # run it while still registered as waiting.
                                driver.start(s, idx[s])
                                started[s] = True
                        if not started[s]:
                            continue
                    else:
                        if waiting[s] is not None:
                            driver.release(waiting[s])   # may insert a return move at idx
                            waiting[s] = None
                        driver.start(s, idx[s])
                        started[s] = True
                if driver.step(s):
                    driver.finished(s, idx[s])
                    finished.add((s, idx[s]))
                    idx[s] += 1
                    started[s] = False
            if not driver.tick():
                return False
            if all(waiting[s] is not None for s in SIDES if idx[s] < n(s)) and \
                    all(idx[s] < n(s) for s in SIDES):
                # both blocked on each other: impossible for a DAG, guard anyway
                driver.deadlock(waiting)
                return False
        return True
