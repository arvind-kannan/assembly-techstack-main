"""Supply slots in the cell frame.

The core keeps a supply slot in its supply station, with a pose relative to the station (AT 174). Planners
and plan functions need the slot where it is in the cell: `slots_in_cell_frame` gives every slot of the cell
with its pose composed with its station's pose. The planning request carries these.

A slot's `robots` limits it to those robots: screwing in honours it; gripping at the supply does not yet (no
cell file limits a part slot).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from diss_assembly_dataclasses.resource import Cell, Pose, StationKind, SupplySlot
from scipy.spatial.transform import Rotation

# the core's pose convention: fixed-axis angles about x, then y, then z (roll, pitch, yaw)
EULER = "xyz"


def pose_in_cell_frame(station_pose: Pose, local_pose: Pose) -> Pose:
    """A pose relative to a station, as a pose in the cell frame."""
    station = Rotation.from_euler(EULER, station_pose[3:])
    position = np.asarray(station_pose[:3]) + station.apply(local_pose[:3])
    rotation = (station * Rotation.from_euler(EULER, local_pose[3:])).as_euler(EULER)
    return tuple(float(v) for v in (*position, *rotation))


def slots_in_cell_frame(cell: Cell) -> tuple[SupplySlot, ...]:
    """The supply slots of every supply station of `cell`, their poses in the cell frame."""
    return tuple(
        replace(slot, pose=pose_in_cell_frame(station.pose, slot.pose))
        for station in cell.stations
        if station.kind is StationKind.SUPPLY
        for slot in station.slots
    )


def slot_for_part(slots: tuple[SupplySlot, ...], part: str) -> SupplySlot | None:
    return next((s for s in slots if s.part == part), None)
