from __future__ import annotations

from enum import Enum


class Process(str, Enum):
    """The closed list of processes an operation can apply (the CRC *process*)."""

    GRIPPING = "gripping"
    REMOVING = "removing"
    GRIPPING_AND_REMOVING = "gripping_and_removing"
    UNSCREWING = "unscrewing"
    MILLING = "milling"
    INSERTING = "inserting"
    SCREWING_IN = "screwing_in"
    RELEASING = "releasing"
    HOLDING = "holding"
    INSPECTING = "inspecting"
    # remanufacturing: rework in a machine, and moving pallets and tooling (cell state only)
    REWORK_MILLING = "rework_milling"
    LOADING = "loading"
    UNLOADING = "unloading"
    RECONFIGURING = "reconfiguring"
