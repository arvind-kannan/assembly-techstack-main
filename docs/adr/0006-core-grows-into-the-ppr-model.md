# The core model grows into the PPR model

The MRTAMP app, its MRTAMP planners and the cell simulation must pass the same typed objects across their seams, and the cell simulation lives in another repo (`crc-disassembly-sim` on KIT GitLab, shipped here as the `crc_disassembly_sim/` subtree since #148). We put these contract types into the core dataclasses, so the core grows from the one definition of parts, connections and free directions into the **PPR model** (product, process, resource). This widens the scope of [ADR-0003](0003-core-model-change-process.md); its change process stays: every addition is a scoped `area:core-model` issue approved by a core maintainer and goes upstream via `git subtree push` ([ADR-0002](0002-sub-repos-as-git-subtrees.md)).

The full PPR model is sketched against the CRC ontology ([Pfr-22], DOI 10.1515/auto-2021-0156, and its machine-readable successor, the Circular Factory Core Ontology `cfc:` v0.9.0), but only the set that crosses the MRTAMP app / cell simulation seam is built. Process and capability follow the CRC: a *process* is a technology (unscrewing, inspecting, holding, ...), a *capability* is a resource's ability to carry out one process, a *resource* is the cell, a robot or a tool.

Decided in #92 (point 5) and #95.

## What the types are

- **Plain data.** Types reference parts and connections by name and need numpy only. Behaviour (feasibility, applying an operation to an assembly) stays in motion planning and the capabilities ([ADR-0004](0004-plan-function-is-the-motion-planning-seam.md)).
- **Process**: a closed enum in the core, each process paired with its parameter type. A new process is a small scoped core issue.
- **Resource**: cell (stations; no times since #174, see [ADR-0012](0012-cell-description-holds-no-times.md)), robot (id, NC channel, mountable tools, capabilities), tool (name, process served). Robot base poses, kinematics and zones stay in the cell simulation until calibrated. The one geometry the core carries is the station: a station has a kind (tool storage, supply, dropoff, fixture, machine, process module), a pose in the cell frame (x, y, z, rx, ry, rz in metres and radians, the angles about the fixed x, then y, then z axes, Robot 1's base at the origin), the processes it offers, coupling places with poses relative to the station and the interface each accepts, and, for the supply, supply slots with poses relative to the station (decided in #122, revised by #216). URDFs and any behaviour stay outside the core; the cell file links them by element name in a `descriptions` section.
- **Pallets and tooling** (revised by #216): the cell also lists its pallets (movable fixtures with a clamp) and its tooling (gripper jaws, clamping jaws, press tools), each with the interfaces it carries (SWS, finger interface, NSR-A, zero-point clamping); a tool lists the interfaces it grips. A *machine* has its own control and runs a cycle; a *process module* is operated by a robot.
- **Cell state** (holds, also of parts outside the assembly; mounted tools; since #216 also where each pallet and tooling is, which tooling is in which gripper or process module, and measured poses of moved stations, which override the nominal pose) and the **cell-assembly snapshot**, which pairs the assembly with the cell state and is the one object capabilities, MRTAMP planners and the cell simulation pass around. Holds do not count toward the product state.
- **Operation** (one generic type, one share per robot), **operation plan** (operations per robot, sync points, inspections, plan metrics, start snapshot) and the feedback types (operation outcome, infeasible operation, anomaly).
- Serialised with `to_dict`/`from_dict` to plain dicts of primitives and numpy arrays: pickle-safe on the simulation's wire, JSON for run records. The assembly travels as XML text.

## Considered options

- **Contract types in the MRTAMP app** (outside the core): the cell simulation would import the app, or keep its own copy that drifts.
- **Build the full PPR model now**: most of it has no consumer yet and would freeze guesses into the protected core.

## Consequences

- Robot holds move off `Assembly` into the cell state (resolves #15): the `Assembly` hold API is deprecated but keeps working until the capabilities and `operations.py` migrate to the snapshot, then it is removed.
- The order of the scoped issues is fixed in the spec (#98): process and resource types; cell state and snapshot; operation; operation plan and sync points; feedback types; migration of the capabilities (`area:decision-making`); removal of the hold API.
- Glossary terms for the new types stay in the contexts that use them (Decision making, Motion planning, MRTAMP app), not in the core model's glossary.
