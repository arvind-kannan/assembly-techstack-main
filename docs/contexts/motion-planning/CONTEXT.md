# Motion planning

Turning an action into a geometric, executable operation (direction, path, duration) and checking that the robot can carry it out. The entry point for motion-planner work is the plan function ([ADR-0004](../../adr/0004-plan-function-is-the-motion-planning-seam.md)).

## Language

**Operation**:
The geometric, executable plan that carries out one action on an assembly with the resources it needs (one or more robots, their tools): per robot a path and duration, plus the removal or insertion direction and the resulting assembly. The robots of one operation act at the same time. On the cell, each robot carries out its share by calling a skill with the operation's values as parameters (see *Skill* in [Decision making](../decision-making/CONTEXT.md)). This is the CRC *operation* (a task on resources) narrowed to one action. The operations of a run, ordered and synchronised, form an *operation plan* (see [MRTAMP app](../mrtamp-app/CONTEXT.md)). An operation that carries out part of a step of the sequence graph names that step, so an outcome reported for the operation leads back to the step.
_Avoid_: step, task, motion, skill call

**Share**:
One robot's part of an operation: its tool, path, path duration and skill parameters. An operation with one robot has one share; a handover or an inspection has one per robot involved. Each share becomes NC commands in that robot's NC program: skill calls (preferred), Cartesian TCP moves with a fixed arm posture, or, rarely, joint moves.
_Avoid_: role, subtask, part (reserved for the product)

**Plan function**:
The component that fills in an operation for a given assembly, including the path of each of its shares, and returns it, optionally with the result assembly. Each process has its own plan function; inserting and screwing in have their own and are not inverted removal operations, even where they reuse removal geometry. A plan function that finds no plan for its share raises `PlanFailure`; the MRTAMP planner then marks the operation with the reason and that share's robot, and the cell simulation reports it as infeasible. Any other exception is a bug and propagates.
_Avoid_: planner (alone), motion policy

**Removal plan core**:
The one place that judges a removal plan (unscrewing, gripping and removing): whether it failed and why, the unit removal direction (falling back to the part's degrees of freedom) and the result assembly, present only if the part is really gone. The gym's capabilities and the MRTAMP plan functions only translate its result ([ADR-0004](../../adr/0004-plan-function-is-the-motion-planning-seam.md)).
_Avoid_: removal adapter, plan adapter

**Motion planner**:
The component in the cell simulation that turns one operation into robot motion: from the robots' start configurations and the cell-assembly snapshot it returns the NC commands of every share, and for an operation whose robots move together (coupled motion) it may set phase marks between them. Selected once per cell-simulation config; what it cannot plan, it reports as an infeasible operation ([ADR-0010](../../adr/0010-cell-simulation-planner-plans-one-operation-and-returns-nc-commands.md)).
_Avoid_: planner (alone), MRTAMP planner (that is the task layer), plan function (that plans paths, not motion)

**Removal direction**:
The one direction an operation actually moves a part in, chosen from the part's free directions.
_Avoid_: pull direction, DOF (for a single direction)

**Insertion direction**:
The one direction an inserting operation moves a part in: the reverse of one of the part's free directions in the result assembly.
_Avoid_: push direction, joining direction

**Path**:
The sequence of tool poses an operation follows relative to the parts: approach, the process motion itself (removal, insertion, screw-in, cut) and retract. Planned by the operation's plan function; the cell simulation runs it unchanged. Poses are in metres, like every pose the cell simulation reads; the plan functions convert the product model's millimetres.
_Avoid_: trajectory, toolpath (outside milling), process path

**Transit**:
A robot's motion between two of its operations (from one path's end to the next path's start, or to the tool storage, supply or dropoff). Planned by the cell simulation, not by plan functions, against the other robots' motions.
_Avoid_: travel, free motion, path (for this)

**Reference part**:
The part an operation's poses are given relative to, usually the base part in the fixture. On the cell, its pose measured by an inspection places the operation; the operation itself is not replanned.
_Avoid_: anchor part, frame part

**Path duration**:
The time the robot needs to follow the path.
_Avoid_: cost (reserved for the planner search graph)

**Result assembly**:
The assembly as it is after the operation succeeded.
_Avoid_: next state, follow assembly

**Feasibility check**:
A global test that the operation can really be executed (e.g. collision-free sweep along the path, reachability), beyond the local free directions.
_Avoid_: validity check, exec gate

**Visibility check**:
A test whether a part can be seen from a set of viewing directions.
_Avoid_: line-of-sight check

**Known gap**:
Free, removal and insertion directions are translations only; rotations and screw motions are not modelled.
