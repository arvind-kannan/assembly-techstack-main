# MRTAMP app

Planning disassembly or assembly of a product on the cell from its sequence graph with an MRTAMP planner, executing the plan on the real cell or a cell simulation, and replanning when an operation fails. The parts (MRTAMP planners, cell simulation, run controller) are library modules that researchers can also combine in their own scripts; the MRTAMP app is the ready-made GUI on top of them. Cell vocabulary (cell, robot, tool, skill, NC program) is defined in [Decision making](../decision-making/CONTEXT.md).

## Language

### Roles

**Cell operator**:
The person at the real cell who runs a plan, confirms what the run controller asks for, starts NC programs and, until it is automated, starts the cycle of a machine.
_Avoid_: user, worker

**Researcher**:
The person who runs plans in a cell simulation and compares MRTAMP planners. The same person may inspect products in the [Product explorer](../product-explorer/CONTEXT.md).
_Avoid_: experimenter, developer

### Planning

**MRTAMP app**:
The GUI in which the cell operator and the researcher use the run controller, MRTAMP planners and the cell simulation; a starting point, not the only way to use them.
_Avoid_: cell app, MRTAMP (alone), DASG app, HMI, app (for the components behind it)

**Run controller**:
The component that carries out a run: it asks an MRTAMP planner for an operation plan, sends it through the cell simulation and on to the real cell, reads the inspections, collects the feedback and replans. Used by the MRTAMP app and directly by researchers' code.
_Avoid_: executive, orchestrator, run loop, app

**MRTAMP planner**:
A multi-robot task and motion planner behind the MRTAMP seam: a library module that takes a planning request and returns an operation plan or reports that the goal is unreachable. A baseline ships with the techstack, theses add others.
_Avoid_: MRTAMP (alone), scheduler, solver

**Makespan**:
The estimated time from the start of an operation plan to the end of its last operation, including transits, tool changes, holding and inspections. One of the plan metrics every operation plan reports, beside each robot's busy time, the number of tool changes and the number of inspections; the baseline MRTAMP planner minimises it, other planners may choose their own objective. The cycle time of a line (the busiest station's total time, see [Sequence graph](../sequence-graph/CONTEXT.md)) is a different measure and is not used for a makespan.
_Avoid_: cost, cycle time, duration (alone)

**Ablation**:
A configuration of our own pipeline with one feature switched off (e.g. one robot instead of two, a given sequence instead of a planned one, parametric skills instead of a motion planner, a fixed product pose instead of an optimised one); the baseline a gain is measured against. Ablations form a ladder per layer (operation planning, motion planning, reachability), each rung switching on one more feature, so that a gain can be attributed to its layer.
_Avoid_: baseline (alone; the baseline MRTAMP planner is one rung), variant, feature flag

**Benchmark product**:
A product folder in the benchmark set, chosen for the ablation rungs it stresses (e.g. the battery replacement product for parallel screw groups, the extended angle grinder for reachability); it has an owner (whoever brought it into the project) and a target state (cell simulation only, or also the real cell).
_Avoid_: benchmark case, test product (the bolt-and-nut case is a test product, not a benchmark product)

**Planning request**:
What the run controller hands an MRTAMP planner, once at the start of a run and again for each replanning: the complete sequence graph and its product model (with the product folder it is read from), the cell and its travel-time table, the start cell-assembly snapshot, the run settings, the supply slots of the cell (poses in the cell frame), the feedback of the run so far (with the earlier operation plans its operation ids refer to) and, optionally, a cell simulation to send feasibility queries to. Replanning is a new planning request from a new snapshot. The planner answers with an operation plan or reports that the goal is unreachable; it keeps nothing between requests that the run controller relies on.
_Avoid_: planning problem, job, query (alone)

**Operation plan**:
The operations (see [Motion planning](../motion-planning/CONTEXT.md)) of one run, each assigned to the robots it needs, ordered per robot and linked across robots by sync points (robots within one operation act together without one); the output of an MRTAMP planner. Its estimated times are what the planner optimises; the cell does not execute them by the clock.
_Avoid_: cell plan, plan (alone), schedule, temporal plan graph (outside describing its structure)

**Step expansion**:
Turning one step of the sequence graph into the operations that carry it out on the cell, e.g. an inverse inserting step into gripping at the supply, an optional handover, inserting, holding where a part must be held (decided by a person or an algorithm; v1 holds nothing), and releasing. The techstack provides one as a library module; an MRTAMP planner may use it or expand on its own.
_Avoid_: decomposition, refinement, operation template

**Step catalogue**:
The steps of the sequence graph an MRTAMP planner may choose from a product state for a robot, each with its cost after the feedback of the run so far: steps the robot cannot carry out are left out, an infeasible step is excluded, a failed step costs more with every failure and is excluded past the retry limit, and a tool change adds its time. The techstack provides one as a library module next to step expansion; an MRTAMP planner may use it or choose on its own.
_Avoid_: candidate list, step filter, action space (reserved for the POMDP)

**Operation-plan builder**:
What records the steps an MRTAMP planner chose and finishes them into an operation plan: the planner commits each step to a robot, optionally waiting for an earlier step (an order sync point), and may ask for the robots' clocks, the estimated start and end of a committed step and the mounted tools; finishing places the inspections and reports the plan metrics of the plan's operations. The step catalogue answers what a planner can do next, the builder records what it chose. The techstack provides one as a library module next to the step catalogue; an MRTAMP planner may use it or build its operation plan on its own.
_Avoid_: scheduler, plan assembler

**Sync point**:
A place in an operation plan where one robot waits, before one of its operations or transits, until another robot has finished a given operation or transit; it records whether the MRTAMP planner or the cell simulation set it and why (order, handover, holding, paths too close). The MRTAMP planner sets the task-level ones (order from the sequence graph, handovers, the end of a holding operation); the cell simulation adds motion-level ones where the robots' paths come too close, and a motion planner may set phase marks inside an operation whose robots move together (until the cell runs such motion synchronously). On the cell each becomes a `WAITM` mark in both NC programs.
_Avoid_: barrier, handshake, wait mark (for the concept)

**Supply slot**:
A place in the cell's supply station (a pose relative to the station) that serves one part or screw type; gripping at the supply grips the part there. A slot may name the robots it serves (none named: every robot); screwing in fetches its screw from a slot of its screw type that serves its robot. Step expansion needs one for every inserted part.
_Avoid_: feeder, magazine

**Concurrent group**:
The operations of an operation plan that run at the same time, at most one per robot, starting from one cell-assembly snapshot (see [Decision making](../decision-making/CONTEXT.md)). A single operation with the other robots idle is a concurrent group too.
_Avoid_: step (reserved for the sequence graph), plan slice, parallel operations

**Feasibility query**:
A question an MRTAMP planner may ask the cell simulation while planning: can this concurrent group be carried out from its cell-assembly snapshot? The query carries the product model of the planning request, as a simulation request does, because the snapshot lacks the parts an assembly run inserts next; it also carries the cell's travel-time table, from which the cell simulation takes its tool-change times until #235. The answer is feasible or infeasible with a reason, plus estimated durations. It checks the robots' paths against each other, but executes nothing and produces no NC programs.
_Avoid_: dry run (that is the whole operation plan), check (alone), validation

**Run**:
One attempt to disassemble or assemble one product instance, from the first operation plan until the goal is reached or the run is aborted. A run targets either the cell simulation, and ends there, or the real cell, which it reaches only through the cell simulation. Replanning can replace the operation plan several times within one run.
_Avoid_: episode (reserved for the POMDP), job, execution (for the whole attempt)

**Goal**:
The product state a run must reach: the parts that must end present and the parts that must end absent. The goal has no direction of its own; each step's direction follows from comparing the start cell-assembly snapshot with the goal, so one run may remove and insert parts. It may also name **part conditions**: for a part, the rework processes it must have received or that it must be replaced by a spare; such a part must end present, so a remanufacturing run removes it, reworks or replaces it and inserts it again. While a condition is unmet, any part may be removed; once all are met, steps insert only the parts that must end present. A part of another version is another part (absent old part, present new one), not a condition. A present part ends in its product-model pose. The app builds the goal from the product setup (its target parts as the absent parts, its part conditions) when the person sets none; a planner never guesses a goal.
_Avoid_: direction, target (the run's cell-or-simulation choice is the target of the run settings)

**Run settings**:
What the person chooses for one run beside the product: whether it targets the cell simulation or the real cell, the goal (see **Goal**), the risky processes, whose operations must be inspected before anything that depends on them goes on (by default all processes that change the product), and the retry limit: how often a failed step may be tried again before it is excluded (default 1).
_Avoid_: config, scenario, product setup (that belongs to the product)

### Executing

**Cell simulation**:
The simulated cell every operation plan passes through: it plans the robots' motions, executes the operation plan virtually and produces the NC programs the real cell runs up to the next inspection. It sits before the real cell, not beside it, and never falls back to unchecked motion: what it cannot plan it reports as an infeasible operation.
_Avoid_: sim, DisassemblySim (for the concept), digital twin

**Planner worker**:
A motion planner of the cell simulation that runs as its own process, because it needs a GPU or Linux. For each operation the cell simulation sends it a scene description: obstacles, held parts and each robot's start joint configuration. Light planners run inside the cell simulation and need no planner worker.
_Avoid_: planner server, GPU service, sim server (that is the cell simulation itself)

**Operation outcome**:
Whether an executed operation reached its result assembly: *succeeded* or *failed*. On the cell an inspection determines it; later the skills will also report it as IO/NIO.
_Avoid_: result (alone), status, IO/NIO (for the concept)

**Inspection**:
An operation in the operation plan in which Robot 2 scans the product instance with the Zivid camera and the run controller compares the observed geometric state with the expected assembly, giving the operation outcomes of the operations before it. The MRTAMP planner places inspections where an outcome must be known before the run goes on. Its process is *inspecting*, and it involves every robot of the cell (Robot 2 scans, the others hold still), so like any operation with several robots it needs no sync point of its own. Both NC programs end at every inspection; the run controller then either has the rest of the operation plan re-simulated from the inspected snapshot or replans.
_Avoid_: scan (for the operation), perception check, seating check (that is one kind of inspection)

**Machine operation**:
An operation in the operation plan in which a machine (see [Decision making](../decision-making/CONTEXT.md)) carries out its cycle, with the rework process it applies (e.g. rework milling of the plate gear in the DMG DMU 60) and an estimated cycle time. A robot loads the machine before it and unloads it after it. Both NC programs end at every machine operation as at an inspection; the run controller asks the cell operator to start the machine's cycle and to confirm that it is done, then sends the rest of the same operation plan through the cell simulation again. In the cell simulation it is a wait of the cycle time.
_Avoid_: machine job, mill run

**Failed operation**:
An operation that was executed on the cell or in a cell simulation but did not reach its result assembly; it triggers replanning. Unlike a *rejected action* (see [Decision making](../decision-making/CONTEXT.md)) it was started and may have changed the product instance.
_Avoid_: failed step, error, NIO

**Injected failure**:
A failed operation the researcher forces in a run that targets the cell simulation, so that replanning can be tested without the real cell.
_Avoid_: fault, mock failure

**Infeasible operation**:
An operation of an operation plan that motion planning or the cell simulation cannot carry out as planned (no motion found, collision, out of reach, sync deadlock). Unlike a failed operation it never ran on the real cell, so the product instance is unchanged; it triggers replanning from the same product state.
_Avoid_: failed operation (for this), rejected action, error

**Anomaly**:
Something unexpected while executing an operation that is not a failed operation, e.g. a time overrun or a torque outside its band while the skill reports IO. It is logged and shown to the cell operator but does not trigger replanning by itself.
_Avoid_: warning, error, failure

**Simulation report**:
What the cell simulation returns for an operation plan: for each operation whether it is feasible (or the first infeasible operation with its reason), the motion-level sync points it added, the estimated cell times of every operation and transit, the observed snapshot at each inspection it reached, operation outcomes and anomalies, and, for a run that targets the real cell, the NC programs up to and including the next inspection.
_Avoid_: result, response, sim output

**Re-simulation**:
Sending the rest of the same operation plan through the cell simulation again from the cell-assembly snapshot an inspection observed, when the inspection found what was expected, so that motions and NC programs use the measured part poses. The MRTAMP planner is not asked again.
_Avoid_: replanning (that makes a new operation plan), retry

**Replanning**:
Making a new operation plan within a run, from the observed product state and cell state, after a failed or infeasible operation.
_Avoid_: recovery, retry (for the whole replan)
