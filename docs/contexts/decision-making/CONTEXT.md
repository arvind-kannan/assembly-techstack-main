# Decision making

The disassembly POMDP: which actions a robot cell can take, how hidden end-of-life conditions make their outcome uncertain, what the cell observes, and how a policy chooses the next action.

The assembly capabilities (*Inserting*, *Screwing in*, *Releasing*) and the skills that execute capabilities on the cell are defined here too, so both directions share one vocabulary. The env does not implement the assembly capabilities yet (#31).

## Language

### Cell

**Cell**:
The robot workcell as it is set up: its robots (any number), its tools, its stations (tool storage, supply, dropoff, fixture, machine, process module), its pallets and tooling, and its other resources, each with its capabilities and its pose. It holds no times: travel and tool-change times come from the travel-time table, process parameters such as feed rates from the skills. It describes the cell, not what is happening in it at a moment (that is the cell state).
_Avoid_: environment (for the physical cell); station (for the whole cell)

**Station**:
A named place of the cell with a kind (tool storage, supply, dropoff, fixture, machine or process module), a pose in the cell frame (metres and radians, Robot 1's base at the origin), the processes it offers (only machines and process modules offer any) and its coupling places. Plain data: the geometry that draws it (a URDF) is linked by name in the cell file, outside the core. The supply is a station; only it has supply slots. The cell file holds the nominal pose; a station that was moved, e.g. a process module, has a measured pose in the cell state that overrides it. It is not the Station of a line in line balancing (see [Sequence graph](../sequence-graph/CONTEXT.md)): that is a place in a line carrying out the steps of one process.
_Avoid_: module (alone), workstation, special machinery

**Resource**:
Anything in the cell that carries out or supports an operation: the cell itself, its robots and tools. Each resource has capabilities. The CRC *resource*.
_Avoid_: equipment, asset

**Robot**:
A physical arm in the cell that can hold at most one part. The cell knows which tools it can mount and which capabilities it has.
_Avoid_: agent, arm, manipulator

**Busy**:
A robot that holds a part and cannot take another action until it releases it.
_Avoid_: locked, occupied

**Cell state**:
Which part each robot holds and which tool each robot has mounted, where each pallet and each tooling is (on a coupling place or held by a robot), which tooling is in which gripper or process module, and the measured pose of any station that was moved, at one moment; kept apart from the product state (see [Sequence graph](../sequence-graph/CONTEXT.md)). A held part may be outside the assembly, e.g. carried to the dropoff or picked at the supply. Shared by the POMDP and the [MRTAMP app](../mrtamp-app/CONTEXT.md).
_Avoid_: robot state, world state, agents

**Cell-assembly snapshot**:
The assembly together with the cell state at one moment, real (after an inspection) or planned; the one picture capabilities, MRTAMP planners and the cell simulation pass around. Who holds a part is looked up here, not on the assembly.
_Avoid_: snapshot (alone), situation, world state, scene

**Holder**:
The robot that currently holds a part; part of the cell state.
_Avoid_: agent, owner

**Active grasp**:
A holder together with how it holds the part: grasp, gripper opening and tool centre point offset.
_Avoid_: grip, pick

**Tool**:
The end effector mounted on a robot through the SWS, e.g. a screwdriver or the Pizzagreifer. For unscrewing and screwing in, the tool name equals the screw head it drives. A tool that grips items lists the interfaces it grips through. A tool is not tooling.
_Avoid_: gripper (for tools in general), end effector

**Tool change**:
Swapping the mounted tool at the tool storage, through the SWS. Changing a tooling is reconfiguring, not a tool change.

**Travel-time table**:
The nominal times of moving a robot between the stations of the cell and its home pose, including the time spent at the tool storage for a tool change. Set once per cell: by hand, estimated from the station poses, or recorded from cell simulation runs. A cheap look-up for the POMDP, the sequence graph and MRTAMP planners; the cell simulation computes its own times. Kept in its own file next to the cell file, under the cell's fingerprint. Until #235 it holds the flat keys of the former cell settings (feed rates and offsets included), and the cell simulation reads its tool-change and inspection times from it (ADR-0012, interim note).
_Avoid_: cell settings, timing settings

**Dropoff**:
The place where removed parts are put down.

**Supply**:
The place where parts for assembly are provided, e.g. the screw nest plate; the counterpart of the dropoff.
_Avoid_: feeder, magazine (unless it really feeds automatically)

**Supply slot**:
One place in the supply station with its pose (relative to the station) and the part or screw type it provides; part of the cell, taught on the real cell. Gripping at the supply and screwing in pick from a supply slot. A supply slot may name the robots it serves (a screw nest plate per robot in the simulation's cell file); a slot that names none serves every robot.
_Avoid_: nest, pocket (except as the physical feature)

**Base part**:
A part that stays in the fixture and is never removed by gripping and removing, e.g. the EMO motor's lower lid and stator.
_Avoid_: fixed part, ground part

**Fixture**:
The device that holds the base of an assembly at a fixed pose in the cell while parts are inserted and screwed in, e.g. the printed cradle for the angle grinder.
_Avoid_: jig, holder (the holder is a robot)

**Pallet**:
A fixture that a robot carries between stations, with a clamp for the part on it, e.g. to set the gearbox lid subassembly into the DMG DMU 60 and take it out again. It carries the NSR-A interface, by which the Pizzagreifer grips it, and zero-point clamping towards the machine. When not in use it stands on a coupling place of the pallet storage next to the cell door. Where it is now is part of the cell state.
_Avoid_: tool plate, Palette, vice (for the pallet)

**Clamp**:
The device on a pallet that holds a part, e.g. the clamp the gearbox lid subassembly is screwed into.
_Avoid_: vice, Spanner

**Machine**:
A station with its own control that carries out a rework process in a cycle the cell starts and waits for (rework milling on the DMG), e.g. the DMG DMU 60 evo linear (Siemens) or the Ponticon pE3D (Beckhoff). A robot loads and unloads it; today the cell operator starts its cycle.
_Avoid_: special machinery, process module (that is robot-operated), Fräsmodul

**Process module**:
A station without its own control that a robot operates, e.g. the bearing press (the robot turns its crank) or the clamping desk. It may keep its own tooling, as the press keeps its press tools. It may be moved in the cell; its measured pose is then in the cell state.
_Avoid_: machine (that has its own control), module (alone)

**Coupling place**:
A named place of a station (a pose relative to the station) where an item is set down and coupled through one interface, e.g. the zero-point receiver on the DMG table, a press tool seat or store in the press, a place in the finger storage, or the pallet's place in the pallet storage. Robot tools in the tool storage sit on coupling places too.
_Avoid_: slot (that is the supply slot), nest, receiver (alone)

**Interface**:
A mechanical coupling through which an item is gripped or seated. The cell has four: the SWS (robot tools on the robot and in the tool storage), the finger interface (a custom design for gripper jaws and clamping jaws), the Schunk NSR-A (pallets and press tools) and zero-point clamping (a pallet on the zero-point receiver of the DMG table). A tool lists the interfaces it grips, an item those it carries, and a coupling place the one it accepts.
_Avoid_: adapter, quick changer (alone)

**Tooling**:
An exchangeable insert of a gripper, the clamping desk or a process module: gripper jaws and clamping jaws (finger interface, kept in the finger storage), the small clamping system of the clamping desk, the press tools (NSR-A, kept in the press). Where each tooling is and which tooling is in which gripper or module is part of the cell state. Not a tool: a tool is mounted on a robot.
_Avoid_: tool (for tooling), fingers (for all tooling), module tooling

**Pizzagreifer**:
The pneumatic gripper tool of Robot 1 that grips items through the NSR-A: it carries pallets (loading and unloading) and press tools (reconfiguring). Its jaws are on valve 5 of Robot 1's valve island.
_Avoid_: pizza gripper, pallet gripper, process module quick changer

### Acting

**Process**:
A technology an operation applies, independent of the resource that carries it out and of the product: gripping, removing, gripping-and-removing, unscrewing, milling; for rework rework milling; for assembly inserting, screwing in, releasing; holding; for the cell state alone reconfiguring, loading and unloading; and inspecting (see *Inspection* in the [MRTAMP app](../mrtamp-app/CONTEXT.md)). The CRC *process*.
_Avoid_: operation type, capability (for the technology itself)

**Capability**:
A resource's ability to carry out one process, e.g. Robot 2 can inspect, the cell can unscrew. The cell's capabilities each come with their own set of actions. Independent of how it is executed; on the cell a capability is carried out by one or more skills. The CRC *capability*.
_Avoid_: module, skill family

**Action**:
One decision of the policy: a capability, a tool and a target.
_Avoid_: move, command, step (reserved for the sequence graph)

**Target**:
The part, connection or buffer slot an action is applied to.

**Gripping**:
A robot grasps a part so that it holds it, either in the assembly (the part stays in place) or at the supply.
_Avoid_: picking, manipulating

**Removing**:
A held part is pulled out of the assembly along one of its free directions.
_Avoid_: pulling, extracting

**Gripping-and-removing**:
Gripping and removing done as one action.
_Avoid_: manipulating

**Inserting**:
A held part is moved into the assembly along its insertion direction; the inverse of removing. Guided fits (shaft into bore, pins into holes) and resting fits (lid onto a housing) are the same capability, executed by different skills.
_Avoid_: placing (as a capability; fine as a skill name), attaching, joining

**Holding**:
A robot keeps a part it holds at its pose, without moving, while other robots carry out their operations; it lasts until a given operation of another robot has finished. Like any operation, it occupies the robot. Who holds the part is recorded in the cell state either way; the holding operation makes the occupied time visible in the operation plan.
_Avoid_: fixing, clamping (that is the fixture)

**Reconfiguring**:
Changing a tooling: gripper jaws in a gripper, clamping jaws or the small clamping system on the clamping desk, a press tool in the press. Changes only the cell state, not the product state. Robot 1 reconfigures the press with the Pizzagreifer.
_Avoid_: setup change, retooling, tool change (that swaps the robot's tool)

**Loading**:
A robot moves a pallet, with the part clamped on it, from its hand onto a coupling place of a station, e.g. Robot 1 sets the pallet on the DMG table and opens the Pizzagreifer. Changes only the cell state. *Unloading* is the reverse.
_Avoid_: tending, ablegen, uncoupling (the robot keeps its tool)

**Unloading**:
A robot takes a pallet from a coupling place of a station into its hand; the reverse of loading.
_Avoid_: aufnehmen, recoupling

**Releasing**:
A robot lets go of a held part, which stays where it is; the robot is no longer busy.
_Avoid_: dropping, ungripping

**Regrasp**:
Changing the grasp on a part that is already held.

**Handover**:
One robot passes a held part to another: both hold it briefly, then the giving robot releases it and is no longer busy.
_Avoid_: transfer, pass

**Unscrewing**:
Releasing a screw connection with the tool matching its screw head; the screw part leaves the assembly in the same action.
_Avoid_: screwing (direction unclear), unbolting

**Screwing in**:
Fetching a screw part from the supply with the tool matching its screw head and driving it in, which adds the screw part and its screw connection to the assembly; the inverse of unscrewing. The screw is never gripped or inserted as a separate action.
_Avoid_: screwing (direction unclear), bolting, tightening (as a capability)

**Milling**:
Cutting material to release a connection that cannot be released otherwise; may split a part into new parts. Destructive disassembly, not rework (see **Rework milling**).
_Avoid_: cutting, machining

**Rework milling**:
Milling a part in a special machine to refurbish it, e.g. the plate gear of the angle grinder in the mill; the part goes back into the product (see *Rework* in [Sequence graph](../sequence-graph/CONTEXT.md)).
_Avoid_: milling (alone, which releases a connection), machining, reprocessing

**Buffer slot**:
A reserved target for a part that only comes into existence through milling.

**Rejected action**:
An action refused before execution (masked out, wrong tool, out of range); it costs time but changes nothing.
_Avoid_: illegal action, invalid action

**Action attempt**:
One try of an action from a product state and cell state: the action is checked, planned and carried out, succeeding by chance (or always, when uncertainty is switched off), and gives one outcome: a rejected action with its reason, a failure, or the next product state and cell state. A POMDP step and the extraction of the sequence graph both make action attempts, so they agree on what an action does.
_Avoid_: action evaluation, transition, try

**Skill**:
A parametrized NC subroutine that carries out one capability on the cell, e.g. `SCREW_IN` with hole pose and C30S program number. Skills are parametrized by definition. When a planned operation exists, its values become the skill's parameters (see *Operation* in [Motion planning](../motion-planning/CONTEXT.md)). The cell simulation runs the same subroutine to check and time it; a motion planner knows a skill only by its name and the meaning of its parameters. Process parameters without a planned value (feed rates, pullout, dwell times such as the inspection time) are the skill's default parameters, not part of the cell.
_Avoid_: parametric skill, routine, macro

**Movement primitive**:
A parametrized elementary motion that skills are built from, e.g. spiral search or force-controlled approach.
_Avoid_: basic motion, sub-skill

**NC program**:
A SINUMERIK main program (`.mpf`) that the cell runs: a sequence of skill calls and taught motions.
_Avoid_: G-code (for the whole program), robot program

**Symbolic mode / Geometric mode**:
Whether capabilities run without plan functions (symbolic, formerly "Stage 1") or with them (geometric, formerly "Stage 2").
_Avoid_: Stage 1, Stage 2

### Uncertainty

**EOL condition**:
The hidden condition of a part in the product instance: *free*, *stuck* or *missing*.
_Avoid_: EOL state, wear state

**Stuck**:
A part that is seized (e.g. by corrosion or deformation) and resists removal although its connections say it is free.
_Avoid_: rusty, damaged, blocked

**Missing**:
A part present in the product model but absent in the product instance.

**Stuck-part rule**:
What the product setup declares about parts that can be stuck: which parts (by name or name prefix), which sampler prior applies to them (`screw`: stuck and missing, `upper`: stuck only) and which parts a stuck one is fixed to. The env injects those fixed connections into a stuck part's product instance and the world sampler draws stuck and missing parts only among the parts a rule names. A product without rules has no stuck-part effects. Field `stuck_part_rules` in `product_setup.json`, a list of `{"parts": [...], "part_prefixes": [...], "prior": "screw" | "upper", "fixed_to": [...]}` where the first matching rule wins.
_Avoid_: stuck config, EMO rules

**Belief**:
The policy's probability per part of being stuck or missing, given everything observed so far.
_Avoid_: estimate, confidence

**Inspection prior**:
The initial belief, taken from an inspection before disassembly starts.

**Belief mode**:
Whether the belief equals the truth (*fully observable*) or is inferred from observations (*POMDP*).

**Observer**:
The component that measures the product instance; the baseline observer reports which parts are visible.
_Avoid_: sensor, perception (for the simulated observer)

**Measurement**:
One output of the observer.
_Avoid_: observation (reserved for the full input the policy receives)

**Observation**:
Everything the policy receives after an action: measurement, mounted tool, action feedback and, optionally, the belief.

### Deciding

**Policy**:
The decision maker that chooses the next action from observations.
_Avoid_: agent

**Episode**:
One disassembly attempt of one product instance, from reset until all target parts are removed or the step limit is reached.

**Target part**:
A part whose removal ends the episode successfully.
_Avoid_: goal part

**Reward**:
The negative time an action costs, plus penalties for rejected actions, no-ops and running out of steps.
