# Core model

The shared description of a product for disassembly: which parts it has, how they are connected, and in which directions each part can move. Every other context builds on it; it changes only through scoped issues ([ADR-0003](../../adr/0003-core-model-change-process.md)).

## Language

### Product

**Assembly**:
A product described as parts plus the connections between them.
_Avoid_: model, scene, CAD

**Product model**:
The nominal assembly built from CAD, identical for every unit of a product.
_Avoid_: reference assembly, template

**Product instance**:
One physical returned unit, with its own EOL conditions; the agent never sees it directly.
_Avoid_: specimen, real assembly, world

**Part**:
A rigid body of an assembly, identified by its name, which is unique within the assembly and follows no naming scheme (#19).
_Avoid_: component, object, body

**Part condition**:
What has happened to a part of a product instance that its name does not show: a rework process it has received (e.g. rework milling) or that it was replaced by a spare. Part of the product state, so that a goal can demand it. A part of another version is another part, not a condition.
_Avoid_: part status, reworked flag, version (for this)

**Subassembly**:
A group of parts joined before the current process and handled as one unit, e.g. the angle grinder's crown-gear subassembly (crown gear + spindle + bearing + bearing plate as gear-housing lid), or the battery lid (PVC plate + bonded stiffening ladder with grip bar, #131). Modelled as one part.
_Avoid_: module, group

**Demonstrator product**:
A product used to show a process on the cell: a real product (the angle grinder) or a replacement product (the battery).
_Avoid_: demo part, test object

**Replacement product**:
A manufacturable stand-in that keeps the process-relevant geometry of a real product but not its function, e.g. the battery replacement product for the Yonghang pack (#29).
_Avoid_: mock-up, dummy

**Product name**:
The identifier of a product model: one English snake_case name that says what the product is, with a qualifier only to tell two products apart (e.g. `angle_grinder`, `battery_replacement`, `electric_motor`). It names the product model file and is what a sequence graph export records. Folder names in the 3D files repository follow that repository's own convention (#19).
Formerly `EMO_SR_V2` = `electric_motor`. Legacy and upstream test products keep their historical names (`DM_EMO`, `gm`, `rivet`, `EMO_SR_Veasy`).
_Avoid_: `EMO_SR_V2`, EMO (for the product name)

**Degree of damage**:
A per-part attribute whose meaning and scale are not yet defined. Not used by any context (#20).

### Connections

**Connection**:
A relation between exactly two parts that restricts how they move relative to each other.
_Avoid_: joint, link, liaison

**Geometric connection**:
A connection formed by surface contact alone; it blocks only the directions pointing into the contact.
_Avoid_: contact connection, loose connection

**Screw connection**:
A connection between a screw part and the part it fastens; it blocks all directions until the screw part is removed by unscrewing.

**Screw head**:
The drive type of a screw (e.g. `H5`, `T6`); it decides which tool can unscrew it.

**Fixed connection**:
A connection that blocks all directions and cannot be released by any capability.
_Avoid_: glued, welded

**Safety ring connection**:
A connection formed by a retaining ring that blocks all directions until the ring is removed.
_Avoid_: circlip, snap ring

**Liaison graph**:
The graph with parts as nodes and connections as edges.
_Avoid_: connection graph, contact graph

### Motion freedom

**Free directions (DOF)**:
The set of unit directions in which a part can translate locally out of contact with its connected neighbours. A set of directions, never a count; translation only.
_Avoid_: degrees of freedom (as a number), mobility

**Restrictions**:
The directions a connection blocks; the complement of its free directions.

**Free part**:
A part with at least one free direction.
_Avoid_: loose part, removable part

### Robot holds

Robot holds are not part of the assembly: they belong to the *cell state* (*Holder*, *Active grasp*, *Cell-assembly snapshot* in [Decision making](../decision-making/CONTEXT.md)). The hold methods still on the assembly are provisional until they are migrated.
