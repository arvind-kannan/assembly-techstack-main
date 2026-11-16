# Thesis code is a functional reference

Code from a student thesis enters the techstack only through a merge request that the student opens (ADR-0009). Thesis code that comes from anywhere else, such as a thesis repo, a backup or a code package sent to us, is a **functional reference**: it shows which problems the thesis solves and how. Each problem we want in the techstack is rebuilt against our seams, with tests. This holds even when the code is already in a subtree, as Ignacio's cuMotion and TPG code is in `crc_disassembly_sim/modular_planner/`.

An integration starts with a grilling per thesis. The thesis text and any other input from the student are its sources. The grilling lists the problems 1..X the thesis solves, picks the ones we want, and writes down their tests. For each problem it answers three questions:

- What was done, how, and how was it tested?
- What must we watch for?
- Can the code be adapted to our system, or is it simpler to rewrite it?

The result is built as **tools or modules that each solve a few of these problems**, so other projects can reuse them. Examples are "where is a process module (a clamping table, a press, anything we have the CAD of) in my cell", "how and where do I change my product or module pose when motion planning fails" (Hao), and "how do I synchronise the motions of several robots" (Dana).

Decided in #237, on map &6 (Real motion planner for the cell simulation).

## Considered options

- **Wrap the thesis code as it is**: fast, but thesis repos carry hard-coded paths, scenes and old-architecture APIs, have no tests at our seams and no licence file. Their code would become a runtime dependency that nobody maintains.
- **Port thesis code file by file**: keeps the code but also its one-use-case structure. The problems it solves would not become reusable modules.

## Consequences

- The integration tickets of map &6 (#177, #178, #179) start with a grilling, not with a build.
- A thesis result that we want to keep (Hao's regrasp as motion planning, Dana's coupled insertion) needs a test that shows it in the techstack. Otherwise a rewrite loses it unnoticed.
- Dependencies that a thesis improved can still be taken over as a whole. Dana's version of `pybullet_industrial_path_planner` supersedes the upstream package, provided that general access to OMPL stays, so other planner families can be tried quickly.
