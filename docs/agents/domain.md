# Domain Docs

How the engineering skills consume this repo's domain documentation.

## Before exploring, read these

- **`CONTEXT-MAP.md`** at the repo root: lists the six contexts (core model, sequence graph, motion planning, decision making, product explorer, MRTAMP app), where their code lives, and how they relate. Read the `docs/contexts/<context>/CONTEXT.md` of every context the topic touches.
- **`docs/adr/`**: read the ADRs that touch the area you are about to work in. All ADRs are system-wide and live in this one folder.

## Layout

```
/
├── CONTEXT-MAP.md
├── docs/
│   ├── adr/                         ← all decisions
│   └── contexts/
│       ├── core-model/CONTEXT.md
│       ├── sequence-graph/CONTEXT.md
│       ├── motion-planning/CONTEXT.md
│       ├── decision-making/CONTEXT.md
│       ├── product-explorer/CONTEXT.md
│       └── mrtamp-app/CONTEXT.md
├── crc_diss_assembly_dataclasses/   ← core model code (subtree, protected)
├── crc_disassembly_sequence_generator/
├── crc_disassembly_sim/             ← cell simulation (subtree)
└── src/disassembly_pomdp/
```

Context glossaries live under `docs/contexts/`, not next to the code, because the subtree folders are pushed to their upstreams and must stay free of techstack docs.

## Use the glossary's vocabulary

When your output names a domain concept (issue title, refactor proposal, hypothesis, test name), use the term as defined in the relevant `CONTEXT.md`, and none of the words listed under _Avoid_. Code identifiers that still use an avoided word (e.g. `agent`, `manipulating`, `check_feasability`) keep their name until a scoped issue renames them; in prose, use the glossary term.

If the concept you need isn't in a glossary yet, either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0004 (plan function is the motion-planning seam), but worth reopening because…_
