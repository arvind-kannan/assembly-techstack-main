# Triage labels

The skills speak in canonical roles. This file maps them to the scoped labels of the group `kit/wbk/map/imk/sfb1574_kreislauffabrik`. A scoped label allows one value per family: setting `state::ready-for-agent` removes `state::needs-triage`.

| Canonical role | Label | Meaning |
|---|---|---|
| `bug` | `category::bug` | Something is broken |
| `enhancement` | `category::enhancement` | A new feature or an improvement |
| `documentation` | `category::documentation` | Shipped or developer documentation |
| `needs-triage` | `state::needs-triage` | A report a maintainer must evaluate |
| `needs-info` | `state::needs-info` | Waiting for the reporter |
| `ready-for-agent` | `state::ready-for-agent` | Passed the triage gate; an agent can do it unattended |
| `ready-for-human` | `state::ready-for-human` | Needs a human: a decision, a grilling, or human work such as a release cut |
| `wontfix` | `state::wontfix` | Will not be done: rejected, or already implemented (say which) |
| map | `wayfinder::map` | On the epic only |
| grilling, research, prototype, task | `wayfinder::grilling`, `::research`, `::prototype`, `::task` | Decision tickets of a map |
| build | `wayfinder::build` | A build ticket filed by the triage gate |
| audit | `review::audit` | An audit ticket of a map, in place of a `wayfinder::` label; the audit task type takes it |
| activity | `activity::building`, `::parked`, `::landing`, `::handed-back` | What a session is doing with the ticket now; at most one; scripts set it; none on a closed ticket (the land's end clears it) |

When a skill mentions a role (for example "apply the AFK-ready triage label"), use the label from this table. Every ticket carries one `category::` and one `state::` label; a map ticket also carries one `wayfinder::` label, or `review::audit` for an audit ticket.

## Area labels

Every ticket also carries one `area:` project label naming the context it touches (see `CONTEXT-MAP.md`):

| Label | Scope |
| ----- | ----- |
| `area:core-model` | `crc_diss_assembly_dataclasses/`; the only way to change the core model (ADR-0003). Never combine with `state::ready-for-agent` unless a core maintainer has approved the scope in the issue. |
| `area:sequence-graph` | sequence graph extraction and its consumers |
| `area:motion-planning` | operations, plan functions, feasibility checks |
| `area:decision-making` | POMDP env, capabilities, belief, observer, reward |
| `area:product-explorer` | product import, authoring and export |
| `area:mrtamp-app` | MRTAMP app: planning and execution of sequence graphs on the Transformer cell |
| `area:env-setup` | packaging, dependencies, CI, repo tooling |
| `area:dasg-hmi` | legacy DASG app (operator GUI, perception, machine link) |

## Other project labels

- `priority:high`, `priority:medium`, `priority:low`: the order in which `state::ready-for-agent` tickets are started (`wayfinder.sh --ready-for-agent`).
- `work-package`, `DISS`, `AP 1` … `AP 5`: the dissertation work packages on the map "Complete architecture". They are a plan, not tickets, and carry no `category::`, `state::` or `wayfinder::` label.
