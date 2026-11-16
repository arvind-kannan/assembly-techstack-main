# Core model changes only through scoped issues

The core model (`crc_diss_assembly_dataclasses/`) is the one definition of parts, connections and free directions that every thesis builds on; [ADR-0006](0006-core-grows-into-the-ppr-model.md) widens it to the PPR model (product, process, resource). Its architecture was proposed by Zeke (`trajektorie-z`); it is maintained by the core maintainers listed in `.gitlab/CODEOWNERS`. It changes only through a GitLab issue labelled `area:core-model` that scopes one behaviour, and a merge request approved by a core maintainer. Everything else extends the core model from outside (helpers, adapters, subclasses in the consuming context).

Accepted core changes flow back to `WBK-Robotics/crc_diss_assembly_dataclasses` via `git subtree push` ([ADR-0002](0002-sub-repos-as-git-subtrees.md)). Personal forks and copied-in versions of the dataclasses are never a source.

## Consequences

- The baseline core model already contains two changes that never reached WBK upstream: Zeke's open PR #1 (`is_translation_feasible`, FCL cache purity, nearest-neighbour DOF deletion) and Hans Wormuth's robot-hold methods (`agents`, `ActiveGrasp`, `assign_agent`, `release_agent`, `get_part_holder`). Tracked in #14 and #15.
- Robot holds are provisional on `Assembly`; they move to the cell state type in the core, and the hold API on `Assembly` is deprecated, then removed (#15, decided in #95, see [ADR-0006](0006-core-grows-into-the-ppr-model.md)).
