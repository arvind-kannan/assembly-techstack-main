# AGENTS.md

Simulation and planning stack for robotic disassembly of end-of-life products under uncertainty: a core product model, a sequence graph, motion planning, and a POMDP gym on top.

## Orient first

- Read `CONTEXT-MAP.md`, then the `docs/contexts/<context>/CONTEXT.md` of each context your task touches. Name things with glossary terms in prose, issues, commits and test names.
- Read the ADRs in `docs/adr/` that touch your area before changing it; surface any conflict with one instead of working around it.

## Guardrails

- **Core model is protected.** Change `crc_diss_assembly_dataclasses/` only when working an issue labelled `area:core-model` whose scope a core maintainer approved; keep the change to that scope. This covers the whole PPR model the core grows into: product, process and resource types such as cell, robot, tool, cell state, operation and operation plan (ADR-0006). For any other task, extend the core model from the consuming context (helper, adapter, subclass) (ADR-0003).
- **Subtrees go upstream.** `crc_diss_assembly_dataclasses/`, `crc_disassembly_sequence_generator/` and `crc_disassembly_sim/` (the cell simulation) are git subtrees. Put changes to a subtree folder in commits of their own (ADR-0002).
- **Motion planners plug in through the plan function.** Pass a new planner as `plan_func` to the capability; leave capability and env code as is (ADR-0004). MRTAMP planners plug in through the stateless `plan(planning request)` call; leave the run controller and the cell simulation as is (ADR-0007).
- **No unchecked motion to the cell.** NC programs for the real cell come only from the cell simulation; never add a silent fallback to unplanned motion (ADR-0008).
- **Legacy stays as is.** The DASG app and the planner search graph (see `CONTEXT-MAP.md`) are reference code for the real cell; ask before changing them.

## Run

- Tests: `python -m pytest tests/unit` (unit only); `python -m pytest -m "not slow and not geom and not gpu" -n 4` (fast suite = CI gate, about 1 min on 4 cores with pytest-xdist; leave out `-n 4` to run serially); `python -m pytest` for everything. `import disassembly_pomdp` (plus `pytest.ini` and `tests/conftest.py`) sets up the import paths, so no install is needed. Markers: `geom`, `slow`, `integration`, `gpu`, `network` (needs internet; also opt-in through its own environment variable).
- Python ≥ 3.10. Install with `uv sync` (versions from `uv.lock`; see the README, "Setup"). Add dependencies with `uv add`, never by hand.
- **CI is the merge gate (ADR-0009).** Before pushing run `uv run pre-commit run --from-ref origin/main --to-ref HEAD` and the fast suite. Open the MR as draft, watch the pipeline with `glab ci status` / `glab ci trace <job>`, fix up to about three times, and mark it ready only when green; otherwise ask. Agent MRs then merge themselves through the merge train (squashed); catch up with `git merge origin/main`, never rebase or force-push. Cover at least 60 % of the lines you change in `src/`. Mark tests that need an NVIDIA GPU with `gpu`.
- On Windows, set `git config core.longpaths true` before checkout: some test data paths exceed 260 characters.
- Binaries (`png`, `gif`, `stl`, `npy`, `pkl`, `prt`) are in Git LFS outside the subtree folders; `crc_disassembly_sim/` keeps its upstream's own LFS rules (ADR-0002).

## Reference material

- **Sinumerik NC code library**: NC programs (`_N_MPF_DIR/`) written by many students on the cell. Use it as a source of working examples when asked to write or fix NC code for the Sinumerik controller. GitHub: `https://github.com/WBK-Robotics/Sinumerik-NC-Code-Library`. The maintainer also has a local clone at `D:/Sinumerik-NC-Code-Library`; other users and agents clone from GitHub.

## Agent skills

### Issue tracker

GitLab issues in `assembly-techstack` and `crc-disassembly-sim`; maps are epics and milestones of the group `kit/wbk/map/imk/sfb1574_kreislauffabrik`. See `docs/agents/issue-tracker.md`.

### Triage labels

Scoped group labels (`category::`, `state::`, `wayfinder::`) plus the project's `area:` labels. See `docs/agents/triage-labels.md`.

### Domain docs

Multi-context: `CONTEXT-MAP.md` with one `CONTEXT.md` per context under `docs/contexts/`, ADRs in `docs/adr/`. See `docs/agents/domain.md`.
