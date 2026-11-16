# Zeke's gripping rework vs. the subtree: what is missing

Research for #17 Reconcile the two parallel gripping-planner reworks. Related: #128, #234 (`removal_planning.py`), [ADR-0002](../adr/0002-sub-repos-as-git-subtrees.md), [ADR-0003](../adr/0003-core-model-change-process.md), [ADR-0004](../adr/0004-plan-function-is-the-motion-planning-seam.md).

**Question ("Wanted" 1).** Diff Zeke's `e27f85f` against the subtree at `ced2ed9` and list what of Zeke's work is missing. The port/drop decisions ("Wanted" 2) are the maintainer's; this note only recommends.

Source keys:
- `[Z@<sha>/…]` = `WBK-Robotics/crc_disassembly_sequence_generator`, commit `<sha>`, fetched by full SHA (`e27f85f925a4f0b29a77b7f47f7525544506600b`; `1f31006` and `5cfbc30` are its ancestors). Files are at the repo root there (`planning.py`, `operations.py`).
- `[H@ced2ed9/…]` = Hans' `gripping-operation` head, the commit the subtree was added from (`0e487c8`). Same layout.
- `[R/…]` = this repo at `8dc0d8c` (`origin/main`, 2026-10-27), paths under `crc_disassembly_sequence_generator/` unless they start with `src/`. The subtree files differ from `ced2ed9` only by later commits (#112 cell-state refactor of the holder helpers in `operations.py`, #169 HMI/imports; 19 changed lines in `planning.py`, 158 in `operations.py`, checked with `diff` against `git show ced2ed9:<file>`). The gripping planner and the `check_feasability` / stepped-path code are unchanged (`return_contacts=False` at `operations.py:822, 964`).
- `[D/…]` = `crc_diss_assembly_dataclasses/src/diss_assembly_dataclasses/` in this repo.

## Short answer

- **Merge base** of `e27f85f` and `ced2ed9` is `4fa3444` (Zeke's `safe_planning` merge). Zeke's side is 3 commits on it, touching two files: `1f31006` and `5cfbc30` (both 2026-08-07) and `e27f85f` (2026-09-05).
- **Hans took Zeke's work over at `2c64f37` (2026-09-02)**, which is after the two August commits and before `e27f85f`. So `e27f85f` is Zeke's change *after* the takeover, and it is the only thing that can be missing.
- **Most of Zeke's work is in.** The planner rewrite (candidate-DOF loop, feasibility checked before the part is removed from the successor, failure diagnostics) and the whole projected-footprint ray-bundle check are in the subtree, renamed (`plan_gripping` → `plan_gripping_and_removing`, `GrippingOperation` → `GrippingAndRemovingOperation`).
- **One thing is missing, and it matters: Zeke's `return_contacts=True` fix in the local-translation check.** Without it, the local step and Hans' stepped-path stage never reject anything (details below). `e27f85f` also drops the stepped-path stage that Hans kept; that is a design choice to decide together with the fix, not an independent item.

## Item list

| # | Item in Zeke's `e27f85f` | In the subtree? | Recommendation (for the maintainer) |
|---|---|---|---|
| 1 | Local-translation feasibility asks `is_translation_feasible(..., return_contacts=True, depth_threshold=…)` and unpacks `(feasible, contacts)` [Z@e27f85f/operations.py:`check_feasability`] | **Missing.** Subtree passes `return_contacts=False` [R/operations.py:822] | **Port.** Fixes a no-op check (see below). |
| 2 | Stepped actual-mesh path check removed (from `1f31006`/`5cfbc30`, kept by Hans: [R/operations.py:936, 964]) | Present, but inert: it also passes `return_contacts=False` | **Decide with item 1.** Keeping it costs one collision query per 5 mm of pull-out per candidate and, once item 1 is applied, becomes strict for the first time. It exists for hollow parts (lids, covers) where the ray bundle fills the interior. |
| 3 | Candidate-DOF loop over assembly DOFs only, deduplicated; successor assembly (`copy.copy` + `remove_part`) built only after the feasibility check passes [Z@e27f85f/planning.py:`plan_gripping`] | In [R/planning.py:`plan_gripping_and_removing`] (same loop, same order of steps; Hans added direction freezing and secondary-path locks) | No action. |
| 4 | Planner failure diagnostics (`candidate_failures`, `dof_candidate_count`, `raw_dof_count`, `invalid_dof_count`, gripping radius/height) | In; `src/motion_planning/removal_planning.py` already consumes them (`_PLANNER_DETAILS`) | No action. |
| 5 | Defaults for missing `cell_settings.json` keys (`.get(key, default)`) | In. The feed-rate keys were renamed to `feed_rate_gripping_and_removing_*`, which `cell_settings.json` does not define, so the code defaults apply (2000/1000, equal to the json values for `feed_rate_gripping_*`) | No action; optionally rename the json keys later (not a port). |
| 6 | Projected-footprint ray-bundle check and its helpers (`_check_removal_path_projected_footprint_rays`, `_make_projected_footprint_ray_origins`, hull/point-in-polygon/downsample/basis/scene builders) | In, byte-identical except one log prefix (checked with a method-by-method diff of both files) | No action. |
| 7 | `for part in … .parts.values()` fixes in `explore_node` / `static_explore_node` [Z@e27f85f/planning.py:226, 1658] | In [R/planning.py:480] | No action. |
| 8 | `import open3d as o3d` in `operations.py` | In | No action. |
| 9 | Failure codes under the key `failure_reason` / `reason` | Hans uses `failure_stage` for the code and `failure_reason` for the message; `removal_planning.py` and the gym consume Hans' keys | **Drop.** Porting Zeke's naming would break the consumers. |
| 10 | Class name `GrippingOperation` (one class) | Hans split it into `GrippingAndRemovingOperation`, `GrippingOperation`, `RemoveOperation`, `RegraspOperation`; the gym uses the split | **Drop.** |
| 11 | Large-radius grip-point fallback also accepts `part.mesh` when it is a `trimesh.Trimesh` | Only `trimesh_mesh` / `mesh_trimesh` / `tri_mesh` attributes. Parts carry Open3D meshes now, so the `trimesh.Trimesh` branch never triggers | **Drop.** Moot. |

## The defect behind item 1 (verified)

`Assembly.is_translation_feasible` [D/assembly.py:1667] only collects penetration depths from `res.contacts`, and it builds the request with `enable_contact=return_contacts`. With `return_contacts=False` and a `depth_threshold` (both of Hans' callers pass one, 2.5 mm by default), python-fcl returns placeholder contacts whose depth is not positive, so no contact exceeds the threshold and the method returns `True` even for a collision. Zeke's comment in `e27f85f` says this in words ("contact generation must be enabled so that penetration depth can be compared against depth_threshold").

Evidence, run offline on this box with the repo `.venv`:

1. Raw python-fcl, two overlapping 10 mm boxes (`CollisionRequest(enable_contact=…)`): `enable_contact=False` gives one contact with `penetration_depth` 0.0; `True` gives four with 5.0. Same experiment with `fcl.BVHModel` meshes: `False` gives depth −0.707 (garbage, negative); `True` gives 5.0.
2. Through the real classes (two `Part`s as Open3D boxes in an `Assembly`; the second box sits 2 mm above the first, in its +z removal path; test script was throwaway, not committed):
   - `assembly.is_translation_feasible("moving", [0,0,5], return_contacts=False, depth_threshold=2.5)` → `True`; with `return_contacts=True` → `False` (collision found).
   - `GrippingAndRemovingOperation.check_feasability` with `ray_bundle_required=False`: the ray bundle reports "hit=blocker at t=1.0", the stepped path reports feasible, and the result is **`True`**, with the log line "projected-footprint warning ignored … actual-mesh stepped path is feasible".

**Consequence in the subtree today.** For parts whose ray bundle is "required", only the ray bundle decides feasibility. For parts the planner treats as cover-like (`ray_bundle_required: not cover_like`, [R/planning.py:1125]; `cover_like` is a name test, `_looks_like_hollow_cover(part.name)`), the ray-bundle result is ignored and *no collision check is effective at all*: every DOF direction is accepted as feasible. Those are exactly the parts the stepped path was added for.

## Where to fix it (sub-choice for item 1)

- **At the two call sites** in `operations.py` (set `return_contacts=True`, unpack the tuple): a subtree commit under ADR-0002, kept free of other changes; the generator is the sequence-generator subtree, not the core model, so ADR-0003's core-maintainer approval does not apply. Zeke's `e27f85f` does exactly this, so the commit can be authored from it.
- **In the callee** [D/assembly.py:1667] (make `return_contacts=False` with a `depth_threshold` still request contacts internally): fixes all callers at once, but it is a core-model change and goes through ADR-0003.

The two are not exclusive; the call-site change is the smaller and needs no core approval.

## Not checked

- **Blast radius of the fix.** I did not apply the change to the tree and did not run the gripping tests with it. `tests/unit/test_gripping_and_removing_capability.py` and `tests/unit/test_disassembly_pomdp_env.py` use fake parts, so they probably do not exercise the check; whether real cases (`cases/`) start to fail removals (and how many) is untested.
- **Runtime cost of a live stepped path** (item 2) on real parts: not measured.
- **Zeke's intent for dropping the stepped path** in `e27f85f` ("Simplify"): his commit message says only "Simplify gripping feasibility checks"; I read no PR text (the PR list is outside the token's scope).
- **Zeke's other branches** on the fork (`trajektorie-z/…`, `pr/gripping-full-path-feasibility`): only the three commits reachable from `e27f85f` were read.

## Gaps against the ticket's "Done when"

The list is here. The per-item decisions are the maintainer's (triage comment on #17 of 2026-10-27), and no port issue was opened: if items 1 (and 2) are ported, the port issue should carry "one subtree commit, call sites only, from `e27f85f`".
