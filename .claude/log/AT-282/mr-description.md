Closes AT-282

## What changed
- `crc_disassembly_sequence_generator/operations.py`: the local-translation call site of `GrippingAndRemovingOperation` passes `return_contacts=True` and unpacks `(feasible, contacts)`, so cover-like parts get a real 5 mm collision check. The stepped-path site stays at `return_contacts=False` with a comment. One subtree commit (ADR-0002); core dataclasses untouched.
- `tests/unit/test_gripping_collision_check.py`: real `Part`s in an `Assembly`; a blocker 2 mm in a cover's path and a free cover pass; the stepped-path blocker test is a strict xfail until the follow-up enables that site.

## Decided without the user
- Land the local-translation site only; the stepped path stays inert until the direction candidates include the bore axis (follow-up build to be specced; a temporary departure from the reconcile decision to keep the stepped path strict): decided by the user (park 20261029-212826-50c2, resolved C).

## Checks
- New unit tests: 2 pass, 1 strict xfail. Full suite (`not gpu and not network`): 1045 passed, 2 skipped, 1 xfailed, 0 failed.

## Blast radius (measured with both sites fixed, before the decision)
- Removal planning per case part: no removal changes feasibility.
- Sequence graphs: angle grinder 6/5/1 and battery replacement 43/42/1 unchanged; EMO 7/6/1 became 6/5/0 (rotor candidates all tilted 6 to 8 degrees, 2.78 mm contact against the 2.5 mm limit in the stepped path). With the stepped path left as is, EMO keeps its goal.
- Evidence and scripts: `.claude/log/AT-282/`.

## Open points
- Follow-up build: add the bore axis to the gripping direction candidates in `planning.py`, then enable the stepped-path site and remove the xfail. Acceptance in the park decision.
