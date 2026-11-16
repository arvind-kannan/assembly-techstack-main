M · gripping planner · blocks landing the AT 282 collision-check fix · undo: hours of work
With the fix, the EMO demo product's rotor can no longer be removed and four integration tests fail; may the fix land that way?
I recommend (C): land only the local-translation call site now, and hold the stepped-path site until the direction candidates include the bore axis, because the axis pull is collision-free and only the tilted candidates collide.

## Background
The gripping check asks the collision library whether a part, moved a few millimetres along a candidate direction, overlaps another part by more than 2.5 mm. Until now it asked without requesting contacts, so the library reported no depth and the check always said yes. The fix makes it work. The planner offers a few candidate directions per part, derived from the model's degrees of freedom; for the EMO rotor all four are tilted 6 to 8 degrees off the vertical axis.

## Scenario
EMO, screws and upper lid already removed. Planning the rotor: all four tilted candidates now fail in the stepped path after 10 to 15 mm (one contact, depth 2.78 mm against the 2.5 mm limit). Pulling straight up (0, 0, 1) over the full 249.9 mm has no collision. Before the fix the rotor was accepted along a tilted direction, which in reality scrapes its bore.

## Why this came up
Decided on the reconcile ticket (Max Goebels, 2026-10-29): port Zeke's fix and keep the stepped path. The research said the blast radius was untested and the unit tests use fake parts. Measured now: EMO sequence graph goes from 7 nodes, 6 edges, 1 goal to 6 nodes, 5 edges, 0 goals (rotor never removable). Angle grinder (6/5/1) and battery replacement (43/42/1) are unchanged; planning time is unchanged (5.8 vs 5.7 s, 241.7 vs 241.0 s). Failing: test_emo_env scripted policy, two graph-extraction collapse tests, product explorer step-back (no goal in the graph). Evidence: .claude/log/AT-282/.

## How each answer shapes the road
| Option | What is built | Scenario result | Records and measurement | Size and risk |
|---|---|---|---|---|
| A: land as ticketed | Fix at both sites; EMO test expectations changed | Rotor unremovable, the EMO demo and graph lose their goal | Four tests rewritten to expect a broken baseline | Small; breaks the baseline product |
| B: hold the MR | Nothing lands; a follow-up ticket adds the axis candidate | Rotor removable along the axis | Fix lands later, all tests green | Delays the fix; one more ticket |
| C: local site only | Only the first call site; stepped path stays inert until B | EMO keeps its goal; cover-like parts get a real 5 mm check | Deviates from the reconcile decision to keep the stepped path strict | Small; partial fix |

What would make the recommendation wrong: if the user wants the stepped path strict now, since the rotor then needs the axis candidate first.
