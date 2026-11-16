# Issue tracker: GitLab

Issues live on `gitlab.kit.edu`. Maps (epics), milestones and labels live in the group `kit/wbk/map/imk/sfb1574_kreislauffabrik` (id `248019`), so that every project below can use them. A ticket lives in the project whose code it changes.

| Project | id | MRs target | Default branch |
|---|---|---|---|
| `assembly-techstack` | `200809` | `main` | `main` |
| `crc-disassembly-sim` | `200879` | `main` | `main` |

## Rules

1. **A ticket is one decision or one MR-sized build.** A new decision, or a small point, about a feature that already has an open ticket goes into that ticket: one short note and one checkbox. It does not get a ticket of its own.
2. **Shape.** A ticket is of one of six kinds, and each kind but the map has a recommended size for its human part (the description). A size is a recommendation, not a limit: no script, gate or skill refuses a text for its length alone, and when Fable (or anyone) has much to say, the longer wording stays. Structure (headings, tables, short paragraphs) keeps a long text readable; compressing a text to hit a number does not. A decision ticket is one decision of size S, M or L. It holds `## Question` and its options, about one screen (350 words) recommended; the answer is one resolution note. A build ticket is one merge request and uses the `/to-spec` template: about one page (700 words) recommended, no file paths, no line numbers. A report is one finding, about ten lines (120 words) recommended (one finding per report is a rule; the ten lines are not). A research ticket (the `wayfinder::research` label) and an audit ticket (the `review::audit` label) are about 500 words each: a question, what is known, what to decide and the input. Agent-only log files attached to a ticket (briefs, digests, evidence and handoffs under `.claude/log/<ticket>/`) are as verbose as needed. A map is a direction with its tickets. The title states the question or the outcome in plain words. A decision of size L that appears inside another ticket is split out into its own ticket. Build reports, test counts and review findings go in the merge request, not in the ticket.
3. **Maps.** A map is a group epic with a group milestone of the same name. It resolves when its destination is reached; the destination may be an implemented build. See `/wayfinder`.
4. **The gate.** A build ticket is filed only after the triage gate passes it: a grilling ends in `/to-spec`, which runs `/triage` in gate mode as a sub-agent. It is filed as `state::ready-for-agent`, `wayfinder::build`, with one model label (`model::opus` or `model::sonnet`, which the gate names), on the map and its milestone. A draft that fails goes back to the grilling.
5. **Reports.** A finding from outside a grilling (a bug in a build, a defect in a run, feedback) goes into the open ticket that covers it. If none does, it becomes one short `state::needs-triage` report — about ten lines recommended: what happened, how to reproduce it, whether it blocks the current use case — one per run or feedback round, attached to its map.
6. **Labels.** Exactly one each of `category::`, `state::` and `wayfinder::` (map tickets; an audit ticket carries `review::audit` in place of `wayfinder::`). A ticket that a session works on also carries at most one `activity::` label (`building`, `parked`, `landing` or `handed-back`); scripts set it. A closed ticket carries none: the end of its land clears the label, open or closed. See `triage-labels.md`.
7. **Assignee = claim.** One person, set when the work starts, removed when it stops without a result. Unassigned means free. Everyone else gets an `@mention`; "assign this to X" means `@mention X`. Agents work on behalf of the person who listed the task. A handback removes the assignee, sets the Status to "To do" and the label to `activity::handed-back`, and `@mention`s the next person if one is named.
8. **Closing.** A build ticket closes when its work reaches `main`. A merge request into `main` says `Closes <ticket>`. Work merged into another integration branch closes with the cut: the cut merge request into `main` lists `Closes` for every merged build ticket of the milestone.
9. **Triage cadence.** Each map session starts by triaging the map's `state::needs-triage` reports. Reports without a map are triaged in a monthly sweep. An unattended triage run decides by size: an S to-spec outcome is filed with its interim line, every other outcome parks a card, and a session never closes a ticket on its own.
10. **Agents fix, they do not comment.** What these rules determine — labels, links, milestone, epic, closing — an agent sets directly, without a note about it. Notes are only for human events: a decision's resolution, the merge request link, a handback and an appended decision. A disclaimer is one line. A question that needs judgement goes to the human in the session, not into a note.
11. **Names, not numbers.** Refer to a ticket by a short handle with its number — "the dashboard title fix (KSM 402)" — never a bare `#402`. Inside another project's ticket, write the full reference (`group/project#N`, `group/project!N`): a short `!N` links to the current project's merge request.
12. **The State section.** The top of a ticket's description holds a State section between `<!-- state:start -->` and `<!-- state:end -->`: status, holder, what is done, what is open, what blocks the ticket (named in words) and, once the folder exists, the link to the agent log. A script rewrites it at each event, and only it. The rest of the description changes only through the triage gate or a human. The Status widget says "In progress" while a session holds the ticket and "To do" after a handback.
13. **The agent log.** Briefs, evidence, resume notes and handoff data go into `.claude/log/<ticket>/` in the repository, on the normal branches, merged with the work. Layout: `state.md`, `log.md`, `handoff.md`, `history.md`, `briefs/`. Maps use `.claude/log/maps/<map>/` in the map's lead repository. The release pipeline strips the folder. The ticket shows one link to it.
14. **Touching an old ticket.** Every open ticket of KSM, OGM, TSI and JMS got the State section in one pass on 2026-10-14. From then on a script rewrites it at each launch, parking, handback and landing, and sets the `activity::` label, without review. It leaves the rest of the description as it is. Scripts never write on the tooling project's tickets. A rewrite of an old ticket (a bare reference, a missing State section; a long ticket only when its owner asks) happens in a sweep, one map at a time, and the map's owner reviews it. The old prose and notes move to `history.md`.

## Operations

Use the REST API (`https://gitlab.kit.edu/api/v4`) or `glab` with `GITLAB_HOST=gitlab.kit.edu`. Read back every write. Never write an empty description: the API accepts it and wipes the issue.

- **Create:** `POST projects/<id>/issues` with `title`, `description`, `labels`, `milestone_id`.
- **Read:** `GET projects/<id>/issues/<iid>` and `.../notes`.
- **Note / labels / close:** `POST .../issues/<iid>/notes`; `PUT .../issues/<iid>` with `add_labels`, `remove_labels` or `state_event=close`.
- **Move to another project:** `POST .../issues/<iid>/move` with `to_project_id`. The old issue closes with a "moved to" link.

## Wayfinding operations

Used by `/wayfinder` and `/frontier`.

- **Map:** `POST groups/248019/epics` with `labels=wayfinder::map`, and `POST groups/248019/milestones` with the same title.
- **Child ticket:** attach by reference, `POST groups/248019/epics/<epic iid>/issues/<issue global id>`, and set the issue's `milestone_id`.
- **Blocking:** `POST projects/<id>/issues/<iid>/links` with `target_project_id`, `target_issue_iid`, `link_type=is_blocked_by`. A ticket is unblocked when every blocker is closed.
- **Frontier:** `/frontier`, or the epic's open children without an open blocker and without an assignee.
- **Claim:** `PUT projects/<id>/issues/<iid>` with one `assignee_ids` value — the session's first write.
- **Resolve:** the resolution note, close, one line in the map's *Decisions so far*.
