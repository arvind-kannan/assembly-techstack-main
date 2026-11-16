# Handoff AT-243
Ticket: AT-243
Branch: ticket/243-product-explorer-import-mesh-folder
Last commit: 2af671d
MR: https://gitlab.kit.edu/kit/wbk/map/imk/sfb1574_kreislauffabrik/assembly-techstack/-/merge_requests/87
Done: Import, cancel, warnings, notices, first save, browser form implemented; 514 fast tests + ruff green on the pre-fix commit; MR !87 open (target ticket/240 branch, Stacked on !86); review done, cancel race fixed; unsaved-version bug fixed in code (_versions=_version=1)
Open: 1 add two failing-first tests: import,save,rename => unsaved True; import,rename,save,undo => unsaved True; 2 park S cards via park.py (non-empty first-save target refused; bbox warning = largest extent) and replace MR 'Open points' with 'decided without Max Goebels (Maintainer) (park <id>)' lines; 3 no node on machine: app.js not parse-checked, say so under Checks in MR; 4 rerun import + edit_history tests and fast suite, push, wait green, update MR checks counts; 5 ticket_state.py via uv run python; 6 fetch and merge origin/ticket/240 branch if moved
Next step: Add the two unsaved tests in tests/integration/test_product_explorer_import.py, run them with the edit-history tests, then park the two S cards
Test state: import tests 20 green before the last fix; fast suite not rerun
Parked decisions: -
Continue with: /implement AT-243
