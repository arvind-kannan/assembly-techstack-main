# Handoff AT-228
Ticket: AT-228
Branch: ticket/228-step-catalogue
Last commit: ac6fb98 refactor(mrtamp): review fixes for the step catalogue (#228)
MR: https://gitlab.kit.edu/kit/wbk/map/imk/sfb1574_kreislauffabrik/assembly-techstack/-/merge_requests/111
Done: core StepRef + Operation.step (own commit); plan_step names the step; penalties read op.step; StepCatalogue module; v1 and B1 on it; tests (catalogue, assembly-run replanning, core round trip); code review fixed; draft MR !111
Open: pipeline 751644 to go green, then mark MR ready; core-maintainer approval of the core commit
Next step: When pipeline 751644 is green, mark MR !111 ready (remove Draft)
Test state: fast suite 756 passed 2 skipped; sim+contract 145 passed 1 skipped; all green locally
Parked decisions: -
Continue with: /implement AT-228
