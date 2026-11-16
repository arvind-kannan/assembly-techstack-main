"""Reports a red pipeline on main after a merge (ADR-0009).

Agent merge requests merge themselves through the merge train, which runs the fast stage only; the
slow and geom tests run on main afterwards. When a job of that main pipeline fails, this script keeps
one open issue labelled `ci::main-red`: it opens the issue with the commit, the failed jobs and the
pipeline link, or adds a note to the issue that is already open. Someone closes it once main is green.

Run by the CI job `main-red-issue` (`when: on_failure`). Needs CI_API_V4_URL, CI_PROJECT_ID,
CI_PIPELINE_ID and a token with `api` scope: MAIN_RED_TOKEN, else DRIFT_TOKEN (both protected, so
main can read them). With `--dry-run`, or without a token, it prints the text instead.
"""

import json
import os
import sys
import urllib.parse
import urllib.request

LABEL = "ci::main-red"
LABELS = f"{LABEL},category::bug,area:env-setup"


def env(name, default=""):
    return os.environ.get(name, default)


class GitLab:
    def __init__(self, token):
        self.api = f"{env('CI_API_V4_URL')}/projects/{env('CI_PROJECT_ID')}"
        self.token = token

    def call(self, method, url, data=None):
        body = json.dumps(data).encode() if data is not None else None
        req = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={"PRIVATE-TOKEN": self.token, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            return json.load(resp)

    def failed_jobs(self):
        url = f"{self.api}/pipelines/{env('CI_PIPELINE_ID')}/jobs?scope[]=failed&per_page=100"
        return [(job["name"], job["web_url"]) for job in self.call("GET", url)]

    def open_issue(self):
        query = urllib.parse.urlencode(
            {"labels": LABEL, "state": "opened", "per_page": 1}
        )
        found = self.call("GET", f"{self.api}/issues?{query}")
        return found[0] if found else None


def report(jobs):
    sha = env("CI_COMMIT_SHA", "?")
    job_lines = [f"- [{name}]({url})" for name, url in jobs] or [
        "- (not listed: no token)"
    ]
    return "\n".join(
        [
            f"The pipeline on `main` failed after commit {sha[:8]} "
            f"({env('CI_COMMIT_TITLE', '?')}, by {env('CI_COMMIT_AUTHOR', '?')}).",
            "",
            "Failed jobs:",
            *job_lines,
            "",
            f"Pipeline: {env('CI_PIPELINE_URL', '?')}",
        ]
    )


def main():
    token = env("MAIN_RED_TOKEN") or env("DRIFT_TOKEN")
    dry_run = "--dry-run" in sys.argv or not token
    gitlab = None if dry_run else GitLab(token)
    jobs = [] if dry_run else gitlab.failed_jobs()
    names = ", ".join(name for name, _ in jobs if name != "main-red-issue") or "a job"
    title = f"main is red: {names} at {env('CI_COMMIT_SHORT_SHA', '?')}"
    body = report(jobs)
    if dry_run:
        print(f"[dry run] {title}\n\n{body}")
        return
    issue = gitlab.open_issue()
    if issue:
        gitlab.call(
            "POST",
            f"{gitlab.api}/issues/{issue['iid']}/notes",
            {"body": f"Again: {title}\n\n{body}"},
        )
        print(f"noted on #{issue['iid']} {issue['title']}")
    else:
        new = gitlab.call(
            "POST",
            f"{gitlab.api}/issues",
            {"title": title, "description": body, "labels": LABELS},
        )
        print(f"opened #{new['iid']} {new['title']}")


if __name__ == "__main__":
    main()
