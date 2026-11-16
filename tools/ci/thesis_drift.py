"""Two-weekly drift check of thesis branches against main (ADR-0009).

For every `thesis/<topic>` branch: merge main into it in a scratch worktree, list the
conflicts, run the fast tests on the merged tree, and list the main commits that touch files the
thesis changed. The facts go into one tracking issue per thesis, assigned to the users the branch's
protection rule lets push (the student; unassigned if none); the issue
is closed while the branch merges cleanly and the tests pass, and reopened when it drifts again.

Run by the scheduled CI job `thesis-drift`. Needs CI_API_V4_URL, CI_PROJECT_ID and DRIFT_TOKEN
(a project access token with `api` scope). With `--dry-run`, or without DRIFT_TOKEN (e.g. a manual
pipeline on an MR branch, which cannot read the protected variable), it prints the issue bodies instead.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

MAIN = "origin/" + os.environ.get("CI_DEFAULT_BRANCH", "main")
FAST_TESTS = [
    "uv",
    "run",
    "pytest",
    "-q",
    "-p",
    "no:cacheprovider",
    "-m",
    "not slow and not geom and not gpu",
]
TITLE = "Thesis drift: {branch} vs main"
LIST_LIMIT = 30


def git(*args, cwd=None, check=True):
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and done.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed ({done.returncode}):\n{done.stderr}")
    return done.stdout.strip()


def lines(text):
    return [line for line in text.splitlines() if line]


def thesis_branches():
    refs = lines(
        git("for-each-ref", "--format=%(refname:short)", "refs/remotes/origin/thesis/")
    )
    return [ref.removeprefix("origin/") for ref in refs]


def failing_tests(junit_path):
    if not os.path.exists(junit_path):
        return ["(no test report: the test run did not start, see the job log)"]
    failed = []
    for case in ET.parse(junit_path).iter("testcase"):
        if case.find("failure") is not None or case.find("error") is not None:
            failed.append(f"{case.get('classname')}::{case.get('name')}")
    return failed


def check_branch(branch):
    ref = "origin/" + branch
    base = git("merge-base", MAIN, ref)
    behind = len(lines(git("rev-list", f"{base}..{MAIN}")))
    thesis_files = lines(git("diff", "--name-only", base, ref))
    overlap = (
        lines(
            git("log", "--format=%h %s (%an)", f"{base}..{MAIN}", "--", *thesis_files)
        )
        if thesis_files
        else []
    )

    scratch = tempfile.mkdtemp(prefix="drift-")
    git("worktree", "add", "--detach", scratch, ref)
    try:
        merged = subprocess.run(
            [
                "git",
                "-c",
                "user.name=drift",
                "-c",
                "user.email=drift@ci",
                "merge",
                "--no-commit",
                "--no-ff",
                MAIN,
            ],
            cwd=scratch,
            capture_output=True,
            text=True,
        )
        conflicts = lines(git("diff", "--name-only", "--diff-filter=U", cwd=scratch))
        failed = []
        if merged.returncode == 0 and not conflicts:
            junit = os.path.join(scratch, "drift-report.xml")
            subprocess.run([*FAST_TESTS, f"--junitxml={junit}"], cwd=scratch)
            failed = failing_tests(junit)
    finally:
        git("worktree", "remove", "--force", scratch, check=False)
        shutil.rmtree(scratch, ignore_errors=True)

    return {
        "branch": branch,
        "behind": behind,
        "conflicts": conflicts,
        "failed": failed,
        "overlap": overlap,
    }


def issue_body(r):
    in_sync = not r["conflicts"] and not r["failed"]

    def bullets(items):
        shown = [f"- `{item}`" for item in items[:LIST_LIMIT]]
        if len(items) > LIST_LIMIT:
            shown.append(f"- … and {len(items) - LIST_LIMIT} more")
        return "\n".join(shown) or "- none"

    status = (
        "Merges cleanly with main and the fast tests pass."
        if in_sync
        else "Needs attention before the next milestone MR."
    )
    return (
        f"""Updated by the scheduled `thesis-drift` job ({os.environ.get("CI_JOB_URL", "local run")}).

**`{r["branch"]}` is {r["behind"]} commits behind main.** {status}

## Merge conflicts with main
{bullets(r["conflicts"])}

## Fast tests failing after merging main
{bullets(r["failed"]) if not r["conflicts"] else "- not run: resolve the conflicts first"}

## Main commits touching files this thesis changed
{bullets(r["overlap"])}

To catch up: `git fetch origin && git merge origin/main` on your thesis branch, fix the conflicts and tests, push.
You decide when; this issue only tracks the gap. It closes itself once the branch is in sync.
""",
        in_sync,
    )


class GitLab:
    def __init__(self):
        self.api = (
            f"{os.environ['CI_API_V4_URL']}/projects/{os.environ['CI_PROJECT_ID']}"
        )
        self.token = os.environ["DRIFT_TOKEN"]

    def call(self, method, url, data=None):
        body = json.dumps(data).encode() if data is not None else None
        req = urllib.request.Request(
            url,
            body,
            method=method,
            headers={"PRIVATE-TOKEN": self.token, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def find_issue(self, title):
        query = urllib.parse.urlencode({"search": title, "in": "title", "state": "all"})
        return next(
            (
                i
                for i in self.call("GET", f"{self.api}/issues?{query}")
                if i["title"] == title
            ),
            None,
        )

    def students(self, branch):
        """Users named under "Allowed to push and merge" in the branch's protection rule."""
        url = f"{self.api}/protected_branches/{urllib.parse.quote(branch, safe='')}"
        try:
            rule = self.call("GET", url)
        except urllib.error.HTTPError as err:
            if err.code == 404:  # branch not protected
                return []
            if err.code in (401, 403):
                print(
                    f"{branch}: DRIFT_TOKEN may not read branch protection; issue unassigned"
                )
                return []
            raise
        return [a["user_id"] for a in rule["push_access_levels"] if a.get("user_id")]

    def upsert(self, branch, body, in_sync):
        title = TITLE.format(branch=branch)
        issue = self.find_issue(title)
        if issue is None:
            if in_sync:
                return "in sync, no issue"
            data = {"title": title, "description": body, "labels": "thesis-drift"}
            students = self.students(branch)
            if students:
                data["assignee_ids"] = students
            return "opened #%d" % self.call("POST", f"{self.api}/issues", data)["iid"]
        data = {"description": body, "state_event": "close" if in_sync else "reopen"}
        self.call("PUT", f"{self.api}/issues/{issue['iid']}", data)
        return "updated #%d (%s)" % (issue["iid"], "closed" if in_sync else "open")


def main():
    dry_run = "--dry-run" in sys.argv or not os.environ.get("DRIFT_TOKEN")
    gitlab = None if dry_run else GitLab()
    for branch in thesis_branches():
        body, in_sync = issue_body(check_branch(branch))
        if dry_run:
            print(f"===== {TITLE.format(branch=branch)} (in sync: {in_sync})\n{body}")
        else:
            print(branch, "->", gitlab.upsert(branch, body, in_sync))


if __name__ == "__main__":
    main()
