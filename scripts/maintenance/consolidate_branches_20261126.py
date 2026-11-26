#!/usr/bin/env python3
"""One-time, SHA-pinned remote branch consolidation. No numerical computation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
INV = ROOT / "notes/maintenance/branch-consolidation-20261126.json"
DATA = json.loads(INV.read_text())
REPO = DATA["repository"]
PROTECTED = {"main", "experiment025-results"}
assert REPO == "heri-espino/maxkcut" and set(DATA["protected"]) == PROTECTED
assert len({x["branch"] for x in DATA["candidates"]}) == len(DATA["candidates"])
assert not PROTECTED.intersection(x["branch"] for x in DATA["candidates"])

def cmd(*args, check=True):
    p = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    if check and p.returncode:
        raise RuntimeError(f"{args[0]} {args[1]} failed: {p.stderr[-800:]}")
    return p

def api(path, *args):
    return json.loads(cmd("gh", "api", path, *args).stdout)

def ref_sha(ref):
    lines = cmd("git", "ls-remote", "--refs", "origin", ref).stdout.splitlines()
    if not lines:
        return None
    assert len(lines) == 1 and lines[0].split()[1] == ref
    return lines[0].split()[0]

def open_prs():
    return api(f"repos/{REPO}/pulls?state=open&per_page=1")

def active_heads():
    result = set()
    for state in ("in_progress", "queued", "waiting", "requested", "pending"):
        pages = api(f"repos/{REPO}/actions/runs?status={state}&per_page=100",
                    "--paginate", "--slurp")
        for page in pages:
            result.update(run["head_branch"] for run in page["workflow_runs"])
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--execute", action="store_true")
    p.add_argument("--wait-seconds", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("branch-consolidation-audit.json"))
    args = p.parse_args()
    if os.environ.get("GITHUB_REPOSITORY") not in (None, REPO):
        raise RuntimeError("wrong repository")
    origin = cmd("git", "remote", "get-url", "origin").stdout.strip().lower()
    if "github.com/heri-espino/maxkcut" not in origin:
        raise RuntimeError("wrong remote")
    if open_prs():
        raise RuntimeError("Open PRs exist; refusing cleanup")
    assert all(ref_sha("refs/heads/" + name) for name in PROTECTED)
    candidates = {x["branch"] for x in DATA["candidates"]}
    # Allow already-running hosted checks to finish, without starting any new checks.
    deadline = time.monotonic() + args.wait_seconds
    while True:
        active = active_heads() & candidates
        if not active or time.monotonic() >= deadline:
            break
        print("Active branches: " + ", ".join(sorted(active)), flush=True)
        time.sleep(min(20, max(0, deadline - time.monotonic())))
    rows = []
    for row in DATA["candidates"]:
        row = dict(row)
        name, sha = row["branch"], row["sha"]
        branch_ref = "refs/heads/" + name
        tag_ref = "refs/tags/" + row["archive_tag"]
        actual = ref_sha(branch_ref)
        existing = ref_sha(tag_ref)
        if actual is None:
            row["status"] = "ALREADY_ARCHIVED" if existing == sha else "ABSENT_WITHOUT_VERIFIED_TAG"
        elif actual != sha:
            row["status"] = "TIP_CHANGED_SKIP"
        elif name in active:
            row["status"] = "ACTIVE_RUN_SKIP"
        elif existing not in (None, sha):
            row["status"] = "TAG_CONFLICT_SKIP"
        elif not args.execute:
            row["status"] = "READY"
        else:
            if existing is None:
                # REST creates a lightweight tag directly from the existing remote commit.
                cmd("gh", "api", "--method", "POST", f"repos/{REPO}/git/refs",
                    "-f", "ref=" + tag_ref, "-f", "sha=" + sha)
            if ref_sha(tag_ref) != sha:
                raise RuntimeError("Archive tag verification failed: " + name)
            row["archive_verified"] = True
            if open_prs():
                raise RuntimeError("A new PR appeared; refusing further deletion")
            if ref_sha(branch_ref) != sha:
                row["status"] = "TIP_CHANGED_SKIP"
            else:
                # Server-side lease rejects deletion if a concurrent writer advanced the branch.
                q = cmd("git", "push", "--force-with-lease=" + branch_ref + ":" + sha,
                        "origin", ":" + branch_ref, check=False)
                row["status"] = "DELETED" if ref_sha(branch_ref) is None else "DELETE_FAILED"
                if row["status"] == "DELETE_FAILED":
                    row["error"] = q.stderr[-800:]
        print(row["status"], name, flush=True)
        rows.append(row)
        args.output.write_text(json.dumps(rows, indent=2) + "\n")
    if not all(ref_sha("refs/heads/" + name) for name in PROTECTED):
        raise RuntimeError("Protected branch missing")
    print(json.dumps({"execute": args.execute, "deleted": sum(x["status"]=="DELETED" for x in rows),
                      "archived_already":sum(x["status"]=="ALREADY_ARCHIVED" for x in rows),
                      "preserved":sorted(PROTECTED), "results":rows}, indent=2))
    acceptable = {"DELETED", "ALREADY_ARCHIVED"} if args.execute else {"READY", "ALREADY_ARCHIVED"}
    return 0 if all(x["status"] in acceptable for x in rows) else 1

if __name__ == "__main__":
    raise SystemExit(main())
