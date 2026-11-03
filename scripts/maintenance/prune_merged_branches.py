#!/usr/bin/env python3
"""Prune only SHA-pinned, already-merged GitHub PR branches.

Default = read-only dry-run. --execute is explicitly destructive.
Never deletes main, experiment025-results, or unmerged experimental work.
Requires Git and a network-capable, authenticated 'origin' on --execute.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "notes/maintenance/branch-cleanup-inventory.json"
PROTECTED = frozenset(("main", "experiment025-results"))


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    if check and proc.returncode:
        raise RuntimeError(
            f"git {args[0]} failed ({proc.returncode}): {proc.stderr[-1000:]}"
        )
    return proc


def remote_refs() -> dict[str, str]:
    result = git("ls-remote", "--heads", "origin")
    result_map = {}
    for line in result.stdout.splitlines():
        sha, ref = line.split(None, 1)
        if ref.startswith("refs/heads/"):
            result_map[ref[len("refs/heads/"):]] = sha
    return result_map


def is_ancestor(older: str, newer: str) -> bool:
    proc = git("merge-base", "--is-ancestor", older, newer, check=False)
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"merge-base failed: {proc.stderr[-500:]}")
    return proc.returncode == 0


def inspect(candidates: list[dict], refs: dict[str, str]) -> list[dict]:
    outcome = []
    for item in candidates:
        branch, expected_sha = item["branch"], item["expected_sha"]
        row = {"branch": branch, "expected_sha": expected_sha,
               "actual_sha": refs.get(branch), "merged_prs": item["merged_prs"],
               "status": None}
        if branch in PROTECTED or not item["merged_prs"]:
            row["status"] = "PROTECTED_OR_NO_MERGED_PR"
        elif not row["actual_sha"]:
            row["status"] = "ALREADY_ABSENT"
        elif row["actual_sha"] != expected_sha:
            row["status"] = "TIP_CHANGED_SKIP"
        elif not (is_ancestor(expected_sha, "origin/main")
                  or is_ancestor(expected_sha, "origin/experiment025-results")):
            # A merged PR may have been squash/rebase merged, or advanced
            # afterward. Preserve this unique exact tip under a permanent
            # Git tag BEFORE any branch deletion.
            row["status"] = "REQUIRES_ARCHIVE_TAG"
            row["archive_tag"] = f"archive/merged/{branch}"
        else:
            row["status"] = "SAFE_TO_DELETE"
        outcome.append(row)
    return outcome


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="Delete only safe, unchanged, merged branches")
    parser.add_argument("--audit-output", type=Path, default=None,
                        help="Optional JSON audit output path")
    args = parser.parse_args(argv)

    data = json.loads(INVENTORY.read_text(encoding="utf-8"))
    candidates = data["merged_candidates"]
    preserve = {item["branch"] for item in data["unmerged_preserve"]}
    assert len(candidates) == 62, "Reaudit candidate set before deleting"
    assert len(preserve) == 6, "Reaudit preservation set before deleting"
    assert len({x["branch"] for x in candidates}) == len(candidates)
    assert all(x["branch"] not in PROTECTED | preserve for x in candidates)
    assert data["repository"] == "heri-espino/maxkcut"

    origin = git("remote", "get-url", "origin").stdout.strip().lower()
    if "heri-espino/maxkcut" not in origin:
        raise RuntimeError("Unexpected origin: refusing deletion")

    # Full fetch is necessary to check reachability of historical merge tips.
    git("fetch", "--no-tags", "origin",
        "+refs/heads/main:refs/remotes/origin/main",
        "+refs/heads/experiment025-results:refs/remotes/origin/experiment025-results")
    initial = remote_refs()
    if any(p not in initial for p in PROTECTED | preserve):
        raise RuntimeError("A protected or unmerged historical branch is absent: abort")

    outcome = inspect(candidates, initial)
    if args.execute:
        for row in outcome:
            if row["status"] not in ("SAFE_TO_DELETE", "REQUIRES_ARCHIVE_TAG"):
                continue
            name = row["branch"]
            expected = row["expected_sha"]
            # Compare again, immediately before changing any refs. Never
            # delete a tip changed by another author since the inventory.
            latest = git("ls-remote", "--heads", "origin", f"refs/heads/{name}")
            parts = latest.stdout.split()
            if len(parts) != 2 or parts[0] != expected:
                row["status"] = "CHANGED_DURING_EXECUTION_SKIP"
                continue
            if row["status"] == "REQUIRES_ARCHIVE_TAG":
                # Preserve otherwise-unreachable history. A lightweight tag
                # points to the EXACT original branch commit. Never move or
                # overwrite an existing archive tag.
                tag = row["archive_tag"]
                ref = f"refs/tags/{tag}"
                prior = git("ls-remote", "--tags", "origin", ref).stdout.split()
                if prior and (len(prior) != 2 or prior[0] != expected):
                    row["status"] = "ARCHIVE_TAG_CONFLICT_SKIP"
                    continue
                if not prior:
                    proc = git("push", "origin", f"{expected}:{ref}",
                               check=False)
                    if proc.returncode:
                        row["archive_push_stderr"] = proc.stderr[-500:]
                receipt = git("ls-remote", "--tags", "origin", ref).stdout.split()
                if len(receipt) != 2 or receipt[0] != expected:
                    row["status"] = "ARCHIVE_TAG_NOT_CONFIRMED_SKIP"
                    continue
                row["archive_verified"] = True
                # A collision or writer could have advanced the branch
                # during tag publication; verify the original SHA once more.
                post_tag = git("ls-remote", "--heads", "origin",
                               f"refs/heads/{name}").stdout.split()
                if len(post_tag) != 2 or post_tag[0] != expected:
                    row["status"] = "CHANGED_DURING_EXECUTION_SKIP"
                    continue
            proc = git("push", "origin", "--delete", f"refs/heads/{name}",
                       check=False)
            # As with Git LFS publishing, a transport error after the ref is
            # deleted is ambiguous. Check the actual remote ref receipt.
            remaining = git("ls-remote", "--heads", "origin",
                             f"refs/heads/{name}").stdout.strip()
            if not remaining:
                row["status"] = "DELETED"
            else:
                row["status"] = "DELETE_FAILED"
                row["error"] = proc.stderr[-500:]
            print(f"{row['status']} {name}", flush=True)

    final = remote_refs()
    summary = {
        "mode": "execute" if args.execute else "dry-run",
        "candidates": len(candidates),
        "deleted": sum(x["status"] == "DELETED" for x in outcome),
        "already_absent": sum(x["status"] == "ALREADY_ABSENT" for x in outcome),
        "safe_to_delete_remaining": sum(x["status"] == "SAFE_TO_DELETE" for x in outcome),
        "archive_tags_needed": sum(x["status"] == "REQUIRES_ARCHIVE_TAG" for x in outcome),
        "archive_tags_verified": sum(bool(x.get("archive_verified")) for x in outcome),
        "skipped_or_failed": [x for x in outcome if x["status"] not in (
            "DELETED", "ALREADY_ABSENT", "SAFE_TO_DELETE",
            "REQUIRES_ARCHIVE_TAG")],
        "protected_present": sorted(PROTECTED.intersection(final)),
        "preserved_research_present": sorted(preserve.intersection(final)),
        "remaining_remote_branches": len(final),
        "details": outcome,
    }
    txt = json.dumps(summary, indent=2) + "\n"
    print(txt)
    if args.audit_output is not None:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(txt, encoding="utf-8")
    if args.execute and summary["skipped_or_failed"]:
        return 1
    if summary["protected_present"] != sorted(PROTECTED):
        return 2
    if summary["preserved_research_present"] != sorted(preserve):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
