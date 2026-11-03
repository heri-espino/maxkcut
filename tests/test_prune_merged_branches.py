"""Regression tests for fail-closed, SHA-pinned branch cleanup.

No GitHub API calls, remote branch deletions, or network requests occur.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def script_module():
    p = (Path(__file__).resolve().parents[1] / "scripts" /
         "maintenance" / "prune_merged_branches.py")
    spec = importlib.util.spec_from_file_location("prune_merged_branches", p)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_has_exact_authorized_scope():
    module = script_module()
    import json

    data = json.loads(module.INVENTORY.read_text(encoding="utf-8"))
    candidates = data["merged_candidates"]
    preserved = data["unmerged_preserve"]
    assert len(candidates) == 62
    assert len(preserved) == 6
    names = [x["branch"] for x in candidates]
    assert len(names) == len(set(names))
    assert not set(names).intersection(module.PROTECTED)
    assert not set(names).intersection(x["branch"] for x in preserved)
    assert all(len(x["expected_sha"]) == 40 and x["merged_prs"]
               for x in candidates)


def test_unchanged_merged_tip_required(monkeypatch):
    module = script_module()
    monkeypatch.setattr(module, "is_ancestor", lambda *args: True)
    row = dict(branch="feature/merged", expected_sha="a" * 40,
               merged_prs=[9])
    matches = module.inspect([row], {"feature/merged": "a" * 40})
    assert matches[0]["status"] == "SAFE_TO_DELETE"
    changed = module.inspect([row], {"feature/merged": "b" * 40})
    assert changed[0]["status"] == "TIP_CHANGED_SKIP"
    missing = module.inspect([row], {})
    assert missing[0]["status"] == "ALREADY_ABSENT"


def test_unreachable_branch_never_deleted(monkeypatch):
    module = script_module()
    monkeypatch.setattr(module, "is_ancestor", lambda *args: False)
    rows = module.inspect([dict(branch="feature/unmerged",
                                expected_sha="a" * 40,
                                merged_prs=[10])],
                          {"feature/unmerged": "a" * 40})
    assert rows[0]["status"] == "REQUIRES_ARCHIVE_TAG"
    assert rows[0]["archive_tag"] == "archive/merged/feature/unmerged"


def test_default_execution_dry_run_and_never_push(monkeypatch, capsys):
    module = script_module()
    import json

    inventory = json.loads(module.INVENTORY.read_text(encoding="utf-8"))
    refs = {item["branch"]: item["expected_sha"]
            for item in inventory["merged_candidates"] + inventory["unmerged_preserve"]}
    refs["main"] = "f" * 40
    refs["experiment025-results"] = "e" * 40

    commands = []

    def git(*args, **kwargs):
        commands.append(args)
        if args[:2] == ("remote", "get-url"):
            return SimpleNamespace(stdout="https://github.com/heri-espino/maxkcut.git")
        if args[0] == "fetch":
            return SimpleNamespace(stdout="")
        raise AssertionError(f"unexpected dry-run git action: {args}")

    monkeypatch.setattr(module, "git", git)
    monkeypatch.setattr(module, "remote_refs", lambda: refs)
    monkeypatch.setattr(module, "is_ancestor", lambda *a: True)
    assert module.main([]) == 0
    assert all(x[0] not in ("push",) for x in commands)
    assert '"mode": "dry-run"' in capsys.readouterr().out


def test_protected_branch_absent_aborts_all_deletions(monkeypatch):
    module = script_module()
    import json
    inventory = json.loads(module.INVENTORY.read_text(encoding="utf-8"))
    refs = {item["branch"]: item["expected_sha"]
            for item in inventory["merged_candidates"] + inventory["unmerged_preserve"]}
    refs["main"] = "a" * 40  # experiment025-results absent.

    def git(*args, **kwargs):
        if args[:2] == ("remote", "get-url"):
            return SimpleNamespace(stdout="https://github.com/heri-espino/maxkcut")
        if args[0] == "fetch":
            return SimpleNamespace(stdout="")
        raise AssertionError(args)

    monkeypatch.setattr(module, "git", git)
    monkeypatch.setattr(module, "remote_refs", lambda: refs)
    with pytest.raises(RuntimeError, match="protected or unmerged"):
        module.main(["--execute"])


def test_unreachable_branch_archived_before_deletion(monkeypatch):
    """An otherwise orphaned tip cannot disappear without a verified tag."""
    module = script_module()
    import json
    inventory = json.loads(module.INVENTORY.read_text(encoding="utf-8"))
    merged = inventory["merged_candidates"]
    kept = inventory["unmerged_preserve"]
    refs = {x["branch"]: x["expected_sha"] for x in merged + kept}
    refs["main"] = "f" * 40
    refs["experiment025-results"] = "e" * 40
    tags = {}
    trace = []

    def fake_git(*args, **kwargs):
        trace.append(args)
        if args[:2] == ("remote", "get-url"):
            return SimpleNamespace(stdout="https://github.com/heri-espino/maxkcut")
        if args[0] == "fetch":
            return SimpleNamespace(stdout="", returncode=0)
        if args[:2] == ("ls-remote", "--tags"):
            tag = args[-1]
            value = tags.get(tag, "")
            return SimpleNamespace(stdout=(f"{value}\t{tag}\n" if value else ""))
        if args[:2] == ("ls-remote", "--heads"):
            b = args[-1].removeprefix("refs/heads/")
            v = refs.get(b)
            return SimpleNamespace(stdout=(f"{v}\trefs/heads/{b}\n" if v else ""))
        if args[:2] == ("push", "origin"):
            if args[2] == "--delete":
                name = args[3].removeprefix("refs/heads/")
                assert any(x[0:2] == ("ls-remote", "--tags") for x in trace), (
                    "no deletion until archival receipt exists")
                refs.pop(name, None)
                return SimpleNamespace(stdout="", stderr="", returncode=0)
            sha, tag = args[2].split(":", 1)
            assert tag not in tags
            tags[tag] = sha
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        raise AssertionError(args)

    monkeypatch.setattr(module, "git", fake_git)
    monkeypatch.setattr(module, "remote_refs", lambda: dict(refs))
    monkeypatch.setattr(module, "is_ancestor", lambda *a: False)
    assert module.main(["--execute"]) == 0
    assert len(tags) == 62
    assert all(t.startswith("refs/tags/archive/merged/") for t in tags)
    assert len(refs) == 8
    assert refs["main"] == "f" * 40
    assert refs["experiment025-results"] == "e" * 40
    assert {x["branch"] for x in kept}.issubset(refs)
