"""GitHub Release uploader: SHA verification, immutable versions, no LFS."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "experiment027_publish_release", ROOT / "scripts" / "experiment027_publish_release.py"
)
rel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rel)


def test_release_asset_remote_sha_and_size_verified(tmp_path):
    arc = tmp_path / "G1-immutable.zip"
    arc.write_bytes(b"tiny but signed")
    sha = rel.legacy.digest(arc)
    asset = dict(name=arc.name, state="uploaded", size=arc.stat().st_size, digest="sha256:"+sha)
    rel.check_asset(asset, arc, sha)
    rel.verify_uploaded({"assets": [asset]}, [(arc, sha)])
    with pytest.raises(ValueError, match="SHA256"):
        rel.check_asset(dict(asset, digest="sha256:"+"0"*64), arc, sha)
    with pytest.raises(ValueError, match="size"):
        rel.check_asset(dict(asset, size=1000), arc, sha)
    with pytest.raises(FileNotFoundError, match="asset"):
        rel.verify_uploaded({"assets": []}, [(arc, sha)])


def test_immutable_release_index_and_manifest(tmp_path, monkeypatch):
    graph = "G1"
    archive = tmp_path / "rawassets" / f"{graph}-abc.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"verified ZIP representative")
    checksum = rel.legacy.digest(archive)
    manifest = tmp_path / "rawassets" / f"{graph}-abc.json"
    manifest.write_text(json.dumps(dict(graph=graph, zip_file="results/staging/temp.zip",
                                         zip_sha256=checksum, member_count=2,
                                         member_hashes=[dict(path="full/G1/0000.json", sha256="a"*64)])))
    audit = tmp_path / "sourceanalysis"
    audit.mkdir()
    for name in rel.legacy.AUDIT_FILES:
        if name == "audited_comparison.json":
            (audit / name).write_text(json.dumps(dict(status="independently_verified_archives",
                                                       graphs=1, analysis_directory="C:/private/vm/path")))
        else:
            (audit / name).write_text("header\n1\n")
    monkeypatch.setattr(rel, "HERE", tmp_path)
    tag = "experiment027-data-1g-123456"
    folder, index = rel.publish_index(
        tag, [(graph, archive, manifest, checksum)],
        tmp_path / "published/experiment027/releases",
        audit, ["G1"]
    )
    assert index["graph_count"] == 1
    assert index["complete_new_seeds"] == 48
    assert index["assets"][0]["url"].startswith("https://github.com/")
    published_manifest = json.loads(next((folder/"manifests").glob("*.json")).read_text())
    assert published_manifest["member_hashes"][0]["sha256"] == "a"*64
    assert "staging" not in published_manifest["zip_file"]
    result = json.loads((folder/"analysis"/"audited_comparison.json").read_text())
    assert "private/vm" not in result["analysis_directory"]


def test_no_git_and_no_release_for_dry_run(tmp_path, monkeypatch, capsys):
    import experiment027_publish as legacy
    folder = tmp_path / "new" / "full" / "G1"
    for method in legacy.METHODS:
        d = folder / method
        (d/"progress"/"0000").mkdir(parents=True)
        (d/"status.json").write_text(json.dumps({"status": "complete_16", "starts_completed":16}))
        for i in range(16):
            (d/f"{i:04d}.json").write_text("{}")
            (d/f"{i:04d}.npz").write_bytes(b"dummy")
            (d/"progress"/f"{i:04d}").mkdir(exist_ok=True)
            (d/"progress"/f"{i:04d}"/"events.jsonl").write_text("{}\n")
    (folder/"gw16_common.npz").write_bytes(b"fake")
    (folder/"gw16_common.json").write_text("{}")
    monkeypatch.setattr(rel, "repo_info", lambda: pytest.fail("auth not necessary for dry-run"))
    monkeypatch.setattr(rel, "run", lambda *a, **k: pytest.fail("dry-run cannot call Git or optimization"))
    assert rel.main(["--graphs","G1","--dry-run","--source-root",str(tmp_path/"new")]) == 0
    assert "GitHub Release" in capsys.readouterr().out


def test_validates_graph_ids_before_any_computation(tmp_path, monkeypatch):
    monkeypatch.setattr(rel.legacy, "validate_graph", lambda *a: pytest.fail("invalid arguments should fail first"))
    with pytest.raises(SystemExit):
        rel.main(["--graphs", "G14", "G14", "--dry-run", "--source-root", str(tmp_path)])


def test_tag_is_stable_for_archive_list():
    a = [("G1","a"*64),("G11","b"*64)]
    b = [("G1","a"*64),("G11","b"*64)]
    def tag(data):
        import hashlib
        return hashlib.sha256(rel.stable_bytes(data)).hexdigest()[:20]
    assert tag(a) == tag(b)
    assert tag(a) != tag(a[::-1])


@pytest.mark.parametrize("checkout", ["experiment027-angular-greedy", "exp027-local"])
def test_repo_info_allows_vm_worktree_without_switch_when_fast_forward_safe(monkeypatch, checkout):
    from types import SimpleNamespace
    calls = []
    def fake_run(args, **kwargs):
        calls.append(tuple(args))
        if args[:3] == ["git", "branch", "--show-current"]:
            return SimpleNamespace(stdout=checkout + "\n", stderr="", returncode=0)
        if args[:3] == ["git", "rev-parse", "refs/remotes/origin/experiment027-angular-greedy"]:
            return SimpleNamespace(stdout="a" * 40 + "\n", stderr="", returncode=0)
        return SimpleNamespace(stdout="", stderr="", returncode=0)
    monkeypatch.setattr(rel, "run", fake_run)
    monkeypatch.setattr(rel.shutil, "which", lambda name: "C:/portable/gh.exe")
    assert rel.repo_info() == "C:/portable/gh.exe"
    assert ("git", "fetch", "origin", "experiment027-angular-greedy") in calls
    assert ("git", "merge-base", "--is-ancestor", "a" * 40, "HEAD") in calls
    assert all("reset" not in cmd and "switch" not in cmd and "checkout" not in cmd for cmd in calls)


def test_repo_info_blocks_vm_branch_without_remote_ancestry(monkeypatch):
    from types import SimpleNamespace
    def fake_run(args, **kwargs):
        if args[:3] == ["git", "branch", "--show-current"]:
            return SimpleNamespace(stdout="exp027-local\n", stderr="", returncode=0)
        if args[:3] == ["git", "rev-parse", "refs/remotes/origin/experiment027-angular-greedy"]:
            return SimpleNamespace(stdout="b" * 40 + "\n", stderr="", returncode=0)
        if args[:3] == ["git", "merge-base", "--is-ancestor"]:
            return SimpleNamespace(stdout="", stderr="", returncode=1)
        return SimpleNamespace(stdout="", stderr="", returncode=0)
    monkeypatch.setattr(rel, "run", fake_run)
    with pytest.raises(RuntimeError, match="does not contain"):
        rel.repo_info()


def test_repo_info_rejects_other_worktrees(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(rel, "run", lambda args, **kwargs:
                        SimpleNamespace(stdout="experiment025-results\n", stderr="", returncode=0))
    with pytest.raises(RuntimeError, match="Expected Experiment027 checkout"):
        rel.repo_info()
