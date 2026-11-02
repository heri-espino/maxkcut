"""Experiment025 execution preflight: source pin, scoped LFS and fixed study size."""
from pathlib import Path
import subprocess
from maxkcut.experiment025_bestcut import (
    algorithm_source_sha256, fingerprint, GRAPH_IDS, METHODS
)


def test_source_hash_stable_across_checkpoint_commits():
    a=algorithm_source_sha256()
    b=algorithm_source_sha256()
    assert a==b and len(a)==64
    assert len(GRAPH_IDS)==71 and len(METHODS)==18
    config=dict(starts=16,kl_passes=30,ils_cycles=80,ils_kl_passes=8)
    f=fingerprint("graph","factor","directions","gw-rr","full",config)
    assert len(f)==24
    assert f==fingerprint("graph","factor","directions","gw-rr","full",config)
    assert f!=fingerprint("graph","factor-changed","directions","gw-rr","full",config)


def test_source_mutation_invalidates_fingerprint(monkeypatch):
    from maxkcut import experiment025_bestcut as model
    config=dict(starts=16)
    old=model.fingerprint("g","f","d","gw-rr","full",config)
    monkeypatch.setattr(model,"algorithm_source_sha256",lambda: "a"*64)
    new=model.fingerprint("g","f","d","gw-rr","full",config)
    assert old!=new


def test_lfs_attributes_scope_only_new_result_files():
    root=Path(__file__).resolve().parents[1]
    path=root/".gitattributes"
    content=path.read_text(encoding="utf-8")
    assert "experiment025-gw16-bestcut-quality/**/*.npz filter=lfs" in content
    assert "experiment025-gw16-bestcut-quality/**/*.jsonl filter=lfs" in content
    assert "experiment021" not in content
    if (root/".git").exists():
        rel="results/gset/experiment025-gw16-bestcut-quality/full/G11/gw-rr/0000.npz"
        cp=subprocess.run(["git","check-attr","filter","--",rel],
                          cwd=root,text=True,capture_output=True,check=True)
        assert cp.stdout.strip().endswith("filter: lfs")


def test_new_vm_runner_is_cleanly_resumable_and_manual_only():
    root=Path(__file__).resolve().parents[1]
    runner=(root/"scripts"/"experiment025-vm.ps1").read_text(encoding="utf-8")
    workflow=(root/".github"/"workflows"/"experiment025-quality-first.yml").read_text(encoding="utf-8")
    assert "experiment025-results" in runner
    assert "git lfs install --local" in runner
    assert "Publish-Checkpoint" in runner
    assert "conda activate maxkcut" in runner
    assert "experiment025-gw16-bestcut-quality" not in workflow.split("jobs:")[0]
    assert "workflow_dispatch:" in workflow
    assert "push:" not in workflow.split("jobs:")[0]
    assert "ref: ${{ github.sha }}" in workflow  # stronger: exact dispatch revision
    assert "-Method $env:EXPERIMENT_METHOD" in workflow
