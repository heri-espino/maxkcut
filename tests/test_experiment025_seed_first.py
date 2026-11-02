"""Experiment025 seed-first invariants: 16 individually dispatched paired GW runs."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from maxkcut import experiment025_bestcut as e
from scripts.experiment025_actions_matrix import (
    GRAPHS, METHODS, METHOD_BATCHES, PILOT_GRAPHS, matrix,
)


def test_seed_first_batches_cover_all_graphs_and_methods_exactly_once():
    cases=[]
    pilot=[]
    for batch in range(6):
        full=matrix("seed-batch",batch=batch,phase="full")["include"]
        short=matrix("seed-batch",batch=batch,phase="pilot")["include"]
        assert len(full)==213
        assert len(short)==9
        assert set(x["graph"] for x in full)==set(GRAPHS)
        assert set(x["method"] for x in full)==set(METHOD_BATCHES[batch])
        pilot.extend((x["graph"],x["method"]) for x in short)
        cases.extend((x["graph"],x["method"]) for x in full)
    assert len(cases)==1278 and len(set(cases))==1278
    assert set(cases)=={(g,m) for g in GRAPHS for m in METHODS}
    assert len(pilot)==54 and len(set(pilot))==54
    assert set(pilot)=={(g,m) for g in PILOT_GRAPHS for m in METHODS}


def test_seed_first_invalid_matrix_rejected():
    for batch in (-1,6,100):
        with pytest.raises(ValueError):
            matrix("seed-batch",batch=batch)
    with pytest.raises(ValueError):
        matrix("seed-batch",phase="other")


def test_status_uses_real_seed_index_not_completion_order():
    docs=[
        {"index":12,"best_seen_cut":12., "stages":{"Round":5.,"Final":12.}},
        {"index":0,"best_seen_cut":23., "stages":{"Round":8.,"Final":23.}},
    ]
    r=e._status("G11","gw-rr","full","test",2,docs,0.)
    assert r["starts_completed"]==2
    assert r["completed_seed_indices"]==[0,12]
    assert r["current_best_seed_index"]==0
    assert r["selected_seed_index"] is None
    with pytest.raises(ValueError,match="Duplicate"):
        e._status("G11","gw-rr","full","test",2,[docs[0],docs[0]],0.)


def test_individual_seed_recovery_out_of_order(tmp_path,monkeypatch):
    rng=np.random.default_rng(17)
    y=rng.standard_normal((8,4))
    y/=np.linalg.norm(y,axis=1,keepdims=True)
    dense=np.zeros((8,8))
    for i in range(8):
        for j in range(i+1,8):
            if (i+j)%3:
                dense[i,j]=1. if (i+j)%2 else -1.
                dense[j,i]=dense[i,j]
    graph=sparse.csr_matrix(dense)
    graph.sort_indices()
    monkeypatch.setattr(e,"select_factor",lambda g:tmp_path/"frozen")
    monkeypatch.setattr(e,"_factor_and_case",
        lambda path,folder:(graph,y,None,"graph_sha","factor_sha"))
    monkeypatch.setattr(e,"load_references",
        lambda:({"G11":{"reference_cut":50.,"reference_kind":"historical"}},{}))
    monkeypatch.setattr(e,"compare_cut",lambda value,record:{"verified_cut":value})
    cfg=dict(starts=16,kmeans_iters=2,direction_trials=1,
             line_directions=1,em_stages=1,em_iters=1,
             cem_generations=1,cem_population=4,gibbs_sweeps=1,
             kl_passes=2,ils_cycles=1,ils_kl_passes=1)
    monkeypatch.setitem(e.SETTINGS,"pilot",cfg)
    folder=tmp_path/"data"
    for requested,expected in [(11,[11]),(0,[0,11]),(15,[0,11,15])]:
        row=e.run_method("G11","gw-rr","pilot",root=folder,seed_index=requested)
        assert row["completed_seed_indices"]==expected
        assert row["starts_completed"]==len(expected)
        assert row["status"]=="partially_refined"
        assert row["best_seen"]>=row["best_initial"]
    before=(folder/"pilot"/"G11"/"gw-rr"/"0011.npz").read_bytes()
    again=e.run_method("G11","gw-rr","pilot",root=folder,seed_index=11)
    assert again["starts_completed"]==3
    assert (folder/"pilot"/"G11"/"gw-rr"/"0011.npz").read_bytes()==before
    # Complete arbitrary missing seeds with the existing sequential batch mode.
    for _ in range(4):
        final=e.run_method("G11","gw-rr","pilot",root=folder,max_new_seeds=4)
    assert final["status"]=="complete_16_best_seen"
    assert final["starts_completed"]==16
    assert final["completed_seed_indices"]==list(range(16))
    assert final["selected_seed_index"] in range(16)
    all_cases=[
        json.loads((folder/"pilot"/"G11"/"gw-rr"/f"{i:04d}.json").read_text())
        for i in range(16)
    ]
    assert final["best_seen"]==max(r["best_seen_cut"] for r in all_cases)
    assert final["selected_seed_index"]==max(
        range(16),key=lambda i:all_cases[i]["best_seen_cut"]
    )


def test_seed_first_workflow_pins_source_and_serializes_method_shards():
    root=Path(__file__).resolve().parents[1]
    parent=(root/".github/workflows/experiment025-seed-round.yml").read_text()
    shard=(root/".github/workflows/experiment025-seed-shard.yml").read_text()
    ps=(root/"scripts/experiment025-vm.ps1").read_text()
    assert "workflow_dispatch:" in parent
    assert "queue: max" in parent
    assert "runs-on: [self-hosted, Windows, X64]" in shard
    assert "RUNNER_NAME -ne 'PCAG103A-13'" in shard
    assert "timeout-minutes: 360" in shard
    assert "max-parallel: 1" in shard
    assert "seed-batch" in shard
    assert "ExpectedSourceSha256" in shard
    assert "seed-index $seedIndex" in ps
    assert "Seed = 0" in ps
    for i in range(6):
        assert f"batch{i}:" in parent
    assert "fromJSON(inputs.seed)" in parent
