"""No premature wins: best-so-far curves require matched completed seed prefixes."""
import csv
import json
from pathlib import Path

import pytest

from maxkcut import experiment025_seed_progress as progress
from maxkcut.experiment024_paired_gw16 import METHODS


def test_seed_progress_strictly_matched_prefixes_and_topology(tmp_path,monkeypatch):
    monkeypatch.setattr(progress,"load_references",lambda:(
        {g:{"reference_cut":1000.} for g in ("G1","G14","G11")},{}
    ))
    base=tmp_path/"pilot"
    for g in ("G1","G14","G11"):
        for rank,m in enumerate(METHODS):
            path=base/g/m
            path.mkdir(parents=True)
            for s in [0,1] if m=="gw-rr" else [0]:
                document={"graph":g,"method":m,"index":s,
                          "best_seen_cut":float(100+rank+s),
                          "stages":{"Round":60.,"Flip":80.,"KL":90.,
                                    "Final":float(100+rank+s)}}
                (path/f"{s:04d}.json").write_text(json.dumps(document))
                (path/f"{s:04d}.npz").write_bytes(b"example archive")
    report=progress.cumulative(root=tmp_path,phase="pilot")
    assert report["graphs"]==3
    assert report["eligible_seed_prefixes"]["1"]["completed_graph_method_prefixes"]==54
    assert report["eligible_seed_prefixes"]["1"]["fully_paired_graphs"]==3
    assert report["eligible_seed_prefixes"]["2"]["completed_graph_method_prefixes"]==3
    assert report["eligible_seed_prefixes"]["2"]["fully_paired_graphs"]==0
    assert not report["final_full_coverage"]
    with (tmp_path/"analysis/pilot_seed_progress.csv").open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==3*18*16
    assert sum(int(r["tied_graph_best"]) for r in rows if r["seed_prefix"]=="1")==3
    # No unequal seed counts masquerading as a paired method win.
    assert all(r["tied_graph_best"]=="" for r in rows if r["seed_prefix"]=="2")
    assert all(r["best_seen_cut"]=="" for r in rows
               if r["seed_prefix"]=="2" and r["method"]!="gw-rr")
    assert (tmp_path/"analysis/pilot_seed_completion.csv").is_file()


def test_progress_refuses_missing_saved_partition(tmp_path,monkeypatch):
    monkeypatch.setattr(progress,"load_references",lambda:(
        {g:{"reference_cut":1000.} for g in ("G1","G14","G11")},{}
    ))
    path=tmp_path/"pilot"/"G1"/"gw-rr"
    path.mkdir(parents=True)
    (path/"0000.json").write_text(json.dumps({
        "graph":"G1","method":"gw-rr","index":0,
        "best_seen_cut":10., "stages":{"Final":10.}}))
    with pytest.raises(ValueError,match="Missing binary"):
        progress.cumulative(root=tmp_path,phase="pilot")


def test_progress_refuses_wrong_best_seen_cut(tmp_path,monkeypatch):
    monkeypatch.setattr(progress,"load_references",lambda:(
        {g:{"reference_cut":1000.} for g in ("G1","G14","G11")},{}
    ))
    path=tmp_path/"pilot"/"G1"/"gw-rr"
    path.mkdir(parents=True)
    (path/"0000.json").write_text(json.dumps({
        "graph":"G1","method":"gw-rr","index":0,
        "best_seen_cut":11., "stages":{"Final":10.}}))
    (path/"0000.npz").write_bytes(b"invalid fixture")
    with pytest.raises(ValueError,match="highest signed cut"):
        progress.cumulative(root=tmp_path,phase="pilot")
