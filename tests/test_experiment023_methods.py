"""Experiment023 smoke/unit invariants (no EVA solves or large graph reads)."""
import json
import numpy as np
import pytest

from maxkcut import experiment023_methods as e


def test_complete_graph_and_method_registry():
    assert len(e.GRAPH_IDS) == 71
    assert len(set(e.GRAPH_IDS)) == 71
    assert e.GRAPH_IDS[-4:] == ("G67", "G70", "G72", "G77", "G81")[-4:]
    assert len(e.METHODS) == 17
    assert len(set(e.METHODS)) == 17
    assert "em-cem" in e.METHODS and "mst-l2-multicut" in e.METHODS


def test_fingerprint_changes_with_method_factor_and_budget():
    conf=e.config("pilot")
    a=e.fingerprint("ga", "fa", "em-hard", "pilot", conf)
    assert len(a) == 20
    assert a != e.fingerprint("ga", "fa", "em-gibbs", "pilot", conf)
    assert a != e.fingerprint("ga", "fb", "em-hard", "pilot", conf)
    assert a != e.fingerprint("ga", "fa", "em-hard", "full", e.config("full"))
    assert a == e.fingerprint("ga", "fa", "em-hard", "pilot", conf)


def test_factor_selection_never_modifies_or_guesses(tmp_path):
    sources=[tmp_path/"one",tmp_path/"two"]
    for folder in sources:
        (folder/"cases").mkdir(parents=True)
        (folder/"factors").mkdir()
    with pytest.raises(FileNotFoundError):e.select_factor("G1", sources)
    (sources[1]/"cases"/"G1.json").write_text("{}")
    with pytest.raises(FileNotFoundError):e.select_factor("G1", sources)
    (sources[1]/"factors"/"G1.npz").write_bytes(b"dummy")
    assert e.select_factor("G1", sources)==sources[1]


def test_atomic_archive_requires_exact_factor_identity_and_rescoring(tmp_path):
    from scipy import sparse
    graph=sparse.csr_matrix(np.array([[0.,2.,-1.],[2.,0.,3.],[-1.,3.,0.]]))
    edges=(np.array([0,0,1]),np.array([1,2,2]),np.array([2.,-1.,3.]))
    labels=np.array([0,1,0],dtype=np.int8)
    score=e._score(edges,labels)
    info={"fingerprint":"sig","factor_sha256":"factor", "graph_sha256":"graph"}
    e.write_case(tmp_path,"em-hard",0,labels,{"Round":score,"Flip":score,"KL":score,"Final":score},info,
                 {"Round":labels,"Flip":labels,"KL":labels,"Final":labels})
    assert e.verify_case(tmp_path,"em-hard",0,graph,edges,"sig") is not None
    with pytest.raises(ValueError):e.verify_case(tmp_path,"em-hard",0,graph,edges,"wrong")
    file=tmp_path/"em-hard"/"0000.json"
    saved=json.loads(file.read_text());saved["stages"]["Final"]=score+1
    e.atomic_json(file,saved)
    with pytest.raises(ValueError):e.verify_case(tmp_path,"em-hard",0,graph,edges,"sig")


def test_not_run_status_is_never_implicit_success(tmp_path):
    s=e.aggregate(tmp_path,"full")
    assert s["complete_graph_method_rows"]==0
    assert s["status_counts"]["not_run"]==71*len(e.METHODS)
    assert (tmp_path/"analysis"/"full_coverage.csv").is_file()
