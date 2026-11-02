"""Experiment025: reproducible iteration histories and best-EVER partitions."""
import json
import numpy as np
import pytest
from scipy import sparse

from maxkcut.io import edge_arrays
from maxkcut.experiment025_journal import IterationJournal, refine_with_journal
from maxkcut.experiment025_bestcut import _status


def graph():
    mat=np.array([[0,4,-1,2],[4,0,3,-1],[-1,3,0,2],[2,-1,2,0]],dtype=float)
    g=sparse.csr_matrix(mat)
    g.sort_indices()
    return g,edge_arrays(g)


def test_journal_retains_intermediate_best_not_last(tmp_path):
    g,edges=graph()
    x=np.array([0,1,0,1],dtype=np.int8)
    y=np.array([0,1,1,0],dtype=np.int8)
    journal=IterationJournal(tmp_path/"seed0000",edges,4,"fixture")
    journal.observe({"stage":"SoftEM","iteration":1},x)
    journal.observe({"stage":"SoftEM","iteration":2},y)
    best,score=journal.assert_best()
    assert score==max(journal._verify(x),journal._verify(y))
    journal.observe({"stage":"Gibbs","sweep":3,"current_cut":0.},
                    np.zeros(4,dtype=np.int8))
    best2,score2=journal.assert_best()
    assert score2==score
    assert np.array_equal(best2,best)
    assert journal.events==3
    logged=[json.loads(x) for x in (tmp_path/"seed0000"/"events.jsonl").read_text().splitlines()]
    assert len(logged)==3
    assert logged[-1]["best_so_far_cut"]==score
    reopened=IterationJournal(tmp_path/"seed0000",edges,4,"fixture")
    assert reopened.best_cut==score and reopened.events==3
    with pytest.raises(ValueError,match="identity"):
        IterationJournal(tmp_path/"seed0000",edges,4,"other")


def test_journal_rejects_invalid_partition(tmp_path):
    _,edges=graph()
    journal=IterationJournal(tmp_path/"checkpoint",edges,4,"id")
    with pytest.raises(ValueError,match="Invalid binary"):
        journal.observe({"stage":"Flip"},np.array([0,1,3,0]))
    with pytest.raises(ValueError,match="Invalid binary"):
        journal.observe({"stage":"Flip"},np.zeros(5,dtype=np.int8))


def test_selected_method_winner_is_best_seen_not_last(tmp_path):
    records=[
      dict(stages=dict(Round=10.,Flip=14.,KL=16.,Final=16.),best_seen_cut=16.),
      dict(stages=dict(Round=11.,Flip=12.,KL=13.,Final=13.),best_seen_cut=17.),
    ]
    status=_status("G11","em-gibbs","pilot","fp",2,records[0:2],0.)
    assert status["best_seen"]==17.
    assert status["best_final"]==17.
    assert status["selection_metric"].startswith("max_best_seen")
    # Only 2 of 16 starts: no false claim of completed benchmark.
    assert status["status"]=="partially_refined"
    assert status["selected_seed_index"] is None


def test_flip_kl_ils_callbacks_and_best_journal(tmp_path):
    from maxkcut.local_search import improve_by_flips
    from maxkcut.eva_elite_search import optimize_kl_fast,optimize_elite
    g,edges=graph()
    initial=np.array([0,0,1,1],dtype=np.int8)
    journal=IterationJournal(tmp_path/"full",edges,4,"algo")
    journal.observe({"stage":"GW","iteration":0},initial)
    flip,d=improve_by_flips(g,initial.copy(),np.random.default_rng(5),
                            max_sweeps=3,on_iteration=journal.observe)
    assert d["sweeps"]>=1
    k,kd=optimize_kl_fast(g,flip.copy(),np.random.default_rng(4),
                          passes=3,on_iteration=journal.observe)
    fin,trace=optimize_elite(g,k,[(k,journal._verify(k))],
            np.random.default_rng(3),method="ILS-KL",cycles=2,
            kl_passes=2,kick_sizes=(1,2),on_iteration=journal.observe)
    j=journal.summary()
    names={json.loads(z)["stage"] for z in
           (tmp_path/"full"/"events.jsonl").read_text().splitlines()}
    assert {"GW","Flip","KL","ILS"}<=names
    assert j["best_seen_cut"]>=max(
        journal._verify(initial),journal._verify(flip),
        journal._verify(k),journal._verify(fin))
    assert len(trace["trace"])==2


def test_refinement_records_all_three_stages(tmp_path):
    g,edges=graph()
    initial=np.array([0,0,1,1],dtype=np.int8)
    journal=IterationJournal(tmp_path/"case",edges,4,"case")
    cfg=dict(kl_passes=2,ils_cycles=2,ils_kl_passes=2)
    final,stages,diagnostics,parts=refine_with_journal(
        g,edges,initial,"gw-rr",0,11,cfg,journal)
    assert set(stages)=={"Round","Flip","KL","Final"}
    assert stages["Final"]>=max(stages.values())
    assert journal._verify(parts["Final"])==stages["Final"]
    assert diagnostics["iteration_events"]>=4
    assert diagnostics["best_partition_improvements"]>=1
    assert diagnostics["ils_cycles_completed"]==2



def test_seed_chunk_checkpoint_resume_and_best_ever(monkeypatch,tmp_path):
    """An interrupted 16-seed study can resume without losing its records."""
    import maxkcut.experiment025_bestcut as study
    g,edges=graph()
    y=np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]],dtype=float)
    monkeypatch.setattr(study,"select_factor",lambda graph: tmp_path/"frozen_factor")
    monkeypatch.setattr(study,"_factor_and_case",
                        lambda path,folder:(g,y,None,"graphhash","factorhash"))
    monkeypatch.setattr(study,"load_references",
                        lambda:({"G11":{"reference_cut":11.,
                                         "published_upper_bound":None}},{}))
    monkeypatch.setitem(study.SETTINGS,"pilot",
        dict(starts=16,kmeans_iters=2,direction_trials=1,
             line_directions=1,em_stages=1,em_iters=1,
             cem_generations=1,cem_population=4,gibbs_sweeps=1,
             kl_passes=1,ils_cycles=1,ils_kl_passes=1))
    root=tmp_path/"results"
    a=study.run_method("G11","gw-rr","pilot",root=root,max_new_seeds=4)
    assert a["starts_completed"]==4
    assert a["status"]=="partially_refined"
    b=study.run_method("G11","gw-rr","pilot",root=root,max_new_seeds=4)
    assert b["starts_completed"]==8
    c=study.run_method("G11","gw-rr","pilot",root=root,max_new_seeds=4)
    assert c["starts_completed"]==12
    d=study.run_method("G11","gw-rr","pilot",root=root,max_new_seeds=4)
    assert d["starts_completed"]==16
    assert d["status"]=="complete_16_best_seen"
    assert d["best_seen"]>=d["best_last_ils_candidate"]
    assert d["selected_seed_index"] in range(16)
    # Every start preserves its independent stage snapshots and iteration log.
    target=root/"pilot"/"G11"/"gw-rr"
    for seed in range(16):
        case=study.result_for_seed(root/"pilot"/"G11",
                                   "gw-rr",seed,g,edges,d["fingerprint"])
        assert case is not None
        assert case["best_seen_cut"]>=case["last_iteration_cut"]
        assert (target/"progress"/f"{seed:04d}"/"events.jsonl").exists()
    # Repeating after all sixteen starts cannot erase earlier best partitions.
    e=study.run_method("G11","gw-rr","pilot",root=root,max_new_seeds=4)
    assert e["best_seen"]==d["best_seen"]
    assert e["selected_seed_index"]==d["selected_seed_index"]
