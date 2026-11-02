"""Pure synthetic tests: no large benchmark/SDP/EVA execution."""
import csv
from pathlib import Path
from maxkcut import graph_structure_methods as mod


def test_complete_unique_family_mapping():
    assert len(mod.validate_classification()) == 71
    assert tuple(map(len, mod.FAMILIES.values())) == (30, 24, 17)
    assert mod.labels('G63') == ('planar', 'positive')
    assert mod.labels('G11') == ('toroidal', 'signed')
    # The external label table mistakenly lists G48-G50 as signed;
    # the canonical files and 2020 paper establish positive-only edges.
    assert mod.labels('G48') == ('toroidal', 'positive')
    assert mod.labels('G49') == ('toroidal', 'positive')
    assert mod.labels('G50') == ('toroidal', 'positive')


def test_exact_triangle_and_graph_metadata(tmp_path):
    f=tmp_path/'G1.txt'
    f.write_text('5 4\n1 2 1\n2 3 -1\n1 3 1\n4 5 -1\n')
    result=mod.graph_features(f)
    assert result['n']==5 and result['m']==4
    assert result['triangles']==1
    assert result['components']==2
    assert result['cycle_rank']==1
    assert result['transitivity']==1
    assert result['negative_edge_fraction']==0.5


def test_fisher_exact_known_table():
    assert abs(mod.fisher_two_sided(11,6,13,41)-0.0032989118)<1e-8
    assert abs(mod.fisher_two_sided(8,6,2,20)-0.0054502582)<1e-8


def test_winners_fractional_ties_and_loo():
    # All 71 graph IDs supplied synthetically; test simple method switches.
    scores={}
    for i in mod.validate_classification():
        g=f'G{i}'
        scores[g]={m:{'best':100.,'mean':100.,'seconds':1.,'extra_eval':16 if m=='gw-screen16' else 0} for m in mod.METHODS}
        if i in mod.FAMILIES['random']:
            scores[g]['sample-pair']['best']=101.
        else:
            scores[g]['gw-screen16']['best']=102.
    gr, mr=mod.winner_tables(scores)
    assert len(gr)==71 and len(mr)==639
    assert sum(r['fractional_win'] for r in mr)==71
    cv=mod.loo_selection(scores)
    assert len(cv)==142
    for r in cv:
        assert r['selected_method'] in mod.METHODS
        assert r['regret']>=0
    assert all(r['selected_method']=='sample-pair' for r in cv if r['policy']=='family' and r['family']=='random')


def test_load_csv_rejects_missing_methods(tmp_path):
    f=tmp_path/'results.csv'
    f.write_text('graph,initializer,restarts,best_final,mean_final,mean_seconds,additional_initial_cut_evaluations\nG1,paired,50,10,9,1,0\n')
    try:
        mod.load_results(f)
    except ValueError as exc:
        assert '71 graph' in str(exc)
    else:
        raise AssertionError('Must reject partial experimental data')