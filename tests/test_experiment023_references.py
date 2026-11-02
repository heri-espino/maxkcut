"""Reference-table integrity tests; no graph solving or saved factors."""
import csv
from maxkcut import experiment023_references as refs


def test_71_graph_archived_references_with_provenance(tmp_path):
    p=tmp_path/"experiment021.csv"
    h=tmp_path/"history.csv"
    with p.open("w",newline="",encoding="utf8") as f:
        w=csv.writer(f)
        w.writerow(["graph","initializer","restarts","best_final"])
        for graph in refs.GRAPHS:
            for i in range(9):
                w.writerow([graph,f"init{i}",50,500+i])
    with h.open("w",newline="",encoding="utf8") as f:
        w=csv.writer(f)
        w.writerow(["graph","experiment","method","best_cut"])
        w.writerow(["G63","013","SavedKMeansFlip+KL+ILS",26917])
    values,meta=refs.load_references(str(p),str(h))
    assert len(values)==71
    assert meta["toroidal_paper_count"]==17
    assert values["G1"]["reference_cut"]==508
    assert values["G63"]["reference_cut"]==26917
    assert values["G11"]["reference_cut"]==564
    assert values["G11"]["reference_kind"]=="conditional_proven_optimum"
    assert values["G12"]["published_upper_bound"]==558
    assert refs.compare_cut(550,values["G11"])["reference_gap"]==14
    assert refs.compare_cut(565,values["G11"])["reference_assessment"].startswith("ABOVE_PUBLISHED_UPPER")
    assert refs.compare_cut(None,values["G11"])["reference_assessment"]=="not_measured"


def test_missing_or_incomplete_reference_fails_closed(tmp_path):
    p=tmp_path/"short.csv"
    p.write_text("graph,initializer,restarts,best_final\nG1,paired,50,12\n")
    h=tmp_path/"history.csv"
    h.write_text("graph,experiment,method,best_cut\nG63,013,SavedKMeansFlip+KL+ILS,26917\n")
    import pytest
    with pytest.raises(ValueError,match="71 graph"):
        refs.load_references(str(p),str(h))
    with pytest.raises(FileNotFoundError):
        refs.load_references(str(tmp_path/"missing.csv"),str(h))
