"""Exact 54-pilot / six 213-case job batches for PCAG103A-13 Actions."""
import pytest

from scripts.experiment025_actions_matrix import (
    GRAPHS, METHODS, METHOD_BATCHES, PILOT_GRAPHS, matrix
)


def test_pilot_produces_exactly_54_independent_graph_method_jobs():
    rows=matrix("pilot")["include"]
    assert len(rows)==54
    assert {(r["graph"],r["method"]) for r in rows} == {
        (g,m) for g in PILOT_GRAPHS for m in METHODS
    }
    assert all(r["phase"]=="pilot" for r in rows)


def test_six_full_batches_cover_every_canonical_pair_once():
    cases=[]
    assert len(GRAPHS)==71 and len(METHODS)==18
    for batch in range(6):
        jobs=matrix("full-batch",batch=batch)["include"]
        assert len(jobs)==213  # under GitHub 256 matrix limit
        assert all(r["phase"]=="full" for r in jobs)
        assert {r["method"] for r in jobs}==set(METHOD_BATCHES[batch])
        cases.extend((r["graph"],r["method"]) for r in jobs)
    assert len(cases)==1278
    assert len(set(cases))==1278
    assert set(cases)=={(g,m) for g in GRAPHS for m in METHODS}


def test_single_and_coverage_are_one_job():
    assert matrix("single",graph="G11",method="gw-rr",phase="full") == {
        "include":[{"graph":"G11","method":"gw-rr","phase":"full"}]
    }
    assert matrix("coverage")["include"] == [
        {"graph":"G11","method":"gw-rr","phase":"coverage"}
    ]


@pytest.mark.parametrize("bad",[-1,6,80])
def test_invalid_batch_fails(bad):
    with pytest.raises(ValueError):
        matrix("full-batch",batch=bad)


def test_invalid_graph_or_method_fails():
    with pytest.raises(ValueError):
        matrix("single",graph="G999")
    with pytest.raises(ValueError):
        matrix("single",method="unregistered")
    with pytest.raises(ValueError):
        matrix("not-a-mode")


def test_workflow_is_manual_only_and_explicitly_targets_named_runner():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    workflow=(root/".github/workflows/experiment025-quality-first.yml").read_text(encoding="utf8")
    assert "workflow_dispatch:" in workflow
    assert "\n  push:" not in workflow
    assert "runs-on: [self-hosted, Windows, X64]" in workflow
    assert "runs-on: [self-hosted, Windows, X64, PCAG103A-13]" not in workflow
    assert "RUNNER_NAME -ne 'PCAG103A-13'" in workflow
    assert "max-parallel: 1" in workflow
    assert "ExpectedSourceSha256" in workflow
    assert "OnlyGraph" in workflow
