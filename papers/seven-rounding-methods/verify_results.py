#!/usr/bin/env python3
"""Provenance gate for the 7-method manuscript; never runs an optimizer.

Check that all quoted ranking data match the original existing E025 aggregate
and E027 independently audited release, as well as our exact manuscript
selection table. Standard-library Python only. Exit nonzero on any drift.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parent
E025 = ROOT / "results/gset/experiment025-gw16-bestcut-quality/analysis"
E027 = ROOT / "published/experiment027/releases/experiment027-data-71g-2b313a624780291eb4dd/analysis"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def close(got: object, expected: object, label: str, tolerance: float = 1e-6) -> None:
    if not math.isclose(float(got), float(expected), rel_tol=0, abs_tol=tolerance):
        raise AssertionError(f"{label}: got {got!r} not {expected!r}")


def main() -> None:
    local = rows(PAPER / "data/top_seven.csv")
    rank025 = {x["method"]: x for x in rows(E025 / "experiment025_method_ranking.csv")}
    rank027 = {x["method"]: x for x in rows(E027 / "audited_method_leaderboard.csv")}
    all025 = rows(E025 / "full_best_final_by_graph.csv")
    audited = json.loads((E027 / "audited_comparison.json").read_text(encoding="utf-8"))
    paired = rows(E027 / "audited_paired_seed_deltas.csv")

    expect = (
        "hyperplane-sweep", "angular-gain-greedy", "angular-margin-greedy",
        "angular-pair-greedy", "control-paired", "em-gibbs", "cutaware-kmeans"
    )
    actual = tuple(x["method"] for x in local)
    assert actual == expect, f"Selection/order drifted: {actual!r}"
    assert len(all025) == 71 * 18 == 1278
    assert len({(x["graph"], x["method"]) for x in all025}) == 1278
    assert {x["status"] for x in all025} == {"complete_16_best_seen"}
    assert {int(x["refined_starts"]) for x in all025} == {16}
    assert audited["status"] == "independently_verified_archives"
    assert audited["graphs"] == 71
    assert audited["completed_new_seeds"] == 71 * 16 * 3 == 3408
    assert audited["completed_historical_comparator_seeds"] == 71 * 16 * 2 == 2272
    assert len(paired) == 71 * 16 * 3 * 2 == 6816

    for idx, item in enumerate(local, 1):
        method = item["method"]
        assert int(item["rank"]) == idx
        provenance = item["source"]
        src = rank025[method] if provenance == "experiment025" else rank027[method]
        mappings = (
            ("mean_gain_pct", "mean_relative_gw_pct"),
            ("bootstrap_low_pct", "bootstrap95_low"),
            ("bootstrap_high_pct", "bootstrap95_high"),
            ("wins_gw", "wins_vs_gw"),
            ("ties_gw", "ties_vs_gw"),
            ("losses_gw", "losses_vs_gw"),
        )
        if provenance == "experiment025":
            mappings = (
                ("mean_gain_pct", "mean_gain_pct"),
                ("bootstrap_low_pct", "bootstrap_95_ci_low_pct"),
                ("bootstrap_high_pct", "bootstrap_95_ci_high_pct"),
                ("wins_gw", "graphs_better_than_gw"),
                ("ties_gw", "graphs_tied_with_gw"),
                ("losses_gw", "graphs_worse_than_gw"),
            )
        assert int(item["wins_gw"]) + int(item["ties_gw"]) + int(item["losses_gw"]) == 71
        assert int(item["wins_hp"]) + int(item["ties_hp"]) + int(item["losses_hp"]) == 71
        for key, from_src in mappings:
            close(item[key], src[from_src], f"{method} / {key}")
        if provenance == "experiment027":
            for key, from_src in (
                ("mean_cut_minus_hp", "mean_cut_difference_vs_hyperplane"),
                ("wins_hp", "wins_vs_hyperplane"),
                ("ties_hp", "ties_vs_hyperplane"),
                ("losses_hp", "losses_vs_hyperplane")
            ):
                close(item[key], src[from_src], f"{method} / {key}")
        elif method == "hyperplane-sweep":
            assert [int(item[k]) for k in ("wins_hp", "ties_hp", "losses_hp")] == [0, 71, 0]
        else:
            historical = rows(E025 / "experiment025_leader_head_to_head.csv")
            source = next(x for x in historical if x["comparison"] == "hyperplane-sweep minus " + method)
            assert int(item["wins_hp"]) == int(source["hyper_losses"])
            assert int(item["ties_hp"]) == int(source["ties"])
            assert int(item["losses_hp"]) == int(source["hyper_wins"])
    gains = [float(x["mean_gain_pct"]) for x in local]
    assert gains == sorted(gains, reverse=True)

    doc = (PAPER / "main.tex").read_text(encoding="utf-8")
    files = (
        "01_problem_and_protocol", "hyperplane_sweep", "angular_gain",
        "angular_margin", "angular_pair", "control_paired", "em_gibbs",
        "cutaware_kmeans", "09_empirical_results", "10_discussion",
        "appendix_derivations", "appendix_reproducibility"
    )
    for name in files:
        path = PAPER / "sections" / (name + ".tex")
        assert path.is_file() and len(path.read_text(encoding="utf-8")) > 1800, name
        assert r"\input{sections/" + name + "}" in doc, name
    for name in ("rounding_pipeline", "ranking", "arc_events", "raw_gap"):
        assert (PAPER / "figures" / (name + ".tex")).is_file(), name
    assert (PAPER / "references.bib").is_file()
    print(json.dumps({
        "status": "source_values_and_manuscript_modules_verified",
        "seven_methods": actual, "graphs": 71, "experiment025_rows": len(all025),
        "experiment027_paired_rows": len(paired), "paper_sections": len(files)
    }, indent=2))


if __name__ == "__main__":
    main()
