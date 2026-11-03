#!/usr/bin/env python
"""Produce a lightweight, explicitly provisional Experiment025 publication preview.

Reads only the existing aggregation written by experiment025_bestcut.summary.
Never changes seed archives, recomputes EVA, stages files, or invokes Git.
Run AFTER the local scheduler has stopped so the input snapshot is consistent.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("results/gset/experiment025-gw16-bestcut-quality")


def generate_preview(root: Path = ROOT, phase: str = "full") -> str:
    analysis = Path(root) / "analysis"
    coverage = json.loads((analysis / f"{phase}_coverage.json").read_text(encoding="utf-8"))
    with (analysis / f"{phase}_best_final_by_graph.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        rows = list(csv.DictReader(f))
    expected_cells = int(coverage["planned_cells"])
    if len(rows) != expected_cells:
        raise ValueError(f"Expected {expected_cells} graph-method rows, found {len(rows)}")

    def complete(row: dict) -> bool:
        return (row["status"] == "complete_16_best_seen"
                and int(row["refined_starts"]) == 16
                and row["best_seen_cut"] != "")

    completed = sum(complete(r) for r in rows)
    if completed != int(coverage["completed_cells"]):
        raise ValueError("CSV and coverage JSON disagree; regenerate snapshot after the run stops")
    completed_seeds = sum(int(r["refined_starts"]) for r in rows)
    started = len({r["graph"] for r in rows if int(r["refined_starts"]) > 0})
    methods = list(dict.fromkeys(r["method"] for r in rows))
    by_method = defaultdict(list)
    by_graph = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
        by_graph[row["graph"]].append(row)
    fully_completed_graphs = [
        g for g, members in by_graph.items()
        if len(members) == len(methods) and all(complete(r) for r in members)
    ]
    all_complete = bool(coverage["all_complete"])
    if all_complete != (completed == expected_cells):
        raise ValueError("Completeness flag inconsistent with aggregate data")

    lines = [
        f"# Experiment025 — {phase}-quality results preview",
        "",
        "**Status:** " + ("Complete coverage (independent audit still required)"
                         if all_complete else "PROVISIONAL — incomplete benchmark; not a final ranking"),
        "",
        "## Protocol",
        "",
        "- Original signed Max-Cut objective on 71 canonical G-set graphs",
        "- 18 method protocols; 16 matched Goemans–Williamson starts per method and graph",
        "- Frozen EVA factors; no SDP/EVA recomputation in this preview",
        "- Best verified feasible cut encountered across all meaningful iterations",
        "",
        "## Saved progress",
        "",
        f"- Fully completed graph-method cells: **{completed:,}/{expected_cells:,}** "
        f"({100*completed/expected_cells:.2f}%)",
        f"- Finished seed checkpoints: **{completed_seeds:,}/"
        f"{int(coverage['planned_independent_full_refinements']):,}**",
        f"- Graphs with at least one finished seed: **{started:,}/71**",
        f"- Graphs with all 18 methods complete: **{len(fully_completed_graphs):,}/71**",
        "",
        "## Method coverage",
        "",
        "| Method | Complete graph-method cells | Saved seeds |",
        "|:--|--:|--:|",
    ]
    for method in methods:
        items = by_method[method]
        lines.append(f"| \`{method}\` | {sum(complete(r) for r in items)}"
                     f" | {sum(int(r['refined_starts']) for r in items)} |")

    lines += ["", "## Example graph-level comparisons", ""]
    if not fully_completed_graphs:
        lines.append("No graph has complete 18-method coverage yet. Comparisons are "
                     "withheld to avoid ranking partially evaluated methods.")
    else:
        lines += [
            "Only graphs with all 18 protocols complete are included. The reference "
            "is GW-RR after the identical refinement protocol, not raw unrefined GW.",
            "",
            "| Graph | Refined GW-RR cut | Best observed method | Best cut | Difference |",
            "|:--|--:|:--|--:|--:|",
        ]
        for graph in fully_completed_graphs[:10]:
            members = by_graph[graph]
            gw = next(r for r in members if r["method"] == "gw-rr")
            best = max(members, key=lambda r: float(r["best_seen_cut"]))
            delta = float(best["best_seen_cut"]) - float(gw["best_seen_cut"])
            lines.append(f"| {graph} | {float(gw['best_seen_cut']):g} | "
                         f"\`{best['method']}\` | {float(best['best_seen_cut']):g} "
                         f"| {delta:+g} |")

    lines += [
        "",
        "## Reproducibility and interpretation",
        "",
        "This page is a compact *index*, not the complete numerical archive. The VM "
        "retains JSONL iteration histories, SHA-verified improved partitions, "
        "individual seed results, logs and checkpoints. Maintain a verified "
        "off-VM backup; do not treat the VM as the sole durable archive.",
        "",
        "Incomplete graph-method cells are never treated as zero and cannot enter "
        "full-budget rankings. G-set published best-known cuts are not necessarily "
        "certified optima. EVA low-rank factors are not certified SDP optima. "
        "Methods have unequal computational budgets. Any final method comparison "
        "requires paired-seed analysis and an independent original-graph rescore.",
        "",
        "Source: \`analysis/" + phase + "_best_final_by_graph.csv\` "
        "and \`analysis/" + phase + "_coverage.json\`.",
        "",
    ]
    output = analysis / f"{phase}_preview.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return str(output)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--phase", choices=("pilot", "full"), default="full")
    args = p.parse_args()
    print(generate_preview(args.root, args.phase))


if __name__ == "__main__":
    main()
