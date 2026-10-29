"""Experiment021: complete the canonical 71-graph G-set without rerunning 68 graphs.

Experiment020 remains immutable. Only G60, G61 and G62 need new
hash-verified, Numba EVA factors and 9 x 50 paired cut searches.
The consolidated statistical outputs are stored under Experiment021.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .eva_cli import _write_json
from .eva_rounding_cli import _factor_and_case
from .eva_kmeans_init_all_cli import (
    _read_long_rows, aggregate as aggregate_exp020,
)

BASE = Path("results/gset")
OLD = BASE / "experiment020-all-gset-paired-inits"
NEW = BASE / "experiment021-complete-71-gset"
NEW_FACTORS = NEW / "eva-numba"
MISSING = ("G60", "G61", "G62")
EXPECTED_GRAPHS = 71
METHODS = 9
RESTARTS = 50


def make_inventory(*, old=OLD, new=NEW, factor_dir=NEW_FACTORS,
                   data_dir=Path("data"), restarts=RESTARTS,
                   factor_reader=_factor_and_case,
                   rows_reader=_read_long_rows):
    """Create a cross-experiment manifest, refusing altered historical inputs.

    Completed Experiment020 records point to their ORIGINAL output paths.
    New cut records point only under Experiment021. The original inventory,
    per-run archives, source factors and scores are never rewritten here.
    """
    old, new, factor_dir, data_dir = map(
        Path, (old, new, factor_dir, data_dir)
    )
    original = json.loads(
        (old / "inventory.json").read_text(encoding="utf8")
    )
    records = []
    missing = []
    seen = set()
    for prior in original["records"]:
        graph = prior["graph"]
        if graph in seen:
            raise RuntimeError(f"Duplicate graph in Experiment020: {graph}")
        seen.add(graph)
        path = data_dir / f"{graph}.txt"
        if not path.is_file():
            raise FileNotFoundError(path)
        r = dict(prior)
        if graph in MISSING:
            if prior["status"] != "no_completed_EVA_factor":
                raise RuntimeError(
                    f"Experiment020 already had an EVA factor for {graph}; "
                    "do not silently alter the frozen experiment"
                )
            r["output_dir"] = str(new / graph)
            case = factor_dir / "cases" / f"{graph}.json"
            factor = factor_dir / "factors" / f"{graph}.npz"
            if case.is_file() != factor.is_file():
                raise RuntimeError(
                    f"{graph} has a partially published EVA factor/case"
                )
            if not case.is_file():
                r["status"] = "no_completed_EVA_factor"
                missing.append(graph)
            else:
                adjacency, vectors, meta, graph_sha, factor_sha = (
                    factor_reader(path, factor_dir)
                )
                if graph_sha != hashlib.sha256(path.read_bytes()).hexdigest():
                    raise RuntimeError(f"Input hash mismatch for {graph}")
                r.update({
                    "status": "ready",
                    "vertices": int(adjacency.shape[0]),
                    "rank": int(vectors.shape[1]),
                    "graph_sha256": graph_sha,
                    "factor_sha256": factor_sha,
                    "factor_source": str(factor_dir),
                })
        else:
            if prior["status"] not in ("ready", "reused_019"):
                raise RuntimeError(
                    f"Previously incomplete graph unexpectedly not in "
                    f"missing set: {graph}"
                )
            _, _, _, graph_sha, factor_sha = factor_reader(
                path, Path(prior["factor_source"])
            )
            if (graph_sha != prior["graph_sha256"]
                    or factor_sha != prior["factor_sha256"]):
                raise RuntimeError(
                    f"Experiment020 factor or graph changed: {graph}"
                )
            rows = rows_reader(prior, restarts=restarts)
            if rows is None or len(rows) != METHODS * restarts:
                raise RuntimeError(
                    f"Experiment020 {graph} lacks {METHODS * restarts} "
                    "verified, matched trial rows"
                )
        records.append(r)

    if len(records) != EXPECTED_GRAPHS or set(MISSING) - seen:
        raise RuntimeError(
            f"Expected all {EXPECTED_GRAPHS} canonical graphs, "
            f"found {len(records)}; absent: {sorted(set(MISSING)-seen)}"
        )
    manifest = {
        "version": "experiment021-inventory-v1",
        "graph_count": len(records),
        "original_experiment020_inventory": str(old / "inventory.json"),
        "new_factor_dir": str(factor_dir),
        "new_graphs": list(MISSING),
        "missing_factors": missing,
        "verified_ready": sum(r["status"] == "ready" for r in records),
        "reused_verified_candidate": sum(
            r["status"] == "reused_019" for r in records
        ),
        "unavailable_count": len(missing),
        "records": records,
    }
    _write_json(new / "inventory.json", manifest)
    return manifest


def require_full_coverage(report, *, graphs=EXPECTED_GRAPHS,
                          methods=METHODS, restarts=RESTARTS):
    """Reject partial coverage even if the legacy aggregator tolerated missing EVA."""
    expected = graphs * methods * restarts
    required = {
        "graph_count": graphs,
        "complete_graphs": graphs,
        "no_EVA_factor_count": 0,
        "incomplete_candidate_count": 0,
        "runs_in_all_runs_csv": expected,
    }
    for key, value in required.items():
        if report.get(key) != value:
            raise RuntimeError(
                f"Experiment021 not complete: {key}="
                f"{report.get(key)!r}, expected {value!r}"
            )
    return report


def consolidate(*, old=OLD, new=NEW, factor_dir=NEW_FACTORS,
                data_dir=Path("data"), restarts=RESTARTS):
    manifest = make_inventory(
        old=old, new=new, factor_dir=factor_dir,
        data_dir=data_dir, restarts=restarts,
    )
    report = aggregate_exp020(
        out=Path(new), restarts=restarts, require_complete=True,
    )
    require_full_coverage(report, restarts=restarts)
    report["experiment021_status"] = "all_71_complete"
    report["provenance"] = {
        "old_experiment": str(old),
        "new_factor_source": str(factor_dir),
        "new_graphs": list(MISSING),
        "existing_runs_reused": (EXPECTED_GRAPHS - len(MISSING)) * METHODS * restarts,
        "new_runs": len(MISSING) * METHODS * restarts,
        "all_71_graphs_have_verified_factors": True,
        "new_eva_solved_only_for": list(MISSING),
        "warning": "Nonconvex EVA factors are not SDP certificates",
    }
    _write_json(Path(new) / "statistics" / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("inventory", "aggregate"))
    p.add_argument("--old", type=Path, default=OLD)
    p.add_argument("--new", type=Path, default=NEW)
    p.add_argument("--factor-dir", type=Path, default=NEW_FACTORS)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--restarts", type=int, default=RESTARTS)
    args = p.parse_args()
    if args.restarts != RESTARTS:
        p.error("Experiment021 must use the frozen 50-restarter protocol")
    if args.phase == "inventory":
        manifest = make_inventory(
            old=args.old, new=args.new, factor_dir=args.factor_dir,
            data_dir=args.data_dir, restarts=args.restarts,
        )
        print(
            f"Experiment021: {manifest['verified_ready'] + manifest['reused_verified_candidate']}"
            f"/{manifest['graph_count']} graphs have validated factors; "
            f"remaining: {manifest['missing_factors']}", flush=True,
        )
    else:
        print(json.dumps(consolidate(
            old=args.old, new=args.new, factor_dir=args.factor_dir,
            data_dir=args.data_dir, restarts=args.restarts,
        ), indent=2), flush=True)


if __name__ == "__main__":
    main()
