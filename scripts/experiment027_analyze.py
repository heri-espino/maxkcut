"""Read-only independent audit of Experiment027 against frozen Experiment025.

Usage (from Experiment027 worktree with its own src/ on Python path):
 python scripts/experiment027_analyze.py --graphs G1 G11 G14
 python scripts/experiment027_analyze.py --graphs ALL

Only reads original signed graphs, saved EVA factors, and two results archives.
Writes compact CSV/JSON *analysis* files in the new Experiment027 results tree.
Never calls optimization, generates new GW normals, or changes raw archives.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import numpy as np

from maxkcut.eva_rounding_cli import _factor_and_case
from maxkcut.experiment023_methods import sha, verify_case
from maxkcut.io import edge_arrays, load_graph
from maxkcut.objective import cut_weight

from experiment027_from_archives import ALL_GRAPHS, select_pinned_factor

NEW_METHODS = (
    "angular-margin-greedy",
    "angular-gain-greedy",
    "angular-pair-greedy",
)
OLD_METHODS = ("gw-rr", "hyperplane-sweep")
TOL = 1e-6


def _read_json(path):
    if not path.is_file():
        raise FileNotFoundError("Missing required archive: " + str(path))
    return json.loads(path.read_text(encoding="utf8"))


def _equal(actual, expected, context):
    if not np.isclose(float(actual), float(expected), atol=TOL, rtol=0):
        raise ValueError(f"{context}: recorded {expected}, recomputed {actual}")


def _labels(value, n, context):
    x = np.asarray(value)
    if x.shape != (n,) or not np.isin(x, (0, 1)).all():
        raise ValueError(f"{context}: nonbinary or wrong-shape partition")
    return x.astype(np.int8)


def _score(edges, labels):
    return float(cut_weight(edges, labels))


def audit_seed(folder, method, index, graph, edges, expected):
    """Recheck stage NPZ checksum, signed objectives, and journal strict best."""
    where = folder / method
    label = f"{folder.name}/{method}/{index:04d}"
    meta = _read_json(where / f"{index:04d}.json")
    if meta.get("index") != index or meta.get("fingerprint") != expected["fingerprint"]:
        raise ValueError(f"{label}: wrong seed index or experiment fingerprint")
    for key in ("graph_sha256", "factor_sha256"):
        if meta.get(key) != expected[key]:
            raise ValueError(f"{label}: {key} mismatch")
    if meta.get(expected["direction_key"]) != expected["direction_sha"]:
        raise ValueError(f"{label}: 16-start GW direction provenance mismatch")
    part = where / f"{index:04d}.npz"
    if not part.is_file() or sha(part) != meta.get("labels_sha256"):
        raise ValueError(f"{label}: missing or corrupted stage NPZ")
    n = graph.shape[0]
    with np.load(part, allow_pickle=False) as data:
        if set(data.files) != {"Round", "Flip", "KL", "Final"}:
            raise ValueError(f"{label}: four required stage partitions missing")
        for stage in ("Round", "Flip", "KL", "Final"):
            arr = _labels(data[stage], n, f"{label}/{stage}")
            _equal(_score(edges, arr), meta["stages"][stage], f"{label}/{stage}")
    if meta["stages"]["Final"] > meta["best_seen_cut"] + TOL:
        raise ValueError(f"{label}: best-ever score below final stage")
    progress = where / "progress" / f"{index:04d}"
    best = _read_json(progress / "best.json")
    partition = progress / best["partition"]
    if not partition.is_file() or sha(partition) != best.get("sha256"):
        raise ValueError(f"{label}: missing or corrupted strict-best snapshot")
    with np.load(partition, allow_pickle=False) as data:
        best_labels = _labels(data["labels"], n, f"{label}/strict-best")
    independent_best = _score(edges, best_labels)
    _equal(independent_best, best["best_cut"], f"{label}/journal")
    _equal(independent_best, meta["best_seen_cut"], f"{label}/best-ever")
    if best.get("fingerprint") != meta["fingerprint"]:
        raise ValueError(f"{label}: mismatched journal fingerprint")
    events = progress / "events.jsonl"
    if not events.is_file():
        raise FileNotFoundError(f"{label}: missing per-iteration event journal")
    count = 0
    with events.open(encoding="utf8") as stream:
        for count, line in enumerate(stream, 1):
            obj = json.loads(line)
            if obj.get("event_number") != count or obj.get("fingerprint") != meta["fingerprint"]:
                raise ValueError(f"{label}: nonsequential/foreign journal event {count}")
    if count != meta.get("progress_events"):
        raise ValueError(f"{label}: event journal count mismatch")
    return {
        "graph": folder.name, "method": method, "seed": index,
        "raw": float(meta["stages"]["Round"]),
        "flip": float(meta["stages"]["Flip"]),
        "kl": float(meta["stages"]["KL"]),
        "final_stage": float(meta["stages"]["Final"]),
        "best": independent_best,
        "seconds": float(meta.get("seed_seconds", float("nan"))),
        "events": count,
        "paths": int(meta.get("angular_summary", {}).get("paths_evaluated", 0)),
    }


def audit_normals(folder, method, index, y, edges, meta):
    """Verify every accepted high-dimensional normal reproduces its saved cut."""
    path = folder / method / "progress" / f"{index:04d}" / "accepted_normals.npz"
    if not path.is_file() or sha(path) != meta.get("accepted_normals_sha256"):
        raise ValueError(f"{folder.name}/{method}/{index}: accepted normals corrupted")
    with np.load(path, allow_pickle=False) as z:
        normals, scores, indices = z["normals"], z["cuts"], z["iterations"]
    if normals.ndim != 2 or normals.shape[1] != y.shape[1]:
        raise ValueError(f"{folder.name}/{method}/{index}: invalid normal dimensions")
    if len(normals) != len(scores) or len(scores) != len(indices) or len(scores) < 1:
        raise ValueError(f"{folder.name}/{method}/{index}: invalid trajectory lengths")
    if not np.all(np.isfinite(normals)) or not np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-8):
        raise ValueError(f"{folder.name}/{method}/{index}: nonunit or nonfinite normals")
    if int(indices[0]) != 0 or not np.all(np.diff(indices) > 0):
        raise ValueError(f"{folder.name}/{method}/{index}: invalid accepted-step indices")
    if not np.all(np.diff(scores) > -TOL):
        raise ValueError(f"{folder.name}/{method}/{index}: nonmonotone accepted cuts")
    for r, score in zip(normals, scores):
        labels = (y @ r > 0).astype(np.int8)
        _equal(_score(edges, labels), score, f"{folder.name}/{method}/{index}/normal")
    _equal(scores[-1], meta["raw_cut"], f"{folder.name}/{method}/{index}/round")
    return len(scores)


def audit_graph(graph_id, original, oldroot, newroot):
    graph_file = original / "data" / f"{graph_id}.txt"
    source, pinned_sha = select_pinned_factor(graph_id, original, oldroot, original / "data")
    graph, y, case, graph_sha, factor_sha = _factor_and_case(graph_file, source)
    if factor_sha != pinned_sha:
        raise ValueError(f"{graph_id}: factor changed after source selection")
    edges = edge_arrays(graph)
    old_dir = oldroot / "full" / graph_id
    new_dir = newroot / "full" / graph_id
    old_status = {m: _read_json(old_dir / m / "status.json") for m in OLD_METHODS}
    new_status = {m: _read_json(new_dir / m / "status.json") for m in NEW_METHODS}
    for m, status in old_status.items():
        if (status.get("status") != "complete_16_best_seen" or status.get("starts_completed") != 16
                or status.get("graph_sha256") != graph_sha or status.get("factor_sha256") != factor_sha):
            raise ValueError(f"{graph_id}/{m}: historical Experiment025 baseline incomplete/incompatible")
    for m, status in new_status.items():
        if (status.get("status") != "complete_16" or status.get("starts_completed") != 16
                or status.get("graph_sha256") != graph_sha or status.get("factor_sha256") != factor_sha):
            raise ValueError(f"{graph_id}/{m}: Experiment027 new method incomplete/incompatible")
    with np.load(old_dir / "gw16_common.npz", allow_pickle=False) as a, np.load(
        new_dir / "gw16_common.npz", allow_pickle=False
    ) as b:
        old_normals, new_normals = a["normals"], b["normals"]
        if old_normals.shape != (16, y.shape[1]) or not np.array_equal(old_normals, new_normals):
            raise ValueError(f"{graph_id}: GW normals not exactly paired")
        for index in range(16):
            if not np.array_equal(a["gw_partitions"][index], b["gw_partitions"][index]):
                raise ValueError(f"{graph_id}: GW starts have different partitions")
    direction_sha = hashlib.sha256(old_normals.tobytes()).hexdigest()
    rows = []
    for m, status in list(old_status.items()) + list(new_status.items()):
        key = "shared_directions_sha256" if m in OLD_METHODS else "shared_directions_sha256"
        if status.get(key) != direction_sha:
            raise ValueError(f"{graph_id}/{m}: incompatible GW direction digest")
        expected = {
            "fingerprint": status["fingerprint"],
            "graph_sha256": graph_sha,
            "factor_sha256": factor_sha,
            "direction_sha": direction_sha,
            "direction_key": "shared_direction_sha256",
        }
        for i in range(16):
            seed = audit_seed(old_dir if m in OLD_METHODS else new_dir, m, i, graph, edges, expected)
            if m in NEW_METHODS:
                metadata = _read_json(new_dir / m / f"{i:04d}.json")
                audit_normals(new_dir, m, i, y, edges, metadata)
            rows.append(seed)
        items = [z for z in rows if z["method"] == m]
        _equal(max(z["best"] for z in items), status["best_final"], f"{graph_id}/{m}/best16")
        _equal(np.mean([z["best"] for z in items]), status["mean_final"], f"{graph_id}/{m}/mean16")
    return rows


def summarize(rows, graphs, bootstraps=2000):
    """Graph-clustered comparison; seed pairs are nested, not independent graphs."""
    by = {(r["graph"], r["method"], r["seed"]): r for r in rows}
    graph_rows = []
    for g in graphs:
        best = {m: max(by[(g, m, s)]["best"] for s in range(16))
                for m in OLD_METHODS + NEW_METHODS}
        raw = {m: max(by[(g, m, s)]["raw"] for s in range(16))
               for m in OLD_METHODS + NEW_METHODS}
        for m in NEW_METHODS:
            delta_gw = best[m] - best["gw-rr"]
            delta_hp = best[m] - best["hyperplane-sweep"]
            graph_rows.append(dict(
                graph=g, method=m, best=best[m], raw=raw[m],
                gw_best=best["gw-rr"], hyperplane_best=best["hyperplane-sweep"],
                delta_gw=delta_gw, delta_hyperplane=delta_hp,
                relative_gw_pct=100 * delta_gw / abs(best["gw-rr"])
                if best["gw-rr"] != 0 else float("nan"),
                relative_hyperplane_pct=100 * delta_hp / abs(best["hyperplane-sweep"])
                if best["hyperplane-sweep"] != 0 else float("nan"),
            ))
    leaders = []
    for method in NEW_METHODS:
        group = [r for r in graph_rows if r["method"] == method]
        d = np.array([r["relative_gw_pct"] for r in group], dtype=float)
        dg = np.array([r["delta_gw"] for r in group], dtype=float)
        dh = np.array([r["delta_hyperplane"] for r in group], dtype=float)
        if np.any(~np.isfinite(d)):
            mean, ci = None, [None, None]
        else:
            mean = float(d.mean())
            rng = np.random.default_rng(np.random.SeedSequence([2026, 27, len(graphs)]))
            indices = rng.integers(0, len(graphs), size=(bootstraps, len(graphs)))
            sample = d[indices].mean(axis=1)
            ci = np.percentile(sample, [2.5, 97.5]).tolist()
        leaders.append(dict(
            method=method, graphs=len(graphs),
            mean_relative_gw_pct=mean, bootstrap95_low=ci[0],
            bootstrap95_high=ci[1],
            wins_vs_gw=int(np.sum(dg > TOL)),
            ties_vs_gw=int(np.sum(np.abs(dg) <= TOL)),
            losses_vs_gw=int(np.sum(dg < -TOL)),
            wins_vs_hyperplane=int(np.sum(dh > TOL)),
            ties_vs_hyperplane=int(np.sum(np.abs(dh) <= TOL)),
            losses_vs_hyperplane=int(np.sum(dh < -TOL)),
            mean_cut_difference_vs_hyperplane=float(dh.mean()),
            total_new_sweep_paths=sum(r["paths"] for r in rows if r["method"] == method),
        ))
    pairs = []
    for g in graphs:
        for m in NEW_METHODS:
            for s in range(16):
                new = by[g, m, s]
                for reference in OLD_METHODS:
                    old = by[g, reference, s]
                    pairs.append(dict(graph=g, method=m, baseline=reference, seed=s,
                                      delta_raw=new["raw"] - old["raw"],
                                      delta_best=new["best"] - old["best"],
                                      new_raw=new["raw"], old_raw=old["raw"],
                                      new_best=new["best"], old_best=old["best"],
                                      new_seconds=new["seconds"], old_seconds=old["seconds"],
                                      new_sweep_paths=new["paths"]))
    return graph_rows, leaders, pairs


def _write_csv(path, records):
    if not records:
        return
    with path.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=Path.home() / "maxkcut")
    parser.add_argument("--new-root", type=Path, default=PROJECT / "results/gset/experiment027-angular-greedy")
    parser.add_argument("--graphs", nargs="+", default=["G1", "G11", "G14"])
    parser.add_argument("--bootstraps", type=int, default=2000)
    args = parser.parse_args(argv)
    graph_ids = list(ALL_GRAPHS if args.graphs == ["ALL"] else args.graphs)
    if not graph_ids or len(graph_ids) != len(set(graph_ids)) or any(g not in ALL_GRAPHS for g in graph_ids):
        parser.error("--graphs must be distinct G-set IDs or ALL")
    if args.bootstraps < 100:
        parser.error("--bootstraps must be >= 100")
    original = args.original.resolve()
    new_root = args.new_root.resolve()
    old_root = original / "results/gset/experiment025-gw16-bestcut-quality"
    all_rows = []
    for graph_id in graph_ids:
        rows = audit_graph(graph_id, original, old_root, new_root)
        all_rows.extend(rows)
        print(json.dumps(dict(graph=graph_id, audited_seed_archives=len(rows),
                              historical=32, new=48)), flush=True)
    graph_rows, leaders, pairs = summarize(all_rows, graph_ids, bootstraps=args.bootstraps)
    out = new_root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "audited_seed_scores.csv", all_rows)
    _write_csv(out / "audited_graph_comparisons.csv", graph_rows)
    _write_csv(out / "audited_method_leaderboard.csv", leaders)
    _write_csv(out / "audited_paired_seed_deltas.csv", pairs)
    report = dict(
        status="independently_verified_archives", graphs=len(graph_ids),
        original_methods=list(OLD_METHODS), new_methods=list(NEW_METHODS),
        audited_method_seed_archives=len(all_rows),
        completed_new_seeds=len(graph_ids) * 3 * 16,
        completed_historical_comparator_seeds=len(graph_ids) * 2 * 16,
        bootstrap_replicates=args.bootstraps,
        note="Paired graph-cluster comparison; not independent 16-seed pseudoreplication; no new optimization",
        leaderboard=leaders,
        analysis_directory=str(out),
    )
    (out / "audited_comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
