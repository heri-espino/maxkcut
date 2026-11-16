"""Experiment028: paired EVA dimension test on 5 small G-set graphs.

No historical factors, plots, archives or source code are modified. Every
execution has its own rank/seed directory under the new experiment root.
The only scientific optimizer invoked is the existing compiled EVA backend.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
from html import escape
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from maxkcut.eva import vector_objective
from maxkcut.eva_numba import solve_eva_numba
from maxkcut.io import edge_arrays, load_graph
from maxkcut.objective import cut_weight
from maxkcut.rounding import round_fj, round_spherical_kmeans

GRAPHS = ("G1", "G6", "G11", "G14", "G18")
SEEDS = (2026, 2027, 2028, 2029, 2030)
METHODS = ("FJ_EVA", "K-Means2N_EVA", "K-MeansNM_EVA")
FIELDS = (
    "graph", "family", "weight_type", "n", "m", "graph_sha256", "rank_mode",
    "p", "seed", "method", "continuous_objective", "cut_weight", "sweeps",
    "converged", "max_angle", "eva_seconds", "jit_seconds",
    "rounding_seconds", "cluster_iterations", "factor_sha256", "partition_sha256",
)


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def graph_byte_hashes(path: Path) -> dict[str, str]:
    """Preserve raw file fingerprint and handle Git Windows/Linux EOL checkout.

    Historical Experiment022 hashes were generated on Windows. The same
    logical G-set edge list can be checked out with LF or CRLF; accepting
    only these three byte-identical-after-EOL-normalization forms does not
    relax the edge/weight-content integrity requirement.
    """
    data = path.read_bytes()
    lf = data.replace(b"\r\n", b"\n")
    return {
        "raw": hashlib.sha256(data).hexdigest(),
        "lf": hashlib.sha256(lf).hexdigest(),
        "crlf": hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest(),
    }



def metadata_from_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["graph"]: row for row in csv.DictReader(stream)}


def seed_sequence(seed: int, name: str, method: str) -> np.random.SeedSequence:
    """Identical independent rounding stream for both ranks, per method."""
    method_index = {"FJ_EVA": 0, "K-Means2N_EVA": 1, "K-MeansNM_EVA": 2}[method]
    return np.random.SeedSequence([seed, int(name[1:]), 2, method_index, 28])


def partition_hash(labels: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(np.asarray(labels, dtype=np.uint8)).tobytes()).hexdigest()


def audit_case(case_path: Path, a, edges) -> None:
    """Independently rescore all archived original signed graph partitions."""
    from maxkcut.eva import load_eva_factor
    obj = json.loads(case_path.read_text(encoding="utf-8"))
    directory = case_path.parent
    factor, partitions = directory / "factor.npz", directory / "partitions.npz"
    if sha256(factor) != obj["factor_sha256"]:
        raise RuntimeError(f"EVA factor SHA mismatch: {factor}")
    if sha256(partitions) != obj["partitions_sha256"]:
        raise RuntimeError(f"Partition archive SHA mismatch: {partitions}")
    y, state = load_eva_factor(factor)
    settings = obj["settings"]
    if y.shape != (a.shape[0],settings["p"]) or int(state["sweeps"]) != int(obj["rows"][0]["sweeps"]):
        raise RuntimeError(f"Invalid factor shape or sweep metadata: {factor}")
    with np.load(partitions, allow_pickle=False) as stored:
        if set(stored.files) != set(METHODS):
            raise RuntimeError(f"Unexpected methods in partition archive: {partitions}")
        for row in obj["rows"]:
            method=row["method"]
            packed=stored[method]
            if hashlib.sha256(packed.tobytes()).hexdigest() != row["partition_sha256"]:
                raise RuntimeError(f"Partition SHA mismatch: {case_path} / {method}")
            labels=np.unpackbits(packed)[:a.shape[0]]
            if not np.isclose(cut_weight(edges,labels),float(row["cut_weight"]),atol=1e-9,rtol=0):
                raise RuntimeError(f"Original signed cut rescoring failed: {case_path} / {method}")



def one_case(graph_name: str, a, edges, info: dict[str, str], rank_mode: str,
             seed: int, output: Path, *, max_sweeps: int, tolerance: float,
             checkpoint_every: int) -> list[dict]:
    n = a.shape[0]
    p = int(np.ceil(np.sqrt(2 * (n + 1)))) if rank_mode == "default" else n
    directory = output / "runs" / graph_name / rank_mode / f"seed-{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    case = directory / "case.json"
    factor = directory / "factor.npz"
    trace = directory / "eva_sweeps.jsonl"
    partitions = directory / "partitions.npz"
    graph_hash = info["graph_sha256"]
    source_digest = hashlib.sha256()
    for source in ("src/maxkcut/eva_numba.py", "src/maxkcut/eva.py",
                   "src/maxkcut/rounding.py", "src/maxkcut/objective.py",
                   "scripts/experiment028_fullrank_pilot.py"):
        source_path = Path(source)
        if source_path.is_file():
            source_digest.update(source_path.read_bytes())
    setting = {
        "implementation_sha256": source_digest.hexdigest(),
        "graph": graph_name, "graph_sha256": graph_hash,
        "raw_checkout_sha256": info.get("raw_checkout_sha256"),
        "matching_line_ending": info.get("matching_line_ending"), "n": n,
        "p": p, "rank_mode": rank_mode, "seed": seed,
        "tolerance": tolerance, "max_sweeps": max_sweeps,
        "checkpoint_every": checkpoint_every, "methods": list(METHODS),
        "optimizer": "maxkcut.eva_numba.solve_eva_numba",
    }
    if case.is_file():
        old = json.loads(case.read_text(encoding="utf-8"))
        if old.get("settings") != setting:
            raise RuntimeError(f"Refusing incompatible cached run {case}")
        if not factor.exists() or not partitions.exists():
            raise RuntimeError(f"A completed record has missing factor/partitions: {case}")
        if old.get("factor_sha256") != sha256(factor):
            raise RuntimeError(f"Factor changed since completed run {case}")
        if old.get("partitions_sha256") != sha256(partitions):
            raise RuntimeError(f"Partitions changed since completed run {case}")
        audit_case(case, a, edges)
        return old["rows"]

    # The EVA implementation writes checkpoint metadata and RNG state every
    # checkpoint interval; no existing Experiment002 or Experiment021 file is touched.
    previous_sweeps = -1
    if factor.is_file():
        from maxkcut.eva import load_eva_factor
        _, state = load_eva_factor(factor)
        previous_sweeps = int(state["sweeps"])
        if trace.is_file():
            retained = [line for line in trace.read_text(encoding="utf-8").splitlines()
                        if line.strip() and int(json.loads(line)["sweep"]) <= previous_sweeps]
            trace.write_text("\n".join(retained) + ("\n" if retained else ""), encoding="utf-8")

    def on_sweep(state: dict) -> None:
        with trace.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "sweep": int(state["sweeps"]),
                "objective": float(state["objective"]),
                "max_angle": float(state["max_angle"]),
                "elapsed_seconds": float(state["elapsed_seconds"]),
                "converged": bool(state["converged"]),
            }, sort_keys=True) + "\n")

    print(f"EVA {graph_name} {rank_mode} p={p} seed={seed}", flush=True)
    y, eva = solve_eva_numba(
        a, rank=p, seed=seed, max_sweeps=max_sweeps, tolerance=tolerance,
        checkpoint_path=factor, checkpoint_every=checkpoint_every,
        on_sweep=on_sweep,
    )
    assert y.shape == (n, p)
    assert np.max(np.abs(np.linalg.norm(y, axis=1) - 1.0)) < 1e-6
    val = float(vector_objective(a, y))
    assert np.isclose(val, float(eva["objective"]), atol=1e-5, rtol=1e-8)
    factor_digest = sha256(factor)
    labels_by_method = {}
    rows = []
    for method in METHODS:
        rng = np.random.default_rng(seed_sequence(seed, graph_name, method))
        start = time.perf_counter()
        iterations = 0
        if method == "FJ_EVA":
            labels = round_fj(y, 2, rng)
        else:
            labels, iterations = round_spherical_kmeans(
                y, 2, rng, antipodal=(method == "K-MeansNM_EVA"),
                edges=edges if method == "K-MeansNM_EVA" else None,
                keep_best=(method == "K-MeansNM_EVA"),
            )
        rounded_seconds = time.perf_counter() - start
        labels = np.asarray(labels, dtype=np.uint8)
        assert labels.shape == (n,) and set(np.unique(labels)).issubset({0, 1})
        score = cut_weight(edges, labels)
        packed = np.packbits(labels)
        assert np.array_equal(np.unpackbits(packed)[:n], labels)
        labels_by_method[method] = packed
        row = {
            "graph": graph_name, "family": info["family"],
            "weight_type": info["weight_type"], "n": n, "m": int(len(edges[0])),
            "graph_sha256": graph_hash, "rank_mode": rank_mode,
            "p": p, "seed": seed, "method": method,
            "continuous_objective": val, "cut_weight": score,
            "sweeps": int(eva["sweeps"]), "converged": bool(eva["converged"]),
            "max_angle": float(eva["max_angle"]),
            "eva_seconds": float(eva["elapsed_seconds"]),
            "jit_seconds": float(eva.get("compile_seconds", 0.0)),
            "rounding_seconds": rounded_seconds,
            "cluster_iterations": int(iterations),
            "factor_sha256": factor_digest,
            "partition_sha256": hashlib.sha256(packed.tobytes()).hexdigest(),
        }
        rows.append(row)
        print(f"  {method} cut={score:.0f}; iterations={iterations}", flush=True)

    np.savez_compressed(partitions, **labels_by_method)
    result = {"settings": setting, "factor_sha256": factor_digest,
              "partitions_sha256": sha256(partitions), "rows": rows,
              "environment": {"python": sys.version.split()[0],
                              "numpy": np.__version__, "platform": platform.platform()}}
    tmp = case.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, case)
    audit_case(case, a, edges)
    return rows


def report(output: Path, expected_graphs: tuple[str, ...], seeds: tuple[int, ...],
           *, require_complete: bool = True) -> dict:
    cases = list((output / "runs").glob("*/**/case.json"))
    rows: list[dict] = []
    for file in cases:
        obj = json.loads(file.read_text(encoding="utf-8"))
        rows.extend(obj["rows"])
    rows.sort(key=lambda row: (row["graph"], row["seed"], row["rank_mode"], row["method"]))
    target = {(g, seed, rank, method) for g in expected_graphs for seed in seeds
              for rank in ("default", "full") for method in METHODS}
    actual = {(row["graph"], int(row["seed"]), row["rank_mode"], row["method"]) for row in rows}
    if len(actual) != len(rows):
        raise RuntimeError("Duplicate results detected")
    unexpected = actual - target
    if unexpected:
        raise RuntimeError(f"Unexpected graph/rank/seed/method entries: {unexpected}")
    if require_complete and actual != target:
        raise RuntimeError(f"Incomplete experiment: missing {len(target - actual)} results")
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "all_runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if not rows:
        return {"rows": 0, "complete": False}

    matched = []
    for g in expected_graphs:
        for seed in seeds:
            for method in METHODS:
                pair = [x for x in rows if x["graph"] == g and x["seed"] == seed and x["method"] == method]
                if len(pair) != 2:
                    if require_complete:
                        raise RuntimeError("Missing matched rank comparison")
                    continue
                default = next(x for x in pair if x["rank_mode"] == "default")
                full = next(x for x in pair if x["rank_mode"] == "full")
                matched.append({"graph": g, "seed": seed, "method": method,
                                "default_cut": default["cut_weight"],
                                "full_cut": full["cut_weight"],
                                "delta_full_minus_default": full["cut_weight"] - default["cut_weight"],
                                "default_objective": default["continuous_objective"],
                                "full_objective": full["continuous_objective"]})
    with (output / "paired_differences.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["graph", "seed", "method", "default_cut", "full_cut",
                                                         "delta_full_minus_default", "default_objective", "full_objective"])
        writer.writeheader()
        writer.writerows(matched)
    summary = {"graph_count": len(expected_graphs), "seeds_per_graph": len(seeds),
               "rank_modes": ["default", "full"], "methods": list(METHODS),
               "expected_rows": len(target), "observed_rows": len(rows),
               "matched_comparisons": len(matched), "complete": actual == target,
               "graphs": list(expected_graphs), "seeds": list(seeds)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    table = ["# Experiment028 — Full-rank EVA paired pilot", "",
             "This is a preliminary, five-graph experiment, **not** a certified SDP solve.",
             "Both rank settings use independent seeded EVA trajectories; matched seed identifiers",
             "do not imply identical initial embeddings across different dimensions.",
             "", f"Coverage: {len(rows)}/{len(target)} graph × seed × rank × method rows.", "",
             "| Graph | Method | Default mean | Full mean | Paired mean delta | Full better / tied / worse |",
             "|---|---|---:|---:|---:|---:|"]
    panel = []
    for g in expected_graphs:
        for method in METHODS:
            sub = [x for x in matched if x["graph"] == g and x["method"] == method]
            if not sub:
                continue
            a = np.array([x["default_cut"] for x in sub]); b = np.array([x["full_cut"] for x in sub]); d=b-a
            table.append(f"| {g} | {method} | {a.mean():.3f} | {b.mean():.3f} | {d.mean():+.3f} | "
                         f"{int((d>0).sum())}/{int((d==0).sum())}/{int((d<0).sum())} |")
            panel.append((g, method, float(100*d.mean()/abs(a.mean())) if a.mean() else 0.0))
    table.extend(["", "## Best-of-five feasible cuts (not a paired estimator)", "",
                  "| Graph | Method | Best default | Best full | Best-full minus best-default |",
                  "|---|---|---:|---:|---:|"])
    for g in expected_graphs:
        for method in METHODS:
            rr = [x for x in rows if x["graph"] == g and x["method"] == method]
            d = [float(x["cut_weight"]) for x in rr if x["rank_mode"] == "default"]
            f = [float(x["cut_weight"]) for x in rr if x["rank_mode"] == "full"]
            if d and f:
                table.append(f"| {g} | {method} | {max(d):.0f} | {max(f):.0f} | {max(f)-max(d):+.0f} |")
    table.extend(["", "## EVA objective and computational cost", "",
                  "| Graph | Rank | Mean continuous EVA objective | Converged / 5 | Mean EVA time (s) |",
                  "|---|---|---:|---:|---:|"])
    for g in expected_graphs:
        for rank in ("default", "full"):
            rr = [x for x in rows if x["graph"] == g and x["rank_mode"] == rank and x["method"] == "FJ_EVA"]
            if rr:
                table.append(f"| {g} | {rank} | {np.mean([x['continuous_objective'] for x in rr]):.6f} | "
                             f"{sum(x['converged'] for x in rr)}/{len(rr)} | {np.mean([x['eva_seconds'] for x in rr]):.2f} |")
    table.extend(["", "Every objective, sampled partition and completed EVA sweep is archived by run.",
                  "The comparisons are exploratory and not a universal rank-improvement claim.",
                  "The optimizer is nonconvex and convergence does not certify SDP optimality.", ""])
    if panel:
        maximum=max(0.1,max(abs(v) for _,_,v in panel))
        width,height=1040,105+35*len(panel)
        zero=560
        scale=365/maximum
        parts=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
               '<rect width="100%" height="100%" fill="#ffffff"/>',
               '<text x="24" y="37" font-family="Arial" font-size="24" font-weight="bold" fill="#1e293b">EVA p=n versus p=41: paired mean cut differences</text>',
               '<text x="24" y="61" font-family="Arial" font-size="14" fill="#64748b">Relative to each graph-method default-rank mean; five paired seeds, exploratory</text>',
               f'<line x1="{zero}" y1="75" x2="{zero}" y2="{height-13}" stroke="#475569" stroke-width="1.8"/>']
        for i,(g,m,pct) in enumerate(panel):
            y=94+i*35
            pretty={"FJ_EVA":"GW", "K-Means2N_EVA":"K-Means2N", "K-MeansNM_EVA":"K-MeansNM"}[m]
            parts.append(f'<text x="24" y="{y+8}" font-family="Arial" font-size="15" fill="#334155">{escape(g+" — "+pretty)}</text>')
            length=abs(pct)*scale
            x=zero if pct>=0 else zero-length
            color="#2563eb" if pct>=0 else "#c2410c"
            parts.append(f'<rect x="{x:.2f}" y="{y-9}" width="{length:.2f}" height="20" fill="{color}" rx="2"/>')
            value_x=x+length+8 if pct>=0 else x-80
            parts.append(f'<text x="{value_x:.2f}" y="{y+6}" font-family="Arial" font-size="14" fill="{color}">{pct:+.3f}%</text>')
        parts.append('</svg>')
        (output / "paired_mean_deltas.svg").write_text("\n".join(parts),encoding="utf-8")
    (output / "report.md").write_text("\n".join(table), encoding="utf-8")
    return summary


def run_graph(g: str, a, edges, info: dict[str, str], output: Path,
              seeds: tuple[int, ...], max_sweeps: int, tolerance: float,
              checkpoint_every: int) -> None:
    """One independent process per graph, no shared file writes."""
    for seed in seeds:
        for mode in ("default", "full"):
            one_case(g, a, edges, info, mode, seed, output,
                     max_sweeps=max_sweeps, tolerance=tolerance,
                     checkpoint_every=checkpoint_every)
        print(f"Completed {g}/seed {seed}", flush=True)



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+", default=list(GRAPHS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--graph-dir", type=Path, default=Path("data"))
    parser.add_argument("--features", type=Path, default=Path("results/gset/experiment022-graph-structure-methods/graph_features.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/gset/experiment028-eva-fullrank-pilot"))
    parser.add_argument("--max-sweeps", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--workers", type=int, default=1, help="Independent graph processes; max 5")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--validate-inputs-only", action="store_true", help="Verify graph hashes and dimensions, no optimization")
    parser.add_argument("--allow-synthetic", action="store_true", help="Tests only: allow n<800")
    args = parser.parse_args(argv)
    if not args.graphs or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("A nonempty, unique seed list is required")
    if len(set(args.graphs)) != len(args.graphs):
        parser.error("Graph list must contain unique IDs")
    if args.workers < 1 or args.workers > 5:
        parser.error("Workers must be 1 through 5")
    if args.max_sweeps < 1 or args.tolerance <= 0 or args.checkpoint_every < 1:
        parser.error("Invalid EVA settings")
    if not args.allow_synthetic and (tuple(args.graphs) != GRAPHS or tuple(args.seeds) != SEEDS):
        parser.error("Pilot protocol fixed to five audited 800-node graphs and five EVA seeds")
    features = metadata_from_csv(args.features)
    graph_cache = {}
    for g in args.graphs:
        path = args.graph_dir / (g + ".txt")
        if not path.is_file():
            raise FileNotFoundError(path)
        a = load_graph(path)
        if not args.allow_synthetic and a.shape != (800, 800):
            raise ValueError(f"Pilot graph {g} expected 800 vertices, found {a.shape}")
        if g not in features:
            raise ValueError(f"Missing feature provenance for {g}")
        hashes = graph_byte_hashes(path)
        info = dict(features[g])
        matches = [style for style, digest in hashes.items() if digest == info["graph_sha256"]]
        if not matches:
            raise RuntimeError(f"Graph SHA mismatch for {g}: {hashes} vs archived {info['graph_sha256']}")
        info["raw_checkout_sha256"] = hashes["raw"]
        info["matching_line_ending"] = matches[0]
        print(f"{g}: graph file identity confirmed as {matches[0]} hash",flush=True)
        if int(info["n"]) != a.shape[0]:
            raise RuntimeError(f"Graph size metadata mismatch for {g}")
        graph_cache[g] = (a, edge_arrays(a), info)
    if args.validate_inputs_only:
        print("PASS: all selected graph sizes, identities and metadata match", flush=True)
        return
    if args.check_only:
        result = report(args.output, tuple(args.graphs), tuple(args.seeds), require_complete=True)
        audited=0
        for g in args.graphs:
            a, edges, _ = graph_cache[g]
            for seed in args.seeds:
                for mode in ("default", "full"):
                    audit_case(args.output / "runs" / g / mode / f"seed-{seed}" / "case.json", a, edges)
                    audited += 1
        result["audited_factor_partition_cases"] = audited
        print(json.dumps(result, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if args.workers == 1:
        for g in args.graphs:
            a, edges, info = graph_cache[g]
            run_graph(g, a, edges, info, args.output, tuple(args.seeds),
                      args.max_sweeps, args.tolerance, args.checkpoint_every)
            report(args.output, tuple(args.graphs), tuple(args.seeds), require_complete=False)
    else:
        # Graphs occupy disjoint run directories; no shared checkpoint writer.
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(run_graph, g, *graph_cache[g], args.output,
                            tuple(args.seeds), args.max_sweeps, args.tolerance,
                            args.checkpoint_every): g for g in args.graphs
            }
            for future in as_completed(futures):
                future.result()
                print(f"Graph complete: {futures[future]}", flush=True)
    result = report(args.output, tuple(args.graphs), tuple(args.seeds), require_complete=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()