"""Strict, archive-verified full convex SDP versus CUDA low-rank comparison.

Consumes existing JSONL and independent archived graph snapshots. Produces
one graph per CSV row. Missing SDP results stay missing; non-identical graphs
cannot silently receive a delta, literature optimum ratio or speedup.

The convex SDP solve must be performed separately by the unchanged
`maxkcut` CLI; this module does NOT launch any solver or allocate a Gram
matrix. CUDA and SDP output use different relaxation/rounding labels.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .io import edge_arrays, load_graph
from .references import reference_for_graph


SDP_METHODS = ("FJ", "K-Means2N", "K-MeansNM")
CUDA_METHODS = (
    "GW_torchlowrank", "K-Means2N_torchlowrank",
    "K-MeansNM_torchlowrank",
)
UNSOLVED_THESIS_GRAPHS = (
    "G55", "G56", "G60", "G61", "G70", "G77", "G81",
)
COLUMNS = (
    "graph", "n", "edges", "graph_identity", "sdp_status",
    "sdp_GW_best", "sdp_KMeans2N_best", "sdp_KMeansNM_best",
    "sdp_best_method", "sdp_best_cut", "sdp_best_mean",
    "sdp_restarts_per_method", "sdp_seconds_original", "sdp_cache_hit",
    "sdp_solver", "sdp_primal_objective_uncertified",
    "cuda_GW_best", "cuda_KMeans2N_best", "cuda_KMeansNM_best",
    "cuda_best_method", "cuda_best_cut", "cuda_best_mean",
    "cuda_rounding_restarts_per_method",
    "cuda_seconds_original", "cuda_cache_hit", "cuda_rank",
    "cuda_epochs", "cuda_optimizer_starts",
    "cuda_device", "sdp_minus_cuda_cut",
    "sdp_vs_cuda_optimizer_time_ratio_uncached",
    "paper_cut", "paper_upper_bound", "reference_status",
    "cuda_gap_to_paper", "sdp_gap_to_paper",
    "cuda_percent_of_proven_optimum",
    "sdp_percent_of_proven_optimum",
    "paper_assessment", "sdp_source", "cuda_source",
)

def _load_jsonl(path: Path, *, expected: tuple[str, ...]) -> dict:
    path = Path(path)
    records = {}
    with path.open("r", encoding="utf8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            if int(row.get("k", -1)) != 2:
                raise ValueError(f"{path}:{line_number}: comparison supports k=2 only")
            method = row.get("method")
            if method not in expected:
                raise ValueError(
                    f"{path}:{line_number}: unexpected method {method!r}"
                )
            if "graph" not in row or "n" not in row or "cut_weight" not in row:
                raise ValueError(f"{path}:{line_number}: missing graph, n or cut_weight")
            graph = str(row["graph"])
            if re.fullmatch(r"G[1-9][0-9]*", graph) is None:
                raise ValueError(f"{path}:{line_number}: invalid G-set graph {graph!r}")
            cut = float(row["cut_weight"])
            if not np.isfinite(cut):
                raise ValueError(f"{path}:{line_number}: non-finite cut")
            run_id = row.get("artifact_run_id")
            if (not isinstance(run_id, str)
                    or re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None):
                raise ValueError(
                    f"{path}:{line_number}: missing archive run identifier; "
                    "run with the default artifact archiving enabled"
                )
            entry = records.setdefault(graph, {
                "n": int(row["n"]), "run_id": run_id,
                "methods": {name: [] for name in expected}, "sample": row,
                "source": str(path),
            })
            if entry["n"] != int(row["n"]) or entry["run_id"] != run_id:
                raise ValueError(
                    f"{path}: multiple runs for {graph}; supply a single "
                    "run per graph per input file to avoid mixing budgets"
                )
            entry["methods"][method].append(cut)
    return records


def _graph_fingerprint(n: int, edges: tuple) -> str:
    """Fingerprint normalized, sorted signed weighted graph, not file bytes."""
    i, j, w = edges
    i = np.asarray(i, dtype="<i8")
    j = np.asarray(j, dtype="<i8")
    w = np.asarray(w, dtype="<f8")
    order = np.lexsort((j, i))
    digest = hashlib.sha256()
    digest.update(np.asarray([n, len(i)], dtype="<i8").tobytes())
    for array in (i, j, w):
        digest.update(np.ascontiguousarray(array[order]).tobytes())
    return digest.hexdigest()


def _archive_folder(source: dict) -> Path:
    return (
        Path(source["source"]).parent / "artifacts"
        / source["run_id"] / f"{source['sample']['graph']}-k2"
    )


def _archived_settings(source: dict | None) -> dict:
    if source is None:
        return {}
    path = _archive_folder(source) / "manifest.json"
    if not path.is_file():
        return {}
    manifest = json.loads(path.read_text(encoding="utf8"))
    return manifest.get("settings", {})


def _archived_identity(source: dict) -> str | None:
    path = _archive_folder(source) / "graph.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as saved:
        n = int(saved["n"].item())
        return _graph_fingerprint(
            n, (saved["i"], saved["j"], saved["weights"])
        )


def _summarize(source: dict, methods: tuple[str, ...]) -> dict:
    stats = {}
    for method in methods:
        scores = source["methods"][method]
        stats[method] = (
            None if not scores
            else {
                "best": max(scores),
                "mean": float(np.mean(scores)),
                "n": len(scores),
            }
        )
    present = [m for m in methods if stats[m] is not None]
    if not present:
        return {
            "stats": stats, "best_method": None, "best_cut": None,
            "best_mean": None, "restarts": None,
        }
    top = max(present, key=lambda m: stats[m]["best"])
    return {
        "stats": stats, "best_method": top, "best_cut": stats[top]["best"],
        "best_mean": stats[top]["mean"],
        "restarts": (
            stats[present[0]]["n"]
            if len({stats[m]["n"] for m in present}) == 1 else None
        ),
    }


def _original_seconds(source: dict | None, field: str) -> float | None:
    if not source:
        return None
    value = source["sample"].get(field)
    return float(value) if value is not None else None


def _canonical_fingerprint(gset_dir: Path, graph: str) -> tuple[int, int, str] | None:
    filename = Path(gset_dir) / f"{graph}.txt"
    if not filename.exists():
        return None
    adjacency = load_graph(filename)
    edges = edge_arrays(adjacency)
    return (
        adjacency.shape[0], len(edges[0]),
        _graph_fingerprint(adjacency.shape[0], edges),
    )


def compare_saved_runs(
    sdp_jsonl: list[Path], cuda_jsonl: Path, gset_dir: Path,
) -> list[dict]:
    """Compare exact cuts only after archived graph snapshots match canonical data.

    The performance comparison is between two DIFFERENT relaxation solvers,
    evaluated with the same named clustering algorithms. The full SDP primal
    score is explicitly not called a certified upper bound.
    """
    cuda = _load_jsonl(cuda_jsonl, expected=CUDA_METHODS)
    sdp = {}
    for path in sdp_jsonl:
        for graph, entry in _load_jsonl(path, expected=SDP_METHODS).items():
            if graph in sdp:
                raise ValueError(
                    f"Multiple full-SDP runs for {graph}: "
                    "provide exactly one selected configuration"
                )
            sdp[graph] = entry
    output = []
    for graph in sorted(set(cuda) | set(sdp),
                        key=lambda name: int(name[1:])):
        a = sdp.get(graph)
        b = cuda.get(graph)
        canonical = _canonical_fingerprint(gset_dir, graph)
        sdp_hash = _archived_identity(a) if a else None
        cuda_hash = _archived_identity(b) if b else None
        if canonical is None:
            identity = "NO_CANONICAL_TXT"
        elif ((a and sdp_hash is None) or (b and cuda_hash is None)):
            identity = "MISSING_ARCHIVE"
        elif ((a and a["n"] != canonical[0]) or
              (b and b["n"] != canonical[0]) or
              (a and sdp_hash != canonical[2]) or
              (b and cuda_hash != canonical[2])):
            identity = "GRAPH_MISMATCH"
        else:
            identity = "VERIFIED_CANONICAL"

        s = _summarize(a, SDP_METHODS) if a else None
        c = _summarize(b, CUDA_METHODS) if b else None
        sb = s["best_cut"] if s else None
        cb = c["best_cut"] if c else None
        stime = _original_seconds(a, "sdp_original_solve_seconds")
        ctime = _original_seconds(b, "original_optimizer_seconds")
        cuda_settings = _archived_settings(b)
        s_cache = a["sample"].get("cache_hit") if a else None
        c_cache = b["sample"].get("factor_cache_hit") if b else None
        ref = reference_for_graph(graph)
        same_and_complete = (identity == "VERIFIED_CANONICAL")
        ref_ok = (
            same_and_complete and ref is not None and canonical is not None
            and canonical[0] == ref["reference_vertices"]
        )
        paper_cut = ref["published_cut"] if ref else None
        upper = ref["published_upper_bound"] if ref else None
        if ref is None:
            paper_assessment = "NO_PAPER_TABLE_1_REFERENCE"
        elif not ref_ok:
            paper_assessment = "UNVERIFIED_GRAPH_IDENTITY_OR_SIZE"
        elif any(val is not None and val > upper + 1e-8 for val in (sb, cb)):
            paper_assessment = "ABOVE_PUBLISHED_BOUND_CHECK_DATA"
        else:
            paper_assessment = "COMPARABLE"
        best_sdp_method = s["best_method"] if s else None
        best_cuda_method = c["best_method"] if c else None
        row = {
            "graph": graph, "n": canonical[0] if canonical else
                (a or b)["n"],
            "edges": canonical[1] if canonical else None,
            "graph_identity": identity,
            "sdp_status": a["sample"].get("solver_status") if a else "NOT_RUN",
            "sdp_GW_best": s["stats"]["FJ"]["best"] if s and s["stats"]["FJ"] else None,
            "sdp_KMeans2N_best": s["stats"]["K-Means2N"]["best"]
                if s and s["stats"]["K-Means2N"] else None,
            "sdp_KMeansNM_best": s["stats"]["K-MeansNM"]["best"]
                if s and s["stats"]["K-MeansNM"] else None,
            "sdp_best_method": best_sdp_method,
            "sdp_best_cut": sb, "sdp_best_mean": s["best_mean"] if s else None,
            "sdp_restarts_per_method": s["restarts"] if s else None,
            "sdp_seconds_original": stime, "sdp_cache_hit": s_cache,
            "sdp_solver": a["sample"].get("solver") if a else None,
            "sdp_primal_objective_uncertified":
                a["sample"].get("sdp_objective_primal") if a else None,
            "cuda_GW_best": c["stats"]["GW_torchlowrank"]["best"]
                if c and c["stats"]["GW_torchlowrank"] else None,
            "cuda_KMeans2N_best": c["stats"]["K-Means2N_torchlowrank"]["best"]
                if c and c["stats"]["K-Means2N_torchlowrank"] else None,
            "cuda_KMeansNM_best": c["stats"]["K-MeansNM_torchlowrank"]["best"]
                if c and c["stats"]["K-MeansNM_torchlowrank"] else None,
            "cuda_best_method": best_cuda_method,
            "cuda_best_cut": cb, "cuda_best_mean": c["best_mean"] if c else None,
            "cuda_rounding_restarts_per_method": c["restarts"] if c else None,
            "cuda_seconds_original": ctime, "cuda_cache_hit": c_cache,
            "cuda_rank": b["sample"].get("rank") if b else None,
            "cuda_epochs": cuda_settings.get("epochs"),
            "cuda_optimizer_starts": cuda_settings.get("optimizer_starts"),
            "cuda_device": b["sample"].get("device") if b else None,
            "sdp_minus_cuda_cut": (
                sb - cb if same_and_complete and sb is not None and cb is not None
                else None
            ),
            "sdp_vs_cuda_optimizer_time_ratio_uncached": (
                stime / ctime
                if same_and_complete and stime is not None and ctime is not None
                and stime > 0 and ctime > 0
                else None
            ),
            "paper_cut": paper_cut,
            "paper_upper_bound": upper,
            "reference_status": (
                ref["reference_status"] if ref else "unavailable"
            ),
            "cuda_gap_to_paper": (
                paper_cut - cb
                if ref_ok and cb is not None else None
            ),
            "sdp_gap_to_paper": (
                paper_cut - sb
                if ref_ok and sb is not None else None
            ),
            "cuda_percent_of_proven_optimum": (
                100 * cb / paper_cut
                if ref_ok and cb is not None
                and ref["reference_status"] == "proven_optimum"
                and cb <= upper + 1e-8 else None
            ),
            "sdp_percent_of_proven_optimum": (
                100 * sb / paper_cut
                if ref_ok and sb is not None
                and ref["reference_status"] == "proven_optimum"
                and sb <= upper + 1e-8 else None
            ),
            "paper_assessment": paper_assessment,
            "sdp_source": a["source"] if a else None,
            "cuda_source": b["source"] if b else None,
        }
        output.append(row)
    return output


def write_study_comparison(
    sdp_jsonl: list[Path], cuda_jsonl: Path, gset_dir: Path, output: Path,
) -> list[dict]:
    rows = compare_saved_runs(sdp_jsonl, cuda_jsonl, gset_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def full_sdp_preflight(gset_dir: Path, graphs: tuple[str, ...]) -> list[dict]:
    """Cost FLOOR for a dense SDP, not a promise that a solve will fit."""
    results = []
    for graph in graphs:
        filename = Path(gset_dir) / f"{graph}.txt"
        if not filename.is_file():
            raise FileNotFoundError(f"Missing canonical G-set input: {filename}")
        with filename.open("r", encoding="utf8") as handle:
            header = handle.readline().split()
        if len(header) != 2:
            raise ValueError(f"Bad canonical G-set header: {filename}")
        n, m = map(int, header)
        if n < 2 or m < 0:
            raise ValueError(f"Invalid graph dimensions: {filename}")
        results.append({
            "graph": graph, "n": n, "edges": m,
            "one_dense_float64_gram_gib": round(8 * n * n / 2 ** 30, 3),
            "message": (
                "VERY_LARGE_FULL_SDP_avoid_dense_CVXPY_attempt"
                if n > 3000 else
                "LARGE_FULL_SDP_use_explicit_budget"
                if n > 1000 else "small_pilot_candidate"
            ),
        })
    return results
