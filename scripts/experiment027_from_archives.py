"""Experiment027: run NEW angular methods against exact Experiment025 EVA factors.

This is a thin, read-only factor-source selector and graph-parallel launcher.
It intentionally DOES NOT edit Experiment027's source-hashed numerical runner,
Experiment025 results, or source EVA factors. Source availability and SHA must
be validated for every requested graph before any new numerical process starts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

FACTOR_RELATIVE_DIRS = (
    "results/gset/experiment021-complete-71-gset/eva-numba",
    "results/gset/experiment003-eva-numba",
    "results/gset/experiment002-eva",
)
ALL_GRAPHS = tuple([f"G{i}" for i in range(1, 68)] + ["G70", "G72", "G77", "G81"])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_pinned_factor(graph: str, original: Path, baseline_root: Path, data_dir: Path):
    """Find ONLY the original verified factor used for graph in Experiment025.

    Never select by directory order alone when historical SHA evidence exists.
    A missing historical baseline is an error, not an excuse to replace the factor.
    """
    status_path = baseline_root / "full" / graph / "gw-rr" / "status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"Experiment025 full GW baseline unavailable: {status_path}")
    baseline = json.loads(status_path.read_text(encoding="utf-8"))
    if baseline.get("status") != "complete_16_best_seen" or baseline.get("starts_completed") != 16:
        raise ValueError(f"Experiment025 full GW baseline not complete for {graph}")
    historical_factor = baseline.get("factor_sha256")
    historical_graph = baseline.get("graph_sha256")
    if not isinstance(historical_factor, str) or len(historical_factor) != 64:
        raise ValueError(f"Missing historical factor SHA256 for {graph}")
    graph_file = data_dir / f"{graph}.txt"
    if not graph_file.is_file() or file_sha256(graph_file) != historical_graph:
        raise ValueError(f"Historical original graph SHA256 mismatch/missing for {graph}: {graph_file}")

    attempted = []
    for relative in FACTOR_RELATIVE_DIRS:
        folder = original / relative
        case_file = folder / "cases" / f"{graph}.json"
        factor_file = folder / "factors" / f"{graph}.npz"
        if not case_file.is_file() or not factor_file.is_file():
            attempted.append(f"{folder} [case or factor missing]")
            continue
        case = json.loads(case_file.read_text(encoding="utf-8"))
        if case.get("graph_sha256") != historical_graph or case.get("status") not in ("converged", "iteration_limit"):
            attempted.append(f"{folder} [case status/graph mismatch]")
            continue
        actual_sha = file_sha256(factor_file)
        if actual_sha == historical_factor:
            return folder, historical_factor
        attempted.append(f"{folder} [factor SHA mismatch: {actual_sha[:16]}]")
    raise FileNotFoundError(
        f"Cannot find EXACT Experiment025 factor for {graph} ({historical_factor[:16]}).\n"
        + "\n".join(attempted)
        + "\nDo not rerun EVA or substitute another factor. Check the original VM archives."
    )


def _run_graph(graph: str, source: Path, args, project: Path, baseline: Path, data: Path, dest: Path):
    script = project / "scripts" / "experiment027_angular.py"
    cmd = [sys.executable, str(script), "run", "--graphs", graph,
           "--phase", "full", "--method", "all", "--workers", "1",
           "--factor-dir", str(source), "--data-dir", str(data),
           "--baseline-root", str(baseline), "--root", str(dest)]
    if args.seed is not None:
        cmd.extend(("--seed", str(args.seed)))
    if args.rounds is not None:
        cmd.extend(("--rounds", str(args.rounds)))
    if args.candidates is not None:
        cmd.extend(("--candidates", str(args.candidates)))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[name] = "1"
    result = subprocess.run(cmd, cwd=project, env=env, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", check=False)
    logdir = dest / "launcher-logs"
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / f"{graph}.log").write_text(
        f"exit_code={result.returncode}\n" + result.stdout + "\nSTDERR\n" + result.stderr,
        encoding="utf-8")
    return graph, result.returncode, result.stdout[-1200:], result.stderr[-1200:]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+", default=["G1", "G11", "G14"],
                        help="Graph identifiers or ALL")
    parser.add_argument("--original", type=Path, default=Path.home() / "maxkcut")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, choices=range(16), default=None,
                        help="Optional 0..15 starting direction; omit for all sixteen")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--candidates", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.rounds is not None and args.rounds < 1) or (args.candidates is not None and args.candidates < 1):
        parser.error("workers, rounds and candidates must be positive")
    graphs = list(ALL_GRAPHS if args.graphs == ["ALL"] else args.graphs)
    if not graphs or len(set(graphs)) != len(graphs) or any(g not in ALL_GRAPHS for g in graphs):
        parser.error("Graphs must be unique canonical G-set identifiers")
    project = Path(__file__).resolve().parents[1]
    original = args.original.resolve()
    baseline = original / "results/gset/experiment025-gw16-bestcut-quality"
    data = original / "data"
    dest = project / "results/gset/experiment027-angular-greedy"
    if not (project / "scripts/experiment027_angular.py").is_file():
        parser.error("Experiment027 numerical runner missing from current worktree")
    # Resolve every requested graph BEFORE starting any new computation.
    selected = {g: select_pinned_factor(g, original, baseline, data) for g in graphs}
    plan = dict(experiment="experiment027", graphs=len(graphs), methods=3,
                starts=1 if args.seed is not None else 16,
                new_runs_max=len(graphs)*3*(1 if args.seed is not None else 16),
                repeated_eva=False, repeated_historical_methods=False,
                factor_sources={g:str(path) for g,(path,_) in selected.items()},
                output=str(dest), project=str(project))
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return 0
    errors = []
    with ThreadPoolExecutor(max_workers=min(len(graphs),args.workers)) as pool:
        futures = {pool.submit(_run_graph,g,selected[g][0],args,project,baseline,data,dest):g for g in graphs}
        for future in as_completed(futures):
            graph, code, stdout, stderr = future.result()
            print(json.dumps(dict(graph=graph, exit_code=code, stdout_tail=stdout, stderr_tail=stderr)),flush=True)
            if code:
                errors.append(graph)
    if errors:
        print(f"FAILED graphs: {', '.join(errors)}. Inspect {dest/'launcher-logs'}",file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
