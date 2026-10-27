"""Reproducible G11 full-SDP backend and PSD-eigenprojection comparison.

Execution is intentionally subprocess-isolated per backend to contain
SDPA native memory failures and GPU CUDA solver errors. Numerical outcomes
and limitations are persisted as JSON/CSV/Markdown for Git publication.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
from scipy import sparse

from .atomic_fs import replace_atomic
from .io import load_graph, edge_arrays
from .objective import cut_weight
from .rounding import factorize, round_fj, round_spherical_kmeans


BACKENDS = (
    "cpu-scs64", "cpu-sdpa64",
    "gpu-admm32", "gpu-admm64",
    "gpu-admm32-adaptive", "gpu-admm64-adaptive",
)
EIG_BACKENDS = ("cpu-eigh64", "gpu-eigh32", "gpu-eigh64")
COLUMNS = (
    "backend", "type", "status", "solver_status", "precision",
    "solve_seconds", "eigh_kernel_seconds", "eigh_total_seconds",
    "best_FJ", "best_K-Means2N", "best_K-MeansNM",
    "cut_best", "continuous_primal_uncertified",
    "min_eigenvalue", "max_diag_error", "admm_converged",
    "primal_residual", "dual_residual",
    "rho_initial", "rho_final", "rho_updates",
    "relative_projector_error",
    "error",
)


def _jsonable(value):
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf8") as fp:
            json.dump(_jsonable(data), fp, indent=2, sort_keys=True, allow_nan=False)
            fp.write("\n")
        replace_atomic(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def _round(matrix: np.ndarray, adjacency: sparse.spmatrix, graph: str,
           restarts: int, seed: int):
    vectors, minimum, repair = factorize(
        matrix, relative_negative_tolerance=1e-3, return_diagnostics=True,
    )
    edges = edge_arrays(adjacency)
    distribution = []
    results = {}
    graph_number = int(graph[1:])
    for method in ("FJ", "K-Means2N", "K-MeansNM"):
        scores = []
        for repeat in range(restarts):
            rng = np.random.default_rng(np.random.SeedSequence(
                [seed, graph_number, 2, repeat,
                 0 if method in ("FJ", "K-MeansNM") else 1],
            ))
            if method == "FJ":
                labels = round_fj(vectors, 2, rng)
            else:
                labels, _ = round_spherical_kmeans(
                    vectors, 2, rng,
                    antipodal=method == "K-MeansNM",
                    edges=edges if method == "K-MeansNM" else None,
                    keep_best=method == "K-MeansNM",
                )
            score = float(cut_weight(edges, labels))
            scores.append(score)
            distribution.append({"method": method, "restart": repeat,
                                 "cut_weight": score})
        results[method] = {
            "best": max(scores), "mean": float(np.mean(scores)),
            "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
        }
    return results, distribution, minimum, repair


def _solver_worker(args, graph: Path) -> dict:
    a = load_graph(graph)
    backend = args.backend
    if backend.startswith("cpu-"):
        from .sdp import solve_sdp
        from .hardware import blas_limit
        solver = "SCS" if backend == "cpu-scs64" else "SDPA"
        with blas_limit(args.cpu_threads):
            matrix, solver_info = solve_sdp(
                a, k=2, solver=solver, eps=args.scs_eps,
                max_iters=args.scs_iters, verbose=False,
                compute_eigenvalues=False,
                sdpa_num_threads=args.cpu_threads if solver == "SDPA" else None,
            )
        info = {
            "solver_status": solver_info["solver_status"],
            "sdp_seconds": solver_info["sdp_seconds"],
            "sdp_objective_primal_uncertified":
                solver_info["sdp_objective_primal"],
            "solver_iterations": solver_info["solver_iterations"],
            "solver_info": solver_info,
            "precision": "float64",
        }
    else:
        from .gpu_full_sdp import full_sdp_admm_torch
        precision = "float32" if backend.startswith("gpu-admm32") else "float64"
        adaptive = backend.endswith("-adaptive")
        matrix, info = full_sdp_admm_torch(
            a, device="cuda", precision=precision,
            rho=args.rho, max_iters=args.admm_iters,
            eps_abs=args.admm_abs, eps_rel=args.admm_rel,
            log_every=args.log_every,
            adaptive_rho=adaptive,
            rho_update_interval=args.rho_update_interval,
            rho_balance_mu=args.rho_balance_mu,
            rho_scale=args.rho_scale,
            rho_min=args.rho_min, rho_max=args.rho_max,
        )
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("Solver produced nonfinite SDP matrix")
    diag_error = float(np.max(np.abs(np.diag(matrix) - 1)))
    if diag_error > 5e-3:
        raise ValueError(f"Solver diagonal infeasibility: {diag_error}")
    # Re-evaluate original unmodified SDP primal in common float64 arithmetic.
    degrees = np.asarray(a.sum(axis=1)).ravel()
    edges = edge_arrays(a)
    ii, jj, weights = edges
    actual_primal = float(
        (np.dot(degrees, np.diag(matrix)) -
         2 * np.dot(weights, matrix[ii, jj])) / 4
    )
    try:
        methods, trials, min_eig, repair_info = _round(
            matrix, a, graph.stem, args.restarts, args.seed,
        )
        rounding_status, rounding_error = "validated", None
    except (RuntimeError, ValueError) as exc:
        # An SDP solver may return an infeasible/indefinite matrix;
        # preserve its objective and numerical diagnostics for comparison.
        methods, trials, repair_info = {}, [], {}
        min_eig = float(np.linalg.eigvalsh((matrix + matrix.T) / 2)[0])
        rounding_status, rounding_error = "failed_validation", (
            f"{type(exc).__name__}: {exc}"
        )
    return {
        "backend": backend, "type": "full_convex_sdp",
        "status": "computed",
        "solver_status": info["solver_status"],
        "precision": info.get("precision", "float64"),
        "graph": graph.stem, "n": a.shape[0],
        "m": len(weights), "graph_sha256": hashlib.sha256(graph.read_bytes()).hexdigest(),
        "solve_seconds": info["sdp_seconds"],
        "continuous_primal_uncertified": actual_primal,
        "continuous_objective_reported": info.get("sdp_objective_primal_uncertified"),
        "min_eigenvalue": min_eig,
        "max_diag_error": diag_error,
        "repair": repair_info,
        "rounding_validation": rounding_status,
        "rounding_error": rounding_error,
        "admm_converged": info.get("converged"),
        "primal_residual": (info.get("residuals") or {}).get("primal_residual"),
        "dual_residual": (info.get("residuals") or {}).get("dual_residual"),
        "rho_initial": info.get("rho"),
        "rho_final": info.get("rho_final"),
        "rho_updates": info.get("rho_update_count"),
        "solver_metadata": info,
        "rounding": methods, "trials": trials,
        "environment": _environment_info(),
    }


def _environment_info() -> dict:
    try:
        import importlib.metadata
        return {
            "python": sys.version.split()[0], "platform": platform.platform(),
            "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
            "numpy": np.__version__,
            **{pkg: _safe_version(pkg) for pkg in
               ("scipy", "torch", "cvxpy", "scs", "sdpa-python")},
        }
    except Exception:
        return {"python": sys.version.split()[0]}


def _safe_version(pkg: str) -> str | None:
    import importlib.metadata
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def _eigh_worker(args, graph: Path) -> dict:
    from .gpu_full_sdp import psd_projection_eigh
    a = load_graph(graph)
    n = a.shape[0]
    rng = np.random.default_rng(args.seed)
    random = rng.standard_normal((n, n))
    matrix = (random + random.T) / (2 * np.sqrt(n))
    # Same input and true float64 reference for every backend.
    start_reference = time.perf_counter()
    w, q = np.linalg.eigh(matrix)
    reference = (q * w.clip(min=0)[None, :]) @ q.T
    reference_seconds = time.perf_counter() - start_reference
    device = "cpu" if args.backend == "cpu-eigh64" else "cuda"
    precision = "float32" if args.backend == "gpu-eigh32" else "float64"
    # Repeat timings without mixing reference computation into GPU timings.
    outcomes = []
    errors = []
    for _ in range(args.eigh_repeats):
        projected, measurement = psd_projection_eigh(
            matrix, device=device, precision=precision,
        )
        outcomes.append(measurement)
        errors.append(float(np.linalg.norm(projected - reference) /
                            max(np.linalg.norm(reference), 1e-12)))
    return {
        "backend": args.backend, "type": "psd_eigenprojection_only",
        "status": "computed", "solver_status": "not_an_SDP_solve",
        "precision": precision, "graph": graph.stem, "n": n,
        "graph_sha256": hashlib.sha256(graph.read_bytes()).hexdigest(),
        "eigh_kernel_seconds": min(v["kernel_seconds"] for v in outcomes),
        "eigh_total_seconds": min(v["end_to_end_seconds"] for v in outcomes),
        "eigh_repeats": len(outcomes),
        "relative_projector_error": min(errors),
        "reference_numpy_float64_seconds": reference_seconds,
        "measurements": outcomes, "environment": _environment_info(),
    }


def _summary(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for item in results:
            rounding = item.get("rounding", {})
            row = {k: item.get(k) for k in COLUMNS}
            for method in ("FJ", "K-Means2N", "K-MeansNM"):
                row["best_" + method] = rounding.get(method, {}).get("best")
            row["cut_best"] = (max((v["best"] for v in rounding.values()))
                               if rounding else None)
            writer.writerow(_jsonable(row))


def _publish(results_dir: Path, public_dir: Path, graph: str,
             results: list[dict]) -> None:
    public_dir.mkdir(parents=True, exist_ok=True)
    _summary(results, public_dir / "comparison.csv")
    trial_file = public_dir / "rounding-trials.csv"
    with trial_file.open("w", newline="", encoding="utf8") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=("backend", "graph", "method", "restart", "cut_weight"),
        )
        writer.writeheader()
        for item in results:
            for trial in item.get("trials", []):
                writer.writerow({"backend": item["backend"],
                                 "graph": graph, **trial})
    cases = public_dir / "cases"
    cases.mkdir(exist_ok=True)
    public_logs = public_dir / "logs"
    public_logs.mkdir(exist_ok=True)
    for item in results:
        # Result JSON contains no absolute local source/log paths.
        atomic_json(cases / (item["backend"] + ".json"), item)
        raw = results_dir / (item["backend"] + ".log")
        if raw.is_file():
            with raw.open("rb") as log:
                log.seek(0, 2)
                size = log.tell()
                log.seek(max(0, size - 200000))
                tail = log.read().decode("utf8", errors="replace")
            tail = tail.replace(str(Path.home()), "<HOME>")
            tail = tail.replace(str(Path.cwd()), "<REPO>")
            (public_logs / (item["backend"] + ".txt")).write_text(
                tail, encoding="utf8",
            )
    lines = [
        f"# {graph} — full convex SDP GPU precision and penalty-parameter study",
        "",
        "This is a **new numerical comparison**, not a claim of improved "
        "quality or runtime until observed measurements are present.",
        "",
        "## Approaches",
        "",
        "- CPU SCS (float64), original full convex SDP reference.",
        "- CPU SDPA (float64), alternative full convex solver.",
        "- GPU full-rank PSD ADMM (float32 and float64), fixed or " 
        "residual-balanced adaptive rho: the **same convex "
        "SDP formulation but a different first-order algorithm**. Its "
        "iteration limit is not an optimality certificate.",
        "- Independent CPU/GPU PSD eigenprojection microbenchmark: "
        "**kernel time is NOT SDP solver time**.",
        "",
        "## Saved results",
        "",
        "[comparison.csv](comparison.csv) — solver status, timings, "
        "objective, feasibility and cut quality.",
        "",
        "[rounding-trials.csv](rounding-trials.csv) — all rounded cuts.",
        "",
        "[cases/](cases/) — complete backend JSON diagnostics.",

        "[logs/](logs/) — sanitized bounded execution-log tails.",
        "",
        "| Backend | Status | Solver status | Seconds | Best cut |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for item in results:
        bests = item.get("rounding", {})
        best = max((v["best"] for v in bests.values()), default=None)
        duration = item.get("solve_seconds", item.get("eigh_total_seconds"))
        lines.append(
            f"| {item['backend']} | {item.get('status')} | "
            f"{item.get('solver_status','—')} | "
            f"{duration if duration is not None else '—'} | "
            f"{best if best is not None else '—'} |"
        )
    lines.extend([
        "",
        "## Scientific caveats",
        "",
        "- GPU ADMM is a new experimental solver, **not** a patch to SCS.",
        "- Adaptive rho changes the numerical penalty only, not the original " 
        "convex SDP; the *scaled* dual is rescaled whenever rho changes.",
        "- Residual-balanced convergence does not certify SDP optimality " 
        "or a numerical upper bound. Compare against fixed rho at the same " 
        "absolute/relative tolerances, graph SHA and iteration budgets.",
        "- Same convex feasible region, but different first-order numerical "
        "iterations, residual criteria and finite-precision effects.",
        "- float32 trades precision for potentially faster matrix operations; "
        "do not assume it reproduces float64 solver outputs.",
        "- An SDP primal objective is **not** a certified upper bound.",
        "- All rounded cuts are re-evaluated against original signed edges.",
        "- An ADMM iteration-limit result is approximate/unconverged.",
        "- Never compare eigenprojection kernel time to total SCS solve time.",
        "- Failed cases are retained and shown, not silently omitted.",
        "",
    ])
    (public_dir / "README.md").write_text("\n".join(lines), encoding="utf8")


def _push_publication(public_dir: Path) -> None:
    """Opt-in publish only this experiment's results, never ignored reports.

    Commit/push is non-forced and requires main; a remote divergence
    must be reconciled manually, without disrupting numerical results.
    """
    try:
        project = Path(subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()).resolve()
        relative = public_dir.resolve().relative_to(project).as_posix()
        if not relative.startswith("results/"):
            raise ValueError("Refusing to stage outside public results/")
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if branch != "main":
            raise RuntimeError(f"GitHub publishing requires main, not {branch!r}")
        subprocess.run(["git", "add", "-A", "--", relative], check=True)
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--", relative],
            check=False,
        )
        if diff.returncode == 0:
            return
        if diff.returncode != 1:
            raise RuntimeError("Failed to compare staged research result files")
        subprocess.run([
            "git", "commit", "-m",
            "results: checkpoint G11 CPU SDPA SCS GPU FP32/FP64 benchmark",
            "--", relative,
        ], check=True)
        subprocess.run(
            ["git", "push", "origin", "HEAD:main"],
            check=True, timeout=120,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        print("Benchmark checkpoint pushed to GitHub.", flush=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            RuntimeError, ValueError, OSError) as exc:
        print(
            f"WARNING: GitHub publication failed: {type(exc).__name__}: {exc}. "
            "Local results retained; no forced update attempted.",
            file=sys.stderr, flush=True,
        )


def _backend_worker_prefix(backend: str) -> list[str]:
    """Launch CUDA/torch first, before SciPy/CVXPY load native Windows DLLs.

    Each solver is an independent Python process; keeping torch first
    avoids Windows shm.dll WinError 127 seen on late torch import after
    the scientific stack in a long-running pytest session.
    """
    if backend.startswith("gpu-") or backend == "cpu-eigh64":
        return [
            sys.executable, "-u", "-c",
            "import torch; from maxkcut.precision_study import main; "
            "raise SystemExit(main())",
        ]
    return [sys.executable, "-u", "-m", "maxkcut.precision_study"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="G11",
                        help="Canonical modest-sized G-set graph (default G11, n=800)")
    parser.add_argument("--gset-dir", type=Path, default=Path("data"))
    parser.add_argument("--backend", choices=(*BACKENDS, *EIG_BACKENDS))
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--push", action="store_true",
                        help="Commit/push this study's sanitized results after EACH backend")
    parser.add_argument("--backends", nargs="+",
                        choices=(*BACKENDS, *EIG_BACKENDS),
                        default=[*BACKENDS[:4], *EIG_BACKENDS])
    parser.add_argument("--output-dir", type=Path,
                        default=Path("reports/gset/precision-psd-benchmark"))
    parser.add_argument("--publish-dir", type=Path,
                        default=Path("results/gset/precision-psd-benchmark"))
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--scs-eps", type=float, default=1e-3)
    parser.add_argument("--scs-iters", type=int, default=3000)
    parser.add_argument("--admm-iters", type=int, default=120)
    parser.add_argument("--admm-abs", type=float, default=1e-4)
    parser.add_argument("--admm-rel", type=float, default=1e-3)
    parser.add_argument("--rho", type=float, default=1.)
    parser.add_argument("--rho-update-interval", type=int, default=25)
    parser.add_argument("--rho-balance-mu", type=float, default=10.)
    parser.add_argument("--rho-scale", type=float, default=2.)
    parser.add_argument("--rho-min", type=float, default=0.0625)
    parser.add_argument("--rho-max", type=float, default=16.)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eigh-repeats", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args(argv)
    if (args.cpu_threads < 1 or args.restarts < 1 or args.admm_iters < 1
            or args.scs_iters < 1 or args.eigh_repeats < 1
            or args.timeout_seconds < 60
            or args.rho <= 0 or args.rho_update_interval < 1
            or args.rho_balance_mu <= 1 or args.rho_scale <= 1
            or args.rho_min <= 0 or args.rho_max < args.rho_min):
        parser.error("Invalid execution budget")
    try:
        from .sweep import discover_graphs
        graph_path = discover_graphs(args.gset_dir, [args.graph])[0]
    except (OSError, ValueError, IndexError) as exc:
        parser.error(f"Cannot locate requested graph: {exc}")
    if args.worker:
        if args.backend is None:
            parser.error("--worker needs --backend")
        try:
            result = (_eigh_worker(args, graph_path)
                      if args.backend in EIG_BACKENDS else
                      _solver_worker(args, graph_path))
            atomic_json(args.output_dir / "cases" /
                        (args.backend + ".json"), result)
            return 0
        except Exception as exc:
            traceback.print_exc()
            atomic_json(args.output_dir / "cases" /
                        (args.backend + ".json"), {
                            "backend": args.backend,
                            "type": "psd_eigenprojection_only" if args.backend in EIG_BACKENDS
                                    else "full_convex_sdp",
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "solver_status": "failed",
                        })
            return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    study_plan = {
        "graph": graph_path.stem,
        "graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
        "backends": list(args.backends), "seed": args.seed,
        "restarts": args.restarts, "cpu_threads": args.cpu_threads,
        "scs_eps": args.scs_eps, "scs_iters": args.scs_iters,
        "admm_iters": args.admm_iters,
        "admm_abs": args.admm_abs, "admm_rel": args.admm_rel,
        "rho": args.rho, "eigh_repeats": args.eigh_repeats,
        "rho_update_interval": args.rho_update_interval,
        "rho_balance_mu": args.rho_balance_mu,
        "rho_scale": args.rho_scale,
        "rho_min": args.rho_min, "rho_max": args.rho_max,
    }
    plan_file = args.output_dir / "study-plan.json"
    if plan_file.is_file() and json.loads(plan_file.read_text()) != study_plan:
        parser.error("Experiment settings changed; use another --output-dir")
    atomic_json(plan_file, study_plan)
    atomic_json(args.publish_dir / "study-plan.json", study_plan)
    for backend in args.backends:
        case = args.output_dir / "cases" / (backend + ".json")
        case.unlink(missing_ok=True)  # Never reuse a stale success on a failed rerun.
        cmd = _backend_worker_prefix(backend) + [
            "--worker", "--backend", backend,
            "--graph", graph_path.stem,
            "--gset-dir", str(args.gset_dir),
            "--output-dir", str(args.output_dir),
            "--cpu-threads", str(args.cpu_threads),
            "--restarts", str(args.restarts),
            "--seed", str(args.seed),
            "--scs-eps", str(args.scs_eps),
            "--scs-iters", str(args.scs_iters),
            "--admm-iters", str(args.admm_iters),
            "--admm-abs", str(args.admm_abs),
            "--admm-rel", str(args.admm_rel),
            "--rho", str(args.rho),
            "--rho-update-interval", str(args.rho_update_interval),
            "--rho-balance-mu", str(args.rho_balance_mu),
            "--rho-scale", str(args.rho_scale),
            "--rho-min", str(args.rho_min),
            "--rho-max", str(args.rho_max),
            "--log-every", str(args.log_every),
            "--eigh-repeats", str(args.eigh_repeats),
        ]
        print(f"Starting {backend} (independent subprocess)", flush=True)
        log = args.output_dir / (backend + ".log")
        start = time.perf_counter()
        try:
            with log.open("w", encoding="utf8") as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                               timeout=args.timeout_seconds, check=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            if not case.is_file():
                atomic_json(case, {"backend": backend, "status": "failed",
                                   "solver_status": "failed",
                                   "error": f"{type(exc).__name__}: {exc}",
                                   "type": "psd_eigenprojection_only" if backend in EIG_BACKENDS
                                           else "full_convex_sdp"})
        print(f"{backend} finished in {time.perf_counter()-start:.1f} sec; "
              f"case={case}", flush=True)
        saved = [
            json.loads(path.read_text(encoding="utf8"))
            for path in sorted((args.output_dir / "cases").glob("*.json"))
            if path.stem in args.backends
        ]
        _publish(args.output_dir, args.publish_dir, graph_path.stem, saved)
        if args.push:
            _push_publication(args.publish_dir)
    print(f"GitHub-ready files: {args.publish_dir}", flush=True)
    print(
        "Results are locally preserved. "
        + ("Per-backend GitHub push was enabled." if args.push else
           "Use --push to publish after each backend, or commit/push manually."),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
