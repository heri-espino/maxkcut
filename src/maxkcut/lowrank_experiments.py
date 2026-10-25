"""GPU/CPU low-rank Max-Cut benchmark reusing the thesis rounding functions.

Unlike src/maxkcut/experiments.py this does NOT solve the full convex SDP.
The output method labels and JSONL include an explicit 'torch_lowrank' tag,
preventing accidental combination with thesis SDP-based GW/K-Means runs.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from .archive import (
    method_archive_path, new_archive, new_run_id, save_method, update_manifest,
)
from .io import load_graph, edge_arrays
from .lowrank import optimize_torch_lowrank, torch_device_info
from .lowrank_cache import (
    factor_cache_path, factor_identity, load_factor, save_factor,
)
from .objective import cut_weight
from .rounding import round_fj, round_spherical_kmeans


def benchmark_lowrank(paths, output, *, device="cuda", rank=32,
                      epochs=200, optimizer_starts=2, learning_rate=0.05,
                      log_every=10, restarts=20, seed=2026, max_n=1000,
                      factor_cache_dir=Path("reports/gset/lowrank_cache"),
                      use_factor_cache=True, save_artifacts=True,
                      artifacts_dir=None, argv=None) -> None:
    """Optimize one rank-r vector factor per graph, round and archive all cuts.

    Performs a single low-rank training procedure (with the requested
    *optimizer_starts*) per graph; each of three thesis-inspired rounding
    methods gets its own *restarts* randomized trials on the same factor.
    """
    if restarts < 1 or max_n < 2:
        raise ValueError("restarts must be positive and max_n >= 2")
    # Fail before writing output if the user requested unavailable CUDA.
    hardware = torch_device_info(device)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if artifacts_dir is None:
        artifacts_dir = output.parent / "artifacts"

    with output.open("a", encoding="utf8") as stream:
        for path in paths:
            path = Path(path)
            graph = load_graph(path)
            n = graph.shape[0]
            if n > max_n:
                raise ValueError(
                    f"{path.name}: n={n} > --max-n={max_n}; "
                    "increase max-n explicitly for large instances"
                )
            if rank > n:
                raise ValueError(f"{path.name}: --rank={rank} exceeds n={n}")
            edges = edge_arrays(graph)
            case_seed = int(
                np.random.SeedSequence([seed, int(path.stem[1:])])
                .generate_state(1, dtype=np.uint32)[0]
            )
            settings = {
                "backend": "torch_lowrank",
                "device": device,
                "rank": rank,
                "epochs": epochs,
                "optimizer_starts": optimizer_starts,
                "learning_rate": learning_rate,
                "log_every": log_every,
                "optimizer_seed": case_seed,
                "torch_version": hardware["torch_version"],
                "torch_cuda_version": hardware["torch_cuda_version"],
                "gpu_name": hardware["gpu_name"],
                "dtype": "float32",
            }
            identity = factor_identity(path, settings)
            cache_path = factor_cache_path(factor_cache_dir, path, identity)

            archive = None
            if save_artifacts:
                archive_settings = dict(settings)
                archive_settings.update({
                    "rounding_restarts": restarts,
                    "rounding_seed": seed,
                    "max_n": max_n,
                    "factor_cache_enabled": bool(use_factor_cache),
                })
                archive = new_archive(
                    artifacts_dir, new_run_id(), path, n, edges, 2,
                    archive_settings, jsonl_output=output, argv=argv,
                )
                update_manifest(
                    archive, backend="torch_lowrank",
                    interpretation="nonconvex_rank_restricted_primal_heuristic",
                    valid_sdp_upper_bound=False,
                )

            started = time.monotonic()
            cached = (
                load_factor(cache_path, identity, n=n, rank=rank)
                if use_factor_cache else None
            )
            cache_hit = cached is not None
            if cached:
                vectors, optimizer_info = cached
                optimizer_info = dict(optimizer_info)
            else:
                vectors, optimizer_info = optimize_torch_lowrank(
                    graph, device=device, rank=rank, epochs=epochs,
                    optimizer_starts=optimizer_starts,
                    learning_rate=learning_rate, log_every=log_every,
                    seed=case_seed,
                )
                if use_factor_cache:
                    save_factor(cache_path, identity, vectors, optimizer_info)
            prep_seconds = time.monotonic() - started
            if archive is not None:
                # Store one reusable rank-r factor per run, NEVER n-by-n X.
                factor_path = archive["path"] / "lowrank_factors.npz"
                save_factor(factor_path, identity, vectors, optimizer_info)
                update_manifest(
                    archive, status="factor_ready",
                    factor_file=factor_path.name,
                    factor_cache_path=str(cache_path) if use_factor_cache else None,
                    cache_hit=cache_hit,
                    optimizer_info=optimizer_info,
                )

            methods = ("GW_torchlowrank", "K-Means2N_torchlowrank",
                       "K-MeansNM_torchlowrank")
            for method in methods:
                method_labels = []
                method_scores = []
                best = float("-inf")
                for rep in range(restarts):
                    # The same seed must be used for paired GW vs K-MeansNM.
                    rng = np.random.default_rng(np.random.SeedSequence([
                        seed, int(path.stem[1:]), 2, rep,
                        0 if method in ("GW_torchlowrank",
                                        "K-MeansNM_torchlowrank") else 1,
                    ]))
                    start_round = time.monotonic()
                    if method == "GW_torchlowrank":
                        labels = round_fj(vectors, 2, rng)
                        iterations = 1
                    else:
                        labels, iterations = round_spherical_kmeans(
                            vectors, 2, rng,
                            antipodal=(method == "K-MeansNM_torchlowrank"),
                            edges=edges if method == "K-MeansNM_torchlowrank"
                            else None,
                            keep_best=(method == "K-MeansNM_torchlowrank"),
                        )
                    round_seconds = time.monotonic() - start_round
                    score = cut_weight(edges, labels)
                    best = max(best, score)
                    if archive is not None:
                        method_labels.append(np.asarray(labels).copy())
                        method_scores.append(score)
                    row = {
                        "graph": path.stem,
                        "n": n,
                        "edges": len(edges[0]),
                        "k": 2,
                        "backend": "torch_lowrank",
                        "optimization_type": "nonconvex_lowrank_primal_heuristic",
                        "is_full_convex_sdp": False,
                        "sdp_upper_bound_certified": False,
                        "device": device,
                        "gpu_name": optimizer_info["gpu_name"],
                        "torch_version": optimizer_info["torch_version"],
                        "rank": rank,
                        "method": method,
                        "restart": rep,
                        "rounding_seed": seed,
                        "cut_weight": score,
                        "rounding_seconds": round_seconds,
                        "rounding_iterations": iterations,
                        "lowrank_vector_objective":
                            optimizer_info["lowrank_vector_objective"],
                        "optimizer_seconds": (
                            0.0 if cache_hit else optimizer_info["optimizer_seconds"]
                        ),
                        "original_optimizer_seconds":
                            optimizer_info["optimizer_seconds"],
                        "optimizer_prepare_seconds": prep_seconds,
                        "factor_cache_hit": cache_hit,
                        "factor_cache_path": (
                            str(cache_path) if use_factor_cache else None
                        ),
                        "artifact_run_id": (
                            archive["manifest"]["run_id"] if archive else None
                        ),
                        "partition_archive": (
                            str(method_archive_path(archive, method))
                            if archive else None
                        ),
                        "partition_index": rep if archive else None,
                    }
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                if archive is not None:
                    save_method(
                        archive, method, method_labels, method_scores, k=2
                    )
                print(
                    f"{path.stem} k=2 {method}: best={best} / "
                    f"{restarts} restarts (GPU={device=='cuda'}, "
                    f"factor cache hit={cache_hit})", flush=True,
                )
            if archive is not None:
                update_manifest(archive, status="complete")
                print(f"Experiment archive: {archive['path']}", flush=True)
