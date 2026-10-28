"""Experiment013: objective-scored, reproducible EVA rounding/refinement grid.

All pipelines reuse the SAME independently verified original EVA factor
and signed G-set graph. A source restart is the original experiment008
K-MeansNM-Flip restart index; each method gets the same paired index.
Families: saved KMeansNM-Flip, raw KMeansNM, raw KMeans2N, GW/FJ,
GW hyperplane + exact signed threshold sweep, and cut-aware KMeansNM.

Independent refiners: optional positive-gain flips, generalized fast KL
with multiple independently seeded attempts, optional iterated KL basin
hopping. The final returned partition never scores below the selected
geometric stage for THAT pipeline (refiners are monotone best-of-so-far).
It need not dominate another pipeline's saved 1-flip baseline.

Grid families are specified in JSON as explicit base values and small
Cartesian-product axes, expanded deterministically with an exact
fingerprint per configuration.
"""
from __future__ import annotations

from itertools import product
import hashlib
import json
import time

import numpy as np

from .eva_cutaware_kmeans import improve_kmeans_direction, optimal_threshold
from .eva_elite_search import optimize_elite, optimize_kl_fast
from .eva_rounding_cli import _round_one
from .io import edge_arrays
from .local_search import improve_by_flips
from .objective import cut_weight


ROUNDERS = (
    "saved-kmeans-flip", "kmeansnm", "kmeans2n", "gw",
    "cutaware-kmeans", "gw-threshold",
)
STAGES = ("Round", "Flip", "KL", "Final")
DEFAULTS = {
    "direction_trials": 0,
    "kmeans_iters": 100,
    "flip_sweeps": 0,
    "kl_passes": 0,
    "kl_repeats": 1,
    "ils_cycles": 0,
    "ils_kl_passes": 8,
    "kick_sizes": [8, 16, 32, 64],
    "perturb_scales": [0.03, 0.08, 0.16, 0.35],
    "temperature": 12.0,
}


def _canonical(config):
    return json.dumps(config, sort_keys=True, allow_nan=False, separators=(",", ":"))


def expand_grid(spec):
    """Expand validated families into unique, fully specified configurations."""
    if not isinstance(spec, dict) or spec.get("version") != 1:
        raise ValueError("Grid JSON requires version=1")
    families = spec.get("families")
    if not isinstance(families, list) or not families:
        raise ValueError("Grid requires a nonempty families list")
    configs = []
    identities = set()
    for family in families:
        name, rounder = family["name"], family["rounder"]
        if (not isinstance(name, str) or not name
                or rounder not in ROUNDERS):
            raise ValueError("Unrecognized grid family/rounder")
        base = family.get("base", {})
        axes = family.get("axes", {})
        if not isinstance(base, dict) or not isinstance(axes, dict):
            raise ValueError("Grid base and axes must be dictionaries")
        if set(base).intersection(axes):
            raise ValueError("Each parameter must occur in base OR axes")
        if any(k not in DEFAULTS for k in (*base, *axes)):
            raise ValueError("Unknown grid hyperparameter")
        if not all(isinstance(v, list) and v for v in axes.values()):
            raise ValueError("Grid axes must have nonempty value lists")
        keys = sorted(axes)
        values = [axes[k] for k in keys]
        for combination in product(*values):
            cfg = {
                **DEFAULTS,
                **base,
                **dict(zip(keys, combination)),
                "name": name,
                "rounder": rounder,
            }
            for key in (
                "direction_trials", "kmeans_iters", "flip_sweeps",
                "kl_passes", "kl_repeats", "ils_cycles", "ils_kl_passes",
            ):
                if not isinstance(cfg[key], int) or isinstance(cfg[key], bool):
                    raise ValueError(f"{key} must be an integer")
            if (cfg["direction_trials"] < 0
                    or cfg["kmeans_iters"] < 1
                    or cfg["flip_sweeps"] < 0
                    or cfg["kl_passes"] < 0
                    or cfg["kl_repeats"] < 1
                    or cfg["ils_cycles"] < 0
                    or cfg["ils_kl_passes"] < 1):
                raise ValueError("Invalid grid iteration/restart settings")
            if cfg["kl_passes"] == 0 and cfg["kl_repeats"] != 1:
                raise ValueError("kl_repeats>1 requires positive kl_passes")
            if cfg["direction_trials"] and rounder != "cutaware-kmeans":
                raise ValueError("direction_trials only apply to cutaware-kmeans")
            if (rounder == "saved-kmeans-flip" and cfg["direction_trials"]):
                raise ValueError("Saved flip starts cannot modify directions")
            for seq in ("kick_sizes", "perturb_scales"):
                if not isinstance(cfg[seq], list) or not cfg[seq]:
                    raise ValueError(f"{seq} must be a nonempty list")
            if any(int(k) != k or k < 1 for k in cfg["kick_sizes"]):
                raise ValueError("Kick sizes must be positive integers")
            if any(not np.isfinite(x) or x <= 0 for x in cfg["perturb_scales"]):
                raise ValueError("Perturb scales must be positive finite numbers")
            if not np.isfinite(cfg["temperature"]) or cfg["temperature"] < 0:
                raise ValueError("Temperature must be finite nonnegative")
            config_id = hashlib.sha256(
                _canonical(cfg).encode("utf8")
            ).hexdigest()[:16]
            if config_id in identities:
                raise ValueError("Duplicate grid configuration")
            identities.add(config_id)
            cfg["id"] = config_id
            configs.append(cfg)
    if len(configs) > 500:
        raise ValueError("Grid has >500 configurations; reduce axes first")
    return configs


def evaluate_pipeline(
    graph, vectors, *, saved_labels, saved_score,
    restart, graph_number, rounding_seed, search_seed, config,
    target=None,
):
    """Return each stage's binary partitions and original signed cut scores.

    The saved-KMeansNMFlip source partition is verified separately and
    included as a paired comparison, even for other rounders. Reusing
    the original seed convention is deliberate: every experiment's
    restart 37 means the same seeded KMeans/GW baseline as experiment008.
    """
    edges = edge_arrays(graph)
    actual_saved = float(cut_weight(edges, saved_labels))
    if not np.isclose(actual_saved, saved_score, atol=1e-8, rtol=0):
        raise ValueError("Saved source label score mismatch")
    cfg = config
    method = cfg["rounder"]
    rng = np.random.default_rng(np.random.SeedSequence(
        [rounding_seed, graph_number, 2, restart,
         1 if method == "kmeans2n" else 0],
    ))
    started = time.monotonic()
    details = {"rounder": method}
    if method == "saved-kmeans-flip":
        raw = np.asarray(saved_labels, dtype=np.int8).copy()
    elif method in ("kmeansnm", "kmeans2n", "gw"):
        method_name = {
            "kmeansnm": "K-MeansNM_EVA",
            "kmeans2n": "K-Means2N_EVA",
            "gw": "FJ_EVA",
        }[method]
        raw, stats = _round_one(
            vectors, graph, edges, method_name,
            graph_number, restart, rounding_seed,
        )
        details.update(stats)
    elif method == "gw-threshold":
        direction = rng.standard_normal(vectors.shape[1])
        raw, report = optimal_threshold(graph, vectors @ direction)
        details["threshold_score"] = float(report["cut"])
    elif method == "cutaware-kmeans":
        raw, report = improve_kmeans_direction(
            vectors, graph, rng,
            direction_trials=cfg["direction_trials"],
            kmeans_iters=cfg["kmeans_iters"],
            perturb_scales=tuple(cfg["perturb_scales"]),
            target=target,
        )
        details.update({
            "raw_kmeans_best": float(report["raw_kmeans_best_cut"]),
            "thresholded_directions": report["direction_candidates"],
            "direction_best_source": report["best_source"],
        })
    else:
        raise ValueError("Unknown rounding method")
    raw = np.asarray(raw, dtype=np.int8)
    score_round = float(cut_weight(edges, raw))
    stages = {"Round": raw.copy()}
    scores = {"Round": score_round}
    timings = {"round_seconds": time.monotonic() - started}

    if cfg["flip_sweeps"] > 0:
        flip_rng = np.random.default_rng(np.random.SeedSequence(
            [search_seed, graph_number, restart, 1313],
        ))
        flipped, flip_diag = improve_by_flips(
            graph, raw.copy(), flip_rng,
            max_sweeps=cfg["flip_sweeps"],
        )
        details["flip_termination"] = flip_diag["termination"]
        details["flip_count"] = int(flip_diag["flips"])
        timings["flip_seconds"] = float(flip_diag["seconds"])
    else:
        flipped = raw.copy()
        timings["flip_seconds"] = 0.
    score_flip = float(cut_weight(edges, flipped))
    if score_flip + 1e-8 < score_round:
        raise AssertionError("Flip stage worsened original graph objective")
    stages["Flip"] = flipped.copy()
    scores["Flip"] = score_flip

    kl_start = time.monotonic()
    if cfg["kl_passes"] > 0:
        best_kl_score = score_flip
        kl_result = flipped.copy()
        repeated_scores = []
        for repeat in range(cfg["kl_repeats"]):
            kl_rng = np.random.default_rng(np.random.SeedSequence(
                [search_seed, graph_number, restart, 101, repeat],
            ))
            result, diag = optimize_kl_fast(
                graph, flipped.copy(), kl_rng, passes=cfg["kl_passes"],
            )
            score = float(cut_weight(edges, result))
            repeated_scores.append(score)
            if score > best_kl_score:
                best_kl_score = score
                kl_result = result.copy()
        details["kl_repeat_scores"] = repeated_scores
    else:
        kl_result = flipped.copy()
    timings["kl_seconds"] = time.monotonic() - kl_start
    score_kl = float(cut_weight(edges, kl_result))
    if score_kl + 1e-8 < score_flip:
        raise AssertionError("KL stage worsened original graph objective")
    stages["KL"] = np.asarray(kl_result, dtype=np.int8).copy()
    scores["KL"] = score_kl

    if cfg["ils_cycles"] > 0:
        ils_rng = np.random.default_rng(np.random.SeedSequence(
            [search_seed, graph_number, restart, 1401],
        ))
        final, diag = optimize_elite(
            graph, kl_result, [(kl_result, score_kl)], ils_rng,
            method="ILS-KL",
            cycles=cfg["ils_cycles"],
            kl_passes=cfg["ils_kl_passes"],
            kick_sizes=tuple(min(graph.shape[0], int(k))
                             for k in cfg["kick_sizes"]),
            temperature=cfg["temperature"],
            target=target,
        )
        timings["ils_seconds"] = float(diag["seconds"])
        details["ils_cycles_completed"] = diag["cycles_completed"]
        details["ils_best_events"] = diag["best_events"]
    else:
        final = kl_result.copy()
        timings["ils_seconds"] = 0.
    score_final = float(cut_weight(edges, final))
    if score_final + 1e-8 < score_kl:
        raise AssertionError("ILS stage worsened original graph objective")
    stages["Final"] = np.asarray(final, dtype=np.int8).copy()
    scores["Final"] = score_final
    details["source_saved_cut"] = float(saved_score)
    timings["total_seconds"] = time.monotonic() - started
    return stages, scores, details, timings
