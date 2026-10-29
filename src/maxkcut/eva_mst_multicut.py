"""Multi-edge MST forest warm starts for the *binary* signed-weight Max-Cut.

An existing full-ambient EVA MST is split by deleting q longest tree edges.
This makes q+1 forest components (NOT q+1 Max-Cut classes). Contract the
original signed graph onto those components, optimize a TWO-way cut on
that contracted graph, and then locally improve individual vertex labels.

With a fixed tree and seed, component partitions for increasing q can
inherit the prior q solution. The best coarse objective is therefore
nondecreasing in q for the requested nested schedule. A separate
post-refinement step is also nondecreasing relative to its own coarse
start but its results need not be monotone across q.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .eva_mst_center import _tree_structure
from .io import edge_arrays
from .local_search import improve_by_flips
from .objective import cut_weight


def longest_edge_forest(parent, lengths, q):
    """Return component ids and removed tree child indices for top-q lengths.

    Distance ties are resolved by child vertex index ascending. Component
    identifiers are deterministic BFS order; component count is exactly
    q+1 even for zero-length edges.
    """
    parent, lengths, children, order, depth, subtree = _tree_structure(
        parent, lengths,
    )
    n = len(parent)
    if not isinstance(q, (int, np.integer)) or not 1 <= q < n:
        raise ValueError(f"Expected 1 <= q < {n}, got {q!r}")
    # Stable sort (longest first, then child index) without n-by-n arrays.
    removed = np.lexsort((np.arange(1, n), -lengths[1:]))[:q] + 1
    removed_mask = np.zeros(n, dtype=np.bool_)
    removed_mask[removed] = True
    components = np.zeros(n, dtype=np.int32)
    next_id = 1
    for vertex in order[1:]:
        if removed_mask[vertex]:
            components[vertex] = next_id
            next_id += 1
        else:
            components[vertex] = components[parent[vertex]]
    if next_id != q + 1 or len(np.unique(components)) != q + 1:
        raise AssertionError("Deleting q MST edges must yield q+1 components")
    return components, removed.astype(np.int32)


def contracted_adjacency(graph, components):
    """Aggregate signed original weights between different forest components.

    For ANY binary component labeling c, original graph Max-Cut equals
    the contracted graph Max-Cut at labels c[components], exactly.
    """
    components = np.asarray(components, dtype=np.int32)
    n = graph.shape[0]
    if (components.shape != (n,) or np.any(components < 0)
            or not np.array_equal(np.unique(components),
                                  np.arange(components.max() + 1))):
        raise ValueError("Expected consecutive component ids for all vertices")
    k = int(components.max() + 1)
    u, v, w = edge_arrays(graph)
    a, b = components[u], components[v]
    between = a != b
    aa, bb, ww = a[between], b[between], w[between]
    coo = sparse.coo_matrix(
        (np.concatenate((ww, ww)),
         (np.concatenate((aa, bb)), np.concatenate((bb, aa)))),
        shape=(k, k), dtype=np.float64,
    )
    result = coo.tocsr()
    result.sum_duplicates()
    result.eliminate_zeros()
    return result


def _component_labels_from_vertices(components, labels):
    """Project ONLY partitions constant on each forest component."""
    labels = np.asarray(labels, dtype=np.int8)
    k = int(components.max() + 1)
    if labels.shape != components.shape:
        raise ValueError("Labels do not match vertex count")
    values = np.full(k, -1, dtype=np.int8)
    for comp, label in zip(components, labels):
        if values[comp] == -1:
            values[comp] = label
        elif values[comp] != label:
            raise ValueError("Previous partition splits a new forest component")
    if np.any(values == -1):
        raise AssertionError("Missing contracted component")
    return values


def multi_edge_candidates(
    graph, parent, lengths, q_values, *,
    coarse_restarts=20, coarse_sweeps=20, flip_sweeps=20,
    seed=2026, graph_number=0, metric_index=2,
):
    """Yield one q-level experiment at a time; include all coarse trials.

    The first trial at each q inherits the previous best component
    partition, guaranteeing that the *best coarse* cut does not decline
    as q increases. Other trials use independent random component
    assignments and signed-weight component-flip optimization. After
    selecting the best contracted cut, a SINGLE unrestricted signed
    vertex-flip search refines the result.

    Returned objects include lists of every coarse binary vertex
    partition for independent archival. This function is pure apart
    from its time-based diagnostics: no EVA/MST recomputation or I/O.
    """
    parent, lengths, _, _, _, _ = _tree_structure(parent, lengths)
    q_values = tuple(int(q) for q in q_values)
    if (not q_values or list(q_values) != sorted(set(q_values))
            or q_values[0] != 1 or q_values[-1] >= len(parent)):
        raise ValueError("q_values must be unique increasing values beginning at 1")
    if coarse_restarts < 0 or coarse_sweeps < 1 or flip_sweeps < 1 or seed < 0:
        raise ValueError("Invalid restart, sweep or seed settings")

    original_edges = edge_arrays(graph)
    previous_best = None
    previous_score = float("-inf")
    for q in q_values:
        components, removed = longest_edge_forest(parent, lengths, q)
        coarse_graph = contracted_adjacency(graph, components)
        if previous_best is None:
            # The first deletion (longest edge) always gives two components.
            # Preserve the original deterministic longest-edge baseline.
            initial_component_labels = _component_labels_from_vertices(
                components, (components != components[0]).astype(np.int8),
            )
        else:
            initial_component_labels = _component_labels_from_vertices(
                components, previous_best,
            )
        # q=1 has a single seeded baseline; q>1 adds randomized coarse starts.
        trial_count = (1 + coarse_restarts) if q > 1 else 1
        trials = []
        all_coarse_partitions = []
        all_coarse_scores = []
        best_coarse = None
        best_coarse_score = float("-inf")
        for trial in range(trial_count):
            trial_seed = np.random.SeedSequence([
                seed, graph_number, metric_index, q, trial, 91,
            ])
            rng = np.random.default_rng(trial_seed)
            if trial == 0:
                start = initial_component_labels.copy()
                source = "inherited_longest_edge" if q == 1 else "nested_previous_best"
            else:
                start = rng.integers(
                    0, 2, size=q + 1, dtype=np.int8,
                )
                source = "random_components"
            initial_full = start[components]
            initial_score = float(cut_weight(original_edges, initial_full))
            if q == 1:
                optimized_components = start
                diag = {
                    "flips": 0, "sweeps": 0, "termination": "single_edge_baseline",
                    "seconds": 0.0,
                }
            else:
                optimized_components, diag = improve_by_flips(
                    coarse_graph, start, rng, max_sweeps=coarse_sweeps,
                )
            vertex_labels = np.asarray(
                optimized_components[components], dtype=np.int8,
            )
            score = float(cut_weight(original_edges, vertex_labels))
            contracted_score = float(cut_weight(
                edge_arrays(coarse_graph), optimized_components,
            ))
            if not np.isclose(score, contracted_score, rtol=0, atol=1e-7):
                raise AssertionError("Component contracted-graph score mismatch")
            if score + 1e-8 < initial_score:
                raise AssertionError("Component optimization worsened initial cut")
            trials.append({
                "trial_index": trial,
                "source": source,
                "initial_cut": initial_score,
                "coarse_cut": score,
                "component_flips": int(diag["flips"]),
                "component_sweeps": int(diag["sweeps"]),
                "component_termination": diag["termination"],
                "component_seconds": float(diag["seconds"]),
            })
            all_coarse_partitions.append(vertex_labels)
            all_coarse_scores.append(score)
            if score > best_coarse_score:
                best_coarse_score = score
                best_coarse = vertex_labels.copy()
        if best_coarse_score + 1e-8 < previous_score:
            raise AssertionError("Nested MST multi-edge coarse scores decreased")
        previous_best = best_coarse.copy()
        previous_score = best_coarse_score

        refined_rng = np.random.default_rng(np.random.SeedSequence(
            [seed, graph_number, metric_index, q, 555]
        ))
        final_labels, final_diag = improve_by_flips(
            graph, best_coarse.copy(), refined_rng, max_sweeps=flip_sweeps,
        )
        final_score = float(cut_weight(original_edges, final_labels))
        if final_score + 1e-8 < best_coarse_score:
            raise AssertionError("Vertex refinement worsened MST multi-edge cut")
        yield {
            "q": q,
            "forest_component_count": q + 1,
            "removed_edge_children": removed.tolist(),
            "removed_edge_lengths": lengths[removed].tolist(),
            "component_sizes": np.bincount(components).tolist(),
            "coarse_restarts": trial_count - 1,
            "coarse_trials": trials,
            "coarse_labels": all_coarse_partitions,
            "coarse_scores": all_coarse_scores,
            "best_coarse_trial_index": int(np.argmax(all_coarse_scores)),
            "best_coarse_cut": best_coarse_score,
            "best_coarse_labels": best_coarse,
            "final_labels": np.asarray(final_labels, dtype=np.int8),
            "final_cut": final_score,
            "vertex_improvement": final_score - best_coarse_score,
            "vertex_flips": int(final_diag["flips"]),
            "vertex_sweeps": int(final_diag["sweeps"]),
            "vertex_termination": final_diag["termination"],
            "vertex_seconds": float(final_diag["seconds"]),
        }
