"""Interpret WHY the best original-graph Max-Cut edge is selected from an EVA MST.

Each full-space MST edge corresponds to a two-component partition.
The score is a signed, original-graph-weighted tree-path load:
C(e) = sum_{(u,v,w) in original G} w * 1[e lies on T-path(u,v)].

Analysis is exact conditional on the saved MST; all-edge score
calculation uses the LCA path-sum algorithm from eva_mst_center.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .eva_mst_center import (
    _tree_structure, all_tree_edge_cut_scores, diameter_center_edge,
    labels_cutting_edge,
)
from .objective import cut_weight


def _rank_fraction(values, chosen_index):
    """Midrank percentile on [0,1] with a meaningful 0.5 for all ties."""
    target = values[chosen_index]
    return float(
        (np.count_nonzero(values < target)
         + 0.5 * np.count_nonzero(values == target)) / len(values)
    )


def _spearman_or_none(left, right):
    """Return None when tied constants make rank correlation undefined."""
    if len(left) < 3 or np.all(left == left[0]) or np.all(right == right[0]):
        return None
    result = float(spearmanr(left, right).statistic)
    return result if np.isfinite(result) else None


def characterize_best_edge(parent, lengths, original_edges):
    """Characterize e* against every other MST edge on the same factor.

    Reports geometric distance rank, structural balance rank,
    correlation of graph score with both and decomposition of the
    selected signed cut. Includes complete arrays for plotting and
    independent verification; these are NOT random restarts.

    The mean cross-pair weight treats missing original graph edges
    as weight 0. This implies the exact identity:
      cut(e) = |S_e|*(n-|S_e|)*mean_cross_pair_weight(e).
    """
    parent, lengths, children, order, depth, subtree = _tree_structure(
        parent, lengths,
    )
    n = len(parent)
    scores = all_tree_edge_cut_scores(parent, lengths, original_edges)
    children_ids = np.arange(1, n, dtype=np.int32)
    cut_scores = np.ascontiguousarray(scores[1:])
    edge_lengths = np.ascontiguousarray(lengths[1:])
    split_sizes = np.ascontiguousarray(subtree[1:])
    balance = split_sizes * (n - split_sizes)
    best_idx = int(np.argmax(cut_scores))
    best_child = best_idx + 1
    best_length = float(edge_lengths[best_idx])
    best_balance = int(balance[best_idx])
    sizes = [int(n - split_sizes[best_idx]),
             int(split_sizes[best_idx])]
    best_labels = labels_cutting_edge(parent, best_child)
    ii, jj, ww = original_edges
    ii = np.asarray(ii, dtype=np.int64)
    jj = np.asarray(jj, dtype=np.int64)
    ww = np.asarray(ww, dtype=float)
    crossing_weights = ww[best_labels[ii] != best_labels[jj]]
    pos = crossing_weights[crossing_weights > 0]
    neg = crossing_weights[crossing_weights < 0]
    direct_score = float(cut_weight(original_edges, best_labels))
    if not np.isclose(direct_score, scores[best_child], rtol=1e-10, atol=1e-7):
        raise AssertionError(
            "MST signed path sum differs from independent original-graph cut"
        )
    diameter_child, _ = diameter_center_edge(parent, lengths)
    positive_weight = float(np.sum(pos))
    negative_weight = float(np.sum(neg))
    best_score = float(scores[best_child])
    potential_cross_pairs = int(balance[best_idx])
    mean_pair_weight = best_score / potential_cross_pairs
    report = {
        "best_edge_child": best_child,
        "best_edge_parent": int(parent[best_child]),
        "best_edge_cut_score": best_score,
        "best_edge_length": best_length,
        "best_edge_balance_pairs": best_balance,
        "component_sizes": sizes,
        "component_imbalance": int(abs(sizes[0] - sizes[1])),
        "best_edge_is_longest": bool(best_length == np.max(edge_lengths)),
        "best_edge_is_most_balanced": bool(best_balance == np.max(balance)),
        "best_edge_is_diameter_center_choice": bool(
            best_child == diameter_child
        ),
        "best_edge_length_percentile": _rank_fraction(
            edge_lengths, best_idx,
        ),
        "best_edge_balance_percentile": _rank_fraction(
            balance, best_idx,
        ),
        "spearman_cut_vs_length": _spearman_or_none(
            cut_scores, edge_lengths,
        ),
        "spearman_cut_vs_balance": _spearman_or_none(
            cut_scores, balance.astype(float),
        ),
        "spearman_length_vs_balance": _spearman_or_none(
            edge_lengths, balance.astype(float),
        ),
        "positive_cross_weight": positive_weight,
        "negative_cross_weight": negative_weight,
        "positive_cross_edge_count": int(len(pos)),
        "negative_cross_edge_count": int(len(neg)),
        "zero_cross_edge_count": int(
            np.count_nonzero(crossing_weights == 0)
        ),
        "existing_cross_edge_count": int(len(crossing_weights)),
        "potential_cross_pair_count": potential_cross_pairs,
        "mean_signed_weight_per_potential_cross_pair": mean_pair_weight,
        "mean_signed_weight_per_existing_cross_edge": (
            best_score / len(crossing_weights)
            if len(crossing_weights) else None
        ),
        "cut_decomposition": "balance_pairs * mean_signed_cross_pair_weight",
        "weighted_betweenness_interpretation":
            "sum of original signed edge weights routed over this MST edge",
        "graph_objective_not_mst_length": True,
        "best_only_among_single_tree_edge_splits": True,
        "total_tree_edge_candidates": int(n - 1),
        "percentile_convention":
            "midrank fraction (count lower + 0.5 * count tied) / count",
    }
    profiles = {
        "edge_child": children_ids,
        "edge_parent": parent[1:].astype(np.int32),
        "tree_edge_length": edge_lengths,
        "cut_weight": cut_scores,
        "subtree_size": split_sizes,
        "balance_pairs": balance,
        "mean_cross_pair_weight": cut_scores / balance,
    }
    return report, profiles
