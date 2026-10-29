"""Interpretability diagnostics for the optimal original-graph MST edge."""
from __future__ import annotations

import numpy as np
import pytest

from maxkcut.eva_mst_center import (
    balanced_edge, best_cut_edge, labels_cutting_edge,
)
from maxkcut.eva_mst_interpret import characterize_best_edge
from maxkcut.objective import cut_weight


def test_complete_uniform_graph_best_tree_edge_is_most_balanced():
    # Tree of seven nodes. All pairs are original graph edges of weight 1.
    parent = np.array([-1, 0, 1, 1, 3, 3, 4], dtype=int)
    lengths = np.array([0., 2., 3., 1., 2., 1., 5.])
    n = len(parent)
    ii, jj = np.triu_indices(n, k=1)
    weights = np.ones(len(ii), dtype=float)
    edges = ii, jj, weights
    diag, profile = characterize_best_edge(parent, lengths, edges)
    child, stats = best_cut_edge(parent, lengths, edges)
    balanced_child, _ = balanced_edge(parent, lengths)
    assert child == balanced_child == diag["best_edge_child"] == 3
    assert diag["best_edge_is_most_balanced"]
    assert not diag["best_edge_is_longest"]
    assert diag["best_edge_cut_score"] == 12.0
    assert diag["best_edge_balance_pairs"] == 12
    assert diag["mean_signed_weight_per_potential_cross_pair"] == 1.
    np.testing.assert_allclose(profile["cut_weight"], profile["balance_pairs"])
    assert diag["spearman_cut_vs_balance"] == pytest.approx(1.0)
    assert diag["positive_cross_weight"] == 12.
    assert diag["negative_cross_weight"] == 0.
    assert diag["best_edge_length_percentile"] < 1.


def test_signed_weight_scores_and_split_decomposition():
    parent = np.array([-1, 0, 1, 2, 3], dtype=int)
    lengths = np.array([0., 1., 6., 1., 2.])
    edges = (
        np.array([0, 0, 1, 1, 2, 3], dtype=int),
        np.array([2, 4, 2, 4, 4, 4], dtype=int),
        np.array([10., 1., -4., 1., -1., 2.]),
    )
    diag, profile = characterize_best_edge(parent, lengths, edges)
    child = diag["best_edge_child"]
    labels = labels_cutting_edge(parent, child)
    independent = float(cut_weight(edges, labels))
    assert independent == pytest.approx(diag["best_edge_cut_score"])
    assert diag["positive_cross_weight"] + diag["negative_cross_weight"] == pytest.approx(independent)
    assert diag["best_edge_balance_pairs"] * diag["mean_signed_weight_per_potential_cross_pair"] == pytest.approx(independent)
    assert len(profile["cut_weight"]) == 4
    assert 0 <= diag["best_edge_length_percentile"] <= 1
    assert 0 <= diag["best_edge_balance_percentile"] <= 1
    assert diag["positive_cross_edge_count"] >= 0
    assert diag["negative_cross_edge_count"] >= 0
    assert diag["existing_cross_edge_count"] >= 1


def test_all_ties_have_midrank_percentile_not_100_percent():
    parent = np.array([-1, 0, 1, 2], dtype=int)
    lengths = np.zeros(4, dtype=float)
    edges = (np.array([], dtype=int), np.array([], dtype=int),
             np.array([], dtype=float))
    diag, profile = characterize_best_edge(parent, lengths, edges)
    assert diag["best_edge_length_percentile"] == 0.5
    assert diag["spearman_cut_vs_length"] is None
    assert diag["spearman_cut_vs_balance"] is None
    assert diag["positive_cross_weight"] == 0
    assert diag["negative_cross_weight"] == 0
    assert np.all(profile["mean_cross_pair_weight"] == 0)
