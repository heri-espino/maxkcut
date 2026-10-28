"""Central and cut-optimal edge choices for an already-computed exact MST.

A tree has distinct possible centers:
  - balanced: most even component sizes (maximum edge betweenness);
  - diameter_center: edge containing the weighted-tree diameter midpoint;
  - bestcut: edge whose removal maximizes the ORIGINAL signed Max-Cut.

No PCA, no distance recalculation, no EVA optimization, no random seeds.
For n vertices, m original graph edges, candidate cut scores for ALL
n-1 tree edges cost O((n + m) log n) using binary LCA and tree-path sums.
"""
from __future__ import annotations

import numpy as np


RULES = ("balanced", "diameter_center", "bestcut")
METHOD_PREFIX = {
    "balanced": "MST-Balanced",
    "diameter_center": "MST-DiameterCenter",
    "bestcut": "MST-BestCut",
}


def _tree_structure(parent, lengths):
    parent = np.asarray(parent, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.float64)
    n = len(parent)
    if (n < 2 or parent.ndim != 1 or lengths.shape != parent.shape
            or parent[0] != -1 or not np.isfinite(lengths).all()
            or np.any(lengths < 0) or np.any(parent[1:] < 0)
            or np.any(parent[1:] >= n)):
        raise ValueError("Invalid rooted tree or edge lengths")
    children = [[] for _ in range(n)]
    for vertex in range(1, n):
        if parent[vertex] == vertex:
            raise ValueError("Self-parenting vertex in tree")
        children[int(parent[vertex])].append(vertex)
    order = [0]
    depth = np.zeros(n, dtype=np.int32)
    for vertex in order:
        for child in children[vertex]:
            depth[child] = depth[vertex] + 1
            order.append(child)
    if len(order) != n:
        raise ValueError("Parent edges contain a cycle or disconnected nodes")
    subtree = np.ones(n, dtype=np.int64)
    for vertex in reversed(order[1:]):
        subtree[parent[vertex]] += subtree[vertex]
    return parent, lengths, children, order, depth, subtree


def labels_cutting_edge(parent, removed_child):
    parent = np.asarray(parent, dtype=np.int64)
    n = len(parent)
    child = int(removed_child)
    if not 1 <= child < n or parent[0] != -1:
        raise ValueError("Removed MST edge must have a non-root child")
    children = [[] for _ in range(n)]
    for vertex in range(1, n):
        children[int(parent[vertex])].append(vertex)
    labels = np.zeros(n, dtype=np.int8)
    stack = [child]
    while stack:
        v = stack.pop()
        labels[v] = 1
        stack.extend(children[v])
    counts = np.bincount(labels, minlength=2)
    if min(counts) < 1:
        raise RuntimeError("An MST edge must split the tree in two")
    return labels


def balanced_edge(parent, lengths):
    """Maximum |S|(n-|S|), equivalent to minimizing split imbalance."""
    parent, lengths, children, order, depth, subtree = _tree_structure(
        parent, lengths,
    )
    n = len(parent)
    balance = subtree[1:] * (n - subtree[1:])
    child = int(np.argmax(balance) + 1)
    return child, {
        "selection_rule": "balanced",
        "edge_betweenness_pairs": int(balance[child - 1]),
        "imbalance_vertices": int(abs(n - 2 * subtree[child])),
        "center_definition": "maximize the number of vertex pairs split",
        "selected_subtree_size": int(subtree[child]),
    }


def diameter_center_edge(parent, lengths):
    """Tree edge containing midpoint of diameter under MST edge lengths.

    A tree's metric center is a point, not necessarily a vertex or edge.
    When the midpoint is exactly a vertex, choose an incident diameter
    edge deterministically (the edge arriving at that vertex).
    Degenerate zero-weight diameter: use balanced-edge fallback.
    """
    parent, lengths, children, order, depth, subtree = _tree_structure(
        parent, lengths,
    )
    n = len(parent)
    neighbors = [[] for _ in range(n)]
    for v in range(1, n):
        p = int(parent[v])
        w = float(lengths[v])
        neighbors[v].append((p, w))
        neighbors[p].append((v, w))

    def traverse(origin):
        distances = np.full(n, -1.0, dtype=np.float64)
        ancestors = np.full(n, -1, dtype=np.int64)
        distances[origin] = 0.0
        stack = [origin]
        while stack:
            vertex = stack.pop()
            for nxt, w in neighbors[vertex]:
                if nxt == ancestors[vertex]:
                    continue
                ancestors[nxt] = vertex
                distances[nxt] = distances[vertex] + w
                stack.append(nxt)
        endpoint = int(np.argmax(distances))
        return endpoint, distances, ancestors

    left, _, _ = traverse(0)
    right, distances, predecessors = traverse(left)
    diameter = float(distances[right])
    if diameter == 0:
        child, details = balanced_edge(parent, lengths)
        details.update({
            "selection_rule": "diameter_center",
            "tree_diameter_length": 0.0,
            "degenerate_zero_diameter": True,
            "center_definition": "zero-length diameter; balanced-edge fallback",
        })
        return child, details

    path = [right]
    while path[-1] != left:
        path.append(int(predecessors[path[-1]]))
    path.reverse()
    halfway = diameter / 2.0
    accumulated = 0.0
    selected_child = None
    center_is_vertex = False
    for u, v in zip(path[:-1], path[1:]):
        child = v if parent[v] == u else u
        w = float(lengths[child])
        if accumulated + w >= halfway:
            selected_child = child
            center_is_vertex = bool(
                np.isclose(accumulated + w, halfway, atol=1e-12, rtol=0)
            )
            break
        accumulated += w
    if selected_child is None:
        raise RuntimeError("Could not locate weighted tree diameter midpoint")
    return int(selected_child), {
        "selection_rule": "diameter_center",
        "center_definition": "weighted diameter midpoint edge",
        "diameter_endpoints": [int(left), int(right)],
        "tree_diameter_length": diameter,
        "midpoint_at_vertex": center_is_vertex,
        "degenerate_zero_diameter": False,
    }


def all_tree_edge_cut_scores(parent, lengths, edges):
    """Score all single-edge tree splits using LCA path sums.

    The original weighted edge (u,v,w) crosses the split induced by
    removing a tree edge e iff e lies on the unique tree path u--v.
    Hence cut_score[e] = sum_{(u,v) original} w_uv * 1[e in path(u,v)].
    Add w at u and v, subtract 2w at LCA(u,v), and propagate the sums
    upward to score all n-1 tree edges efficiently. Signed w is valid.
    """
    parent, lengths, children, order, depth, subtree = _tree_structure(
        parent, lengths,
    )
    n = len(parent)
    levels = n.bit_length()
    up = np.empty((levels, n), dtype=np.int64)
    up[0] = parent
    up[0, 0] = 0
    for level in range(1, levels):
        up[level] = up[level - 1, up[level - 1]]
    delta = np.zeros(n, dtype=np.float64)
    ii, jj, weights = edges
    if not (len(ii) == len(jj) == len(weights)):
        raise ValueError("Original graph edge arrays have different lengths")
    for u, v, w in zip(ii, jj, weights):
        u, v, w = int(u), int(v), float(w)
        if not (0 <= u < n and 0 <= v < n and np.isfinite(w)):
            raise ValueError("Nonfinite weight or vertex outside saved MST")
        if u == v:
            continue
        delta[u] += w
        delta[v] += w
        a, b = u, v
        if depth[a] < depth[b]:
            a, b = b, a
        step = int(depth[a] - depth[b])
        bit = 0
        while step:
            if step & 1:
                a = int(up[bit, a])
            step >>= 1
            bit += 1
        if a != b:
            for level in range(levels - 1, -1, -1):
                if up[level, a] != up[level, b]:
                    a = int(up[level, a])
                    b = int(up[level, b])
            a = int(up[0, a])
        lca = a
        delta[lca] -= 2 * w
    scores = np.full(n, np.nan, dtype=np.float64)
    for vertex in reversed(order[1:]):
        scores[vertex] = delta[vertex]
        delta[parent[vertex]] += delta[vertex]
    return scores


def best_cut_edge(parent, lengths, edges):
    scores = all_tree_edge_cut_scores(parent, lengths, edges)
    child = int(np.argmax(scores[1:]) + 1)
    return child, {
        "selection_rule": "bestcut",
        "center_definition": "not a geometric center; maximize actual cut",
        "best_tree_edge_cut_weight": float(scores[child]),
        "evaluated_tree_edges": int(len(scores) - 1),
    }
