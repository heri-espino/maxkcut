"""Small exact-reference example: SDP relaxation of the unit-weight triangle.

Run only by explicit invocation:
    python examples/triangle_sdp.py
"""
import numpy as np
from scipy.sparse import csr_matrix
from maxkcut import cut_weight, edge_arrays, factorize, round_fj, solve_sdp


def main():
    adjacency = csr_matrix(
        [[0, 1, 1],
         [1, 0, 1],
         [1, 1, 0]]
    )
    gram, diagnostics = solve_sdp(
        adjacency, k=2, solver="SCS", eps=1e-5, max_iters=5000
    )
    vectors, _ = factorize(gram)
    edges = edge_arrays(adjacency)
    rng = np.random.default_rng(2026)
    partition = round_fj(vectors, k=2, rng=rng)
    print("Triangle GW SDP primal objective:", diagnostics["sdp_objective_primal"])
    print("One rounded cut value:", cut_weight(edges, partition))
    print("Exact combinatorial optimum:", 2)
    print("Solver status:", diagnostics["solver_status"])
    # The SDP relaxation value for K3 is 9/4; it is not the integer optimum.


if __name__ == "__main__":
    main()
