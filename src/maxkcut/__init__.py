"""Max-k-Cut semidefinite programming and structured rounding research tools.

Importing the package never starts an SDP optimization.
"""
from .io import load_graph, edge_arrays
from .objective import cut_weight
from .rounding import factorize, round_fj, round_spherical_kmeans
from .sdp import solve_sdp
from .experiments import benchmark

__version__ = "0.1.0"
__all__ = ["load_graph", "edge_arrays", "cut_weight", "factorize",
           "round_fj", "round_spherical_kmeans", "solve_sdp", "benchmark"]
