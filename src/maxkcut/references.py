"""Published reference results for the 17 toroidal G-set benchmark instances.

Glass and Feder, 'Maximum Cut Algorithms and Upper Bounds for Planar and
Toroidal Graphs', arXiv:2606.28478v1 (June 2026), page 8, Table 1.

These numbers are *reported by the paper*, not results computed by our code.
A graph name by itself does not prove local data identity. For an instance
whose paper result and proven upper bound coincide, the reported value is a
proved optimum conditional on matching the canonical graph.
"""
from __future__ import annotations


SOURCE = (
    "Glass and Feder (2026), arXiv:2606.28478v1, Table 1, page 8"
)
SOURCE_URL = "https://arxiv.org/abs/2606.28478"

# G-set number: (vertices, prior-known cut, valid upper bound, paper cut)
TOROIDAL_TABLE_1 = {
    11: (800, 564, 564, 564),
    12: (800, 556, 558, 556),
    13: (800, 582, 583, 582),
    32: (2000, 1410, 1414, 1410),
    33: (2000, 1382, 1384, 1382),
    34: (2000, 1384, 1386, 1384),
    48: (3000, 6000, 6000, 6000),
    49: (3000, 6000, 6000, 6000),
    50: (3000, 5880, 6000, 5880),
    57: (5000, 3494, 3494, 3494),
    62: (7000, 4870, 4872, 4872),
    65: (8000, 5562, 5562, 5562),
    66: (9000, 6364, 6366, 6364),
    67: (10000, 6950, 6952, 6950),
    72: (10000, 7008, 7008, 7008),
    77: (14000, 9940, 9940, 9940),
    81: (20000, 14060, 14060, 14060),
}


def reference_for_graph(name: str):
    """Return the paper's reference with an explicit proof-status label."""
    if not isinstance(name, str) or not name.startswith("G"):
        return None
    try:
        number = int(name[1:])
    except ValueError:
        return None
    if name != f"G{number}":
        return None
    record = TOROIDAL_TABLE_1.get(number)
    if record is None:
        return None
    n, known, upper, paper = record
    if not (known <= paper <= upper):
        raise ValueError("Invalid literature reference row")
    return {
        "graph": name,
        "reference_vertices": n,
        "previous_known_cut": known,
        "published_cut": paper,
        "published_upper_bound": upper,
        "reference_status": (
            "proven_optimum" if paper == upper else "best_known_not_proven"
        ),
        "source": SOURCE,
        "source_url": SOURCE_URL,
    }
