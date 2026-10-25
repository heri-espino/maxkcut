# maxkcut

**Semidefinite relaxations and geometric rounding for Max-$k$-Cut.**

This documentation describes a Python package for reproducible experiments motivated by Ángel Eduardo Rodríguez Fernández's *On the Study of Fitness Landscapes and the Max-Cut Problem* (2021), particularly Chapters 6–7. The implementation supports the Goemans–Williamson/Frieze–Jerrum SDP, randomized rounding, and spherical $k$-means variants.

**Research status:** a numerical experiment toolkit, not an established new approximation guarantee. An approximate primal SDP objective is not automatically a certified optimum or upper bound.

```{toctree}
:maxdepth: 2
:caption: Get started

installation
quickstart
windows-powershell
experiments
experiment-archives
full-gset-sweep
canonical-data-comparison
gset-reference-values
performance
gpu-cpu-performance
cuda-lowrank
full-sdp-comparison
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
research
architecture
agent-guide
contributing
```

For detailed source-grounded derivations and benchmarks, see the repository's [research background](https://github.com/heri-espino/maxkcut/tree/main/notes/background) and [G-set reproduction notes](https://github.com/heri-espino/maxkcut/tree/main/notes/experiments).

The existing `G-Set/` files are a Max-Cut corpus and should not be confused with the A/B/C/D random-graph experiments in Chapter 7.
