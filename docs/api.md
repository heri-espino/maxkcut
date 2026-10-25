# Python API

The package exposes stable **research entry points** at the top level. Each function can also be imported from its specialized module.

## Graph I/O

```{automodule} maxkcut.io
:members:
:undoc-members:
:show-inheritance:
```

## Exact objective

```{automodule} maxkcut.objective
:members:
```

## SDP solver

```{automodule} maxkcut.sdp
:members:
```

## Rounding

```{automodule} maxkcut.rounding
:members:
```

## Experiment orchestration

```{automodule} maxkcut.experiments
:members:
```

## Canonical G-set data and reference comparisons

```{automodule} maxkcut.references
:members:
```

```{automodule} maxkcut.comparison
:members:
```

```{automodule} maxkcut.graph_compare
:members:
```

## Resumable sparse G-set sweep

```{automodule} maxkcut.sweep
:members:
```

## Optional CUDA low-rank Max-Cut (k=2 only)

```{automodule} maxkcut.lowrank
:members:
```

```{automodule} maxkcut.lowrank_cache
:members:
```

```{automodule} maxkcut.lowrank_experiments
:members:
```

```{automodule} maxkcut.lowrank_cli
:members:
```

## Full convex SDP versus CUDA comparisons

```{automodule} maxkcut.study_compare
:members:
```

```{automodule} maxkcut.study_cli
:members:
```

## Resource-bounded full convex SDP G-set batch

```{automodule} maxkcut.sdp_batch
:members:
```

```{automodule} maxkcut.sdp_batch_cli
:members:
```

## CLI

```{automodule} maxkcut.cli
:members:
```

**Important:** `solve_sdp(...)` returns a numerical *primal solution* and solver diagnostics; downstream analyses must check feasibility and distinguish primal objective estimates from valid upper bounds.
