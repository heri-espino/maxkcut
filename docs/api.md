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

## CLI

```{automodule} maxkcut.cli
:members:
```

**Important:** `solve_sdp(...)` returns a numerical *primal solution* and solver diagnostics; downstream analyses must check feasibility and distinguish primal objective estimates from valid upper bounds.
