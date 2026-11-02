# Experiment025 — Maximize the best cut encountered, including intermediate iterations

## Research objective

Only original signed Max-Cut quality matters for method rankings. Runtime and
additional proposal counts are archived for audit, not used as ranking criteria.

For a graph with original signed weights w and a binary cut x, define
C(x) = sum over original edges (i,j) of w_ij times 1[x_i != x_j].

For method m and one of the 16 COMMON Goemans–Williamson Gaussian starting
directions s, let P(m,s) include every feasible hard partition encountered
during its initializer, all Flip sweeps, all generalized KL passes, and every
ILS outer cycle (including the best incumbent). The result is

    BEST(m,s) = max { C(x) : x is in P(m,s) }
    BEST(m)   = max { BEST(m,s) : s = 0,...,15 }.

The final iteration of an optimizer can be worse than its previous best.
Therefore BEST(m), not the last walking state or the raw initial cut, chooses
the winning graph/method/seed. The known best feasible cut from the original
repository is a separate descriptive benchmark (not necessarily an optimum).

## Complete per-iteration and per-record preservation

Each graph / method / GW seed keeps independently reproducible archives:

- progress/0000/events.jsonl: append-only original-cut score and method
  diagnostics for every meaningful iteration (e.g., K-Means iteration,
  cut-aware proposal, hyperplane search iteration, multigroup binary merge,
  soft EM iteration, CEM generation, Gibbs sweep, Flip sweep, KL pass, and
  outer ILS cycle). Reported current candidate scores are distinguished
  from the stored current best partition when the ILS walk can deteriorate.
- progress/0000/best.json: best-so-far CUT on the original signed graph,
  event of discovery, immutable binary partition filename, SHA256 checksum.
  On each strict improvement, preserve one compressed versioned partition
  (best_000001.npz, best_000002.npz, ...). Non-improving iteration scores
  and statistics are preserved in the event log, without unnecessary
  full-size partition copies.
- method/0000.npz and method/0000.json: four independently rescored stage
  partitions (Round, Flip, KL, Final). Here the legacy Final key means
  the BEST cut ever seen in the run, not the last walking state.
  Records contain best_seen_cut, best_seen_event, last_iteration_cut and
  the number of iteration and improvement events. A recorded best can
  come from an intermediate initializer state, not only a local-search stage.
- method/status.json and statuses/method.json: completed seed count,
  best-ever cut across all 16 seeds, selected seed index, mean best-seen,
  last ILS candidate score, graph and factor SHA and best-known-cut gap.

A progress record is flushed to disk after EVERY meaningful iteration.
Improved partitions are saved atomically as soon as discovered; a crashed
process can be retried without losing its previous best. Intermediate
scores are not reconstructed from the final result: they are logged when
the algorithm actually produces them.

Deterministic MST methods receive the 16 GW starting partitions as retained
incumbents and are explicitly GW-anchored HYBRIDS; their tree itself is not
randomly restarted.

## Quality-first computational protocol

- Same 71 original G-set graphs, 18 methods, and 16 matched GW unit
  directions per graph as the prior Experiment024 protocol.
- ALL SIXTEEN initial candidates for EACH method are separately refined,
  rather than refining only the strongest raw candidate.
- Full protocol: up to 20 Flip sweeps, 30 KL passes, and 80 ILS outer cycles
  (up to 8 inner KL passes each). Pilot uses shorter settings.
- No method-level wall-clock timeout or target-based early stop. GitHub
  Actions still imposes a finite platform job duration; it is an operational
  constraint, not an optimization performance criterion.
- Checkpoints are pushed back to the experiment branch after batches of
  4 completed seeds, along with all their iteration logs, best partitions,
  signed cuts and diagnostics. Unfinished methods are reported as such.
- 71 x 18 = 1,278 graph-method cells and 20,448 seed-level full
  refinements are planned. Only 16-of-16 complete, independently verified
  graph-method cells are eligible for the primary comparison.

## Reproduction

From a full checkout of the existing repository and Conda environment:

    python -m pytest tests/test_experiment025_bestcut.py -q
    python -m maxkcut.experiment025_bestcut single --graph G11 --method gw-rr --phase pilot --max-new-seeds 4
    python -m maxkcut.experiment025_bestcut single --graph G11 --method gw-rr --phase pilot
    python -m maxkcut.experiment025_bestcut summary --phase full

Files:
- src/maxkcut/experiment025_bestcut.py
- src/maxkcut/experiment025_journal.py
- tests/test_experiment025_bestcut.py
- scripts/experiment025-windows.ps1
- .github/workflows/experiment025-quality-tests.yml
- .github/workflows/experiment025-quality-first.yml

The callbacks added to existing optimizers are optional: their behavior
remains unchanged when no observer is supplied. Completed Experiment021,
Experiment022, Experiment023 and Experiment024 data are not modified.

## Research restrictions and future analysis

This study answers which algorithm produces the largest VERIFIED feasible
original-graph signed cut under the fixed 16-GW-start protocol. It does
not assert an approximation guarantee or equal computational effort. All
winners, per-family fractional ties, mean best-seen differences relative
to GW-RR, and gaps to published best-known feasible cuts must be computed
from verified full data, not from queued jobs or truncated intermediate
results. When result uploads arrive after implementation PR merge, open
and merge a separate results PR. Never force-push or replace source factors.
