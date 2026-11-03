# Branch cleanup and scientific provenance — 2026-11-03

This repository historically kept a branch for every experiment, CI repair, or
research note. The 2026-11-03 inventory found 70 remote branches and 68 pull
requests, of which 62 distinct *merged* PR heads remained as branches. Both
primary branches were also present:

- `main`: integrated code, documentation and audited methods
- `experiment025-results`: actively accumulating full-phase 71-graph,
  18-method, 16-seed numerical records, iteration journals and SHA-pinned
  original signed Max-Cut partitions

No action should reset, force-push, or delete either primary branch. A published
PR merge is not, by itself, proof that a branch's **current** tip is already
reachable from the primary history; a branch may have advanced afterward.

## Audited branch deletion

[`branch-cleanup-inventory.json`](branch-cleanup-inventory.json) is the
immutable starting inventory. It records the SHA and merged PR numbers for
62 candidate heads and six research branches deliberately excluded.

[`scripts/maintenance/prune_merged_branches.py`](../../scripts/maintenance/prune_merged_branches.py)
is fail-closed and dry-run by default:

```powershell
# From a safe, independent worktree, not the actively writing Experiment025 job.
python scripts/maintenance/prune_merged_branches.py
# Use only when all proposed deletion checks pass:
python scripts/maintenance/prune_merged_branches.py --execute
```

Before deleting a branch, the script verifies that its current remote SHA
exactly matches the recorded SHA. The hosted non-mutating preflight determined
that **31/62 candidate tips were reachable** from `main` or
`experiment025-results`. For the other **31 merged-but-unreachable tips**,
the script first publishes `refs/tags/archive/merged/<original-branch-name>`
at the **exact original SHA**, confirms the remote tag receipt, and only
then permits deletion. These tags preserve unique commits even if a PR was
squash-merged or its feature head later advanced. All tips are checked again
immediately before deletion; no archive tag is overwritten.
It refuses an unexpected origin, missing primary branches or missing unmerged
research branches. It never uses force-push, rebase, or history rewriting.

The one-time GitHub Actions workflow `.github/workflows/merged-branch-prune.yml`
has a non-mutating PR audit and a deletion step invoked **only when the
reviewed inventory first lands on main** (or deliberately dispatched with the
explicit delete input). It uses a hosted runner and a GitHub-generated short-
lived token, never the university's active Windows worktree. The archived
Git tags are not new experiments or validated cuts; they preserve history. Its JSON audit
is uploaded as a GitHub Actions artifact. If a branch advances, it is skipped
rather than deleted.

## Unmerged work kept as branches

These six heads contain distinct experiments, data, code or documentation not
proven to be reachable from `main`. They are **not candidates** for automatic
deletion and should be separately reviewed, integrated or archived to permanent
commit-specific Git tags before their references are removed.

| Historical branch | Source | Recommendation |
| --- | --- | --- |
| `experiment001-g77-gpu-fp32-single-cut` | Closed PR #31 | Keep as archive; separate GPU methodology |
| `feature/interactive-sdp-eva-geometry` | Closed PR #34 | Read-only dashboard integrated separately as `maxkcut_viz` optional extra, with Experiment025 SHA isolation; retain historical branch until integration and tests are verified |
| `docs/illustrated-maxcut-sdp-eva-tutorial-20261028` | Closed PR #35 | Keep full TeX, compiled PDF, figures and build provenance |
| `docs/gset-71-literature-references` | Closed PR #43 | Keep original literature dataset until reconciled with newer references |
| `experiment014-gw-expected-refine` | Closed PR #44 | Preserve all G63 factor and partition archives; different objective |
| `experiment007-eva-mst-bestcut-flips` | No PR | Four commits outside current `main`; audit and archive separately |

The closed PR conversations retain decisions and linked source history.
Git LFS backing objects must be verified before retiring a branch that
contains LFS pointers. A source commit SHA alone does not prove that all
binary LFS objects are currently downloadable.

## Visualization integration rationale

The old Streamlit exploratory dashboard is safe and useful on `main`, but it
must remain **optional**. The maintained module is `src/maxkcut_viz/` (not
`src/maxkcut/`). Install with `pip install -e ".[viz]"`; optional nonlinear
UMAP projection with `.[viz,viz-umap]`. See
[`docs/geometry-dashboard.md`](../../docs/geometry-dashboard.md).

Experiment025's `algorithm_source_sha256` fingerprints the Python source
files directly under `src/maxkcut/`; adding a new module there, even if it
only visualizes data, would change the numerical protocol identity. Keeping
the dashboard in a separate sibling package avoids that change. It reads
existing archives; it neither solves EVA again nor modifies saved cuts.

## Next operations

Review the deletion Action artifact and the remaining branch listing. Never
assume 62 deletions occurred merely because the workflow launched; report the
actual number of successful ref deletions and skip reasons.

Do not delete the remaining six research branch refs just for a tidy branch
menu. Instead make a separate, reviewed archival work item, with LFS
verification and immutable tags, once their outstanding content is
scientifically reconciled.
