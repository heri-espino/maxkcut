# experiment004 — Interactive 3D geometry of archived SDP/EVA solutions

**Status:** implemented on an isolated visualization branch;
synthetic GitHub CI / actual Windows dashboard smoke test pending.
**Branch:** `feature/interactive-sdp-eva-geometry`.
**Base:** `experiment002-eva-factor-reproduction`.
**No full SDP or EVA optimization is performed.**

## Aim

Study the high-dimensional unit-sphere geometry of archived Max-Cut
vector-relaxation solutions and how clustering converts them into
discrete graph bipartitions. Compare G-sets without assuming identical
coordinate frames. Inspect genuine optimization *scalar* histories
without inventing missing intermediate vector factors.

Primary research inspiration: Rodríguez Fernández's 2021 dissertation,
Chapters 6 and 7; Rodríguez-Fernández et al. (2020), *Computation* 8(3),
75, especially Figure 1 on clustering distortion versus cut quality.
The independent 2026 EVA algorithm updates vectors one at a time.
The 2022 JMLR fixed-point method is a separate approach and is not
misrepresented as EVA.

## Inputs

- Canonical `data/G*.txt`; preserved signed edge weights and graph hashes.
- Existing `results/gset/experiment002-eva/factors/Gxx.npz` and
  `cases/Gxx.json` with complete graph/settings/factor validation.
- Archived per-method partitions in `archives/` and extension
  `rounding-extensions/`; independently verified exact weighted cuts.
- Existing `logs/Gxx-sweeps.jsonl` scalar histories.
- Optional local `reports/gset/sdp_cache/Gxx-k2-*.npz`
  (CVXPY/SCS Gram matrices). Since these are large and ignored by Git,
  no automatic public availability is presumed.

## Procedures

1. Centered PCA of rows of full saved EVA vector factor, report
   retained explained variance. Selected seed controls vertex sampling.
2. Optional t-SNE/UMAP as **non-metric** exploratory views only.
3. Color by the exact archived partition, graph degree/strength, or
   positive-edge-only Louvain communities (ignore negative affinities).
4. Replay k-means rounding on a fixed saved factor; save/plot only in
   memory, do not persist changed optimizer states or manipulate archives.
5. Plot actual EVA objective/angular histories and inspect frozen
   coordinate update residuals. Real factor trajectory views are only
   enabled if the corresponding intermediate snapshots exist.
6. Compare independent small multiples across at most six G-sets;
   spectra/cosine distributions permit more robust structural comparisons.
7. Support optional local cached full convex Gram spectrum using
   top-three uncentered eigenvectors with an explicit size cap.

## Output and safeguards

Pure visualization/CSV exports on demand through Streamlit; no new
numerical benchmark result claimed. No cache rewriting. Methods
`src/maxkcut/geometry_viz.py`, dashboard
`src/maxkcut/geometry_dashboard.py`, tests
`tests/test_geometry_viz.py`, runbook
`docs/geometry-dashboard.md` and PowerShell launcher
`scripts/launch_geometry_dashboard.ps1`.

Known limits: PCA projection does not retain full unit norms; t-SNE/
UMAP distort global geometry; separately fit G-sets are not comparable
on shared axes; no recorded EVA intermediate vectors for existing runs.
Check performance and browser interaction on the Windows VM.
Next step: if future optimization runs are authorized, save periodic
EVA vector snapshots with provenance, without changing historical
experiment002 files.
