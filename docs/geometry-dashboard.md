# Interactive SDP and EVA geometry explorer — optional visualization

This read-only visualization uses existing EVA vectors, graph snapshots, saved
rounding partitions, scalar solver histories, and optional locally cached convex
SDP Gram matrices. It does **not** execute EVA, SCS, or any continuous optimizer.
The code is packaged separately as `maxkcut_viz`, outside `src/maxkcut/` so
Experiment025's frozen optimizer source fingerprint remains unchanged.

## Install and run (Windows PowerShell, from a safe main checkout)

```powershell
conda activate maxkcut
python -m pip install -e ".[viz]"
python -m pytest tests/test_geometry_viz.py tests/test_geometry_dashboard.py -q
powershell.exe -NoProfile -File .\scripts\launch_geometry_dashboard.ps1
```

Alternatively:

```powershell
python -m streamlit run src/maxkcut_viz/geometry_dashboard.py
```

For optional UMAP projections:

```powershell
python -m pip install -e ".[viz,viz-umap]"
```

Streamlit usually prints a local URL such as `http://localhost:8501`.
On a shared university machine, keep access private or use authenticated
tunneling; do not expose archival research files publicly.

**Never switch branches inside a directory running Experiment025.**
The dashboard must read available files from its own checkout or from
a user-supplied, verified existing source path.

## Panels

1. **3D geometry**: G-set graph, PCA/t-SNE/UMAP, sampling seed, marker
   size, cut side, k-means cluster, positive-edge communities, degree,
   signed/absolute weighted degree, or vertex index. Select any
   available archived rounding method and restart; all colors use the
   stored vertex assignments, not reconstructed guesses.
2. **Clustering replay**: deterministic replay of the *rounding only*
   from the saved fixed vectors, reproducing the seeded K-Means2N or
   antipodal K-MeansNM initialization, assignment/centroid iterations
   and per-step signed-weight cut, including best cut so far. Play/pause
   controls and a slider show the actual clustering trajectory.
3. **EVA progress**: scalar objective and maximum-angle vs sweep from
   the existing `logs/Gxx-sweeps.jsonl`. These logs **do not contain**
   intermediate vector coordinates: they cannot show the authentic
   evolving high-dimensional sphere without saved intermediate factors.
4. **Compare G-sets**: up to six independently projected 3D clouds
   with consistent viewing style, vertex counts, factor rank and
   explained variance; **not** one jointly aligned embedding.
5. **Diagnostics**: pairwise cosine histogram from sampled vertex
   pairs (no dense n×n Gram), factor eigenvalue spectrum via small
   `Y.T @ Y`, and distributions for *all* archived cut methods.

The default graph is G22 if available. At the time of implementation,
archived G22 has a complete factor and 100 saved rounding restarts per
original method. Availability is discovered dynamically; the UI does
not manufacture missing graph cases or partitions.

## Numerical interpretation

### EVA unit sphere and PCA

EVA saves one unit-norm vector per graph vertex,

\[
Y\in\mathbb{R}^{n\times p},\qquad \|y_i\|_2=1,\qquad X_{\mathrm{EVA}}=YY^\top.
\]

The dashboard projects rows with *centered* PCA and reports the fraction
of variance in three coordinates. Because projection discards the
remaining \(p-3\) dimensions and PCA centers the data, the displayed
cloud is **not necessarily a unit sphere** even though rows of \(Y\) are.

For the full convex SDP cache, the app reads the stored numerical Gram
\(X_{\mathrm{SCS}}\) and displays the leading 3 **uncentered** eigenpair
coordinates \(U_3\sqrt{\Lambda_3}\). This uses a partial symmetric
eigensolver and avoids the second full-rank spectral factorization,
but is not identical to centered PCA. Currently disabled for
\(n>7500\) to prevent a very large full matrix read. Cache files
live under ignored `reports/gset/sdp_cache/` and must be available
*on the same machine*; they are not downloaded automatically from GitHub.

The SDP tab overlays **EVA cut labels** when coloring by a partition;
the dashboard explicitly marks this cross-solver combination rather
than mislabeling it as SCS's own rounding.

### Distortion, convergence and multiple graphs

t-SNE and UMAP are local-neighborhood exploratory projections, not
isometries of the hypersphere. Distances, intercluster gaps, visual
volumes, and global orientation should not be interpreted literally.
The sample is fixed by a recorded random seed.

Each G-set embedding has a different \(n,p\), and a different arbitrary
orthogonal factor orientation; independently projected diagrams are
**not on a shared axis**. Compare robust descriptors such as pairwise
cosine distributions, spectral concentrations and rounded cut scores.

"Graph community" is explicitly a **positive-edge-only** Louvain
partition of the graph. Negative edges are excluded as affinities,
so it is a heuristic topology summary, not a signed-modularity optimum
or a ground-truth partition; the option is bounded to \(n\le5000\).

### Authentic algorithm animation: saved data boundary

The existing factor cache is only the *final* \(Y\) (or current
checkpoint). Scalar logs can animate the convergence plot but cannot
recover the exact intermediate \(\{Y^{(t)}\}\). There is no mathematical
inverse of the map from full factor trajectories to one endpoint and
scalar objectives. We therefore do **not** fabricate a moving-vector
animation or rerun EVA behind the user's back.

The clustering replay *does* rerun k-means assignments using the
original fixed vector factor and recorded seed; it does **not** solve
or change the continuous relaxation and is the appropriate existing-data
animation. Adding optional immutable intermediate EVA factor snapshots
to a **future solver run** is a separate solver/provenance change.

## Reproducibility and safety

- Source and exact graph SHA256 are validated before drawing EVA data.
- Factor identity and sweep number are verified against the corresponding
  case record; archives are independently rescored with signed weights.
- Extended partitions load only when graph **and factor** SHA256 match.
- Visualization exports are user-requested CSV, never written over
  experiment002 results. Heavy nonlinear projection is capped to 3500
  points; the dashboard never forms a full n-by-n pairwise cosine matrix.
- No heavy numerical work is started when importing this module.
- Project files and results live in the repo, not in chat or transient VM.

## Remaining validation

Run the committed `tests/test_geometry_viz.py` on the real Windows
environment. The original Streamlit visual rendering and full G-set
interaction require a manual browser smoke test; the authoring
environment cannot access the Windows VM or its ignored Gram caches.

## Next research steps

Once explicitly requested, a separate prospective experiment can
capture `Y(t)` snapshots during future EVA runs to enable fixed-basis
3D trajectory animations (with optional orthogonal Procrustes
alignment), ensuring hashes and time indices are recorded.


## EVA's one-vertex update inspector and future snapshots

The sixth dashboard tab evaluates \(h_i = \sum_j w_{ij}y_j\) at the
**existing saved factor**, showing the angular difference between
\(y_i\) and \(-h_i/\|h_i\|\), and the potential continuous objective gain

\[
\Delta_i = \frac12(\|h_i\| + y_i^\top h_i).
\]

These measurements indicate near-coordinate-stationarity of the saved
factor. They are not the historical update path and are **not**
discrete Max-Cut flip improvements.

The third tab can also visualize *actual* intermediate EVA factors
if these are archived prospectively under
`results/gset/experiment002-eva/trajectory/Gxx/sweep-00001.npz`
with `vectors` (n-by-p), `sweep` (integer), and `graph_sha256`
(string) arrays. Their identity, shape, finite entries and row norms
are verified. All frames are placed on the **same final-factor PCA
basis**, so they can be compared without repeatedly refitting PCA.
Existing experiment002 runs did not record these snapshots, and this
visualization branch does not generate missing snapshots by solving EVA.


## Safe separate worktree while Experiment025 is active

The running numerical worktree must remain on `experiment025-results`.
Use a second checkout of `main` for visualization:

```powershell
# Run from the current repository root, without changing its branch.
git fetch origin main
git worktree add --detach ..\maxkcut-geometry origin/main
Set-Location ..\maxkcut-geometry
conda activate maxkcut
python -m pip install -e ".[viz]"
python -m streamlit run src/maxkcut_viz/geometry_dashboard.py
```

This second worktree contains only already committed factors in the fetched
`main` tree, not automatically the newest `experiment025-results` outcomes.
Select an existing verified source directory in the dashboard; never copy or
overwrite the active output tree. Locally ignored convex SDP caches are also
not shared by default. The sidebar's *Existing convex SDP cache* field can
point read-only to the original directory.

## Scientific and repository boundaries

The dashboard's 3D view, clustering replay, EVA sweep history, multi-graph
gallery, diagnostics, and coordinate-update inspector are exploratory.
Neither PCA nor t-SNE nor UMAP proves an optimal cut, and scalar histories
cannot reconstruct intermediate vectors absent saved snapshots.
All signed cuts and factor identities are verified against archived inputs.
Adding this separate optional package does not authorize new EVA solves.
