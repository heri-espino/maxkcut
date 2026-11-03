# Experiment025 completed-results seminar — Marp 001–025 edition

**Slide source:** `presentations/maxkcut_full_research_001_025.marp.md`, 94 slides after removing a blank slide from the historical deck. **Original 001–023 Marp and PDF remain intact** as archived October 29 evidence.

**Scientific status:** full 71 × 18 × 16 = 20,448 numerical refinements recorded completed on the university VM. The 1,278-row aggregate CSV is the only numerical Experiment025 source in this presentation. The full VM-local raw JSONL/NPZ archive has not been independently rescored or verified backed up off the VM. This is a reproducible **descriptive scientific presentation**, not a certified method superiority / equal-compute / new-record claim.

## Inputs and provenance

| Input | Purpose |
| --- | --- |
| `results/gset/experiment025-gw16-bestcut-quality/analysis/full_best_final_by_graph.csv` | Exactly 1,278 graph-method rows, 16/16 status complete; best-of-16 and mean-of-16 cut scores |
| `results/gset/experiment025-gw16-bestcut-quality/analysis/experiment025_method_ranking.csv` | Aggregate best-of-16 graph bootstrap intervals from [Experiment025 report](../notes/experiments/experiment025-aggregate-analysis.md) |
| `results/gset/experiment022-graph-structure-methods/graph_features.csv` | Original-edge structural features (n, m, average degree, degree CV, transitivity, triangles, negative-edge fraction, etc.) |
| `presentations/build_experiment025_figures.mjs` | Deterministic Node 22 SVG figure renderer, no Python scientific package / optimizer |
| `presentations/experiment025-figure-manifest.json` | Machine-readable figure inventory and input SHA |
| `presentations/assets/experiment025_*.svg` | **12 committed, source-reproducible vector charts**, included by relative path in the deck |
| `presentations/verify_experiment025_presentation.mjs` | Checks slide boundaries, broken image paths, complete chart references and source manifest |

**Source SHA256:** `8a9d84cfc35f1d424519b862676facc288a6366bc1820e52975a2bf76678de6c` for the full aggregate CSV. The figure generator checks that hash and verifies all 71 graphs, 18 methods, 1,278 cells, 16 completed starts per cell and a consistent join to 71 original-edge feature records. Any changes to inputs require an explicit, reviewed update; it refuses to silently build from a different source.

## Figure design and interpretation

The twelve plots are: all-18 method ranking; graph-bootstrap intervals; per-family method heatmap; positive-vs-signed heatmap; 71-graph fixed-method-vs-oracle gain; family fractional winner credit; degree association (log scale); transitivity association; negative-edge fraction association; 71×10 graph-method heatmap; mean-of-16 vs best-of-16 comparison; and eight **candidate** archived feasible-reference improvements.

Normalization is graph-specific: `100 × (best_cut[g,m] − best_cut[g,GW]) / best_cut[g,GW]`, then **equal weight per graph**. This prevents the highest-cut instances dominating due to scale. Winner credit for ties is split equally among winning methods. The graphwise 18-way oracle is selected *after* observing outcomes and therefore not an operational out-of-sample method. Bootstrap intervals concern resampled graphs, **not** 20,448 independent GW seeds. Structure associations do not establish causation or held-out predictive power. Generator-family labels are metadata, **not** planarity proofs, and family may be confounded with degree and signedness.

The 001–023 historical slides are kept to preserve provenance; old Experiment023 statements are explicitly labeled as a historical status snapshot. The 001–025 edition opens with the new completed computation and closes with its descriptive findings and outstanding scientific audit.

## Regenerate all plots without numerical optimization

From the **repository root** with Node 22 and Python 3 installed:

```powershell
python -m pip install numpy matplotlib
python presentations/build_assets.py
python presentations/build_full_research_figures.py
python presentations/make_extra_graphics.py
node presentations/build_experiment025_figures.mjs
node presentations/build_experiment025_figures.mjs --check
node presentations/verify_experiment025_presentation.mjs
```

All old and new SVGs remain in `presentations/assets/`. For a quick reproducibility check without Python plotting dependencies, run **only** the two Node scripts; all 12 new Experiment025 figures are committed and require no regeneration.

## Render complete PDF, PowerPoint and HTML

```powershell
npm install --no-save --no-package-lock @marp-team/marp-cli@4.5.1
New-Item -ItemType Directory -Force .\presentations\build | Out-Null
npx --no-install marp .\presentations\maxkcut_full_research_001_025.marp.md --allow-local-files --pdf -o .\presentations\build\maxkcut_full_research_001_025.pdf
npx --no-install marp .\presentations\maxkcut_full_research_001_025.marp.md --allow-local-files --pptx -o .\presentations\build\maxkcut_full_research_001_025.pptx
npx --no-install marp .\presentations\maxkcut_full_research_001_025.marp.md --allow-local-files --html -o .\presentations\build\maxkcut_full_research_001_025.html
```

Marp CLI requires an installed Chrome, Edge or Firefox for PDF/PPTX conversion. The browser auto-discovery should work on a normal interactive Windows VM, but **you do not need the university VM** for these exports. The GitHub Actions workflow uses `ubuntu-latest`, **not** the self-hosted PCAG103A-13 runner.

## Automated GitHub Actions

Workflow: `.github/workflows/experiment025-marp-presentation.yml`. It runs on pushes to the presentation branch (and main after integration), relevant PRs, and manual `workflow_dispatch` after the workflow exists on the default branch.

Pipeline: check out source → regenerate existing historical SVGs and all 12 Experiment025 vectors → verify committed new SVGs are byte-identical → check every Marp image reference and slide count → install Marp CLI v4.5.1 → render **PDF + PPTX + HTML** → validate outputs and slide count → upload artifact named `maxkcut-experiment025-structure-marp`.

The generated PDF/PPTX/HTML are **temporary workflow artifacts (30-day retention)**, not permanent repo binaries. All figure **source**, all 12 generated SVGs, original data inputs, presentation Markdown, renderer source, manifests and reproducibility instructions are committed. For durable final binaries, publish the validated artifacts to an authorized permanent repository release/archive after review. Never pass the original full experiment archive through this workflow.

## Acceptance / future agents

- Full deck renders with **94** slides, 12 new figures, no broken relative links or empty slides.
- Workflow successfully produces a nonempty PDF, PPTX and HTML and downloads as GitHub artifact.
- Graph family / weight type are separately stratified; original graph structure plots use Experiment022 original-edge features.
- No EVA, SDP, new GW roundings, Flip/KL/ILS refinements, or Windows VM actions occur during figure or slide builds.
- Independent saved-partition rescore and off-VM raw backup remain pending. The eight archived feasible reference improvements are *candidates*, not independently verified public records.
- All new research results are saved under this presentation branch / analysis PR until the broader Experiment025 audit authorizes final scientific claims.
