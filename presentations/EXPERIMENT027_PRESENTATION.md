## New algorithm-name appendix — 9 + 18 + 3 methods (9 November 2026)

User requested an **explanation of each confusing method name inside the existing Marp**, rather than a separate slide deck. The current `presentations/maxkcut_full_research_001_027.marp.md` has **137 slides** (97 historical + 20 Experiment027 + **20 new appendix slides**, A0–A19). The appendix is at the **end** and differentiates fixed-embedding rounding from direct graph partition refinement. It explains the literal names, encoded hyperparameters, construction mechanisms, and applicability to a proposed raw-only geometric experiment. It covers **all 9 Experiment021 initializers** (crosschecked against `eva_kmeans_initializers.py::INIT2`), **all 18 Experiment025 methods** (crosschecked against `experiment023_methods.py::METHODS`), **all 3 Experiment027 methods** (crosschecked against `scripts/experiment027_angular.py::METHODS`), and five archived raw EVA variants. It also defines GW, RR, FJ, PCA, MST, L1/L2/L∞, EM, CEM, Gibbs, KL, ILS, Tabu and explains GW-anchored baseline retention and the exact-vs-global difference for threshold/semicircle searches.

**Implementation grounding:** primary source files inspected were `src/maxkcut/eva_rounding_cli.py`, `eva_kmeans_initializers.py`, `experiment023_methods.py`, `experiment024_paired_gw16.py`, `eva_cutaware_kmeans.py`, `eva_direct_rounding.py`, `eva_multigroup_kmeans.py`, `eva_soft_em_bernoulli.py`, `eva_mst_center.py`, `eva_mst_multicut.py`, `eva_elite_search.py`, and `scripts/experiment027_angular.py`. No code or numerical result was altered. The verifier checks the exact code catalogs to prevent a source method from silently disappearing from the appendix.

**Build:** the PDF is now **137 pages**, and the HTML is self-contained. The hosted `experiment027-marp-presentation.yml` build checks the new page count and continues to generate **PDF and HTML only—NO PowerPoint**. The historical 97-slide Experiment025 and current source/results remain intact.

**Current export policy (9 November 2026):** The user explicitly requested **no PowerPoint**. All future Experiments001–027 hosted builds export **PDF and portable HTML only**; the editable Marp and committed SVGs remain. Historical job #42570516934 produced a PowerPoint artifact before this preference was stated; that earlier ZIP is historical only, not the active output format. The current GitHub workflow excludes PPTX creation, validation and upload.

## Validated hosted build (2026-11-09 UTC / 2026-11-08 Mexico)

[GitHub Actions run #42570516934](https://github.com/heri-espino/maxkcut/actions/runs/42570516934) **succeeded** on the hosted `ubuntu-latest` runner. Its checks independently regenerated and compared all 13 new committed E027 SVGs (and the historical E025 figures), verified the 71-graph/6,816-pair source data, validated the exact 117-slide merged Markdown, and exported PDF, PPTX and embedded-SVG standalone HTML (historical result before the no-PPTX change). The historical run verified **117 PDF pages** and **117 PPTX slides**; subsequent workflows omit PPTX as requested. Its [`maxkcut-experiment027-complete-marp` build artifact](https://github.com/heri-espino/maxkcut/actions/runs/42570516934) is a 40,072,286-byte ZIP containing all three formats; the 30-day artifact retention is not permanent storage. The **editable Markdown and 13 reproducible SVGs are permanently committed** to this research branch.

**Line-ending provenance:** Git checkout may normalize Windows CSV CRLF sequences to LF. The E027 source validators accept only raw SHA256 equality **or** exact LF→CRLF round-trip equality against the immutable Release index checksum, and still fail closed on altered numerical data. No floating-point output, method definition, or source CSV row was modified by this compatibility fix.

# Full research seminar — Experiment001–027 merged Marp edition

**Source:** [`maxkcut_full_research_001_027.marp.md`](maxkcut_full_research_001_027.marp.md), **137 slides** (including 20 algorithm-name appendix slides). The original [001–025 97-slide edition](maxkcut_full_research_001_025.marp.md) remains unmodified and available for historical comparison.

**Publication date:** 2026-11-08. **Scientific scope:** original 71 G-set signed Max-Cut graphs, saved EVA Gram factors, Experiment025's 18-method descriptive census, and new Experiment027 angular-guided rounding. Do **not** interpret numerical cut improvements as certified global optima or as approximation guarantees for Max-`k`-Cut with `k>2`.

## What is integrated

The new section sits **after the original Experiment025 graphs, before the final synthesis** and contains 20 new slides across two chapters:

- **Chapter 12 — Experiment027 design:** no new SDP solves, exactly 16 existing GW normal vectors per graph/method, margin/gain/pair ranking rules, exact single-arc sweeps (not global optimality), common refinement, factor and original signed-score identity.
- **Chapter 13 — Experiment027 results:** comparison to refined GW and Hyperplane Sweep, the official graph-bootstrap 95% CIs for GW gains, 71-by-3 quality matrix, raw-to-refined effect, matched-seed distribution, stage effects, recorded runtime, generator families and sign type, best-of-three ex post portfolio with explicit budget caveat, graph-level extremes, next controlled experiments and immutable publication pointers.
- **Revised cover, completed-study summary, conclusions and references** reflect the November 8 results while keeping E025's older provisional statements scoped to **the still-unpublished 18-method full raw archive**. E027's audited five-method subset is not confused with an audit of all 18 E025 methods.

## Grounded source data and figures

**Immutable numerical data:** [Experiment027 GitHub Release, 71 ZIPs](https://github.com/heri-espino/maxkcut/releases/tag/experiment027-data-71g-2b313a624780291eb4dd). On this branch: [latest-release pointer](../published/experiment027/latest_release.json), [content-addressed index](../published/experiment027/releases/experiment027-data-71g-2b313a624780291eb4dd/index.json), and five audited CSV/JSON products in `published/experiment027/releases/experiment027-data-71g-2b313a624780291eb4dd/analysis/`.

- `audited_graph_comparisons.csv`: **213** new method×graph best-of-16 rows.
- `audited_method_leaderboard.csv`: official E027 gains vs refined GW with graph-level bootstrap CIs; comparisons with historical HP.
- `audited_paired_seed_deltas.csv`: **6,816** matched 71×3×16×2 seed-pair deltas against historical GW and HP.
- `audited_seed_scores.csv`: **5,680** rescored method×seed rows (3 new and 2 old controls), every Round/Flip/KL/Final stage.
- `results/gset/experiment022-graph-structure-methods/graph_features.csv`: **71** recorded original-edge family and sign features.

**Reproducible vector graphics**: `presentations/build_experiment027_figures.mjs` (Node 22, standard library only) generates **13 SVGs** (`presentations/assets/experiment027_*.svg`), with source SHA256 pinned in `experiment027-figure-manifest.json`. The existing E025 historical figures and source remain intact. The generator neither accesses the network nor performs SDP/EVA/refinement/optimization.

| Figure filename | Meaning |
|---|---|
| `experiment027_protocol.svg` | Saved GW normal → candidate ranking → exact arc search → common refinement |
| `experiment027_gain_vs_gw.svg` | Mean relative best-of-16 gain over refined GW and official bootstrap CIs |
| `experiment027_vs_hyperplane.svg` | Single-method graph wins/ties/losses against HP |
| `experiment027_delta_hp_ci.svg` | Exploratory graph-bootstrap 95% intervals for mean signed cut difference versus HP |
| `experiment027_graph_matrix.svg` | All 71×3 graph-level differences vs HP |
| `experiment027_raw_refined.svg` | Mean matched raw and best-refined seed deltas vs HP |
| `experiment027_refinement_stages.svg` | Raw/Flip/KL/Final matched-stage differences |
| `experiment027_paired_cdf.svg` | Empirical distributions of correlated matched seed differences |
| `experiment027_quality_time.svg` | Archived mean per-seed seconds and best-of-16 gain (descriptive only) |
| `experiment027_family.svg` | Within-generator-family descriptive gains vs HP |
| `experiment027_signedness.svg` | Positive vs signed graph strata (confounded; descriptive) |
| `experiment027_portfolio.svg` | Three-method ex post portfolio: 42 W / 13 T / 16 L against HP |
| `experiment027_extremes.svg` | Largest observed portfolio improvements and deficits |

**Key values** copied from immutable auditor: graph-weighted mean best-of-16 improvements over refined GW are Angular Gain **+0.316704%**, Margin **+0.302682%**, Pair **+0.281426%**. No new single method unambiguously dominates historical HP. Mean cut advantage over HP for Gain/Margin/Pair is **−0.3099 / −1.0000 / −1.0845**; the ex post best-of-three has **42 W / 13 T / 16 L** and **+2.7606** cut units, but spends three methods' computational budgets. Mean seed wall times are descriptive and not a matched-hardware causal speed benchmark. The paired-seed CDF contains correlated within-graph starts; graph-cluster intervals should be used for inference.

**Figure provenance rule:** the new generator SHA256-verifies all four published CSV sources against the immutable release index and fails if rows, graph IDs, archived starts, or figures drift. `--check` compares generated SVG bytes and manifest to the committed outputs. The CI's `git diff` check is independent and fails on any nondeterministic SVG regeneration.

## Offline reproduction and validation

From repository root (Node.js 22):

```powershell
node presentations/build_experiment027_figures.mjs
node presentations/build_experiment027_figures.mjs --check
node presentations/verify_experiment027_presentation.mjs
```

The verifier requires **137 nonblank slides**, all 13 new SVGs, all historical figures, source SHA checks, the published release inventory and the previous MathJax formula count.

**Full figure refresh** for historical + E025 + E027:

```powershell
python -m pip install "numpy>=1.26,<3" "matplotlib>=3.8,<4"
python presentations/build_assets.py
python presentations/build_full_research_figures.py
python presentations/make_extra_graphics.py
node presentations/build_experiment025_figures.mjs
node presentations/build_experiment025_reference_ratio_figures.mjs
node presentations/build_experiment027_figures.mjs
node presentations/verify_experiment025_presentation.mjs
node presentations/verify_experiment027_presentation.mjs
```

**Export manually** (Chrome/Chromium or Edge installed; Marp CLI version pinned by CI):

```powershell
npx @marp-team/marp-cli@4.5.1 presentations/maxkcut_full_research_001_027.marp.md --allow-local-files --pdf -o presentations/build/maxkcut_full_research_001_027.pdf
npx @marp-team/marp-cli@4.5.1 presentations/maxkcut_full_research_001_027.marp.md --allow-local-files --html -o presentations/build/maxkcut_full_research_001_027.html
node presentations/embed_experiment027_html_assets.mjs
```

The Marp 001–027 PDF/standalone HTML are automatically exported in the dedicated [hosted Actions workflow](../.github/workflows/experiment027-marp-presentation.yml), which confirms 137 PDF pages and publishes an Actions artifact **`maxkcut-experiment027-complete-marp`**. Export runs only on `ubuntu-latest`, never the university Windows VM. The editable source and SVGs remain permanently committed; the CI export artifact has **30-day retention**, not permanent storage.

## Handoff restrictions

Keep the original older editions and files; do not overwrite the Experiment025 97-slide source or the frozen numerical optimizer or its identity/fingerprints. Do **not** rerun EVA, SDP, Experiment025, or Experiment027 to make presentation figures. Edit the Figure generator and audit provenance as a pair. Changes to scientific claims must be traced to the primary CSV/JSON and corresponding Release SHA. The branch/PR remains unmerged unless the user expressly authorizes merging.
