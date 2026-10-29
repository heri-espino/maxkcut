# Full Max-Cut research seminar — Marp edition

**Audience:** Professor Ángel E. Rodríguez Fernández. **Date:** 29 October 2026.

This is a **separate 74-slide deck**, derived from the repository’s original `presentations/maxkcut_research_journey.marp.md`. That 44-slide historical deck remains unchanged.

## Contents

1. Signed Max-Cut, SDP reference, nonconvex EVA geometry, GW, spherical K-means.
2. Early experiment chronology: MST, flip, KL, Tabu, cut-aware, hyperplanes, EM and multigroup clustering.
3. Verified Experiments019–021: nine two-centroid initializers, 71 graphs, 31,950 saved searches; best-of-50 and mean-of-50 comparisons.
4. Retrospective Experiment022: random/planar-family/toroidal classification, structural audit, method wins, negative LOOCV selection result.
5. Additional historical 62-graph raw GW/K-means results and G63-only MST/EM comparisons with warnings about incomparable budgets.
6. Experiment023: its 17 registered algorithm protocols, pilot/runner/coverage design, and a clear **NOT VERIFIED COMPLETE** status.
7. Reproducibility, citations, and future validation.

**No Experiment023 numerical outcomes are presented.** It was implemented but a complete final report was not present in the repository at the time the deck was prepared.

## Open in Marp

From the repository root, with Node.js installed:

```powershell
npx @marp-team/marp-cli presentations/maxkcut_full_research_001_023.marp.md --allow-local-files -o maxkcut_full_research.pdf
```

Or open the Markdown in VS Code with the **Marp for VS Code** extension and use Preview/Export. The `assets/` directory resolves relative to the Markdown source.

The pre-rendered PDF companion is `MaxCut_Full_Research_Experiments_001_023_Angel.pdf`. Its reference rendering used MathJax and WeasyPrint; minor typography differences can arise with Marp CLI/Chromium exports.

## Rebuild new SVG diagrams

```powershell
python presentations/build_full_research_figures.py
```

The script uses Python's standard library. The older scientific SVGs remain under `presentations/assets/`; the new diagrams were designed from source-audited Experiment021/022 statistics. **Illustrative graph drawings are not benchmark node coordinates.**

## Evidence boundaries

- Source-of-record numerical results: `results/gset/experiment021-complete-71-gset/statistics/` and `results/gset/experiment022-graph-structure-methods/`.
- Historical raw rounding: archived summaries from Experiments002/003/008; 62 graphs, 20 starts per method.
- Historical G63: heterogeneous archived Experiments009/012/013/016/017/018/019. In particular, elite-EM retains an existing 26,917 cut rather than discovering it.
- Experiment023 is a **registered prospective experiment** (`notes/experiments/experiment023-all-method-topology.md`), not a completed result in this deck.
- Generator-family labels are not strict graph-theoretic planarity certificates.
- Never treat repeated restarts or 1,207 planned method cells as independent graphs, or claim equal-compute superiority without an equal-budget result.

The original 44-slide talk has not been edited.