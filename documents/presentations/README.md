**Experiment029 100-seed rank addendum (2026-11-16):** The canonical [104-slide full-research Marp source](maxkcut_full_research_001_025.marp.md) now appends **seven independently verified Experiment028–029 slides** after the historical bibliography. These contrast EVA p=41 with p=n=800 on G1/G6/G11/G14/G18, 100 seeds per graph and width, using **antipodal K-MeansNM only**, 500 matched rank contrasts, nominal paired 95% bootstrap intervals and best-of-100 cut/runtime analysis. [Reproducible chart generator](build_experiment029_rank_figure.mjs) reads the archived [per-graph CSV](../../results/gset/experiment029-eva-kmeansnm-100seed/per_graph.csv) and writes [the committed SVG](assets/experiment029_rank_effect_ci.svg); use `node documents/presentations/build_experiment029_rank_figure.mjs --check` to verify it without optimization. The [hosted PDF/HTML workflow](../../.github/workflows/experiment025-marp-presentation.yml) expects 104 slides and does not use the Windows self-hosted runner. See the [complete Experiment029 research note](../../notes/experiments/experiment029-eva-kmeansnm-100seed.md) for provenance and limits.

**Experiment025 complete-results edition (2026-11-03):** [94-slide Marp deck with 12 graph-structure figures](maxkcut_full_research_001_025.marp.md), [reproducible build and Actions instructions](EXPERIMENT025_PRESENTATION.md), [figure generator](build_experiment025_figures.mjs), and [GitHub Actions hosted renderer](../../.github/workflows/experiment025-marp-presentation.yml). The old 44-slide and 74-slide decks remain historical; no EVA, SDP or refinement is rerun for this edition. PDF/HTML exports are uploaded as time-limited workflow artifacts; full raw VM results still require independent audit and backup.

# maxkcut — Marp research presentation

## Included files

- `maxkcut_research_journey.marp.md`: editable, 16:9 Marp deck, **44 slides**.
- `assets/pipeline.svg`: the experiment pipeline.
- `assets/wins.svg`: winning method graph counts (including ties).
- `assets/mean_differences.svg`: paired mean cut-score differences.
- `build_assets.py`: deterministic core SVG source generator (stdlib only).
- `make_extra_graphics.py`: 16 additional explanatory SVG figures (Matplotlib + NumPy); these are schematics unless explicitly labeled with experiment data.
- `render_pdf.py`: offline HTML/MathJax/WeasyPrint renderer used for the illustrated professor edition (requires Python markdown-it-py, WeasyPrint, PyMuPDF, Node.js and mathjax-full).
- [`MaxCut_Research_71_Gset_For_Professor_Angel.pdf`](MaxCut_Research_71_Gset_For_Professor_Angel.pdf): **archived, complete 44-page illustrated PDF**, prepared for Professor Ángel E. Rodríguez Fernández.

## Build extra scientific graphics

```powershell
python .\make_extra_graphics.py
```

Existing figure files are committed. Regenerating the graphics does not rerun EVA, rounding, or statistical benchmarks. For the exact repository-preserved PDF use the linked PDF above; `render_pdf.py` is an offline build script intended for an environment with the listed renderer dependencies.

## Present and export

From this directory, with Node.js and Marp CLI installed:

```powershell
npx @marp-team/marp-cli .\maxkcut_research_journey.marp.md --html --allow-local-files -o .\maxkcut_research_journey.html
npx @marp-team/marp-cli .\maxkcut_research_journey.marp.md --allow-local-files --pdf -o .\maxkcut_research_journey.pdf
npx @marp-team/marp-cli .\maxkcut_research_journey.marp.md --allow-local-files --pptx -o .\maxkcut_research_journey.pptx
```

If CLI is already installed, use `marp` in place of `npx @marp-team/marp-cli`.

## Numerical provenance

Every graph-level result comes from `experiment021-complete-71-gset/statistics/` on the repository's `main` branch (originally Experiment021). Rankings use each initializer's 50-restart `best_final` for a graph. Ties counted for all winners or split as `1/t` fractional wins. Average effects are unweighted averages of graph-specific matched mean differences; raw scores from different graphs are not pooled.

See the [repository research handoff](../../RESEARCH_HANDOFF.md), [canonical agent guide](../../AGENTS.md), and Experiments019–021 notes for exact programs, commands, graph/factor SHA256 hashes, seeds and limitations. This is documentation only. Exporting the deck must not rerun expensive optimizations.

The 2020 and 2022 papers and the 2021 dissertation are credited in the deck; there are no uncredited reused figures.

Experiment025 benchmark reference update (2026-11-05): Three method-colored graph-family bar charts compare best-of-16 signed cut quality with a version-pinned external reference snapshot (Quicopt and OpenIsing). Rebuild using `node documents/presentations/build_experiment025_reference_ratio_figures.mjs`; see `EXPERIMENT025_PRESENTATION.md`. Updated slide count: 104 (including seven E028–E029 addendum slides); export remains PDF and HTML only.
