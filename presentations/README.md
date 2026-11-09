**Export preference (9 November 2026):** For the current full Experiments001–027 deck, generate only **PDF + standalone HTML**. No PowerPoint/PPTX. See [current GitHub Actions build](../.github/workflows/experiment027-marp-presentation.yml); older records may mention historical PPTX artifacts.

**NEW Experiment001–027 merged edition (2026-11-08):** [Full 117-slide Marp source](maxkcut_full_research_001_027.marp.md), [13 deterministic source-grounded SVG figures](assets/), [Experiment027 chart generator](build_experiment027_figures.mjs), [figure source SHA manifest](experiment027-figure-manifest.json), [complete provenance and build instructions](EXPERIMENT027_PRESENTATION.md), and [hosted PDF/HTML export workflow](../.github/workflows/experiment027-marp-presentation.yml). This preserves the previous 97-slide Experiment001–025 edition unchanged. The new results use the remotely published 71-graph/3,408-new-seed Experiment027 audit, not fresh EVA/SDP or numerical optimization. Graph-level best-of-16 +0.3167% (Gain), +0.3027% (Margin), +0.2814% (Pair) over refined GW; no single-arm superiority over HP; 3-arm 42/13/16 is post hoc and unequal budget. Use `node presentations/build_experiment027_figures.mjs --check` and `node presentations/verify_experiment027_presentation.mjs` for validation. Draft PR #80; no merge without authorization.

**Experiment025 complete-results edition (2026-11-03):** [94-slide Marp deck with 12 graph-structure figures](maxkcut_full_research_001_025.marp.md), [reproducible build and Actions instructions](EXPERIMENT025_PRESENTATION.md), [figure generator](build_experiment025_figures.mjs), and [GitHub Actions hosted renderer](../.github/workflows/experiment025-marp-presentation.yml). The old 44-slide and 74-slide decks remain historical; no EVA, SDP or refinement is rerun for this edition. PDF/HTML exports are uploaded as time-limited workflow artifacts; full raw VM results still require independent audit and backup.

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

See `RESEARCH_HANDOFF.md`, `AGENTS.md`, and Experiments019–021 notes for exact programs, commands, graph/factor SHA256 hashes, seeds and limitations. This is documentation only. Exporting the deck must not rerun expensive optimizations.

The 2020 and 2022 papers and the 2021 dissertation are credited in the deck; there are no uncredited reused figures.

Experiment025 benchmark reference update (2026-11-05): Three method-colored graph-family bar charts compare best-of-16 signed cut quality with a version-pinned external reference snapshot (Quicopt and OpenIsing). Rebuild using `node presentations/build_experiment025_reference_ratio_figures.mjs`; see `EXPERIMENT025_PRESENTATION.md`. Updated slide count: 97; export remains PDF and HTML only.
