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