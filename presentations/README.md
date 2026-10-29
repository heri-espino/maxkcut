# maxkcut — Marp research presentation

## Included files

- `maxkcut_research_journey.marp.md`: editable, 16:9 Marp deck, 45 slides.
- `assets/pipeline.svg`: the experiment pipeline.
- `assets/wins.svg`: winning method graph counts (including ties).
- `assets/mean_differences.svg`: paired mean cut-score differences.
- `build_assets.py`: deterministic SVG source generator (stdlib only).

## Present and export

From this directory, with Node.js and Marp CLI installed:

```powershell
npx @marp-team/marp-cli .\maxkcut_research_journey.marp.md --html --allow-local-files -o .\maxkcut_research_journey.html
npx @marp-team/marp-cli .\maxkcut_research_journey.marp.md --allow-local-files --pdf -o .\maxkcut_research_journey.pdf
npx @marp-team/marp-cli .\maxkcut_research_journey.marp.md --allow-local-files --pptx -o .\maxkcut_research_journey.pptx
```

If CLI is already installed, use `marp` in place of `npx @marp-team/marp-cli`.

## Numerical provenance

Every graph-level result comes from `experiment021-complete-71-gset/statistics/` on GitHub's `experiment021-complete-71-gset` branch. Rankings use each initializer's 50-restart `best_final` for a graph. Ties counted for all winners or split as `1/t` fractional wins. Average effects are unweighted averages of graph-specific matched mean differences; raw scores from different graphs are not pooled.

See `RESEARCH_HANDOFF.md`, `AGENTS.md`, and Experiments019–021 notes for exact programs, commands, graph/factor SHA256 hashes, seeds and limitations. This is documentation only. Exporting the deck must not rerun expensive optimizations.

The 2020 and 2022 papers and the 2021 dissertation are credited in the deck; there are no uncredited reused figures.