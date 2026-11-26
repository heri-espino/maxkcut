# Hyperplane Sweep — educational companion

This directory is the canonical location for the six-page mathematical explainer requested by the user, not the active research manuscript. The dashboard remains in [dashboard/](../../dashboard/) and the manuscript in [papers/hyperplane-sweep/](../../papers/hyperplane-sweep/).

## Archive status

The educational LaTeX source and Python figure generator are reconstructed from the earlier ChatGPT draft and committed here. The original binary bundle could not be retrieved in this session; regenerated figures and PDF are built from these sources. The expected archived files are:

- `hyperplane_sweep.pdf` — compiled six-page mathematical explainer
- `hyperplane_sweep.tex` — editable LaTeX source
- `make_figures.py` — Python figure generator
- `r2.pdf`, `r3.pdf`, `events.pdf`, `worked_r3.pdf` — vector figures

The hosted [document build workflow](../../.github/workflows/hyperplane-sweep-document.yml) generates the four figures, compiles the PDF, checks it, records SHA-256 hashes in `source-manifest.json` and archives the generated files on main. No original binary equivalence is claimed. No other research documents are moved.

## Reproduction

With Python, NumPy, Matplotlib and a LaTeX installation (including Latin Modern):

```sh
cd documents/hyperplane-sweep
python -m pip install numpy matplotlib
python make_figures.py
pdflatex -halt-on-error hyperplane_sweep.tex
pdflatex -halt-on-error hyperplane_sweep.tex
```

The dashboard generator is maintained separately under `dashboard/`. The worked example has eight synthetic unit vectors and 13 signed weighted edges; its initial and first-sweep best cuts are 14 and 23. These are illustrative results, not G-set benchmarks. No EVA/SDP solves or numerical experiments are required.

## Publication

The hosted Pages workflow looks for `documents/hyperplane-sweep/hyperplane_sweep.pdf` and publishes it only if present. The dashboard can deploy without the PDF. The document build is separate from the Pages deployment; after the compiled PDF is archived, the Pages workflow can be run manually to publish the updated site. Until the build succeeds, the PDF is not yet verified on main.
