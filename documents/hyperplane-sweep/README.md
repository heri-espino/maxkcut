# Hyperplane Sweep — educational companion

This directory is the canonical location for the six-page mathematical explainer requested by the user, not the active research manuscript. The dashboard remains in [dashboard/](../../dashboard/) and the manuscript in [papers/hyperplane-sweep/](../../papers/hyperplane-sweep/).

## Archive status

The original compiled PDF and source bundle were produced in ChatGPT but are not yet committed here. Preserve the original files together (do not replace them with placeholders):

- `hyperplane_sweep.pdf` — compiled six-page mathematical explainer
- `hyperplane_sweep.tex` — editable LaTeX source
- `make_figures.py` — Python figure generator
- `r2.pdf`, `r3.pdf`, `events.pdf`, `worked_r3.pdf` — vector figures

Extract these files from the supplied `hyperplane_sweep_complete.zip` into this directory. Do not move or overwrite other research documents. Record SHA-256 digests of the imported originals; check that the PDF renders and that all referenced figures exist.

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

The hosted Pages workflow looks for `documents/hyperplane-sweep/hyperplane_sweep.pdf` and publishes it only if present. The dashboard can deploy without the PDF. This is a path correction and archive preparation, not a verified PDF upload or deployment.
