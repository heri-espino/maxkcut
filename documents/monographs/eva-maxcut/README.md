# Max-Cut SDP, Cholesky / Gram factorization, and EVA — modular LaTeX monograph

Two parts: rigorous mathematical theory and nontrivial worked examples. The main file is `main.tex`, which imports **20 separately editable concept files** using `\input{sections/...}` (including preface, bibliography, and appendix).

## Build

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

If `latexmk` is unavailable, run `pdflatex main.tex` twice. Requires standard TeX Live packages `amsmath, amsthm, mathtools, booktabs, siunitx, tcolorbox, tikz, hyperref, cleveref, listings` and their dependencies. There are no external image downloads required.

## Reproduce examples

```bash
python -m pip install numpy matplotlib
python code/check_examples.py
# Optional independent CVXPY/SCS solution:
python -m pip install cvxpy scs
python code/solve_sdp_compare.py
```

`code/check_examples.py` explicitly checks primal feasibility, exact dual PSD and complementary slackness of the analytical constructions, exhaustive discrete optima, and monotone seeded EVA updates. Results go to `experiments/experiment001/results.json`, and figures to `figures/`.

## Source and interpretive caution

Primary references are Gonzalez-Torres et al. (2026), Goemans–Williamson (1995), and rank/factorization literature. The user's earlier thesis and 2020 clustering paper are cited, as is the 2022 fixed-point rounding study. The 2026 paper's global-optimality/rate claims are carefully separated from conditional or independently demonstrated theorems. This is a standalone pedagogical project and does **not** alter or replace historical factors/results in the `maxkcut` GitHub repository.
