# From Max-Cut to SDP, EVA and rounding

This self-contained, two-column tutorial is intended for readers without prior knowledge of graph optimization. It derives the Max-Cut objective, its sign/matrix/PSD representations, the convex SDP relaxation, and the nonconvex sequential EVA update. It then develops GW random-hyperplane rounding, K-Means2N, and K-MeansNM step by step, including their probability/initialization differences, G-set graphs, 1,000-restart experiments, and exact saved-factor workflow.

- **Source:** [`maxcut_sdp_eva_rounding_tutorial.tex`](maxcut_sdp_eva_rounding_tutorial.tex)
- **PDF:** [`maxcut_sdp_eva_rounding_tutorial.pdf`](maxcut_sdp_eva_rounding_tutorial.pdf)

The typography and two-column layout are based on the uploaded `trend_comparison_two_column.tex` template. All figures are generated in native TikZ; no external figures or proprietary fonts are required.

To compile locally from the repository root (with TeX Live or MiKTeX):

```powershell
pdflatex -interaction=nonstopmode -halt-on-error docs/tutorials/maxcut_sdp_eva_rounding_tutorial.tex
pdflatex -interaction=nonstopmode -halt-on-error docs/tutorials/maxcut_sdp_eva_rounding_tutorial.tex
```

Running inside `docs/tutorials/` will place the PDF next to the `.tex`. The included manual GitHub Actions workflow can rebuild and commit the PDF to the documentation branch. Keep generated `.aux`, `.log`, and `.out` files untracked.

**Scientific caution:** EVA's factorized vector problem is nonconvex. A completed EVA solve, a numerical SDP primal value, and a proven optimal discrete Max-Cut are distinct claims. The classical 0.87856 GW guarantee assumes nonnegative edge weights and an appropriate SDP solution; G-set instances often include negative weights.
