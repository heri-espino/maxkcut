# Hyperplane Sweep: mathematical explainer and interactive dashboard

This educational companion is separate from the research manuscript in [papers/hyperplane-sweep](https://github.com/heri-espino/maxkcut/tree/main/papers/hyperplane-sweep). The supplied six-page explanation and four PDF figures belong in [docs/hyperplane-sweep](https://github.com/heri-espino/maxkcut/tree/main/docs/hyperplane-sweep), and the standalone HTML and its generator in [dashboard](https://github.com/heri-espino/maxkcut/tree/main/dashboard).

The demo uses eight fixed unit vectors and 13 signed edges. It is illustrative, not a benchmark or a new approximation guarantee. No EVA or SDP solve is required.

Once published, the intended [interactive dashboard](https://heri-espino.github.io/maxkcut/dashboard/) and [PDF](https://heri-espino.github.io/maxkcut/docs/hyperplane-sweep/hyperplane_sweep.pdf) are served by GitHub Pages. The workflow publishes only the HTML and explainer PDF, not the repository's experimental archives. Enable Settings → Pages → Source → GitHub Actions before deploying.

The dashboard HTML is required. The explainer PDF is optional for deployment: if missing, the workflow emits a warning and omits the PDF link until the file is committed. See the companion folder README for reproduction commands.


**Pages deployment recovery (2026-11-26):** [Run 45888110773](https://github.com/heri-espino/maxkcut/actions/runs/45888110773) failed during static site preparation because the original workflow required an explainer PDF that had not been committed. The standalone HTML is present on main. The revised workflow requires only the HTML, warns if the PDF is absent, and includes its link only when the PDF is available. This is a publication-only change; the PDF, LaTeX and figures still need to be archived separately. No numerical computation or university runner is needed.
