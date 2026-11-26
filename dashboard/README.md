# Hyperplane Sweep interactive dashboard

Open `index.html` directly in a browser, or use the intended GitHub Pages address: https://heri-espino.github.io/maxkcut/dashboard/ . The original standalone HTML is preserved byte for byte; the supplied filename was changed to `index.html` for Pages routing. It embeds Plotly.js and does not require a backend or external network.

To reproduce (Python 3):

```sh
python -m pip install plotly
python build_dashboard.py
```

The original generator writes `hyperplane_sweep_dashboard.html` in this directory. To intentionally refresh the published page, replace `index.html` with that generated file after review. See `../docs/hyperplane-sweep/source-manifest.json` for provenance. No numerical benchmark is run.

The GitHub-hosted Pages workflow deploys only the dashboard and the explainer PDF. The separate research manuscript and experimental archives are not published by it. Enable GitHub Actions as the Pages source in repository settings before deployment.
