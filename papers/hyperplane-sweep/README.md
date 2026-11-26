# Event-Driven Hyperplane Sweep Research Paper

**Status:** LaTeX research manuscript, November 2026. Numeric findings are **provisional** until the Experiment025 independent original signed-cut partition audit is complete. This manuscript is not a verified new-record announcement or a submitted publication.

## Source and reproducibility

- Paper: main.tex; references: references.bib.
- Actual algorithm: src/maxkcut/eva_direct_rounding.py (optimize_semicircle, optimize_hyperplane and _guided_tangent).
- Input benchmark: draft PR #75 (analysis/experiment025-complete-aggregate-review), whose compact full_best_final_by_graph.csv was reported with SHA-256 8a9d84cfc35f1d424519b862676facc288a6366bc1820e52975a2bf76678de6c.
- Computation: 71 graphs x 18 methods x 16 common GW seeds = 20,448 checkpoints. Reported primary score is best-of-16 *after* common Flip/KL/ILS refinement. The paper does not claim isolated raw-sweep improvement, equal-compute superiority, or independently verified records.
- Source warning: gain-guided selects negative flip gains despite a comment about improving flips; boundary-guided selects small absolute gains rather than small geometric margins. The paper documents these semantics, without modifying the validated historical computation.

## Build

With a standard TeX Live installation:

    cd papers/hyperplane-sweep
    latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex

A GitHub Actions workflow compiles the LaTeX and BibTeX and publishes the PDF as an Actions artifact. The workflow runs on GitHub-hosted Ubuntu and deliberately does not touch the university Windows runner or its unpublished numerical checkpoints.

## Research integrity / next steps

1. Verify the off-VM backup of raw partitions and iteration journals, hashes, and all graph-method source fingerprints.
2. Independently rescore best binary partitions against original signed graph edges.
3. Export a compact 20,448-row matched-seed table with raw, Flip, KL and ILS best-seen stages.
4. Run fixed-SDP-factor ablations comparing random, negative-gain, positive-gain and margin-guided tangents with equal number of sweeps and equal CPU budgets.
5. Complete an exhaustive prior-art review before making originality claims about the combined method.

Do not silently replace the provisional aggregate or numerical limitations with claims of certified performance.
