# experiment001 — analytic verification and six-vertex EVA trace

**What:** Numerically verify exact primal/dual SDP certificates for the five-cycle and six-vertex wheel; run and save a sequential EVA trace on the unit-weight wheel.

**Why:** Provide an independently verifiable, nontrivial mathematical example that separates exact SDP optimality, continuous EVA convergence, factorization, and discrete cuts.

**How:** From the monograph root, run `python code/check_examples.py` (NumPy required; matplotlib optional for the plotted trace). The script executes exact graph enumeration for `n=5,6`, primal/dual PSD eigenvalue checks, complementary slackness, and seeded EVA with `p=4`, seed `2026`, angle tolerance `1e-6`. It regenerates `results.json` and trace figures. Optional: `python code/solve_sdp_compare.py` after installing CVXPY/SCS.

**Findings:** See `results.json` and Chapters 13–15 in the PDF. Exact derived SDP optima exceed integer cuts on both graphs; EVA converges numerically to the known wheel optimum under the specified seed and tolerance. All analytic claims are verified with primal and dual matrices.

**Next:** Test additional seeds and rank limits, compare primal/dual diagnostics and rounded cuts on a common fixed factor. Do not confuse this self-contained `experiment001` with numbering in the existing maxkcut repository.
