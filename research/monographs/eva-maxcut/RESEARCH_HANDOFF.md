# Handoff: EVA / SDP theoretical monograph

## Status

Self-contained LaTeX source is in `main.tex` and `sections/`. Part I derives Max-Cut IQP, PSD geometry, SDP primal and dual, factorization, GW rounding, Burer–Monteiro low-rank theory, EVA updates and convergence, dual certification, cross-disciplinary connections. Part II proves the exact SDP values for C5 and the weighted W6(q) family and provides reproducible EVA trace, CVXPY comparison script, and an experimental protocol.

## Proven analytic results

- C5: integer optimum 4, SDP optimum (25+5sqrt(5))/8 = 4.5225424859; primal rank 2; exact dual PSD certificate.
- W6(1): integer optimum 7, SDP optimum 7.368033988749895; primal rank 3; exact dual PSD certificate.
- For 0<q<=2(1+cos(pi/5)), formula: (5/2)(1+cos(pi/5)+q)+5q^2/[8(1+cos(pi/5))].
- EVA coordinate gain delta_i = ||h_i||(1-cos theta_i)/2. Monotonicity by exact block optimization.
- Conditional angular sweep bound O(epsilon^-2) under field norm lower bound. More general statements in cited paper require separate hypothesis checking.

## Reproduce / verify

`python code/check_examples.py` regenerates `experiments/experiment001/results.json`, figures and asserted primal-dual checks. `latexmk -pdf main.tex` compiles. `python code/solve_sdp_compare.py` requires optional CVXPY/SCS.

## Cautions / next tasks

The 2026 paper's broad global convergence phrasing should be cross-checked against original low-rank results. We prove a rank-deficient second-order global-optimality statement and give direct dual certificate tests. This bundle is uncommitted to the user's GitHub repository; do not claim repository publication. Further work: independently test another solver and formalize adaptive signed-weight bounds.
