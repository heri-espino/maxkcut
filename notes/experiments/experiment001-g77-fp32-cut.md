# experiment001 — one feasible G77 cut using GPU FP32

**Why:** Prior G77 1-step full PSD-projection probe succeeded but had
no cut. The user requested *only* GPU ADMM float32 and a single
meaningful result. The existing default CPU full eigendecomposition
could dominate runtime and memory at n=14,000.

**What:** Reuse the full GPU PSD eigenvectors from the final ADMM
iteration to draw GW Gaussian hyperplanes; save every packed
14,000-vertex partition and score on original signed edge weights.
The full convex SDP is unchanged. Optional adaptive rho changes
the solver's algorithmic penalty, not precision.

**How to reproduce:** Follow
`docs/g77-gpu-fp32-one-cut.md`. All execution stays on a
feature branch and writes into distinct `experiment001` result
directories; opt-in pushes record outcomes before VM teardown.

**Known prior facts:** G77 canonical 14,000 vertices, 28,000 signed
edges. The earlier one-iteration preflight took 4.33 seconds,
peak PyTorch GPU allocation 7,850,976,256 bytes, numerical dual
residual 59.16 versus 0.14 stopping tolerance. No SDP convergence
or partition resulted from the preflight.

**Findings:** Not yet measured for experiment001. Do not invent a
cut value or treat the preflight's numerical primal as a proof.

**Next:** Run tests on the Windows RTX 4080 SUPER, then the
50-iteration single-backend experiment; verify `FJ` assignments
against the original signed graph and inspect ADMM convergence
diagnostics. If successful, create `experiment002` with longer
iterations, without overwriting experiment001.
