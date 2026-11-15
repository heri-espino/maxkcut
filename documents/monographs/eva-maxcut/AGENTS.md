# Research / AI agent guide

1. `main.tex` is the canonical entry point. Each `sections/NN_*.tex` is one conceptual module; retain file names and `\input` order unless a deliberate restructuring is documented.
2. Read `README.md`, all source modules, `experiments/experiment001/README.md`, and `RESEARCH_HANDOFF.md` before editing.
3. All mathematical numerical claims in worked examples must be reproducible by `python code/check_examples.py`. Exact proofs are in chapters 13–14; floating-point outputs are explicitly approximate.
4. Keep `code/`, plots, `results.json`, the PDF, and the TeX source together when redistributing. Use `latexmk -pdf` and inspect the rendered PDF.
5. For additional campaigns, use sequential `experiment002`, `experiment003`, etc. Document what, why, commands, results, and next steps. Never silently replace archived numerical results.
6. This bundle is not a live clone of the GitHub project. Do not represent its inclusion in any GitHub repository, or any commit/push, as accomplished without verification.
7. Do not claim unconditional EVA global optimality based on a finite-tolerance or first-order KKT stop. Check rank assumptions and dual PSD certificates.
