# Seven-rounding-methods manuscript handoff

- Branch: **paper-seven-rounding-methods**, derived from **experiment027-angular-greedy**. Stack a draft PR against that base and leave both PRs unmerged.
- Editable LaTeX: **main.tex**, twelve source chapters in **sections/** (seven methods, two general chapters, empirical, plus two appendices), four source vector figures in **figures/** and **references.bib**.
- Data: **data/top_seven.csv**, source checked by **verify_results.py** against 1,278 Experiment025 aggregate rows and 6,816 Experiment027 paired rows.
- Conclusions: Sweep 0.317618%, Angular Gain 0.316704%, Angular Margin 0.302682%, Angular Pair 0.281426%, control-paired 0.271877%, EM–Gibbs 0.270688%, cut-aware K-Means 0.266807%. All post-Flip/KL/ILS, equal graph weighting, best of 16.
- E025 full raw archive audit is still incomplete; E027 independently audits the three new arms plus two historical comparators. Not a definitive statistical winner.
- Code nuance: Hyperplane Sweep labelled gain proposals use **negative** one-flip gain, whereas Angular Gain Greedy uses **positive** gain / angular margin. Do not hide discrepancy.
- Compile: hosted **seven-rounding-methods-manuscript.yml**, PDF artifact and LaTeX log. No PPTX. Never rerun numerical code.
- Next: compile and visually inspect PDF, correct LaTeX warnings if any, verify bibliography metadata, refine source literature review and deeper pure-rounder methods without conflating Experiment028 proposals with completed data.
