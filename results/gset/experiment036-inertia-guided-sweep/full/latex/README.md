# Experiment036 — numerical appendix (LaTeX source)

Self-contained document project using STIX2 text and math and the audited AwesomeFigures PDF assets (hybrid/vector export, original 600 dpi rasterization). Source: run 45885883213, artifact 15149891652; 716,826 verified records. No optimization or audit rerun was required. Compact CSV/JSON summaries are included. Large candidate/event tables and binary trajectories are intentionally excluded and retained separately under the VM archival policy.

## Build

Install TeX Live (latexmk, latex-extra, science, fonts-extra / STIX2) and Python 3. Then run:

    python generate_results.py
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

The rendered output is main.pdf. results.tex is generated from data/summary.json. figures/ contains all 17 original figure PDFs. The style is reusable: set \ExperimentNumber and \AppendixFooter before loading maxcut-appendix.sty; use normal LaTeX sections, captions, tables and the \resultfigure helper. The original plotting implementation is in scripts/experiment036_report.py in the maxkcut repository. This redesign does not regenerate or change the original plots.

provenance.json records the source artifact and checksums of the compact data and figure files. The older VM storage manifest may reference the pre-redesign PDF, not this derived LaTeX edition. The raw experiment remains unchanged.
