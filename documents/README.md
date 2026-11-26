# Research documents and deliverables

This is the **canonical home for human-facing research deliverables**.
It contains editable sources, figures, build instructions, and any verified
compiled artifacts that are intentionally preserved in Git.

| Collection | Contents | Entry point |
|---|---|---|
| [Presentations](presentations/) | Marp research seminars, archived PDFs, reproducible SVGs, and build tools | [Presentation README](presentations/README.md) |
| [Hyperplane Sweep](hyperplane-sweep/) | Educational algorithm explanation, LaTeX and figures (archive pending) | [Companion README](hyperplane-sweep/README.md) |
| [Monographs](monographs/) | Long-form theoretical expositions with modular LaTeX, code, figures, and experiments | [EVA/Max-Cut monograph](monographs/eva-maxcut/README.md) |

## What does **not** belong here

- `docs/` remains the Sphinx/software documentation and engineering guides.
- `notes/` remains research chronology, historical records, decisions, and experiments.
- `results/`, `reports/`, and `G-Set/` remain numerical data, results,
  local scratch outputs, and benchmarks. Do not move them as document cleanup.
- `src/`, `tests/`, `scripts/`, and `.github/workflows/` retain their roles.
- `papers/hyperplane-sweep/` is **temporarily retained at its existing path**:
  an open manuscript PR uses that location. Do not move an active manuscript
  underneath another agent without first rebasing/coordinating the open PR.
  [Current manuscript](../papers/hyperplane-sweep/README.md).

## Source vs. rendered outputs

Keep source and verifiable source-generating inputs together. Ephemeral build
products can go into `build/` under each collection (globally gitignored).
Only intentionally archived, validated PDF/HTML/PPTX outputs should be
committed or published as durable releases. Never confuse a 30-day Actions
artifact with a permanently archived file.

## Adding new documents

For a standalone research deliverable use `documents/<kind>/<slug>/` and include
`README.md` with scope, provenance, source entry point, rebuild instructions,
verification steps, and scholarly status. Use sequential experiment names
(`experiment001`, `experiment002`, ...) inside research projects when appropriate.
Never discard a source or raw result just to tidy the repository.

The [restructure manifest](migration-manifest.json) maps each prior path to
its new location and gives SHA-256 byte digests. The migration is a move,
not a regeneration or reinterpretation of experimental findings.
