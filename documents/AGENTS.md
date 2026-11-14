# AGENTS.md — Research documents

- `documents/` contains deliverables and their reproducible sources, not raw
  numerical experiment archives. Read the root `AGENTS.md` first.
- Before editing an individual deliverable, read its local `README.md`,
  `AGENTS.md`, `RESEARCH_HANDOFF.md` if present, plus its build workflows.
- Preserve editability: Marp Markdown, modular LaTeX, diagram sources,
  data snapshots, references, archived outputs, and regeneration scripts.
- Every claim based on G-set benchmarks must retain its original citations,
  data identity, experimental status, and uncertainty caveats.
- Do not rerun EVA/SDP/Experiment025 to rebuild documents. Build on a hosted
  GitHub runner or separate clean worktree, never the shared university VM.
- Use `documents/presentations/` for slide decks and
  `documents/monographs/<slug>/` for multi-section theory documents.
- `papers/hyperplane-sweep/` remains at its existing location while its PR
  is active; coordinate with authors before moving live paper branches.
