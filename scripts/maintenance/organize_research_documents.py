#!/usr/bin/env python3
"""One-time, reproducible migration of research deliverables to documents/.

This utility NEVER touches any optimizer, graph, experiment result, or remote VM.
Use from a clean, dedicated Git branch at the repository root:
    python scripts/maintenance/organize_research_documents.py --apply
    python scripts/maintenance/organize_research_documents.py --check

Renames are performed by git mv so binary Git blobs and historical provenance
are preserved byte-for-byte. The migration manifest records every moved file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MOVES = (
    ("presentations", "documents/presentations"),
    ("research/monographs/eva-maxcut", "documents/monographs/eva-maxcut"),
)
TEXT_SUFFIXES = {".md", ".mjs", ".js", ".py", ".yml", ".yaml", ".tex", ".json", ".bib"}
BINARY_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".pptx", ".gif"}


def git(*args: str, capture: bool = True) -> str:
    out = subprocess.run(["git", *args], cwd=ROOT, check=True,
                         stdout=subprocess.PIPE if capture else None,
                         stderr=subprocess.PIPE if capture else None)
    return out.stdout.decode("utf-8", errors="replace") if capture else ""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tracked_paths() -> list[str]:
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return sorted(p.decode("utf-8") for p in raw.split(b"\0") if p)


def transform(s: str, path: str) -> str:
    # Absolute repository paths and source links: avoid already-migrated paths.
    s = s.replace("research/monographs/eva-maxcut", "documents/monographs/eva-maxcut")
    s = re.sub(r"(?<!documents/)presentations/", "documents/presentations/", s)
    if path.startswith("documents/presentations/"):
        # This collection is one level deeper in the repository. Repair links
        # that previously left presentations/ for repository-root siblings.
        s = re.sub(r"(?<![./])\.\./(?=(?:\.github|notes|results|docs|AGENTS\.md|RESEARCH_HANDOFF\.md|README\.md)(?:/|\b))", "../../", s)
        # A few Node scripts used the *bare directory name* in path.join().
        if path.endswith((".mjs", ".js")):
            s = s.replace('"presentations"', '"documents/presentations"')
            s = s.replace("'presentations'", "'documents/presentations'")
    return s


def editable(path: str) -> bool:
    p = Path(path)
    if p.suffix.lower() not in TEXT_SUFFIXES:
        return False
    if path in {"README.md", "AGENTS.md", "RESEARCH_HANDOFF.md", "docs/architecture.md"}:
        return True
    if path == ".github/workflows/experiment025-marp-presentation.yml":
        return True
    if path.startswith("documents/presentations/"):
        return True
    if path.startswith("documents/monographs/eva-maxcut/"):
        return True
    if path.startswith("docs/") and p.suffix == ".md":
        return True
    if path.startswith("notes/") and p.suffix == ".md":
        return True
    return False


CATALOGUE = """# Research documents and deliverables

This is the **canonical home for human-facing research deliverables**.
It contains editable sources, figures, build instructions, and any verified
compiled artifacts that are intentionally preserved in Git.

| Collection | Contents | Entry point |
|---|---|---|
| [Presentations](presentations/) | Marp research seminars, archived PDFs, reproducible SVGs, and build tools | [Presentation README](presentations/README.md) |
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
"""

LOCAL_AGENTS = """# AGENTS.md — Research documents

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
"""


def prepend_after_title(path: Path, title: str, text: str) -> None:
    current = path.read_text(encoding="utf-8")
    if text.splitlines()[0] in current:
        return
    if current.startswith(title):
        revised = title + "\n" + text + "\n" + current[len(title):].lstrip("\n")
    else:
        revised = text + "\n" + current
    path.write_text(revised.rstrip("\n") + "\n", encoding="utf-8")


def apply() -> None:
    branch = git("branch", "--show-current").strip()
    if not branch or branch in {"main", "master", "experiment025-results"}:
        raise RuntimeError("Use a dedicated named documentation branch; never run this on main/VM results")
    if git("status", "--porcelain").strip():
        raise RuntimeError("Refusing to migrate a dirty worktree")
    before = tracked_paths()
    collected: list[dict] = []
    for old, new in MOVES:
        src, dst = ROOT / old, ROOT / new
        if not src.is_dir() or dst.exists():
            raise RuntimeError(f"Missing source or existing destination: {old} -> {new}")
        matching = [p for p in before if p.startswith(old + "/")]
        if not matching:
            raise RuntimeError(f"No tracked paths found in {old}")
        for p in matching:
            f = ROOT / p
            if not f.is_file() or f.is_symlink():
                raise RuntimeError(f"Unexpected absent or symlinked asset: {p}")
            collected.append({"old": p, "new": new + p[len(old):],
                              "sha256": sha256(f), "bytes": f.stat().st_size})
        dst.parent.mkdir(parents=True, exist_ok=True)
        git("mv", "--", old, new)

    # Patch links/document generator paths only; never alter optimizer sources,
    # archived numeric data or graph inputs. The manifest checks all moves.
    for path in tracked_paths():
        if not editable(path):
            continue
        f = ROOT / path
        try:
            old_text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"Expected UTF-8 file: {path}") from exc
        new_text = transform(old_text, path)
        if new_text != old_text:
            f.write_text(new_text, encoding="utf-8")

    docs = ROOT / "documents"
    (docs / "README.md").write_text(CATALOGUE, encoding="utf-8")
    (docs / "AGENTS.md").write_text(LOCAL_AGENTS, encoding="utf-8")
    manifest = {
        "schema": "maxkcut-documents-migration-v1",
        "purpose": "Preserve research deliverables without touching experimental data",
        "moved_count": len(collected),
        "collections": [{"from": a, "to": b} for a, b in MOVES],
        "files": sorted(collected, key=lambda r: r["old"]),
    }
    (docs / "migration-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    root_readme = ROOT / "README.md"
    prepend_after_title(root_readme, "# maxkcut", """## Research documents

[Browse presentations, the mathematical monograph, and research deliverables](documents/README.md).
The editable sources, figures, and verified PDFs live together under `documents/`;
engineering documentation stays in `docs/`, and raw experiments stay in
`notes/` and `results/`. The active hyperplane-sweep manuscript remains in
`papers/` until its open PR can be coordinated.
""")
    prepend_after_title(ROOT / "AGENTS.md", "# AGENTS.md — Research and coding agent instructions", """**Document organization (2026-11-14):** Human-facing presentation and
monograph collections are now under [`documents/`](documents/README.md).
Use `documents/presentations/` and `documents/monographs/eva-maxcut/` instead of
historical paths. See `documents/AGENTS.md` and the SHA256 migration manifest.
Do not move `papers/hyperplane-sweep/` while its manuscript PR is open, and do
not touch university VM experimental data as part of layout cleanup.
""")
    prepend_after_title(ROOT / "RESEARCH_HANDOFF.md", "# Research handoff — maxkcut", """**Document collection migration (2026-11-14):** The archived presentation
sources/PDFs/plots and 49-page modular EVA monograph were moved (not
regenerated) to [`documents/`](documents/README.md) on an isolated branch.
All old/new paths and file hashes are recorded in
[`documents/migration-manifest.json`](documents/migration-manifest.json).
No numerical optimizer source, G-set corpus, frozen EVA factor, or
Experiment025 result was modified. `papers/hyperplane-sweep/` remains in place
until coordination with its open manuscript PR.
""")
    # Append a durable engineering layout note without rewriting the old
    # static tree diagram and its historical snapshot semantics.
    architecture = ROOT / "docs" / "architecture.md"
    arch_text = architecture.read_text(encoding="utf-8")
    arch_note = """\n## Human-facing research documents (2026-11-14)\n\n`documents/presentations/` contains editable Marp decks, vector diagrams,\nbuild scripts, and archived rendered slides. `documents/monographs/` contains\nthe modular EVA/Max-Cut theoretical book and its reproduction scripts.\nSee [`documents/README.md`](../documents/README.md) for routing. The\n`papers/hyperplane-sweep/` path is intentionally not moved while a PR edits it.\nSoftware docs (`docs/`), experimental ledgers (`notes/`), signed cut archives\n(`results/`) and optimizers (`src/`) remain in their established locations.\n"""
    if "## Human-facing research documents (2026-11-14)" not in arch_text:
        architecture.write_text(arch_text.rstrip() + "\n" + arch_note, encoding="utf-8")

    # Strong byte equality for every moved file not intentionally rewritten;
    # for revised text verify the exact new location and its content category.
    for record in collected:
        moved = ROOT / record["new"]
        if not moved.exists():
            raise RuntimeError(f"Missing migrated item: {record['new']}")
        if Path(record["old"]).suffix.lower() in BINARY_SUFFIXES:
            if sha256(moved) != record["sha256"]:
                raise RuntimeError(f"Binary payload changed during move: {record['old']}")
    print(json.dumps({"status": "migrated", "branch": branch,
                      "files_moved": len(collected),
                      "binary_files_preserved": sum(Path(r["old"]).suffix.lower() in BINARY_SUFFIXES for r in collected)}, indent=2))
    check()


def check() -> None:
    for old, new in MOVES:
        if (ROOT / old).exists() or not (ROOT / new).exists():
            raise RuntimeError(f"Unexpected layout: {old} -> {new}")
    raw = json.loads((ROOT / "documents/migration-manifest.json").read_text(encoding="utf-8"))
    for f in raw["files"]:
        target = ROOT / f["new"]
        if not target.is_file() or (ROOT / f["old"]).exists():
            raise RuntimeError(f"Missing path, or old path retained: {f}")
        if target.suffix.lower() in BINARY_SUFFIXES and sha256(target) != f["sha256"]:
            raise RuntimeError(f"Binary hash mismatch: {f['new']}")
    # All executable references must be updated to the relocated directory.
    must_check = [".github/workflows/experiment025-marp-presentation.yml"]
    must_check += [p for p in tracked_paths() if p.startswith("documents/presentations/") and p.endswith((".mjs", ".js"))]
    for p in must_check:
        content = (ROOT / p).read_text(encoding="utf-8")
        if re.search(r"(?<!documents/)presentations/", content):
            raise RuntimeError(f"Stale executable presentation path in {p}")
        if p.endswith((".mjs", ".js")) and re.search(r"[\"']presentations[\"']", content):
            raise RuntimeError(f"Stale bare presentation directory in {p}")
    assert (ROOT / "documents/presentations/maxkcut_full_research_001_025.marp.md").exists()
    assert (ROOT / "documents/monographs/eva-maxcut/main.tex").exists()
    assert "documents/presentations" in (ROOT / ".github/workflows/experiment025-marp-presentation.yml").read_text(encoding="utf-8")
    print(f"Layout check passed: {raw['moved_count']} migrated files; no lost binaries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--apply", action="store_true", help="Perform isolated git-mv migration")
    group.add_argument("--check", action="store_true", help="Verify completed migration")
    args = parser.parse_args()
    if args.apply:
        apply()
    else:
        check()