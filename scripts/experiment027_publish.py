"""Publish independently audited Experiment027 data to Git / Git LFS.

Read-only source inputs:
  original Experiment025 VM archive; new Experiment027 full/ and analysis/.
Writes only published/experiment027/ plus explicitly staged Git metadata.
Does NOT solve EVA, run numerical search, or mutate source archives.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "results" / "gset" / "experiment027-angular-greedy"
PUBLISH = PROJECT / "published" / "experiment027"
METHODS = ("angular-margin-greedy", "angular-gain-greedy", "angular-pair-greedy")
AUDIT_FILES = (
    "audited_comparison.json",
    "audited_seed_scores.csv",
    "audited_graph_comparisons.csv",
    "audited_method_leaderboard.csv",
    "audited_paired_seed_deltas.csv",
)
ALL_GRAPHS = tuple([f"G{i}" for i in range(1, 68)] + ["G70", "G72", "G77", "G81"])
CHUNK = 1024 * 1024


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf8")


def record_files(folder):
    """Enumerate source bytes deterministically; disallow transient/symlink inputs."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing graph archive: {folder}")
    out = []
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink in archived source, refusing: {path}")
        if path.is_dir():
            continue
        if path.suffix in (".tmp", ".lock") or ".temporary." in path.name:
            raise RuntimeError(f"Temporary/in-flight source found: {path}")
        if not path.is_file():
            raise ValueError(f"Unexpected source object: {path}")
        out.append(path)
    if not out:
        raise ValueError(f"No source files: {folder}")
    return out


def file_records(folder, paths):
    return [
        dict(path=p.relative_to(folder).as_posix(), size=p.stat().st_size, sha256=digest(p))
        for p in paths
    ]


def validate_graph(new_root, graph):
    folder = new_root / "full" / graph
    for method in METHODS:
        base = folder / method
        meta = base / "status.json"
        if not meta.is_file():
            raise FileNotFoundError(f"Missing method status: {meta}")
        row = json.loads(meta.read_text(encoding="utf8"))
        if row.get("status") != "complete_16" or row.get("starts_completed") != 16:
            raise ValueError(f"Unfinished Experiment027 method {graph}/{method}")
        for seed in range(16):
            for suffix in (".json", ".npz"):
                if not (base / f"{seed:04d}{suffix}").is_file():
                    raise FileNotFoundError(f"Missing completed seed {graph}/{method}/{seed:04d}{suffix}")
            if not (base / "progress" / f"{seed:04d}" / "events.jsonl").is_file():
                raise FileNotFoundError(f"Missing iteration history {graph}/{method}/{seed:04d}")
    for name in ("gw16_common.npz", "gw16_common.json"):
        if not (folder / name).is_file():
            raise FileNotFoundError(f"Missing paired GW directions {graph}/{name}")
    return folder


def validate_audit(new_root, graphs):
    where = new_root / "analysis"
    for name in AUDIT_FILES:
        if not (where / name).is_file():
            raise FileNotFoundError(
                f"Missing {where/name}; first run python scripts/experiment027_analyze.py --graphs "
                + " ".join(graphs))
    report = json.loads((where / "audited_comparison.json").read_text(encoding="utf8"))
    if (report.get("status") != "independently_verified_archives"
            or report.get("graphs") != len(graphs)
            or report.get("completed_new_seeds") != 48 * len(graphs)
            or report.get("audited_method_seed_archives") != 80 * len(graphs)):
        raise ValueError("Audited comparison does not establish complete requested coverage")
    with (where / "audited_graph_comparisons.csv").open(
        newline="", encoding="utf8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    observed = set(r["graph"] for r in rows)
    if (observed != set(graphs) or len(rows) != 3 * len(graphs)
            or any({r["method"] for r in rows if r["graph"] == graph} != set(METHODS)
                   for graph in graphs)):
        raise ValueError("Audited comparison was generated for different graph IDs")
    return where


def archived_graph(source_dir, dest_dir, graph, max_mib=1900):
    """Create append-only content-addressed ZIP and per-file SHA manifest."""
    files = record_files(source_dir)
    entries = file_records(source_dir, files)
    provenance = dict(
        schema="experiment027-raw-graph-archive-v1", graph=graph,
        methods=list(METHODS), starts=16,
        source_root="results/gset/experiment027-angular-greedy/full/" + graph,
        member_hashes=entries,
        # Entire archive membership is identified by its original bytes.
    )
    identity = hashlib.sha256(canonical(provenance)).hexdigest()
    archive = dest_dir / f"{graph}-{identity[:20]}.zip"
    manifest = dest_dir.parent / "manifests" / f"{graph}-{identity[:20]}.json"
    if manifest.is_file() and archive.is_file():
        recorded = json.loads(manifest.read_text(encoding="utf8"))
        if (recorded.get("source_identity") == identity
                and recorded.get("zip_sha256") == digest(archive)):
            return archive, manifest, recorded
        raise ValueError(f"Existing content-addressed archive differs: {archive}")
    if archive.exists() or manifest.exists():
        raise ValueError(f"Incomplete/stale archive pair: {archive} / {manifest}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fd, tempname = tempfile.mkstemp(prefix=f".{graph}-", suffix=".zip.tmp", dir=dest_dir)
    os.close(fd)
    temp = Path(tempname)
    try:
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as out:
            for path, entry in zip(files, entries):
                # Preserve member paths without absolute VM usernames/directories.
                name = f"full/{graph}/{entry['path']}"
                zi = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                zi.compress_type = zipfile.ZIP_DEFLATED
                zi.external_attr = (0o100644 << 16)
                with path.open("rb") as reader, out.open(zi, "w", force_zip64=True) as writer:
                    shutil.copyfileobj(reader, writer, CHUNK)
        size = temp.stat().st_size
        if size > max_mib * 1024 * 1024:
            raise ValueError(
                f"{graph} ZIP is {size/(1024*1024):.0f} MiB; above --max-archive-mib={max_mib}. "
                "Don't bypass remote per-object limits; split by method or use documented external storage.")
        with zipfile.ZipFile(temp) as check:
            bad = check.testzip()
            if bad is not None:
                raise ValueError(f"Corrupt ZIP entry: {bad}")
            if len(check.namelist()) != len(entries):
                raise ValueError(f"ZIP lost files for {graph}")
        recorded = dict(
            **provenance, source_identity=identity,
            zip_file=archive.relative_to(PROJECT).as_posix(),
            zip_sha256=digest(temp), zip_bytes=size,
            member_count=len(entries),
        )
        # Never replace an existing content-addressed archive.
        temp.rename(archive)
        manifest.write_bytes(canonical(recorded))
    finally:
        temp.unlink(missing_ok=True)
    return archive, manifest, recorded


def command(args, cwd=PROJECT, capture=False):
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=capture)
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(map(str,args))}"
            + (f"\n{result.stdout}\n{result.stderr}" if capture else "")
        )
    return result.stdout if capture else ""


def publish_to_git(paths, *, push, branch):
    # Do not merge with another agent's staged changes.
    staged = command(["git", "diff", "--cached", "--name-only"], capture=True).strip()
    if staged:
        raise RuntimeError(f"Unrelated staged changes present: {staged}")
    command(["git", "lfs", "version"], capture=True)
    sample = next(p for p in paths if p.suffix == ".zip")
    attr = command(["git", "check-attr", "filter", "--", str(sample.relative_to(PROJECT))],
                   capture=True)
    if not attr.strip().endswith("filter: lfs"):
        raise RuntimeError(f"Refusing large Git blob; .gitattributes does not apply: {attr}")
    git_branch = command(["git", "branch", "--show-current"], capture=True).strip()
    if not git_branch or git_branch in ("main", "master"):
        raise RuntimeError(f"Refusing publication from protected branch {git_branch!r}")
    # Stage only explicitly generated paths, never old/new raw local source.
    rel = [str(p.relative_to(PROJECT)).replace("\\", "/") for p in paths]
    command(["git", "add", "--"] + rel)
    names = command(["git", "diff", "--cached", "--name-only"], capture=True).splitlines()
    if not set(names).issubset(set(rel)):
        raise RuntimeError("Unexpected staged file; inspect Git index before continuing")
    if not names:
        print("No new Git changes. Already published locally.")
    else:
        command(["git", "commit", "-m", "experiment027: publish audited cuts, trajectories and SHA manifests"])
    if push:
        # Never force-push. Remote rejection requires fetch/review/retry.
        command(["git", "push", "origin", f"HEAD:refs/heads/{branch}"])
        print(f"Pushed to origin/{branch}. Verify remote LFS availability separately.")
    else:
        print(f"Committed locally; push with: git push origin HEAD:{branch}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+", required=True,
                        help="G1 G11 G14, or ALL for all 71 graphs")
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--published-root", type=Path, default=PUBLISH)
    parser.add_argument("--max-archive-mib", type=int, default=1900)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--push", action="store_true",
                        help="Commit audited outputs and git push, without merge")
    parser.add_argument("--branch", default="experiment027-angular-greedy")
    args = parser.parse_args(argv)
    graphs = list(ALL_GRAPHS if args.graphs == ["ALL"] else args.graphs)
    if not graphs or any(g not in ALL_GRAPHS for g in graphs) or len(set(graphs)) != len(graphs):
        parser.error("Use unique canonical graph IDs or ALL")
    if args.max_archive_mib < 1:
        parser.error("--max-archive-mib must be positive")
    root = args.source_root.resolve()
    output = args.published_root.resolve()
    if not output.is_relative_to(PROJECT):
        parser.error("Published output must stay inside the current repository worktree")
    if root == output or output.is_relative_to(root):
        parser.error("Publisher cannot write into the read-only raw source tree")
    source = {}
    bytes_expected = 0
    for graph in graphs:
        folder = validate_graph(root, graph)
        files = record_files(folder)
        size = sum(p.stat().st_size for p in files)
        bytes_expected += size
        source[graph] = folder
    if args.dry_run:
        print(json.dumps(dict(graphs=graphs, graph_count=len(graphs),
                              new_method_seed_archives=len(graphs)*48,
                              source_bytes=bytes_expected,
                              source_gib=round(bytes_expected/1024**3, 3),
                              action="no files created; run independent audit and publish without --dry-run"),
                         indent=2))
        return 0
    # Refresh the independent audit against CURRENT VM bytes before export.
    # This checks saved signed-graph partitions and trajectories; it never optimizes.
    command([sys.executable, str(PROJECT / "scripts" / "experiment027_analyze.py"),
             "--graphs", *graphs, "--new-root", str(root)])
    audit = validate_audit(root, graphs)
    # The independent audit covers the same graph list and is required BEFORE publishing.
    generated = []
    manifests = []
    for graph in graphs:
        archive, manifest, info = archived_graph(source[graph], output / "archives", graph,
                                                max_mib=args.max_archive_mib)
        generated.extend((archive, manifest))
        manifests.append(dict(graph=graph, archive=archive.relative_to(PROJECT).as_posix(),
                              archive_sha256=info["zip_sha256"],
                              source_identity=info["source_identity"],
                              member_count=info["member_count"],
                              zip_bytes=info["zip_bytes"]))
        print(f"Archived {graph}: {info['member_count']} files, "
              f"{info['zip_bytes'] / (1024*1024):.1f} MiB -> {archive.name}", flush=True)
    out_analyses = output / "analysis"
    out_analyses.mkdir(parents=True, exist_ok=True)
    for name in AUDIT_FILES:
        src = audit / name
        dest = out_analyses / name
        if name == "audited_comparison.json":
            doc = json.loads(src.read_text(encoding="utf8"))
            doc["analysis_directory"] = "published/experiment027/analysis"
            dest.write_bytes(canonical(doc))
        else:
            shutil.copy2(src, dest)
        generated.append(dest)
    index = dict(schema="experiment027-publication-v1", audit_graphs=graphs,
                 status="independently_audited_new_method_results",
                 new_methods=list(METHODS), starts=16,
                 historical_eva_recomputed=False,
                 historical_methods_rerun=False,
                 archived_graphs=manifests,
                 analysis={name: digest(out_analyses / name) for name in AUDIT_FILES})
    index_file = output / "index.json"
    index_file.write_bytes(canonical(index))
    generated.append(index_file)
    print(json.dumps(dict(graph_count=len(graphs), LFS_archives=len(manifests),
                          published_path=str(output), status="ready_for_git"),
                     indent=2))
    publish_to_git(generated, push=args.push, branch=args.branch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
