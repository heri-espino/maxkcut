"""Publish verified Experiment027 archives to GitHub Releases, without Git LFS.

Only the current Windows VM can access the raw Experiment027 and Experiment025
outputs. This script audits them without optimization, creates content-addressed
per-graph ZIPs, uploads immutable release assets with GitHub CLI, confirms their
server-side SHA256, and commits tiny manifests/analyses/index to ordinary Git.

Resumable after upload interruptions; never force-pushes or clobbers assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))

import experiment027_publish as legacy

REPO = "heri-espino/maxkcut"
BRANCH = "experiment027-angular-greedy"
STAGING = HERE / "results/gset/experiment027-release-staging"
DEST = HERE / "published/experiment027/releases"


def stable_bytes(obj):
    return (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf8")


def run(cmd, *, ok_codes=(0,), text=True):
    proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=text, check=False)
    if proc.returncode not in ok_codes:
        raise RuntimeError(
            f"Command failed (code {proc.returncode}): {' '.join(map(str, cmd))}\n"
            + (proc.stderr[-2000:] if isinstance(proc.stderr, str) else "")
        )
    return proc


def repo_info():
    branch = run(["git", "branch", "--show-current"]).stdout.strip()
    if branch != BRANCH:
        raise RuntimeError(f"Expected checkout {BRANCH}, found {branch!r}")
    if run(["git", "diff", "--cached", "--name-only"]).stdout.strip():
        raise RuntimeError("Git index contains unrelated staged files; refuse publication")
    gh = shutil.which("gh")
    if gh is None:
        raise RuntimeError(
            "GitHub CLI missing. On Windows run winget install --id GitHub.cli -e, "
            "then open a new terminal and run gh auth login."
        )
    run(["gh", "auth", "status"])
    return gh


def get_release(tag):
    cmd = ["gh", "api", f"repos/{REPO}/releases/tags/{tag}"]
    result = run(cmd, ok_codes=(0, 1))
    if result.returncode == 1:
        if "404" in result.stderr or "Not Found" in result.stderr:
            return None
        raise RuntimeError("Unable to read release; check GitHub authorization/network")
    return json.loads(result.stdout)


def ensure_release(tag, count):
    existing = get_release(tag)
    if existing is not None:
        return existing
    remote = run(["git", "ls-remote", "origin", f"refs/heads/{BRANCH}"]).stdout.strip()
    if not remote:
        raise RuntimeError(f"Remote branch origin/{BRANCH} is missing")
    remote_sha = remote.split()[0]
    # Create as DRAFT: never publish a partially uploaded set of archives.
    run(["gh", "release", "create", tag,
         "--repo", REPO,
         "--target", remote_sha,
         "--title", f"Experiment027 verified signed Max-Cut data ({count} graphs)",
         "--notes", (
             "Experiment027: independently audited full per-iteration data from "
             "the saved EVA factors and 16 matched GW starts. Download assets "
             "using the SHA256 manifests committed under "
             "published/experiment027/releases/. No new SDP or old methods ran."
         ),
         "--draft", "--latest=false"])
    out = get_release(tag)
    if out is None:
        raise RuntimeError("Release creation did not become visible")
    return out


def check_asset(asset, archive, sha):
    if asset.get("name") != archive.name:
        raise ValueError(f"Wrong asset name for {archive.name}")
    if asset.get("state") != "uploaded":
        raise ValueError(f"Asset not completely uploaded: {archive.name}")
    if int(asset.get("size", -1)) != archive.stat().st_size:
        raise ValueError(f"Remote asset size differs: {archive.name}")
    # GitHub REST release-asset API includes a server-computed digest.
    if asset.get("digest") != "sha256:" + sha:
        raise ValueError(
            f"Missing or mismatched GitHub-computed SHA256 for {archive.name}; "
            "refusing to mark the data as published."
        )


def verify_uploaded(release, archives):
    by_name = {x["name"]: x for x in release.get("assets", [])}
    if len(by_name) != len(release.get("assets", [])):
        raise ValueError("Duplicate asset names in release")
    for archive, expected_sha in archives:
        obj = by_name.get(archive.name)
        if obj is None:
            raise FileNotFoundError(f"Missing GitHub release asset: {archive.name}")
        check_asset(obj, archive, expected_sha)
    return by_name


def copy_analysis(source, dest):
    dest.mkdir(parents=True, exist_ok=True)
    for name in legacy.AUDIT_FILES:
        original = source / name
        target = dest / name
        if name == "audited_comparison.json":
            d = json.loads(original.read_text(encoding="utf8"))
            d["analysis_directory"] = "published/experiment027/releases"
            target.write_bytes(stable_bytes(d))
        else:
            shutil.copyfile(original, target)


def publish_index(tag, info, root, audit_dir, graphs):
    dest = root / tag
    dest.mkdir(parents=True, exist_ok=True)
    copy_analysis(audit_dir, dest / "analysis")
    manifests = dest / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    outputs = []
    for graph, archive, original_manifest, checksum in info:
        source = json.loads(original_manifest.read_text(encoding="utf8"))
        if source.get("zip_sha256") != checksum:
            raise ValueError(f"Manifest ZIP digest mismatch: {graph}")
        # Keep full member SHA256 and provenance; remove local ZIP staging path.
        source["zip_file"] = f"{REPO}/releases/download/{tag}/{archive.name}"
        source["asset_name"] = archive.name
        source["release_tag"] = tag
        source["download_url"] = f"https://github.com/{REPO}/releases/download/{tag}/{archive.name}"
        output = manifests / original_manifest.name
        output.write_bytes(stable_bytes(source))
        outputs.append(dict(graph=graph, asset=archive.name, sha256=checksum,
                            size_bytes=archive.stat().st_size, member_count=source["member_count"],
                            manifest=output.relative_to(HERE).as_posix(),
                            url=source["download_url"]))
    index = dict(
        schema="experiment027-github-release-v1", release_tag=tag,
        release_url=f"https://github.com/{REPO}/releases/tag/{tag}",
        graph_count=len(graphs), graphs=graphs, new_methods=list(legacy.METHODS),
        starts_per_method=16, complete_new_seeds=48*len(graphs),
        eva_recomputed=False, historical_methods_rerun=False,
        assets=outputs, analysis_checksums={
            name: legacy.digest(dest / "analysis" / name)
            for name in legacy.AUDIT_FILES
        },
    )
    (dest / "index.json").write_bytes(stable_bytes(index))
    return dest, index


def commit_small_files(folder, tag, latest=None):
    if run(["git", "diff", "--cached", "--name-only"]).stdout.strip():
        raise RuntimeError("Unrelated staged Git changes; refuse to commit")
    # Only ordinary-Git publication outputs, never the ZIPs or VM history.
    path = folder.relative_to(HERE).as_posix()
    paths = [path]
    if latest is not None:
        paths.append(latest.relative_to(HERE).as_posix())
    run(["git", "add", "--", *paths])
    staged = run(["git", "diff", "--cached", "--name-only"]).stdout.splitlines()
    if not all(s.startswith(path + "/") or s in paths for s in staged):
        raise RuntimeError("Unexpected staged file; do not push")
    if staged:
        run(["git", "commit", "-m", f"experiment027: publish verified Release metadata {tag}"])
    run(["git", "push", "origin", f"HEAD:refs/heads/{BRANCH}"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graphs", nargs="+", required=True,
                   help="G1 G11 G14 or ALL")
    p.add_argument("--dry-run", action="store_true",
                   help="Check existence/size only; do not audit, package, upload, or push")
    p.add_argument("--source-root", type=Path, default=legacy.SOURCE)
    p.add_argument("--max-archive-mib", type=int, default=1900)
    args = p.parse_args(argv)
    graphs = list(legacy.ALL_GRAPHS if args.graphs == ["ALL"] else args.graphs)
    if not graphs or len(set(graphs)) != len(graphs) or any(g not in legacy.ALL_GRAPHS for g in graphs):
        p.error("Graphs must be unique supported G-set graph IDs, or ALL")
    if args.max_archive_mib < 1 or args.max_archive_mib > 1900:
        p.error("Archives must be smaller than the GitHub Release 2 GiB asset limit")
    src = args.source_root.resolve()
    folders = [(g, legacy.validate_graph(src, g)) for g in graphs]
    size = sum(f.stat().st_size for _, base in folders for f in legacy.record_files(base))
    plan = dict(graphs=graphs, complete_new_seeds=len(graphs)*48,
                uncompressed_bytes=size, uncompressed_gib=round(size/1024**3, 3),
                storage="GitHub Release assets (no Git LFS)", no_optimization=True)
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return 0
    repo_info()
    # Fresh independent audit of every historical comparator and new seed;
    # never trust a stale local summary.
    run([sys.executable, str(HERE / "scripts/experiment027_analyze.py"),
         "--graphs", *graphs, "--new-root", str(src)])
    audit = legacy.validate_audit(src, graphs)
    items = []
    for g, folder in folders:
        archive, manifest, metadata = legacy.archived_graph(
            folder, STAGING / "archives", g, max_mib=args.max_archive_mib
        )
        items.append((g, archive, manifest, metadata["zip_sha256"]))
        print(f"Packed {g}: {archive.name}, {archive.stat().st_size / 1024**2:.1f} MiB", flush=True)
    if len(items) > 1000:
        raise ValueError("More than GitHub Release's 1000-asset allowance")
    fingerprint = hashlib.sha256(stable_bytes(
        [(g, checksum) for g, _, _, checksum in items]
    )).hexdigest()[:20]
    tag = f"experiment027-data-{len(graphs)}g-{fingerprint}"
    release = ensure_release(tag, len(graphs))
    for g, archive, _, checksum in items:
        remote = next((a for a in release.get("assets", []) if a["name"] == archive.name), None)
        if remote is not None:
            check_asset(remote, archive, checksum)
            print(f"Verified preexisting release asset {g}", flush=True)
            continue
        # Never --clobber: preserve immutable published data and partial progress.
        run(["gh", "release", "upload", tag, str(archive), "--repo", REPO])
        release = get_release(tag)
        if release is None:
            raise RuntimeError("Release disappeared during upload")
        verify_uploaded(release, [(archive, checksum)])
        print(f"Uploaded and server-digest-verified {g}", flush=True)
    release = get_release(tag)
    verify_uploaded(release, [(archive, checksum) for _, archive, _, checksum in items])
    if release.get("draft"):
        run(["gh", "release", "edit", tag, "--repo", REPO, "--draft=false", "--latest=false"])
        release = get_release(tag)
        if release is None or release.get("draft"):
            raise RuntimeError("Release remained draft after publication")
    folder, idx = publish_index(tag, items, DEST, audit, graphs)
    # Stable discovery path for future agents without guessing content tags.
    # A smaller pilot must never downgrade a previously published full study.
    pointer = DEST.parent / "latest_release.json"
    previous = json.loads(pointer.read_text(encoding="utf8")) if pointer.is_file() else None
    update_pointer = previous is None or len(graphs) >= int(previous.get("graph_count", 0))
    if update_pointer:
        pointer.write_bytes(stable_bytes(dict(
            schema="experiment027-latest-release-v1",
            graph_count=len(graphs), release_tag=tag, release_url=idx["release_url"],
            index=(folder / "index.json").relative_to(HERE).as_posix(),
        )))
    commit_small_files(folder, tag, latest=pointer if update_pointer else None)
    print(json.dumps(dict(status="published", release=idx["release_url"],
                          index=str(folder / "index.json"), graph_count=len(graphs)),
                     indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
