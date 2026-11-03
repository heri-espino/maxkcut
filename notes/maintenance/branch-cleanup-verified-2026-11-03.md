# Verified branch cleanup — 2026-11-03

**Repository:** `heri-espino/maxkcut`. **Approved action:** retire 62 merged-PR
branches; preserve unpublished research. **Execution workflow:** [Actions
#41660307392](https://github.com/heri-espino/maxkcut/actions/runs/41660307392),
merged from [PR #69](https://github.com/heri-espino/maxkcut/pull/69) into
`main` at commit `8b8d887d97f0060f6da6217b62babfbf736acb1d`.

## Final independently verified facts

| Metric | Verified result |
| --- | ---: |
| Branches before maintenance feature branch was created | 70 |
| Branches immediately before deletion (including maintenance branch) | 71 |
| Exact SHA-verified merged branch refs deleted | **62/62** |
| Remote archive tags created and confirmed for otherwise unreachable merged tips | **31** |
| Deletions skipped or failed | **0** |
| Branches remaining | **9** |
| Closed-but-unmerged research branches retained | **6/6** |
| Protected active branches retained | **2/2** |
| Workflow outcome | **success** |

The executor confirmed `deleted=62`, `archive_tags_verified=31`,
`skipped_or_failed=[]`, `remaining_remote_branches=9`, and both
`main` and `experiment025-results` present in its final JSON audit.
The audit was uploaded as an artifact named `merged-branch-cleanup-audit`.
GitHub's Git matching-refs API independently returned exactly 31
`refs/tags/archive/merged/...` references, pointing to their recorded
historic commits. These tags intentionally preserve work not otherwise
reachable from the retained branch tips.

Remaining branches, as returned by GitHub's live branch listing:

- `main` — stable source and optional visualization
- `experiment025-results` — live numerical EVA/G-set experiment, untouched
- `maintenance/prune-merged-branches-and-optional-viz` — completed
  maintenance PR #69 head (eligible for later routine deletion)
- `docs/gset-71-literature-references` — original closed unmerged PR #43
- `docs/illustrated-maxcut-sdp-eva-tutorial-20261028` — original
  closed unmerged PR #35
- `experiment001-g77-gpu-fp32-single-cut` — original closed unmerged PR #31
- `experiment007-eva-mst-bestcut-flips` — four non-main commits, no PR
- `experiment014-gw-expected-refine` — original closed unmerged PR #44
- `feature/interactive-sdp-eva-geometry` — original closed unmerged PR #34,
  source copied to the optional `maxkcut_viz` package on `main`

## Follow-up handling

Do **not** delete the six unmerged branches without explicitly archiving
their distinct code/data/binary artifacts and verifying Git LFS backing.
The visualization's maintained integration is now on `main` via
`src/maxkcut_viz/` and `pip install -e ".[viz]"`; its historical source
branch is retained for comparative provenance. The maintenance branch can
be deleted as a normal merged PR branch after no agent is depending on it.

The active Experiment025 source fingerprint remains based on the unchanged
`src/maxkcut/*.py`, so this visual package does not alter its numerical
protocol identity. No new EVA/SDP solve or full experimental rerun took
place during branch cleanup.

## Recovery and reproduction

[Inventory](branch-cleanup-inventory.json) and
[read-only-first script](../../scripts/maintenance/prune_merged_branches.py)
remain in the repo. The manifest's 62 SHA-pinned refs are now all absent;
subsequent dry runs report `ALREADY_ABSENT` and never touch the remaining
branches. Historical unreachable merged tips can be checked with
`git ls-remote --tags origin "refs/tags/archive/merged/*"`.
Do not reset `experiment025-results` to reconstruct any earlier branch.
