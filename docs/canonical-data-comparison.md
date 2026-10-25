# The complete canonical G-set and literature comparisons

## What changed

The repository now contains the **complete canonical G-set in text format**:

- `data/G1.txt` through `data/G67.txt`,
- `data/G70.txt`, `data/G72.txt`, `data/G77.txt`, and `data/G81.txt`.

These are exactly **71 graph files**. Some identifiers between 1 and 81
were never part of the standardized G-set corpus. The earlier
`G-Set/` directory also contains 71 MATLAB files but is a *different
stored representation*; do not assume exact identity from filename.

## Canonical source format

Every text graph has this format:

~~~text
800 1600
1 793 1
1 9 -1
1 8 1
...
~~~

The first line is the true graph size `n m` (including isolated
vertices). Each remaining row is a single undirected edge `i j w`,
using **one-based** identifiers, including potentially **negative**
weights. The parser verifies the header, row count, finite weights,
indices, self-loops, and duplicate edges, then produces a symmetric
sparse weighted adjacency. G11 contains **800 vertices and 1600
edges**, and G81 **20,000 vertices and 40,000 edges**.

The ordinary SDP command `maxkcut` and heuristic batch command
`maxkcut-sweep` now **default to `--gset-dir data`**. For previous
MATLAB experiments, pass `--gset-dir G-Set` explicitly.

Changing from MAT to TXT changes the input file's SHA256 and therefore
does **not** reuse a legacy SDP cache: a G11 SDP for the new text input
must be solved again unless the new input has an independently
validated equivalent cached matrix. The package does not silently
substitute or transplant the old solver matrix.

## Recommended Windows PowerShell sequence

Update and run lightweight tests:

~~~powershell
git pull origin main
conda activate maxkcut
python -m pip install -e .
python -m pytest -q
~~~

Inspect all 71 text inputs without optimizing:

~~~powershell
maxkcut-sweep --inspect-only
~~~

**Audit old versus new graph content**, including every signed edge
weight (no SDP solve):

~~~powershell
maxkcut-sweep --compare-inputs
Import-Csv reports/gset/full-first-pass/input_comparison.csv | Format-Table graph,status,n_canonical,n_legacy,max_absolute_difference
~~~

An `identical_graph` row means exact equality of the normalized
weighted adjacency, not equal file hashes or just matching graph size.
Other statuses flag dimension differences, weight/topology differences,
missing old files, or loading failures. Inspect **G11** before
transferring its previously observed cut of 546 to this canonical
benchmark.

Try two graphs with a bounded heuristic (not the SDP):

~~~powershell
maxkcut-sweep --max-graphs 2 --output-dir reports/gset/canonical-first-pass
~~~

Then run the remaining graphs with the **same settings**:

~~~powershell
maxkcut-sweep --output-dir reports/gset/canonical-first-pass
~~~

The runner automatically verifies and skips successful existing cases.
The default is five random starts and a soft 60-second limit per
graph. Each result records its graph SHA256, weighted edge list, all
partitions, scores, timings, seeds and software environment. Failed
graphs are explicit; no solution is fabricated. The outputs are:

- `summary.csv`: all observed graph results;
- `comparison.csv`: results joined with the 17 published toroidal
  references, with exact proof-status labels;
- `input_comparison.csv`: old MAT graph versus canonical TXT audit;
- `graphs/G*.json`, `trials/G*.jsonl`, `artifacts/`: restart records
  and independently verifiable cut assignments.

Regenerate the literature comparison from a saved `summary.csv`
without any numerical optimization:

~~~powershell
maxkcut-sweep --output-dir reports/gset/canonical-first-pass --compare-only
Import-Csv reports/gset/canonical-first-pass/comparison.csv | Where-Object {$_.paper_cut -ne ''} | Format-Table graph,our_best_cut,paper_cut,paper_upper_bound,reference_status,assessment
~~~

## Paper comparison: Glass and Feder (June 2026)

Source: Mark Glass and Meir Feder, *Maximum Cut Algorithms and Upper
Bounds for Planar and Toroidal Graphs*, arXiv:2606.28478v1,
**Table 1 on page 8**:
https://arxiv.org/abs/2606.28478

The paper's table includes **17 toroidal G-set instances**, not all 71
G-set instances. It reports (i) previous known cut, (ii) independently
computed upper bound, and (iii) the paper's feasible cut.

| Graph | Paper cut | Published upper bound | Classification |
| --- | ---: | ---: | --- |
| G11 | 564 | 564 | Proven optimum |
| G12 | 556 | 558 | Best known, not proven |
| G13 | 582 | 583 | Best known, not proven |
| G32 | 1410 | 1414 | Best known, not proven |
| G33 | 1382 | 1384 | Best known, not proven |
| G34 | 1384 | 1386 | Best known, not proven |
| G48 | 6000 | 6000 | Proven optimum |
| G49 | 6000 | 6000 | Proven optimum |
| G50 | 5880 | 6000 | Best known, not proven |
| G57 | 3494 | 3494 | Proven optimum |
| G62 | 4872 | 4872 | Proven optimum, improved by paper |
| G65 | 5562 | 5562 | Proven optimum |
| G66 | 6364 | 6366 | Best known, not proven |
| G67 | 6950 | 6952 | Best known, not proven |
| G72 | 7008 | 7008 | Proven optimum |
| G77 | 9940 | 9940 | Proven optimum |
| G81 | 14060 | 14060 | Proven optimum |

A proven optimum means the paper identifies a feasible cut that
matches a valid upper bound. When the bound is larger than the best
reported cut, the latter remains a **best-known solution**, not an
optimality certificate.

For a locally observed cut `C`, the comparison reports (where
applicable) absolute gap `P-C` and `100*C/P`, where `P` is the paper's
cut. If `P` is certified optimal and the local graph truly matches
the canonical instance, that fraction is an **achieved optimality
ratio**. This does *not* prove an approximation ratio for the algorithm
on arbitrary graphs.

If a computed score exceeds the paper's independently proven upper
bound, the script flags an **input/objective/certificate mismatch** for
investigation; do not immediately publish it as a new record.

## Limits

The new full-corpus first pass is a separate **sparse positive-flip
heuristic**. It is not the same as solving the full Goemans–Williamson
SDP and applying K-Means2N/K-MeansNM rounding. Use the existing
`maxkcut --graphs G11 --k 2 ...` command for a targeted (expensive)
canonical SDP solve. This new TXT input has a distinct cache identity.

These changes install a **repeatable analysis protocol**. They do not
claim that any all-71 run has been executed in this GitHub authoring
environment, nor that stored Windows-local files under Git-ignored
`reports/` were accessible remotely.
