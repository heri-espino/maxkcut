# G-set reference values: proven optima versus best-known cuts

**Do not treat every published "best known cut" as a proven global optimum.**

For some benchmark instances there is a *feasible cut* of value $L$ and
an independently derived *valid upper bound* $U$. If $L=U$, the
Max-Cut optimum is proved equal to that value. If $L<U$, the published
cut is a **lower bound** only and the instance remains uncertified from
that evidence.

## Recent toroidal proofs

Mark Glass and Meir Feder, [*Maximum Cut Algorithms and Upper Bounds for
Planar and Toroidal Graphs*, arXiv:2606.28478 (2026)](https://arxiv.org/html/2606.28478),
prove that discarding the global toroidal cycle parity constraints and
solving the corresponding planar relaxation gives a valid **upper bound**.
Their Table 1 reports feasible cuts and independently computed upper bounds
for 17 toroidal G-set instances.

| Graph | Feasible cut reported | Upper bound | Classification |
| --- | ---: | ---: | --- |
| G11 | 564 | 564 | **Proven optimum** |
| G12 | 556 | 558 | Best known only |
| G13 | 582 | 583 | Best known only |
| G32 | 1410 | 1414 | Best known only |
| G33 | 1382 | 1384 | Best known only |
| G34 | 1384 | 1386 | Best known only |
| G48 | 6000 | 6000 | **Proven optimum** |
| G49 | 6000 | 6000 | **Proven optimum** |
| G50 | 5880 | 6000 | Best known only |
| G57 | 3494 | 3494 | **Proven optimum** |
| G62 | 4872 | 4872 | **Proven optimum, improved feasible cut in this paper** |
| G65 | 5562 | 5562 | **Proven optimum** |
| G66 | 6364 | 6366 | Best known only |
| G67 | 6950 | 6952 | Best known only |
| G72 | 7008 | 7008 | **Proven optimum** |
| G77 | 9940 | 9940 | **Proven optimum** |
| G81 | 14060 | 14060 | **Proven optimum** |

**Source:** Glass–Feder (2026), Sections 5–7, especially Lemma 6 and
Table 1. This table reports the paper's results, **not** values we have
independently certified from local graph files. Their numerical bounds
depend on graph topology and the specified G-set benchmark data.

For standard **G11**, the reported feasible solution $L=564$ and proven
upper bound $U=564$ imply $OPT=564$. Our initial G11 run found
$C=546$, so the *empirically achieved* optimality ratio is

$$
C/OPT=546/564\approx 0.9681,
$$

**conditional on confirming our `G-Set/G11.mat` matches canonical G11
and uses the same signed-weight cut objective.**

A 2025/earlier "best-known" column in heuristic benchmark tables did not
by itself prove this optimality; the 2026 upper-bound argument closes
the gap.

## Future benchmark protocol

- Save the **exact SHA256** of each graph input, dimensions, edge counts,
  sign/weight distribution and original edge arrays in the experiment archive.
- Confirm the local graph against the reference dataset before copying
  literature values or claiming the above proof applies to that file.
- For a proved optimum and positive objective, report $C/OPT$ and absolute
  gap $OPT-C$. This is a measured solution-quality ratio, **not** a newly
  proved approximation factor for the algorithm.
- For a best-known feasible cut $B$ without proof, label $C/B$ only as
  *fraction of best-known value*, not the actual approximation ratio.
- If a separately certified upper bound $U>0$ is available and $C$ is a
  valid nonnegative cut, report $C/U$ as a **lower bound** on $C/OPT$;
  do not use an approximate SDP primal objective as if it were $U$.
- Do not infer a theorem or population-level statistical significance from
  the highest cut seen in 20 restarts of one graph.

## Restarts and statistical experimental units

The default is **20 random restarts per method**, not 100. The SDP solve
normally dominates cost and is reused from its fingerprinted cache.

Use a predeclared number of restarts for method comparisons and analyze
the **entire distribution**. GW and K-MeansNM reuse matched Gaussian
starts on the same Gram vectors, making their *per-restart differences*
meaningful within a graph. K-Means2N uses independent initialization:
do not label its restart-index comparison with GW a paired experiment.

Repeated restarts on one fixed graph and one fixed SDP do **not** create
independent graph-level observations. For broader inference, compare
methods across independent graph instances and consider differences in
runtime and cut value as well as success probabilities for reaching a
specified known cut.
