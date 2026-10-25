# Mathematical foundations and sources

## The discrete problem

For an undirected weighted graph with class labels $c(i)\in\{1,\ldots,k\}$,

$$
F(c)=\sum_{i<j}w_{ij}\,\mathbf 1\{c(i)\ne c(j)\}.
$$

A regular $(k-1)$-simplex encodes each class by a unit vector $a_j$ with distinct-class dot products $-1/(k-1)$.

## Semidefinite relaxation

Let $X_{ij}=v_i^\top v_j$ and $L=D-W$. The thesis's Chapter 7 formulation is

$$
\begin{aligned}
\max_X\quad&\frac{k-1}{2k}\operatorname{Tr}(LX)\\
\text{s.t.}\quad&X\succeq 0,\;X_{ii}=1,\;X_{ij}\ge-\frac1{k-1}.
\end{aligned}
$$

For $k=2$, the lower bound is redundant and this is the Goemans–Williamson relaxation.

## Converting vectors into partitions

- **GW for $k=2$:** a random Gaussian hyperplane.
- **FJ for $k>2$:** maximum dot product with $k$ independent Gaussian direction vectors.
- **Spherical $k$-means:** iteratively cluster the *relaxed SDP vectors* using normalized centroid directions. The clustering distortion is not the original graph objective.
- **K-MeansNM:** for $k=2$, start with antipodal Gaussian centroid directions, retaining the highest cut visited along the trajectory.

## Cite the original contribution correctly

- Rodríguez Fernández, A. E. (2021). *On the Study of Fitness Landscapes and the Max-Cut Problem*. Dissertation, Leipzig. Ch. 6 (Max-Cut), Ch. 7 (Max-$k$-Cut).
- Rodríguez-Fernández, A. E., González-Torres, B., Menchaca-Méndez, R. and Stadler, P. F. (2020). *Clustering Improves the Goemans–Williamson Approximation for the Max-Cut Problem*. Computation 8(3):75. DOI [10.3390/computation8030075](https://doi.org/10.3390/computation8030075).
- Felzenszwalb, P., Klivans, C. and Paul, A. (2022). *Clustering with Semidefinite Programming and Fixed Point Iteration*. JMLR 23. [Paper](https://jmlr.org/papers/v23/21-0402.html).

The 2021 thesis **already proposes general $k$ clustering-based rounding**. The 2022 method uses iterated SDP optimization and is conceptually different from a single $k$-means pass. **Neither source demonstrates a strictly improved universal approximation factor for spherical $k$-means.**

Further source-location detail: see `notes/background/README.md` and `notes/background/02-rounding-methods.md` in the repository.
