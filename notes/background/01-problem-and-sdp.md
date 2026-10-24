# Max-$k$-Cut: graph partitions, simplex geometry, and SDP

**Source:** [TH21, Chapter 7, §§7.1–7.2, PDF pp. 88–92 / printed pp. 78–82]. See [source index](README.md).

## 1. Combinatorial objective

For an undirected graph $G=(V,E)$ with symmetric weights $w_{ij}$ (zero for nonedges) and label assignment $c:V\to\{1,\ldots,k\}$,

$
F(c)=\sum_{\{i,j\}\in E}w_{ij}\,\mathbf 1[c(i)\neq c(j)].

$
Max-$k$-Cut maximizes $F$. For **nonnegative weights**, splitting a class cannot decrease the objective; for signed weights this monotonicity fails. A convention permitting empty color classes therefore matters, especially when implementations or papers say “$k$ clusters.” [TH21, §7.1] writes a $k$-partition with $|\mathcal A|=k$; rounding algorithms may yield fewer than $k$ *nonempty* classes unless controlled.

The Petersen graph in [TH21, Fig. 25; PDF p. 89 / printed p. 79] admits a proper 3-coloring, hence its unweighted maximum 3-cut is all 15 edges.

## 2. Regular-simplex encoding (Lemma 5)

Represent the $k$ color classes by unit vectors $a_1,\ldots,a_k$, the vertices of a regular simplex. Their Gram relations are

$
a_r^\top a_s =
\begin{cases}
1 & r=s,\\
-1/(k-1) & r\ne s.
\end{cases}

$
An explicit, unambiguous construction in $\mathbb R^k$, on the $(k-1)$-dimensional subspace $\mathbf 1^\perp$, is

$
a_r=\sqrt{\frac{k}{k-1}}
\left(e_r-\frac{1}{k}\mathbf 1\right).

$
(The coordinate representation in thesis Eq. **7.2** uses an equivalent simplex orientation/sign convention.) It immediately gives $\|a_r\|=1$, $\sum_r a_r=0$, and the pairwise inner products above. The notation $\mathbb R^{k-1}$ refers to the **intrinsic dimension**; explicit $k$-coordinate vectors live in $\mathbf 1^\perp\subset\mathbb R^k$.

**Why the lower bound?** Suppose all $k$ unit vectors have common off-diagonal inner product $\beta$. Then

$
0\le\left\|\sum_{r=1}^k a_r\right\|^2
=k+k(k-1)\beta
\quad\Longrightarrow\quad
\beta\ge-\frac1{k-1}.

$
The simplex attains equality: it gives maximal equal angular separation, not arbitrary pairwise antipodal vectors for $k>2$.

Given a partition, put $y_i=a_{c(i)}$. Then

$
\mathbf1[c(i)\ne c(j)]
=\frac{k-1}{k}\left(1-y_i^\top y_j\right).

$
Check: same class gives $0$; different classes gives $\frac{k-1}{k}(1+1/(k-1))=1$. This converts the combinatorial problem to Eq. **7.3**:

$
\max_{y_i\in\{a_1,\ldots,a_k\}}\;
\frac{k-1}{k}\sum_{i<j}w_{ij}(1-y_i^\top y_j).

$
For $k=2$, the simplex is the antipodal pair $\{-1,+1\}$, recovering the standard Max-Cut encoding.

## 3. Vector relaxation: why an extra constraint is essential

Replace the finite simplex choices by general **unit vectors** $v_i$, but retain the lower bound that every discrete simplex assignment satisfies:

$
\begin{aligned}
\max_{\{v_i\}}\quad&
\frac{k-1}{k}\sum_{i<j}w_{ij}(1-v_i^\top v_j),\\
\text{subject to}\quad&
\|v_i\|=1,\qquad
v_i^\top v_j\ge-\frac1{k-1}\quad(i\ne j).
\end{aligned}
\tag{TH21 Eq. 7.4}

$
Without the second constraint, arbitrary unit vectors may have inner product $-1$; when $k>2$, this is more negative than $-1/(k-1)$ and does not respect the discrete simplex geometry. This is the Frieze–Jerrum-style $k$-way relaxation in the thesis.

## 4. Gram matrix and semidefinite program

Let $X_{ij}=v_i^\top v_j$, so $X=X^\top\succeq0$ and $X_{ii}=1$. Conversely every real PSD $X$ with unit diagonal admits a Gram factor $X=V^\top V$ (the Cholesky factor can be replaced by an eigenvalue factorization when singular).

Then thesis Eq. **7.5** is

$
\boxed{
\begin{aligned}
S_{\mathrm{SDP}}=\max_X\quad&
\frac{k-1}{k}\sum_{i<j}w_{ij}(1-X_{ij})\\
\text{subject to}\quad&
X\succeq0,\quad X_{ii}=1,\quad
X_{ij}\ge-\frac1{k-1}\ \ (i\ne j).
\end{aligned}}

$
All partition matrices are feasible, so for a **correctly solved** relaxation $S_{\mathrm{SDP}}\ge F^\star$. A general $X$ is not necessarily a discrete partition matrix, which is precisely why **rounding** is necessary.

### Laplacian trace derivation (thesis Eq. 7.6)

Let $W=(w_{ij})$, $D=\operatorname{diag}(\sum_j w_{ij})$, and $L=D-W$ (the weighted graph Laplacian). Since $X_{ii}=1$,

$
\begin{aligned}
\operatorname{Tr}(LX)
&=\sum_i\left(\sum_jw_{ij}\right)X_{ii}
 -\sum_{i,j}w_{ij}X_{ij}\\
&=2\sum_{i<j}w_{ij}(1-X_{ij}).
\end{aligned}

$
Therefore the same objective equals

$
\boxed{\frac{k-1}{2k}\operatorname{Tr}(LX).}

$
This is thesis **Eq. 7.6**. The equality uses symmetric $W$, zero diagonal, and the unit-diagonal constraints. It is an algebraic identity **even when some weights are negative**; with signed weights the Laplacian need not itself be PSD.

## 5. What the SDP solution tells us geometrically

For a **positive** edge weight $w_{ij}$, the objective rewards making $X_{ij}=v_i^\top v_j$ smaller (more angular separation); for a **negative** edge weight it rewards making the inner product larger. This is a weighted, globally constrained compromise between competing pairwise preferences — *not* a guarantee that every positive edge reaches its minimum inner product.

An SDP factor $V$ yields a unit-vector representation on a sphere in dimension at most $\operatorname{rank}(X)$. Partition labels are still absent. The key distinction:

- **SDP optimization:** maximizes the relaxed *cut* objective over feasible Gram matrices.
- **Rounding:** maps SDP vectors to one of at most $k$ color labels.
- **Spherical $k$-means:** minimizes a within-cluster distortion (or maximizes within-cluster cosine similarity); this is **not identical** to maximizing cut weight.

The possible correlation between these objectives is an **empirical hypothesis**, studied in [RF20] for $k=2$ and [TH21, §7.4] for $k=3,4,5$, not a general equivalence.

## 6. Caveats for derivations and implementations

1. **Labels versus vectors:** $a_r$ are the fixed *color/simplex representatives*; $v_i$ are the *relaxed vertex vectors*; $X$ is the *Gram matrix*. They play different roles.
2. **PSD and feasibility:** floating-point output may have tiny negative eigenvalues and slightly violated constraints. Repairing only the spectrum may change $X_{ii}$ or $X_{ij}$ bounds. Validate *all* constraints after repair.
3. **Sparsity:** zero-weight pairs still participate in the global PSD and lower-bound constraints; omitting their constraints changes the SDP.
4. **Upper bounds:** the primal maximization value of a numerically approximate, possibly infeasible $X$ is not automatically a mathematically certified upper bound; reliable bounds require solver certificates/tolerances or a validated dual.
5. **Problem terminology:** for the optimization version it is more precise to call Max-$k$-Cut **NP-hard**; an appropriately thresholded *decision version* is NP-complete. The thesis uses “NP-Complete” in §7.1.

**Next:** [Rounding algorithms and clustering](02-rounding-methods.md).
