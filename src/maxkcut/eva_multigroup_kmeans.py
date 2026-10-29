"""Experiment017: antipodally initialized multi-cluster spherical K-MeansNM.

Keep the original full EVA factor Y fixed. Cluster its *rows* into
k = 2,4,6,8,10,... spherical groups. At EACH Lloyd iteration solve,
EXACTLY, the much smaller binary Max-Cut obtained by merging these
groups into two sides of the ORIGINAL signed weighted graph.

There are 2^(k-1) distinct binary assignments up to a global
complement, including the trivial one-side cut; hence at k=10 there
are 511 nontrivial assignments. Internal edges within a group
never contribute. The group-pair signed edge sums suffice to
compute the original graph cut of every merging exactly.

For k=2 with init='paired' and top_merges=1, the initial RNG,
Lloyd trajectory, and best intermediate partition reproduce the
repository's original K-MeansNM (before its optional flip step).
For even k>2, 'paired' seeds k/2 opposing center pairs; standard
Lloyd updates are independent afterward (not forced antipodal).
'independent' seeds k standard random spherical centers as control.

This optimizes group MERGING exactly conditional on clustering,
not the globally best Max-Cut or the Lloyd centroid objective.
"""
from __future__ import annotations

import time
import numpy as np

from .io import edge_arrays
from .eva_kmeans_initializers import INIT2, centers_for_k2
from .objective import cut_weight


def _canonical(labels):
    labels=np.asarray(labels,dtype=np.int8)
    if labels[0]:
        labels=1-labels
    return np.packbits(labels,bitorder="little").tobytes()


def enumerate_group_merges(edges, groups, k, *, top=4):
    """Exactly rank all 2^(k-1) signed two-way group combinations.

    Fix group 0 on side 0 to quotient the global complement symmetry.
    Include the all-zero cut: on negative-weight graphs, zero is
    legitimately preferable to every nontrivial coarse partition.
    Returns [(cut, mask, binary_labels)] in stable descending cut
    order, then smaller integer mask order.
    """
    g=np.asarray(groups)
    if (g.ndim!=1 or not np.issubdtype(g.dtype,np.integer)
            or k<2 or k>16 or top<1
            or len(g)==0 or np.any(g<0) or np.any(g>=k)):
        raise ValueError("Invalid group labels/count; k must be in 2..16")
    i,j,w=edges
    i,j=np.asarray(i),np.asarray(j)
    w=np.asarray(w,dtype=np.float64)
    if (len(i)!=len(j) or len(j)!=len(w)
            or np.any(i<0) or np.any(j<0)
            or np.any(i>=len(g)) or np.any(j>=len(g))
            or not np.isfinite(w).all()):
        raise ValueError("Group merge edges invalid")
    low=np.minimum(g[i],g[j])
    high=np.maximum(g[i],g[j])
    graph_by_group=np.bincount(
        low*k+high,weights=w,minlength=k*k,
    ).reshape(k,k)
    u,v=np.triu_indices(k,1)
    signed_pair_weights=graph_by_group[u,v]
    masks=np.arange(1 << (k-1),dtype=np.uint32)
    sides=np.zeros((len(masks),k),dtype=np.uint8)
    sides[:,1:]=(
        masks[:,None]>>np.arange(k-1,dtype=np.uint32)[None,:]
    ) & 1
    separation=(sides[:,u]!=sides[:,v])
    # Tiny k-group enumeration requires no BLAS. The explicit
    # reduction is also robust when a self-hosted Windows runner
    # launches Conda Python without a fully initialized DLL PATH.
    scores=np.sum(
        separation * signed_pair_weights[None,:],
        axis=1,dtype=np.float64,
    )
    ranking=np.lexsort((masks,-scores))[:min(top,len(masks))]
    result=[]
    for idx in ranking:
        cut=float(scores[idx])
        mask=int(masks[idx])
        labels=sides[idx,g].astype(np.int8)
        observed=float(cut_weight(edges,labels))
        if not np.isclose(observed,cut,atol=1e-7,rtol=0):
            raise ArithmeticError(
                "Group contraction score differs from original signed cut"
            )
        result.append((observed,mask,labels))
    return result


def optimize_multigroup_kmeans(
    vectors, graph, rng, *, k=4, init="paired",
    max_iter=100, top_merges=4,
):
    """Keep best original signed graph binary merges over Lloyd iterations.

    All returned binary partitions are coupled to their recorded
    cluster assignment and integer bit mask. top_merges controls
    the TOTAL returned unique coarsenings across iterations,
    not the number of group cuts tested (ALL are always tested).
    """
    y=np.asarray(vectors,dtype=np.float64)
    if (y.ndim!=2 or y.shape[0]!=graph.shape[0]
            or y.shape[0]<2 or y.shape[1]<1 or not np.isfinite(y).all()
            or k<2 or k>16 or k>len(y) or max_iter<1
            or top_merges<1 or init not in INIT2
            or (k>2 and init not in ("paired","independent"))
            or (init=="paired" and k%2)):
        raise ValueError("Invalid multigroup K-Means configuration")
    if np.any(np.abs(np.linalg.norm(y,axis=1)-1)>1e-5):
        raise ValueError("Expected original EVA unit-row factor")
    edges=edge_arrays(graph)
    n,p=y.shape
    if k==2:
        centers,init_info=centers_for_k2(
            y,rng,init=init,edges=edges,
        )
    else:
        centers=rng.standard_normal((k,p))
        init_info={"screened_hard_cuts":0}
        if init=="paired":
            for group in range(0,k,2):
                centers[group+1]=-centers[group]
        centers/=np.linalg.norm(centers,axis=1)[:,None]
    previous=None
    trace=[]
    found={}
    began=time.monotonic()
    for iteration in range(1,max_iter+1):
        groups=np.argmax(y@centers.T,axis=1).astype(np.int16)
        merges=enumerate_group_merges(
            edges,groups,k,top=top_merges,
        )
        for cut,mask,binary in merges:
            key=_canonical(binary)
            if key not in found:
                found[key]={
                    "score":cut,"mask":mask,
                    "labels":binary.copy(),
                    "groups":groups.copy(),
                    "iteration":iteration,
                }
        trace.append({
            "iteration":iteration,
            "best_merge_cut":float(merges[0][0]),
            "best_merge_mask":int(merges[0][1]),
            "groups_nonempty":int(np.count_nonzero(
                np.bincount(groups,minlength=k)
            )),
            "group_sizes":np.bincount(
                groups,minlength=k,
            ).astype(int).tolist(),
            "evaluated_group_merges":int(1<<(k-1)),
        })
        if previous is not None and np.array_equal(groups,previous):
            break
        previous=groups.copy()
        for c in range(k):
            members=y[groups==c]
            if not len(members):
                centers[c]=y[rng.integers(len(y))]
            else:
                subtotal=np.sum(members,axis=0)
                magnitude=np.linalg.norm(subtotal)
                centers[c]=(
                    subtotal/magnitude if magnitude>1e-12
                    else y[rng.integers(len(y))]
                )
    candidates=sorted(
        found.values(),
        key=lambda row:(-row["score"],row["iteration"],row["mask"]),
    )[:top_merges]
    if not candidates:
        raise AssertionError("No group-merging candidate")
    for candidate in candidates:
        if not np.isclose(
            cut_weight(edges,candidate["labels"]),
            candidate["score"],atol=1e-8,rtol=0,
        ):
            raise AssertionError("Saved candidate cut does not match G-set")
    return candidates,{
        "k":int(k),"init":init,
        "initialization_details":init_info,
        "max_iter":int(max_iter),
        "lloyd_iterations":len(trace),
        "distinct_binary_merges_seen":len(found),
        "best_raw_cut":float(candidates[0]["score"]),
        "tested_group_assignments_per_iteration":1<<(k-1),
        "nontrivial_group_assignments_per_iteration":(1<<(k-1))-1,
        "trace":trace,"seconds":time.monotonic()-began,
    }
