"""Experiment033: raw random-tangent sweep versus independent GW, fixed EVA.

Run from repository root with PYTHONPATH=src. Never solves EVA/SDP or refines labels.
Each graph has a separately verified compressed archive, including all normals,
packed partitions, scores, wall times, candidate counts and incumbent trajectories.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from maxkcut.eva_direct_rounding import optimize_semicircle
from maxkcut.eva import _identity, _prepared_graph, load_eva_factor
from maxkcut.io import edge_arrays, load_graph
from maxkcut.objective import cut_weight

ROOT = Path("results/gset/experiment033-raw-hyperplane-sweep")
INVENTORY = Path("results/gset/experiment021-complete-71-gset/inventory.json")
GRAPHS = tuple([f"G{i}" for i in range(1, 68)] + ["G70", "G72", "G77", "G81"])
PILOT = ("G1", "G11", "G14")
STARTS, STEPS, SEED = 16, 100, 2026
METHODS = ("gw", "sweep")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def unit(z):
    return z / np.linalg.norm(z)


def source_sha(path):
    # Cross-platform source identity: Git may check out Python with CRLF.
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n",b"\n")).hexdigest()


def inputs(g):
    # Match E032's verified cross-platform original graph identity procedure.
    obj = json.loads(INVENTORY.read_text(encoding="utf8"))
    recs = {r["graph"]: r for r in obj["records"]}
    if set(recs) != set(GRAPHS) or len(obj["records"]) != len(GRAPHS) or obj["missing_factors"]:
        raise ValueError("Canonical inventory differs")
    rec = recs[g]
    folder = Path(rec["factor_source"].replace("\\", "/"))
    path = Path(rec["graph_path"].replace("\\", "/"))
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    accepted = {hashlib.sha256(s).hexdigest() for s in (raw, lf, lf.replace(b"\n", b"\r\n"))}
    if rec["graph_sha256"] not in accepted:
        raise ValueError(f"{g}: original graph SHA mismatch, even after EOL normalization")
    graph = load_graph(path)
    factor_path = folder / "factors" / f"{g}.npz"
    case = json.loads((folder / "cases" / f"{g}.json").read_text(encoding="utf8"))
    if (case["graph_sha256"] != rec["graph_sha256"] or case["factor_rank"] != rec["rank"]
            or case["status"] not in ("converged", "iteration_limit")
            or sha(factor_path) != rec["factor_sha256"]):
        raise ValueError(f"{g}: frozen source identity mismatch")
    y, meta = load_eva_factor(factor_path)
    if (y.shape != (rec["vertices"], rec["rank"]) or graph.shape[0] != len(y)
            or meta["sweeps"] != case["factor_sweeps"]):
        raise ValueError(f"{g}: original dimensions or sweep metadata mismatch")
    prepared = _prepared_graph(graph)
    provenance = case["provenance"]
    numba_backend = provenance.get("optimizer_backend", "numpy") == "numba"
    if numba_backend:
        prepared.sort_indices()
    expected = _identity(prepared, y.shape[1], provenance["seed"], provenance["tolerance"])
    if numba_backend:
        expected["algorithm"] = "eva-sequential-numba-v1"
    if meta["identity"] != expected:
        raise ValueError(f"{g}: immutable EVA optimizer identity mismatch")
    if not np.isfinite(y).all() or y.shape[1] < 2:
        raise ValueError(f"{g}: invalid embedding")
    return graph, np.ascontiguousarray(y, dtype=np.float64), rec

def score(edges, labels):
    return float(cut_weight(edges, labels))


def trial(graph, y, start, proposals):
    """Coupled Gaussian draws: independent GW normal or tangent at incumbent.
    Time excludes RNG generation, archive writes and external auditing.
    Each returned candidate is rescored on the original signed graph.
    """
    edges = edge_arrays(graph)
    t = len(proposals) + 1
    p, n = y.shape[1], y.shape[0]
    normals = np.empty((2,t,p))
    labels = np.empty((2,t,(n+7)//8), dtype=np.uint8)
    cuts = np.empty((2,t))
    times = np.zeros((2,t))
    intervals = np.ones((2,t), dtype=np.int64)
    valid = np.ones(t, dtype=bool)
    angles = np.zeros(t)
    predicted = np.zeros(t)
    t0 = time.perf_counter()
    x = (y @ start > 0).astype(np.uint8)
    c = score(edges,x)
    initial_time = time.perf_counter()-t0
    normals[:,0] = start
    labels[:,0] = np.packbits(x)
    cuts[:,0] = c
    times[:,0] = initial_time
    predicted[0] = c
    r = start.copy()
    best = c
    for k,z in enumerate(proposals, 1):
        t0 = time.perf_counter()
        g = unit(z)
        gx = (y @ g > 0).astype(np.uint8)
        gc = score(edges,gx)
        times[0,k] = time.perf_counter()-t0
        normals[0,k], labels[0,k], cuts[0,k] = g, np.packbits(gx), gc

        t0 = time.perf_counter()
        u = z - (z @ r)*r
        if np.linalg.norm(u) <= 1e-13:
            # A deterministic fallback only for a numerically parallel proposal.
            u = np.eye(p)[np.argmin(np.abs(r))]
            u -= (u @ r)*r
        u = unit(u)
        rr, sx, info = optimize_semicircle(graph,y,r,u)
        sc = score(edges,sx)
        times[1,k] = time.perf_counter()-t0
        if not np.isclose(sc,info["realized_arc_cut"],atol=1e-7,rtol=0):
            raise ValueError("Sweep score mismatch")
        if sc < best-1e-7:
            raise ValueError("Sweep lost incumbent")
        normals[1,k], labels[1,k], cuts[1,k] = rr, np.packbits(sx), sc
        intervals[1,k] = info["distinct_events"]+1
        valid[k], angles[k], predicted[k] = info["numerically_valid"], info["theta"], info["predicted_arc_max"]
        if sc > best+1e-8:
            r, best = rr.copy(), sc
    return dict(normals=normals, labels=labels, cuts=cuts, seconds=times,
                intervals=intervals, valid=valid, angles=angles, predicted=predicted,
                best=np.maximum.accumulate(cuts,axis=1))


def warmup():
    # Compile the existing Numba kernel outside measured trajectories.
    a = sparse.csr_matrix(np.array([[0.,1.,-1.],[1.,0.,2.],[-1.,2.,0.]]))
    y = np.array([[1.,0.],[0.,1.],[-1.,0.]])
    optimize_semicircle(a,y,np.array([1.,0.]),np.array([0.,1.]))


def location(phase,g):
    return ROOT/phase/"graphs"/g


def verify(g, phase, starts, steps):
    graph, y, rec = inputs(g)
    folder = location(phase,g)
    meta = json.loads((folder/"manifest.json").read_text())
    expected = dict(graph=g, graph_sha256=rec["graph_sha256"],
                    factor_sha256=rec["factor_sha256"], starts=starts, steps=steps,
                    seed=SEED, vertices=len(y), rank=y.shape[1],
                    driver_sha256=source_sha(__file__),
                    sweep_sha256=source_sha("src/maxkcut/eva_direct_rounding.py"))
    if any(meta.get(k)!=v for k,v in expected.items()) or sha(folder/"trajectory.npz") != meta["archive_sha256"]:
        raise ValueError(f"{g}: incompatible checkpoint")
    with np.load(folder/"trajectory.npz",allow_pickle=False) as f:
        data = {k:f[k] for k in f.files}
    c, b = data["cuts"], data["best"]
    if c.shape!=(starts,2,steps+1) or not np.isfinite(c).all():
        raise ValueError("Invalid trace shape or scores")
    np.testing.assert_allclose(b,np.maximum.accumulate(c,axis=2),rtol=0,atol=1e-7)
    np.testing.assert_allclose(c[:,0,0],c[:,1,0],rtol=0,atol=1e-7)
    if np.any(np.diff(c[:,1,:],axis=1)<-1e-7):
        raise ValueError("Nonmonotone sweep")
    if np.any(data["seconds"]<0) or not np.isfinite(data["seconds"]).all():
        raise ValueError("Invalid times")
    # Independent text-edge evaluation, not the sweep objective implementation.
    path = Path(rec["graph_path"].replace("\\","/"))
    with path.open() as f:
        n,m = map(int,f.readline().split())
    rows = np.loadtxt(path,skiprows=1,ndmin=2)
    assert rows.shape==(m,3) and n==len(y)
    i,j = rows[:,:2].astype(np.int64).T-1
    w = rows[:,2]
    for s in range(starts):
        for arm in range(2):
            nn = data["normals"][s,arm]
            np.testing.assert_allclose(np.linalg.norm(nn,axis=1),1.,atol=1e-9,rtol=0)
            # Reconstruct using the same vector dot product, and independently
            # rescore every archived candidate, including non-improving draws.
            for t in range(steps+1):
                x = np.unpackbits(data["labels"][s,arm,t])[:n]
                q = y @ nn[t]
                mismatch = (q>0)!=x
                if np.any(np.abs(q[mismatch])>1e-10):
                    raise ValueError(f"{g}: normal/partition mismatch")
                if abs(float(w[x[i]!=x[j]].sum())-c[s,arm,t])>1e-7:
                    raise ValueError(f"{g}: independent cut mismatch")
    return data,meta


def compute(job):
    g,phase,starts,steps=job
    with threadpool_limits(limits=1):
        folder=location(phase,g)
        if (folder/"manifest.json").exists():
            verify(g,phase,starts,steps)
            return g,"verified existing"
        graph,y,rec=inputs(g)
        warmup()
        all_trials=[]
        for s in range(starts):
            # Match initial normals from E024/E025; new proposal stream for E033.
            initial = unit(np.random.default_rng(np.random.SeedSequence(
                [SEED,int(g[1:]),2401,s])).standard_normal(y.shape[1]))
            rng=np.random.default_rng(np.random.SeedSequence([SEED,33,int(g[1:]),s]))
            all_trials.append(trial(graph,y,initial,rng.standard_normal((steps,y.shape[1]))))
        folder.mkdir(parents=True,exist_ok=True)
        tmp=folder/"trajectory.tmp.npz"
        np.savez_compressed(tmp,**{k:np.stack([a[k] for a in all_trials]) for k in all_trials[0]})
        os.replace(tmp,folder/"trajectory.npz")
        meta=dict(graph=g,graph_sha256=rec["graph_sha256"],factor_sha256=rec["factor_sha256"],
                  factor_source=rec["factor_source"],starts=starts,steps=steps,seed=SEED,
                  vertices=len(y),rank=y.shape[1],driver_sha256=source_sha(__file__),
                  sweep_sha256=source_sha("src/maxkcut/eva_direct_rounding.py"),
                  archive_sha256=sha(folder/"trajectory.npz"))
        tmp=folder/"manifest.tmp"
        tmp.write_text(json.dumps(meta,indent=2)+"\n",encoding="utf8")
        os.replace(tmp,folder/"manifest.json")
        verify(g,phase,starts,steps)
        return g,"computed and rescored"


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--phase",choices=["pilot","full"],default="full")
    ap.add_argument("--action",choices=["compute","verify","validate","test"],default="compute")
    ap.add_argument("--workers",type=int,default=8)
    args=ap.parse_args()
    if args.action=="test":
        warmup()
        rng=np.random.default_rng(17)
        y=rng.normal(size=(12,5));y/=np.linalg.norm(y,axis=1)[:,None]
        a=np.triu(rng.integers(-3,6,size=(12,12)),1)
        g=sparse.csr_matrix(a+a.T,dtype=float)
        r=unit(rng.normal(size=5))
        d=trial(g,y,r,rng.normal(size=(8,5)))
        assert d["cuts"].shape==(2,9)
        assert np.all(np.diff(d["best"],axis=1)>=-1e-8)
        assert np.all(np.diff(d["cuts"][1])>=-1e-8)
        assert d["cuts"][0,0]==d["cuts"][1,0]
        for arm in range(2):
            for t in range(9):
                x=np.unpackbits(d["labels"][arm,t])[:len(y)]
                assert np.array_equal(x,y@d["normals"][arm,t]>0)
                assert d["cuts"][arm,t]==score(edge_arrays(g),x)
        print("PASS signed synthetic trajectories");return
    graphs,starts,steps=(PILOT,2,10) if args.phase=="pilot" else (GRAPHS,STARTS,STEPS)
    if args.action=="validate":
        for g in GRAPHS:
            inputs(g)
        print("PASS all 71 frozen graph/factor identities");return
    if args.action=="verify":
        for g in graphs:
            verify(g,args.phase,starts,steps)
        print(f"PASS {len(graphs)*starts*2*(steps+1)} signed candidate cuts");return
    # Serial precompilation prevents concurrent Numba cache writes.
    warmup()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(compute,(g,args.phase,starts,steps)) for g in graphs]
        for f in as_completed(jobs):
            print(*f.result(),flush=True)


if __name__=="__main__":
    main()
