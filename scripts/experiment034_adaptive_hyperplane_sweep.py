"""E034: extend audited E033 trajectories without repeating the first 100 draws."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
import experiment033_raw_hyperplane_sweep as e33

ROOT = Path("results/gset/experiment034-adaptive-hyperplane-sweep")
PATIENCE, LIMIT, EPS = 5, 1000, 1e-8
FIELDS = ("normals", "labels", "cuts", "best", "seconds", "intervals", "valid", "angles", "predicted")


def stopping(best, patience=PATIENCE):
    """First joint patience hit, with history starting at the E033 endpoint."""
    stale = 0
    for i in range(1, len(best)):
        stale = 0 if np.any(best[i] > best[i-1] + EPS) else stale + 1
        if stale == patience:
            return i
    return None


def extend(graph, y, g, s, prefix):
    # The E033 sweep retains the FIRST strict-best normal, not the last tied one.
    b = prefix["best"][s, :, -1].copy()
    r = prefix["normals"][s, 1, np.argmax(prefix["cuts"][s, 1])].copy()
    rng = np.random.default_rng(np.random.SeedSequence([e33.SEED, 33, int(g[1:]), s]))
    rng.standard_normal((e33.STEPS, y.shape[1]))  # advance RNG only; no old cuts
    edges = e33.edge_arrays(graph)
    records = {k: [] for k in FIELDS}
    stale = 0
    for t in range(e33.STEPS + 1, LIMIT + 1):
        z = rng.standard_normal(y.shape[1])
        clock = time.perf_counter()
        gw = e33.unit(z)
        gx = (y @ gw > 0).astype(np.uint8)
        gc = e33.score(edges, gx)
        gt = time.perf_counter() - clock
        clock = time.perf_counter()
        u = z - (z @ r) * r
        if np.linalg.norm(u) <= 1e-13:
            u = np.eye(len(r))[np.argmin(np.abs(r))]
            u -= (u @ r) * r
        rr, sx, info = e33.optimize_semicircle(graph, y, r, e33.unit(u))
        sc = e33.score(edges, sx)
        st = time.perf_counter() - clock
        if abs(sc-info["realized_arc_cut"]) > 1e-7 or sc < b[1]-1e-7:
            raise ValueError("Sweep score or monotonicity failure")
        if sc > b[1] + EPS:
            r = rr.copy()
        candidate = np.array([gc, sc])
        next_b = np.maximum(b, candidate)
        stale = 0 if np.any(next_b > b + EPS) else stale + 1
        b = next_b
        values = (np.stack([gw, rr]), np.stack([np.packbits(gx), np.packbits(sx)]),
                  candidate, b.copy(), [gt, st], [1, info["distinct_events"]+1],
                  info["numerically_valid"], info["theta"], info["predicted_arc_max"])
        for key, value in zip(FIELDS, values):
            records[key].append(value)
        if stale == PATIENCE:
            break
    return {k: np.asarray(v) for k, v in records.items()}


def folder(phase, g):
    return ROOT / phase / "graphs" / g


def identity(g, phase, starts, prefix_meta):
    return dict(graph=g, phase=phase, starts=starts, patience=PATIENCE, max_total_draws=LIMIT,
                prefix_steps=e33.STEPS, prefix_archive_sha256=prefix_meta["archive_sha256"],
                graph_sha256=prefix_meta["graph_sha256"], factor_sha256=prefix_meta["factor_sha256"],
                driver_sha256=e33.source_sha(__file__), prefix_driver_sha256=e33.source_sha(e33.__file__),
                sweep_sha256=e33.source_sha("src/maxkcut/eva_direct_rounding.py"))


def verify_extension(g, phase, starts, graph, y, rec, prefix, prefix_meta):
    out = folder(phase, g)
    meta = json.loads((out/"manifest.json").read_text(encoding="utf8"))
    expected = identity(g, phase, starts, prefix_meta)
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError(f"{g}: checkpoint identity mismatch")
    if e33.sha(out/"continuation.npz") != meta["archive_sha256"]:
        raise ValueError(f"{g}: continuation archive hash mismatch")
    with np.load(out/"continuation.npz", allow_pickle=False) as data:
        all_runs = [{k: data[f"{s:02d}_{k}"] for k in FIELDS} for s in range(starts)]
    path = Path(rec["graph_path"].replace("\\", "/"))
    rows = np.loadtxt(path, skiprows=1, ndmin=2)
    i, j = rows[:, :2].astype(np.int64).T - 1
    w = rows[:, 2]
    for s, d in enumerate(all_runs):
        length = len(d["cuts"])
        if not PATIENCE <= length <= LIMIT-e33.STEPS:
            raise ValueError("Invalid continuation length")
        if d["cuts"].shape != (length, 2) or not np.isfinite(d["cuts"]).all():
            raise ValueError("Invalid cuts")
        if any(len(v) != length for v in d.values()):
            raise ValueError("Inconsistent trace lengths")
        if not np.isfinite(d["seconds"]).all() or np.any(d["seconds"] < 0):
            raise ValueError("Invalid timings")
        initial = prefix["best"][s, :, -1]
        best = np.maximum.accumulate(np.vstack([initial, d["cuts"]]), axis=0)
        np.testing.assert_allclose(d["best"], best[1:], atol=1e-7, rtol=0)
        if np.any(d["cuts"][:, 1] < best[:-1, 1]-1e-7):
            raise ValueError("Sweep lost incumbent")
        hit = stopping(best)
        if hit != length and not (hit is None and length == LIMIT-e33.STEPS):
            raise ValueError("Invalid stopping rule")
        rng = np.random.default_rng(np.random.SeedSequence([e33.SEED,33,int(g[1:]),s]))
        rng.standard_normal((e33.STEPS, y.shape[1]))
        for t in range(length):
            z = rng.standard_normal(y.shape[1])
            np.testing.assert_allclose(d["normals"][t,0], e33.unit(z), atol=1e-12, rtol=0)
            for arm in range(2):
                normal = d["normals"][t,arm]
                np.testing.assert_allclose(np.linalg.norm(normal),1.,atol=1e-9,rtol=0)
                x = np.unpackbits(d["labels"][t,arm])[:len(y)]
                q = y @ normal
                if np.any(np.abs(q[(q>0)!=x]) > 1e-10):
                    raise ValueError("Normal/partition mismatch")
                if abs(float(w[x[i]!=x[j]].sum()) - d["cuts"][t,arm]) > 1e-7:
                    raise ValueError("Independent signed-cut audit failed")
    return all_runs


def load_prefix(g):
    prefix, meta = e33.verify(g, "full", e33.STARTS, e33.STEPS)
    graph, y, rec = e33.inputs(g)
    return graph, y, rec, prefix, meta


def compute(job):
    g, phase, starts = job
    with threadpool_limits(limits=1):
        graph, y, rec, prefix, meta = load_prefix(g)
        out = folder(phase,g)
        if not (out/"manifest.json").exists():
            e33.warmup()
            all_runs = [extend(graph,y,g,s,prefix) for s in range(starts)]
            out.mkdir(parents=True, exist_ok=True)
            tmp = out/"continuation.tmp.npz"
            np.savez_compressed(tmp, **{f"{s:02d}_{k}": v for s,d in enumerate(all_runs) for k,v in d.items()})
            os.replace(tmp,out/"continuation.npz")
            manifest = identity(g,phase,starts,meta)
            manifest["archive_sha256"] = e33.sha(out/"continuation.npz")
            tmp = out/"manifest.tmp"
            tmp.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf8")
            os.replace(tmp,out/"manifest.json")
        runs = verify_extension(g,phase,starts,graph,y,rec,prefix,meta)
        return g, [len(d["cuts"]) for d in runs]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    per, seeds, trace, curves = [], [], [], []
    root = ROOT/"full"
    for g in e33.GRAPHS:
        graph,y,rec,prefix,meta = load_prefix(g)
        runs = verify_extension(g,"full",e33.STARTS,graph,y,rec,prefix,meta)
        baseline = float(prefix["best"][:,0,0].mean())
        if baseline <= 0:
            raise ValueError("Nonpositive E033 normalization baseline")
        before = prefix["best"][:,:,-1]
        final = np.stack([d["best"][-1] for d in runs])
        gains = 100*(final-before)/baseline
        initial = prefix["best"][:,0,0]
        total = 100*(final-initial[:,None])/baseline
        durations = np.stack([d["seconds"].sum(axis=0) for d in runs])
        lengths = np.array([len(d["cuts"]) for d in runs])
        cap = [stopping(np.vstack([before[s],d["best"]])) is None for s,d in enumerate(runs)]
        per.append(dict(graph=g,gw_extra_gain_pct=float(gains[:,0].mean()),
                        sweep_extra_gain_pct=float(gains[:,1].mean()),
                        gw_total_gain_pct=float(total[:,0].mean()),sweep_total_gain_pct=float(total[:,1].mean()),
                        gw_final_mean=float(final[:,0].mean()),sweep_final_mean=float(final[:,1].mean()),
                        mean_extra_draws=float(lengths.mean()),max_extra_draws=int(lengths.max()),
                        cap_hits=sum(cap),gw_extra_seconds=float(durations[:,0].mean()),
                        sweep_extra_seconds=float(durations[:,1].mean())))
        padded=[]
        for s,d in enumerate(runs):
            seeds.append(dict(graph=g,seed=s,extra_draws=int(lengths[s]),
                              stop_reason="cap" if cap[s] else "joint_patience",
                              gw_at_100=float(before[s,0]),sweep_at_100=float(before[s,1]),
                              gw_final=float(final[s,0]),sweep_final=float(final[s,1]),
                              gw_extra_seconds=float(durations[s,0]),sweep_extra_seconds=float(durations[s,1]),
                              sweep_intervals=int(d["intervals"][:,1].sum()),
                              numerical_fallbacks=int((~d["valid"]).sum())))
            for t in range(lengths[s]):
                trace.append(dict(graph=g,seed=s,iteration=e33.STEPS+t+1,
                                  gw_candidate=float(d["cuts"][t,0]),sweep_candidate=float(d["cuts"][t,1]),
                                  gw_best=float(d["best"][t,0]),sweep_best=float(d["best"][t,1]),
                                  gw_seconds=float(d["seconds"][t,0]),sweep_seconds=float(d["seconds"][t,1]),
                                  sweep_intervals=int(d["intervals"][t,1]),sweep_valid=bool(d["valid"][t])))
            history=np.concatenate([prefix["best"][s].T,d["best"]],axis=0)
            history=np.pad(history,((0,LIMIT+1-len(history)),(0,0)),mode="edge")
            padded.append(100*(history-initial[s])/baseline)
        curves.append(np.mean(padded,axis=0))
        print("Audited",g,flush=True)
    write_csv(root/"per_graph.csv",per)
    write_csv(root/"per_seed.csv",seeds)
    write_csv(root/"iterations.csv",trace)
    effects=np.array([p["sweep_total_gain_pct"]-p["gw_total_gain_pct"] for p in per])
    rng=np.random.default_rng(3401)
    ci=np.quantile(effects[rng.integers(0,len(effects),size=(10000,len(effects)))].mean(axis=1),[.025,.975])
    steps=np.array([s["extra_draws"] for s in seeds])
    summary=dict(complete=True,graphs=len(per),paired_trajectories=len(seeds),
                 new_eva_solves=0,reused_prefix_records=len(seeds)*2*(e33.STEPS+1),
                 new_candidate_records=2*int(steps.sum()),patience=PATIENCE,max_total_draws=LIMIT,
                 min_extra_draws=int(steps.min()),median_extra_draws=float(np.median(steps)),
                 mean_extra_draws=float(steps.mean()),max_extra_draws=int(steps.max()),
                 cap_hits=sum(s["stop_reason"]=="cap" for s in seeds),
                 graph_wins=int((effects>1e-9).sum()),graph_ties=int((abs(effects)<=1e-9).sum()),
                 graph_losses=int((effects<-1e-9).sum()),mean_paired_gain_pct=float(effects.mean()),
                 graph_bootstrap_95_pct=ci.tolist(),
                 numerical_fallbacks=sum(s["numerical_fallbacks"] for s in seeds))
    for k in ("gw_extra_gain_pct","sweep_extra_gain_pct","gw_total_gain_pct","sweep_total_gain_pct",
              "gw_extra_seconds","sweep_extra_seconds"):
        summary["mean_"+k]=float(np.mean([p[k] for p in per]))
    (root/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf8")
    a=np.array(curves)
    end=e33.STEPS+int(steps.max())
    fig,ax=plt.subplots(figsize=(7.5,4))
    ax.plot(a[:,:end+1,0].mean(axis=0),label="Repeated GW")
    ax.plot(a[:,:end+1,1].mean(axis=0),label="Hyperplane Sweep")
    ax.axvline(e33.STEPS,color="grey",linestyle="--",label="E033 endpoint")
    ax.set(xlabel="Gaussian draws (stopped curves held constant)",ylabel="Mean gain from initial GW (%)")
    ax.legend();fig.tight_layout();fig.savefig(root/"convergence.pdf");plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4))
    ax.hist(steps,bins=min(30,max(1,int(steps.max()-steps.min()+1))))
    ax.set(xlabel="Additional draws until stop",ylabel="Paired starts")
    fig.tight_layout();fig.savefig(root/"stopping_distribution.pdf");plt.close(fig)
    with PdfPages(root/"all_71_curves.pdf") as pdf:
        for first in range(0,len(per),12):
            fig,axes=plt.subplots(4,3,figsize=(8.27,11.69))
            for j,ax in enumerate(axes.flat):
                idx=first+j
                if idx>=len(per):
                    ax.axis("off");continue
                stop=e33.STEPS+per[idx]["max_extra_draws"]
                ax.plot(a[idx,:stop+1,0],label="GW")
                ax.plot(a[idx,:stop+1,1],label="Sweep")
                ax.axvline(e33.STEPS,color="grey",linestyle="--")
                ax.set_title(per[idx]["graph"])
                if j==0: ax.legend(fontsize=7)
            fig.supxlabel("Draws");fig.supylabel("Gain (%)")
            fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    md = "# Experiment034: paired continuation with adaptive stopping\n\n"
    md += "All original E033 prefixes independently verified and reused. Both arms receive identical additional Gaussian draws; five consecutive draws without improvement by either arm stop a trajectory. A cap of 1,000 total draws protects resources. This is a heuristic plateau, not a convergence proof. Equal draws are not equal time or equal cut evaluations.\n\n"
    md += "| Metric | Result |\n|---|---:|\n" + "".join(f"| {k} | {v} |\n" for k,v in summary.items())
    (root/"report.md").write_text(md,encoding="utf8")
    print(json.dumps(summary,indent=2))


def test():
    assert stopping(np.array([[10,10],[11,10],[11,10],[11,10],[11,10],[11,10],[11,10]])) == 6
    assert stopping(np.array([[10,10],[10,10],[10,11],[10,11],[10,11],[10,11]])) is None
    assert stopping(np.ones((6,2))) == 5
    rng=np.random.default_rng(4)
    y=rng.normal(size=(12,5));y/=np.linalg.norm(y,axis=1)[:,None]
    from scipy import sparse
    a=np.triu(rng.integers(-3,6,size=(12,12)),1)
    graph=sparse.csr_matrix(a+a.T,dtype=float)
    r=e33.unit(rng.normal(size=5))
    d=e33.trial(graph,y,r,rng.normal(size=(e33.STEPS,5)))
    prefix={k:v[None] for k,v in d.items()}
    ext=extend(graph,y,"G1",0,prefix)
    b=np.vstack([prefix["best"][0,:,-1],ext["best"]])
    assert stopping(b)==len(ext["cuts"]) or len(ext["cuts"])==LIMIT-e33.STEPS
    for t in range(len(ext["cuts"])):
        for arm in range(2):
            labels=np.unpackbits(ext["labels"][t,arm])[:len(y)]
            assert ext["cuts"][t,arm]==e33.score(e33.edge_arrays(graph),labels)
    print("PASS adaptive stopping and signed continuation")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=["pilot","full"],default="full")
    parser.add_argument("--action",choices=["compute","test","report"],default="compute")
    parser.add_argument("--workers",type=int,default=8)
    args=parser.parse_args()
    if args.action=="test":
        test();return
    if args.action=="report":
        report();return
    graphs,starts=(e33.PILOT,2) if args.phase=="pilot" else (e33.GRAPHS,e33.STARTS)
    e33.warmup()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(compute,(g,args.phase,starts)) for g in graphs]):
            print(*f.result(),flush=True)


if __name__=="__main__":
    main()
