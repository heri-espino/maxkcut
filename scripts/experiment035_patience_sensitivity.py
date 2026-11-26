"""E035: reuse E033/E034 and continue paired searches to patience 100."""
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
import experiment034_adaptive_hyperplane_sweep as e34

ROOT = Path("results/gset/experiment035-patience-sensitivity")
PATIENCES = (5, 10, 20, 50, 100)
PATIENCE, LIMIT, EPS = max(PATIENCES), 1000, 1e-8
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
    old = prefix["_prior"][s]
    # Recover the first strict-best sweep normal and the existing failure streak.
    stale = 0
    for t, candidate in enumerate(old["cuts"]):
        if candidate[1] > b[1] + EPS:
            r = old["normals"][t, 1].copy()
        next_b = np.maximum(b, candidate)
        stale = 0 if np.any(next_b > b + EPS) else stale + 1
        b = next_b
    np.testing.assert_allclose(b, old["best"][-1], atol=1e-7, rtol=0)
    rng.standard_normal((e33.STEPS + len(old["cuts"]), y.shape[1]))
    edges = e33.edge_arrays(graph)
    records = {k: list(old[k]) for k in FIELDS}
    for t in range(e33.STEPS + len(old["cuts"]) + 1, LIMIT + 1):
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
                e34_archive_sha256=prefix_meta["e34_archive_sha256"],
                e34_driver_sha256=e33.source_sha(e34.__file__),
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
        old = prefix["_prior"][s]
        for k in FIELDS:
            np.testing.assert_array_equal(d[k][:len(old[k])], old[k])
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
    prior = e34.verify_extension(g, "full", e33.STARTS, graph, y, rec, prefix, meta)
    prefix["_prior"] = prior
    meta = dict(meta, e34_archive_sha256=e33.sha(e34.folder("full", g)/"continuation.npz"))
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
            tmp.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf8",newline="\n")
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
    per, seeds, trace, curves, policies = [], [], [], [], []
    reused_e34 = 0
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
            old = prefix["_prior"][s]
            reused_e34 += len(old["cuts"]) * 2
            history_best = np.vstack([before[s], d["best"]])
            increments = np.diff(history_best, axis=0)
            cumulative_time = np.cumsum(d["seconds"], axis=0)
            cumulative_intervals = np.cumsum(d["intervals"], axis=0)
            for patience in PATIENCES:
                hit = stopping(history_best, patience)
                n = hit if hit is not None else len(d["cuts"])
                if patience == 5:
                    assert n == len(old["cuts"])
                final_p = d["best"][n-1]
                delta = increments[:n]
                policies.append(dict(graph=g, seed=s, patience=patience,
                    stop_iteration=e33.STEPS+n, extra_draws=n, cap_hit=hit is None,
                    gw_gain_pct=float(100*(final_p[0]-initial[s])/baseline),
                    sweep_gain_pct=float(100*(final_p[1]-initial[s])/baseline),
                    gw_extra_gain_pct=float(100*(final_p[0]-before[s,0])/baseline),
                    sweep_extra_gain_pct=float(100*(final_p[1]-before[s,1])/baseline),
                    gw_final=float(final_p[0]), sweep_final=float(final_p[1]),
                    gw_seconds=float(cumulative_time[n-1,0]), sweep_seconds=float(cumulative_time[n-1,1]),
                    gw_records=int((delta[:,0]>EPS).sum()), sweep_records=int((delta[:,1]>EPS).sum()),
                    joint_records=int(np.any(delta>EPS,axis=1).sum()),
                    sweep_intervals=int(cumulative_intervals[n-1,1])))
            stale = np.zeros(3,dtype=int)
            incumbent = [prefix["labels"][s,a,np.argmax(prefix["cuts"][s,a])].copy() for a in range(2)]
            incumbent_normal = [prefix["normals"][s,a,np.argmax(prefix["cuts"][s,a])].copy() for a in range(2)]
            seeds.append(dict(graph=g,seed=s,extra_draws=int(lengths[s]),
                              stop_reason="cap" if cap[s] else "joint_patience",
                              gw_at_100=float(before[s,0]),sweep_at_100=float(before[s,1]),
                              gw_final=float(final[s,0]),sweep_final=float(final[s,1]),
                              gw_extra_seconds=float(durations[s,0]),sweep_extra_seconds=float(durations[s,1]),
                              sweep_intervals=int(d["intervals"][:,1].sum()),
                              numerical_fallbacks=int((~d["valid"]).sum())))
            for t in range(lengths[s]):
                gain = increments[t]
                flags = gain > EPS
                stale = np.where(np.r_[flags, flags.any()],0,stale+1)
                turnover, distance, balance = [], [], []
                for arm in range(2):
                    bits = np.unpackbits(d["labels"][t,arm])[:len(y)]
                    prev = np.unpackbits(incumbent[arm])[:len(y)]
                    h = float(np.mean(bits != prev))
                    turnover.append(min(h,1-h))
                    cosine = abs(float(d["normals"][t,arm] @ incumbent_normal[arm]))
                    distance.append(float(np.arccos(np.clip(cosine,0,1))))
                    fraction = float(bits.mean())
                    balance.append(min(fraction,1-fraction))
                    if flags[arm]:
                        incumbent[arm] = d["labels"][t,arm].copy()
                        incumbent_normal[arm] = d["normals"][t,arm].copy()
                trace.append(dict(graph=g,seed=s,iteration=e33.STEPS+t+1,
                                  reused_e34=t<len(old["cuts"]),
                                  gw_delta=float(gain[0]),sweep_delta=float(gain[1]),
                                  gw_delta_pct=float(100*gain[0]/baseline),sweep_delta_pct=float(100*gain[1]/baseline),
                                  gw_improved=bool(flags[0]),sweep_improved=bool(flags[1]),joint_improved=bool(flags.any()),
                                  gw_stale=int(stale[0]),sweep_stale=int(stale[1]),joint_stale=int(stale[2]),
                                  gw_turnover=turnover[0],sweep_turnover=turnover[1],
                                  gw_projective_angle=distance[0],sweep_projective_angle=distance[1],
                                  gw_balance=balance[0],sweep_balance=balance[1],
                                  gw_cumulative_seconds=float(cumulative_time[t,0]),sweep_cumulative_seconds=float(cumulative_time[t,1]),
                                  sweep_theta=float(d["angles"][t]),sweep_predicted=float(d["predicted"][t]),
                                  sweep_prediction_gap=float(d["predicted"][t]-d["cuts"][t,1]),
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
    write_csv(root/"patience_per_seed.csv",policies)
    analyze_policies(root, policies, trace)
    effects=np.array([p["sweep_total_gain_pct"]-p["gw_total_gain_pct"] for p in per])
    rng=np.random.default_rng(3401)
    ci=np.quantile(effects[rng.integers(0,len(effects),size=(10000,len(effects)))].mean(axis=1),[.025,.975])
    steps=np.array([s["extra_draws"] for s in seeds])
    summary=dict(complete=True,graphs=len(per),paired_trajectories=len(seeds),
                 new_eva_solves=0,reused_prefix_records=len(seeds)*2*(e33.STEPS+1),
                 reused_e34_records=reused_e34,
                 new_candidate_records=2*int(steps.sum())-reused_e34,patience=PATIENCE,max_total_draws=LIMIT,
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
    md = "# Experiment035: paired patience sensitivity and convergence diagnostics\n\n"
    md += "All E033 and E034 prefixes independently verified and reused. Both arms receive identical additional Gaussian draws; 100 consecutive draws without improvement by either arm stop a trajectory. Smaller patience policies are exact retrospective prefixes. A cap of 1,000 total draws protects resources. This is a heuristic plateau, not a convergence proof. Equal draws are not equal time or equal cut evaluations.\n\n"
    md += "| Metric | Result |\n|---|---:|\n" + "".join(f"| {k} | {v} |\n" for k,v in summary.items())
    (root/"report.md").write_text(md,encoding="utf8")
    print(json.dumps(summary,indent=2))


def analyze_policies(root, policies, trace):
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import statistics
    by_graph, overall = [], []
    keys = ("extra_draws","gw_gain_pct","sweep_gain_pct","gw_extra_gain_pct",
            "sweep_extra_gain_pct","gw_seconds","sweep_seconds","gw_records",
            "sweep_records","joint_records","sweep_intervals")
    rng = np.random.default_rng(3501)
    for p in PATIENCES:
        group = [r for r in policies if r["patience"] == p]
        graph_rows = []
        for g in e33.GRAPHS:
            rows = [r for r in group if r["graph"] == g]
            row = dict(graph=g,patience=p,cap_hits=sum(r["cap_hit"] for r in rows))
            row.update({k:statistics.mean(r[k] for r in rows) for k in keys})
            graph_rows.append(row)
        by_graph.extend(graph_rows)
        effects = np.array([r["sweep_gain_pct"]-r["gw_gain_pct"] for r in graph_rows])
        ci = np.quantile(effects[rng.integers(0,len(effects),size=(10000,len(effects)))].mean(axis=1),[.025,.975])
        out = dict(patience=p,trajectories=len(group),cap_hits=sum(r["cap_hit"] for r in group),
                   median_draws=float(np.median([r["extra_draws"] for r in group])),
                   max_draws=max(r["extra_draws"] for r in group),
                   paired_advantage_pct=float(effects.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]))
        out.update({k:statistics.mean(r[k] for r in graph_rows) for k in keys})
        overall.append(out)
    write_csv(root/"patience_per_graph.csv",by_graph)
    write_csv(root/"patience_summary.csv",overall)
    buckets = defaultdict(list)
    hazards=defaultdict(lambda:[0,0])
    previous={}
    for r in trace:
        buckets[r["iteration"]].append(r)
        key=(r["graph"],r["seed"])
        for arm in ("gw","sweep","joint"):
            k=key+(arm,)
            streak=previous.get(k,0)
            h=hazards[(arm,streak)]
            h[0]+=1
            h[1]+=int(r[arm+"_improved"])
            previous[k]=r[arm+"_stale"]
    hazard_rows=[dict(method=arm,prior_stale=stale,attempts=v[0],improvements=v[1],
                      probability=v[1]/v[0]) for (arm,stale),v in sorted(hazards.items())]
    write_csv(root/"improvement_hazard.csv",hazard_rows)
    iter_rows=[]
    for t,rows in sorted(buckets.items()):
        row=dict(iteration=t,active_trajectories=len(rows))
        for arm in ("gw","sweep"):
            for field in ("delta_pct","improved","turnover","projective_angle","balance","seconds"):
                row[arm+"_"+field]=float(np.mean([r[arm+"_"+field] for r in rows]))
        row["joint_improved"]=float(np.mean([r["joint_improved"] for r in rows]))
        iter_rows.append(row)
    write_csv(root/"iteration_diagnostics.csv",iter_rows)
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for arm,label in (("gw","Repeated GW"),("sweep","Hyperplane Sweep")):
        axes[0,0].plot(PATIENCES,[r[arm+"_gain_pct"] for r in overall],"o-",label=label)
        axes[0,1].plot([r[arm+"_seconds"] for r in overall],[r[arm+"_gain_pct"] for r in overall],"o-",label=label)
        for r in overall:
            axes[0,1].annotate(str(r["patience"]),(r[arm+"_seconds"],r[arm+"_gain_pct"]),fontsize=7)
        axes[1,0].plot([r["iteration"] for r in iter_rows],[r[arm+"_improved"] for r in iter_rows],label=label)
        hr=[r for r in hazard_rows if r["method"]==arm and r["attempts"]>=30]
        axes[1,1].plot([r["prior_stale"] for r in hr],[r["probability"] for r in hr],label=label)
    axes[0,0].set(xscale="log",xlabel="Patience",ylabel="Mean gain from initial GW (%)")
    axes[0,0].set_xticks(PATIENCES,labels=[str(p) for p in PATIENCES])
    axes[0,1].set(xlabel="Mean additional method seconds",ylabel="Mean gain (%)")
    axes[1,0].set(xlabel="Draw (active trajectories only)",ylabel="Strict improvement fraction")
    axes[1,1].set(xlabel="Prior failures (at least 30 attempts)",ylabel="Conditional improvement fraction")
    for ax in axes.flat: ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(root/"patience_diagnostics.pdf");plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for arm,label in (("gw","Repeated GW"),("sweep","Hyperplane Sweep")):
        for ax,field in zip(axes.flat,("delta_pct","turnover","projective_angle","balance")):
            ax.plot([r["iteration"] for r in iter_rows],[r[arm+"_"+field] for r in iter_rows],label=label)
    for ax,field in zip(axes.flat,("Marginal gain (%)","Complement-invariant turnover","Projective angle (rad)","Minority-side fraction")):
        ax.set(xlabel="Draw (active trajectories only)",ylabel=field);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(root/"geometry_diagnostics.pdf");plt.close(fig)
    print("Patience summary",json.dumps(overall,indent=2),flush=True)


def test():
    assert stopping(np.array([[10,10],[11,10],[11,10],[11,10],[11,10],[11,10],[11,10]]),5) == 6
    assert stopping(np.array([[10,10],[10,10],[10,11],[10,11],[10,11],[10,11]])) is None
    assert stopping(np.ones((6,2)),5) == 5
    assert stopping(np.ones((101,2)),100) == 100
    rng=np.random.default_rng(4)
    y=rng.normal(size=(12,5));y/=np.linalg.norm(y,axis=1)[:,None]
    from scipy import sparse
    a=np.triu(rng.integers(-3,6,size=(12,12)),1)
    graph=sparse.csr_matrix(a+a.T,dtype=float)
    r=e33.unit(rng.normal(size=5))
    d=e33.trial(graph,y,r,rng.normal(size=(e33.STEPS,5)))
    prefix={k:v[None] for k,v in d.items()}
    prefix["_prior"] = [e34.extend(graph,y,"G1",0,prefix)]
    ext=extend(graph,y,"G1",0,prefix)
    for k in FIELDS:
        np.testing.assert_array_equal(ext[k][:len(prefix["_prior"][0][k])],prefix["_prior"][0][k])
    b=np.vstack([prefix["best"][0,:,-1],ext["best"]])
    assert stopping(b)==len(ext["cuts"]) or len(ext["cuts"])==LIMIT-e33.STEPS
    for t in range(len(ext["cuts"])):
        for arm in range(2):
            labels=np.unpackbits(ext["labels"][t,arm])[:len(y)]
            assert ext["cuts"][t,arm]==e33.score(e33.edge_arrays(graph),labels)
    print("PASS exact E034 prefix reuse, adaptive stopping and signed continuation")


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
