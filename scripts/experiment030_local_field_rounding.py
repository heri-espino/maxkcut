"""Experiment030: compare weighted local-field spherical clusterings on frozen EVA factors.

NO EVA/SDP/graph optimization. All 1,000 factor arrays come unchanged from
Experiments028/029. Original signed Max-Cut scores and 100 paired seed IDs
are preserved. Save all clustering iterations, exact labels, and source SHA256.

Run from repository root with PYTHONPATH=src. Only touch experiment030 outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from maxkcut.eva import load_eva_factor
from maxkcut.objective import cut_weight
from maxkcut.rounding import round_spherical_kmeans
from experiment029_kmeansnm_100seeds import (GRAPHS, ALL_SEEDS, EX028, ROOT as EX029,
    load_verified_graph, properties, rng_for)

ROOT = Path('results/gset/experiment030-local-field-clustering')
MODES = ('default', 'full')
METHODS = ('baseline_y', 'unit_minus_h', 'raw_minus_h',
           'field_strength_y', 'coherence_y', 'abs_degree_y')
TRIAL_FIELDS = ('graph','rank_mode','p','seed','method','cut_weight','baseline_cut',
    'delta','partition_sha256','source_factor_sha256','source_experiment',
    'cluster_iterations','round_seconds','min_field_norm','max_field_norm',
    'mean_field_norm','mean_coherence','mean_cosine_to_minus_h',
    'min_cosine_to_minus_h','zero_field_count')
EPS = 1.e-15


def sha_file(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as fh:
        for block in iter(lambda:fh.read(1024*1024),b''):
            d.update(block)
    return d.hexdigest()


def archived_row_index(path=EX029/'all_runs.csv'):
    with Path(path).open(newline='',encoding='utf8') as f:
        rows=list(csv.DictReader(f))
    index={(r['graph'],int(r['seed']),r['rank_mode']):r for r in rows}
    if len(rows)!=1000 or len(index)!=1000:
        raise ValueError('Experiment029 baseline must provide exactly 1000 unique cases')
    return index


def load_exact_frozen(row,n,edges):
    factor=Path(row['factor_path'])
    part=Path(row['partition_path'])
    case=Path(row['case_path'])
    if not (factor.exists() and part.exists() and case.exists()):
        raise FileNotFoundError(f'Frozen factor, partition or case is missing: {factor}')
    if not str(factor).startswith(str(EX028)) and not str(factor).startswith(str(EX029)):
        raise ValueError('Reject foreign factor path')
    if sha_file(factor)!=row['factor_sha256']:
        raise ValueError(f'Frozen factor checksum mismatch: {factor}')
    y,state=load_eva_factor(factor)
    if y.shape!=(n,int(row['p'])) or not np.all(np.isfinite(y)):
        raise ValueError('Frozen matrix has unexpected dimensions or nonfinite entries')
    if np.max(np.abs(np.linalg.norm(y,axis=1)-1))>1e-6:
        raise ValueError('Frozen EVA factor is not unit row normalized')
    with np.load(part,allow_pickle=False) as archive:
        packed=np.asarray(archive['K-MeansNM_EVA'],dtype=np.uint8)
    if hashlib.sha256(packed.tobytes()).hexdigest()!=row['partition_sha256']:
        raise ValueError(f'Frozen partition checksum mismatch: {part}')
    labels=np.unpackbits(packed)[:n]
    score=float(cut_weight(edges,labels))
    if abs(score-float(row['cut_weight']))>1e-9:
        raise ValueError(f'Frozen original signed cut score mismatch {case}')
    metadata=json.loads(case.read_text(encoding='utf8'))
    if str(factor).startswith(str(EX028)):
        if metadata['factor_sha256']!=row['factor_sha256']:
            raise ValueError('Experiment028 case checksum mismatch')
    elif metadata['row']['factor_sha256']!=row['factor_sha256']:
        raise ValueError('Experiment029 case checksum mismatch')
    return y,labels,score


def derive_fields(a,y):
    """Return method vectors and diagnostics, with zero-field rows handled explicitly.

    The sign -H aligns direction with Y at a coordinate-ascent fixed point.
    Scaling all rows by a common positive number does not alter spherical KMeans.
    """
    h=np.asarray(a @ y,dtype=np.float64)
    d=np.linalg.norm(h,axis=1)
    deg=np.asarray(np.abs(a).sum(axis=1)).reshape(-1)
    if np.any(d>deg+1e-7*np.maximum(1.,deg)):
        raise ValueError('Triangle inequality violated: weighted field too large')
    nonzero=d>EPS
    u=y.copy()
    u[nonzero]=-h[nonzero]/d[nonzero,None]
    q=np.ones(len(d),dtype=np.float64)
    valid=deg>EPS
    q[valid]=d[valid]/deg[valid]
    q=np.clip(q,1e-12,1.0)
    # Row weights are positive, so the initial antipodal partition is unchanged.
    # Per-cluster centroids then become field-strength/degree weighted.
    strength=np.maximum(d,1e-12)
    abs_degree=np.maximum(deg,1e-12)
    vectors={
        'baseline_y':y,
        'unit_minus_h':u,
        'raw_minus_h':np.where(nonzero[:,None],-h,y),
        'field_strength_y':strength[:,None]*y,
        'coherence_y':q[:,None]*y,
        'abs_degree_y':abs_degree[:,None]*y,
    }
    cos=np.sum(y[nonzero]*u[nonzero],axis=1)
    diag={
       'min_field_norm':float(np.min(d)),
       'max_field_norm':float(np.max(d)),
       'mean_field_norm':float(np.mean(d)),
       'mean_coherence':float(np.mean(q)),
       'mean_cosine_to_minus_h':float(np.mean(cos)) if cos.size else float('nan'),
       'min_cosine_to_minus_h':float(np.min(cos)) if cos.size else float('nan'),
       'zero_field_count':int(np.count_nonzero(~nonzero)),
    }
    return vectors,diag


def cluster_with_trace(vectors,edges,seed,graph,max_iter=100):
    """Exact traced counterpart of repository round_spherical_kmeans(..., antipodal=True).

    Each iteration logs the original signed cut and best-so-far. The caller
    checks final labels against unchanged repository implementation for every case.
    """
    rng=rng_for(seed,graph)
    centers=rng.standard_normal((2,vectors.shape[1]))
    centers[1]=-centers[0]
    centers/=np.linalg.norm(centers,axis=1)[:,None]
    previous=None
    best_labels=None
    best_score=float('-inf')
    history=[]
    for iteration in range(1,max_iter+1):
        labels=np.argmax(vectors @ centers.T,axis=1)
        score=float(cut_weight(edges,labels))
        improved=score>best_score
        if improved:
            best_labels,best_score=labels.copy(),score
        stable=previous is not None and np.array_equal(labels,previous)
        history.append({'iteration':iteration,'cut_weight':score,'best_so_far':best_score,
                        'improved':bool(improved),'stable':bool(stable)})
        if stable:
            return best_labels,iteration,history
        previous=labels.copy()
        for c in range(2):
            members=vectors[labels==c]
            if not len(members):
                centers[c]=vectors[rng.integers(len(vectors))]
            else:
                centroid=np.sum(members,axis=0)
                magnitude=np.linalg.norm(centroid)
                centers[c]=centroid/magnitude if magnitude>1e-12 else vectors[rng.integers(len(vectors))]
    return best_labels,max_iter,history


def run_case(graph,seed,mode,a,edges,rows,raw_dir,trace_handle):
    row=rows[(graph,seed,mode)]
    n=a.shape[0]
    y,baseline_labels,baseline_score=load_exact_frozen(row,n,edges)
    vectors,diag=derive_fields(a,y)
    outputs=[]
    partitions={}
    for method in METHODS:
        data=vectors[method]
        if not np.all(np.isfinite(data)):
            raise ValueError('Nonfinite method representation')
        began=time.perf_counter()
        labels,iterations,history=cluster_with_trace(data,edges,seed,graph)
        elapsed=time.perf_counter()-began
        # Check the traced implementation against the actual published rounder.
        control,control_it=round_spherical_kmeans(data,2,rng_for(seed,graph),
            antipodal=True,edges=edges,keep_best=True)
        if iterations!=control_it or not np.array_equal(labels,control):
            raise AssertionError(f'Trace diverged from baseline K-MeansNM: {graph} {seed} {mode} {method}')
        if method=='baseline_y' and not (np.array_equal(labels,baseline_labels) and
                abs(float(cut_weight(edges,labels))-baseline_score)<1e-9):
            raise AssertionError(f'Cannot reproduce frozen baseline {graph} seed={seed} {mode}')
        packed=np.packbits(np.asarray(labels,dtype=np.uint8))
        sha=hashlib.sha256(packed.tobytes()).hexdigest()
        score=float(cut_weight(edges,labels))
        label_key=f'{mode}:{seed}:{method}'
        partitions[label_key]=packed
        record={'graph':graph,'rank_mode':mode,'p':int(row['p']),'seed':int(seed),
            'method':method,'cut_weight':score,'baseline_cut':baseline_score,
            'delta':score-baseline_score,'partition_sha256':sha,
            'source_factor_sha256':row['factor_sha256'],
            'source_experiment':row['source_experiment'],
            'cluster_iterations':int(iterations),'round_seconds':elapsed,**diag}
        outputs.append(record)
        for t in history:
            trace_handle.write(json.dumps({'graph':graph,'rank_mode':mode,'seed':seed,
                 'method':method,**t},sort_keys=True,separators=(',',':'))+'\n')
    return outputs,partitions


def do_graph(graph,output=ROOT,baseline=EX029,max_seeds=100):
    if max_seeds<1 or max_seeds>100:raise ValueError('max_seeds must be 1..100')
    props=properties()
    a,edges,meta=load_verified_graph(graph,props)
    allrows=archived_row_index(baseline/'all_runs.csv')
    out=Path(output)/'graphs'/graph
    out.mkdir(parents=True,exist_ok=True)
    files=[out/'cases.csv',out/'partitions.npz',out/'trajectories.jsonl',out/'manifest.json']
    if any(x.exists() for x in files):
        if not all(x.exists() for x in files):
            raise RuntimeError('Incomplete prior graph publication: preserve outputs; no overwrite')
        verify_graph(graph,output,baseline,expected_seed_count=max_seeds)
        print(f'REUSED existing validated Experiment030 graph {graph}',flush=True)
        return
    records=[];partitions={}
    with (out/'trajectories.jsonl').open('w',encoding='utf8',newline='\n') as trace:
        for mode in MODES:
            for seed in ALL_SEEDS[:max_seeds]:
                case,packed=run_case(graph,seed,mode,a,edges,allrows,out,trace)
                records.extend(case)
                partitions.update(packed)
                print(f'EX030 {graph} {mode} seed={seed}: '+', '.join(
                    f"{z['method']}={z['cut_weight']:.0f}" for z in case),flush=True)
    with (out/'cases.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=TRIAL_FIELDS,lineterminator='\n')
        w.writeheader();w.writerows(records)
    np.savez_compressed(out/'partitions.npz',**partitions)
    manifest={'schema':'experiment030-graph-v1','graph':graph,'ranks':list(MODES),
         'methods':list(METHODS),'seeds':list(ALL_SEEDS[:max_seeds]),
         'n':int(a.shape[0]),'source_graph_sha256':meta['graph_sha256'],
         'cases':len(records),'partitions':len(partitions),
         'cut_recompute':'original-signed-edges','factor_recompute':False,
         'case_csv_sha256':sha_file(out/'cases.csv'),
         'partition_archive_sha256':sha_file(out/'partitions.npz'),
         'trajectory_sha256':sha_file(out/'trajectories.jsonl')}
    (out/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n',encoding='utf8',newline='\n')
    verify_graph(graph,output,baseline,expected_seed_count=max_seeds)
    print(f'PASS EX030 {graph}: {len(records)} exact signed partitions archived',flush=True)


def verify_graph(graph,output=ROOT,baseline=EX029,expected_seed_count=100):
    props=properties()
    a,edges,meta=load_verified_graph(graph,props)
    d=Path(output)/'graphs'/graph
    manifest=json.loads((d/'manifest.json').read_text(encoding='utf8'))
    if manifest['graph']!=graph or manifest['source_graph_sha256']!=meta['graph_sha256']:
        raise AssertionError('Graph identity mismatch')
    for filename,field in [('cases.csv','case_csv_sha256'),('partitions.npz','partition_archive_sha256'),('trajectories.jsonl','trajectory_sha256')]:
        if sha_file(d/filename)!=manifest[field]:raise AssertionError(f'Changed archive: {filename}')
    with (d/'cases.csv').open(newline='',encoding='utf8') as f:
        cases=list(csv.DictReader(f))
    expected={(m,s,k) for m in MODES for s in ALL_SEEDS[:expected_seed_count] for k in METHODS}
    actual={(z['rank_mode'],int(z['seed']),z['method']) for z in cases}
    if len(cases)!=len(expected) or actual!=expected:
        raise AssertionError(f'Missing or repeated cases for {graph}')
    baseline_rows=archived_row_index(Path(baseline)/'all_runs.csv')
    with np.load(d/'partitions.npz',allow_pickle=False) as archive:
        if set(archive.files)!={f'{mode}:{seed}:{method}' for mode,seed,method in expected}:
            raise AssertionError('Missing packed partitions')
        for z in cases:
            key=f"{z['rank_mode']}:{z['seed']}:{z['method']}"
            packed=archive[key]
            if hashlib.sha256(packed.tobytes()).hexdigest()!=z['partition_sha256']:
                raise AssertionError('Partition packed SHA mismatch')
            bits=np.unpackbits(packed)[:a.shape[0]]
            actual_score=cut_weight(edges,bits)
            if not np.isclose(actual_score,float(z['cut_weight']),atol=1e-9,rtol=0):
                raise AssertionError('Partition score mismatch: '+key)
            prev=baseline_rows[(graph,int(z['seed']),z['rank_mode'])]
            if (prev['factor_sha256']!=z['source_factor_sha256'] or
                float(prev['cut_weight'])!=float(z['baseline_cut'])):
                raise AssertionError('Wrong historical reference: '+key)
            if z['method']=='baseline_y' and (z['partition_sha256']!=prev['partition_sha256'] or
                 float(z['cut_weight'])!=float(prev['cut_weight'])):
                raise AssertionError('Baseline labels changed: '+key)
    seen=set();entries=0;reconstructed={}
    with (d/'trajectories.jsonl').open('r',encoding='utf8') as f:
        for line in f:
            t=json.loads(line);idx=(t['rank_mode'],int(t['seed']),t['method'])
            if idx not in expected or int(t['iteration'])<1:raise AssertionError('Bad clustering trace identity')
            if idx in reconstructed and int(t['iteration'])!=reconstructed[idx][0]+1:
                raise AssertionError('Nonconsecutive trace')
            if idx not in reconstructed and int(t['iteration'])!=1:
                raise AssertionError('First trace iteration not 1')
            if idx in reconstructed and float(t['best_so_far'])<reconstructed[idx][1]:
                raise AssertionError('Best-so-far decreased')
            reconstructed[idx]=(int(t['iteration']),float(t['best_so_far']))
            entries+=1
    for z in cases:
        idx=(z['rank_mode'],int(z['seed']),z['method'])
        if idx not in reconstructed or reconstructed[idx][0]!=int(z['cluster_iterations']):
            raise AssertionError('Clustering iteration coverage mismatch')
        if abs(reconstructed[idx][1]-float(z['cut_weight']))>1e-9:
            raise AssertionError('Final best cut mismatch with trace')
    return {'graph':graph,'cases':len(cases),'trace_entries':entries,
        'baseline_exact':2*expected_seed_count,'verified':True}


def summarize(root=ROOT,baseline=EX029,bootstrap=5000,require_all=True):
    import numpy as np
    root=Path(root);seed_range=ALL_SEEDS
    rows=[];audits=[]
    for graph in GRAPHS:
        audits.append(verify_graph(graph,root,baseline,expected_seed_count=len(seed_range)))
        with (root/'graphs'/graph/'cases.csv').open(newline='',encoding='utf8') as f:
            rows.extend(csv.DictReader(f))
    index={(z['graph'],z['rank_mode'],int(z['seed']),z['method']):float(z['cut_weight']) for z in rows}
    stats=[];pairs=[]
    for graph in GRAPHS:
        for mode in MODES:
            base=np.array([index[(graph,mode,s,'baseline_y')] for s in seed_range])
            for method in METHODS:
                score=np.array([index[(graph,mode,s,method)] for s in seed_range])
                delta=score-base
                rng=np.random.default_rng(np.random.SeedSequence([30030,int(graph[1:]),0 if mode=='default' else 1,METHODS.index(method)]))
                draws=rng.integers(0,len(delta),size=(bootstrap,len(delta)))
                means=delta[draws].mean(axis=1)
                lo,hi=np.percentile(means,[2.5,97.5])
                stats.append(dict(graph=graph,rank_mode=mode,method=method,n=100,
                     mean_cut=float(score.mean()),mean_baseline=float(base.mean()),
                     mean_delta=float(delta.mean()),mean_relative_pct=float(100*delta.mean()/base.mean()),
                     ci95_low=float(lo),ci95_high=float(hi),wins=int((delta>0).sum()),
                     ties=int((delta==0).sum()),losses=int((delta<0).sum()),
                     best=float(score.max()),baseline_best=float(base.max())))
                for seed,sc,ba in zip(seed_range,score,base):
                    pairs.append(dict(graph=graph,rank_mode=mode,seed=seed,method=method,
                        cut_weight=sc,baseline=ba,delta=sc-ba))
    summary={'schema':'experiment030-v1','graphs':list(GRAPHS),'seeds':len(seed_range),
             'factor_cases':len(rows)//len(METHODS),'rounded_records':len(rows),
             'methods':list(METHODS),'rank_modes':list(MODES),'input_factors_reused':1000,
             'new_eva_solves':0,'graph_audits':audits,'bootstrap_reps':bootstrap,'complete':len(rows)==6000}
    root.mkdir(parents=True,exist_ok=True)
    def csv_out(name,records):
        with (root/name).open('w',newline='',encoding='utf8') as fh:
            w=csv.DictWriter(fh,fieldnames=list(records[0]),lineterminator='\n')
            w.writeheader();w.writerows(records)
    csv_out('all_rounds.csv',rows)
    csv_out('paired_differences.csv',pairs)
    csv_out('per_graph_method.csv',stats)
    (root/'summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n',encoding='utf8',newline='\n')
    report=['# Experiment030 — Local-field spherical clustering on frozen EVA embeddings','',
      '**Verified scope:** Five G-set graphs, 100 EVA seeds, two factor widths, six K-MeansNM variants; no new EVA or SDP solves.',
      'Every baseline hard partition reproduces Experiment029 exactly. All scores use original signed graph weights.','',
      '| Graph | Rank | Method | Mean cut | Mean Δ vs Y | 95% paired CI | W/T/L | Best |',
      '|---|---|---|---:|---:|---|---:|---:|']
    for z in stats:
        report.append(f"| {z['graph']} | {z['rank_mode']} | {z['method']} | {z['mean_cut']:.2f} | {z['mean_delta']:+.2f} | [{z['ci95_low']:+.2f}, {z['ci95_high']:+.2f}] | {z['wins']}/{z['ties']}/{z['losses']} | {z['best']:.0f} |")
    report+=['','**Confidence intervals**: percentile paired bootstrap within each fixed graph and rank; these do not infer population-level performance or correct for multiple comparisons.',
       '**Interpretation**: -H at exact EVA fixed points is D Y, so raw-field clustering is (approximately) field-strength-weighted spherical clustering, not a fundamentally different directional embedding.',
       'Raw-field and field-weighted-Y outcomes should be interpreted alongside their observed angular deviations; unnormalized rows affect centroid weighting.',
       '**Bias control**: absolute-weighted-degree-Y isolates a degree/weight-magnitude effect. Coherence weights are norm(H)/weighted-degree, bounded by one.',
       '**Caveat**: The six methods were studied on the same five graphs; any winner chosen after seeing these results is exploratory.','']
    (root/'report.md').write_text('\n'.join(report)+'\n',encoding='utf8',newline='\n')
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT)
    parser.add_argument('--baseline',type=Path,default=EX029)
    parser.add_argument('--graphs',nargs='+',choices=GRAPHS,default=list(GRAPHS))
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--max-seeds',type=int,default=100,help='Synthetic/debug partial run only; production exactly 100')
    a=parser.parse_args(argv)
    props=properties()
    for g in a.graphs: load_verified_graph(g,props)
    index=archived_row_index(a.baseline/'all_runs.csv')
    for g in a.graphs:
        for m in MODES:
            for seed in ALL_SEEDS[:a.max_seeds]:
                if (g,seed,m) not in index:raise RuntimeError('Missing frozen baseline record')
    if a.validate_only:
        print('PASS: five graph hashes and 1000 original factor identities exist in metadata')
        return 0
    if a.verify_only:
        for g in a.graphs: print(verify_graph(g,a.output,a.baseline,expected_seed_count=a.max_seeds))
        if len(a.graphs)==5 and a.max_seeds==100:print(summarize(a.output,a.baseline))
        return 0
    if a.workers<1 or a.workers>5:raise ValueError('workers must be 1..5')
    if a.workers==1:
        for g in a.graphs:do_graph(g,a.output,a.baseline,a.max_seeds)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as executor:
            futures=[executor.submit(do_graph,g,a.output,a.baseline,a.max_seeds) for g in a.graphs]
            for future in as_completed(futures):future.result()
    if len(a.graphs)==5 and a.max_seeds==100:
        print(summarize(a.output,a.baseline))
    return 0


if __name__=='__main__':raise SystemExit(main())