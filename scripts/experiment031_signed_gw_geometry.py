"""Experiment031: signed-edge geometry and matched GW hyperplanes on frozen EVA factors.

No EVA/SDP/iterative clustering/local search is run. Each case loads the exact
archived Experiment028/029 EVA factor and checks its source SHA-256.

Designed for execution on the dedicated PCAG103A-13 Actions runner, with
independent hosted verification of every archived hard cut and source ID.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import sparse

GRAPHS = ('G1', 'G6', 'G11', 'G14', 'G18')
MODES = ('default', 'full')
SEEDS = tuple(range(2026, 2126))
DIRECTIONS = 16
ROOT = Path('results/gset/experiment031-signed-gw-geometry')
SOURCE = Path('results/gset/experiment029-eva-kmeansnm-100seed/all_runs.csv')
EPS = 1e-12

CASE_FIELDS = ('graph','rank_mode','seed','p','graph_sha256','factor_sha256',
    'positive_edges','negative_edges','positive_abs_weight','negative_abs_weight',
    'pos_angle_deg','neg_angle_deg','pos_cut_probability','neg_cut_probability',
    'pos_unfavorable_frac','neg_unfavorable_frac',
    'expected_positive_cut','expected_negative_cut','expected_signed_cut',
    'relaxation_objective','min_alignment_cosine','mean_alignment_cosine',
    'zero_fields','negative_vertex_fraction','field_norm_mean','coherence_mean')
TRIAL_FIELDS = ('graph','rank_mode','seed','direction_index','factor_sha256',
    'cut_y','cut_minus_h','cut_positive_only','delta_field_minus_y',
    'mismatched_field_vertices','changed_negative_ablation_vertices',
    'negative_dominance_frac','negative_projection_mean_abs',
    'positive_projection_mean_abs','positive_cut_y','negative_cut_y')


def sha_file(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def sha_text_file(path:Path)->str:
    """Stable source-index fingerprint across Windows CRLF and Linux LF."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n",b"\n")).hexdigest()


def rng_normals(graph: str, mode: str, seed: int, p: int, count: int = DIRECTIONS) -> np.ndarray:
    """Independent deterministic GW hyperplanes matched across Y/H ablations."""
    if not (1<=count<=DIRECTIONS):
        raise ValueError('direction count must be between 1 and 16')
    mode_index=MODES.index(mode)
    rng=np.random.default_rng(np.random.SeedSequence([31031,int(graph[1:]),mode_index,seed]))
    return rng.standard_normal((p,count))


def signed_fields(a: sparse.csr_matrix, y: np.ndarray):
    plus=a.copy()
    plus.data=np.maximum(plus.data,0.0)
    plus.eliminate_zeros()
    minus=a.copy()
    minus.data=np.minimum(minus.data,0.0)
    minus.eliminate_zeros()
    hp=np.asarray(plus@y)
    hn=np.asarray(minus@y)
    h=hp+hn
    d=np.linalg.norm(h,axis=1)
    degree=np.asarray(np.abs(a).sum(axis=1)).ravel()
    cos=np.einsum('ij,ij->i',y,-h,optimize=True)/np.maximum(d,EPS)
    actual=d>EPS
    if np.any(d>degree+1e-7*np.maximum(1,degree)):
        raise AssertionError('Field magnitude violates triangle inequality')
    return hp,hn,h,d,degree,cos,actual


def edge_angles(y:np.ndarray, edges):
    """Exact GW angular edge separation probabilities for unit vectors."""
    i,j,w=edges
    dot=np.einsum('ij,ij->i',y[i],y[j],optimize=True)
    dot=np.clip(dot,-1.0,1.0)
    prob=np.arccos(dot)/np.pi
    return dot,prob


def signed_cut_batch(edges, labels:np.ndarray):
    """For K directions, score K binary n-vertex partitions on original signed edges."""
    i,j,w=edges
    if labels.ndim!=2:
        raise ValueError('labels must be n by directions')
    # Loop across directions to cap transient memory for large sparse graphs.
    return np.asarray([float(w[labels[i,k]!=labels[j,k]].sum()) for k in range(labels.shape[1])])


def edge_summary(y,edges):
    dot,prob=edge_angles(y,edges)
    _,_,w=edges
    pos=w>0;neg=w<0
    if np.any(~(pos|neg)):
        raise ValueError('Zero-weight edge in canonical graph')
    result={}
    for key,mask in [('pos',pos),('neg',neg)]:
        aw=np.abs(w[mask]);total=float(aw.sum())
        result[f'{key}_edges']=int(mask.sum())
        result[f'{key}_abs_weight']=total
        result[f'{key}_angle_deg']=float(180*np.average(prob[mask],weights=aw)) if total else None
        result[f'{key}_cut_probability']=float(np.average(prob[mask],weights=aw)) if total else None
        # w*(y_i dot y_j)>0 is geometrically unfavorable for an edge.
        unfavorable=(dot[mask]>0) if key=='pos' else (dot[mask]<0)
        result[f'{key}_unfavorable_frac']=float(np.average(unfavorable,weights=aw)) if total else None
    result['expected_positive_cut']=float(np.sum(w[pos]*prob[pos]))
    result['expected_negative_cut']=float(np.sum(w[neg]*prob[neg]))
    result['expected_signed_cut']=float(np.sum(w*prob))
    result['relaxation_objective']=float(np.sum(w*(1-dot)*0.5))
    return result


def score_decomposition(edges,labels):
    i,j,w=edges
    crossings=labels[i]!=labels[j]
    return (float(w[(w>0)&crossings].sum()),float(w[(w<0)&crossings].sum()))


def compute_case(graph,mode,seed,a,edges,row,nhyp=DIRECTIONS):
    from maxkcut.eva import load_eva_factor,vector_objective
    factor=Path(row['factor_path'])
    if not factor.is_file() or sha_file(factor)!=row['factor_sha256']:
        raise ValueError('Frozen factor missing or source SHA changed: '+str(factor))
    y,state=load_eva_factor(factor)
    n,p=y.shape
    if n!=int(row['n']) or p!=int(row['p']) or np.max(np.abs(np.linalg.norm(y,axis=1)-1))>1e-6:
        raise AssertionError('Invalid frozen embedding')
    if abs(vector_objective(a,y)-float(row['continuous_objective']))>1e-4:
        raise AssertionError('Frozen SDP objective changed')
    hp,hn,h,d,deg,cos,nonzero=signed_fields(a,y)
    if not np.all(np.isfinite(h)):
        raise AssertionError('Nonfinite local field')
    geom=edge_summary(y,edges)
    normals=rng_normals(graph,mode,seed,p,nhyp)
    proj_y=y@normals
    proj_p=hp@normals
    proj_n=hn@normals
    proj_h=proj_p+proj_n
    if np.max(np.abs(proj_h-(h@normals)))>1e-6:
        raise AssertionError('Signed field decomposition incorrect')
    z_y=proj_y>=0
    # -h, not h, aligns with y at an exact EVA fixed point.
    z_h=(-proj_h)>=0
    # Positive-only control freezes Y and drops negative contributions to H.
    # For a vertex with no positive neighbors, preserve its original Y sign.
    plus_norm=np.linalg.norm(hp,axis=1)
    z_pos=np.where(plus_norm[:,None]>EPS,-proj_p>=0,z_y)
    scores_y=signed_cut_batch(edges,z_y)
    scores_h=signed_cut_batch(edges,z_h)
    scores_pos=signed_cut_batch(edges,z_pos)
    key=f'{mode}:{seed}'
    case=dict(graph=graph,rank_mode=mode,seed=seed,p=p,
              graph_sha256=row['graph_sha256'],factor_sha256=row['factor_sha256'],
              positive_edges=geom['pos_edges'],negative_edges=geom['neg_edges'],
              positive_abs_weight=geom['pos_abs_weight'],negative_abs_weight=geom['neg_abs_weight'],
              pos_angle_deg=geom['pos_angle_deg'],neg_angle_deg=geom['neg_angle_deg'],
              pos_cut_probability=geom['pos_cut_probability'],neg_cut_probability=geom['neg_cut_probability'],
              pos_unfavorable_frac=geom['pos_unfavorable_frac'],neg_unfavorable_frac=geom['neg_unfavorable_frac'],
              expected_positive_cut=geom['expected_positive_cut'],
              expected_negative_cut=geom['expected_negative_cut'],
              expected_signed_cut=geom['expected_signed_cut'],
              relaxation_objective=geom['relaxation_objective'],
              min_alignment_cosine=float(np.min(cos[nonzero])) if nonzero.any() else None,
              mean_alignment_cosine=float(np.mean(cos[nonzero])) if nonzero.any() else None,
              zero_fields=int(np.sum(~nonzero)),
              negative_vertex_fraction=float(np.mean(np.linalg.norm(hn,axis=1)>EPS)),
              field_norm_mean=float(np.mean(d)),
              coherence_mean=float(np.mean(d/np.maximum(deg,EPS))))
    trials=[]
    for k in range(nhyp):
        pos_score,neg_score=score_decomposition(edges,z_y[:,k])
        trials.append(dict(graph=graph,rank_mode=mode,seed=seed,direction_index=k,
            factor_sha256=row['factor_sha256'],cut_y=float(scores_y[k]),
            cut_minus_h=float(scores_h[k]),cut_positive_only=float(scores_pos[k]),
            delta_field_minus_y=float(scores_h[k]-scores_y[k]),
            mismatched_field_vertices=int(np.count_nonzero(z_y[:,k]!=z_h[:,k])),
            changed_negative_ablation_vertices=int(np.count_nonzero(z_h[:,k]!=z_pos[:,k])),
            negative_dominance_frac=float(np.mean(np.abs(proj_n[:,k])>np.abs(proj_p[:,k]))),
            negative_projection_mean_abs=float(np.mean(np.abs(proj_n[:,k]))),
            positive_projection_mean_abs=float(np.mean(np.abs(proj_p[:,k]))),
            positive_cut_y=pos_score,negative_cut_y=neg_score))
    arrays={
        f'{key}:y':np.packbits(z_y.astype(np.uint8),axis=0).T,
        f'{key}:minus_h':np.packbits(z_h.astype(np.uint8),axis=0).T,
        f'{key}:positive_only':np.packbits(z_pos.astype(np.uint8),axis=0).T,
    }
    return case,trials,arrays


def source_rows(source=SOURCE):
    with Path(source).open(newline='',encoding='utf8') as f:
        rows=list(csv.DictReader(f))
    index={(r['graph'],r['rank_mode'],int(r['seed'])):r for r in rows}
    if len(rows)!=1000 or len(index)!=1000:
        raise AssertionError('Need exactly 1000 distinct prior EVA embeddings')
    return index


def write_csv(path,rows,fields):
    with Path(path).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n')
        w.writeheader();w.writerows(rows)


def run_graph(graph,output=ROOT,source=SOURCE,nhyp=DIRECTIONS,max_seeds=100):
    from experiment029_kmeansnm_100seeds import properties,load_verified_graph
    props=properties()
    a,edges,meta=load_verified_graph(graph,props)
    rows=source_rows(source)
    dest=Path(output)/'graphs'/graph
    dest.mkdir(parents=True,exist_ok=True)
    files=[dest/x for x in ('cases.csv','trials.csv','partitions.npz','manifest.json')]
    if any(x.exists() for x in files):
        if not all(x.exists() for x in files):raise RuntimeError('Incomplete existing graph output; do not overwrite')
        verify_graph(graph,output,source,nhyp,max_seeds)
        print(f'REUSED VERIFIED {graph}',flush=True);return
    cs=[];ts=[];pa={}
    started=time.perf_counter()
    for mode in MODES:
        for seed in SEEDS[:max_seeds]:
            source_row=rows[(graph,mode,seed)]
            if source_row['graph_sha256']!=meta['graph_sha256']:
                raise AssertionError('Graph source fingerprint mismatch')
            case,trials,arrays=compute_case(graph,mode,seed,a,edges,source_row,nhyp)
            cs.append(case);ts.extend(trials);pa.update(arrays)
        print(f'{graph} {mode} completed {max_seeds} factor geometries',flush=True)
    write_csv(dest/'cases.csv',cs,CASE_FIELDS)
    write_csv(dest/'trials.csv',ts,TRIAL_FIELDS)
    np.savez_compressed(dest/'partitions.npz',**pa)
    manifest=dict(schema='experiment031-graph-v1',graph=graph,source_graph_sha256=meta['graph_sha256'],
        n=int(a.shape[0]),seeds=list(SEEDS[:max_seeds]),ranks=list(MODES),nhyp=nhyp,
        cases=len(cs),trials=len(ts),reused_eva_factors=len(cs),new_eva_solves=0,
        source_index_sha256=sha_text_file(source),
        case_csv_sha256=sha_file(dest/'cases.csv'),trial_csv_sha256=sha_file(dest/'trials.csv'),
        partition_sha256=sha_file(dest/'partitions.npz'),runtime_seconds=time.perf_counter()-started)
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf8',newline='\n')
    verify_graph(graph,output,source,nhyp,max_seeds)
    print(f'PASS {graph}: {len(cs)} factors, {len(ts)} matched GW hyperplanes, all verified',flush=True)


def verify_graph(graph,output=ROOT,source=SOURCE,nhyp=DIRECTIONS,max_seeds=100):
    """Independent-of-factor audit: compare saved partition bits with exact signed graph."""
    from experiment029_kmeansnm_100seeds import properties,load_verified_graph
    a,edges,meta=load_verified_graph(graph,properties())
    root=Path(output)/'graphs'/graph
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf8'))
    if manifest['source_graph_sha256']!=meta['graph_sha256'] or manifest['graph']!=graph:
        raise AssertionError('Graph provenance mismatch')
    for name,key in [('cases.csv','case_csv_sha256'),('trials.csv','trial_csv_sha256'),('partitions.npz','partition_sha256')]:
        if sha_file(root/name)!=manifest[key]:raise AssertionError('Output checksum mismatch: '+name)
    if manifest['source_index_sha256']!=sha_text_file(source):raise AssertionError('Source index changed')
    with (root/'cases.csv').open(newline='',encoding='utf8') as f:cases=list(csv.DictReader(f))
    with (root/'trials.csv').open(newline='',encoding='utf8') as f:trials=list(csv.DictReader(f))
    expected_cases={(mode,seed) for mode in MODES for seed in SEEDS[:max_seeds]}
    actual_cases={(r['rank_mode'],int(r['seed'])) for r in cases}
    if len(cases)!=len(expected_cases) or expected_cases!=actual_cases:raise AssertionError('Case coverage invalid')
    prior=source_rows(source)
    for c in cases:
        ref=prior[(graph,c['rank_mode'],int(c['seed']))]
        if c['factor_sha256']!=ref['factor_sha256'] or c['graph_sha256']!=ref['graph_sha256']:
            raise AssertionError('Prior factor identity mismatch')
    expected_trials={(mode,seed,k) for mode in MODES for seed in SEEDS[:max_seeds] for k in range(nhyp)}
    if len(trials)!=len(expected_trials) or {(r['rank_mode'],int(r['seed']),int(r['direction_index'])) for r in trials}!=expected_trials:
        raise AssertionError('GW trial coverage invalid')
    with np.load(root/'partitions.npz',allow_pickle=False) as archive:
        if set(archive.files)!={f'{mode}:{seed}:{label}' for mode,seed in expected_cases for label in ('y','minus_h','positive_only')}:
            raise AssertionError('Packed partition keys incomplete')
        for t in trials:
            mode,seed,k=t['rank_mode'],int(t['seed']),int(t['direction_index'])
            ref=prior[(graph,mode,seed)]
            if t['factor_sha256']!=ref['factor_sha256']:raise AssertionError('Mismatch factor in GW trial')
            bits={q:np.unpackbits(archive[f'{mode}:{seed}:{q}'][k])[:a.shape[0]] for q in ('y','minus_h','positive_only')}
            for q,field in [('y','cut_y'),('minus_h','cut_minus_h'),('positive_only','cut_positive_only')]:
                actual=float(edges[2][bits[q][edges[0]]!=bits[q][edges[1]]].sum())
                if abs(actual-float(t[field]))>1e-9:raise AssertionError(f'Original signed cut mismatch {graph} {mode} {seed} {k} {q}')
            if np.count_nonzero(bits['y']!=bits['minus_h'])!=int(t['mismatched_field_vertices']):
                raise AssertionError('GW invariance mismatch')
            if np.count_nonzero(bits['minus_h']!=bits['positive_only'])!=int(t['changed_negative_ablation_vertices']):
                raise AssertionError('Negative-ablation discrepancy invalid')
            if abs(float(t['positive_cut_y'])+float(t['negative_cut_y'])-float(t['cut_y']))>1e-9:
                raise AssertionError('Signed cut decomposition incorrect')
    return {'graph':graph,'factor_cases':len(cases),'gw_pairs':len(trials),'all_signed_cuts_validated':3*len(trials)}


def summarize(root=ROOT,source=SOURCE,bootstrap=5000):
    root=Path(root)
    records=[];trials=[];audits=[]
    for graph in GRAPHS:
        audits.append(verify_graph(graph,root,source))
        with (root/'graphs'/graph/'cases.csv').open(newline='',encoding='utf8') as f:records+=list(csv.DictReader(f))
        with (root/'graphs'/graph/'trials.csv').open(newline='',encoding='utf8') as f:trials+=list(csv.DictReader(f))
    case_idx={(r['graph'],r['rank_mode'],int(r['seed'])):r for r in records}
    trial_idx={}
    for t in trials:
        key=(t['graph'],t['rank_mode'],int(t['seed']))
        trial_idx.setdefault(key,[]).append(t)
    groups=[]
    for graph in GRAPHS:
        for mode in MODES:
            cells=[case_idx[(graph,mode,seed)] for seed in SEEDS]
            ts=[trial_idx[(graph,mode,seed)] for seed in SEEDS]
            def cm(key):
                vals=[float(z[key]) for z in cells if z[key] not in ('','None','nan')]
                return float(np.mean(vals)) if vals else None
            def tm(key):
                return float(np.mean([float(t[key]) for trial in ts for t in trial]))
            seed_delta=np.asarray([np.mean([float(t['delta_field_minus_y']) for t in trial]) for trial in ts])
            rng=np.random.default_rng(np.random.SeedSequence([31031,int(graph[1:]),MODES.index(mode),100]))
            draws=rng.integers(0,len(seed_delta),size=(bootstrap,len(seed_delta)))
            lo,hi=np.percentile(seed_delta[draws].mean(axis=1),[2.5,97.5])
            mean_expected=cm('expected_signed_cut')
            mean_empirical=tm('cut_y')
            groups.append(dict(graph=graph,rank_mode=mode,n=100,gw_directions=DIRECTIONS,
                positive_edges=int(cells[0]['positive_edges']),negative_edges=int(cells[0]['negative_edges']),
                mean_positive_angle_deg=cm('pos_angle_deg'),mean_negative_angle_deg=cm('neg_angle_deg'),
                mean_positive_cut_probability=cm('pos_cut_probability'),
                mean_negative_cut_probability=cm('neg_cut_probability'),
                mean_pos_unfavorable_frac=cm('pos_unfavorable_frac'),
                mean_neg_unfavorable_frac=cm('neg_unfavorable_frac'),
                expected_signed_cut=mean_expected,empirical_gw_mean=mean_empirical,
                empirical_minus_theory=mean_empirical-mean_expected,
                mean_field_minus_y=tm('delta_field_minus_y'),
                ci95_delta_low=float(lo),ci95_delta_high=float(hi),
                mismatch_fraction=tm('mismatched_field_vertices')/800,
                negative_ablation_fraction=tm('changed_negative_ablation_vertices')/800,
                negative_projection_dominance_fraction=tm('negative_dominance_frac'),
                min_alignment_cosine=float(min(float(z['min_alignment_cosine']) for z in cells)),
                mean_alignment_cosine=cm('mean_alignment_cosine')))
    write_csv(root/'all_cases.csv',records,CASE_FIELDS)
    write_csv(root/'all_gw_trials.csv',trials,TRIAL_FIELDS)
    write_csv(root/'per_graph_rank.csv',groups,tuple(groups[0]))
    summary={'schema':'experiment031-v1','complete':True,'graphs':list(GRAPHS),
        'cases':len(records),'paired_hyperplanes':len(trials),'verified_hard_cuts':len(trials)*3,
        'new_eva_solves':0,'gw_directions_per_factor':DIRECTIONS,'bootstrap_replicates':bootstrap,
        'input_eva_factors_reused':len(records),'graph_audits':audits}
    (root/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf8',newline='\n')
    lines=['# Experiment031 — Signed-edge geometry and GW projection diagnostics','',
        '**Frozen factors only:** 1,000 independently archived EVA embeddings, 16 matched Gaussian hyperplanes per embedding, 16,000 paired GW comparisons; no EVA or SDP optimization.','',
        '| Graph | p | Positive-edge mean angle | Negative-edge mean angle | Analytic expected cut | Empirical GW cut | -H vs Y mean Δ | Field-label mismatch | Negative-edge field ablation |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for z in groups:
        a='n/a' if z['mean_negative_angle_deg'] is None else f"{z['mean_negative_angle_deg']:.2f}°"
        lines.append(f"| {z['graph']} | {41 if z['rank_mode']=='default' else 800} | {z['mean_positive_angle_deg']:.2f}° | {a} | {z['expected_signed_cut']:.2f} | {z['empirical_gw_mean']:.2f} | {z['mean_field_minus_y']:+.3f} | {100*z['mismatch_fraction']:.4f}% | {100*z['negative_ablation_fraction']:.2f}% |")
    lines+=['', 'The exact angular GW expectation is Σ_(i,j) w_ij arccos(y_i·y_j)/π, including negative weights with their original signs.',
        'The negative-edge field ablation removes W_- from the projected field **at frozen Y**; it does not solve or reoptimize a positive-only graph.',
        'EVA fixed-point alignment predicts that -H and Y produce the same GW signs except near hyperplane boundaries at finite convergence tolerance.',
        'The 16 hyperplanes share each underlying factor; uncertainty must be resampled over the 100 EVA seeds, not over 1,600 independent observations.',
        'All conclusions describe these five graphs. Positive-only G1/G14 are controls. No approximation guarantee or global SDP certificate follows.',
        '',f"**Coverage:** {len(records)} source factors, {len(trials)} matched normals, {3*len(trials)} independently re-scorable signed hard cuts, with full bit-packed labels and SHA-256 manifests."]
    (root/'report.md').write_text('\n'.join(lines).rstrip('\n')+'\n',encoding='utf8',newline='\n')
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--validate-only',action='store_true')
    ap.add_argument('--graph',choices=GRAPHS)
    ap.add_argument('--workers',type=int,default=5)
    ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--output',type=Path,default=ROOT)
    args=ap.parse_args()
    from experiment029_kmeansnm_100seeds import properties,load_verified_graph
    idx=source_rows();props=properties()
    for graph in GRAPHS:
        a,edges,metadata=load_verified_graph(graph,props)
        for mode in MODES:
            for seed in SEEDS:
                row=idx[(graph,mode,seed)]
                if row['graph_sha256']!=metadata['graph_sha256'] or int(row['n'])!=800:
                    raise AssertionError('Bad archived graph index '+graph)
    if args.validate_only:
        print('PASS: five original G-set SHA fingerprints and all 1000 frozen EVA metadata references');return
    if args.check_only:
        summary=summarize(root=args.output)
        print(json.dumps(summary,indent=2));return
    if args.graph:
        run_graph(args.graph,output=args.output)
    else:
        if args.workers<1 or args.workers>5:raise ValueError('workers must be 1..5')
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(run_graph,graph,args.output):graph for graph in GRAPHS}
            for future in as_completed(futures):
                future.result()
    summary=summarize(root=args.output)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()