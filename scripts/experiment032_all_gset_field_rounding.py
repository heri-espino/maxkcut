"""Experiment032: paired GW / K-MeansNM on Y vs -WY, all 71 frozen G-set EVA factors.

No EVA/SDP/Flip/KL/ILS. Complete signed cut verification and per-Lloyd-step
traces. All source factors stay immutable in Experiments002/003/021.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from maxkcut.eva import _identity, _prepared_graph, load_eva_factor
from maxkcut.io import load_graph, edge_arrays
from maxkcut.objective import cut_weight
from maxkcut.rounding import round_spherical_kmeans

ROOT = Path('results/gset/experiment032-all-gset-field-rounding')
INVENTORY = Path('results/gset/experiment021-complete-71-gset/inventory.json')
FEATURES = Path('results/gset/experiment022-graph-structure-methods/graph_features.csv')
METHODS = ('GW_Y', 'GW_minusH', 'KM_Y', 'KM_minusH')
RESTARTS = 50
SEED = 2026
GRAPH_FIELDS = ('graph','restart','method','cut_weight','partition_sha256',
    'source_factor_sha256','source_graph_sha256','n','p','m','family','weight_type',
    'round_seconds','iterations','initial_gw_cut','field_alignment_min_cos',
    'field_alignment_mean_cos','zero_fields','gw_label_mismatch_fraction')


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def sha_bytes(buf):return hashlib.sha256(buf).hexdigest()


def dump_json(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')


def write_csv(path,fields,rows):
    with Path(path).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)


def read_inventory(inventory=INVENTORY,features=FEATURES):
    obj=json.loads(Path(inventory).read_text(encoding='utf8'))
    records=obj['records']
    if len(records)!=71 or len({r['graph'] for r in records})!=71 or obj['missing_factors']:
        raise ValueError('Expected exactly 71 archived source factor identities')
    if any(r['status'] not in ('ready','reused_019') for r in records):
        raise ValueError('Input EVA factor inventory has unresolved cases')
    with Path(features).open(encoding='utf8',newline='') as f:
        feature_map={r['graph']:r for r in csv.DictReader(f)}
    if len(feature_map)!=71 or set(feature_map)!={r['graph'] for r in records}:
        raise ValueError('Graph feature table does not cover same 71 instances')
    for r in records:
        if r['graph_sha256']!=feature_map[r['graph']]['graph_sha256']:
            raise ValueError('Graph SHA disagrees with independent structure manifest: '+r['graph'])
        if len(r['factor_sha256'])!=64 or len(r['graph_sha256'])!=64:
            raise ValueError('Missing original SHA identity: '+r['graph'])
    return sorted(records,key=lambda z:int(z['graph'][1:])),feature_map


def accepted_eol_hashes(path):
    raw=Path(path).read_bytes()
    lf=raw.replace(b'\r\n',b'\n')
    return {sha_bytes(raw),sha_bytes(lf),sha_bytes(lf.replace(b'\n',b'\r\n'))}


def factor_location(record):
    src=Path(record['factor_source'].replace('\\','/'))
    return src/'factors'/f"{record['graph']}.npz", src/'cases'/f"{record['graph']}.json"


def load_frozen(record,features,*,require_factor=True):
    graph=record['graph']; src=Path('data')/(graph+'.txt')
    if not src.exists():raise FileNotFoundError(src)
    if record['graph_sha256'] not in accepted_eol_hashes(src):
        raise ValueError('Graph graph-source SHA mismatch, including EOL normalization: '+graph)
    a=load_graph(src)
    if int(a.shape[0])!=int(record['vertices']):raise ValueError('Graph vertex count mismatch: '+graph)
    factor_path,case_path=factor_location(record)
    if not case_path.is_file():raise FileNotFoundError(case_path)
    case=json.loads(case_path.read_text(encoding='utf8'))
    if case['graph_sha256']!=record['graph_sha256'] or case['factor_rank']!=record['rank']:
        raise ValueError('Original EVA case mismatch: '+graph)
    if case['status'] not in ('converged','iteration_limit'):
        raise ValueError('Archived EVA solver not finished: '+graph)
    if not require_factor:return a,None,case
    if not factor_path.is_file():raise FileNotFoundError(factor_path)
    if sha_file(factor_path)!=record['factor_sha256']:
        raise ValueError('Original factor archive SHA mismatch: '+graph)
    y,meta=load_eva_factor(factor_path)
    if y.shape!=(a.shape[0],int(record['rank'])) or meta['sweeps']!=case['factor_sweeps']:
        raise ValueError('Original factor/metadata dimensions or sweeps differ: '+graph)
    prepared=_prepared_graph(a)
    seed=case['provenance']['seed']; tol=case['provenance']['tolerance']
    if case['provenance'].get('optimizer_backend','numpy')=='numba':
        prepared.sort_indices()
    expected=_identity(prepared,y.shape[1],seed,tol)
    if case['provenance'].get('optimizer_backend','numpy')=='numba':
        expected['algorithm']='eva-sequential-numba-v1'
    if meta['identity']!=expected:raise ValueError('Immutable EVA identity mismatch: '+graph)
    return a,y,case


def signed_cut(edges,labels):return float(cut_weight(edges,labels))


def normals(graph_number, restart, rank):
    rng=np.random.default_rng(np.random.SeedSequence([SEED,graph_number,2,restart,0]))
    raw=rng.standard_normal((2,rank))
    raw[1]=-raw[0]
    raw/=np.linalg.norm(raw,axis=1)[:,None]
    return raw,rng


def cluster_trace(v,edges,initial,rng,max_iter=100):
    """Rounder-equivalent spherical KMeansNM with full best-so-far histories."""
    centers=initial.copy()
    previous=None;best=None;best_cut=-np.inf
    history=[]
    for it in range(1,max_iter+1):
        labels=np.argmax(v @ centers.T,axis=1).astype(np.uint8)
        score=signed_cut(edges,labels)
        improved=score>best_cut
        if improved:best_cut=score;best=labels.copy()
        stable=previous is not None and np.array_equal(labels,previous)
        history.append({'iteration':it,'cut_weight':score,'best_so_far':best_cut,'improved':bool(improved),'stable':bool(stable)})
        if stable:return best,it,history
        previous=labels.copy()
        for c in range(2):
            members=v[labels==c]
            if len(members)==0:
                centers[c]=v[rng.integers(len(v))]
            else:
                raw=members.sum(axis=0)
                norm=np.linalg.norm(raw)
                centers[c]=raw/norm if norm>1e-12 else v[rng.integers(len(v))]
    return best,max_iter,history


def prepare_representations(a,y):
    h=np.asarray(a @ y,dtype=np.float64)
    d=np.linalg.norm(h,axis=1)
    nonzero=d>1e-12
    z=-h
    z[~nonzero]=y[~nonzero]
    alignment=np.einsum('ij,ij->i',y[nonzero],z[nonzero],optimize=True)/d[nonzero]
    return {'Y':y,'minusH':z},dict(field_alignment_min_cos=float(np.min(alignment)) if alignment.size else 1.,
        field_alignment_mean_cos=float(np.mean(alignment)) if alignment.size else 1.,
        zero_fields=int((~nonzero).sum()))


def pack(labels):return np.packbits(np.asarray(labels,dtype=np.uint8))


def execute_graph(graph, output=ROOT,inventory=INVENTORY,features=FEATURES,restarts=RESTARTS):
    records,feature_map=read_inventory(inventory,features)
    record=next(r for r in records if r['graph']==graph)
    out=Path(output)/'graphs'/graph
    out.mkdir(parents=True,exist_ok=True)
    expected_files=[out/n for n in ('cases.csv','partitions.npz','traces.jsonl','manifest.json')]
    if any(p.exists() for p in expected_files):
        if not all(p.exists() for p in expected_files):
            raise RuntimeError('Partial prior output; preserve and inspect, do not overwrite: '+graph)
        verify_graph(graph,output,inventory,features,restarts)
        print(f'REUSE verified {graph}',flush=True)
        return
    a,y,_=load_frozen(record,feature_map)
    edge=edge_arrays(a)
    feat=feature_map[graph]
    n,p=y.shape
    reps,diag=prepare_representations(a,y)
    cuts=[];partitions={};trace_count=0
    with (out/'traces.jsonl').open('w',encoding='utf8',newline='\n') as log:
        for restart in range(restarts):
            cent,_=normals(int(graph[1:]),restart,p)
            projected={name:reps[name] @ cent[0] for name in reps}
            gw_labels={name:(proj>=0).astype(np.uint8) for name,proj in projected.items()}
            mismatch=float(np.mean(gw_labels['Y']!=gw_labels['minusH']))
            for method in METHODS:
                rep='Y' if method.endswith('_Y') else 'minusH'
                t0=time.perf_counter()
                if method.startswith('GW_'):
                    labels=gw_labels[rep]; iters=1;history=[]
                else:
                    _,km_rng=normals(int(graph[1:]),restart,p)
                    labels,iters,history=cluster_trace(reps[rep],edge,cent,km_rng)
                    if signed_cut(edge,labels)+1e-8<signed_cut(edge,gw_labels[rep]):
                        raise AssertionError('K-MeansNM best-ever must contain GW starting cut')
                    if restart<2: # numerical cross-check with unchanged repository rounder
                        seed_rng=np.random.default_rng(np.random.SeedSequence([SEED,int(graph[1:]),2,restart,0]))
                        published,nit=round_spherical_kmeans(reps[rep],2,seed_rng,antipodal=True,
                            edges=edge,keep_best=True)
                        if nit!=iters or not np.array_equal(published,labels):
                            raise AssertionError('Trace diverged from source implementation')
                duration=time.perf_counter()-t0
                score=signed_cut(edge,labels)
                bits=pack(labels)
                partitions[f'{method}_r{restart:03d}']=bits
                cuts.append(dict(graph=graph,restart=restart,method=method,cut_weight=score,
                    partition_sha256=sha_bytes(bits.tobytes()),source_factor_sha256=record['factor_sha256'],
                    source_graph_sha256=record['graph_sha256'],n=n,p=p,m=int(a.nnz//2),
                    family=feat['family'],weight_type=feat['weight_type'],round_seconds=duration,
                    iterations=iters,initial_gw_cut=signed_cut(edge,gw_labels[rep]),
                    gw_label_mismatch_fraction=mismatch,**diag))
                for item in history:
                    log.write(json.dumps({'graph':graph,'restart':restart,'method':method,**item},
                        separators=(',',':'),sort_keys=True)+'\n')
                    trace_count+=1
            if restart in (0,9,24,49):
                print(f'{graph}: rounded {restart+1}/{restarts} shared starts',flush=True)
    write_csv(out/'cases.csv',GRAPH_FIELDS,cuts)
    np.savez_compressed(out/'partitions.npz',**partitions)
    manifest=dict(schema='experiment032-graph-v1',graph=graph,n=n,p=p,
        restarts=restarts,method_order=list(METHODS),records=len(cuts),partitions=len(partitions),
        trajectory_rows=trace_count,original_graph_sha256=record['graph_sha256'],
        original_factor_sha256=record['factor_sha256'],
        factor_source=record['factor_source'],no_new_eva=True,
        hashes={name:sha_file(out/name) for name in ('cases.csv','partitions.npz','traces.jsonl')})
    dump_json(out/'manifest.json',manifest)
    verify_graph(graph,output,inventory,features,restarts)
    print(f'PASS signed-score audit {graph}: {len(cuts)} cases, {trace_count} steps',flush=True)


def verify_graph(graph,output=ROOT,inventory=INVENTORY,features=FEATURES,restarts=RESTARTS):
    records,feature_map=read_inventory(inventory,features)
    source=next(r for r in records if r['graph']==graph)
    a,_,_=load_frozen(source,feature_map,require_factor=False)
    edges=edge_arrays(a)
    folder=Path(output)/'graphs'/graph
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf8'))
    if (manifest['original_factor_sha256']!=source['factor_sha256'] or
        manifest['original_graph_sha256']!=source['graph_sha256'] or
        manifest['records']!=restarts*4 or manifest['partitions']!=restarts*4 or
        manifest['restarts']!=restarts or manifest['method_order']!=list(METHODS)):
        raise ValueError('Changed input identity or missing coverage: '+graph)
    for name,sha in manifest['hashes'].items():
        if sha_file(folder/name)!=sha:raise ValueError('Archived file SHA mismatch: '+graph+' '+name)
    with (folder/'cases.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
    expected={(r,m) for r in range(restarts) for m in METHODS}
    if len(rows)!=4*restarts or {(int(x['restart']),x['method']) for x in rows}!=expected:
        raise ValueError('Incomplete or duplicate method/restart identities: '+graph)
    scoremap={}
    with np.load(folder/'partitions.npz',allow_pickle=False) as arc:
        if set(arc.files)!={f'{method}_r{restart:03d}' for restart,method in expected}:
            raise ValueError('Missing packed partitions: '+graph)
        for row in rows:
            key=(int(row['restart']),row['method'])
            bits=arc[f'{row["method"]}_r{int(row["restart"]):03d}']
            if sha_bytes(bits.tobytes())!=row['partition_sha256']:
                raise ValueError('Changed partition SHA: '+graph+str(key))
            labels=np.unpackbits(bits)[:a.shape[0]]
            score=signed_cut(edges,labels)
            if abs(score-float(row['cut_weight']))>1e-7:
                raise ValueError('Original signed cut does not match: '+graph+str(key))
            scoremap[key]=score
    traces={};total=0
    with (folder/'traces.jsonl').open(encoding='utf8') as f:
        for line in f:
            z=json.loads(line);key=(int(z['restart']),z['method'])
            if key not in expected or not key[1].startswith('KM_'):
                raise ValueError('Unexpected clustering trace: '+graph)
            if key in traces and (int(z['iteration'])!=traces[key][0]+1 or
               float(z['best_so_far'])+1e-7<traces[key][1]):
                raise ValueError('Non-consecutive or nonmonotone clustering trace: '+graph)
            if key not in traces and int(z['iteration'])!=1:raise ValueError('Trace starts after 1')
            traces[key]=(int(z['iteration']),float(z['best_so_far']))
            total+=1
    for row in rows:
        key=(int(row['restart']),row['method'])
        if key[1].startswith('KM_'):
            if key not in traces or traces[key][0]!=int(row['iterations']) or abs(traces[key][1]-scoremap[key])>1e-7:
                raise ValueError('Incomplete/bad best-ever clustering trace: '+graph+str(key))
        elif int(row['iterations'])!=1:raise ValueError('GW rounds must have single initial cut')
    if manifest['trajectory_rows']!=total:raise ValueError('Trace row count mismatch')
    return dict(graph=graph,partitions=len(rows),trajectories=total,verified=True)


def bootstrap_mean(deltas, graph, method, reps=3000):
    x=np.asarray(deltas,dtype=float)
    rng=np.random.default_rng(np.random.SeedSequence([32032,int(graph[1:]),METHODS.index(method)]))
    index=rng.integers(0,len(x),size=(reps,len(x)))
    return tuple(map(float,np.percentile(np.mean(x[index],axis=1),[2.5,97.5])))


def aggregate(output=ROOT,inventory=INVENTORY,features=FEATURES,restarts=RESTARTS):
    root=Path(output);source,feat=read_inventory(inventory,features)
    cases=[];stats=[];checks=[]
    for r in source:
        graph=r['graph'];checks.append(verify_graph(graph,root,inventory,features,restarts))
        with (root/'graphs'/graph/'cases.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
        cases+=rows
        by={(z['method'],int(z['restart'])):float(z['cut_weight']) for z in rows}
        obj=dict(graph=graph,n=r['vertices'],p=r['rank'],family=feat[graph]['family'],weight_type=feat[graph]['weight_type'],
            original_graph_sha256=r['graph_sha256'],original_factor_sha256=r['factor_sha256'])
        deltas=[]
        for pair in [('GW_minusH','GW_Y'),('KM_minusH','KM_Y')]:
            m,base=pair
            y=np.array([by[(base,t)] for t in range(restarts)],dtype=float)
            h=np.array([by[(m,t)] for t in range(restarts)],dtype=float)
            ds=h-y
            lo,hi=bootstrap_mean(ds,graph,m)
            obj.update({f'{base}_mean':float(y.mean()),f'{m}_mean':float(h.mean()),
                f'{m}_delta':float(ds.mean()),f'{m}_relative_pct':float(100*ds.mean()/max(abs(y.mean()),1.)),
                f'{m}_ci_low':lo,f'{m}_ci_high':hi,
                f'{base}_best':float(y.max()),f'{m}_best':float(h.max()),
                f'{m}_wins':int((ds>0).sum()),f'{m}_ties':int((ds==0).sum()),f'{m}_losses':int((ds<0).sum())})
        obj['gw_mean_label_mismatch_pct']=float(100*np.mean([float(z['gw_label_mismatch_fraction']) for z in rows if z['method']=='GW_Y']))
        obj['min_alignment_cos']=float(min(float(z['field_alignment_min_cos']) for z in rows))
        obj['zero_fields']=int(rows[0]['zero_fields'])
        stats.append(obj)
    casecols=GRAPH_FIELDS
    write_csv(root/'all_runs.csv',casecols,cases)
    fields=list(stats[0]);write_csv(root/'per_graph.csv',fields,stats)
    signed=[r for r in stats if r['weight_type']=='signed']
    positive=[r for r in stats if r['weight_type']=='positive']
    def macro(grp,key):return float(np.mean([r[key] for r in grp])) if grp else None
    macro_keys=['GW_minusH_relative_pct','KM_minusH_relative_pct']
    summary=dict(schema='experiment032-v1',complete=True,graphs=71,starts_per_graph=restarts,
        source_eva_factors_reused=71,new_eva_solves=0,total_cut_records=len(cases),
        comparisons=71*restarts*2,gw_rounds=71*restarts*2,kmeans_rounds=71*restarts*2,
        signed_graph_count=len(signed),positive_graph_count=len(positive),
        all_verified=all(x['verified'] for x in checks),graph_checks=checks,
        macro_relative_pct={k:{'all':macro(stats,k),'signed':macro(signed,k),'positive':macro(positive,k)} for k in macro_keys},
        mean_vertex_mismatch_pct=macro(stats,'gw_mean_label_mismatch_pct'))
    dump_json(root/'summary.json',summary)
    text=['# Experiment032 — All 71 G-set graphs: GW and K-MeansNM on Y versus -WY','',
        '**Completed coverage:** 71 original EVA factors, 50 matched normals/factor, 4 rounding arms, 14,200 independently verified original signed cuts. No EVA, SDP, Flip, KL or ILS.','',
        'All paired differences are **raw -H minus Y** at fixed source factor and shared initial Gaussian normal. The first K-MeansNM assignment is the corresponding GW partition (up to label complement); K-MeansNM retains the best signed cut along its own Lloyd path.','',
        '## All-graph mean percentage effects (graphs equally weighted)','',
        '| Algorithm | All 71 | Positive-weight | Signed-weight |',
        '|---|---:|---:|---:|']
    for key,label in [('GW_minusH_relative_pct','GW'),('KM_minusH_relative_pct','K-MeansNM')]:
        vals=summary['macro_relative_pct'][key]
        text.append(f"| {label} | {vals['all']:+.5f}% | {vals['positive']:+.5f}% | {vals['signed']:+.5f}% |")
    text.extend(['','## Per-graph original signed cut comparisons','',
        '| Graph | Family | Sign | GW mean Δ | KM mean Δ | KM best Δ | KM 95% paired CI |',
        '|---|---|---|---:|---:|---:|---|'])
    for r in stats:
        text.append(f"| {r['graph']} | {r['family']} | {r['weight_type']} | {r['GW_minusH_delta']:+.2f} | {r['KM_minusH_delta']:+.2f} | {r['KM_minusH_best']-r['KM_Y_best']:+.0f} | [{r['KM_minusH_ci_low']:+.2f}, {r['KM_minusH_ci_high']:+.2f}] |")
    text.extend(['','**Interpretation limitations:** One historical EVA factor is frozen for each graph; 50 starts quantify rounding stochasticity, not independent EVA initialization uncertainty. Graph families are correlated; graphwise CIs condition on each fixed factor and are not multiplicity-adjusted. Compare equal-graph-relative percentages, not pooled signed cut weights. No positive-weight-only reoptimization or GW approximation guarantee for signed weights.','',
        '**Archive:** `all_runs.csv`, `per_graph.csv`, `summary.json`, and all 71 `graphs/G*/{cases.csv,partitions.npz,traces.jsonl,manifest.json}` with immutable factor SHA identities and signed-cut independent rescoring.',''])
    (root/'report.md').write_text('\n'.join(text).rstrip()+'\n',encoding='utf8',newline='\n')
    return summary


def validate_only():
    records,features=read_inventory()
    for record in records:
        a,_,_=load_frozen(record,features,require_factor=False)
        assert a.shape[0]==int(record['vertices'])
    print(f'PASS: 71 immutable case identities and graph SHA checks (without downloading EVA factors)',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=['validate','compute','verify','aggregate'],required=True)
    ap.add_argument('--workers',type=int,default=5)
    args=ap.parse_args()
    if args.phase=='validate':return validate_only()
    if args.phase=='verify':
        cases,feat=read_inventory()
        audit=[verify_graph(c['graph'],ROOT,INVENTORY,FEATURES) for c in cases]
        print(f'PASS independently rescored {sum(x["partitions"] for x in audit)} signed cuts and {sum(x["trajectories"] for x in audit)} clustering iterations',flush=True)
        return
    if args.phase=='aggregate':
        s=aggregate();print(json.dumps({k:v for k,v in s.items() if k!='graph_checks'},sort_keys=True),flush=True);return
    if args.workers not in range(1,9):raise ValueError('workers must be 1 to 8; prevent BLAS/memory oversubscription')
    cases,_=read_inventory()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        fs={executor.submit(execute_graph,c['graph']):c['graph'] for c in cases}
        for f in as_completed(fs):
            graph=fs[f];f.result()
            print(f'COMPLETED {graph}',flush=True)
    summary=aggregate()
    print('EX032 ALL 71 completed, '+str(summary['total_cut_records'])+' cuts',flush=True)

if __name__=='__main__':main()