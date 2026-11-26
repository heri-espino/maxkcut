"""Create TeX macros from audited summary; no new optimization or numerical audit."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
s=json.loads((ROOT/'data/summary.json').read_text())
assert s['complete'] and s['graphs']==71 and s['paired_starts']==1136 and s['new_eva_solves']==0
names={'gw':'Repeated GW','random-sweep':'Random sweep','spherical-kmeans':'Spherical two-means','centroid-sweep':'Centroid sweep','antipodal-sweep':'Antipodal sweep','inertia-sweep':'Inertia sweep','hybrid-sweep':'Hybrid sweep'}
m={r['method']:r for r in s['methods']}; top=max(m.values(),key=lambda r:r['gain_pct']); base=m['random-sweep']; c=m['centroid-sweep']; a=m['antipodal-sweep']; h=m['hybrid-sweep']; i=m['inertia-sweep']; l=m['spherical-kmeans']; g=m['gw']
def gain(r): return f"{r['gain_pct']:.3f}\\%"
def pp(r): return f"{r['advantage_pp']:+.3f} pp"
def ci(r): return f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
def wt(r): return f"{r['wins']}/{r['ties']}/{r['losses']}"
macros={
'PrincipalFinding':f"The {names[top['method']].lower()} reached a mean gain of {gain(top)}, versus {gain(base)} for random sweep ({pp(top)}). All three cut-guided variants improved on random sweep in the equal-graph mean; inertia-only search did not.",
'ConvergenceFinding':f"Repeated GW achieves {gain(g)}, random sweep {gain(base)}, and the centroid, antipodal and hybrid variants {gain(c)}, {gain(a)} and {gain(h)}, respectively. Gains are measured relative to the common initial cut.",
'EffectFinding':f"Centroid: {pp(c)}, interval {ci(c)}; antipodal: {pp(a)}, interval {ci(a)}; hybrid: {pp(h)}, interval {ci(h)}. These intervals exclude zero, but do not establish a ranking among the three guided variants.",
'CostFinding':f"Mean archived seconds are {base['seconds']:.3f} for random sweep and {c['seconds']:.3f} for centroid sweep; inertia sweep takes {i['seconds']:.3f}. The latter yields lower cut quality despite examining a similar number of intervals.",
'RecordFinding':f"Random sweep registers {base['records']:.2f} strict records on average, compared with {c['records']:.2f} for centroid sweep. Spherical two-means stops after {l['attempts']:.2f} attempts on average. More records do not imply a better final incumbent.",
'GraphFinding':f"Centroid wins on {c['wins']} of 71 graphs, antipodal on {a['wins']}, and hybrid on {h['wins']}. Their respective losses ({c['losses']}, {a['losses']}, {h['losses']}) rule out universal dominance.",
'GeometryFinding':f"Direct inertia minimization attains {gain(i)} versus {gain(base)} for random sweep ({pp(i)}). Spherical two-means achieves {gain(l)}. The cut-guided variants perform better on average than these geometry-only updates.",
'AuditFinding':f"The completed audit covered {s['audited_records']:,} records, including {s['new_records']:,} new and {s['reused_records']:,} reused records; {s['sampled_event_records']:,} sampled event intervals were archived. All methods recorded zero invalid candidate flags."
}
rows=[]
for r in s['methods']:
 name=names[r['method']]
 if r is top:name=r'\textbf{'+name+'}'
 rows.append(f"{name} & {r['gain_pct']:.3f} & {r['advantage_pp']:+.3f} & {ci(r)} & {wt(r)} "+r'\\')
(ROOT/'results.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n\\newcommand{\\ResultRows}{\n'+'\n'.join(rows)+'\n}\n',encoding='utf-8')
