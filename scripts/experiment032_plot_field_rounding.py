"""Deterministic, standard-library SVG charts from Experiment032's audited 71-graph CSV.

Usage: python scripts/experiment032_plot_field_rounding.py [--check]
This script NEVER loads EVA factors or reruns any rounding.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import math
from pathlib import Path
from xml.sax.saxutils import escape

BASE=Path('results/gset/experiment032-all-gset-field-rounding')
ASSETS=Path('documents/presentations/assets')
CHART1='experiment032_km_relative_by_graph.svg'
CHART2='experiment032_field_effect_by_weight_type.svg'


def load():
    source=BASE/'per_graph.csv'
    raw=source.read_bytes()
    rows=list(csv.DictReader(raw.decode('utf8').splitlines()))
    if len(rows)!=71 or len({r['graph'] for r in rows})!=71:
        raise ValueError('Only a complete 71-graph audited input is valid')
    if any(r['weight_type'] not in ('positive','signed') for r in rows):
        raise ValueError('Unexpected sign family')
    digest=hashlib.sha256(raw).hexdigest()
    return sorted(rows,key=lambda r:int(r['graph'][1:])),digest


def xml(s):return escape(str(s),{'"':'&quot;'})


def begin(w,h,label,sha):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="{xml(label)}">',
        '<rect x="0" y="0" width="100%" height="100%" rx="14" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;} .title{font-size:23px;font-weight:700;fill:#153047;} .sub{font-size:14px;fill:#5a7283} .tick{font-size:13px;fill:#516b7b} .annot{font-size:13px;fill:#173b51;font-weight:600}</style>',
        f'<!-- deterministic_source_sha256={sha}; n_graphs=71; no numerical rerun -->']


def chart_distribution(rows,sha):
    w,h=1140,520
    v=[float(z['KM_minusH_relative_pct']) for z in rows]
    maxabs=max(max(abs(x) for x in v),0.02)
    axis=math.ceil(maxabs*1.05*100)/100
    l,r,t,b=88,w-32,90,h-95
    zero=t+(b-t)/2
    y=lambda value:zero-(value/axis)*(b-t)/2
    xslot=(r-l)/len(v)
    out=begin(w,h,'Experiment032: K-MeansNM relative change from -WY across all 71 G-set graphs',sha)
    out.extend([f'<text x="{l}" y="40" class="title">K-MeansNM: local-field versus unit embeddings</text>',
                f'<text x="{l}" y="64" class="sub">Mean paired signed-cut change from -WY, 50 common starts per frozen EVA factor (one factor per graph)</text>'])
    for frac in (-1,-.5,0,.5,1):
        yy=y(frac*axis)
        out.append(f'<line x1="{l}" y1="{yy:.1f}" x2="{r}" y2="{yy:.1f}" stroke="#{"547585" if frac==0 else "e4edf2"}" stroke-width="{1.7 if frac==0 else 1}" stroke-dasharray="{ "" if frac==0 else "4 4" }"/>')
        out.append(f'<text x="{l-9}" y="{yy+5:.1f}" class="tick" text-anchor="end">{frac*axis:+.2f}%</text>')
    for i,(row,delta) in enumerate(zip(rows,v)):
        x=l+i*xslot+1.5
        width=max(2.,xslot-2.6)
        yy=y(delta)
        top=min(yy,zero)
        height=max(1.5,abs(yy-zero))
        color='#276f9c' if row['weight_type']=='signed' else '#ce8752'
        out.append(f'<rect x="{x:.2f}" y="{top:.2f}" width="{width:.2f}" height="{height:.2f}" rx="1.2" fill="{color}"/>')
        if i==0 or i==70 or ((i+1)%10==0 and i<69):
            out.append(f'<text x="{x+width/2:.1f}" y="{b+20}" class="tick" text-anchor="middle">{xml(row["graph"])}</text>')
    out.extend([f'<text x="{(l+r)/2}" y="{h-54}" class="sub" text-anchor="middle">Canonical G-set instance (numeric order)</text>',
                f'<rect x="{l}" y="{h-36}" width="13" height="13" rx="2" fill="#276f9c"/>',
                f'<text x="{l+20}" y="{h-25}" class="tick">Signed edges present</text>',
                f'<rect x="{l+235}" y="{h-36}" width="13" height="13" rx="2" fill="#ce8752"/>',
                f'<text x="{l+255}" y="{h-25}" class="tick">Positive-only graph</text>',
                f'<text x="{r}" y="{h-25}" class="tick" text-anchor="end">Zero = identical mean cut; higher is better</text>',
                '</svg>'])
    return '\n'.join(out)+'\n'


def chart_groups(rows,sha):
    w,h=990,405
    groups=[('GW','Positive-only', 'GW_minusH_relative_pct','positive'),('GW','Signed', 'GW_minusH_relative_pct','signed'),
            ('K-MeansNM','Positive-only','KM_minusH_relative_pct','positive'),('K-MeansNM','Signed','KM_minusH_relative_pct','signed')]
    vals=[]
    for algorithm,kind,key,sign in groups:
        data=[float(r[key]) for r in rows if r['weight_type']==sign]
        vals.append((algorithm,kind,float(sum(data)/len(data)),len(data),sign))
    maxabs=max(max(abs(q[2]) for q in vals),.02)
    axis=math.ceil(maxabs*1.2*100)/100
    l,r,t,b=248,w-38,92,h-70
    zero=l+(r-l)/2
    xx=lambda d:zero+(d/axis)*(r-l)/2
    out=begin(w,h,'All G-set: graph-equal mean percentage changes by algorithm and sign family',sha)
    out.extend(['<text x="38" y="40" class="title">Across all 71 graphs: graph-equal relative effects</text>',
        '<text x="38" y="63" class="sub">Relative mean signed cut changes from -WY compared with original Y; one vote per graph</text>'])
    for f in (-1,-.5,0,.5,1):
        x=xx(f*axis)
        out.append(f'<line x1="{x:.1f}" y1="{t-14}" x2="{x:.1f}" y2="{b+2}" stroke="#{"47718a" if f==0 else "e1eaf0"}" stroke-width="{2 if f==0 else 1}"/>')
        out.append(f'<text x="{x:.1f}" y="{b+26}" text-anchor="middle" class="tick">{f*axis:+.3f}%</text>')
    for i,(alg,kind,delta,n,sign) in enumerate(vals):
        yy=t+22+i*62
        x=xx(delta)
        color='#276f9c' if sign=='signed' else '#ce8752'
        out.append(f'<rect x="{min(x,zero):.2f}" y="{yy-12}" width="{max(1.6,abs(x-zero)):.2f}" height="25" rx="3" fill="{color}"/>')
        out.append(f'<text x="38" y="{yy+5}" class="annot">{xml(alg)} · {xml(kind)} (n={n})</text>')
        right=x+10 if delta>=0 else x-10
        anchor='start' if delta>=0 else 'end'
        out.append(f'<text x="{right:.1f}" y="{yy+5}" class="annot" text-anchor="{anchor}">{delta:+.5f}%</text>')
    out.extend([f'<text x="{(l+r)/2:.1f}" y="{h-18}" class="sub" text-anchor="middle">Averaged per-graph percentage, not a pooled raw-cut estimate; exploratory fixed 71-graph corpus</text>', '</svg>'])
    return '\n'.join(out)+'\n'


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--check',action='store_true');a=ap.parse_args()
    rows,sha=load()
    generated={CHART1:chart_distribution(rows,sha),CHART2:chart_groups(rows,sha)}
    if not a.check:ASSETS.mkdir(parents=True,exist_ok=True)
    for name,body in generated.items():
        dest=ASSETS/name
        if a.check:
            if not dest.is_file() or dest.read_text(encoding='utf8')!=body:
                raise SystemExit('Non-reproducible Experiment032 figure: '+str(dest))
            print('Verified',dest)
        else:
            dest.write_text(body,encoding='utf8',newline='\n')
            print('Wrote',dest)

if __name__=='__main__':main()