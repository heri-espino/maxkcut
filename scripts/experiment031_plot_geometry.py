"""Deterministic standard-library SVG figures for Experiment031 (no solver)."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

ROOT=Path('results/gset/experiment031-signed-gw-geometry')
GRAPHS=('G1','G6','G11','G14','G18')


def data(path):
    with path.open(newline='',encoding='utf8') as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=10 or {(z['graph'],z['rank_mode']) for z in rows}!={(g,m) for g in GRAPHS for m in ('default','full')}:
        raise ValueError('Expected exactly 5 graphs x 2 embedding widths')
    return {(r['graph'],r['rank_mode']):r for r in rows}


def make_angles(rows):
    W,H=850,440
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Experiment031 positive versus negative signed edge angles">',
    '<rect width="850" height="440" fill="#ffffff"/>',
    '<text x="34" y="36" font-size="19" font-family="sans-serif" font-weight="700" fill="#1b3141">Signed-edge EVA angles</text>',
    '<text x="34" y="55" font-size="12" font-family="sans-serif" fill="#64788a">100 seeds per graph · p=41 · means weighted by |edge weight|</text>']
    x0,x1=145,776;scale=(x1-x0)/180
    for tick in range(0,181,45):
        x=x0+tick*scale
        out.append(f'<line x1="{x:.1f}" y1="86" x2="{x:.1f}" y2="386" stroke="#e0e5eb"/>')
        out.append(f'<text x="{x:.1f}" y="404" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#64788a">{tick}°</text>')
    for k,g in enumerate(GRAPHS):
        r=rows[(g,'default')]
        y=110+59*k
        out.append(f'<text x="25" y="{y+19}" font-family="sans-serif" font-size="15" font-weight="700" fill="#1b3141">{g}</text>')
        a=float(r['mean_positive_angle_deg']);neg=r['mean_negative_angle_deg']
        out.append(f'<rect x="{x0}" y="{y}" width="{a*scale:.2f}" height="17" rx="3" fill="#377dae"/>')
        out.append(f'<text x="{x0+a*scale+6:.2f}" y="{y+13}" font-family="sans-serif" font-size="12" fill="#1b3141">{a:.1f}°</text>')
        if neg not in ('','None'):
            b=float(neg);out.append(f'<rect x="{x0}" y="{y+21}" width="{b*scale:.2f}" height="17" rx="3" fill="#d48c3a"/>')
            out.append(f'<text x="{x0+b*scale+6:.2f}" y="{y+34}" font-family="sans-serif" font-size="12" fill="#1b3141">{b:.1f}°</text>')
        else:out.append(f'<text x="{x0+7}" y="{y+34}" font-family="sans-serif" font-size="11" fill="#64788a">No negative edges</text>')
    out+=['<rect x="36" y="422" width="11" height="11" fill="#377dae"/><text x="53" y="431" font-size="12" font-family="sans-serif" fill="#1b3141">Positive</text>',
    '<rect x="157" y="422" width="11" height="11" fill="#d48c3a"/><text x="174" y="431" font-size="12" font-family="sans-serif" fill="#1b3141">Negative</text>','</svg>']
    return '\n'.join(out)+'\n'


def make_projection(rows):
    W,H=850,440
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Experiment031 negative-neighbor field ablation and GW invariance">',
    '<rect width="850" height="440" fill="#fff"/>',
    '<text x="34" y="35" font-family="sans-serif" font-size="19" font-weight="700" fill="#1b3141">Which GW labels depend on negative neighbors?</text>',
    '<text x="34" y="55" font-family="sans-serif" font-size="12" fill="#64788a">p=41 · same GW normals · negative-field removal evaluated at frozen Y, not reoptimized</text>']
    x0=150;scale=520/20
    for tick in (0,5,10,15,20):
        x=x0+tick*scale
        out.append(f'<line x1="{x:.1f}" y1="88" x2="{x:.1f}" y2="383" stroke="#e0e5eb"/>')
        out.append(f'<text x="{x:.1f}" y="401" font-size="11" font-family="sans-serif" text-anchor="middle" fill="#64788a">{tick}%</text>')
    for k,g in enumerate(GRAPHS):
        r=rows[(g,'default')]
        frac=100*float(r['negative_ablation_fraction'])
        mismatch=100*float(r['mismatch_fraction'])
        y=109+k*57
        out.append(f'<text x="30" y="{y+22}" font-family="sans-serif" font-size="15" font-weight="700" fill="#1b3141">{g}</text>')
        if frac:
            out.append(f'<rect x="{x0}" y="{y+4}" width="{frac*scale:.1f}" height="24" rx="3" fill="#d48c3a"/>')
            out.append(f'<text x="{x0+frac*scale+8:.1f}" y="{y+22}" font-family="sans-serif" font-size="13" fill="#1b3141">{frac:.2f}%</text>')
        else:out.append(f'<text x="{x0+7}" y="{y+22}" font-family="sans-serif" font-size="12" fill="#64788a">0% (positive-only control)</text>')
        out.append(f'<text x="700" y="{y+22}" font-family="sans-serif" font-size="11" fill="#1b3141">Y/−H: {mismatch:.4f}%</text>')
    out+=['<text x="34" y="426" font-size="11" font-family="sans-serif" fill="#64788a">Bars: fraction of vertex signs changed by deleting negative field terms. Right: finite-tolerance mismatch of Y and −H.</text>','</svg>']
    return '\n'.join(out)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true')
    p.add_argument('--root',type=Path,default=ROOT)
    args=p.parse_args()
    rows=data(args.root/'per_graph_rank.csv')
    for name,content in [('signed_edge_angles.svg',make_angles(rows)),('gw_negative_field_ablation.svg',make_projection(rows))]:
        target=args.root/name
        if args.check:
            if not target.is_file() or target.read_text(encoding='utf8')!=content:
                raise SystemExit('Not deterministic or missing: '+str(target))
        else:
            target.write_text(content,encoding='utf8',newline='\n')
        print('Verified' if args.check else 'Generated',target)

if __name__=='__main__':main()