"""Deterministic, stdlib-only SVG of Experiment030 paired raw-field effects.

Usage (from repository root):
  python scripts/experiment030_plot_effects.py
  python scripts/experiment030_plot_effects.py --check
This script reads only archived Experiment030 CSV; it never recomputes EVA/clustering.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
from pathlib import Path
from xml.sax.saxutils import escape

ROOT=Path('results/gset/experiment030-local-field-clustering')
SOURCE=ROOT/'per_graph_method.csv'
TARGET=ROOT/'raw_field_relative_effects.svg'
GRAPHS=('G1','G6','G11','G14','G18')
MODES=('default','full')
COLORS={'default':'#14698c','full':'#cf7d37'}

def generate(source:Path=SOURCE)->str:
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    with source.open(newline='',encoding='utf8') as f:
        data={(x['graph'],x['rank_mode'],x['method']):x for x in csv.DictReader(f)}
    expect={(g,m,'raw_minus_h') for g in GRAPHS for m in MODES}
    if not expect.issubset(data):raise ValueError('Archived raw-field rows incomplete')
    width,height=1110,690
    left,right=370,1030
    ytop,step=168,94
    low,high=-.22,.50
    def sx(v):return left+(v-low)/(high-low)*(right-left)
    def txt(x,y,s,size=18,color='#233a50',weight='400',anchor='start'):
        return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" font-family="Arial, Helvetica, sans-serif" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{escape(str(s))}</text>'
    items=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="1110" height="690" fill="#fbfcfe"/>',
           txt(34,45,'Experiment030: signed local-field weighting',29,'#0c3c61','700'),
           txt(34,78,'Raw -H versus original Y · antipodal K-MeansNM · 100 EVA seeds per graph and rank',17,'#546b7f'),
           txt(35,116,'Mean relative change in original signed Max-Cut (%) and 95% paired seed-bootstrap CI',17,'#546b7f')]
    for tick in (-.2,-.1,0,.1,.2,.3,.4,.5):
        x=sx(tick)
        items.append(f'<line x1="{x:.2f}" y1="145" x2="{x:.2f}" y2="632" stroke="{("#92a7b8" if tick==0 else "#e1e8ee")}" stroke-width="{(2.4 if tick==0 else 1)}" stroke-dasharray="{("0" if tick==0 else "4 6")}"/>')
        items.append(txt(x,651,f'{tick:+.1f}%',15,anchor='middle'))
    for gi,g in enumerate(GRAPHS):
        yy=ytop+gi*step
        if gi%2==0:items.append(f'<rect x="26" y="{yy-21}" width="1058" height="74" fill="#ecf3f8" opacity="0.66" rx="9"/>')
        items.append(txt(35,yy+13,g,23,'#183b55','700'))
        for mode,dy in (('default',0),('full',29)):
            row=data[(g,mode,'raw_minus_h')]
            b=float(row['mean_baseline'])
            m=100*float(row['mean_delta'])/b
            lo=100*float(row['ci95_low'])/b
            hi=100*float(row['ci95_high'])/b
            if not (low<lo<high and low<hi<high):raise ValueError('Plot would clip an interval')
            sy=yy+dy
            color=COLORS[mode]
            items.extend([f'<line x1="{sx(lo):.2f}" y1="{sy:.2f}" x2="{sx(hi):.2f}" y2="{sy:.2f}" stroke="{color}" stroke-width="4"/>',
                f'<line x1="{sx(lo):.2f}" y1="{sy-6:.2f}" x2="{sx(lo):.2f}" y2="{sy+6:.2f}" stroke="{color}" stroke-width="2"/>',
                f'<line x1="{sx(hi):.2f}" y1="{sy-6:.2f}" x2="{sx(hi):.2f}" y2="{sy+6:.2f}" stroke="{color}" stroke-width="2"/>',
                f'<circle cx="{sx(m):.2f}" cy="{sy:.2f}" r="6.5" fill="{color}" stroke="white" stroke-width="1.5"/>',
                txt(101,sy+6,('p=41' if mode=='default' else 'p=800'),16,color,'650'),
                txt(285,sy+6,f'{m:+.3f}%',17,color,'700',anchor='end')])
    items.append(txt(35,676,'Blue = p=41  ·  Orange = p=800  ·  5 fixed graphs  ·  Unadjusted intervals; between-graph uncertainty not estimated',14,'#536a7f'))
    items.append(f'<!-- source-sha256:{digest}; generator: scripts/experiment030_plot_effects.py; no optimization -->')
    items.append('</svg>')
    return '\n'.join(items)+'\n'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--check',action='store_true')
    ap.add_argument('--source',type=Path,default=SOURCE)
    ap.add_argument('--output',type=Path,default=TARGET)
    args=ap.parse_args();text=generate(args.source)
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding='utf8')!=text:
            raise SystemExit('Experiment030 SVG differs from deterministic source')
        print('PASS: Experiment030 deterministic forest plot')
    else:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text,encoding='utf8',newline='\n')
        print(f'Wrote {args.output}')

if __name__=='__main__':main()