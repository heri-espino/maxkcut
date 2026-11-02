#!/usr/bin/env python3
"""Deterministic compact SVG figures for the expanded maxkcut Marp deck.
Numbers transcribed from verified Experiment021/022 reports and original-edge audits.
No numerical search, EVA solve or model fit is performed here.
"""
from pathlib import Path
from math import pi, sin, cos
import random
from html import escape
R=Path(__file__).parent/'assets';R.mkdir(exist_ok=True)
BLUE='#113554';TEAL='#0b718b';SKY='#3b9cbd';MUTED='#64748b';DARK='#0b2944';BG='#f7fafc';GOLD='#e0a32c';LIGHT='#e3eef5';INK='#1d415c';RED='#c06164'

def svg(w,h,parts,name):
    header=f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><style>text{{font-family:Inter,Arial,sans-serif}}</style><rect width="{w}" height="{h}" fill="{BG}" rx="12"/>'
    (R/name).write_text(header+''.join(parts)+'</svg>',encoding='utf-8')
def rect(x,y,w,h,fill,rx=0,stroke='none',sw=0):return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
def t(x,y,text,size=18,color=INK,weight=500,anchor='start'):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(str(text))}</text>'
def line(x,y,X,Y,stroke='#a5c2d4',width=2,dash=''):return f'<line x1="{x}" y1="{y}" x2="{X}" y2="{Y}" stroke="{stroke}" stroke-width="{width}" stroke-dasharray="{dash}"/>'
def circle(x,y,r,fill,stroke='#fff',sw=1.5):return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'

# G-set family composition: 30,24,17.
p=[t(30,45,'G-set generator families',28,BLUE,800)]
for i,(lab,count,col) in enumerate([('Random',30,TEAL),('Almost-planar family',24,SKY),('Toroidal',17,GOLD)]):
    y=94+i*78;p +=[t(35,y+20,lab,19,BLUE,600),rect(260,y-3,570,36,LIGHT,9),rect(260,y-3,count/30*570,36,col,9),t(850,y+23,str(count)+' graphs',18,col,800)]
p+=[t(32,348,'71 original signed/positive weighted benchmark instances — family labels are generator metadata',17,MUTED,500)]
svg(1000,370,p,'families.svg')

# Three conceptual small graphs; no claims about actual embedded coordinates.
p=[t(26,38,'Connectivity patterns — schematics, not benchmark coordinates',24,BLUE,750)]
random.seed(10)
for j,(name,color) in enumerate([('Random edges',TEAL),('Almost-planar local links',SKY),('Toroidal / periodic grid',GOLD)]):
    x0=34+j*322
    p += [rect(x0,55,292,290,'#ffffff',14,'#dae8f0',1),t(x0+14,82,name,19,BLUE,700)]
    if j==0:
        coords=[(x0+33+random.random()*221,108+random.random()*178) for _ in range(18)]
        for k in range(21):
            a,b=random.sample(range(len(coords)),2);A,B=coords[a],coords[b];p.append(line(*A,*B,color,1.25))
    elif j==1:
        coords=[]
        for row in range(4):
            for col in range(5):coords.append((x0+43+col*48+random.randrange(-9,10),112+row*49+random.randrange(-10,11)))
        for row in range(4):
            for col in range(5):
                a=row*5+col
                if col<4:p.append(line(*coords[a],*coords[a+1],color,1.5))
                if row<3:p.append(line(*coords[a],*coords[a+5],color,1.5))
        for a,b in [(0,6),(2,8),(13,19)]:p.append(line(*coords[a],*coords[b],color,1.2))
    else:
        coords=[(x0+42+c*49,108+r*52) for r in range(4) for c in range(5)]
        for r in range(4):
            for c in range(5):
                a=r*5+c
                if c<4:p.append(line(*coords[a],*coords[a+1],color,1.6))
                if r<3:p.append(line(*coords[a],*coords[a+5],color,1.6))
        # conceptual periodic wrap edges shown curved
        for r in range(4):
            a=coords[r*5];b=coords[r*5+4];p.append(f'<path d="M {a[0]} {a[1]} Q {x0+150} {a[1]-38} {b[0]} {b[1]}" fill="none" stroke="{color}" stroke-dasharray="5 4" opacity=".78"/>')
    for xy in coords:p.append(circle(*xy,5,color))
p.append(t(30,380,'Topological features must come from original edges, not the drawing algorithm',18,MUTED,600))
svg(1000,401,p,'topology_schematics.svg')

# graph feature case examples, observed originals
examples=[('G1','random','47.94','0.132','0.059'),('G14','planar-family','11.74','0.983','0.154'),('G11','toroidal','4.00','0','0'),('G48','toroidal','4.00','0','0')]
p=[t(30,40,'Structural signatures measured from original edges',25,BLUE,750)]
cols=[40,155,400,570,740]
for x,s in zip(cols,['Graph','Generator family','Mean degree','Degree CV','Transitivity']):p.append(t(x,95,s,18,TEAL,750))
for i,rec in enumerate(examples):
    y=126+i*58;p.append(rect(20,y-18,950,48,'#edf4f8' if i%2==0 else '#ffffff',7))
    for x,s in zip(cols,rec):p.append(t(x,y+11,s,18,BLUE,650 if x==40 else 500))
p.append(t(30,384,'G14 exceeds the strict planar graph edge bound — “planar-family” is a benchmark label',16,RED,650))
svg(1000,410,p,'graph_signatures.svg')

# best-cut heatmap 9x3 from observed 022 method_by_family.csv
win={
'paired':[7,2,4], 'independent':[8,5,4], 'sample-antipodal':[10,4,5],
'sample-pair':[5,5,4], 'farthest-pair':[9,3,6], 'spherical-pp':[7,4,7],
'mean-perturb':[7,4,6], 'pca-antipodal':[7,3,6], 'gw-screen16':[5,8,11]}
labels=['Original paired','Independent','Sample-antipodal','Sample-pair','Farthest-pair','Spherical++','Mean-perturb','PCA-antipodal','GW screen-16']
p=[t(32,41,'Best observed cut: number of graphs won or tied',25,BLUE,750)]
for i,head in enumerate(['Random · 30','Planar family · 24','Toroidal · 17']):p.append(t(565+140*i,86,head,15,TEAL,700,'middle'))
for i,(method,counts) in enumerate(win.items()):
    y=111+i*40;p.append(t(34,y+19,labels[i],17,BLUE,650 if method=='gw-screen16' else 500))
    for j,v in enumerate(counts):
        x=507+j*140; intensity=v/11
        col=['#e1f0f5','#cae7f0','#a7d2e0','#75b7cb','#2a8da9','#0c607e'][min(5,int(v/2))]
        p.append(rect(x,y,112,32,col,7));p.append(t(x+56,y+22,v,18, '#fff' if v>7 else BLUE,800,'middle'))
p.append(t(30,502,'All ties count. GW screen-16 performs 16 extra trial-cut evaluations per restart.',16,MUTED,600))
svg(965,524,p,'topology_wins.svg')

# GW screen 16 frequencies, with explicit all-ties
p=[t(28,39,'GW screen-16: topology association',26,BLUE,800)]
for i,(lab,n,k) in enumerate([('Random',30,5),('Planar-family',24,8),('Toroidal',17,11)]):
    y=89+92*i;p+=[t(28,y+23,lab,20,BLUE,680),rect(245,y,607,37,LIGHT,9),rect(245,y,k/n*607,37,[TEAL,SKY,GOLD][i],9),t(872,y+25,f'{k}/{n} ({100*k/n:.1f}%)',19,BLUE,760)]
p +=[t(28,383,'Caution: G48–G50 tie across all 9 methods. Informative toroidal: 8/14.',17,RED,600)]
svg(1000,405,p,'gw_topology.svg')

# Leave one graph out impact control.
p=[t(28,42,'Held-out graph method selection',25,BLUE,800),t(30,74,'Performance of a method chosen without using the held-out graph',16,MUTED)]
vals=[('Globally trained selector',0.7323943662,TEAL),('Family-trained selector',0.1126760563,SKY)]
for i,(lab,val,col) in enumerate(vals):
    y=118+i*95;p+=[t(28,y+25,lab,20,BLUE,600),rect(340,y,555,42,LIGHT,8),rect(340,y,555*val/0.85,42,col,8),t(910,y+29,f'+{val:.3f}',20,BLUE,750)]
p+=[t(27,357,'The family rule loses to one globally chosen initializer in leave-one-graph-out analysis.',17,RED,650),t(28,383,'Outcome: mean absolute best-cut gain vs original paired method, in cut-weight units.',15,MUTED)]
svg(1000,404,p,'heldout_policy.svg')

# raw round winners by family
p=[t(30,42,'Historical 20-start raw rounding: family best/ties',26,BLUE,800)]
for i,(lab,total,a,b,c) in enumerate([('Random',27,26,1,0),('Planar-family',23,23,0,0),('Toroidal',12,12,3,2)]):
    y=96+i*98;p.append(t(30,y+24,lab,20,BLUE,750))
    for j,(k,clr) in enumerate([(a,TEAL),(b,SKY),(c,GOLD)]):
        x=270+j*226;p.append(rect(x,y,172,33,LIGHT,8));p.append(rect(x,y,k/27*172,33,clr,8));p.append(t(x+185,y+23,k,18,BLUE,700))
for j,(name,clr) in enumerate([('Antipodal K-MeansNM',TEAL),('Independent K-Means2N',SKY),('GW/FJ hyperplane',GOLD)]):
    x=30+j*310;p.extend([rect(x,409,17,17,clr,3),t(x+25,423,name,16,BLUE,540)])
p.append(t(30,470,'62 graphs archived; same 20 initial attempts. These are NOT 71-graph final pipeline results.',15,MUTED,580))
svg(1000,488,p,'historical_raw.svg')

# G63 archive best-cut contextual bars measured, compare 25k-27k; adapted labels
methods=[('K-means + Flip + KL + ILS',26917,TEAL),('Elite EM: retained source',26917,'#b8a15c'),('Cut-aware + KL + ILS',26909,SKY),('Farthest-pair full protocol',26906,SKY),('Cut-aware + KL',26877,'#92a9bf'),('Mean-start EM + KL',26834,'#a7b7c7'),('K-means + Flips',26682,'#adb9c7'),('Multi-edge MST + flips',25761,RED)]
p=[t(28,38,'G63: historical best scores at heterogeneous budgets',23,BLUE,800)]
for i,(name,k,col) in enumerate(methods):
    y=70+i*48;p += [t(24,y+20,name,17,BLUE,650 if i<2 else 500),rect(350,y+1,545,26,LIGHT,6),rect(350,y+1,max(0,(k-25000)/2000*545),26,col,6),t(908,y+21,f'{k:,}',17,BLUE,740)]
p +=[t(27,481,'Elite EM matches 26,917 because its verified input already achieved 26,917.',16,RED,650)]
svg(1000,505,p,'historical_g63.svg')

# evidence breadth
p=[t(30,42,'How much evidence exists for each method family?',26,BLUE,800)]
for i,(lab,count,col) in enumerate([('Nine init. + full pipeline · Exp021',71,TEAL),('Raw GW / K-means · Exp002/003/008',62,SKY),('EM, MST, multigroup, cut-aware · G63',1,GOLD),('Experiment023 verified final report',0,RED)]):
    y=90+i*90;p+=[t(32,y,lab,18,BLUE,660),rect(34,y+12,805,28,LIGHT,7)]
    if count:p.append(rect(34,y+12,805*count/71,28,col,7))
    p.append(t(858,y+35,f'{count}/71',19,BLUE,800))
p.append(t(30,481,'Coverage is a property of saved verified results—not an indication that a method was never implemented.',15,MUTED))
svg(1000,504,p,'evidence_coverage.svg')

# 17 protocol family taxonomy (draw boxes)
groups=[('Control',2,BLUE),('Hyperplane / cut-aware',3,TEAL),('Multi-group',2,SKY),('Soft EM / sampling',3,'#688ab9'),('MST variants',7,GOLD)]
p=[t(30,42,'Experiment023 — 17 registered protocols',27,BLUE,800)];x=34
for name,count,col in groups:
    w=count*52+82;p.extend([rect(x,95,w,123,'#ffffff',12,'#ccdde6',1),rect(x,95,w,9,col,4),t(x+17,144,name,18,BLUE,740),t(x+17,193,str(count)+' protocols',19,col,800)]);x+=w+12
p.extend([t(35,277,'All methods → SAME 20 Flips → 30 KL passes → 80 ILS cycles',20,BLUE,700),line(55,307,940,307,TEAL,3),circle(57,307,7,TEAL),circle(940,307,7,TEAL),t(35,350,'Exception: Tabu control changes the refiner. MST preprocessing is costly/deterministic.',18,RED,580),t(35,392,'This slide shows the registered design; it is not a graph of completed outcomes.',17,MUTED)])
svg(1000,417,p,'exp023_taxonomy.svg')

# Exp023 process gate diagram
p=[t(26,40,'Experiment023: pilot-gated run and verification',25,BLUE,800)]
boxdata=[('Pilot','G1 · G14 · G11','three generators'),('Full sweep','71 × 17 cells','1,207 scheduled'),('Coverage audit','verified vs missing','hard timeouts kept')]
for i,(title,sub,note) in enumerate(boxdata):
    x=25+i*323;p.extend([rect(x,95,288,175,'#ffffff',15,'#d1e1eb',1.5),rect(x,95,288,12,[TEAL,SKY,GOLD][i],8),t(x+20,155,title,26,BLUE,780),t(x+20,194,sub,19,BLUE,600),t(x+20,230,note,16,MUTED)])
    if i<2:p.extend([line(x+290,173,x+316,173,TEAL,3),f'<path d="M {x+305} 167 L {x+315} 173 L {x+305} 179" stroke="{TEAL}" fill="none" stroke-width="3"/>'])
p.extend([t(28,326,'Stop condition: if pilot fails, do not claim the 71-graph run commenced.',18,RED,650),t(28,365,'Persist per-case graph/factor SHA, stage cuts, runtime, seed and failure status.',16,MUTED)])
svg(1000,393,p,'exp023_pipeline.svg')

# taxonomy of empirical vs planned flags
p=[t(30,42,'Evidence boundary: what can we report today?',26,BLUE,800)]
for i,(label,headline,detail,col) in enumerate([
 ('VERIFIED','Experiment021','71 × 9 × 50 = 31,950 complete searches',TEAL),
 ('RETROSPECTIVE','Experiment022','Topology associations + historical G63 audit',SKY),
 ('NOT VERIFIED COMPLETE','Experiment023','17 × 71 registered; final results absent',RED)
]):
    y=80+i*122;p+=[rect(26,y,946,99,'#ffffff',13,'#dce8ee',1),rect(26,y,9,99,col,6),t(51,y+28,label,14,col,800),t(50,y+63,headline,26,BLUE,760),t(385,y+60,detail,17,MUTED,500)]
svg(1000,485,p,'evidence_boundary.svg')

print('Generated',len(list(R.glob('*.svg'))),'total assets, including copied prior figures')