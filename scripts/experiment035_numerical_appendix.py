"""Build E035's standalone numerical appendix from the audited report directory.
Usage: python build_appendix.py REPORT_DIRECTORY OUTPUT.pdf
Dependencies: reportlab, pymupdf. Does not run numerical optimization.
"""
import csv, json, sys
from pathlib import Path
from io import BytesIO
import fitz
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph

root, out = map(Path, sys.argv[1:3])
s = json.loads((root/'summary.json').read_text())
assert s['complete'] and s['graphs']==71 and s['paired_trajectories']==1136
with (root/'patience_summary.csv').open(newline='') as f:
    rows=list(csv.DictReader(f))
assert [int(r['patience']) for r in rows]==[5,10,20,50,100]
W,H=A4; M=42; CW=W-2*M
ink=HexColor('#18283B'); muted=HexColor('#4C6073'); rule=HexColor('#D0D8DF'); blue=HexColor('#20577A')
body=ParagraphStyle('body',fontName='Times-Roman',fontSize=11.2,leading=15,textColor=ink,spaceAfter=8)
small=ParagraphStyle('small',parent=body,fontSize=9.6,leading=12.6,textColor=muted)
result=fitz.open(); figures=[]

def start(title, section):
    b=BytesIO(); c=canvas.Canvas(b,pagesize=A4)
    c.setFillColor(blue); c.setFont('Helvetica-Bold',8); c.drawString(M,H-32,'MAX-CUT  /  EXPERIMENT 035')
    c.setFillColor(muted); c.setFont('Helvetica',8); c.drawRightString(W-M,H-32,section.upper())
    c.setStrokeColor(rule); c.line(M,H-42,W-M,H-42); c.line(M,38,W-M,38)
    c.setFont('Helvetica',8); c.setFillColor(muted); c.drawString(M,23,'Numerical appendix  |  26 November 2026')
    c.drawRightString(W-M,23,f'{len(result)+1:02d}')
    c.setFont('Times-Bold',22); c.setFillColor(ink); c.drawString(M,H-76,title)
    return b,c,H-96

def para(c,y,text,style=body):
    p=Paragraph(text,style); _,h=p.wrap(CW,700); p.drawOn(c,M,y-h); return y-h-10

def heading(c,y,text):
    c.setFont('Helvetica-Bold',10); c.setFillColor(blue); c.drawString(M,y-11,text);return y-22

def finish(b,c,fig=None,index=0,rect=None):
    c.showPage(); c.save(); d=fitz.open(stream=b.getvalue(),filetype='pdf');result.insert_pdf(d);d.close()
    if fig:
        with fitz.open(root/fig) as src:
            result[-1].show_pdf_page(fitz.Rect(*rect),src,index,keep_proportion=True)
        figures.append((fig,index))

b,c,y=start('Adaptive hyperplane search','Overview')
y=para(c,y,'<b>Numerical appendix: patience sensitivity and convergence diagnostics.</b> Repeated independent Goemans-Williamson (GW) rounding versus the random-tangent Hyperplane Sweep, with a fixed EVA embedding on each G-set graph.')
# compact key results band
for j,(v,label) in enumerate([(f"{s['mean_gw_total_gain_pct']:.3f}%",'GW mean gain'),(f"{s['mean_sweep_total_gain_pct']:.3f}%",'Sweep mean gain'),(f"{s['mean_paired_gain_pct']:.3f} pp",'Sweep advantage')]):
    x=M+j*CW/3; c.setFillColor(HexColor('#EEF3F6'));c.rect(x,y-67,CW/3-5,67,fill=1,stroke=0)
    c.setFillColor(ink);c.setFont('Times-Bold',22);c.drawString(x+9,y-29,v);c.setFont('Helvetica',9);c.drawString(x+9,y-48,label)
y-=84
y=heading(c,y,'Experimental design')
y=para(c,y,'71 graphs, 16 paired starts per graph (1,136 trajectories). Both methods start from the same GW cut and consume the same Gaussian draws. The original 100-draw histories and the patience-5 continuations are reused; the EVA factors remain fixed. The long trajectory stops after 100 consecutive draws without a strict improvement by either arm, or at 1,000 total draws. Shorter patience policies are exact prefixes of the same trajectories.')
y=heading(c,y,'Scoring and inference')
y=para(c,y,'Cuts are scored on the original signed weighted graph: C(x) = sum of w(i,j) over edges whose endpoints have different binary labels. A strict incumbent gain exceeds the 1e-8 tolerance. Percent gains use each graph\'s mean initial GW cut as denominator; graphs receive equal weight. The paired 95% graph-bootstrap interval for the final Sweep-minus-GW advantage is [{:.3f}, {:.3f}] percentage points (10,000 resamples). This is a descriptive benchmark interval, not a universal approximation guarantee.'.format(*s['graph_bootstrap_95_pct']))
y=heading(c,y,'Audit and scope')
y=para(c,y,f"{s['new_candidate_records']:,} new candidate records; {s['reused_prefix_records']:,} earlier records and {s['reused_e34_records']:,} patience-5 records reused. All candidate cuts independently rescored; no new EVA solves, numerical fallbacks or cap hits. Sweep wins on {s['graph_wins']} graph-level means, ties on {s['graph_ties']} and loses on {s['graph_losses']}. These are raw rounding results, not local-flip refinements or certified global optima.")
y=para(c,y,'Source: verified GitHub Actions run 45816398818, report artifact 15115708726; archived result commit ae82791f429ecc6dad36938c2b60775fb74c86cf. All figures reproduced from the audited report; no optimization rerun.',small)
assert y>48
finish(b,c)

b,c,y=start('Patience sensitivity','Numerical results')
y=para(c,y,'Table 1. Equal-graph means for each joint stopping policy. Gains are percentages relative to the common initial GW baseline; the advantage is the difference in percentage points. Time is additional method computation only, excluding archive and audit overhead.')
cols=[('Patience',46),('Draws',62),('GW gain',66),('Sweep gain',74),('Adv. (pp)',65),('GW (s)',65),('Sweep (s)',65)]
width=sum(w for _,w in cols); x0=M; c.setFillColor(ink);c.rect(x0,y-24,width,24,fill=1,stroke=0); x=x0
for label,w in cols:
    c.setFillColor(HexColor('#FFFFFF'));c.setFont('Helvetica-Bold',8.4);c.drawRightString(x+w-6,y-15,label);x+=w
y-=24
for i,r in enumerate(rows):
    c.setFillColor(HexColor('#EEF3F6') if i%2==0 else HexColor('#FFFFFF'));c.rect(x0,y-24,width,24,fill=1,stroke=0)
    vals=[r['patience'],f"{float(r['extra_draws']):.2f}",f"{float(r['gw_gain_pct']):.3f}%",f"{float(r['sweep_gain_pct']):.3f}%",f"{float(r['paired_advantage_pct']):.3f}",f"{float(r['gw_seconds']):.4f}",f"{float(r['sweep_seconds']):.4f}"]
    x=x0;c.setFont('Helvetica',9);c.setFillColor(ink)
    for v,(_,w) in zip(vals,cols): c.drawRightString(x+w-6,y-15,v);x+=w
    y-=24
y-=20
y=heading(c,y,'Interpretation')
y=para(c,y,'The Sweep gain rises from 4.265% at patience 5 to 4.496% at patience 100 (+0.231 percentage points). The GW gain rises from 2.431% to 2.697% (+0.266 points). The gap is largest descriptively at patience 20 (1.840 points) and remains positive at patience 100 (1.800 points). The paired intervals overlap substantially; no formal superiority of one patience policy over another is established.')
y=para(c,y,'Increasing patience from 5 to 100 increases mean additional draws from 6.00 to 191.89 (about 32 times). Mean Sweep time rises from 0.0111 s to 0.3036 s, and GW time from 0.00163 s to 0.04665 s. Sweep requires about 6.51 times the additional method time of GW at patience 100; matching Gaussian draws does not match computation or cut evaluations.')
y=para(c,y,'The strict record count at patience 100 averages 4.27 for Sweep and 0.91 for GW per trajectory across the post-100 segment. Only strict improvements count: monotone retention does not imply that every draw improves the solution. The five policies are nested counterfactual stopping points, not independent experimental replicates.')
y=heading(c,y,'Reading the figures')
y=para(c,y,'Figures 1–4 give aggregate convergence, stopping lengths, patience and geometric diagnostics. Figures 5a–5f contain every graph-specific convergence curve; their vertical scales are graph-specific. In all plots, blue denotes GW and orange denotes Sweep.')
assert y>48
finish(b,c)

pages=[
('Convergence of the incumbent','Figure 1','convergence.pdf',
 'Both methods retain the best feasible cut seen so far. The dashed line is draw 100, the endpoint of the initial shared benchmark. Stopped trajectories are held at their last incumbent so the cross-graph mean remains comparable across draws.',
 'The rapid initial gains flatten into a long tail of smaller improvements. Sweep remains above repeated GW throughout the aggregate trajectory. Final mean gains are 4.496% and 2.697%, respectively. Beyond draw 100 the extra gains are 0.257 and 0.286 percentage points; longer searching improves absolute quality but slightly narrows the mean gap.',(42,180,553,480)),
('When does the search stop?','Figure 2','stopping_distribution.pdf',
 'Histogram of additional Gaussian draws until 100 joint failures. Each observation is one paired start. The 1,000-total-draw cap was never reached.',
 'The median is 170 additional draws; the mean is 191.89, with minimum 100 and maximum 597. The right-skewed distribution indicates that some graph/start combinations continue to discover records much longer than others. The minimum is imposed by the patience rule, not by a mathematical convergence certificate.',(42,180,553,480)),
('Quality, cost and record frequency','Figure 3','patience_diagnostics.pdf',
 'Top row: final mean cut gain by patience and additional method time. Bottom row: strict improvement frequency by active draw, and empirical conditional improvement probability given the preceding failure streak (at least 30 observations per plotted streak).',
 'The quality/cost frontier shows diminishing returns. Sweep is more likely to record an improvement after a short failure streak, but the probability falls as the streak lengthens. Late iteration spikes are noisy: fewer trajectories remain active. Conditional probabilities are pooled descriptive estimates, not independent Bernoulli trials or evidence of stationarity.',(36,172,559,548)),
('Geometry of successive proposals','Figure 4','geometry_diagnostics.pdf',
 'The four panels report marginal incumbent gain, complement-invariant candidate-to-incumbent Hamming turnover, projective angular displacement arccos(|r_candidate dot r_incumbent|), and minority-side fraction. The last metric is min(f, 1-f), where f is the fraction of vertices in one side.',
 'Sweep usually returns a nearby partition and normal, while independent GW proposes a distant orientation and changes many labels. At draw 101, mean turnover is about 0.078% for Sweep versus 40.36% for GW; mean projective angle is 0.023 versus 1.463 radians. Both generate nearly balanced partitions. These are candidate-relative, not accepted-only, measurements. Active-only averages become increasingly selection-sensitive at late draws.',(36,172,559,548))]
for title,num,fig,intro,interpret,rect in pages:
    b,c,y=start(title,num);y=para(c,y,intro)
    y=H-rect[3]-20;y=heading(c,y,'Interpretation');y=para(c,y,interpret)
    assert y>48
    finish(b,c,fig,rect=rect)

with fitz.open(root/'all_71_curves.pdf') as src: n=len(src)
assert n==6
for i in range(n):
    b,c,y=start(f'Graphwise convergence ({i+1} of {n})',f'Figure 5{chr(97+i)}')
    y=para(c,y,'Per-graph mean over 16 paired starts. Dashed line: draw 100. Stopped curves remain constant; axes vary by graph.',small)
    finish(b,c,'all_71_curves.pdf',i,(34,128,561,H-43))
assert len(figures)==10 and len(result)==12
result.set_metadata({'title':'Experiment035 | Numerical appendix','subject':'Adaptive hyperplane sweep: patience sensitivity and convergence','keywords':'Max-Cut, G-set, EVA, Goemans-Williamson, Hyperplane Sweep'})
result.save(out,garbage=4,deflate=True); result.close()
