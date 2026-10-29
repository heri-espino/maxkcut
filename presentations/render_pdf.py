#!/usr/bin/env python3
"""Build a fully offline, 16:9 visual PDF from our Marp Markdown source.

Uses markdown-it-py for text, MathJax SVG (locally installed) for LaTeX,
and headless Chromium PDF printing. No benchmark reruns; deterministic art.
"""
from pathlib import Path
import re, json, subprocess, html, argparse, tempfile, shutil
from markdown_it import MarkdownIt

HERE=Path(__file__).parent
MD=HERE/'maxkcut_research_journey.marp.md'
OUTPUT=HERE/'MaxCut_Research_71_Gset_For_Professor_Angel.pdf'
NODE_ROOT='/opt/nvm/versions/node/v22.16.0/lib/node_modules/mathjax-full/js'
CHROMIUM=shutil.which('chromium')

VISUALS={
 'Formal objective: signed Max-Cut':'signed_graph.svg',
 'Classical SDP relaxation':'eva_caveat.svg',
 'Goemans–Williamson: random hyperplane rounding':'hyperplane.svg',
 'How spherical K-means enters Max-Cut rounding':'spherical_centroids.svg',
 'Three foundations from the literature':'research_triangle.svg',
 'Step 1 — Build a verifiable experimental substrate':'research_triangle.svg',
 'Step 2 — Obtain geometry with EVA':'eva_caveat.svg',
 'Step 3 — Test alternatives to naive rounding':'multigroup.svg',
 'Step 4 — Improve a cut with exact vertex gains':'flip_gain.svg',
 'Step 5 — Escape the one-flip local optimum':'local_search_basins.svg',
 'Step 6 — Select the full refinement pipeline':'runner.svg',
 'Reproducible RNG design: matched downstream seeds':'seeds.svg',
 'Experiment019 — Controlled pilot on G63':'initializers.svg',
 'Experiment020 — Expand to the 68 graphs with saved factors':'coverage.svg',
 'Why the self-hosted runner mattered':'runner.svg',
 'Experiment021 — Generate exactly the missing EVA factors':'new_graph_gains.svg',
 'Final completeness proof is a machine-checked ledger':'coverage.svg',
 'Important graph-specific landmarks':'initializers.svg',
 'Why 58/71 is not an equal-budget victory':'portfolio_pie.svg',
 'What the statistics do—and do not—show':'seeds.svg',
 'Experimental limitations':'eva_caveat.svg',
 'Experiment022 — The next testable question':'equal_budget.svg',
}

SRC=MD.read_text(encoding='utf-8')
if SRC.startswith('---\n'):
 # all Marp slide breaks occur on single stand-alone --- lines
 SRC=SRC.split('\n---\n',1)[1]
slides=SRC.split('\n---\n')
# direct image cases are handled by markdown-it

# Protect inline/display maths from markdown's backslash escaping.
math_text=[]
def isolate_math(txt):
    def sub(m):
        kind='display' if m.group(1)=='\\[' else 'inline'
        raw=m.group(2)
        tok=f'MATHTOKEN_{len(math_text):04d}_END'
        math_text.append((raw,kind))
        return ('\n\n' if kind=='display' else '')+tok+('\n\n' if kind=='display' else '')
    return re.sub(r'(\\\[|\\\()([\s\S]*?)(?:\\\]|\\\))',sub,txt)

slides=[isolate_math(s) for s in slides]
script=HERE/'_mathjax_batch.cjs'
script.write_text(r'''
const fs=require('fs');
const root='/opt/nvm/versions/node/v22.16.0/lib/node_modules/mathjax-full/js';
const {mathjax}=require(root+'/mathjax.js');
const {TeX}=require(root+'/input/tex.js');
const {SVG}=require(root+'/output/svg.js');
const {AllPackages}=require(root+'/input/tex/AllPackages.js');
const {liteAdaptor}=require(root+'/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler}=require(root+'/handlers/html.js');
const adaptor=liteAdaptor(); RegisterHTMLHandler(adaptor);
const html=mathjax.document('',{InputJax:new TeX({packages:AllPackages}),OutputJax:new SVG({fontCache:'none'})});
const input=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const out=input.map(([src,kind])=>{
 try{return adaptor.outerHTML(html.convert(src,{display:kind==='display'}))}
 catch(e){return '<span class="math-error">'+src.replace(/</g,'&lt;')+'</span>'}
});
fs.writeFileSync(process.argv[3],JSON.stringify(out));
''',encoding='utf-8')
(HERE/'_math.json').write_text(json.dumps(math_text),encoding='utf-8')
subprocess.run(['node',str(script),str(HERE/'_math.json'),str(HERE/'_math_render.json')],check=True,timeout=70)
math_render=json.loads((HERE/'_math_render.json').read_text(encoding='utf-8'))

md=MarkdownIt('commonmark',{'html':True,'linkify':True}).enable('table').enable('strikethrough')
page_html=[]
for ix,src in enumerate(slides,1):
    cl=re.search(r'<!--\s*_class:\s*(\S+)\s*-->',src)
    slide_class=cl.group(1) if cl else ''
    src=re.sub(r'<!--\s*_class:[\s\S]*?-->','',src)
    # Preserve code and original full-width images, but side images are injected by title matching.
    head=re.search(r'^#{1,3}\s+(.+)$',src,re.M)
    heading=head.group(1).strip() if head else ''
    new_figure=None
    if heading in VISUALS:
        new_figure=VISUALS[heading]
        # if source also includes Marp right-background image then remove it here (we render aside)
        src=re.sub(r'^!\[bg\s+right:[^\]]+\]\([^)]*\)\s*$', '',src,flags=re.M)
    # Markdown images absolute local file paths for chromium.
    inner=md.render(src.strip())
    for j,s in enumerate(math_render):
        token=f'MATHTOKEN_{j:04d}_END'
        if token in inner:
            inner=inner.replace('<p>'+token+'</p>',s) if 'display="true"' in s else inner
            inner=inner.replace(token,s)
    # Remove unwanted link query tracking, but preserve references.
    aside=f'<aside class="graphic"><img alt="Scientific figure" src="assets/{new_figure}"/><span class="diagram-tag">{("Conceptual schematic" if new_figure not in ["coverage.svg","new_graph_gains.svg","portfolio_pie.svg"] else "Verified experiment values")}</span></aside>' if new_figure else ''
    if slide_class=='cover':
        inner=inner.replace('<h1>', '<h1>',1)+"<p class='professor'>Prepared for Professor Ángel E. Rodríguez Fernández</p>"
    content='<article class="content">'+inner+'</article>'
    page_html.append(f'<section class="slide {slide_class} {"has-graphic" if aside else ""}" id="page-{ix}"><div class="topline"><span>MAX-CUT RESEARCH</span><span>G-SET · 29 OCT 2026</span></div><div class="slide-content">{content}{aside}</div><div class="foot"><span>heri-espino/maxkcut · Experiments 001–021</span><span>{ix:02d} / {len(slides):02d}</span></div></section>')

styles=r'''
@page {size: 13.3333in 7.5in; margin: 0;}
* {box-sizing:border-box;}
body {margin:0;padding:0;background:#e5eaf0;color:#102a52;font-family:Arial,'DejaVu Sans',sans-serif;-webkit-print-color-adjust:exact;print-color-adjust:exact;}
.slide { position:relative; width:13.3333in; height:7.5in; overflow:hidden;page-break-after:always;break-after:page; background:linear-gradient(180deg,#f8fafc 0%, #f6f9fb 100%);padding:40px 66px 46px;}
.topline {display:flex;justify-content:space-between;text-transform:uppercase;letter-spacing:1.9px;color:#1686a4;font-size:12px;font-weight:800;border-bottom:2px solid #d7e8ef;padding-bottom:12px;height:29px;}
.slide-content{display:flex;position:absolute;top:89px;left:66px;right:66px;bottom:66px;align-items:flex-start;gap:35px;}
.content{flex:1;min-width:0;font-size:25px;line-height:1.27;overflow:visible;}
h1,h2,h3,p{margin-top:0;}
h1{font-size:53px;line-height:1.05;color:#0d2b4d;letter-spacing:-1.2px;margin-bottom:29px;}
h2{font-size:40px;line-height:1.1;color:#0d2b4d;letter-spacing:-.5px;margin-bottom:25px;}
h3{font-size:28px;line-height:1.13;color:#126c8c;margin-bottom:18px;}
p{margin:13px 0 16px;}strong{color:#086e8b;}ol,ul{padding-left:1.18em;margin:15px 0;}li{padding-left:2px;margin:10px 0;}a{color:#126b8a;text-decoration:none;}
blockquote{border-left:4px solid #0d87a2;margin:18px 0;padding:8px 18px;background:#e8f2f7;}
code{font-family:'DejaVu Sans Mono',monospace;background:#e6f1f7;padding:2px 5px;border-radius:3px;font-size:.87em;color:#1e6380;}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#112d46;border-radius:12px;padding:18px 22px;color:#e9f4fc;font-size:19px;line-height:1.24;margin:16px 0;}
pre code{background:transparent;color:inherit;padding:0;font-size:inherit;}
table{border-collapse:collapse;width:100%;font-size:20px;line-height:1.18;}
th,td{padding:9px 11px;border-bottom:1px solid #cddce5;vertical-align:top;text-align:left;}th{background:#e2f0f7;color:#173b58;font-weight:bold;}tr:nth-child(even){background:#eff5f8;}
.content img{display:block;max-width:100%;max-height:455px;margin:18px auto;object-fit:contain;}
.graphic{width:43%;flex:none;display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;background:linear-gradient(180deg,#eff8fa,#f9fcfd);border:1px solid #dcecf0;border-radius:18px;padding:18px;}
.graphic img{width:100%;max-width:100%;max-height:420px;object-fit:contain;}
.diagram-tag{margin-top:18px;font-size:12px;text-transform:uppercase;letter-spacing:1.3px;color:#628294;font-weight:700;}
.has-graphic .content{font-size:23px;line-height:1.25;}
.has-graphic h2{font-size:36px;}
.has-graphic pre{font-size:15px;}
.has-graphic li{margin:8px 0;}
.content mjx-container{font-size:1em!important;color:#193f59;max-width:100%;}
.content mjx-container[display='true']{margin:14px 0 22px;}
.content mjx-container[display='true'] svg{max-width:100%;}
.foot{position:absolute;bottom:19px;left:66px;right:66px;display:flex;justify-content:space-between;color:#8299a8;font-size:13px;border-top:1px solid #e2eaf0;padding-top:9px;}
.slide.cover{background:linear-gradient(125deg,#0a1c34,#0e3c58 65%,#0e7792);color:white;}
.cover .topline{color:#cce4ed;border-color:#5a8da6;}.cover .content{color:#dceefa;font-size:29px;}.cover h1{color:white;font-size:66px;}.cover strong{color:#e0f3fa;}.cover .foot{color:#bcd8e5;border-color:#5489a0;}.cover .professor{font-size:25px;color:#b8ecf7;border-top:2px solid #5ab1c8;padding-top:22px;margin-top:25px;}
.slide.chapter{background:linear-gradient(130deg,#0b2946,#0b6587);color:white;}
.chapter .topline{color:#b5e1ec;border-color:#4885a1;}.chapter .slide-content{align-items:center;}.chapter .content{color:#d7ebf3;font-size:30px;}.chapter h1{color:white;font-size:67px;}.chapter .foot{color:#b5d4e1;border-color:#4c8198;}
.slide.compact .content{font-size:20.5px;}.slide.compact h2{font-size:37px;}.slide.compact table{font-size:17.5px;}.slide.compact td,.slide.compact th{padding:7px 9px;}
.slide.small .content{font-size:21px;}.slide.small table{font-size:17px;}
#page-31 .content{font-size:20px;}#page-31 .content p{margin:7px 0;}#page-37 .content{font-size:19px;}#page-37 .content p{margin:7px 0;}
.slide.center .content{text-align:center;}
.slide.center .slide-content{align-items:center;}
.slide.center p{font-size:30px;}
@media screen {.slide{margin:20px auto;box-shadow:0 12px 45px #8598a5;}}
@media print {.slide{margin:0!important;box-shadow:none!important;}body{background:white;}a{color:#126b8a;}}
'''
js=r'''
function fitSlides(){
 const slides=document.querySelectorAll('.slide');
 for(const slide of slides){
  const art=slide.querySelector('.content');
  const wrap=slide.querySelector('.slide-content');
  if(!art)continue;
  const h=wrap.clientHeight;
  // Scale entire content using font-size where necessary (tables/pre are covered below).
  let n=0;
  while(art.scrollHeight>h+1 && n<19){
   const ratio=parseFloat(getComputedStyle(art).fontSize);
   art.style.fontSize=(ratio*.957)+'px';
   art.querySelectorAll('h1,h2,h3').forEach(x=>x.style.fontSize=(parseFloat(getComputedStyle(x).fontSize)*.957)+'px');
   art.querySelectorAll('table').forEach(x=>x.style.fontSize=(parseFloat(getComputedStyle(x).fontSize)*.957)+'px');
   art.querySelectorAll('pre').forEach(x=>x.style.fontSize=(parseFloat(getComputedStyle(x).fontSize)*.957)+'px');
   art.querySelectorAll('li').forEach(x=>x.style.margin='6px 0');
   n++;
  }
  if(art.scrollHeight>h+2){slide.dataset.overflow=art.scrollHeight-h;}
 }
 window.__slideFitDone=true;
}
window.addEventListener('load',()=>{fitSlides()});
'''
html_doc=f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Max-Cut research report for Professor Ángel E. Rodríguez Fernández</title><style>{styles}</style></head><body>{"".join(page_html)}<script>{js}</script></body></html>'
html_path=HERE/'_rendered_slides.html';html_path.write_text(html_doc,encoding='utf-8')
print('Slides:',len(slides),'figures:',len(re.findall('class="graphic"',html_doc))+len(re.findall('src="assets/',html_doc)),'math items:',len(math_text),flush=True)
from weasyprint import HTML
HTML(filename=str(html_path),base_url=str(HERE)).write_pdf(str(OUTPUT))
import fitz
pdf=fitz.open(OUTPUT)
if len(pdf)!=len(slides):
    print('Warning: PDF pages',len(pdf),'versus slides',len(slides),'see output',flush=True)
if len(pdf)>0:
    pdf.set_metadata({'title':'From Random Hyperplanes to 71-Graph Evidence','author':'Max-Cut research presentation | heri-espino/maxkcut','subject':'For Professor Ángel E. Rodríguez Fernández | 71 G-set instances; Experiments 001–021','keywords':'Max-Cut, EVA, GW, spherical k-means, local search, G-set'})
    temp=OUTPUT.with_name(OUTPUT.stem+'__metadata.pdf');pdf.save(temp,garbage=3,deflate=True);pdf.close();temp.replace(OUTPUT)
print(f'PDF: {OUTPUT} ({OUTPUT.stat().st_size:,} bytes, {len(fitz.open(OUTPUT))} pages)',flush=True)