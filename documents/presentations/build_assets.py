from pathlib import Path
from html import escape

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

def svg_root(width, height, inner):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><style>text{{font-family:Inter,Arial,sans-serif}} .label{{fill:#193b58;font-size:20px;font-weight:600}} .small{{fill:#47647d;font-size:16px}} .head{{fill:#102a52;font-size:26px;font-weight:800}}</style><rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc"/>{inner}</svg>'

nodes=[('01','Input','Signed G-set'),('02','Factor','EVA unit vectors'),('03','Round','9 initializers'),('04','Refine','Flips → KL → ILS'),('05','Archive','All scores & cuts'),('06','Analyze','71 paired graphs')]
parts=[]
for i,(num,label,sub) in enumerate(nodes):
    x=10+i*185
    parts.extend([f'<rect x="{x}" y="55" width="160" height="116" rx="15" fill="{["#103354","#154c69","#166986","#187d91","#0c7187","#0c607a"][i]}"/>',
                  f'<text x="{x+14}" y="78" font-size="15" fill="#a6dff0">{num}</text>',
                  f'<text x="{x+80}" y="112" font-size="20" text-anchor="middle" font-weight="800" fill="white">{escape(label)}</text>',
                  f'<text x="{x+80}" y="143" font-size="14" text-anchor="middle" fill="#d1e7f5">{escape(sub)}</text>'])
    if i<5:
        parts.append(f'<path d="M {x+163} 113 L {x+181} 113 M {x+174} 107 L {x+181} 113 L {x+174} 119" stroke="#147da2" stroke-width="3" fill="none"/>')
parts.append('<text x="560" y="224" font-size="22" fill="#3a6280" text-anchor="middle">One fixed vector factor per graph · 50 matched restarts · immutable saved partitions</text>')
(ASSETS/'pipeline.svg').write_text(svg_root(1125,255,''.join(parts)))

wins=[('GW screen-16',24),('Sample antipodal',19),('Farthest pair',18),('Spherical K-means++',18),('Mean perturbation',17),('Independent centers',17),('PCA antipodal',16),('Sample pair',14),('Original K-MeansNM',13)]
parts=['<text x="16" y="38" class="head">Best-observed cut wins (ties included)</text>']
for i,(name,val) in enumerate(wins):
    y=69+i*47
    parts.append(f'<text x="16" y="{y+22}" class="label">{escape(name)}</text>')
    parts.append(f'<rect x="336" y="{y}" width="{val*27.5}" height="30" rx="6" fill="{("#0e7490" if i==0 else "#5aa3b9")}"/>')
    parts.append(f'<text x="{350+val*27.5}" y="{y+22}" class="label">{val}</text>')
parts.append('<text x="16" y="528" class="small">71 canonical graphs · nine methods × 50 restarts · ties count for each method</text>')
(ASSETS/'wins.svg').write_text(svg_root(1110,552,''.join(parts)))

means=[('GW screen-16',0.059),('Original K-MeansNM',0.0),('Mean perturbation',-0.379),('Sample pair',-0.536),('Spherical K-means++',-0.615),('Farthest pair',-0.641),('Independent centers',-0.649),('Sample antipodal',-0.686),('PCA antipodal',-0.756)]
parts=['<text x="16" y="38" class="head">71-graph mean final-cut difference vs original K-MeansNM</text>']
zero=1030
parts.append(f'<line x1="{zero}" y1="64" x2="{zero}" y2="501" stroke="#0f7492" stroke-width="2" stroke-dasharray="6 4"/>')
for i,(name,val) in enumerate(means):
    y=72+i*47
    parts.append(f'<text x="16" y="{y+21}" class="label">{escape(name)}</text>')
    if val<0:
        width=-val*650
        parts.append(f'<rect x="{zero-width:.2f}" y="{y}" width="{width:.2f}" height="29" rx="5" fill="#9caab8"/>')
        parts.append(f'<text x="{zero-width-65:.2f}" y="{y+21}" class="label">{val:+.3f}</text>')
    elif val>0:
        width=val*650
        parts.append(f'<rect x="{zero}" y="{y}" width="{width:.2f}" height="29" rx="5" fill="#0e7490"/>')
        parts.append(f'<text x="{zero-width-112:.2f}" y="{y+21}" class="label">{val:+.3f}</text>')
    else:
        parts.append(f'<text x="{zero-85}" y="{y+21}" class="label">0.000</text>')
parts.append('<text x="16" y="529" class="small">Unweighted average of each graph’s 50-restart paired mean differences · original signed cut units</text>')
(ASSETS/'mean_differences.svg').write_text(svg_root(1130,550,''.join(parts)))
print([str(p) for p in ASSETS.glob('*.svg')])
