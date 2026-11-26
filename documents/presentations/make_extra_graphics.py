"""Reproducible vector diagrams for the Max-Cut research presentation.

Schematics are conceptual, not benchmark measurements unless explicitly identified.
No stochastic benchmarking or EVA optimizations are performed.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
from matplotlib.lines import Line2D

ROOT=Path(__file__).parent/'assets'
ROOT.mkdir(parents=True,exist_ok=True)
NAVY='#102a52'; TEAL='#0c7999'; CYAN='#3ca7c1'; GOLD='#db9c31'; RED='#ce6173'; MUTE='#526b83'; BG='#f8fafc'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':BG,'axes.facecolor':BG,'savefig.facecolor':BG})

def save(name,fig):
    fig.savefig(ROOT/name,bbox_inches='tight',pad_inches=.15,format='svg')
    plt.close(fig)
def setup(w=6,h=4.2):
    fig,ax=plt.subplots(figsize=(w,h));ax.set_facecolor(BG);return fig,ax

def graph(ax,xy,edges,sides=None,neg=[]):
    for k,(i,j) in enumerate(edges):
        negedge=k in neg
        ax.plot([xy[i][0],xy[j][0]],[xy[i][1],xy[j][1]],lw=2.6,ls='--' if negedge else '-',color=RED if negedge else '#9cb5c5',zorder=1)
    for i,p in enumerate(xy):
        c=TEAL if sides is None or sides[i]==0 else GOLD
        ax.add_patch(Circle(p,.09,facecolor=c,edgecolor='white',linewidth=1.7,zorder=3))
        ax.text(*p,str(i+1),ha='center',va='center',fontsize=9,color='white',weight='bold',zorder=4)
    ax.set_aspect('equal');ax.axis('off')

# 1 maxcut graph with signed ties
fig,ax=setup();xy=[(-1,0),(-.8,1),(-.25,.6),(.35,.9),(.9,.2),(.45,-.7),(-.5,-.7)];edges=[(0,1),(0,2),(1,2),(2,3),(2,5),(3,4),(4,5),(5,6),(6,0),(1,3),(2,4)];graph(ax,xy,edges,[0,0,1,1,0,0,1],[0,6]);ax.text(-1.25,1.32,'SIGNED MAX-CUT',fontsize=16,weight='bold',color=NAVY);ax.text(-1.25,1.12,'Dashed red = negative weight',fontsize=10,color=MUTE);ax.set_xlim(-1.35,1.25);ax.set_ylim(-1.1,1.6);save('signed_graph.svg',fig)

# 2 abstract SDP factor and hyperplane
fig,ax=setup();t=np.linspace(0,2*np.pi,250);ax.plot(np.cos(t),np.sin(t),color='#cedee7',lw=2);ax.plot([-1.4,1.4],[-.45,.45],color=RED,lw=2,ls='--');angs=np.array([.17,.42,.8,1.2,1.8,2.1,2.7,3.35,3.75,4.1,4.65,5.5,5.8]);pts=np.array([np.cos(angs),np.sin(angs)]).T;side=pts[:,1]-.321*pts[:,0]>0;ax.scatter(pts[side,0],pts[side,1],s=95,c=TEAL,edgecolor='white',lw=1.5);ax.scatter(pts[~side,0],pts[~side,1],s=95,c=GOLD,edgecolor='white',lw=1.5);ax.annotate('r',xy=(-.16,.6),xytext=(-.3,.9),color=RED,arrowprops={'arrowstyle':'->','color':RED},fontsize=15);ax.text(-1.28,1.3,'A RANDOM HYPERPLANE',weight='bold',color=NAVY,fontsize=14);ax.set_xlim(-1.4,1.4);ax.set_ylim(-1.38,1.58);ax.set_aspect('equal');ax.axis('off');save('hyperplane.svg',fig)

# 3 spherical kmeans two centroid update
fig,ax=setup();ax.plot(np.cos(t),np.sin(t),color='#cedee7',lw=2);a=np.array([.1,.35,.52,.9,1.05,1.4,3.2,3.5,3.8,4.18,4.5,5.1]);p=np.array([np.cos(a),np.sin(a)]).T;ax.scatter(p[:6,0],p[:6,1],s=58,c=TEAL);ax.scatter(p[6:,0],p[6:,1],s=58,c=GOLD);forcen=[np.arctan2(np.sin(a[:6]).sum(),np.cos(a[:6]).sum()),np.arctan2(np.sin(a[6:]).sum(),np.cos(a[6:]).sum())];
for ang,c in zip(forcen,[TEAL,GOLD]):ax.annotate('',xy=(1.17*np.cos(ang),1.17*np.sin(ang)),xytext=(0,0),arrowprops={'arrowstyle':'->','lw':3,'color':c})
ax.set_xlim(-1.4,1.4);ax.set_ylim(-1.4,1.4);ax.set_aspect('equal');ax.axis('off');ax.text(-1.3,1.28,'SPHERICAL CENTROID UPDATE',fontsize=13,weight='bold',color=NAVY);save('spherical_centroids.svg',fig)

# 4 2d benefit of local search (schematic)
fig,ax=setup();x=np.linspace(0,100,500);f=.5+2*np.sin(x/14)+.9*np.sin(x/4);ax.plot(x,f,color='#9bb3c5',lw=3);candidates=[10,27,49,68,90];ax.scatter(candidates,np.interp(candidates,x,f),s=60,color=[GOLD,GOLD,TEAL,GOLD,TEAL],zorder=4);ax.annotate('Local basin',xy=(27,np.interp(27,x,f)),xytext=(4,4.8),color=TEAL,arrowprops={'arrowstyle':'->','color':TEAL});ax.annotate('ILS perturbation',xy=(90,np.interp(90,x,f)),xytext=(55,5.4),color=TEAL,arrowprops={'arrowstyle':'->','color':TEAL});ax.set_xlabel('Conceptual partition-search coordinate',color=MUTE);ax.set_ylabel('Conceptual cut quality',color=MUTE);ax.set_yticks([]);ax.set_xticks([]);ax.text(3,6.4,'SCHEMATIC: ESCAPING BASINS',color=NAVY,weight='bold',fontsize=13);save('local_search_basins.svg',fig)

# 5 chronological timeline
fig,ax=setup(6.6,4.4);stages=[('001–003','Factorization'),('005–009','MST/coarsen'),('010–013','KL + ILS'),('015–018','New rounding'),('019','G63 trial'),('020','68 graphs'),('021','71 graphs')];xs=np.linspace(.4,6.2,len(stages));ax.plot(xs,np.zeros_like(xs),lw=5,color='#c4dde7');
for i,(n,name) in enumerate(stages):
    c=TEAL if i>=4 else NAVY;ax.scatter([xs[i]],[0],s=280,color=c,zorder=4,edgecolor='white',lw=2);v=.42 if i%2==0 else -.42;ax.text(xs[i],v,n,ha='center',va='center',fontsize=12,weight='bold',color=NAVY);ax.text(xs[i],v+(.22 if i%2==0 else -.22),name,ha='center',va='center',fontsize=9,color=MUTE)
ax.set_xlim(0,6.6);ax.set_ylim(-1,1);ax.axis('off');ax.text(.2,.9,'EXPERIMENTAL EVOLUTION',fontsize=15,color=NAVY,weight='bold');save('timeline.svg',fig)

# 6 illustrated gain signed graph
fig,ax=setup();xy=[(0,0),(-.8,.7),(.8,.7),(-.7,-.7),(.9,-.75)];edges=[(0,1),(0,2),(0,3),(0,4),(1,2),(3,4)];graph(ax,xy,edges,[0,0,1,0,1],[1]);ax.add_patch(Circle(xy[0],.20,fill=False,lw=2,ls='--',ec=RED));ax.annotate('flip this vertex',xy=(0,0),xytext=(.4,1.22),color=RED,fontsize=11,arrowprops={'arrowstyle':'->','color':RED});ax.text(-1.4,1.45,'LOCAL FLIP: EXACT GAIN',color=NAVY,fontsize=14,weight='bold');ax.set_xlim(-1.5,1.45);ax.set_ylim(-1.1,1.65);save('flip_gain.svg',fig)

# 7 multi cluster to binary mapping
fig,ax=setup();rng=np.random.default_rng(3);means=[(-1,.6),(-.4,-.5),(.6,.7),(1,-.4)];cols=[TEAL,'#68c2ad',GOLD,'#ddb4a1'];
for k,mu in enumerate(means):
    z=rng.normal(mu,.12,(18,2));ax.scatter(z[:,0],z[:,1],s=16,color=cols[k],alpha=.9)
    ax.add_patch(Circle(mu,.43,fill=False,edgecolor=cols[k],lw=2))
ax.text(-1.5,1.4,'FOUR GEOMETRIC GROUPS',weight='bold',fontsize=12,color=NAVY);ax.annotate('',xy=(1.1,-1.12),xytext=(-1.1,-1.12),arrowprops={'arrowstyle':'->','lw':2,'color':TEAL});ax.text(0,-1.35,'merge exactly into two cut sides',ha='center',fontsize=10,color=TEAL);ax.set_xlim(-1.5,1.5);ax.set_ylim(-1.55,1.6);ax.axis('off');save('multigroup.svg',fig)

# 8 initialization geometry icons
fig,ax=setup();ax.set_xlim(-2,2);ax.set_ylim(-1.6,1.5);ax.axis('off');v=[(-1.1,.7),(1.1,.7),(-1.1,-.75),(1.1,-.75)];names=['Antipodal','Independent','Farthest pair','GW screen-16'];
for i,((cx,cy),name) in enumerate(zip(v,names)):
    t=np.linspace(0,2*np.pi,80);ax.plot(cx+.45*np.cos(t),cy+.45*np.sin(t),lw=1.6,color='#b7d3e0');a= [(.35,.35+np.pi),(.4,2.0),(.8,3.7),(.1,.1+np.pi)][i];
    for j,theta in enumerate(a):ax.scatter([cx+.45*np.cos(theta)],[cy+.45*np.sin(theta)],s=85,color=TEAL if j==0 else GOLD,zorder=4)
    ax.text(cx,cy-.63,name,ha='center',fontsize=10,color=NAVY)
ax.text(-1.9,1.35,'FOUR STARTING-CENTER PATTERNS',color=NAVY,weight='bold',fontsize=12);save('initializers.svg',fig)

# 9 matched seeds table drawing
fig,ax=setup();ax.axis('off');ax.set_xlim(0,5);ax.set_ylim(0,4);
for j in range(4):
    for i in range(3):
        ax.add_patch(FancyBboxPatch((.4+i*1.47,2.95-j*.65),1.18,.52,boxstyle='round,pad=.06',facecolor='#cfe9f1' if j else '#d7dde9',edgecolor='white'))
        ax.text(1+i*1.47,3.2-j*.65,['r=0','r=1','r=49'][i] if j==0 else ['Flip','KL','ILS'][j-1],ha='center',va='center',color=NAVY,fontsize=12)
ax.text(0.2,.25,'Shared postprocessor seeds across initializer methods',fontsize=10,color=MUTE);ax.text(.2,3.7,'PAIRED RANDOMNESS DESIGN',fontsize=14,weight='bold',color=NAVY);save('seeds.svg',fig)

# 10 workflow runner
fig,ax=setup();ax.axis('off');ax.set_xlim(0,6);ax.set_ylim(0,4.1);items=[('GitHub Actions',3.6),('Windows / Conda',2.8),('32 search workers',2.0),('Hash checks + CSV',1.2),('Commit + archives',.4)];
for i,(label,y) in enumerate(items):
 ax.add_patch(FancyBboxPatch((1.2,y),3.7,.48,boxstyle='round,pad=.08',facecolor=['#113c5d','#125b7c','#167f9c','#198ba4','#0c6f86'][i],edgecolor='none'));ax.text(3.05,y+.25,label,ha='center',va='center',color='white',fontsize=13,weight='bold')
 if i<4:ax.annotate('',xy=(3.05,y-.24),xytext=(3.05,y-.02),arrowprops={'arrowstyle':'->','lw':2,'color':TEAL})
ax.set_title('REPRODUCIBLE RUNNER',loc='left',color=NAVY,fontweight='bold',fontsize=14);save('runner.svg',fig)

# 11 coverage chart
fig,ax=setup(6.3,3.3);ax.barh([0],[68],height=.54,color=TEAL,label='Reused results');ax.barh([0],[3],left=[68],height=.54,color=GOLD,label='New EVA + searches');ax.text(34,0,'68 reused',ha='center',va='center',color='white',fontsize=14,weight='bold');ax.text(69.4,-.43,'3 new',ha='center',va='center',color=GOLD,fontsize=13,weight='bold');ax.set_xlim(0,72);ax.set_ylim(-.6,.6);ax.axis('off');ax.set_title('68 + 3 = ALL 71 CANONICAL GRAPHS',loc='left',color=NAVY,fontweight='bold',fontsize=14);save('coverage.svg',fig)

# 12 new graphs cut comparison slopes
fig,ax=setup(6.2,3.8);g=['G60','G61','G62'];control=[14103,5718,4808];new=[14124,5721,4816];gain=[21,3,8];ax.barh(np.arange(3),gain,color=[TEAL,'#329fbc',GOLD],height=.48);ax.set_yticks(range(3),g);ax.invert_yaxis();ax.set_xlim(0,25);ax.set_xlabel('Gain over the graph’s best-of-50 control cut');ax.set_title('NEWLY COMPLETED: BEST-OF-NINE GAIN',loc='left',color=NAVY,fontweight='bold',fontsize=13);[ax.text(v+.4,i,f'+{v}',va='center',color=NAVY,fontweight='bold') for i,v in enumerate(gain)];save('new_graph_gains.svg',fig)

# 13 portfolio results
fig,ax=setup(6,3.6);ax.pie([58,13],labels=['Improved: 58','Tied: 13'],colors=[TEAL,'#bdccd7'],startangle=90,wedgeprops={'width':.38,'edgecolor':BG},textprops={'color':NAVY,'fontsize':12});ax.text(0,0,'71',ha='center',va='center',fontsize=36,weight='bold',color=NAVY);ax.set_title('BEST-OF-NINE VS BEST-OF-CONTROL',color=NAVY,fontweight='bold',fontsize=13);save('portfolio_pie.svg',fig)

# 14 model caution side-by-side
fig,ax=setup(6,3.8);ax.set_xlim(0,6);ax.set_ylim(0,4);ax.axis('off');
for i,(title,subtitle) in enumerate([('Primal SDP','Convex full-rank relaxation'),('EVA factor','Nonconvex rank restriction')]):
    x=.3+i*2.9;ax.add_patch(FancyBboxPatch((x,1),2.6,1.9,boxstyle='round,pad=.08',facecolor=['#dcecf4','#e5f4ee'][i],edgecolor=TEAL,lw=1.5));ax.text(x+1.3,2.25,title,ha='center',fontsize=15,fontweight='bold',color=NAVY);ax.text(x+1.3,1.65,subtitle,ha='center',fontsize=10,color=MUTE)
ax.text(.25,3.4,'NOT INTERCHANGEABLE',color=NAVY,fontweight='bold',fontsize=14);ax.text(3,.35,'EVA convergence ≠ certified SDP bound',color=RED,ha='center',fontsize=12,weight='bold');save('eva_caveat.svg',fig)

# 15 equal budget controls conceptual
fig,ax=setup(6,3.7);b=np.linspace(1,100,100);f=lambda a,c:a+c*(1-np.exp(-b/25));ax.plot(b,f(.6,1.8),lw=3,color=TEAL,label='Control-only');ax.plot(b,f(.5,2.05),lw=3,color=GOLD,label='Diversified portfolio');ax.set_xlabel('Matched total evaluation budget');ax.set_ylabel('Illustrative best cut quality');ax.set_xticks([]);ax.set_yticks([]);ax.legend(frameon=False,fontsize=11);ax.set_title('PROPOSED EXPERIMENT022 (SCHEMATIC)',loc='left',color=NAVY,fontweight='bold',fontsize=13);save('equal_budget.svg',fig)

# 16 academic relationship
fig,ax=setup(6,3.7);ax.axis('off');ax.set_xlim(0,6);ax.set_ylim(0,4);ll=[('SDP geometry',.55,2.4),('Clustering',3.8,2.4),('Max-Cut score',2.2,.55)];
for label,x,y in ll:
 ax.add_patch(FancyBboxPatch((x,y),1.75,.85,boxstyle='round,pad=.08',facecolor='#d9edf5',edgecolor=TEAL,lw=2));ax.text(x+.875,y+.42,label,ha='center',va='center',color=NAVY,fontweight='bold',fontsize=11)
for u,v in [((2.3,2.7),(3.7,2.7)),((4.3,2.3),(3.5,1.47)),((1.7,2.3),(2.4,1.47))]:ax.annotate('',xy=v,xytext=u,arrowprops={'arrowstyle':'->','lw':2,'color':TEAL})
ax.text(.3,3.55,'THE SCIENTIFIC HYPOTHESIS',color=NAVY,fontweight='bold',fontsize=14);save('research_triangle.svg',fig)

print('Created',len(list(ROOT.glob('*.svg'))),'SVG figures')