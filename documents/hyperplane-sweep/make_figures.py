"""Generate the four vector figures for the educational Hyperplane Sweep note."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family":"serif","font.size":11,"mathtext.fontset":"stix",
                     "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
BLUE, ORANGE, GRAY = "#215b8c", "#c66b32", "#77818b"
angles=np.deg2rad([0,60,120])
Y=np.c_[np.cos(angles),np.sin(angles)]
edges=[(0,1,3),(0,2,2),(1,2,1)]
def signs(theta):
    return np.where(Y@np.array([np.cos(theta),np.sin(theta)])>0,1,-1)
def score(x):
    return sum(w for i,j,w in edges if x[i]!=x[j])
def save(fig,name):
    fig.savefig(OUT/name,bbox_inches="tight",metadata={"CreationDate":None,"ModDate":None})
    plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(9,3.3),layout="constrained")
for ax,deg in zip(axs,[-60,0,60]):
    theta=np.deg2rad(deg)
    normal=np.array([np.cos(theta),np.sin(theta)])
    tangent=np.array([-normal[1],normal[0]])
    x=signs(theta)
    t=np.linspace(0,2*np.pi,400)
    ax.plot(np.cos(t),np.sin(t),color="#a7adb4",lw=1)
    ax.plot(*np.outer([-1.15,1.15],tangent).T,"--",color=GRAY,lw=1.5)
    ax.quiver(0,0,*(0.77*normal),angles="xy",scale_units="xy",scale=1,color=BLUE,width=.012)
    for i,v in enumerate(Y):
        ax.scatter(*v,color=BLUE if x[i]>0 else ORANGE,s=45,zorder=4)
        ax.text(*(1.16*v),f"$y_{i+1}$",ha="center",va="center",fontsize=12)
    ax.text(*(0.50*normal+np.array([.04,-.10])),r"$r(\theta)$",color=BLUE,fontsize=10)
    ax.set(xlim=(-1.3,1.3),ylim=(-1.3,1.3),aspect="equal",
           title=rf"$\theta={deg}^\circ$ | cut = {score(x)}")
    ax.axis("off")
fig.legend(handles=[Line2D([0],[0],marker="o",color=BLUE,ls="",label="Group +1"),
                    Line2D([0],[0],marker="o",color=ORANGE,ls="",label="Group -1"),
                    Line2D([0],[0],ls="--",color=GRAY,label="Separating line")],
           loc="outside lower center",ncols=3,frameon=False)
save(fig,"r2.pdf")

fig,ax=plt.subplots(figsize=(7,2.65),layout="constrained")
bounds=[-90,-30,30,90]
for lo,hi in zip(bounds[:-1],bounds[1:]):
    v=score(signs(np.deg2rad((lo+hi)/2)))
    ax.hlines(v,lo,hi,lw=3,color=BLUE)
    ax.plot([lo,hi],[v,v],"o",mfc="white",mec=BLUE,ms=5)
for e in bounds[1:-1]:
    ax.axvline(e,ls="--",color=GRAY,lw=1)
ax.scatter([-60],[5],color=ORANGE,s=45,zorder=5)
ax.annotate("winning interval",(-60,5),xytext=(-55,5.8),arrowprops={"arrowstyle":"-","color":GRAY})
ax.set(xlim=(-90,90),ylim=(-.5,6.4),xticks=np.arange(-90,91,30),yticks=np.arange(7),
       xlabel=r"Rotation angle $\theta$ (degrees)",ylabel="Cut weight")
ax.grid(axis="y",alpha=.2)
save(fig,"events.pdf")
assert [score(signs(np.deg2rad(t))) for t in [-60,0,60]]==[5,3,0]

fig=plt.figure(figsize=(8.2,3.6),layout="constrained")
phi=np.linspace(0,np.pi,30); psi=np.linspace(0,2*np.pi,50)
P,Q=np.meshgrid(phi,psi)
X=np.sin(P)*np.cos(Q); YY=np.sin(P)*np.sin(Q); Z=np.cos(P)
r=np.array([0.,0.,1.])
us=[np.array([1.,0.,0.]),np.array([0.,1.,0.])]
t=np.linspace(-np.pi/2,np.pi/2,200)
for idx in range(2):
    ax=fig.add_subplot(1,2,idx+1,projection="3d")
    ax.plot_surface(X,YY,Z,color="#dce4eb",alpha=.22,linewidth=0,shade=False)
    ax.plot(np.cos(psi),np.sin(psi),0*psi,color=GRAY,ls="--",lw=.8)
    for j,u in enumerate(us[:idx+1]):
        path=np.outer(np.cos(t),r)+np.outer(np.sin(t),u)
        c=[BLUE,ORANGE][j]
        ax.plot(*path.T,color=c,lw=2.5)
        ax.quiver(0,0,0,*u,color=c,arrow_length_ratio=.1,lw=1.5)
        ax.text(*(1.14*u),rf"$u_{j+1}$",color=c)
    ax.quiver(0,0,0,*r,color="#263645",arrow_length_ratio=.1,lw=1.5)
    ax.text(0,0,1.12,"$r$")
    ax.set(xlim=(-1.15,1.15),ylim=(-1.15,1.15),zlim=(-1.15,1.15),
           title=["One rotation plane","Two possible rotation planes"][idx])
    ax.set_box_aspect((1,1,1)); ax.view_init(elev=24,azim=38); ax.set_axis_off()
save(fig,"r3.pdf")

Y3=np.array([[.85,.2,.48],[-.75,.35,.56],[.2,-.93,.31],[-.2,-.3,-.93],
             [.55,.75,-.37],[-.65,-.6,-.45],[.15,.9,.4],[.9,-.4,-.15]])
Y3/=np.linalg.norm(Y3,axis=1)[:,None]
E3=[(0,1,3),(0,2,2),(0,4,1),(1,3,4),(1,6,2),(2,3,3),(2,7,2),
    (3,4,2),(4,5,3),(5,6,4),(6,7,3),(0,7,-2),(1,5,-1)]
z=np.array([.9,.6,.8]); h=z-(z@r)*r; u=h/np.linalg.norm(h)
def normal(t):
    return np.cos(t)*r+np.sin(t)*u
def cut(t):
    return sum(w for i,j,w in E3 if (Y3[i]@normal(t)>0)!=(Y3[j]@normal(t)>0))
a=Y3@r; b=Y3@u
bounds=np.r_[-np.pi/2,np.sort(np.arctan(-a/b)),np.pi/2]
mid=(bounds[:-1]+bounds[1:])/2
values=np.array([cut(t) for t in mid]); best=values.argmax(); t=mid[best]; q=normal(t)
assert cut(0)==14 and cut(t)==23 and np.isclose(r@u,0)
print("Verified example: initial",cut(0),"best",cut(t),"interval",np.rad2deg(bounds[best:best+2]),"midpoint",np.rad2deg(t))
fig=plt.figure(figsize=(10.4,4.2),layout="constrained")
ax=fig.add_subplot(121,projection="3d"); bx=fig.add_subplot(122)
ang=np.linspace(0,2*np.pi,100); pol=np.linspace(0,np.pi,20)
for p in pol[1:-1:2]:
    ax.plot(np.sin(p)*np.cos(ang),np.sin(p)*np.sin(ang),np.full_like(ang,np.cos(p)),alpha=.13,color="0.4",lw=.65)
for a0 in np.linspace(0,2*np.pi,10,endpoint=False):
    ax.plot(np.sin(pol)*np.cos(a0),np.sin(pol)*np.sin(a0),np.cos(pol),alpha=.15,color="0.4",lw=.65)
def plane(v,w,color):
    pts=np.cos(ang)[:,None]*v+np.sin(ang)[:,None]*w
    ax.add_collection3d(Poly3DCollection([pts],facecolor=color,alpha=.16,edgecolor=color,linewidth=.8))
plane(r,u,"#8e5baf")
a0=np.cross(q,[0,0,1]); a0/=np.linalg.norm(a0)
plane(a0,np.cross(q,a0),"#12877d")
arc=np.array([normal(t0) for t0 in np.linspace(-np.pi/2,np.pi/2,100)])
ax.plot(*arc.T,color="#8e5baf",lw=2)
for v,label,c in [(r,"$r$","#173e65"),(z,"$z$","#ac6c16"),(u,"$u$","#8e5baf"),(q,r"$r(\theta^*)$","#12877d")]:
    ax.quiver(0,0,0,*v,color=c,arrow_length_ratio=.1,lw=2)
    ax.text(*(1.13*v),label,color=c)
ax.plot(*np.array([z,(z@r)*r]).T,ls="--",color="#ac6c16",lw=1.2)
for i,v in enumerate(Y3):
    ax.scatter(*v,color="#2978b8" if v@q>0 else "#d77736",s=20,depthshade=False)
    ax.text(*(1.07*v),f"$y_{i+1}$",fontsize=7)
ax.set(xlim=(-1.2,1.2),ylim=(-1.2,1.2),zlim=(-1.2,1.2))
ax.set_box_aspect((1,1,1)); ax.view_init(elev=20,azim=90); ax.set_axis_off()
ax.set_title("Rotation plane and separating plane",fontsize=11)
for i,v in enumerate(values):
    c="#12877d" if i==best else "#2978b8"
    bx.hlines(v,*np.rad2deg(bounds[i:i+2]),lw=2.6,color=c)
    bx.plot(np.rad2deg(mid[i]),v,"o",ms=3,color=c)
for e in bounds[1:-1]:
    bx.axvline(np.rad2deg(e),color=".8",ls=":",lw=.8)
bx.axvline(0,color=".4",ls="--",lw=.8)
bx.set(xlim=(-90,90),xlabel="Angle (degrees)",ylabel="Signed cut",title="Nine event-free intervals")
bx.set_xticks(np.arange(-90,91,30)); bx.grid(axis="y",alpha=.2)
save(fig,"worked_r3.pdf")
