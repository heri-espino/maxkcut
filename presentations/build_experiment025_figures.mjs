#!/usr/bin/env node
/** Deterministic Experiment025 Marp SVG figures. Read existing compact CSVs only;
 * no EVA/SDP solve, optimization, network access, or random graphics.
 * Run: node presentations/build_experiment025_figures.mjs [--check]
 */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";

function makeFigures(outcomeCSV,featureCSV,rankingCSV){
  const parseCSV=s=>{
    const lines=s.trim().split(/\r?\n/),keys=lines.shift().split(",");
    return lines.filter(Boolean).map(l=>{
      const values=l.split(",");
      if(values.length!==keys.length)throw Error("Malformed CSV row: "+l.slice(0,60));
      return Object.fromEntries(keys.map((k,i)=>[k,values[i]]));
    });
  };
  const raw=parseCSV(outcomeCSV), features=parseCSV(featureCSV), published=parseCSV(rankingCSV);
  const featureBy=new Map(features.map(r=>[r.graph,r]));
  const groups=new Map();
  for(const r of raw){if(!groups.has(r.graph))groups.set(r.graph,new Map());groups.get(r.graph).set(r.method,r);}
  const ids=[...groups.keys()].sort((a,b)=>Number(a.slice(1))-Number(b.slice(1)));
  const methods=[...groups.get(ids[0]).keys()];
  if(ids.length!==71 || methods.length!==18 || raw.length!==1278 || features.length!==71)throw Error("Required 71×18 census not complete");
  const familyOrder=["random","planar","toroidal"], familyName={random:"Random",planar:"Planar-family",toroidal:"Toroidal"};
  const F={random:"#0B7890",planar:"#347BC3",toroidal:"#D49B37"};
  const palette=["#0C6B91","#258E9D","#7869B1","#C6843D","#51906C","#A35F81","#4A79A3","#8E9560","#AC7A5C","#6B88A0","#9D9CA4"];
  const BLUE="#153858",INK="#17324D",MUTED="#667C91",BORDER="#D6E4EE",BG="#F7FAFD",WHITE="#FFFFFF";
  const esc=s=>String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&apos;");
  const t=(x,y,s,sz=16,co=INK,weight=500,anchor="start")=>`<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" text-anchor="${anchor}" fill="${co}" font-family="Arial, Helvetica, sans-serif" font-size="${sz}" font-weight="${weight}">${esc(s)}</text>`;
  const rect=(x,y,w,h,c,r=0,stroke="none")=>`<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(0,w).toFixed(1)}" height="${Math.max(0,h).toFixed(1)}" rx="${r}" fill="${c}" stroke="${stroke}"/>`;
  const line=(x,y,xx,yy,c=BORDER,w=1,dash="")=>`<line x1="${x.toFixed(1)}" y1="${y.toFixed(1)}" x2="${xx.toFixed(1)}" y2="${yy.toFixed(1)}" stroke="${c}" stroke-width="${w}"${dash?` stroke-dasharray="${dash}"`:""}/>`;
  const dot=(x,y,r,c,stroke=WHITE,sw=1.4)=>`<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r}" fill="${c}" stroke="${stroke}" stroke-width="${sw}"/>`;
  const xfmt=(x,d=3)=>`${x>=0?"+":""}${x.toFixed(d)}%`;
  const box=(w,h,title,subtitle,b)=>`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img"><title>${esc(title)}</title><desc>${esc(subtitle)}</desc>${rect(0,0,w,h,BG,16)}${t(28,42,title,27,BLUE,750)}${t(30,68,subtitle,15,MUTED,500)}${b.join("")}</svg>\n`;
  const avg=a=>a.reduce((x,y)=>x+y,0)/a.length;
  const val=(id,m,key="best_seen_cut")=>Number(groups.get(id).get(m)[key]);
  const base=id=>val(id,"gw-rr");
  const gain=(id,m)=>100*(val(id,m)-base(id))/base(id);
  const graphFam=id=>groups.get(id).get("gw-rr").family;
  for(const id of ids){
    if(!featureBy.has(id))throw Error("Missing topology features for "+id);
    const g=groups.get(id),f=featureBy.get(id);
    if(g.size!==18 || f.family!==graphFam(id) || f.weight_type!==g.get("gw-rr").weight_type)throw Error("Graph-metadata join mismatch "+id);
    for(const m of methods)if(g.get(m).status!=="complete_16_best_seen" || +g.get(m).refined_starts!==16)throw Error("Incomplete "+id+" / "+m);
  }
  const ranked=methods.map(m=>({method:m,mean:avg(ids.map(id=>gain(id,m))),mean16:avg(ids.map(id=>100*(val(id,m,"mean_best_seen")-val(id,"gw-rr","mean_best_seen"))/val(id,"gw-rr","mean_best_seen")))})).sort((a,b)=>b.mean-a.mean);
  const stats=ranked.map((r,i)=>{
    const p=published.find(x=>x.method===r.method);
    if(!p || Math.abs(+p.mean_gain_pct-r.mean)>.00005)throw Error("Ranking conflict "+r.method);
    return {...r,rank:i+1,ciLo:+p.bootstrap_95_ci_low_pct,ciHi:+p.bootstrap_95_ci_high_pct,credit:+p.fractional_winner_credit};
  });
  const methodNames={ "hyperplane-sweep":"Hyperplane sweep","control-paired":"Paired K-Means","em-gibbs":"EM + Gibbs","cutaware-kmeans":"Cut-aware K-Means","em-hard":"EM-hard","em-cem":"EM + CEM","control-tabu":"Paired + Tabu","multigroup-k4":"Multigroup k=4","multigroup-k8":"Multigroup k=8","gw-threshold":"GW threshold","gw-rr":"GW-RR","mst-l2-longest":"MST L2 longest","mst-l2-balanced":"MST L2 balanced","mst-l2-diameter":"MST L2 diameter","mst-l2-bestcut":"MST L2 best-edge","mst-l2-multicut":"MST L2 multicut","mst-l1-bestcut":"MST L1 best-edge","mst-linf-bestcut":"MST Linf best-edge" };
  const label=m=>methodNames[m]||m;
  const files={};
  // All 18: exact equal-graph-weighted, best-of-16 relative gain.
  {
    const b=[], x0=335, scale=1830, y0=104, rowH=24.6, max=.42;
    for(let k=0;k<=4;k++){const xx=x0+scale*k/10;b.push(line(xx,90,xx,550,k===0?"#66849A":BORDER,k===0?2:1),t(xx,572,`${(k/10).toFixed(1)}%`,13,MUTED,500,"middle"));}
    for(let i=0;i<stats.length;i++){
      const z=stats[i], y=y0+i*rowH, width=z.mean*scale, co=i===0?F.random:(i<9?palette[(i-1)%palette.length]:"#92A7B4");
      b.push(t(30,y+13,`${String(z.rank).padStart(2," ")}  ${label(z.method)}`,15,INK,i===0?750:500));
      if(width>=0)b.push(rect(x0,y,width,16,co,3));
      else b.push(rect(x0+width,y,-width,16,"#C25D62",3));
      b.push(t(1110,y+14,xfmt(z.mean),15,i===0?F.random:INK,i===0?750:550,"end"));
    }
    b.push(t(1015,572,"Equal graph weighting · 71 graphs",13,MUTED,500,"end"));
    files["experiment025_method_ranking.svg"]=box(1200,590,"All 18 methods vs refined GW-RR","Mean relative gain of best-of-16 cuts; original signed graph objective",b);
  }
  // Bootstrap confidence-interval plot for leading methods.
  {
    const b=[],top=stats.slice(0,9),left=310,x0=310,scale=1800,y0=120,step=45;
    for(let k=0;k<=4;k++){let xx=x0+scale*k/10;b.push(line(xx,90,xx,510,k===0?"#7494A9":BORDER,k===0?2:1),t(xx,537,(k/10).toFixed(1)+"%",14,MUTED,500,"middle"));}
    for(let i=0;i<top.length;i++){
      const s=top[i],y=y0+i*step,lo=x0+scale*s.ciLo,hi=x0+scale*s.ciHi,mid=x0+scale*s.mean;
      b.push(t(29,y+5,label(s.method),17,INK,i===0?750:500),line(lo,y,hi,y,i===0?F.random:"#6286A2",4),line(lo,y-7,lo,y+7),line(hi,y-7,hi,y+7),dot(mid,y,6,i===0?F.random:"#6286A2"));
      b.push(t(1120,y+5,xfmt(s.mean),16,INK,600,"end"));
    }
    b.push(t(38,565,"95% percentile graph bootstrap intervals from the aggregate report; not per-seed confidence intervals.",14,MUTED));
    files["experiment025_bootstrap_intervals.svg"]=box(1200,585,"Leaders and uncertainty","30,000 graph-level resamples, not independent GW seed bootstrap",b);
  }
  // family mean gain heatmap.
  const top=stats.slice(0,9);
  function heatColor(z){
    if(z>=0){const a=Math.min(z/.52,1);return `rgb(${Math.round(234-162*a)},${Math.round(244-93*a)},${Math.round(249-79*a)})`;}
    const a=Math.min(-z/.3,1);return `rgb(${Math.round(247-15*a)},${Math.round(238-127*a)},${Math.round(238-120*a)})`;
  }
  {
    const b=[],colXs=[365,615,865],rowY=145,rowH=43;
    for(let j=0;j<3;j++){const fam=familyOrder[j];b.push(t(colXs[j]+102,105,`${familyName[fam]} (n=${ids.filter(id=>graphFam(id)===fam).length})`,18,BLUE,700,"middle"));}
    for(let i=0;i<top.length;i++){
      const m=top[i].method,y=rowY+i*rowH;
      b.push(t(34,y+24,label(m),19,INK,i===0?750:500));
      for(let j=0;j<3;j++){
        const fam=familyOrder[j], member=ids.filter(id=>graphFam(id)===fam),g=avg(member.map(id=>gain(id,m))),x=colXs[j];
        b.push(rect(x,y,204,33,heatColor(g),5),t(x+102,y+23,xfmt(g),18,INK,650,"middle"));
      }
    }
    b.push(t(34,569,"Generator family is metadata, not a graph-theoretic planarity certificate. Shading is within-graph % gain.",15,MUTED));
    files["experiment025_family_method_heatmap.svg"]=box(1200,590,"Do method improvements depend on graph family?","Equal-graph mean best-of-16 gain (%) relative to each graph's refined GW baseline",b);
  }
  // weight type heatmap
  {
    const b=[],keys=["positive","signed"],xcols=[410,770],y0=142,step=44;
    for(let j=0;j<2;j++)b.push(t(xcols[j]+120,105,`${keys[j][0].toUpperCase()+keys[j].slice(1)} (n=${ids.filter(id=>groups.get(id).get("gw-rr").weight_type===keys[j]).length})`,19,BLUE,700,"middle"));
    for(let i=0;i<top.length;i++){
      const y=y0+i*step,m=top[i].method;
      b.push(t(34,y+23,label(m),19,INK,i===0?750:500));
      for(let j=0;j<2;j++){
        const member=ids.filter(id=>groups.get(id).get("gw-rr").weight_type===keys[j]),g=avg(member.map(id=>gain(id,m)));
        b.push(rect(xcols[j],y,238,33,heatColor(g),5),t(xcols[j]+119,y+23,xfmt(g),19,INK,650,"middle"));
      }
    }
    b.push(t(34,566,"Signedness is not independently randomized across families; these are descriptive comparisons.",15,MUTED));
    files["experiment025_sign_method_heatmap.svg"]=box(1200,585,"Positive vs signed edge weights","Mean within-graph relative gain (%) by weight-type stratum",b);
  }
  // graph-by-graph gain panel: one prespecified method + oracle.
  const winning=id=>Math.max(...methods.map(m=>val(id,m)));
  {
    const b=[],x0=70,x1=1155,y0=120,y1=440,n=ids.length;
    const all=ids.flatMap(id=>[gain(id,"hyperplane-sweep"),100*(winning(id)-base(id))/base(id)]);
    const mn=Math.min(-.2,...all),mx=Math.max(.6,...all);
    const range=mx-mn;
    const sy=v=>y1-(v-mn)/range*(y1-y0), sx=i=>x0+(x1-x0)*i/(n-1);
    let first=0;
    for(const fam of familyOrder){
      const cnt=ids.filter(id=>graphFam(id)===fam).length;
      const a=sx(first)-7,z=sx(first+cnt-1)+8;
      b.push(rect(a,y0,z-a,y1-y0,F[fam]+"15",0),t((a+z)/2,101,`${familyName[fam]} (${cnt})`,17,F[fam],700,"middle"));first+=cnt;
    }
    for(let k=0;k<=5;k++){
      const v=mn+range*k/5,y=sy(v);b.push(line(x0,y,x1,y,v===0?"#203F5E":BORDER,v===0?2:1,v===0?"4 4":""),t(61,y+5,`${v.toFixed(2)}%`,13,MUTED,500,"end"));
    }
    ids.forEach((id,i)=>{
      const x=sx(i),c=F[graphFam(id)];
      b.push(dot(x,sy(gain(id,"hyperplane-sweep")),4,c,WHITE,1));
      b.push(`<circle cx="${x.toFixed(1)}" cy="${sy(100*(winning(id)-base(id))/base(id)).toFixed(1)}" r="4.5" fill="none" stroke="#263F62" stroke-width="1.2" opacity=".75"/>`);
      if(i%5===0 || i===n-1)b.push(t(x,468,id,11,MUTED,500,"middle"));
    });
    b.push(dot(118,508,6,F.random),t(132,513,"Fixed hyperplane-sweep",16,INK),`<circle cx="417" cy="509" r="6" fill="none" stroke="#263F62" stroke-width="1.8"/>`,t(433,513,"Best of all 18 (in-sample oracle)",16,INK));
    b.push(t(41,557,"Graphwise oracle has selection advantage by construction; it is not a deployable fixed method.",15,MUTED));
    files["experiment025_graphwise_gains.svg"]=box(1200,580,"71 graphs: fixed method vs oracle selection","Best-of-16 relative gain over refined GW-RR; ordered by G-set graph identifier",b);
  }
  // winner credit distribution by family.
  {
    const credit={};
    for(const fam of familyOrder){
      credit[fam]={};
      for(const m of methods)credit[fam][m]=0;
      for(const id of ids.filter(q=>graphFam(q)===fam)){
        const best=winning(id),winners=methods.filter(m=>val(id,m)===best);
        for(const m of winners)credit[fam][m]+=1/winners.length;
      }
    }
    const leaders=[...methods].sort((a,b)=>familyOrder.reduce((s,f)=>s+credit[f][b],0)-familyOrder.reduce((s,f)=>s+credit[f][a],0)).slice(0,6);
    const cols=Object.fromEntries(leaders.map((m,i)=>[m,palette[i]]));cols["Other"]="#A6B5C2";
    const b=[],x0=95,totalW=1030,yRows=[157,286,415];
    familyOrder.forEach((fam,j)=>{
      const memb=ids.filter(id=>graphFam(id)===fam),sum=memb.length,y=yRows[j];
      b.push(t(95,y-19,`${familyName[fam]} · ${sum} graphs`,18,BLUE,700));
      let x=x0;
      for(const m of [...leaders,"Other"]){
        const v=m==="Other"?sum-leaders.reduce((s,n)=>s+credit[fam][n],0):credit[fam][m],w=totalW*v/sum;
        if(w>0){b.push(rect(x,y,w,38,cols[m],0), w>55?t(x+w/2,y+24,(100*v/sum).toFixed(0)+"%",14,WHITE,700,"middle"):"");}
        x+=w;
      }
    });
    let lx=45;
    for(let i=0;i<[...leaders,"Other"].length;i++){
      const m=[...leaders,"Other"][i],wid=i>=4?220:205;
      b.push(rect(lx,490,14,14,cols[m],3),t(lx+20,503,label(m),13,INK,600));lx+=wid;
    }
    b.push(t(45,556,"Each graph contributes exactly one winner credit, divided equally among tied methods.",14,MUTED));
    files["experiment025_family_fractional_winners.svg"]=box(1200,577,"Which methods win within each generator family?","Fractional in-sample winner credit among all 18 methods; ties split 1/k",b);
  }
  // topology metrics: 3 scatter plots; avoid predictive claims.
  const highlight=new Set(["G1","G40","G42","G77","G81"]);
  function scatterFeature(name,key,scaleMode,xTitle){
    const b=[],x0=90,x1=1120,y0=116,y1=428;
    const xs=ids.map(id=>Number(featureBy.get(id)[key]));
    const ys=ids.map(id=>gain(id,"hyperplane-sweep"));
    const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(-.12,...ys),ymax=Math.max(.5,...ys);
    const xx=v=>scaleMode==="log"?Math.log10(v):v;
    const a=xx(xmin),z=xx(xmax),sx=v=>x0+((xx(v)-a)/(z-a||1))*(x1-x0),sy=v=>y1-(v-ymin)/(ymax-ymin)*(y1-y0);
    for(let k=0;k<=5;k++){
      const x=x0+(x1-x0)*k/5,v=scaleMode==="log"?10**(a+(z-a)*k/5):xmin+(xmax-xmin)*k/5;
      b.push(line(x,y0,x,y1,BORDER),t(x,455,key==="negative_edge_fraction"?(100*v).toFixed(0)+"%":v.toFixed(key==="average_degree"?0:2),14,MUTED,500,"middle"));
      const y=y0+(y1-y0)*k/5,rv=ymax-(ymax-ymin)*k/5;
      b.push(line(x0,y,x1,y,rv===0?"#67869B":BORDER,rv===0?2:1,rv===0?"5 4":""),t(77,y+5,rv.toFixed(2)+"%",13,MUTED,500,"end"));
    }
    for(const id of ids){
      const x=sx(Number(featureBy.get(id)[key])),y=sy(gain(id,"hyperplane-sweep")),col=F[graphFam(id)];
      b.push(dot(x,y,5,col,WHITE,1.4));
      if(highlight.has(id))b.push(t(x+9,y-8,id,14,BLUE,750));
    }
    b.push(t((x0+x1)/2,489,xTitle,16,INK,650,"middle"));
    let lx=70;for(const fam of familyOrder){b.push(dot(lx,536,6,F[fam]),t(lx+12,542,familyName[fam],16,INK));lx+=270;}
    b.push(t(44,575,"Point = one original graph; colors are G-set generator families. Association is descriptive, not causal.",14,MUTED));
    return box(1200,592,name,"Observed fixed hyperplane-sweep relative gain vs refined GW-RR (%)",b);
  }
  files["experiment025_degree_association.svg"]=scatterFeature("Structure: degree and improvement","average_degree","log","Mean unweighted degree (log scale)");
  files["experiment025_transitivity_association.svg"]=scatterFeature("Structure: triangles and improvement","transitivity","linear","Graph transitivity (global clustering coefficient)");
  files["experiment025_negative_edges_association.svg"]=scatterFeature("Structure: edge signs and improvement","negative_edge_fraction","linear","Fraction of negative-weight edges (from original graph)");
  // Graph × top 10 methods heatmap.
  {
    const b=[],mths=stats.slice(0,10).map(s=>s.method),x0=232,y0=125,cellW=13.0,cellH=37;
    b.push(t(234,94,"Random (30)                                   Planar-family (24)                      Toroidal (17)",16,BLUE,650));
    for(let i=0;i<mths.length;i++){
      const m=mths[i],yy=y0+i*cellH;
      b.push(t(20,yy+19,label(m),15,INK,i===0?750:500));
      for(let j=0;j<ids.length;j++){
        const g=gain(ids[j],m);
        b.push(rect(x0+j*cellW,yy,cellW-1,cellH-2,heatColor(g),0));
      }
    }
    [30,54].forEach(i=>b.push(line(x0+i*cellW-1,y0-4,x0+i*cellW-1,y0+10*cellH+4,BLUE,2)));
    for(let j=0;j<ids.length;j++)if(j%5===0||j===70)b.push(t(x0+j*cellW+5,y0+10*cellH+24,ids[j],11,MUTED,500,"middle"));
    const y=540;for(let k=0;k<5;k++){const v=-.2+.2*k;b.push(rect(400+k*85,y,80,17,heatColor(v)),t(441+k*85,y+37,xfmt(v,1),12,MUTED,500,"middle"));}
    b.push(t(24,559,"Negative (rose)  /  zero (pale)  /  positive (teal); color saturates above +0.52%.",14,MUTED));
    files["experiment025_graph_method_heatmap.svg"]=box(1200,586,"Structure-stratified method × graph matrix","Each cell = best-of-16 % improvement over graph-matched refined GW-RR",b);
  }
  // Paired distinction: best-of-16 vs mean-of-16 for leading methods.
  {
    const b=[],sel=stats.slice(0,9),maxV=Math.max(...sel.flatMap(s=>[s.mean,s.mean16]))*1.13,y0=112,step=47,x0=335,x1=1085,scale=(x1-x0)/maxV;
    for(let k=0;k<=5;k++){const xx=x0+(x1-x0)*k/5;b.push(line(xx,y0-15,xx,542,BORDER),t(xx,561,(maxV*k/5).toFixed(1)+"%",13,MUTED,500,"middle"));}
    sel.forEach((s,i)=>{
      const y=y0+i*step;
      b.push(t(29,y+13,label(s.method),18,INK,i===0?750:500));
      b.push(rect(x0,y-3,s.mean16*scale,13,"#6DAEBA",2),rect(x0,y+13,s.mean*scale,13,"#204F82",2));
    });
    b.push(rect(390,577,16,12,"#6DAEBA",2),t(414,589,"Mean of 16 refined starts",14,INK),rect(760,577,16,12,"#204F82",2),t(785,589,"Best of 16 refined starts",14,INK));
    files["experiment025_mean_vs_best.svg"]=box(1200,602,"The ranking target changes the story","Two distinct aggregate estimands; not a paired per-seed variance analysis",b);
  }
  // Eight reference candidates
  {
    const candidates=ids.map(id=>({id,best:winning(id),ref:Number(groups.get(id).get("gw-rr").reference_cut)})).filter(x=>x.best>x.ref);
    if(candidates.length!==8)throw Error("Reference candidate count changed, got "+candidates.length);
    const b=[],x0=218,x1=1070,y0=120,rowH=48;
    candidates.forEach((q,i)=>{
      const y=y0+i*rowH,delta=q.best-q.ref;
      b.push(t(40,y+21,q.id,21,BLUE,700),rect(x0,y,delta/5*(x1-x0),31,F.random,4),t(x0+delta/5*(x1-x0)+16,y+21,"+"+delta.toFixed(0),20,INK,750));
      b.push(t(1115,y+21,`${q.ref.toFixed(0)} → ${q.best.toFixed(0)}`,16,MUTED,500,"end"));
    });
    b.push(t(40,550,"Archived feasible references are NOT certified optima; every candidate requires independent partition verification.",15,MUTED));
    files["experiment025_reference_candidates.svg"]=box(1200,572,"Eight candidate improvements to archived feasible references","Gain in original signed cut units; provisional and not independently rescored",b);
  }
  const summary={
    files:Object.keys(files),nGraphs:ids.length,nMethods:methods.length,nCells:raw.length,
    familyCounts:Object.fromEntries(familyOrder.map(f=>[f,ids.filter(id=>graphFam(id)===f).length])),
    refCandidates:ids.filter(id=>winning(id)>Number(groups.get(id).get("gw-rr").reference_cut)),
    leader:stats[0].method,leaderGain:Number(stats[0].mean.toFixed(6)),
    noNumericalSolver:true
  };
  return {files,summary};
}

const root = process.cwd();
const csvFile = path.join(root, "results/gset/experiment025-gw16-bestcut-quality/analysis/full_best_final_by_graph.csv");
const featuresFile = path.join(root, "results/gset/experiment022-graph-structure-methods/graph_features.csv");
const rankingFile = path.join(root, "results/gset/experiment025-gw16-bestcut-quality/analysis/experiment025_method_ranking.csv");
const csv = fs.readFileSync(csvFile,"utf8");
const sha = createHash("sha256").update(fs.readFileSync(csvFile)).digest("hex");
if (sha !== "8a9d84cfc35f1d424519b862676facc288a6366bc1820e52975a2bf76678de6c") {
  throw new Error(`Unexpected Experiment025 source CSV SHA256: ${sha}`);
}
const {files, summary} = makeFigures(csv, fs.readFileSync(featuresFile,"utf8"), fs.readFileSync(rankingFile,"utf8"));
const dest = path.join(root,"presentations","assets");
fs.mkdirSync(dest,{recursive:true});
const check = process.argv.includes("--check");
for(const [name,content] of Object.entries(files)) {
 const p=path.join(dest,name);
 if(check) { if(!fs.existsSync(p)||fs.readFileSync(p,"utf8")!==content) throw Error(`Stale or missing generated SVG: ${p}`); }
 else fs.writeFileSync(p,content,"utf8");
}
const report=JSON.stringify({...summary,csvSha256:sha},null,2)+"\n";
const reportFile=path.join(root,"presentations","experiment025-figure-manifest.json");
if(check){if(!fs.existsSync(reportFile)||fs.readFileSync(reportFile,"utf8")!==report)throw Error("Figure manifest changed.");}
else fs.writeFileSync(reportFile,report,"utf8");
console.log(JSON.stringify({...summary,mode:check?"check":"generate"},null,2));
