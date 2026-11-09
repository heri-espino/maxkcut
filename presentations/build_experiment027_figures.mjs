#!/usr/bin/env node
/** Rebuild Experiment027 figures deterministically from committed verified CSV files.
 * No new EVA/SDP, cut optimization, internet access, or third-party packages.
 * Usage: node presentations/build_experiment027_figures.mjs [--check]
 */
import fs from "node:fs";
import path from "node:path";
import {createHash} from "node:crypto";
function parseCSV(s){const rows=s.trim().split(/\r?\n/);const keys=rows.shift().split(",");return rows.filter(Boolean).map(line=>{const vals=line.split(",");if(vals.length!==keys.length)throw Error("Bad CSV: "+line.slice(0,90));const o={};keys.forEach((k,i)=>{const v=vals[i];o[k]=v!==""&&Number.isFinite(Number(v))?Number(v):v});return o})}

function esc(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;")}

function txt(x,y,t,size=18,color="#17324D",weight=400,anchor="start",other=""){return `<text x="${x}" y="${y}" fill="${color}" font-size="${size}" font-family="Inter,Segoe UI,Arial,sans-serif" font-weight="${weight}" text-anchor="${anchor}" ${other}>${esc(t)}</text>`}

function rect(x,y,w,h,fill,r=0,stroke="none",sw=0){return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`}

function line(x1,y1,x2,y2,color="#D8E5EF",sw=1,dash=""){return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${sw}" ${dash?`stroke-dasharray="${dash}"`:""}/>`}

function circ(x,y,r,fill,stroke="none",sw=0){return `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`}

function frame(title,subtitle,content,foot="SOURCE: Experiment027 published 71-graph audit · 8 Nov 2026"){return `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="700" viewBox="0 0 1200 700" role="img" aria-label="${esc(title)}">${rect(0,0,1200,700,"#FFFFFF")}${rect(0,0,1200,9,"#0B6E92")}${txt(60,69,title,31,"#103453",700)}${txt(60,101,subtitle,18,"#667F91")}${line(60,115,1140,115)}${content}${line(60,656,1140,656)}${txt(62,680,foot,13,"#6C8193")}</svg>\n`}

function mean(a){return a.reduce((s,x)=>s+x,0)/a.length}

function percentile(a,p){const b=a.slice().sort((x,y)=>x-y),z=(b.length-1)*p,x=Math.floor(z);return b[x]+(b[Math.min(b.length-1,x+1)]-b[x])*(z-x)}

function bootCI(vals){let x=2026027;const r=2400,n=vals.length,a=[];for(let j=0;j<r;j++){let total=0;for(let k=0;k<n;k++){x=(Math.imul(x,1664525)+1013904223)>>>0;total+=vals[Math.floor((x/4294967296)*n)]}a.push(total/n)}return [percentile(a,.025),percentile(a,.975)]}

function blendColor(z,max=18){const t=Math.max(-1,Math.min(1,z/max));const mix=(a,b,u)=>Math.round(a*(1-u)+b*u);const neutral=[242,246,249],pos=[18,125,139],neg=[193,87,86],v=t>=0?pos:neg,u=Math.abs(t);return `rgb(${mix(neutral[0],v[0],u)},${mix(neutral[1],v[1],u)},${mix(neutral[2],v[2],u)})`}

function compute(raw){
 const graph=parseCSV(raw.graph), leaderboard=parseCSV(raw.leaderboard), pairs=parseCSV(raw.paired), seed=parseCSV(raw.seed), features=parseCSV(raw.features), index=JSON.parse(raw.index);
 const names=["angular-margin-greedy","angular-gain-greedy","angular-pair-greedy"],short=["Margin","Gain","Pair"],palette=["#0C6B91","#3A9A9B","#C58D37"];
 const graphs=[...new Set(graph.map(r=>r.graph))].sort((a,b)=>+a.slice(1)-+b.slice(1));
 if(graphs.length!==71||graph.length!==213||pairs.length!==6816||seed.length!==5680||features.length!==71||index.assets.length!==71)throw Error("Unexpected Experiment027 source coverage");
 const lookup=new Map(graph.map(r=>[r.graph+"|"+r.method,r]));
 const feat=new Map(features.map(r=>[r.graph,r]));
 const bySeed=new Map(seed.map(r=>[r.graph+"|"+r.method+"|"+r.seed,r]));
 const stats=names.map((m,i)=>{
   const r=graph.filter(x=>x.method===m),s=seed.filter(x=>x.method===m),p=pairs.filter(x=>x.method===m&&x.baseline==="hyperplane-sweep");
   return {method:m,short:short[i],color:palette[i],rows:r,avgTime:mean(s.map(x=>x.seconds)),rawDiffHP:mean(p.map(x=>x.delta_raw)),bestDiffHP:mean(p.map(x=>x.delta_best)),avgPaths:mean(s.map(x=>x.paths)),ciHP:bootCI(r.map(x=>x.delta_hyperplane))};
 });
 const all=graphs.map(g=>{const rs=names.map(m=>lookup.get(g+"|"+m));let best=Math.max(...rs.map(r=>r.best));return {graph:g,delta:best-rs[0].hyperplane_best,hp:rs[0].hyperplane_best,gw:rs[0].gw_best,winner:names[rs.findIndex(r=>r.best===best)]}});
 const chart= {};
 // Diagram: GW seed normal -> rank candidates -> exact sweeps -> accept strict improvement -> common refinement.
 {
 let b="";const boxes=[{x:64,w:195,name:"Shared GW normal",sub:"Same saved EVA factor"},{x:300,w:206,name:"Angular target",sub:"Margin / gain / pair"},{x:546,w:206,name:"Exact arc sweeps",sub:"Up to 8 per round"},{x:792,w:158,name:"Select gain",sub:"Strictly better cut"},{x:987,w:164,name:"Flip → KL → ILS",sub:"Common refiner"}];
 for(let i=0;i<boxes.length;i++){const o=boxes[i];b+=rect(o.x,212,o.w,114,"#E7F3F8",13,"#9CCDDD",2)+txt(o.x+o.w/2,253,o.name,19,"#104A6A",700,"middle")+txt(o.x+o.w/2,283,o.sub,15,"#50788A",400,"middle");if(i<boxes.length-1)b+=line(o.x+o.w,268,boxes[i+1].x-8,268,"#0F7495",3)+`<path d="M${boxes[i+1].x-16} 260 L${boxes[i+1].x-8} 268 L${boxes[i+1].x-16} 276" stroke="#0F7495" stroke-width="3" fill="none"/>`;}
 b+=txt(65,408,"Only strict improvements update the hyperplane; otherwise the angular search stops.",22,"#1D425B",600)+txt(65,451,"Every attempted semicircle is evaluated on the original signed cut objective.",19,"#46677C")+txt(65,494,"32 outer rounds × ≤8 candidate semicircle sweeps · 16 matched GW starts per graph/method.",19,"#46677C")+rect(62,537,1076,75,"#F2F7FA",12)+txt(82,582,"No new SDP/EVA solve · Original Experiment025 factor hashes and 16 GW normals reused.",20,"#16647D",600);
 chart["experiment027_protocol.svg"]=frame("Experiment027 | angular-guided search protocol","Three proposal rules · same signed-cut verifier · same downstream refinement",b);
 }
 // 1 mean relative gain with graph bootstrap
 {
 let b="",ybase=553,scale=860,labels=stats.map(s=>s.short);
 leaderboard.forEach((r,i)=>{let x=136+i*340,h=r.mean_relative_gw_pct*scale;const color=palette[i];b+=rect(x,ybase-h,145,h,color,10);const low=r.bootstrap95_low*scale, high=r.bootstrap95_high*scale,c=x+72;b+=line(c,ybase-low,c,ybase-high,"#0E284A",3)+line(c-15,ybase-low,c+15,ybase-low,"#0E284A",3)+line(c-15,ybase-high,c+15,ybase-high,"#0E284A",3)+txt(c,ybase-h-35,"+"+r.mean_relative_gw_pct.toFixed(3)+"%",23,"#103B59",700,"middle")+txt(c,593,labels[i],22,"#173A55",600,"middle");});
 [0,.1,.2,.3,.4].forEach(t=>{const y=ybase-t*scale;b+=line(70,y,1110,y,"#DDE6EC",1)+txt(61,y+6,t.toFixed(1)+"%",15,"#667A90",400,"end")});b+=txt(600,630,"Equal-weight mean across 71 graphs; 95% bootstrap resamples graphs, not seed runs.",18,"#46687F",400,"middle");
 chart["experiment027_gain_vs_gw.svg"]=frame("Gain over refined GW-RR | best of 16","Relative change = 100 × (best new cut − best refined GW cut) / |best refined GW cut|",b);
 }
 // 2 outcomes HP
 {
 let b="";const ax=140,ww=870,barH=78;stats.forEach((s,i)=>{const x=ax,y=205+i*133,L=71,segment=[leaderboard[i].wins_vs_hyperplane,leaderboard[i].ties_vs_hyperplane,leaderboard[i].losses_vs_hyperplane],col=["#237E8B","#9DAAB7","#C36D6D"];b+=txt(54,y+30,s.short,22,"#20445D",600);let pos=x;segment.forEach((v,j)=>{let w=ww*v/L;b+=rect(pos,y-14,w,barH,col[j],0)+txt(pos+w/2,y+31,String(v),26,"#FFFFFF",700,"middle");pos+=w});});b+=txt(201,629,"Wins",18,"#237E8B",700)+txt(373,629,"Ties",18,"#647A8E",700)+txt(539,629,"Losses",18,"#C36D6D",700)+txt(791,629,"One graph = one vote (71 total)",16,"#667A90");
 chart["experiment027_vs_hyperplane.svg"]=frame("Graphwise outcomes | against Hyperplane Sweep","Higher is better; best-of-16 final cut after the common refinement pipeline",b);
 }
 // 3 mean diffs HP CI
 {
 let b="",zero=370,scale=17;
 [-6,-4,-2,0,2,4,6].forEach(t=>{const y=zero-t*scale;b+=line(92,y,1098,y,t===0?"#173E59":"#DFE9EF",t===0?2:1)+txt(72,y+5,String(t),15,"#647E90",400,"end")});
 stats.forEach((s,i)=>{const x=263+i*335,point=mean(s.rows.map(r=>r.delta_hyperplane)),low=s.ciHP[0],hi=s.ciHP[1];b+=line(x,zero-low*scale,x,zero-hi*scale,s.color,5)+line(x-20,zero-low*scale,x+20,zero-low*scale,s.color,3)+line(x-20,zero-hi*scale,x+20,zero-hi*scale,s.color,3)+circ(x,zero-point*scale,12,s.color)+txt(x,557,s.short,24,"#173A55",700,"middle")+txt(x,598,(point>=0?"+":"")+point.toFixed(2)+" units",21,"#173A55",600,"middle");});
 b+=txt(596,154,"All three graph-bootstrap intervals include 0; no resolved single-method advantage.",19,"#5F7181",600,"middle");b+=txt(600,632,"Exploratory 2,400-replicate graph bootstrap; separate from the published GW intervals.",16,"#657E90",400,"middle");
 chart["experiment027_delta_hp_ci.svg"]=frame("Against Hyperplane Sweep | mean cut differences","Absolute best-of-16 cut difference across graphs · 95% graph bootstrap",b);
 }
 // 4 71 x 3 graphwise heatmap
 {
 let b="",left=185,top=150,colW=240,rowH=6.55;graphs.forEach((g,j)=>{if(j%5===0)b+=txt(175,top+j*rowH+6,g,13,"#4F6B7D",400,"end");stats.forEach((s,i)=>{const value=lookup.get(g+"|"+s.method).delta_hyperplane;b+=rect(left+i*300,top+j*rowH,colW,rowH-.7,blendColor(value,22),0)})});stats.forEach((s,i)=>b+=txt(left+i*300+colW/2,135,s.short,19,"#1E4B64",700,"middle"));b+=txt(975,207,"Difference",19,"#1E4960",700)+txt(975,236,"vs HP",18,"#1E4960");[-20,-10,0,10,20].forEach((v,j)=>{b+=rect(976,258+j*48,50,36,blendColor(v,22))+txt(1037,281+j*48,(v>0?"+":"")+v,18,"#456478")});b+=txt(600,631,"Graph IDs sorted numerically; red = worse than HP, teal = better, near-white = tie.",17,"#5B7284",400,"middle");
 chart["experiment027_graph_matrix.svg"]=frame("Full G-set | graph × method outcome matrix","Each tile = best-of-16 signed-cut difference vs Experiment025 Hyperplane Sweep (cut units)",b);
 }
 // 5 paired raw and refined vs HP
 {
 let b="",zero=370,scale=11;
 [-25,-20,-15,-10,-5,0,5].forEach(v=>{const y=zero-v*scale;b+=line(110,y,1120,y,v===0?"#24485E":"#E3ECF1",v===0?2:1)+txt(90,y+6,String(v),16,"#667B8A",400,"end")});
 stats.forEach((s,i)=>{const x=220+i*330;[s.rawDiffHP,s.bestDiffHP].forEach((v,j)=>{const cx=x+j*89,h=Math.abs(v*scale),top=zero-v*scale,y=Math.min(zero,top);b+=rect(cx,y,54,Math.max(2,h),j===0?"#9DB5C6":s.color,7)+txt(cx+27,v<0?zero+24:top-12,v.toFixed(1),17,"#1E465F",600,"middle")});b+=txt(x+70,583,s.short,23,"#173C58",700,"middle")});
 b+=rect(340,613,18,18,"#9DB5C6")+txt(367,627,"Raw rounding",17,"#4F6B80")+rect(582,613,18,18,"#0C6B91")+txt(610,627,"Best after common refinement",17,"#4F6B80");
 chart["experiment027_raw_refined.svg"]=frame("Matched seeds | raw vs refined difference from HP","Mean signed-cut difference across 71 graphs × 16 matched GW seeds per method",b);
 }
 // 6 stage-wise difference vs HP (Round Flip KL Final)
 {
 let b="",stages=["raw","flip","kl","best"],zero=390,yscale=9,xx=[220,470,720,970];
 [-25,-20,-15,-10,-5,0,5].forEach(v=>{const y=zero-v*yscale;b+=line(150,y,1080,y,v===0?"#193E58":"#E2ECF1",v===0?2:1)+txt(135,y+6,String(v),16,"#5A7485",400,"end")});
 stages.forEach((s,i)=>b+=txt(xx[i],559,s==="best"?"Final best":s[0].toUpperCase()+s.slice(1),20,"#27516C",600,"middle"));
 stats.forEach((s,idx)=>{const values=stages.map(st=>mean(seed.filter(r=>r.method===s.method).map(r=>r[st]-bySeed.get(r.graph+"|hyperplane-sweep|"+r.seed)[st]))),points=values.map((v,i)=>[xx[i],zero-v*yscale]);b+=`<polyline points="${points.map(p=>p.join(",")).join(" ")}" fill="none" stroke="${s.color}" stroke-width="4"/>`;points.forEach(p=>b+=circ(p[0],p[1],8,s.color));b+=txt(xx[3]+14,points[3][1]+(idx-1)*16,values[3].toFixed(1),16,s.color,700);});
 stats.forEach((s,i)=>{b+=rect(300+i*245,604,18,18,s.color)+txt(324+i*245,619,s.short,17,"#466579")});
 chart["experiment027_refinement_stages.svg"]=frame("Refinement collapses the raw rounding gap","Average matched per-seed Δcut vs HP, at each saved stage (not a controlled time series)",b);
 }
 // 7 runtime / quality scatter
 {
 let b="",zeroX=180,yAxis=550,xAxis=980,msPerSec=240,yScale=500;const points=[{name:"GW-RR",time:mean(seed.filter(r=>r.method==="gw-rr").map(r=>r.seconds)),gain:0,color:"#8295A6"},{name:"Hyperplane Sweep",time:mean(seed.filter(r=>r.method==="hyperplane-sweep").map(r=>r.seconds)),gain:mean(graph.filter(r=>r.method===names[0]).map(r=>100*(r.hyperplane_best-r.gw_best)/Math.abs(r.gw_best))),color:"#1D3559"},...stats.map((s,i)=>({name:s.short,time:s.avgTime,gain:leaderboard[i].mean_relative_gw_pct,color:s.color}))];
 [0,.1,.2,.3,.4,.5].forEach(t=>{const y=yAxis-t*yScale;b+=line(zeroX,y,1100,y,"#E2EAF0")+txt(166,y+5,t.toFixed(1)+"%",16,"#587086",400,"end")});
 [.5,1,1.5,2,2.5,3].forEach(t=>{const x=zeroX+t*msPerSec;b+=line(x,180,x,yAxis,"#E7EEF2")+txt(x,582,t.toFixed(1),16,"#60788A",400,"middle")});
 points.forEach(p=>{const x=zeroX+p.time*msPerSec,y=yAxis-p.gain*yScale;b+=circ(x,y,12,p.color,"#FFFFFF",3)+txt(x+18,y-13,p.name==="Hyperplane Sweep"?"Hyperplane Sweep":p.name,17,p.color,700)});
 b+=txt(660,620,"Mean recorded seconds per seed (historical runs; not controlled hardware)",18,"#586F82",400,"middle");
 chart["experiment027_quality_time.svg"]=frame("Quality versus recorded runtime | descriptive","Equal-graph best-of-16 gain over refined GW versus average archived per-seed seconds",b);
 }
 // 8 family heatmap
 {
 let b="",families=[["random","Random (30)"],["planar","Planar-family (24)"],["toroidal","Toroidal (17)"]],left=330,top=211;
 families.forEach(([f,l],j)=>{b+=txt(296,top+j*114+57,l,21,"#234E66",600,"end");stats.forEach((s,i)=>{const d=s.rows.filter(r=>feat.get(r.graph).family===f),v=mean(d.map(x=>x.relative_hyperplane_pct));const x=left+i*265,y=top+j*114;b+=rect(x,y,232,82,blendColor(v,.35),10)+txt(x+116,y+52,(v>=0?"+":"")+v.toFixed(3)+"%",24,Math.abs(v)>.15?"#FFFFFF":"#173B51",700,"middle")})});
 stats.forEach((s,i)=>b+=txt(left+i*265+116,174,s.short,23,"#224C63",700,"middle"));
 b+=txt(600,626,"Exploratory family strata; generator family is not a planarity certificate.",18,"#597386",400,"middle");
 chart["experiment027_family.svg"]=frame("Experiment027 | G-set generator-family differences","Mean graphwise relative best-of-16 advantage vs Hyperplane Sweep, within family",b);
 }
 // 9 sign heatmap
 {
 let b="",sign=[["positive","Positive weights (35)"],["signed","Signed weights (36)"]],left=320,top=222;
 sign.forEach(([f,l],j)=>{b+=txt(299,top+j*151+59,l,20,"#234E66",600,"end");stats.forEach((s,i)=>{const r=s.rows.filter(x=>feat.get(x.graph).weight_type===f),v=mean(r.map(x=>x.relative_hyperplane_pct)),x=left+i*273,y=top+j*151;b+=rect(x,y,235,110,blendColor(v,.28),10)+txt(x+117,y+65,(v>=0?"+":"")+v.toFixed(3)+"%",26,Math.abs(v)>.13?"#FFFFFF":"#173B51",700,"middle")})});
 stats.forEach((s,i)=>b+=txt(left+i*273+117,182,s.short,23,"#214C65",700,"middle"));b+=txt(600,619,"Sign type and graph family are not randomized independently: descriptive association only.",18,"#597386",400,"middle");
 chart["experiment027_signedness.svg"]=frame("Experiment027 | signed versus positive-weight instances","Mean graphwise relative best-of-16 advantage vs Hyperplane Sweep, within sign type",b);
 }
 // 10 best-of-three and caveat
 {
 let b="",counts=[all.filter(x=>x.delta>0).length,all.filter(x=>x.delta===0).length,all.filter(x=>x.delta<0).length],col=["#237E8B","#9DAAB7","#C36D6D"],labels=["Wins","Ties","Losses"];
 counts.forEach((v,i)=>{const x=225+i*325,h=v*9.7;b+=rect(x,564-h,175,h,col[i],10)+txt(x+87,564-h-19,String(v),33,col[i],800,"middle")+txt(x+87,609,labels[i],22,"#1A4561",600,"middle")});
 b+=txt(610,152,"Best of 3 × 16 starts versus 1 × 16 HP starts",20,"#5B7287",700,"middle");
 chart["experiment027_portfolio.svg"]=frame("An ex post portfolio improves graph coverage","Portfolio selects the largest cut across all three new methods after observing outcomes",b,"CAUTION: ≥3× method budget; in-sample oracle, not same-cost superiority");
 }
 // 11 best / worst exemplars
 {
 let b="",ordered=all.slice().sort((a,b)=>b.delta-a.delta),chosen=[...ordered.slice(0,6),...ordered.slice(-6)].sort((a,b)=>b.delta-a.delta);const zero=600,scale=18;
 b+=line(zero,168,zero,596,"#173C52",2);chosen.forEach((o,i)=>{const y=178+i*35,x1=Math.min(zero,zero+o.delta*scale),w=Math.abs(o.delta)*scale;b+=rect(x1,y,w,22,o.delta>=0?"#207F89":"#C87272",4)+txt(o.delta>=0?x1+w+11:x1-11,y+17,(o.delta>0?"+":"")+o.delta,16,"#274E66",700,o.delta>=0?"start":"end")+txt(126,y+17,o.graph,17,"#1B4960",700)});
 b+=txt(1048,203,"Best of three",20,"#2C6178",600,"middle")+txt(1048,239,"vs HP",20,"#2C6178",600,"middle")+txt(1048,284,"+19 on G23",22,"#1F7D87",700,"middle")+txt(1048,322,"−13 on G31",22,"#C87272",700,"middle");
 chart["experiment027_extremes.svg"]=frame("Where the new portfolio helps or hurts","Top six gains and bottom six deficits versus Hyperplane Sweep (absolute cut units)",b);
 }
 // 12 empirical survival across all paired differences - use 3 cumulative curves
 {
 let b="",left=155,right=1090,top=160,bottom=567,xmin=-70,xmax=70;
 for(let v=-60;v<=60;v+=20){const x=left+(v-xmin)/(xmax-xmin)*(right-left);b+=line(x,top,x,bottom,"#E7EDF2")+txt(x,bottom+28,String(v),16,"#5F7489",400,"middle")}
 for(let t=0;t<=1;t+=.25){const y=bottom-t*(bottom-top);b+=line(left,y,right,y,"#E6EDF1")+txt(left-21,y+5,Math.round(100*t)+"%",16,"#60788A",400,"end")}
 stats.forEach(s=>{const sorted=pairs.filter(x=>x.method===s.method&&x.baseline==="hyperplane-sweep").map(x=>x.delta_best).sort((a,b)=>a-b);const pts=sorted.map((v,i)=>[left+(Math.max(xmin,Math.min(xmax,v))-xmin)/(xmax-xmin)*(right-left),bottom-(i+1)/sorted.length*(bottom-top)]);b+=`<polyline points="${pts.map(p=>p.join(",")).join(" ")}" fill="none" stroke="${s.color}" stroke-width="3" opacity=".88"/>`;});
 stats.forEach((s,i)=>{b+=rect(260+i*300,626,20,12,s.color)+txt(292+i*300,637,s.short,18,"#4C6D7D")});
 chart["experiment027_paired_cdf.svg"]=frame("Seed-level matched differences | empirical CDF","Each curve: 71 × 16 paired Δbest cuts vs HP; SEED observations are clustered within graph",b);
 }
 const manifest={schema:"experiment027-marp-figures-v1",release_tag:index.release_tag,graphs:71,new_methods:3,new_seed_archives:3408,audited_seed_archives:5680,full_graph_rows:graph.length,paired_rows:pairs.length,source_sha256:{graph:index.analysis_checksums["audited_graph_comparisons.csv"],leaderboard:index.analysis_checksums["audited_method_leaderboard.csv"],paired:index.analysis_checksums["audited_paired_seed_deltas.csv"],seed:index.analysis_checksums["audited_seed_scores.csv"]},feature_source:"results/gset/experiment022-graph-structure-methods/graph_features.csv",files:Object.keys(chart).sort(),portfolio:{wins:42,ties:13,losses:16,mean_delta_hp:mean(all.map(x=>x.delta))}};
 if(manifest.portfolio.wins!==42||manifest.portfolio.ties!==13||manifest.portfolio.losses!==16)throw Error("Unexpected portfolio tally");
 return {figures:chart,manifest,stats:stats.map(x=>({name:x.method,rawDiffHP:x.rawDiffHP,bestDiffHP:x.bestDiffHP,ciHP:x.ciHP,avgTime:x.avgTime})),portfolio:manifest.portfolio};
}
const root=process.cwd();
const tag="experiment027-data-71g-2b313a624780291eb4dd";
const prefix=path.join(root,"published/experiment027/releases",tag);
const inputs={
 graph:fs.readFileSync(path.join(prefix,"analysis/audited_graph_comparisons.csv"),"utf8"),
 leaderboard:fs.readFileSync(path.join(prefix,"analysis/audited_method_leaderboard.csv"),"utf8"),
 paired:fs.readFileSync(path.join(prefix,"analysis/audited_paired_seed_deltas.csv"),"utf8"),
 seed:fs.readFileSync(path.join(prefix,"analysis/audited_seed_scores.csv"),"utf8"),
 features:fs.readFileSync(path.join(root,"results/gset/experiment022-graph-structure-methods/graph_features.csv"),"utf8"),
 index:fs.readFileSync(path.join(prefix,"index.json"),"utf8")};
const hash=s=>createHash("sha256").update(s,"utf8").digest("hex");
const declared=JSON.parse(inputs.index).analysis_checksums;
for(const [k,f] of [["graph","audited_graph_comparisons.csv"],["leaderboard","audited_method_leaderboard.csv"],["paired","audited_paired_seed_deltas.csv"],["seed","audited_seed_scores.csv"]]){
 if(hash(inputs[k])!==declared[f])throw Error("Changed or corrupted published CSV: "+f);
}
const output=compute(inputs);
const dest=path.join(root,"presentations/assets");
const check=process.argv.includes("--check");
if(!check)fs.mkdirSync(dest,{recursive:true});
for(const [file,svg] of Object.entries(output.figures)){
 const target=path.join(dest,file);
 if(check){if(!fs.existsSync(target)||fs.readFileSync(target,"utf8")!==svg)throw Error("Missing/non-deterministic Experiment027 figure: "+file);}
 else fs.writeFileSync(target,svg,"utf8");
}
const manifest=JSON.stringify(output.manifest,null,2)+"\n";
const target=path.join(root,"presentations/experiment027-figure-manifest.json");
if(check){if(!fs.existsSync(target)||fs.readFileSync(target,"utf8")!==manifest)throw Error("Experiment027 figure manifest drift");}
else fs.writeFileSync(target,manifest,"utf8");
console.log(JSON.stringify({mode:check?"check":"generated",figures:Object.keys(output.figures).length,graphs:71,pairedRows:output.manifest.paired_rows},null,2));
