#!/usr/bin/env node
/** Reproduce 3 Experiment025 external reference charts, no network / optimization.
 * References frozen 2026-11-05: Quicopt fa2b8bc70a8e, OpenIsing 5d22c7d5e375.
 * Usage: node presentations/build_experiment025_reference_ratio_figures.mjs [--check]
 */
import fs from "node:fs";
import path from "node:path";
function generate(refText,fullText){
const parse=s=>{let lines=s.trim().split(/\r?\n/),head=lines.shift().split(",");return lines.map(l=>{let cells=l.split(",");if(cells.length!==head.length)throw Error("CSV malformed");return Object.fromEntries(head.map((k,i)=>[k,cells[i]]))})};
const refs=parse(refText),full=parse(fullText),groups=new Map(),refMap=new Map(refs.map(r=>[r.graph,r]));
for(const r of full){if(!groups.has(r.graph))groups.set(r.graph,new Map());groups.get(r.graph).set(r.method,r)}
const graphs=[...groups.keys()].sort((a,b)=>+a.slice(1)-+b.slice(1));
const methods=[...groups.get(graphs[0]).keys()];
if(graphs.length!==71||groups.size!==71||methods.length!==18||full.length!==1278||refs.length!==71)throw Error("71x18 missing");
const colors={"hyperplane-sweep":"#117E99","control-paired":"#BF812D","em-gibbs":"#7157B4","cutaware-kmeans":"#278C53","em-hard":"#9661AE","em-cem":"#B473C5","control-tabu":"#D55C42","multigroup-k4":"#406FBA","multigroup-k8":"#84944A","gw-threshold":"#A6754B","gw-rr":"#576E85","mst-l2-longest":"#617E6A","mst-l2-balanced":"#708B77","mst-l2-diameter":"#839784","mst-l2-bestcut":"#8FA18F","mst-l2-multicut":"#9DAC9B","mst-l1-bestcut":"#708E9B","mst-linf-bestcut":"#8796AA"};
const names={"hyperplane-sweep":"Hyperplane Sweep","control-paired":"Paired K-Means","em-gibbs":"EM + Gibbs","cutaware-kmeans":"Cut-aware K-Means","em-hard":"EM-hard","em-cem":"EM-CEM","control-tabu":"Paired + Tabu","multigroup-k4":"Multigroup k=4","multigroup-k8":"Multigroup k=8","gw-threshold":"GW threshold","gw-rr":"GW-RR","mst-l2-longest":"MST L2 longest","mst-l2-balanced":"MST L2 balanced","mst-l2-diameter":"MST L2 diameter","mst-l2-bestcut":"MST L2 best-edge","mst-l2-multicut":"MST L2 multicut","mst-l1-bestcut":"MST L1 best-edge","mst-linf-bestcut":"MST Linf best-edge"};
const fams={random:[],planar:[],toroidal:[]},ratioLines=["graph,family,weight_type,method,best_of16_cut,external_reference_cut,ratio_to_reference,reference_basis"];
for(const graph of graphs){const r=refMap.get(graph),group=groups.get(graph);if(!r||!fams[r.family]||group.size!==18)throw Error("Missing reference "+graph);const denominator=+r.reference_cut;let best=-Infinity,winners=[];
for(const method of methods){const datum=group.get(method);const cut=+datum.best_seen_cut;if(!Number.isSafeInteger(cut)||datum.family!==r.family||datum.status!=="complete_16_best_seen"||+datum.refined_starts!==16)throw Error("Incorrect graph-method "+graph+" / "+method);if(cut>denominator)throw Error("Audit candidate above published reference before plotting "+graph);ratioLines.push([graph,r.family,datum.weight_type,method,cut,denominator,(cut/denominator).toFixed(9),r.reference_basis].join(","));if(cut>best){best=cut;winners=[method]}else if(cut===best)winners.push(method)}
const gw=+group.get("gw-rr").best_seen_cut;
fams[r.family].push({graph,best,gw,denominator,bestPct:100*best/denominator,gwPct:100*gw/denominator,winners});
}
if(fams.random.length!==30||fams.planar.length!==24||fams.toroidal.length!==17)throw Error("Graph families changed");
const esc=s=>String(s).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;");
const text=(x,y,s,sz=14,color="#203B54",weight=500,anchor="start")=>`<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" fill="${color}" text-anchor="${anchor}" font-size="${sz}" font-family="Arial,Helvetica,sans-serif" font-weight="${weight}">${esc(s)}</text>`;
const box=(x,y,w,h,c,rx=0)=>`<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${w.toFixed(1)}" height="${Math.max(0,h).toFixed(1)}" rx="${rx}" fill="${c}"/>`;
const ln=(x,y,X,Y,c="#DDE6EE",w=1)=>`<line x1="${x.toFixed(1)}" y1="${y.toFixed(1)}" x2="${X.toFixed(1)}" y2="${Y.toFixed(1)}" stroke="${c}" stroke-width="${w}"/>`;
const files={},stats={};
for(const family of ["random","planar","toroidal"]){const vals=fams[family],W=1400,H=660,b=[],left=86,right=1367,top=191,bottom=522,min=96,max=100,yy=p=>bottom-(p-min)/(max-min)*(bottom-top);
const mean=vals.reduce((a,v)=>a+v.bestPct,0)/vals.length, meanGW=vals.reduce((a,v)=>a+v.gwPct,0)/vals.length;
const credit=new Map();for(const row of vals)for(const method of row.winners)credit.set(method,(credit.get(method)||0)+1/row.winners.length);
const shown=[...credit.keys()].sort((a,b)=>credit.get(b)-credit.get(a)||a.localeCompare(b));
b.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img">`, `<title>${esc(family)} graph best-known reference ratios</title>`,box(0,0,W,H,"#F8FAFD",15),text(29,43,`${family==="planar"?"Planar-family":family==="random"?"Random":"Toroidal"} graphs (${vals.length}) — external reference ratios`,27,"#123154",750),text(30,73,`Best of 18 methods: ${mean.toFixed(3)}% mean  ·  refined GW-RR: ${meanGW.toFixed(3)}% mean`,17,"#53718A"));
b.push(box(30,89,18,14,"#A0AEBB",2),text(56,102,"Refined GW-RR",15),text(222,102,"Colored = highest cut among 18 methods; multicolored = tied winners",15));
shown.forEach((method,i)=>{const col=i%7,row=Math.floor(i/7),x=30+col*196,y=126+25*row;b.push(box(x,y-12,14,13,colors[method],2),text(x+19,y,names[method]||method,13,"#29475D"));});
for(let tick=min;tick<=max+.001;tick+=.5){const y=yy(tick);b.push(ln(left,y,right,y,tick===100?"#67859B":"#DFE7EE",tick===100?2:1),text(73,y+4,`${tick.toFixed(1)}%`,14,tick===100?"#174F6E":"#69849C",600,"end"));}
const cell=(right-left)/vals.length;
vals.forEach((r,i)=>{const x=left+cell*(i+.5),barW=Math.min(11,cell*.27),height1=bottom-yy(r.gwPct),height2=bottom-yy(r.bestPct);b.push(box(x-barW-2,yy(r.gwPct),barW,height1,"#A0AEBB",1));const width=barW/r.winners.length;r.winners.forEach((m,j)=>b.push(box(x+2+j*width,yy(r.bestPct),width+.10,height2,colors[m])));b.push(`<title>${esc(`${r.graph}: best ${r.best}/${r.denominator} = ${r.bestPct.toFixed(3)}%; GW ${r.gw}/${r.denominator} = ${r.gwPct.toFixed(3)}%; best method(s) ${r.winners.join(", ")}`)}</title>`,text(x,545,r.graph,Math.min(15,cell*.43),"#31526A",600,"middle"));});
b.push(text(30,588,"Axis truncated to 96–100% for legibility; bars are best observed cut / listed feasible reference.",15,"#536F84"),text(30,615,`Strongest value in two frozen public tables (Quicopt + OpenIsing, 2026-11-05); not a proved optimum. Exact matches: ${vals.filter(v=>v.bestPct===100).length}.`,14,"#536F84"),text(30,642,"Winner selection is retrospective; equal-width segments represent all tied algorithms.",14,"#536F84"),"</svg>\n");
files[`experiment025_benchmark_ratio_${family}.svg`]=b.join("");
stats[family]={graphs:vals.length,meanBestPct:+mean.toFixed(6),meanGWPct:+meanGW.toFixed(6),exactMatches:vals.filter(x=>x.bestPct===100).length,tiedWins:vals.filter(x=>x.winners.length>1).length};
}
return{files,ratios:ratioLines.join("\n")+"\n",manifest:{files:Object.keys(files),nGraphs:graphs.length,nMethods:methods.length,nCells:full.length,externalReferenceCount:refs.length,quicoptGraded:refs.filter(r=>r.quicopt_best_known!=="").length,stats,noNumericalSolver:true}};
}
const dir=path.join(process.cwd(),"presentations");
const snapshot=fs.readFileSync(path.join(dir,"data/experiment025_gset_reference_snapshot_2026-11-05.csv"),"utf8");
const outcomes=fs.readFileSync(path.join(process.cwd(),"results/gset/experiment025-gw16-bestcut-quality/analysis/full_best_final_by_graph.csv"),"utf8");
const {files,ratios,manifest}=generate(snapshot,outcomes);
const output={...Object.fromEntries(Object.entries(files).map(([k,v])=>[path.join(dir,"assets",k),v])),
[path.join(dir,"data/experiment025_best_of16_method_ratios.csv")]:ratios,
[path.join(dir,"experiment025-reference-ratio-manifest.json")]:JSON.stringify(manifest,null,2)+"\n"};
const checking=process.argv.includes("--check");
for(const [p,s] of Object.entries(output)){if(checking){if(!fs.existsSync(p)||fs.readFileSync(p,"utf8")!==s)throw Error("Stale generated file: "+p)}else{fs.mkdirSync(path.dirname(p),{recursive:true});fs.writeFileSync(p,s,"utf8")}}
console.log(JSON.stringify({...manifest,mode:checking?"check":"generated"},null,2));
