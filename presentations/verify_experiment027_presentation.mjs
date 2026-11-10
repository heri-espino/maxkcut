#!/usr/bin/env node
/** Structural, provenance and figure coverage checks for the 001–027 deck. */
import fs from "node:fs";
import path from "node:path";
import {createHash} from "node:crypto";

const root=process.cwd();
const deckFile=path.join(root,"presentations/maxkcut_full_research_001_027.marp.md");
const oldFile=path.join(root,"presentations/maxkcut_full_research_001_025.marp.md");
const text=fs.readFileSync(deckFile,"utf8");
const older=fs.readFileSync(oldFile,"utf8");
const manifest=JSON.parse(fs.readFileSync(path.join(root,"presentations/experiment027-figure-manifest.json"),"utf8"));
const historical=JSON.parse(fs.readFileSync(path.join(root,"presentations/experiment025-figure-manifest.json"),"utf8"));
const tag=manifest.release_tag;
const index=JSON.parse(fs.readFileSync(path.join(root,"published/experiment027/releases",tag,"index.json"),"utf8"));
const hash=s=>createHash("sha256").update(s,"utf8").digest("hex");
if(!text.startsWith("---\nmarp: true\n"))throw Error("Marp frontmatter missing");
if(!/^math: mathjax$/m.test(text))throw Error("MathJax should match existing deck");
if(!text.includes("Experiments 001–027"))throw Error("Incorrect edition");
if([...text.matchAll(/^\$\$/gm)].length!==18)throw Error("Unexpected historical display-math environments");
const slides=text.split(/^---\s*$/m).slice(2);
if(slides.length!==137)throw Error("Expected 97 historical + 20 Experiment027 + 20 method-appendix slides, got "+slides.length);
for(const [i,slide] of slides.entries()){
 const prose=slide.replace(/<!--[\s\S]*?-->/g,"").trim();
 if(!prose)throw Error("Blank slide at "+(i+1));
 if([...slide.replace(/^\$\$/gm,"").matchAll(/(?<!\\)\$/g)].length%2)throw Error("Unbalanced inline math on slide "+(i+1));
}
// Keep the appendix complete if source method catalogs change.
const appendixAt=text.indexOf("# Appendix A — Method-name decoder");
if(appendixAt<0)throw Error("Missing explanatory appendix");
const appendix=text.slice(appendixAt);
if(slides.slice(-20).filter(s=>s.includes("## A")||s.includes("# Appendix A")).length!==20)
  throw Error("Appendix section is not the final twenty slides");
for(let number=1;number<=19;number++){
 if(!appendix.includes("## A"+number+"."))throw Error("Missing appendix topic A"+number);
}
function tupleNames(filename,identifier){
 const source=fs.readFileSync(path.join(root,filename),"utf8");
 const pattern=new RegExp("(?:^|\\n)"+identifier+"\\s*=\\s*\\(([\\s\\S]*?)\\)");
 const match=source.match(pattern);
 if(!match)throw Error("Missing method catalog: "+filename+" "+identifier);
 return [...match[1].matchAll(/"([^"]+)"/g)].map(x=>x[1]);
}
const catalog025=tupleNames("src/maxkcut/experiment023_methods.py","METHODS");
const catalog021=tupleNames("src/maxkcut/eva_kmeans_initializers.py","INIT2");
const catalog027=tupleNames("scripts/experiment027_angular.py","METHODS");
if(catalog025.length!==18||catalog021.length!==9||catalog027.length!==3)
 throw Error("Unexpected method catalog coverage");
const tick=String.fromCharCode(96);
for(const name of [...catalog025,...catalog021,...catalog027,
 "FJ_EVA","K-Means2N_EVA","K-MeansNM_EVA","FJ-Flip_EVA","K-MeansNM-Flip_EVA"]){
 if(!appendix.includes(tick+name+tick))throw Error("Algorithm missing from appendix: "+name);
}
for(const cue of ["GW-anchored hybrids","not a different EVA","No Experiment028 data",
 "same original GW","graph","Kernighan"]){
 if(!appendix.toLowerCase().includes(cue.toLowerCase()))
  throw Error("Appendix explanatory boundary missing: "+cue);
}

const refs=[...text.matchAll(/!\[[^\]]*\]\(([^)]+)\)/g)].map(m=>m[1]);
for(const ref of refs)if(!fs.existsSync(path.resolve(path.dirname(deckFile),ref)))throw Error("Missing source figure: "+ref);
const newer=refs.filter(x=>x.startsWith("assets/experiment027_"));
const expected=manifest.files.map(file=>"assets/"+file);
if(newer.length!==13||new Set(newer).size!==13)throw Error("Missing/duplicate Experiment027 figure");
for(const e of expected)if(!newer.includes(e))throw Error("New figure not embedded: "+e);
const historicRefs=[...older.matchAll(/!\[[^\]]*\]\(([^)]+)\)/g)].map(m=>m[1]);
for(const r of historicRefs)if(!refs.includes(r))throw Error("Historical figure disappeared: "+r);
if(!historicRefs.includes("assets/experiment025_method_ranking.svg"))throw Error("Historical results missing");
if(manifest.graphs!==71||manifest.new_seed_archives!==3408||manifest.paired_rows!==6816)throw Error("Invalid Experiment027 chart provenance");
if(historical.nGraphs!==71||historical.nCells!==1278)throw Error("Invalid historical chart provenance");
if(index.assets.length!==71||index.complete_new_seeds!==3408)throw Error("Missing published Experiment027 archives");
const files=[
 ["graph","audited_graph_comparisons.csv"],
 ["leaderboard","audited_method_leaderboard.csv"],
 ["paired","audited_paired_seed_deltas.csv"],
 ["seed","audited_seed_scores.csv"]];
for(const [key,filename] of files){
 const csv=fs.readFileSync(path.join(root,"published/experiment027/releases",tag,"analysis",filename),"utf8");
 if(hash(csv)!==index.analysis_checksums[filename] && hash(csv.replace(/\r?\n/g,"\r\n"))!==index.analysis_checksums[filename])throw Error("Source SHA mismatch beyond CRLF normalization: "+filename);
 if(manifest.source_sha256[key]!==index.analysis_checksums[filename])throw Error("Manifest SHA mismatch: "+filename);
}
for(const cue of ["best-of-16","not a fair","graph-bootstrap","16 matched","off-VM","Experiment027"])if(!text.includes(cue))throw Error("Scientific caveat absent: "+cue);
console.log(JSON.stringify({slides:slides.length,historicalPreserved:97,experiment027Slides:20,methodAppendixSlides:20,newFigures:newer.length,historicalFigures:historicRefs.length,release:tag,archives:index.assets.length},null,2));
