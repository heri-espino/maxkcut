#!/usr/bin/env node
/** Embed all source SVG figures in the complete Experiment001–027 Marp HTML.
 * Standalone offline HTML remains readable without checking out the repository.
 */
import fs from "node:fs";
import path from "node:path";

const root=path.join(process.cwd(),"presentations");
const stem="maxkcut_full_research_001_027";
const deck=fs.readFileSync(path.join(root,stem+".marp.md"),"utf8");
const output=path.join(root,"build",stem+".html");
const refs=[...new Set([...deck.matchAll(/!\[[^\]]*\]\((assets\/[A-Za-z0-9_.-]+\.svg)\)/g)].map(x=>x[1]))];
if(refs.length<42)throw Error("Too few source figures in Experiment027 deck: "+refs.length);
if(refs.filter(x=>x.startsWith("assets/experiment027_")).length!==13)throw Error("Expected 13 unique E027 source figures");
let html=fs.readFileSync(output,"utf8");
let total=0;
for(const reference of refs){
 const contents=fs.readFileSync(path.join(root,reference));
 const uri="data:image/svg+xml;base64,"+contents.toString("base64");
 const count=(html.match(new RegExp(reference.replaceAll(".","\\."),"g"))||[]).length;
 if(count===0)throw Error("Missing image in rendered HTML: "+reference);
 html=html.replaceAll("./"+reference,uri).replaceAll(reference,uri);
 total+=count;
}
if(/assets\/[A-Za-z0-9_.-]+\.svg/.test(html))throw Error("Output HTML still points to a local SVG");
const math=(html.match(/<mjx-container\b/g)||[]).length;
if(math<10)throw Error("MathJax content missing in rendered HTML");
fs.writeFileSync(output,html);
console.log(JSON.stringify({html:output,svgCount:refs.length,embeddedReferences:total,mathExpressions:math,bytes:Buffer.byteLength(html)},null,2));
