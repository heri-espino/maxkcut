#!/usr/bin/env node
/** Validate the updated Marp deck without running any scientific optimizer. */
import fs from "node:fs";
import path from "node:path";

const root = process.cwd();
const deckPath = path.join(root, "presentations/maxkcut_full_research_001_025.marp.md");
const deck = fs.readFileSync(deckPath,"utf8");
const directory = path.dirname(deckPath);
const manifest = JSON.parse(fs.readFileSync(path.join(directory,"experiment025-figure-manifest.json"),"utf8"));
if (!deck.startsWith("---\nmarp: true\n")) throw Error("Marp YAML header is missing");
if (!/^math: mathjax$/m.test(deck)) throw Error("MathJax must be declared");
if (/\\[()\[\]]/.test(deck)) throw Error("Old TeX math delimiters remain");
const displayMarkers = [...deck.matchAll(/^\$\$/gm)].length;
if (displayMarkers !== 18) throw Error("Expected exactly 9 display math environments");
const sections = deck.split(/^---\s*$/m);
const bodies = sections.slice(2);
if(bodies.length !== 97) throw Error("Experiment025 extension is absent: "+bodies.length+" slides");
for(const [index,slide] of bodies.entries()){
  const nonDisplay = slide.replace(/^\$\$/gm, "");
  if ([...nonDisplay.matchAll(/(?<!\\)\$/g)].length % 2) throw Error("Unbalanced inline math on slide " + (index+1));

  const prose=slide.replace(/<!--[\s\S]*?-->/g,"").trim();
  if(!prose)throw Error("Blank slide at index "+(index+1));
}
const references=[...deck.matchAll(/!\[[^\]]*]\(([^)]+)\)/g)].map(x=>x[1]);
const missing=references.filter(ref=>!fs.existsSync(path.resolve(directory,ref)));
if(missing.length)throw Error("Broken slide figure paths: "+missing.join(", "));
const extra=JSON.parse(fs.readFileSync(path.join(directory,"experiment025-reference-ratio-manifest.json"),"utf8"));
const ratioRefs=references.filter(x=>x.startsWith("assets/experiment025_benchmark_ratio_"));
if(ratioRefs.length!==3||new Set(ratioRefs).size!==3||extra.nGraphs!==71||extra.nMethods!==18||extra.quicoptGraded!==52)throw Error("External reference ratio graphs missing");
for(const name of extra.files)if(!ratioRefs.includes("assets/"+name))throw Error("Missing generated external benchmark bar chart: "+name);
const e025=references.filter(x=>x.startsWith("assets/experiment025_")&&!x.startsWith("assets/experiment025_benchmark_ratio_"));
const expected=manifest.files.map(name=>"assets/"+name);
if(e025.length!==expected.length || new Set(e025).size!==expected.length) throw Error("Expected exactly "+expected.length+" unique Experiment025 image references");
for(const ref of expected)if(!e025.includes(ref))throw Error("Missing Experiment025 chart: "+ref);
if(manifest.nCells!==1278 || manifest.nGraphs!==71 || manifest.nMethods!==18)throw Error("Invalid full coverage metadata");
if(manifest.csvSha256!=="8a9d84cfc35f1d424519b862676facc288a6366bc1820e52975a2bf76678de6c")throw Error("Unexpected aggregate source hash");
if(!deck.includes("independent") || !deck.includes("off-VM") || !deck.includes("generator"))throw Error("Scientific caveats missing");
console.log(JSON.stringify({slides:bodies.length,referencedImages:references.length,experiment025Charts:e025.length,externalCharts:ratioRefs.length,manifestComplete:true},null,2));
