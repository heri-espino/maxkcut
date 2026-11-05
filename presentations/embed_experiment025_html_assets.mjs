#!/usr/bin/env node
/** Embed all referenced SVGs in the Experiment025 HTML, making it portable. */
import fs from "node:fs";
import path from "node:path";

const dir = path.join(process.cwd(), "presentations");
const deck = fs.readFileSync(path.join(dir, "maxkcut_full_research_001_025.marp.md"), "utf8");
const htmlPath = path.join(dir, "build/maxkcut_full_research_001_025.html");
const refs = [...new Set([...deck.matchAll(/!\[[^\]]*\]\((assets\/[A-Za-z0-9_.-]+\.svg)\)/g)].map(m => m[1]))];
if (refs.length < 29) throw Error("Missing expected source SVGs: " + refs.length);
let html = fs.readFileSync(htmlPath, "utf8");
let embedded = 0;
for (const ref of refs) {
  const uri = "data:image/svg+xml;base64," + fs.readFileSync(path.join(dir, ref)).toString("base64");
  const count = html.split(ref).length - 1;
  if (!count) throw Error("Expected SVG not present in exported HTML: " + ref);
  html = html.replaceAll("./" + ref, uri).replaceAll(ref, uri);
  embedded += count;
}
if (/assets\/[A-Za-z0-9_.-]+\.svg/.test(html)) throw Error("HTML retains an external SVG reference");
const equations = (html.match(/<mjx-container\b/g) || []).length;
if (equations < 10) throw Error("Rendered MathJax equations missing from HTML: " + equations);
fs.writeFileSync(htmlPath, html);
console.log(JSON.stringify({embeddedSvgReferences: embedded, uniqueSvgFigures: refs.length, renderedMathExpressions: equations, htmlBytes: Buffer.byteLength(html)}, null, 2));
