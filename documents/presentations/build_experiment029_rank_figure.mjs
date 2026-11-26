#!/usr/bin/env node
/**
 * Regenerate the Experiment029 Marp SVG from the archived, original signed-cut
 * per-graph summary. No solver, graph-factor loading, network or VM access.
 *
 * From repository root:
 *   node documents/presentations/build_experiment029_rank_figure.mjs
 *   node documents/presentations/build_experiment029_rank_figure.mjs --check
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const root = process.cwd();
const csvFile = path.join(root, 'results/gset/experiment029-eva-kmeansnm-100seed/per_graph.csv');
const figureFile = path.join(root, 'documents/presentations/assets/experiment029_rank_effect_ci.svg');
const check = process.argv.slice(2).includes('--check');
const escape = v => String(v).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const csv = fs.readFileSync(csvFile, 'utf8').replaceAll('\r\n', '\n').trim();
const lines = csv.split('\n');
const headers = lines.shift().split(',');
const records = lines.map(line => Object.fromEntries(line.split(',').map((value, i) => [headers[i], value])));
const required = ['graph','n','default_mean','full_mean','mean_delta','ci95_low','ci95_high','default_converged','full_converged'];
for (const field of required) if (!headers.includes(field)) throw Error('Missing Experiment029 field: ' + field);
if (records.length !== 5 || records.map(r => r.graph).join(',') !== 'G1,G6,G11,G14,G18') throw Error('Unexpected five-graph order or count');
for (const r of records) {
  const d = Number(r.default_mean), f = Number(r.full_mean), delta = Number(r.mean_delta);
  const lo = Number(r.ci95_low), hi = Number(r.ci95_high);
  if (+r.n !== 100 || +r.default_converged !== 100 || +r.full_converged !== 100) throw Error('Incomplete seed coverage for ' + r.graph);
  if (![d,f,delta,lo,hi].every(Number.isFinite) || Math.abs(f-d-delta) > 1e-7 || lo > delta || delta > hi) throw Error('Invalid paired delta/interval for ' + r.graph);
}
const hash = crypto.createHash('sha256').update(fs.readFileSync(csvFile)).digest('hex');
const W=1200,H=590,left=342,right=995,top=155,step=75;
const min=-0.65,max=0.50;
const X = v => left + ((v-min)/(max-min))*(right-left);
const fmt = v => (v>0?'+':'')+v.toFixed(3)+'%';
const line = (x1,y1,x2,y2,color='#cbd5e1',sw=1) => `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="${color}" stroke-width="${sw}"/>`;
const text = (x,y,value,size=18,color='#102a52',weight=500,anchor='start') => `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" font-family="Arial, Helvetica, sans-serif" font-size="${size}" fill="${color}" font-weight="${weight}" text-anchor="${anchor}">${escape(value)}</text>`;
const svg = [];
svg.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Experiment029: paired 95 percent bootstrap intervals for full versus default EVA embedding dimension">`);
svg.push('<rect width="1200" height="590" fill="#f8fafc"/>');
svg.push(text(32,43,'Does p = n improve antipodal K-MeansNM?',28,'#102a52',750));
svg.push(text(32,72,'100 EVA initialization seeds per graph and embedding width · differences in % of default-rank mean cut',17,'#53677d'));
svg.push(text(35,122,'Graph',17,'#53677d',650));
svg.push(text(1005,122,'Mean change',17,'#53677d',650));
for(const tick of [-0.5,-0.25,0,0.25,0.5]) {
  const x=X(tick);
  svg.push(line(x,133,x,518,tick===0?'#34536b':'#dbe4ec',tick===0?2.8:1));
  svg.push(text(x,545,(tick>0?'+':'')+tick.toFixed(2)+'%',15,'#53677d',500,'middle'));
}
records.forEach((r,i)=>{
  const y=top+i*step,mean=Number(r.default_mean);
  const mid=100*Number(r.mean_delta)/mean,lo=100*Number(r.ci95_low)/mean,hi=100*Number(r.ci95_high)/mean;
  if(i%2===0) svg.push(`<rect x="26" y="${y-33}" width="1145" height="61" rx="7" fill="#e9f1f7" opacity="0.52"/>`);
  svg.push(text(35,y+6,r.graph,22,'#102a52',750));
  svg.push(line(X(lo),y,X(hi),y,'#16718f',5));
  svg.push(line(X(lo),y-10,X(lo),y+10,'#16718f',2));
  svg.push(line(X(hi),y-10,X(hi),y+10,'#16718f',2));
  svg.push(`<circle cx="${X(mid).toFixed(1)}" cy="${y}" r="8" fill="${mid<0?'#c45745':'#147b9d'}" stroke="#ffffff" stroke-width="2"/>`);
  svg.push(text(1007,y+6,fmt(mid),20,mid<0?'#a74335':'#116b88',700));
});
svg.push(text(31,580,'Whiskers: nominal 95% paired percentile bootstrap CI (5,000 resamples within each fixed graph). Every interval crosses zero.',14,'#53677d'));
svg.push(`<!-- source: results/gset/experiment029-eva-kmeansnm-100seed/per_graph.csv sha256:${hash} -->`);
svg.push('</svg>');
const output=svg.join('\n')+'\n';
if (check) {
  if (!fs.existsSync(figureFile) || fs.readFileSync(figureFile,'utf8')!==output) throw Error('Experiment029 SVG differs from deterministic generator output');
  console.log('PASS: deterministic Experiment029 paired-confidence-interval SVG, source sha256='+hash);
} else {
  fs.mkdirSync(path.dirname(figureFile),{recursive:true});
  fs.writeFileSync(figureFile,output,'utf8');
  console.log('Wrote '+figureFile+'; source sha256='+hash);
}