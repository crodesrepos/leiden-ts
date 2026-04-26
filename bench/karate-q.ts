/**
 * Quick measurement: actual Karate Q across many seeds for the full
 * Leiden pipeline. Diagnostic only — no budget gate.
 */
import { Graph } from '../src/graph.js';
import { leiden } from '../src/leiden.js';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const k = JSON.parse(
  readFileSync(resolve(__dirname, '../bench/compare/fixtures/karate.json'), 'utf8'),
) as { n: number; edges: ReadonlyArray<[number, number]> };
const g = Graph.fromEdgeList(k.n, k.edges as Array<[number, number]>);

const seeds = [1, 7, 42, 100, 1000, 12345, 0xc0ffee, 0xbeef, 0xdead, 0xface];
console.log('Karate Q (full Leiden) across seeds:');
const results: number[] = [];
for (const seed of seeds) {
  const r = leiden(g, { seed });
  results.push(r.modularity);
  console.log(
    `  seed=${seed.toString().padStart(10)}: Q=${r.modularity.toFixed(6)}  levels=${r.levels}  moves=${r.totalMoves}  merges=${r.totalMerges}`,
  );
}
const sorted = [...results].sort();
const median = sorted[Math.floor(sorted.length / 2)] ?? 0;
const max = Math.max(...results);
const min = Math.min(...results);
console.log(`\nQ stats:  min=${min.toFixed(6)}  median=${median.toFixed(6)}  max=${max.toFixed(6)}`);
console.log(`paper canonical (Traag 2019 Fig.2): ~0.444`);
console.log(`networkx Louvain (seed=42, full):    0.385`);
