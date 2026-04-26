/**
 * Modularity benchmark: measures `modularity(G, P)` on Erdős–Rényi
 * graphs of growing size. Asserts results against `bench/baselines.json`
 * — failure exits non-zero (see Cat-4 in `docs/test-philosophy.md`).
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

import { Graph } from '../src/graph.js';
import { modularity } from '../src/modularity.js';
import type { Edge } from '../src/types.js';
import { bench, formatResult, mulberry32 } from './helpers.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

interface Baselines {
  _meta: { ratio_p99_p50_max: number; regression_factor_max: number };
  budgets: Record<string, { p50_ms_max: number; label: string }>;
}

const baselines = JSON.parse(
  readFileSync(resolve(__dirname, 'baselines.json'), 'utf8'),
) as Baselines;

function buildErdosRenyi(n: number, m: number, seed: number): Graph {
  const rand = mulberry32(seed);
  const edges: Edge[] = [];
  const seen = new Set<number>();
  while (edges.length < m) {
    const u = Math.floor(rand() * n);
    const v = Math.floor(rand() * n);
    if (u === v) continue;
    const lo = u < v ? u : v;
    const hi = u < v ? v : u;
    const key = lo * n + hi;
    if (seen.has(key)) continue;
    seen.add(key);
    edges.push([lo, hi]);
  }
  return Graph.fromEdgeList(n, edges);
}

function buildPartition(n: number, k: number, seed: number): Int32Array {
  const rand = mulberry32(seed);
  const partition = new Int32Array(n);
  for (let i = 0; i < n; i++) partition[i] = Math.floor(rand() * k);
  return partition;
}

function main(): number {
  const cases: ReadonlyArray<{ n: number; m: number; k: number }> = [
    { n: 1_000, m: 5_000, k: 10 },
    { n: 5_000, m: 25_000, k: 25 },
    { n: 10_000, m: 50_000, k: 50 },
    { n: 50_000, m: 250_000, k: 100 },
  ];

  console.log(
    `\n# leiden-ts modularity benchmark — ${new Date().toISOString()}\n`,
  );
  console.log(
    `# Node ${process.version} · platform ${process.platform} · arch ${process.arch}`,
  );
  console.log(
    '#'.padEnd(40, ' '),
    'runs   p50         p90         p99         min         max',
  );

  const failures: string[] = [];
  const ratioMax = baselines._meta.ratio_p99_p50_max;

  for (const { n, m, k } of cases) {
    const graph = buildErdosRenyi(n, m, 0xc0ffee);
    const partition = buildPartition(n, k, 0xbeef);
    const label = `modularity n=${n} m=${m} k=${k}`;

    const result = bench(
      label,
      () => {
        modularity(graph, partition);
      },
      { warmup: 20, runs: 100 },
    );
    console.log(formatResult(result));

    const budget = baselines.budgets[label];
    if (budget !== undefined) {
      // Cat-4 budget gate: p50 must stay under the committed ceiling.
      if (result.p50_ms > budget.p50_ms_max) {
        failures.push(
          `❌ BUDGET p50: ${label}: p50=${result.p50_ms.toFixed(3)}ms > budget ${budget.p50_ms_max}ms`,
        );
      }
      // GC-pause gate: p99/p50 ratio captures latency spikes that mean
      // ratios — passing this even on slow hardware.
      const ratio = result.p99_ms / Math.max(result.p50_ms, 1e-9);
      if (ratio > ratioMax) {
        failures.push(
          `❌ BUDGET p99/p50: ${label}: ratio=${ratio.toFixed(2)} > ${ratioMax} (suggests GC pauses or other variance)`,
        );
      }
    } else {
      console.log(
        `# (no baseline budget for "${label}" — add one to bench/baselines.json)`,
      );
    }
  }

  console.log('');
  if (failures.length > 0) {
    console.error('# Performance regression detected:');
    for (const f of failures) console.error(`  ${f}`);
    console.error(
      '# To intentionally raise budgets, edit bench/baselines.json and explain in commit.',
    );
    return 1;
  }
  console.log('# All performance budgets passed.');
  return 0;
}

const exitCode = main();
if (exitCode !== 0) process.exit(exitCode);
