/**
 * Refinement benchmark — Cat-4 Budget per docs/test-philosophy.md.
 * Measures `refine(graph, parent)` after `localMove(graph)` produces parent.
 * Asserts against `bench/baselines.json`; exits non-zero on regression.
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

import { Graph } from '../src/graph.js';
import { localMove } from '../src/localMove.js';
import { refine } from '../src/refine.js';
import type { Edge } from '../src/types.js';
import { bench, formatResult, mulberry32 } from './helpers.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

interface Baselines {
  _meta: { ratio_p99_p50_max: number };
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

function main(): number {
  const cases: ReadonlyArray<{ n: number; m: number }> = [
    { n: 1_000, m: 5_000 },
    { n: 5_000, m: 25_000 },
    { n: 10_000, m: 50_000 },
  ];

  console.log(
    `\n# leiden-ts refine benchmark — ${new Date().toISOString()}\n`,
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

  for (const { n, m } of cases) {
    const graph = buildErdosRenyi(n, m, 0xc0ffee);
    // Compute parent once — refinement is what we benchmark.
    const lm = localMove(graph, undefined, { seed: 1, maxIterations: 1000 });
    const label = `refine n=${n} m=${m}`;

    const result = bench(
      label,
      () => {
        refine(graph, lm.partition, { seed: 1 });
      },
      { warmup: 2, runs: 10 },
    );
    console.log(formatResult(result));

    const budget = baselines.budgets[label];
    if (budget !== undefined) {
      if (result.p50_ms > budget.p50_ms_max) {
        failures.push(
          `❌ BUDGET p50: ${label}: p50=${result.p50_ms.toFixed(3)}ms > budget ${budget.p50_ms_max}ms`,
        );
      }
      const ratio = result.p99_ms / Math.max(result.p50_ms, 1e-9);
      if (ratio > ratioMax) {
        failures.push(
          `❌ BUDGET p99/p50: ${label}: ratio=${ratio.toFixed(2)} > ${ratioMax}`,
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
    return 1;
  }
  console.log('# All performance budgets passed.');
  return 0;
}

const exitCode = main();
if (exitCode !== 0) process.exit(exitCode);
