/**
 * Cross-validation parity gates against graspologic 3.4.4 across the full
 * fixture portfolio. Reads bench/compare/ref/outputs/<fixture>-graspologic-seed42.json
 * for each fixture, runs leiden-ts at the same seed, asserts the parity
 * gates, and emits diagnostics.
 *
 * Project framing (load-bearing):
 *   leiden-ts is at parity with — or outperforms — graspologic, the
 *   canonical Python Leiden implementation, and our tests can detect
 *   when we have regressed in quality or performance.
 *
 * Parity gates (one-sided; we do NOT fail when we BEAT graspologic):
 *   - Q_ours_median ≥ Q_grasp_median − 0.05
 *   - wallclock_ours_p50 ≤ wallclock_grasp_p50 × 1.0
 *
 * Diagnostic emit (NOT gates — printed to the console only):
 *   - NMI vs graspologic (legitimately < 1.0 under PRNG-driven tie-breaks
 *     at tied Q; not a regression signal)
 *   - NMI vs ground truth where available (the anti-sarāb gate lives in
 *     test/cross-validation.test.ts; here it is diagnostic emit only)
 *   - NMI vs the fixture's `prior_communities` baseline (httpx)
 *   - n_communities agreement
 *
 * Run: `npx tsx bench/cross-validation-report.ts`
 *   - exits 0 if every parity gate passes
 *   - exits 1 if any parity gate fails (regression signal)
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

import { Graph } from '../src/graph.js';
import { leiden } from '../src/leiden.js';
import { nmi, ari } from '../src/internal/metrics.js';
import type { Edge } from '../src/types.js';
import { bench } from './helpers.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

interface GraspologicRef {
  fixture: string;
  graspologic_version: string;
  partition: number[];
  n_communities: number;
  Q_distribution: { median: number; p10: number; p90: number };
  time_ms: { median: number; p50: number; p90: number; p99: number };
  all_connected: boolean;
}

interface FixtureRaw {
  n: number;
  edges: ReadonlyArray<[number, number] | [number, number, number]>;
  ground_truth_assignments?: number[];
  ground_truth_conference_assignments?: number[];
  prior_communities?: number[];
}

interface FixtureSpec {
  /** Filename stem under bench/compare/fixtures/ */
  name: string;
  /** Display label */
  label: string;
  /** Bench warmup runs (LFR-10k is too slow for 20 warmups). */
  warmup: number;
  /** Bench measured runs. */
  runs: number;
  /** Which ground-truth field to compare against, if any. */
  truthKey?: 'ground_truth_assignments' | 'ground_truth_conference_assignments';
  /** Pretty name for the truth source (e.g., "conference"). */
  truthLabel?: string;
  /** Treat the fixture's `prior_communities` as a diagnostic baseline. */
  emitPriorNmi?: boolean;
}

const FIXTURES: FixtureSpec[] = [
  { name: 'karate', label: 'karate', warmup: 20, runs: 200 },
  { name: 'dolphins', label: 'dolphins', warmup: 20, runs: 200 },
  {
    name: 'football',
    label: 'football',
    warmup: 20,
    runs: 200,
    truthKey: 'ground_truth_conference_assignments',
    truthLabel: 'conference',
  },
  {
    name: 'httpx',
    label: 'httpx',
    warmup: 20,
    runs: 200,
    emitPriorNmi: true,
  },
  {
    name: 'lfr-1k-mu03',
    label: 'lfr-1k μ=0.3',
    warmup: 5,
    runs: 50,
    truthKey: 'ground_truth_assignments',
    truthLabel: 'planted',
  },
  {
    name: 'lfr-1k-mu05',
    label: 'lfr-1k μ=0.5',
    warmup: 5,
    runs: 50,
    truthKey: 'ground_truth_assignments',
    truthLabel: 'planted',
  },
  {
    name: 'lfr-10k-mu03',
    label: 'lfr-10k μ=0.3',
    warmup: 2,
    runs: 10,
    truthKey: 'ground_truth_assignments',
    truthLabel: 'planted',
  },
];

const SEEDS = [1, 7, 42, 100, 1000, 12345, 0xc0ffee, 0xbeef, 0xdead, 0xface];

const Q_PARITY_TOLERANCE = 0.05;
const WALLCLOCK_PARITY_RATIO = 1.0;

function loadFixture(name: string): FixtureRaw {
  const path = resolve(__dirname, `compare/fixtures/${name}.json`);
  return JSON.parse(readFileSync(path, 'utf8')) as FixtureRaw;
}

function loadRef(name: string): GraspologicRef {
  const path = resolve(__dirname, `compare/ref/outputs/${name}-graspologic-seed42.json`);
  return JSON.parse(readFileSync(path, 'utf8')) as GraspologicRef;
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)] as number;
}
function minOf(xs: number[]): number {
  return [...xs].sort((a, b) => a - b)[0] as number;
}
function maxOf(xs: number[]): number {
  return [...xs].sort((a, b) => a - b)[xs.length - 1] as number;
}

interface FixtureMetrics {
  spec: FixtureSpec;
  n: number;
  m: number;
  ourQ_median: number;
  refQ_median: number;
  dq_signed: number; // ourQ - refQ (positive => we're better)
  nmi_median: number;
  nmi_min: number;
  nmi_max: number;
  ari_median: number;
  truth_nmi_median?: number;
  truth_nmi_min?: number;
  prior_nmi_median?: number;
  ourCommunities: number;
  refCommunities: number;
  ts_p50_ms: number;
  ts_p99_ms: number;
  grasp_p50_ms: number;
  grasp_p99_ms: number;
  speedup_p50: number;
}

function measure(spec: FixtureSpec): FixtureMetrics {
  const fix = loadFixture(spec.name);
  const ref = loadRef(spec.name);
  const graph = Graph.fromEdgeList(fix.n, fix.edges as Edge[]);
  const refPartition = Int32Array.from(ref.partition);
  const truthRaw = spec.truthKey ? fix[spec.truthKey] : undefined;
  const truthPartition = truthRaw ? Int32Array.from(truthRaw) : undefined;
  const priorRaw = spec.emitPriorNmi ? fix.prior_communities : undefined;
  const priorPartition = priorRaw ? Int32Array.from(priorRaw) : undefined;

  const nmiScores: number[] = [];
  const ariScores: number[] = [];
  const qValues: number[] = [];
  const truthNmiScores: number[] = [];
  const priorNmiScores: number[] = [];
  let lastResultCommunities = 0;

  for (const seed of SEEDS) {
    const result = leiden(graph, { seed });
    const a = result.partition.assignments;
    nmiScores.push(nmi(a, refPartition));
    ariScores.push(ari(a, refPartition));
    qValues.push(result.modularity);
    if (truthPartition) {
      truthNmiScores.push(nmi(a, truthPartition));
    }
    if (priorPartition) {
      priorNmiScores.push(nmi(a, priorPartition));
    }
    let nonEmpty = 0;
    for (let c = 0; c < result.partition.numCommunities; c++) {
      if ((result.partition.communitySize[c] as number) > 0) nonEmpty++;
    }
    lastResultCommunities = nonEmpty;
  }

  const wc = bench(
    `leiden-ts ${spec.label}`,
    () => {
      leiden(graph, { seed: 42 });
    },
    { warmup: spec.warmup, runs: spec.runs },
  );

  const ourQ = median(qValues);
  const refQ = ref.Q_distribution.median;

  return {
    spec,
    n: graph.n,
    m: graph.m,
    ourQ_median: ourQ,
    refQ_median: refQ,
    dq_signed: ourQ - refQ,
    nmi_median: median(nmiScores),
    nmi_min: minOf(nmiScores),
    nmi_max: maxOf(nmiScores),
    ari_median: median(ariScores),
    truth_nmi_median: truthPartition ? median(truthNmiScores) : undefined,
    truth_nmi_min: truthPartition ? minOf(truthNmiScores) : undefined,
    prior_nmi_median: priorPartition ? median(priorNmiScores) : undefined,
    ourCommunities: lastResultCommunities,
    refCommunities: ref.n_communities,
    ts_p50_ms: wc.p50_ms,
    ts_p99_ms: wc.p99_ms,
    grasp_p50_ms: ref.time_ms.median,
    grasp_p99_ms: ref.time_ms.p99,
    speedup_p50: ref.time_ms.median / wc.p50_ms,
  };
}

function fmt(n: number, w: number, d = 4): string {
  return n.toFixed(d).padStart(w);
}

function fmtSigned(n: number, w: number, d = 4): string {
  const sign = n >= 0 ? '+' : '';
  return (sign + n.toFixed(d)).padStart(w);
}

function bar(): void {
  console.log('─'.repeat(110));
}

console.log('\n# Cross-validation parity report — leiden-ts vs graspologic 3.4.4 (full portfolio)\n');
console.log(`Seeds scanned: ${SEEDS.length} for quality; warmup+runs per fixture vary by graph size.\n`);

const all: FixtureMetrics[] = [];
for (const spec of FIXTURES) {
  process.stderr.write(`measuring ${spec.label}…\n`);
  all.push(measure(spec));
}

console.log('## Quality table (Q ours vs graspologic; signed ΔQ; diagnostic NMI / ARI)\n');
bar();
console.log(
  'fixture          n      m       ourQ    refQ    ΔQ        NMI(med) ARI(med)  comm(ours/ref)',
);
bar();
for (const m of all) {
  console.log(
    `${m.spec.label.padEnd(15)} ${String(m.n).padStart(5)}  ${String(m.m).padStart(6)}  ${fmt(m.ourQ_median, 6)}  ${fmt(m.refQ_median, 6)}  ${fmtSigned(m.dq_signed, 8)}  ${fmt(m.nmi_median, 7)}  ${fmt(m.ari_median, 7)}  ${String(m.ourCommunities).padStart(3)} / ${String(m.refCommunities).padStart(3)}`,
  );
}
bar();

console.log('\n## Wall-clock table (p50, p99, speedup vs graspologic at seed=42)\n');
bar();
console.log(
  'fixture          n      m       ts p50    ts p99    grasp p50  grasp p99  speedup(p50)',
);
bar();
for (const m of all) {
  const arrow = m.speedup_p50 > 1 ? '×↑' : '×↓';
  console.log(
    `${m.spec.label.padEnd(15)} ${String(m.n).padStart(5)}  ${String(m.m).padStart(6)}  ${m.ts_p50_ms.toFixed(3).padStart(8)}ms ${m.ts_p99_ms.toFixed(3).padStart(8)}ms  ${m.grasp_p50_ms.toFixed(3).padStart(8)}ms ${m.grasp_p99_ms.toFixed(3).padStart(8)}ms  ${m.speedup_p50.toFixed(2).padStart(5)}${arrow}`,
  );
}
bar();

console.log('\n## Parity gates (one-sided, load-bearing)\n');
console.log(
  `Quality gate:   Q_ours_median ≥ Q_grasp_median − ${Q_PARITY_TOLERANCE.toFixed(2)}\n` +
    `Wall-clock gate: wallclock_ours_p50 ≤ wallclock_grasp_p50 × ${WALLCLOCK_PARITY_RATIO.toFixed(2)} (parity-or-better)\n`,
);
bar();
console.log(
  'fixture          ΔQ        Q-gate  wc ratio (ours/grasp)  wc-gate',
);
bar();
const failures: string[] = [];
for (const m of all) {
  const qPass = m.dq_signed >= -Q_PARITY_TOLERANCE;
  const wcRatio = m.ts_p50_ms / m.grasp_p50_ms;
  const wcPass = wcRatio <= WALLCLOCK_PARITY_RATIO;
  if (!qPass) {
    failures.push(
      `Q parity: ${m.spec.label}: ourQ=${m.ourQ_median.toFixed(4)} < refQ=${m.refQ_median.toFixed(4)} − ${Q_PARITY_TOLERANCE} (ΔQ=${m.dq_signed.toFixed(4)})`,
    );
  }
  if (!wcPass) {
    failures.push(
      `Wall-clock parity: ${m.spec.label}: ours_p50=${m.ts_p50_ms.toFixed(3)}ms > grasp_p50=${m.grasp_p50_ms.toFixed(3)}ms × ${WALLCLOCK_PARITY_RATIO} (ratio=${wcRatio.toFixed(2)})`,
    );
  }
  console.log(
    `${m.spec.label.padEnd(15)}  ${fmtSigned(m.dq_signed, 8)}   ${qPass ? 'PASS' : 'FAIL'}   ${wcRatio.toFixed(3).padStart(7)} (1.0 = parity)   ${wcPass ? 'PASS' : 'FAIL'}`,
  );
}
bar();

console.log('\n## Diagnostics (NOT gates — informational)\n');
bar();
console.log('fixture          NMI vs grasp (med, min/max)  NMI vs truth (med)  NMI vs prior (med)');
bar();
for (const m of all) {
  const truthCol =
    m.truth_nmi_median !== undefined ? fmt(m.truth_nmi_median, 7) : '   n/a ';
  const priorCol =
    m.prior_nmi_median !== undefined ? fmt(m.prior_nmi_median, 7) : '   n/a ';
  console.log(
    `${m.spec.label.padEnd(15)}  ${fmt(m.nmi_median, 7)} (${fmt(m.nmi_min, 5)} / ${fmt(m.nmi_max, 5)})        ${truthCol}             ${priorCol}`,
  );
}
bar();
console.log(
  '\nReminder: NMI vs graspologic legitimately drops below 1.0 under PRNG-driven\n' +
    'tie-breaks at tied Q. Diverging labels at equivalent Q is not a regression.\n',
);

console.log('\n## Performance regime\n');
console.log(
  'speedup > 1 means leiden-ts is faster (lower p50). graspologic numbers include\n' +
    'a JVM-warmed cold start (one un-timed call); leiden-ts numbers discard a\n' +
    'configurable warmup window. Different methodologies — speedup is approximate.\n',
);
for (const m of all) {
  const regime =
    m.speedup_p50 >= 2
      ? 'leiden-ts strongly faster'
      : m.speedup_p50 >= 1
        ? 'leiden-ts faster'
        : m.speedup_p50 >= 0.5
          ? 'parity'
          : 'graspologic faster';
  console.log(`  ${m.spec.label.padEnd(15)} (n=${m.n}, m=${m.m}): ${regime} (${m.speedup_p50.toFixed(2)}×)`);
}

console.log('');
if (failures.length > 0) {
  console.error('# Parity regression detected:');
  for (const f of failures) console.error(`  ❌ ${f}`);
  process.exit(1);
}
console.log('# All parity gates passed.\n');
