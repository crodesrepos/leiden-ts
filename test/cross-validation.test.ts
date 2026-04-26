/**
 * L2 cross-validation against `graspologic.partition.leiden` — the
 * canonical Python Leiden implementation (Microsoft/JHU; Java backend).
 *
 * Project framing (load-bearing):
 *   leiden-ts is at parity with — or outperforms — graspologic, and our
 *   tests can detect when we have regressed in quality or performance.
 *
 * Load-bearing parity gates in this file:
 *   - Q_ours_median ≥ Q_grasp_median − 0.05 (one-sided; we do NOT fail
 *     when we beat graspologic)
 *   - Connectivity (Theorem 1) on every leiden-ts output community —
 *     a correctness invariant, not a graspologic comparison.
 *   - Anti-sarāb gate (truth-NMI parity, only on fixtures with real
 *     planted/canonical truth): NMI_ours_vs_truth ≥ NMI_grasp_vs_truth − 0.05.
 *     "Sarāb" is defined in docs/adr/0004-benchmark-methodology.md — a
 *     partition whose Q is right but whose structure is empty. The gate
 *     catches Q-equivalent-but-wrong partitions that Q-parity alone misses.
 *
 * NOT gates (diagnostic only — emitted via console.log):
 *   - NMI vs graspologic. Two implementations can land at different
 *     equally-modular partitions under PRNG-driven tie-breaks; that is
 *     not a regression.
 *   - n_communities agreement vs graspologic.
 *
 * Wall-clock parity is asserted in `bench/cross-validation-report.ts`,
 * which is wired into `npm run bench`. Timing inside vitest is unreliable.
 *
 * Reference values are committed in
 * `bench/compare/ref/outputs/<fixture>-graspologic-seed42.json`. Regenerate
 * with `python ref/run_leiden.py --all --seed 42 --runs 30` in the
 * Python venv at `bench/compare/.venv/`.
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

import { Graph } from '../src/graph.js';
import { leiden } from '../src/leiden.js';
import { nmi, ari } from '../src/internal/metrics.js';
import type { Edge } from '../src/types.js';

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

const Q_PARITY_TOLERANCE = 0.05;
const TRUTH_NMI_PARITY_TOLERANCE = 0.05;

const SEEDS = [1, 7, 42, 100, 1000, 12345, 0xc0ffee, 0xbeef, 0xdead, 0xface];

function loadFixture(name: string) {
  const path = resolve(__dirname, `../bench/compare/fixtures/${name}.json`);
  const raw = JSON.parse(readFileSync(path, 'utf8')) as {
    n: number;
    edges: ReadonlyArray<[number, number] | [number, number, number]>;
    ground_truth_assignments?: number[];
    ground_truth_conference_assignments?: number[];
    prior_communities?: number[];
  };
  const truth =
    raw.ground_truth_conference_assignments ??
    raw.ground_truth_assignments ??
    undefined;
  const prior = raw.prior_communities;
  return {
    n: raw.n,
    edges: raw.edges as Edge[],
    truth: truth ? Int32Array.from(truth) : undefined,
    prior: prior ? Int32Array.from(prior) : undefined,
  };
}

function loadGraspologicRef(name: string, seed: number): GraspologicRef {
  const path = resolve(
    __dirname,
    `../bench/compare/ref/outputs/${name}-graspologic-seed${seed}.json`,
  );
  return JSON.parse(readFileSync(path, 'utf8'));
}

/**
 * Median of an array. Single-seed Q has natural variance; the median
 * across SEEDS is the honest measure for parity comparison.
 */
function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)] as number;
}

/** BFS over a community's induced subgraph; returns the count of reachable nodes. */
function communityReachable(graph: Graph, assignments: Int32Array, c: number): number {
  let seedNode = -1;
  for (let u = 0; u < graph.n; u++) {
    if ((assignments[u] as number) === c) {
      seedNode = u;
      break;
    }
  }
  if (seedNode === -1) return 0;
  const visited = new Uint8Array(graph.n);
  const queue: number[] = [seedNode];
  visited[seedNode] = 1;
  let count = 1;
  while (queue.length > 0) {
    const u = queue.shift() as number;
    const start = graph.offsets[u] as number;
    const end = graph.offsets[u + 1] as number;
    for (let i = start; i < end; i++) {
      const v = graph.targets[i] as number;
      if ((assignments[v] as number) !== c) continue;
      if (visited[v] === 1) continue;
      visited[v] = 1;
      queue.push(v);
      count++;
    }
  }
  return count;
}

function assertAllCommunitiesConnected(
  graph: Graph,
  assignments: Int32Array,
  numCommunities: number,
  communitySize: ArrayLike<number>,
): void {
  for (let c = 0; c < numCommunities; c++) {
    const size = communitySize[c] as number;
    if (size <= 1) continue;
    const reached = communityReachable(graph, assignments, c);
    expect(reached).toBe(size);
  }
}

/**
 * Quality scan: collect Q, NMI vs grasp, ARI vs grasp, NMI vs truth (if any),
 * across every seed in SEEDS. Returns medians + raw arrays for diagnostic emit.
 */
function scanQuality(
  graph: Graph,
  refPartition: Int32Array,
  truth?: Int32Array,
) {
  const qs: number[] = [];
  const nmiGrasp: number[] = [];
  const ariGrasp: number[] = [];
  const nmiTruth: number[] = [];
  for (const seed of SEEDS) {
    const r = leiden(graph, { seed });
    qs.push(r.modularity);
    nmiGrasp.push(nmi(r.partition.assignments, refPartition));
    ariGrasp.push(ari(r.partition.assignments, refPartition));
    if (truth) {
      nmiTruth.push(nmi(r.partition.assignments, truth));
    }
  }
  return {
    qMedian: median(qs),
    nmiGraspMedian: median(nmiGrasp),
    nmiGraspMin: Math.min(...nmiGrasp),
    nmiGraspMax: Math.max(...nmiGrasp),
    ariGraspMedian: median(ariGrasp),
    nmiTruthMedian: truth ? median(nmiTruth) : undefined,
  };
}

// -----------------------------------------------------------------
// Karate (Zachary 1977)
// -----------------------------------------------------------------

describe('L2 parity — leiden-ts vs graspologic on Karate', () => {
  const { n, edges } = loadFixture('karate');
  const graph = Graph.fromEdgeList(n, edges);
  const ref = loadGraspologicRef('karate', 42);
  const refPartition = Int32Array.from(ref.partition);

  it('graspologic reference is loadable and self-consistent', () => {
    expect(ref.fixture).toBe('karate');
    expect(refPartition.length).toBe(graph.n);
    expect(ref.all_connected).toBe(true);
    expect(ref.Q_distribution.median).toBeCloseTo(0.4193, 3);
  });

  it('leiden-ts Q ≥ graspologic Q − 0.05 on Karate (one-sided parity)', () => {
    const s = scanQuality(graph, refPartition);
    console.log(
      `[diagnostic] karate: ourQ=${s.qMedian.toFixed(4)} refQ=${ref.Q_distribution.median.toFixed(4)} ΔQ=${(s.qMedian - ref.Q_distribution.median).toFixed(4)} | NMI vs grasp med=${s.nmiGraspMedian.toFixed(4)} (min ${s.nmiGraspMin.toFixed(4)} max ${s.nmiGraspMax.toFixed(4)}) | ARI med=${s.ariGraspMedian.toFixed(4)}`,
    );
    expect(s.qMedian).toBeGreaterThanOrEqual(ref.Q_distribution.median - Q_PARITY_TOLERANCE);
  });

  it('every leiden-ts output community on Karate is internally connected', () => {
    const r = leiden(graph, { seed: 42 });
    assertAllCommunitiesConnected(
      graph,
      r.partition.assignments,
      r.partition.numCommunities,
      r.partition.communitySize,
    );
  });
});

// -----------------------------------------------------------------
// Dolphins (Lusseau 2003)
// -----------------------------------------------------------------
//
// Dolphins has Q-degenerate solutions at Q ≈ 0.519 distinguished only by a
// handful of boundary nodes; PRNG-driven tie-breaks legitimately produce
// different equally-modular partitions. The parity claim is one-sided Q.
// NMI vs graspologic floats around 0.82 and is diagnostic only.
// -----------------------------------------------------------------

describe('L2 parity — leiden-ts vs graspologic on Dolphins', () => {
  const { n, edges } = loadFixture('dolphins');
  const graph = Graph.fromEdgeList(n, edges);
  const ref = loadGraspologicRef('dolphins', 42);
  const refPartition = Int32Array.from(ref.partition);

  it('graspologic reference is loadable', () => {
    expect(ref.fixture).toBe('dolphins');
    expect(refPartition.length).toBe(graph.n);
    expect(ref.all_connected).toBe(true);
  });

  it('leiden-ts Q ≥ graspologic Q − 0.05 on Dolphins (one-sided parity)', () => {
    const s = scanQuality(graph, refPartition);
    console.log(
      `[diagnostic] dolphins: ourQ=${s.qMedian.toFixed(4)} refQ=${ref.Q_distribution.median.toFixed(4)} ΔQ=${(s.qMedian - ref.Q_distribution.median).toFixed(4)} | NMI vs grasp med=${s.nmiGraspMedian.toFixed(4)} (min ${s.nmiGraspMin.toFixed(4)} max ${s.nmiGraspMax.toFixed(4)}) — diverging labels at tied Q is normal`,
    );
    expect(s.qMedian).toBeGreaterThanOrEqual(ref.Q_distribution.median - Q_PARITY_TOLERANCE);
  });

  it('every leiden-ts output community on Dolphins is internally connected', () => {
    const r = leiden(graph, { seed: 42 });
    assertAllCommunitiesConnected(
      graph,
      r.partition.assignments,
      r.partition.numCommunities,
      r.partition.communitySize,
    );
  });
});

// -----------------------------------------------------------------
// Football (Girvan & Newman 2002)
// -----------------------------------------------------------------
//
// Canonical ground truth: each team belongs to one of 12 NCAA conferences,
// and games are disproportionately within-conference. We assert one-sided
// Q parity AND that leiden-ts recovers the conferences at least as well as
// graspologic does (within 0.05 NMI).
// -----------------------------------------------------------------

describe('L2 parity — leiden-ts vs graspologic on Football', () => {
  const { n, edges, truth } = loadFixture('football');
  const graph = Graph.fromEdgeList(n, edges);
  const ref = loadGraspologicRef('football', 42);
  const refPartition = Int32Array.from(ref.partition);

  it('fixture has ground-truth conference labels', () => {
    expect(truth).toBeDefined();
    expect(truth!.length).toBe(graph.n);
  });

  it('leiden-ts Q ≥ graspologic Q − 0.05 on Football (one-sided parity)', () => {
    const s = scanQuality(graph, refPartition, truth);
    console.log(
      `[diagnostic] football: ourQ=${s.qMedian.toFixed(4)} refQ=${ref.Q_distribution.median.toFixed(4)} ΔQ=${(s.qMedian - ref.Q_distribution.median).toFixed(4)} | NMI vs grasp med=${s.nmiGraspMedian.toFixed(4)} | NMI vs conferences med=${s.nmiTruthMedian!.toFixed(4)}`,
    );
    expect(s.qMedian).toBeGreaterThanOrEqual(ref.Q_distribution.median - Q_PARITY_TOLERANCE);
  });

  // Anti-sarāb gate (see docs/adr/0004-benchmark-methodology.md):
  // graspologic's measured NMI vs conferences is ~0.89; we assert
  // leiden-ts is within 0.05 of that. Catches Q-equivalent-but-
  // structurally-wrong partitions on Football specifically.
  it('anti-sarāb: leiden-ts NMI vs conferences ≥ 0.84 (≈ graspologic\'s 0.89 − 0.05)', () => {
    const s = scanQuality(graph, refPartition, truth);
    expect(s.nmiTruthMedian!).toBeGreaterThanOrEqual(0.84);
  });

  it('every leiden-ts output community on Football is internally connected', () => {
    const r = leiden(graph, { seed: 42 });
    assertAllCommunitiesConnected(
      graph,
      r.partition.assignments,
      r.partition.numCommunities,
      r.partition.communitySize,
    );
  });
});

// -----------------------------------------------------------------
// httpx — real-world Python-codebase dependency graph
// -----------------------------------------------------------------
//
// A real-world fixture under the parity framing. graspologic is the
// yardstick (no canonical ground truth). The graph's `prior_communities`
// field — an externally-computed clustering carried in alongside the
// edges — is emitted as a diagnostic comparison only.
// -----------------------------------------------------------------

describe('L2 parity — leiden-ts vs graspologic on httpx (real-world dependency graph)', () => {
  const { n, edges, prior } = loadFixture('httpx');
  const graph = Graph.fromEdgeList(n, edges);
  const ref = loadGraspologicRef('httpx', 42);
  const refPartition = Int32Array.from(ref.partition);

  it('fixture has prior_communities (carried through from node-link)', () => {
    expect(prior).toBeDefined();
    expect(prior!.length).toBe(graph.n);
  });

  it('graspologic reference is loadable and connected', () => {
    expect(ref.fixture).toBe('httpx');
    expect(refPartition.length).toBe(graph.n);
    expect(ref.all_connected).toBe(true);
  });

  it('leiden-ts Q ≥ graspologic Q − 0.05 on httpx (one-sided parity)', () => {
    const s = scanQuality(graph, refPartition);
    const priorNmi = prior
      ? median(SEEDS.map((seed) => nmi(leiden(graph, { seed }).partition.assignments, prior!)))
      : undefined;
    console.log(
      `[diagnostic] httpx: ourQ=${s.qMedian.toFixed(4)} refQ=${ref.Q_distribution.median.toFixed(4)} ΔQ=${(s.qMedian - ref.Q_distribution.median).toFixed(4)} | NMI vs grasp med=${s.nmiGraspMedian.toFixed(4)} | NMI vs prior_communities=${priorNmi !== undefined ? priorNmi.toFixed(4) : 'n/a'}`,
    );
    expect(s.qMedian).toBeGreaterThanOrEqual(ref.Q_distribution.median - Q_PARITY_TOLERANCE);
  });

  it('every leiden-ts output community on httpx is internally connected', () => {
    const r = leiden(graph, { seed: 42 });
    assertAllCommunitiesConnected(
      graph,
      r.partition.assignments,
      r.partition.numCommunities,
      r.partition.communitySize,
    );
  });
});

// -----------------------------------------------------------------
// LFR-1k μ=0.3 (clear community structure)
// -----------------------------------------------------------------
//
// μ=0.3: 30% mixing — communities still clearly detectable. graspologic
// recovers planted truth at NMI ≈ 0.95; we assert one-sided Q parity AND
// truth-recovery parity (within 0.05 of graspologic's truth-NMI).
// -----------------------------------------------------------------

describe('L2 parity — leiden-ts vs graspologic on LFR-1k μ=0.3', () => {
  const { n, edges, truth } = loadFixture('lfr-1k-mu03');
  const graph = Graph.fromEdgeList(n, edges);
  const ref = loadGraspologicRef('lfr-1k-mu03', 42);
  const refPartition = Int32Array.from(ref.partition);

  it('fixture has planted ground-truth labels', () => {
    expect(truth).toBeDefined();
    expect(truth!.length).toBe(graph.n);
  });

  it('leiden-ts Q ≥ graspologic Q − 0.05 on LFR-1k μ=0.3 (one-sided parity)', () => {
    const s = scanQuality(graph, refPartition, truth);
    console.log(
      `[diagnostic] lfr-1k μ=0.3: ourQ=${s.qMedian.toFixed(4)} refQ=${ref.Q_distribution.median.toFixed(4)} ΔQ=${(s.qMedian - ref.Q_distribution.median).toFixed(4)} | NMI vs grasp med=${s.nmiGraspMedian.toFixed(4)} | NMI vs planted med=${s.nmiTruthMedian!.toFixed(4)}`,
    );
    expect(s.qMedian).toBeGreaterThanOrEqual(ref.Q_distribution.median - Q_PARITY_TOLERANCE);
  });

  // Anti-sarāb gate (see docs/adr/0004-benchmark-methodology.md):
  // graspologic's measured NMI vs planted is ~0.95; we assert leiden-ts
  // within 0.05. At μ=0.3 communities are clearly detectable, so a
  // sarāb partition here would be a real algorithmic regression.
  it('anti-sarāb: leiden-ts NMI vs planted ≥ 0.90 (≈ graspologic\'s 0.95 − 0.05)', () => {
    const s = scanQuality(graph, refPartition, truth);
    expect(s.nmiTruthMedian!).toBeGreaterThanOrEqual(0.90);
  });
});

// -----------------------------------------------------------------
// LFR-1k μ=0.5 (at/past detectability boundary)
// -----------------------------------------------------------------
//
// At μ=0.5 modularity-based clustering cannot reliably recover the planted
// partition — graspologic's own truth NMI is ~0.22 here. Every partition
// is sarāb at this μ (see ADR-0004); the anti-sarāb gate is therefore
// not asserted on this fixture. The only defensible parity claim is
// Q-equivalence. NMI vs graspologic also drifts (~0.18); diagnostic only.
// -----------------------------------------------------------------

describe('L2 parity — leiden-ts vs graspologic on LFR-1k μ=0.5', () => {
  const { n, edges } = loadFixture('lfr-1k-mu05');
  const graph = Graph.fromEdgeList(n, edges);
  const ref = loadGraspologicRef('lfr-1k-mu05', 42);
  const refPartition = Int32Array.from(ref.partition);

  it('graspologic reference is loadable', () => {
    expect(ref.fixture).toBe('lfr-1k-mu05');
    expect(ref.all_connected).toBe(true);
  });

  it('leiden-ts Q ≥ graspologic Q − 0.05 on LFR-1k μ=0.5 (one-sided parity)', () => {
    const s = scanQuality(graph, refPartition);
    console.log(
      `[diagnostic] lfr-1k μ=0.5 (past detectability): ourQ=${s.qMedian.toFixed(4)} refQ=${ref.Q_distribution.median.toFixed(4)} ΔQ=${(s.qMedian - ref.Q_distribution.median).toFixed(4)} | NMI vs grasp med=${s.nmiGraspMedian.toFixed(4)} (graspologic's own truth-NMI is also ~0.22)`,
    );
    expect(s.qMedian).toBeGreaterThanOrEqual(ref.Q_distribution.median - Q_PARITY_TOLERANCE);
  });

  it('every leiden-ts output community on LFR-1k μ=0.5 is internally connected', () => {
    const r = leiden(graph, { seed: 42 });
    assertAllCommunitiesConnected(
      graph,
      r.partition.assignments,
      r.partition.numCommunities,
      r.partition.communitySize,
    );
  });
});

// -----------------------------------------------------------------
// Note on TRUTH_NMI_PARITY_TOLERANCE
// -----------------------------------------------------------------
// The football "≥ 0.84" and lfr-1k μ=0.3 "≥ 0.90" thresholds derive from
// graspologic's published truth-NMI on those graphs (0.89 and 0.95)
// minus TRUTH_NMI_PARITY_TOLERANCE. They are one-sided: failing means
// "we recover the canonical answer materially worse than graspologic on
// the same graph." Updating the threshold requires updating the
// graspologic ref too — the gate is grounded in measured graspologic
// behavior, not an arbitrary number.
void TRUTH_NMI_PARITY_TOLERANCE; // re-exported via the inline comment above
