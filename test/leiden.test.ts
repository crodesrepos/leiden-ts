/**
 * End-to-end tests for the public leiden() function — the multi-level loop.
 *
 * Categories:
 *   Cat-1 Identity      — Karate Q ≈ 0.444 (paper Fig. 2 canonical value);
 *                         two-triangles → Q=0.5; K_n → Q=0
 *   Cat-2 Reference     — Q on Karate matches or exceeds networkx Louvain
 *   Cat-3 Property      — connectivity guarantee on full pipeline output;
 *                         determinism by seed; warm-start invariance
 *   Cat-5 Contract      — convergence error; option validation
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

import { Graph } from '../src/graph.js';
import { Partition } from '../src/partition.js';
import { leiden, LeidenConvergenceError } from '../src/leiden.js';
import { modularity } from '../src/modularity.js';
import type { Edge } from '../src/types.js';
import { clique, twoTriangles } from './helpers.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

function loadKarate() {
  const path = resolve(__dirname, '../bench/compare/fixtures/karate.json');
  const raw = JSON.parse(readFileSync(path, 'utf8')) as {
    n: number;
    edges: ReadonlyArray<[number, number]>;
  };
  return { n: raw.n, edges: raw.edges as Edge[] };
}

function loadLouvainRef() {
  const path = resolve(
    __dirname,
    '../bench/compare/ref/outputs/karate-louvain-seed42.json',
  );
  return JSON.parse(readFileSync(path, 'utf8')) as {
    modularity: number;
    n_communities: number;
  };
}

function findDisconnectedCommunity(
  graph: Graph,
  partition: Partition,
): number | null {
  for (let c = 0; c < partition.numCommunities; c++) {
    const size = partition.communitySize[c] as number;
    if (size <= 1) continue;
    let seed = -1;
    for (let u = 0; u < graph.n; u++) {
      if ((partition.assignments[u] as number) === c) {
        seed = u;
        break;
      }
    }
    if (seed === -1) continue;

    const visited = new Uint8Array(graph.n);
    const queue: number[] = [seed];
    visited[seed] = 1;
    let count = 1;
    while (queue.length > 0) {
      const u = queue.shift() as number;
      const start = graph.offsets[u] as number;
      const end = graph.offsets[u + 1] as number;
      for (let i = start; i < end; i++) {
        const v = graph.targets[i] as number;
        if ((partition.assignments[v] as number) !== c) continue;
        if (visited[v] === 1) continue;
        visited[v] = 1;
        queue.push(v);
        count++;
      }
    }
    if (count !== size) return c;
  }
  return null;
}

describe('leiden — Cat-1 Identity', () => {
  it('two-triangles → Q = 0.5 (the perfect 2-partition)', () => {
    const g = twoTriangles();
    const result = leiden(g);
    expect(result.modularity).toBeCloseTo(0.5, 10);
  });

  it('K_n → Q = 0 (no community structure; all-one-community is optimal)', () => {
    for (const n of [3, 5, 8]) {
      const g = clique(n);
      const result = leiden(g);
      expect(result.modularity).toBeCloseTo(0, 10);
    }
  });

  it('empty graph → Q = 0', () => {
    const g = Graph.fromEdgeList(5, []);
    const result = leiden(g);
    expect(result.modularity).toBe(0);
  });

  it('two cliques bridged by one edge → 2 communities, Q close to optimal', () => {
    // K3 + K3 + 1 bridge. Optimal Q is ~0.357 (computable: each triangle is
    // a community, the bridge is a cross-community edge).
    const g = Graph.fromEdgeList(6, [
      [0, 1], [0, 2], [1, 2], // triangle A
      [3, 4], [3, 5], [4, 5], // triangle B
      [2, 3],                 // bridge
    ]);
    const result = leiden(g);
    expect(result.modularity).toBeGreaterThan(0.30);
    // 2 non-empty communities expected.
    let nonEmpty = 0;
    for (let c = 0; c < result.partition.numCommunities; c++) {
      if ((result.partition.communitySize[c] as number) > 0) nonEmpty++;
    }
    expect(nonEmpty).toBe(2);
  });
});

describe('leiden — Cat-1 Identity on the Karate club', () => {
  // The headline test: full Leiden on Karate should reach Q ≈ 0.444 (paper
  // Fig. 2 canonical value via Leiden with multiple restarts). Single-run
  // Leiden typically reaches 0.42-0.45 depending on seed; we assert ≥ 0.40.
  // This is the primary M4 quality gate.

  const { n, edges } = loadKarate();
  const graph = Graph.fromEdgeList(n, edges);

  it('reaches Q ≥ 0.40 on Karate (paper-canonical ~0.444 ± seed variance)', () => {
    for (const seed of [1, 7, 42, 1000, 0xc0ffee]) {
      const result = leiden(graph, { seed });
      expect(result.modularity).toBeGreaterThanOrEqual(0.40);
      expect(result.modularity).toBeLessThan(0.50);
    }
  });

  it('matches or exceeds networkx Louvain Q on Karate (Q ≥ 0.385)', () => {
    // networkx Louvain reference: Q ≈ 0.385. Full Leiden should match or
    // exceed because Leiden has a tighter optimum (refinement helps escape
    // local minima Louvain commits to).
    const louvain = loadLouvainRef();
    const result = leiden(graph, { seed: 42 });
    expect(result.modularity).toBeGreaterThanOrEqual(louvain.modularity - 1e-3);
  });

  it('cached and scratch modularity agree on the final partition', () => {
    const result = leiden(graph, { seed: 42 });
    const cached = result.modularity;
    const scratch = modularity(graph, result.partition.assignments);
    expect(cached).toBeCloseTo(scratch, 10);
  });

  it('reports plausible level and move counts', () => {
    const result = leiden(graph, { seed: 42 });
    expect(result.levels).toBeGreaterThan(0);
    expect(result.levels).toBeLessThan(20); // Karate converges in 2-5 levels
    expect(result.totalMoves).toBeGreaterThan(0);
  });
});

describe('leiden — Cat-3 Property: connectivity guarantee on full pipeline', () => {
  it('every output community is internally connected — Karate', () => {
    const { n, edges } = loadKarate();
    const graph = Graph.fromEdgeList(n, edges);
    for (const seed of [1, 7, 42, 1000, 0xb00b]) {
      const result = leiden(graph, { seed });
      const dc = findDisconnectedCommunity(graph, result.partition);
      if (dc !== null) {
        expect.fail(
          `seed=${seed}: output community ${dc} is disconnected`,
        );
      }
    }
  });

  it('every output community is connected — random K_n', () => {
    for (const n of [4, 6, 10]) {
      const g = clique(n);
      const result = leiden(g);
      const dc = findDisconnectedCommunity(g, result.partition);
      expect(dc).toBeNull();
    }
  });
});

describe('leiden — Cat-3 Property: determinism', () => {
  it('same seed → byte-identical assignments and Q', () => {
    const { n, edges } = loadKarate();
    const g = Graph.fromEdgeList(n, edges);
    const a = leiden(g, { seed: 12345 });
    const b = leiden(g, { seed: 12345 });
    for (let u = 0; u < g.n; u++) {
      expect(a.partition.assignments[u]).toBe(b.partition.assignments[u]);
    }
    expect(a.modularity).toBe(b.modularity);
    expect(a.levels).toBe(b.levels);
  });
});

describe('leiden — Cat-3 Property: warm-start (incremental clustering)', () => {
  it('warm-start with an already-optimal partition is a no-op or near-no-op', () => {
    // For two-triangles, we know the optimum: {0,1,2}, {3,4,5} with Q=0.5.
    const g = twoTriangles();
    const optimal = Partition.fromAssignments(
      g,
      new Int32Array([0, 0, 0, 1, 1, 1]),
    );
    const initialQ = optimal.modularity();

    const result = leiden(g, { initialPartition: optimal, seed: 1 });

    // Final Q should be ≥ initial (at most we maintain optimum).
    expect(result.modularity).toBeGreaterThanOrEqual(initialQ - 1e-12);
    expect(result.modularity).toBeCloseTo(0.5, 10);
  });

  it('warm-start does NOT mutate the supplied partition', () => {
    const g = twoTriangles();
    const userPartition = Partition.fromAssignments(
      g,
      new Int32Array([0, 0, 1, 1, 0, 1]), // not optimal
    );
    const userAssignmentsBefore = Array.from(userPartition.assignments);
    leiden(g, { initialPartition: userPartition, seed: 1 });
    expect(Array.from(userPartition.assignments)).toEqual(userAssignmentsBefore);
  });
});

describe('leiden — Cat-5 Contract', () => {
  it('throws on non-finite resolution', () => {
    expect(() => leiden(clique(3), { resolution: NaN })).toThrowError(
      RangeError,
    );
  });

  it('throws on non-positive randomness', () => {
    expect(() => leiden(clique(3), { randomness: 0 })).toThrowError(RangeError);
    expect(() => leiden(clique(3), { randomness: -1 })).toThrowError(RangeError);
  });

  it('throws on zero or non-integer maxLevels', () => {
    expect(() => leiden(clique(3), { maxLevels: 0 })).toThrowError(RangeError);
    expect(() => leiden(clique(3), { maxLevels: 1.5 })).toThrowError(RangeError);
  });

  it('throws when initialPartition belongs to a different graph', () => {
    const g1 = clique(3);
    const g2 = clique(4);
    const wrong = Partition.singletons(g2);
    expect(() => leiden(g1, { initialPartition: wrong })).toThrowError(
      RangeError,
    );
  });

  it('throws LeidenConvergenceError when maxLevels is exceeded', () => {
    // Karate converges in ~2-5 levels; capping at 1 forces non-convergence.
    const { n, edges } = loadKarate();
    const g = Graph.fromEdgeList(n, edges);
    expect(() => leiden(g, { maxLevels: 1, seed: 42 })).toThrowError(
      LeidenConvergenceError,
    );
  });

  it('LFR-1k μ=0.3 seed=0xface terminates (regression: fixed-point at level ≥ 1)', () => {
    // Regression for the bug fixed in src/leiden.ts: the multi-level loop
    // used isAllSingletons() as its only termination signal. On LFR-1k
    // μ=0.3 with seed=0xface, localMove at level 4+ converges with zero
    // moves on a non-singleton partition — refine+aggregate would then
    // produce a same-sized super-graph carrying the same partition, and
    // the next level repeats. The loop ran to maxLevels=100 and threw
    // LeidenConvergenceError despite Q being stable around 0.525.
    //
    // Fix: also break when localMove returns moves==0. See ADR-0012 for
    // why this is paired with isAllSingletons rather than replacing it.
    const path = resolve(__dirname, '../bench/compare/fixtures/lfr-1k-mu03.json');
    const raw = JSON.parse(readFileSync(path, 'utf8')) as {
      n: number;
      edges: ReadonlyArray<[number, number]>;
    };
    const g = Graph.fromEdgeList(raw.n, raw.edges as Edge[]);
    const r = leiden(g, { seed: 0xface });
    expect(r.modularity).toBeGreaterThan(0.5); // Q for LFR-1k μ=0.3 is ~0.53
    expect(r.levels).toBeLessThan(100);
  });
});
