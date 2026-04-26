/**
 * Tests for the local-moving phase.
 *
 * Categories:
 *   Cat-1 Identity      — terminal Q matches hand-derived value on tiny graphs
 *   Cat-3 Property      — monotonic Q across iterations; determinism by seed;
 *                         output communities have non-negative ΔQ neighbors
 *                         (local optimum); shuffle-invariance of Q
 *   Cat-4 Budget        — runs in finite iterations; fast on Karate
 *   Cat-5 Contract      — throws on bad inputs; ConvergenceError on cap
 */

import { describe, expect, it } from 'vitest';
import fc from 'fast-check';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

import { Graph } from '../src/graph.js';
import { Partition } from '../src/partition.js';
import { modularity } from '../src/modularity.js';
import { localMove, ConvergenceError } from '../src/localMove.js';
import type { Edge } from '../src/types.js';
import { clique, twoTriangles, path3 } from './helpers.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

describe('localMove — Cat-1 Identity', () => {
  it('two-triangles converges to the obvious 2-partition (Q=0.5)', () => {
    // The graph has two disjoint triangles. Optimal Q is exactly 0.5
    // for the 2-partition by component. Local moving from singletons
    // must find this — it is a textbook case.
    const g = twoTriangles();
    const result = localMove(g);
    expect(result.modularity).toBeCloseTo(0.5, 10);
    // Two non-empty communities (one per triangle).
    let nonEmpty = 0;
    for (let c = 0; c < result.partition.numCommunities; c++) {
      if ((result.partition.communitySize[c] as number) > 0) nonEmpty++;
    }
    expect(nonEmpty).toBe(2);
  });

  it('K_n converges to the trivial 1-community partition (Q=0)', () => {
    // A complete graph has no community structure; the only locally-optimal
    // partition is the all-one-community partition with Q=0.
    for (const n of [3, 5, 8]) {
      const g = clique(n);
      const result = localMove(g);
      expect(result.modularity).toBeCloseTo(0, 10);
      // All nodes in the same community.
      const target = result.partition.assignments[0] as number;
      for (let u = 0; u < n; u++) {
        expect(result.partition.assignments[u]).toBe(target);
      }
    }
  });

  it('empty graph (no edges) is already converged: Q=0', () => {
    const g = Graph.fromEdgeList(5, []);
    const result = localMove(g);
    expect(result.modularity).toBe(0);
    expect(result.iterations).toBeLessThanOrEqual(1);
  });

  it('single isolated node converges immediately', () => {
    const g = Graph.fromEdgeList(1, []);
    const result = localMove(g);
    expect(result.modularity).toBe(0);
  });
});

describe('localMove — Cat-1 Identity on the Karate club', () => {
  // Karate's globally-optimal Q via *full* Leiden ≈ 0.4449 (Traag 2019 Fig. 2).
  // Local-moving alone (without refinement+aggregation) reaches a *local*
  // optimum, typically Q ∈ [0.30, 0.40] depending on visit order. The full
  // optimum requires M3+M4 (refinement + aggregation across multiple levels).
  //
  // We test what local-moving alone IS supposed to do, not what full Leiden
  // does:
  //   1. Q strictly improves over the singleton initial.
  //   2. Number of communities strictly decreases (merges actually happened).
  //   3. Cached and scratch modularity agree.
  //   4. Output is locally optimal (no single-node move improves Q) — this
  //      is asserted in the separate "local optimum" property test below.

  type KarateFixture = { n: number; edges: ReadonlyArray<[number, number]> };
  const karate = JSON.parse(
    readFileSync(
      resolve(__dirname, '../bench/compare/fixtures/karate.json'),
      'utf8',
    ),
  ) as KarateFixture;
  const g = Graph.fromEdgeList(karate.n, karate.edges as Edge[]);
  const initialQ = Partition.singletons(g).modularity(); // ≈ −0.0498

  it('strictly improves over singleton-partition Q on Karate (any seed)', () => {
    for (const seed of [1, 7, 42, 1000, 0xc0ffee]) {
      const result = localMove(g, undefined, { seed });
      // Strict improvement: real work happened, not "every node stayed".
      expect(result.modularity).toBeGreaterThan(initialQ);
      // Cache vs scratch agreement.
      expect(result.modularity).toBeCloseTo(
        modularity(g, result.partition.assignments),
        10,
      );
    }
  });

  it('strictly reduces community count from n on Karate (merges happened)', () => {
    for (const seed of [1, 7, 42, 1000, 0xc0ffee]) {
      const result = localMove(g, undefined, { seed });
      // Count non-empty communities.
      let nonEmpty = 0;
      for (let c = 0; c < result.partition.numCommunities; c++) {
        if ((result.partition.communitySize[c] as number) > 0) nonEmpty++;
      }
      expect(nonEmpty).toBeLessThan(g.n); // n=34
      expect(nonEmpty).toBeGreaterThan(0);
    }
  });
});

describe('localMove — Cat-3 Property: determinism', () => {
  it('same seed → byte-identical assignments', () => {
    const g = clique(10);
    const a = localMove(g, undefined, { seed: 12345 });
    const b = localMove(g, undefined, { seed: 12345 });
    for (let u = 0; u < g.n; u++) {
      expect(a.partition.assignments[u]).toBe(b.partition.assignments[u]);
    }
    expect(a.modularity).toBe(b.modularity);
    expect(a.iterations).toBe(b.iterations);
    expect(a.moves).toBe(b.moves);
  });

  it('shuffling the input edge list preserves the converged Q (CSR is canonical)', () => {
    const edgesA: Edge[] = [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 0],
      [0, 2],
    ];
    const edgesB: Edge[] = [
      [3, 0],
      [0, 2],
      [0, 1],
      [2, 3],
      [1, 2],
    ];
    const gA = Graph.fromEdgeList(4, edgesA);
    const gB = Graph.fromEdgeList(4, edgesB);
    const rA = localMove(gA, undefined, { seed: 99 });
    const rB = localMove(gB, undefined, { seed: 99 });
    expect(rA.modularity).toBeCloseTo(rB.modularity, 12);
  });
});

describe('localMove — Cat-3 Property: monotonic improvement & local optimum', () => {
  it('final Q ≥ initial Q (non-decreasing)', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 2, max: 12 }),
        fc.integer({ min: 0, max: 0xffffffff }),
        (n, seed) => {
          const g = clique(n);
          const initialQ = Partition.singletons(g).modularity();
          const finalQ = localMove(g, undefined, { seed }).modularity;
          // For K_n, singletons gives Q ≈ -1/n; final is ≥ that.
          return finalQ >= initialQ - 1e-12;
        },
      ),
      { numRuns: 30, seed: 0xface },
    );
  });

  it('every node is in a community that is a non-strict local maximum', () => {
    // After convergence, no single-node move can produce ΔQ > tolerance.
    // Scan every (node, neighboring community) pair and verify.
    const g = twoTriangles();
    const result = localMove(g);
    const p = result.partition;

    for (let u = 0; u < g.n; u++) {
      const fromComm = p.assignments[u] as number;
      // Compute kuToCommunity for u and check ΔQ for each neighboring community.
      const start = g.offsets[u] as number;
      const end = g.offsets[u + 1] as number;
      const candidates = new Set<number>();
      candidates.add(fromComm);
      for (let i = start; i < end; i++) {
        const v = g.targets[i] as number;
        if (v !== u) candidates.add(p.assignments[v] as number);
      }
      // Inline the modularity delta logic for each candidate.
      const m2 = g.totalWeight;
      const ku = g.nodeWeights[u] as number;
      const totFrom = p.communityIncident[fromComm] as number;
      // Compute kuToFrom and kuToCommunity[c].
      let kuToFrom = 0;
      const kuTo = new Map<number, number>();
      for (let i = start; i < end; i++) {
        const v = g.targets[i] as number;
        if (v === u) continue;
        const w = g.weights[i] as number;
        const cv = p.assignments[v] as number;
        if (cv === fromComm) kuToFrom += w;
        else kuTo.set(cv, (kuTo.get(cv) ?? 0) + w);
      }
      for (const [cand, kuToCand] of kuTo) {
        const totTo = p.communityIncident[cand] as number;
        const internalDelta = (2 * (kuToCand - kuToFrom)) / m2;
        const penaltyDelta = (-2 * ku * (totTo - totFrom + ku)) / (m2 * m2);
        const dq = internalDelta + penaltyDelta;
        // Must not be a strict improvement (any improvement above tolerance
        // would indicate convergence is buggy).
        expect(dq).toBeLessThanOrEqual(1e-10);
      }
    }
  });
});

describe('localMove — Cat-3 Property: warm-start (incremental clustering)', () => {
  it('starting from a custom partition produces a result no worse than starting from singletons', () => {
    const g = twoTriangles();
    // Hand-crafted "almost right" partition: nodes 0,1 in comm 0, rest in comm 1.
    const handCrafted = new Int32Array([0, 0, 1, 1, 1, 1]);
    const init = Partition.fromAssignments(g, handCrafted);
    const initialQ = init.modularity();

    const fromHand = localMove(g, init, { seed: 1 });
    const fromSingle = localMove(g, undefined, { seed: 1 });

    // Both should converge to the same optimum (0.5 for two-triangles).
    expect(fromHand.modularity).toBeGreaterThanOrEqual(initialQ);
    expect(fromHand.modularity).toBeCloseTo(0.5, 10);
    expect(fromSingle.modularity).toBeCloseTo(0.5, 10);
  });
});

describe('localMove — Cat-5 Contract', () => {
  it('throws RangeError on non-finite resolution', () => {
    expect(() => localMove(clique(3), undefined, { resolution: NaN })).toThrowError(
      RangeError,
    );
    expect(() =>
      localMove(clique(3), undefined, { resolution: Infinity }),
    ).toThrowError(RangeError);
  });

  it('throws RangeError on negative tolerance', () => {
    expect(() => localMove(clique(3), undefined, { tolerance: -1 })).toThrowError(
      RangeError,
    );
  });

  it('throws RangeError on zero or non-integer maxIterations', () => {
    expect(() => localMove(clique(3), undefined, { maxIterations: 0 })).toThrowError(
      RangeError,
    );
    expect(() =>
      localMove(clique(3), undefined, { maxIterations: 1.5 }),
    ).toThrowError(RangeError);
  });

  it('throws ConvergenceError when maxIterations is exceeded', () => {
    // Construct a graph + extreme tolerance pair guaranteed to "improve"
    // each pass. A single tiny tolerance can't actually reproduce a
    // pathological non-convergence, so we just cap at 1 and check that
    // ConvergenceError is thrown when convergence requires more than 1 pass.
    //
    // Two-triangles requires multiple passes from singletons to reach Q=0.5.
    const g = twoTriangles();
    expect(() =>
      localMove(g, undefined, { maxIterations: 1 }),
    ).toThrowError(ConvergenceError);
  });

  it('ConvergenceError carries iterations and lastModularity', () => {
    const g = twoTriangles();
    try {
      localMove(g, undefined, { maxIterations: 1 });
      expect.fail('expected ConvergenceError');
    } catch (err) {
      expect(err).toBeInstanceOf(ConvergenceError);
      expect((err as ConvergenceError).iterations).toBe(1);
      expect((err as ConvergenceError).lastModularity).toBeGreaterThanOrEqual(-1);
    }
  });
});
