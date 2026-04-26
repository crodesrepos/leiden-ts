/**
 * Tests for Partition.
 *
 * Categories:
 *   Cat-1 Identity      — singletons construction matches manual derivation
 *   Cat-2 Reference     — partition.modularity() agrees with standalone modularity()
 *   Cat-3 Property      — invariant preservation across moves; equivalence with renumbering
 *   Cat-5 Contract      — input validation, error codes
 */

import { describe, expect, it } from 'vitest';
import fc from 'fast-check';

import { Graph } from '../src/graph.js';
import { Partition } from '../src/partition.js';
import { modularity } from '../src/modularity.js';
import type { Edge } from '../src/types.js';
import { clique, twoTriangles, path3 } from './helpers.js';

describe('Partition.singletons — Cat-1 Identity', () => {
  it('singleton init produces n communities of size 1 each', () => {
    const g = clique(5);
    const p = Partition.singletons(g);
    expect(p.numCommunities).toBe(5);
    for (let u = 0; u < 5; u++) {
      expect(p.assignments[u]).toBe(u);
      expect(p.communitySize[u]).toBe(1);
    }
  });

  it('singleton communityIncident[u] == nodeWeights[u]', () => {
    const g = twoTriangles();
    const p = Partition.singletons(g);
    for (let u = 0; u < g.n; u++) {
      expect(p.communityIncident[u]).toBe(g.weightedDegree(u));
    }
  });

  it('singleton communityInternal[u] is 0 in graphs without self-loops', () => {
    const g = clique(4);
    const p = Partition.singletons(g);
    for (let u = 0; u < g.n; u++) expect(p.communityInternal[u]).toBe(0);
  });

  it('singleton communityInternal[u] = 2w for a self-loop on u (Newman §3.4)', () => {
    const g = Graph.fromEdgeList(
      2,
      [
        [0, 0, 3],
        [0, 1, 1],
      ] as Edge[],
      { selfLoops: 'allow' },
    );
    const p = Partition.singletons(g);
    expect(p.communityInternal[0]).toBe(6); // 2 × 3
    expect(p.communityInternal[1]).toBe(0);
  });
});

describe('Partition.fromAssignments — Cat-1 Identity & renumbering', () => {
  it('renumbers non-contiguous community ids to dense [0, k)', () => {
    const g = twoTriangles();
    const sparse = new Int32Array([10, 10, 10, 99, 99, 99]);
    const p = Partition.fromAssignments(g, sparse);
    expect(p.numCommunities).toBe(2);
    // First-seen wins: community 10 → 0, community 99 → 1.
    expect(Array.from(p.assignments)).toEqual([0, 0, 0, 1, 1, 1]);
  });

  it('communityIncident sums correctly for known partitions', () => {
    const g = twoTriangles();
    const assignments = new Int32Array([0, 0, 0, 1, 1, 1]);
    const p = Partition.fromAssignments(g, assignments);
    // Each triangle: 3 nodes, each degree 2 (unit weights). Σₜₒₜ = 6.
    expect(p.communityIncident[0]).toBe(6);
    expect(p.communityIncident[1]).toBe(6);
  });

  it('communityInternal counts each non-self edge twice (paper convention)', () => {
    const g = twoTriangles();
    const assignments = new Int32Array([0, 0, 0, 1, 1, 1]);
    const p = Partition.fromAssignments(g, assignments);
    // Each triangle has 3 edges, weight 1 each, counted twice → Σᵢₙ = 6.
    expect(p.communityInternal[0]).toBe(6);
    expect(p.communityInternal[1]).toBe(6);
  });
});

describe('Partition.modularity — Cat-2 Reference', () => {
  // Cross-check against the standalone modularity() function.

  it('agrees with modularity() on singleton K_n', () => {
    for (const n of [3, 5, 8]) {
      const g = clique(n);
      const p = Partition.singletons(g);
      const fromCache = p.modularity();
      const fromScratch = modularity(g, p.assignments);
      expect(fromCache).toBeCloseTo(fromScratch, 12);
    }
  });

  it('agrees with modularity() on two-triangles perfect partition', () => {
    const g = twoTriangles();
    const p = Partition.fromAssignments(g, new Int32Array([0, 0, 0, 1, 1, 1]));
    expect(p.modularity()).toBeCloseTo(0.5, 12);
    expect(p.modularity()).toBeCloseTo(modularity(g, p.assignments), 12);
  });

  it('agrees at non-default resolution', () => {
    const g = twoTriangles();
    const p = Partition.fromAssignments(g, new Int32Array([0, 0, 0, 1, 1, 1]));
    const Q1 = p.modularity(2.0);
    const Q2 = modularity(g, p.assignments, { resolution: 2.0 });
    expect(Q1).toBeCloseTo(Q2, 12);
  });
});

describe('Partition.move — Cat-3 Property: cache invariant preservation', () => {
  it('after any sequence of moves, cached modularity matches scratch modularity', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 2, max: 12 }),
        fc.integer({ min: 0, max: 0xffffffff }),
        fc.array(fc.tuple(fc.integer({ min: 0 }), fc.integer({ min: 0 })), {
          minLength: 0,
          maxLength: 30,
        }),
        (n, _seed, moves) => {
          // Build a clique so all moves are well-defined.
          const g = clique(n);
          const p = Partition.singletons(g);
          // Apply moves; clamp to legal ranges.
          for (const [uRaw, cRaw] of moves) {
            const u = uRaw % n;
            const c = cRaw % p.numCommunities;
            p.move(u, c);
          }
          const cached = p.modularity();
          const scratch = modularity(g, p.assignments);
          return Math.abs(cached - scratch) < 1e-10;
        },
      ),
      { numRuns: 100, seed: 0xab1de },
    );
  });

  it('after merging singletons into one community, Q = 0 (identity)', () => {
    const g = path3();
    const p = Partition.singletons(g);
    // Merge everything into community 0.
    p.move(1, 0);
    p.move(2, 0);
    expect(p.modularity()).toBeCloseTo(0, 12);
  });

  it('move is a no-op when u is already in toCommunity', () => {
    const g = clique(3);
    const p = Partition.singletons(g);
    const Qbefore = p.modularity();
    const sizesBefore = Array.from(p.communitySize);
    p.move(1, 1); // already there
    expect(p.modularity()).toBe(Qbefore);
    expect(Array.from(p.communitySize)).toEqual(sizesBefore);
  });

  it('source community becomes empty after last node leaves; downstream moves still valid', () => {
    const g = clique(3);
    const p = Partition.singletons(g);
    p.move(0, 1);
    p.move(2, 1);
    expect(p.communitySize[0]).toBe(0);
    expect(p.communitySize[2]).toBe(0);
    expect(p.communitySize[1]).toBe(3);
    // All in community 1: Q = 0 by identity.
    expect(p.modularity()).toBeCloseTo(0, 12);
  });
});

describe('Partition.copy — Cat-3 Property: independence', () => {
  it('mutating the copy does not affect the original', () => {
    const g = clique(4);
    const original = Partition.singletons(g);
    const cloned = original.copy();
    cloned.move(0, 1);
    expect(original.assignments[0]).toBe(0);
    expect(cloned.assignments[0]).toBe(1);
  });
});

describe('Partition — Cat-5 Contract', () => {
  it('throws RangeError on assignments length mismatch', () => {
    const g = clique(3);
    expect(() =>
      Partition.fromAssignments(g, new Int32Array(2)),
    ).toThrowError(RangeError);
    expect(() =>
      Partition.fromAssignments(g, new Int32Array(4)),
    ).toThrowError(RangeError);
  });

  it('throws RangeError on move with out-of-range u', () => {
    const g = clique(3);
    const p = Partition.singletons(g);
    expect(() => p.move(-1, 0)).toThrowError(RangeError);
    expect(() => p.move(3, 0)).toThrowError(RangeError);
    expect(() => p.move(0.5, 0)).toThrowError(RangeError);
  });

  it('throws RangeError on move into a non-existent community id', () => {
    const g = clique(3);
    const p = Partition.singletons(g);
    expect(() => p.move(0, 99)).toThrowError(RangeError);
    expect(() => p.move(0, -1)).toThrowError(RangeError);
  });
});
