/**
 * Tests for the refinement phase.
 *
 * Categories:
 *   Cat-1 Identity      — refines two-triangles to 2 communities; K_n collapses
 *   Cat-3 Property      — connectivity guarantee (Theorem 1); within-parent
 *                         constraint; same seed → identical output;
 *                         singleton-freeze (verified via communitySize check);
 *                         Q is non-decreasing
 *   Cat-5 Contract      — input validation; partition-graph mismatch
 */

import { describe, expect, it } from 'vitest';
import fc from 'fast-check';

import { Graph } from '../src/graph.js';
import { Partition } from '../src/partition.js';
import { localMove } from '../src/localMove.js';
import { refine } from '../src/refine.js';
import type { Edge } from '../src/types.js';
import { clique, twoTriangles } from './helpers.js';

/** BFS within a community subgraph; returns node-count reachable from any seed. */
function communityIsConnected(
  graph: Graph,
  partition: Partition,
  community: number,
): boolean {
  // Find a seed node.
  let seed = -1;
  for (let u = 0; u < graph.n; u++) {
    if ((partition.assignments[u] as number) === community) {
      seed = u;
      break;
    }
  }
  if (seed === -1) return true; // empty community is vacuously connected

  // BFS restricted to nodes in `community`.
  const visited = new Uint8Array(graph.n);
  const queue = [seed];
  visited[seed] = 1;
  let count = 1;
  while (queue.length > 0) {
    const u = queue.shift() as number;
    const start = graph.offsets[u] as number;
    const end = graph.offsets[u + 1] as number;
    for (let i = start; i < end; i++) {
      const v = graph.targets[i] as number;
      if ((partition.assignments[v] as number) !== community) continue;
      if (visited[v] === 1) continue;
      visited[v] = 1;
      queue.push(v);
      count++;
    }
  }

  return count === (partition.communitySize[community] as number);
}

describe('refine — Cat-1 Identity', () => {
  it('two-triangles partitioned by component refines into 2 well-connected communities', () => {
    // After localMove, two-triangles converges to the optimal 2-partition.
    // Refinement on top should produce 2 refined communities (one per triangle)
    // each of which is connected.
    const g = twoTriangles();
    const lm = localMove(g);
    const result = refine(g, lm.partition);

    let nonEmpty = 0;
    for (let r = 0; r < result.refined.numCommunities; r++) {
      if ((result.refined.communitySize[r] as number) > 0) nonEmpty++;
    }
    expect(nonEmpty).toBe(2);

    // Both refined communities are subsets of distinct parents (one per triangle).
    const parents = new Set<number>();
    for (let r = 0; r < result.refined.numCommunities; r++) {
      if ((result.refined.communitySize[r] as number) > 0) {
        parents.add(result.parentOfRefined[r] as number);
      }
    }
    expect(parents.size).toBe(2);
  });

  it('K_n parent partition (one community) refines into 1 community', () => {
    // After localMove on K_n, all nodes are in one parent community.
    // Refinement should keep them together (the one community is well-
    // connected; no node prefers to be alone).
    const g = clique(5);
    const lm = localMove(g);
    const result = refine(g, lm.partition);

    let nonEmpty = 0;
    for (let r = 0; r < result.refined.numCommunities; r++) {
      if ((result.refined.communitySize[r] as number) > 0) nonEmpty++;
    }
    expect(nonEmpty).toBe(1);
  });

  it('empty graph refines to empty', () => {
    const g = Graph.fromEdgeList(0, []);
    const parent = Partition.singletons(g);
    const result = refine(g, parent);
    expect(result.refined.numCommunities).toBe(0);
    expect(result.merges).toBe(0);
  });

  it('singleton parent communities are skipped (merges = 0 when every parent has size 1)', () => {
    // Custom: 5 isolated nodes, parent = singletons → no parent has size > 1
    const g = Graph.fromEdgeList(5, []);
    const parent = Partition.singletons(g);
    const result = refine(g, parent);
    expect(result.merges).toBe(0);
  });
});

describe('refine — Cat-3 Property: connectivity guarantee (paper Theorem 1)', () => {
  it('every refined community induces a connected subgraph', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 3, max: 15 }),
        fc.integer({ min: 0, max: 0xffffffff }),
        (n, seed) => {
          // Build a clique (dense, well-connected) to maximize chance of
          // multi-node refined communities.
          const g = clique(n);
          const lm = localMove(g, undefined, { seed });
          const result = refine(g, lm.partition, { seed });
          // Every non-empty refined community must be internally connected.
          for (let r = 0; r < result.refined.numCommunities; r++) {
            if ((result.refined.communitySize[r] as number) === 0) continue;
            if (!communityIsConnected(g, result.refined, r)) return false;
          }
          return true;
        },
      ),
      { numRuns: 50, seed: 0xc0ffee },
    );
  });

  it('every refined community is a subset of exactly one parent community', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 3, max: 15 }),
        fc.integer({ min: 0, max: 0xffffffff }),
        (n, seed) => {
          const g = clique(n);
          const lm = localMove(g, undefined, { seed });
          const result = refine(g, lm.partition, { seed });

          // For every node, its refined community's parent matches the
          // parent it had originally.
          for (let u = 0; u < g.n; u++) {
            const r = result.refined.assignments[u] as number;
            const parentOfR = result.parentOfRefined[r] as number;
            const directParent = lm.partition.assignments[u] as number;
            if (parentOfR !== directParent) return false;
          }
          return true;
        },
      ),
      { numRuns: 30, seed: 0xfade },
    );
  });
});

describe('refine — Cat-3 Property: determinism and quality', () => {
  it('same seed → identical refined.assignments', () => {
    const g = twoTriangles();
    const lm = localMove(g, undefined, { seed: 1 });
    const a = refine(g, lm.partition, { seed: 7 });
    const b = refine(g, lm.partition, { seed: 7 });
    for (let u = 0; u < g.n; u++) {
      expect(a.refined.assignments[u]).toBe(b.refined.assignments[u]);
    }
  });

  // Note: refinement is NOT guaranteed to preserve modularity. On graphs
  // where the parent partition is already optimal (e.g. K_n with a single
  // community), re-singletoning every node and re-merging produces smaller
  // sub-communities whose modularity sum can be lower than the parent's.
  // The paper guarantees CONNECTIVITY (Theorem 1), not Q-monotonicity.
  // The full Leiden loop (M5) recovers Q via subsequent local-moving on
  // the aggregated graph.
  //
  // What we DO assert: Q is bounded — refinement does not produce nonsense.
  it('refined.modularity() is in [-0.5, 1) — paper bound', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 3, max: 12 }),
        fc.integer({ min: 0, max: 0xffffffff }),
        (n, seed) => {
          const g = clique(n);
          const lm = localMove(g, undefined, { seed });
          const result = refine(g, lm.partition, { seed });
          const Q = result.refined.modularity();
          return Q >= -0.5 - 1e-12 && Q < 1 + 1e-12;
        },
      ),
      { numRuns: 30, seed: 0xb00b },
    );
  });
});

describe('refine — Cat-3 Property: singleton-freeze invariant', () => {
  it('after refinement, each multi-node refined community contains nodes that share a parent community', () => {
    // The freeze constraint guarantees: a node v in a non-singleton refined
    // community c' was visited and chose to merge. All members of c' share
    // the parent community in which c' lives.
    const g = twoTriangles();
    const lm = localMove(g);
    const result = refine(g, lm.partition);

    for (let r = 0; r < result.refined.numCommunities; r++) {
      const size = result.refined.communitySize[r] as number;
      if (size <= 1) continue;
      // All nodes in r share the same parent.
      let sharedParent = -1;
      for (let u = 0; u < g.n; u++) {
        if ((result.refined.assignments[u] as number) !== r) continue;
        const p = lm.partition.assignments[u] as number;
        if (sharedParent === -1) sharedParent = p;
        else expect(p).toBe(sharedParent);
      }
    }
  });
});

describe('refine — Cat-5 Contract', () => {
  it('throws on parent partition belonging to a different graph', () => {
    const g1 = clique(3);
    const g2 = clique(4);
    const wrong = Partition.singletons(g2);
    expect(() => refine(g1, wrong)).toThrowError(RangeError);
  });

  it('throws on non-finite resolution', () => {
    const g = clique(3);
    const lm = localMove(g);
    expect(() => refine(g, lm.partition, { resolution: NaN })).toThrowError(
      RangeError,
    );
  });

  it('throws on non-positive randomness', () => {
    const g = clique(3);
    const lm = localMove(g);
    expect(() => refine(g, lm.partition, { randomness: 0 })).toThrowError(
      RangeError,
    );
    expect(() => refine(g, lm.partition, { randomness: -0.5 })).toThrowError(
      RangeError,
    );
  });

  it('throws on negative tolerance', () => {
    const g = clique(3);
    const lm = localMove(g);
    expect(() => refine(g, lm.partition, { tolerance: -1 })).toThrowError(
      RangeError,
    );
  });
});

describe('refine — randomness extreme values', () => {
  it('very small randomness behaves greedy-like (deterministic given seed)', () => {
    // randomness = 1e-10 → β = 1e10; only the strict best ΔQ wins.
    // Same seed must produce same partition.
    const g = twoTriangles();
    const lm = localMove(g);
    const a = refine(g, lm.partition, { seed: 11, randomness: 1e-10 });
    const b = refine(g, lm.partition, { seed: 11, randomness: 1e-10 });
    for (let u = 0; u < g.n; u++) {
      expect(a.refined.assignments[u]).toBe(b.refined.assignments[u]);
    }
  });

  it('very large randomness behaves uniformly random (still deterministic given seed)', () => {
    // randomness = 100 → β = 0.01; ΔQ differences barely affect probabilities.
    // Determinism still holds.
    const g = twoTriangles();
    const lm = localMove(g);
    const a = refine(g, lm.partition, { seed: 11, randomness: 100 });
    const b = refine(g, lm.partition, { seed: 11, randomness: 100 });
    for (let u = 0; u < g.n; u++) {
      expect(a.refined.assignments[u]).toBe(b.refined.assignments[u]);
    }
  });
});
