/**
 * Tests for aggregation.
 *
 * Categories:
 *   Cat-1 Identity      — small hand-derivable cases
 *   Cat-3 Property      — Q-preservation: modularity on the super-graph with
 *                         the projected partition equals modularity on the
 *                         original graph with the parent partition
 *   Cat-5 Contract      — graph-partition mismatch; size-cap throw
 */

import { describe, expect, it } from 'vitest';
import fc from 'fast-check';

import { Graph } from '../src/graph.js';
import { Partition } from '../src/partition.js';
import { localMove } from '../src/localMove.js';
import { refine } from '../src/refine.js';
import { aggregate } from '../src/aggregate.js';
import { modularity } from '../src/modularity.js';
import type { Edge } from '../src/types.js';
import { clique, twoTriangles } from './helpers.js';

describe('aggregate — Cat-1 Identity on tiny graphs', () => {
  it('two-triangles with refined = 2 communities → super-graph of 2 isolated nodes with self-loops', () => {
    // Two K3 cliques. After localMove + refine, expect 2 refined communities,
    // one per triangle. Aggregating: 2 super-nodes, each with a self-loop
    // weight = 3 (the 3 internal edges of each triangle), no cross-edges.
    const g = twoTriangles();
    const lm = localMove(g);
    const rf = refine(g, lm.partition);

    const agg = aggregate(g, rf.refined, rf.parentOfRefined);

    expect(agg.graph.n).toBe(2);
    expect(agg.graph.hasSelfLoops).toBe(true);

    // Each super-node has a self-loop with weight 3 (sum of 3 internal
    // unit-weight edges in a triangle), no other edges.
    for (let s = 0; s < 2; s++) {
      // Self-loop is the only neighbor (target == s).
      expect(agg.graph.degree(s)).toBe(1);
      // weightedDegree counts self-loops with 2× the loop weight (per
      // Newman convention) → 2 × 3 = 6.
      expect(agg.graph.weightedDegree(s)).toBe(6);
    }

    // No edges between super-nodes (the triangles are disjoint).
    // graph.m counts unique edges including self-loops; expect exactly 2.
    expect(agg.graph.m).toBe(2);
  });

  it('two cliques connected by a bridge — super-graph has 2 nodes + 1 cross-edge', () => {
    // Two triangles connected by one edge.
    // Edges: 0-1, 0-2, 1-2 (triangle A); 3-4, 3-5, 4-5 (triangle B); 2-3 (bridge)
    const g = Graph.fromEdgeList(6, [
      [0, 1], [0, 2], [1, 2],
      [3, 4], [3, 5], [4, 5],
      [2, 3], // bridge
    ]);
    // Hand-craft a refinement: A = {0,1,2}, B = {3,4,5} via fromAssignments.
    const handCrafted = new Int32Array([0, 0, 0, 1, 1, 1]);
    const refinedPartition = Partition.fromAssignments(g, handCrafted);
    const parentOfRefined = new Int32Array([0, 1]); // each refined community is its own parent

    const agg = aggregate(g, refinedPartition, parentOfRefined);

    expect(agg.graph.n).toBe(2);
    // 2 self-loops + 1 cross-edge = 3 unique edges
    expect(agg.graph.m).toBe(3);
    expect(agg.graph.hasSelfLoops).toBe(true);

    // Cross-edge weight = 1 (the single bridge).
    // Self-loop on each super-node = 3 (3 internal edges per triangle).
    // weightedDegree[s] = 2 × 3 (self-loop) + 1 (bridge) = 7.
    expect(agg.graph.weightedDegree(0)).toBe(7);
    expect(agg.graph.weightedDegree(1)).toBe(7);
  });

  it('K_n with refined = singletons → super-graph identical structurally', () => {
    // If refinement keeps everything as singletons, aggregation just
    // canonicalizes the graph (no merging happens).
    const g = clique(4);
    const refinedPartition = Partition.singletons(g);
    const parentOfRefined = new Int32Array(g.n);
    for (let u = 0; u < g.n; u++) parentOfRefined[u] = u;

    const agg = aggregate(g, refinedPartition, parentOfRefined);

    expect(agg.graph.n).toBe(4);
    expect(agg.graph.m).toBe(g.m); // 6 edges
    expect(agg.graph.totalWeight).toBe(g.totalWeight);
  });

  it('superNodeOf maps each original node to its (renumbered) refined community', () => {
    const g = twoTriangles();
    const refinedPartition = Partition.fromAssignments(
      g,
      new Int32Array([7, 7, 7, 99, 99, 99]),
    );
    const parentOfRefined = new Int32Array(refinedPartition.numCommunities);
    parentOfRefined[0] = 0;
    parentOfRefined[1] = 1;

    const agg = aggregate(g, refinedPartition, parentOfRefined);

    // After Partition.fromAssignments, ids are dense [0,2):
    //   node 0,1,2 → 0; node 3,4,5 → 1.
    // Aggregation preserves these (since both communities are non-empty).
    expect(Array.from(agg.superNodeOf)).toEqual([0, 0, 0, 1, 1, 1]);
  });
});

describe('aggregate — Cat-3 Property: Q-preservation across levels', () => {
  // The key correctness property: modularity on the super-graph computed
  // with the projected parent partition must equal modularity on the
  // original graph computed with the parent partition. If aggregation is
  // implemented correctly, the two values are identical (modulo float64).

  it('Q is preserved across aggregation on random small graphs (clique-based)', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 3, max: 12 }),
        fc.integer({ min: 0, max: 0xffffffff }),
        (n, seed) => {
          const g = clique(n);
          const lm = localMove(g, undefined, { seed });
          const rf = refine(g, lm.partition, { seed });

          const Q_original = lm.partition.modularity();
          const agg = aggregate(g, rf.refined, rf.parentOfRefined);
          const Q_super = agg.partition.modularity();

          return Math.abs(Q_original - Q_super) < 1e-10;
        },
      ),
      { numRuns: 50, seed: 0xcafe },
    );
  });

  it('Q-preservation on two-triangles with hand-crafted partition', () => {
    const g = twoTriangles();
    const lm = localMove(g);
    const rf = refine(g, lm.partition);

    const Q_original = lm.partition.modularity();
    const agg = aggregate(g, rf.refined, rf.parentOfRefined);
    const Q_super = agg.partition.modularity();

    expect(Q_super).toBeCloseTo(Q_original, 12);
  });

  it('Q-preservation when refined is a partition of singletons (no merges)', () => {
    // Edge case: if refine does no merging (everyone is in their own
    // singleton refined community), aggregation should produce a graph
    // structurally equivalent to the original, with the same Q.
    const g = clique(5);
    const handRefined = Partition.singletons(g);
    const handParent = new Int32Array(g.n).fill(0); // all in parent community 0
    const parentOfRefined = new Int32Array(g.n).fill(0);

    void handParent;
    const agg = aggregate(g, handRefined, parentOfRefined);
    // Super-partition has all super-nodes in community 0. Q should match.
    const original = Partition.fromAssignments(g, new Int32Array(g.n).fill(0));
    expect(agg.partition.modularity()).toBeCloseTo(original.modularity(), 12);
  });
});

describe('aggregate — Cat-5 Contract', () => {
  it('throws on refined partition belonging to a different graph', () => {
    const g1 = clique(3);
    const g2 = clique(4);
    const wrong = Partition.singletons(g2);
    expect(() => aggregate(g1, wrong, new Int32Array(g2.n))).toThrowError(
      RangeError,
    );
  });

  it('throws on parentOfRefined length mismatch', () => {
    const g = clique(3);
    const refined = Partition.singletons(g);
    expect(() => aggregate(g, refined, new Int32Array(2))).toThrowError(
      RangeError,
    );
    expect(() => aggregate(g, refined, new Int32Array(5))).toThrowError(
      RangeError,
    );
  });
});
