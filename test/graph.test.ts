import { describe, expect, it } from 'vitest';
import { Graph } from '../src/graph.js';
import { GraphValidationError } from '../src/types.js';
import { path3, twoTriangles, square, clique } from './helpers.js';

describe('Graph.fromEdgeList — basic construction', () => {
  it('builds an empty graph with no edges', () => {
    const g = Graph.fromEdgeList(0, []);
    expect(g.n).toBe(0);
    expect(g.m).toBe(0);
    expect(g.offsets).toEqual(new Uint32Array([0]));
    expect(g.targets.length).toBe(0);
    expect(g.weights.length).toBe(0);
    expect(g.totalWeight).toBe(0);
    expect(g.hasSelfLoops).toBe(false);
  });

  it('builds a single isolated node', () => {
    const g = Graph.fromEdgeList(1, []);
    expect(g.n).toBe(1);
    expect(g.m).toBe(0);
    expect(g.degree(0)).toBe(0);
    expect(g.weightedDegree(0)).toBe(0);
  });

  it('builds path 0—1—2 correctly', () => {
    const g = path3();
    expect(g.n).toBe(3);
    expect(g.m).toBe(2);
    // each non-self-loop edge appears as 2 half-edges
    expect(g.targets.length).toBe(4);
    expect(g.weights.length).toBe(4);
    // node 1 has degree 2; nodes 0 and 2 have degree 1
    expect(g.degree(0)).toBe(1);
    expect(g.degree(1)).toBe(2);
    expect(g.degree(2)).toBe(1);
    // weighted degrees: unit weights → identical
    expect(g.weightedDegree(0)).toBe(1);
    expect(g.weightedDegree(1)).toBe(2);
    expect(g.weightedDegree(2)).toBe(1);
    // totalWeight = 2 * sum(unique weights) = 2 * (1 + 1) = 4
    expect(g.totalWeight).toBe(4);
  });

  it('honors explicit edge weights', () => {
    const g = Graph.fromEdgeList(2, [[0, 1, 2.5]]);
    expect(g.weightedDegree(0)).toBe(2.5);
    expect(g.weightedDegree(1)).toBe(2.5);
    expect(g.totalWeight).toBe(5);
  });

  it('builds undirected symmetry: every (u→v) has matching (v→u)', () => {
    const g = path3();
    // walk each half-edge; every (u, v, w) must have a partner (v, u, w)
    for (let u = 0; u < g.n; u++) {
      const start = g.offsets[u]!;
      const end = g.offsets[u + 1]!;
      for (let i = start; i < end; i++) {
        const v = g.targets[i]!;
        const w = g.weights[i]!;
        if (u === v) continue; // self-loops have no partner
        // find u in v's neighbors
        const vStart = g.offsets[v]!;
        const vEnd = g.offsets[v + 1]!;
        let found = false;
        for (let j = vStart; j < vEnd; j++) {
          if (g.targets[j] === u && g.weights[j] === w) {
            found = true;
            break;
          }
        }
        expect(found).toBe(true);
      }
    }
  });

  it('handles multiple edges between distinct node pairs as duplicates (rejected)', () => {
    expect(() =>
      Graph.fromEdgeList(2, [
        [0, 1],
        [0, 1],
      ]),
    ).toThrowError(GraphValidationError);
  });

  it('handles two disjoint triangles', () => {
    const g = twoTriangles();
    expect(g.n).toBe(6);
    expect(g.m).toBe(6);
    // each node in a triangle has degree 2
    for (let u = 0; u < 6; u++) expect(g.degree(u)).toBe(2);
    // totalWeight = 2 * 6 = 12
    expect(g.totalWeight).toBe(12);
  });

  it('builds a clique correctly', () => {
    const g = clique(5);
    expect(g.n).toBe(5);
    expect(g.m).toBe(10); // C(5,2) = 10
    for (let u = 0; u < 5; u++) expect(g.degree(u)).toBe(4);
    expect(g.totalWeight).toBe(20); // 2 * 10
  });

  it('builds a square', () => {
    const g = square();
    expect(g.n).toBe(4);
    expect(g.m).toBe(4);
    for (let u = 0; u < 4; u++) expect(g.degree(u)).toBe(2);
  });
});

describe('Graph.fromEdgeList — validation', () => {
  it('rejects negative node ids', () => {
    expect(() => Graph.fromEdgeList(2, [[-1, 0]])).toThrowError(
      /NODE_OUT_OF_RANGE|out of range/,
    );
  });

  it('rejects out-of-range node ids', () => {
    expect(() => Graph.fromEdgeList(2, [[0, 5]])).toThrowError(
      GraphValidationError,
    );
  });

  it('rejects non-integer node ids', () => {
    expect(() => Graph.fromEdgeList(3, [[0, 1.5]])).toThrowError(
      GraphValidationError,
    );
  });

  it('rejects negative edge weights', () => {
    expect(() => Graph.fromEdgeList(2, [[0, 1, -0.5]])).toThrowError(
      /NEGATIVE_WEIGHT|negative/,
    );
  });

  it('rejects non-finite edge weights', () => {
    expect(() => Graph.fromEdgeList(2, [[0, 1, Infinity]])).toThrowError(
      /NOT_FINITE|finite/,
    );
    expect(() => Graph.fromEdgeList(2, [[0, 1, NaN]])).toThrowError(
      GraphValidationError,
    );
  });

  it('rejects self-loops by default', () => {
    expect(() => Graph.fromEdgeList(2, [[0, 0]])).toThrowError(
      /self-loop/i,
    );
  });

  it("'collapse' silently drops self-loops", () => {
    const g = Graph.fromEdgeList(
      3,
      [
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 2],
      ],
      { selfLoops: 'collapse' },
    );
    expect(g.m).toBe(2);
    expect(g.hasSelfLoops).toBe(false);
  });

  it("'allow' keeps self-loops with paper §3.4 semantics", () => {
    // A self-loop of weight w on node u contributes 2w to nodeWeights[u]
    // (because in the undirected formulation A(u,u) is counted twice in
    // the row sum) — see paper §3.4.
    const g = Graph.fromEdgeList(
      2,
      [
        [0, 0, 1],
        [0, 1, 1],
      ],
      { selfLoops: 'allow' },
    );
    expect(g.hasSelfLoops).toBe(true);
    // node 0: self-loop contributes 2*1 + edge to node 1 contributes 1 = 3
    expect(g.weightedDegree(0)).toBe(3);
    // node 1: just the edge to 0 = 1
    expect(g.weightedDegree(1)).toBe(1);
    // totalWeight = sum of nodeWeights = 4
    expect(g.totalWeight).toBe(4);
  });

  it('rejects directed: true', () => {
    expect(() =>
      Graph.fromEdgeList(2, [[0, 1]], {
        directed: true as unknown as false, // intentionally violate the type
      }),
    ).toThrowError(GraphValidationError);
  });

  it('rejects negative or non-integer nodeCount', () => {
    expect(() => Graph.fromEdgeList(-1, [])).toThrowError(GraphValidationError);
    expect(() => Graph.fromEdgeList(1.5, [])).toThrowError(
      GraphValidationError,
    );
  });

  it('attaches the offending edge to the error', () => {
    try {
      Graph.fromEdgeList(2, [
        [0, 1],
        [0, 1],
      ]);
      expect.fail('expected throw');
    } catch (err) {
      expect(err).toBeInstanceOf(GraphValidationError);
      expect((err as GraphValidationError).code).toBe('DUPLICATE_EDGE');
      // The error preserves the user's original tuple form (2-element here).
      expect((err as GraphValidationError).edge).toEqual([0, 1]);
    }
  });

  it('skips validation when validate: false', () => {
    // out-of-range node id is now silently accepted (and the graph will be
    // structurally invalid; this is the user's promise to keep)
    const g = Graph.fromEdgeList(2, [[0, 1]], { validate: false });
    expect(g.n).toBe(2);
    expect(g.m).toBe(1);
  });
});

describe('Graph.fromCSR', () => {
  it('round-trips an edge-list construction', () => {
    const a = path3();
    const b = Graph.fromCSR({
      n: a.n,
      offsets: new Uint32Array(a.offsets),
      targets: new Uint32Array(a.targets),
      weights: new Float64Array(a.weights),
    });
    expect(b.n).toBe(a.n);
    expect(b.m).toBe(a.m);
    expect(b.totalWeight).toBe(a.totalWeight);
    for (let u = 0; u < a.n; u++) {
      expect(b.weightedDegree(u)).toBe(a.weightedDegree(u));
    }
  });

  it('detects self-loops from CSR layout', () => {
    // graph with one self-loop on node 0 and one edge 0-1
    const offsets = new Uint32Array([0, 2, 3]);
    const targets = new Uint32Array([0, 1, 0]);
    const weights = new Float64Array([1, 1, 1]);
    const g = Graph.fromCSR({ n: 2, offsets, targets, weights });
    expect(g.hasSelfLoops).toBe(true);
    expect(g.m).toBe(2); // self-loop counted as 1, the 0-1 edge counted as 1
  });
});
