import { Graph } from '../src/graph.js';
import type { Edge } from '../src/types.js';

/**
 * Tiny graph helpers for tests. Not exported from the package — these are
 * test fixtures only.
 */

/**
 * A path of three nodes: 0 - 1 - 2 with unit weights.
 */
export function path3(): Graph {
  return Graph.fromEdgeList(3, [
    [0, 1],
    [1, 2],
  ] as Edge[]);
}

/**
 * Two disjoint cliques of 3 each (nodes 0,1,2 and 3,4,5).
 * Q for the obvious 2-partition is exactly 0.5.
 */
export function twoTriangles(): Graph {
  return Graph.fromEdgeList(6, [
    [0, 1],
    [0, 2],
    [1, 2],
    [3, 4],
    [3, 5],
    [4, 5],
  ] as Edge[]);
}

/**
 * A square: 0-1-2-3-0.
 */
export function square(): Graph {
  return Graph.fromEdgeList(4, [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
  ] as Edge[]);
}

/**
 * A complete graph K_n with unit weights.
 */
export function clique(n: number): Graph {
  const edges: Edge[] = [];
  for (let u = 0; u < n; u++) {
    for (let v = u + 1; v < n; v++) {
      edges.push([u, v]);
    }
  }
  return Graph.fromEdgeList(n, edges);
}

/**
 * Compare two partitions modulo community-id permutation.
 *
 * `[0,0,1,1]` and `[3,3,7,7]` are the same partition; `[0,0,1,1]` and
 * `[0,1,0,1]` are not.
 */
export function partitionsEqual(
  a: Int32Array | ReadonlyArray<number>,
  b: Int32Array | ReadonlyArray<number>,
): boolean {
  if (a.length !== b.length) return false;
  const map = new Map<number, number>();
  const seen = new Set<number>();
  for (let i = 0; i < a.length; i++) {
    const ai = a[i] as number;
    const bi = b[i] as number;
    const mapped = map.get(ai);
    if (mapped === undefined) {
      if (seen.has(bi)) return false; // bi already used by a different ai
      map.set(ai, bi);
      seen.add(bi);
    } else if (mapped !== bi) {
      return false;
    }
  }
  return true;
}
