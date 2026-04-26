/**
 * Aggregation phase of the Leiden algorithm (Traag, Waltman & van Eck 2019,
 * Algorithm 4). Collapses each refined community into a single super-node
 * to produce a smaller graph whose modularity matches the original at the
 * coarser level.
 *
 * Construction:
 *   - Each non-empty refined community c' becomes a super-node s_c'.
 *     Empty communities are dropped; super-node ids are dense [0, k).
 *   - Edges within c' (including self-loops in the original) become a
 *     self-loop on s_c' with weight w_self where w_self equals the sum of
 *     within-c' edge weights. Under the Newman convention A(s,s)=2w_self,
 *     this contributes 2 × Σᵢₙ(c')/2 = Σᵢₙ(c') to the super-node's
 *     internal weight — preserving the modularity formula across levels.
 *   - Cross-community edges (u in c'₁, v in c'₂) become super-edges
 *     between s_c'₁ and s_c'₂; multiple original edges are summed.
 *
 * The new partition over the super-graph projects the *parent* partition
 * via `parentOfRefined`: each super-node s_c' belongs to the parent
 * community of its constituents.
 *
 * See ADR-0011 (self-loop handling) and ADR-0001 (canonical CSR via
 * Graph.fromEdgeList).
 */

import { Graph } from './graph.js';
import { Partition } from './partition.js';
import type { Edge } from './types.js';

/** Returned by `aggregate`. */
export interface AggregateResult {
  /** The super-graph: one node per non-empty refined community. */
  readonly graph: Graph;
  /**
   * Partition over the super-graph. assignments[s] is the parent community
   * of super-node s — i.e., the parent partition projected onto the
   * super-graph's nodes.
   */
  readonly partition: Partition;
  /**
   * superNodeOf[u] = super-node id of original node u. Length = original
   * graph.n. Used by `leiden()` to project the final partition back to
   * original-node assignments after multiple aggregation levels.
   */
  readonly superNodeOf: Int32Array;
}

/**
 * Maximum super-graph node count for the numeric edge-key dedup scheme.
 * `lo × k + hi` must fit in Uint32; lo, hi < k means k² ≤ 2^32, so
 * k ≤ 65535. Real workloads compress 100×–1000× per aggregation, so
 * even n=10⁶ graphs land at k ~ 10³–10⁴ after one level.
 */
const MAX_SUPER_NODE_COUNT = 65535;

export function aggregate(
  graph: Graph,
  refined: Partition,
  parentOfRefined: Int32Array,
): AggregateResult {
  if (refined.graph !== graph) {
    throw new RangeError('refined partition does not belong to graph');
  }
  if (parentOfRefined.length !== refined.numCommunities) {
    throw new RangeError(
      `parentOfRefined length ${parentOfRefined.length} does not match ` +
        `refined.numCommunities ${refined.numCommunities}`,
    );
  }

  const n = graph.n;

  // Step 1: dense renumbering of non-empty refined community ids.
  const oldToNew = new Int32Array(refined.numCommunities).fill(-1);
  let k = 0;
  for (let c = 0; c < refined.numCommunities; c++) {
    if ((refined.communitySize[c] as number) > 0) {
      oldToNew[c] = k;
      k++;
    }
  }

  if (k > MAX_SUPER_NODE_COUNT) {
    throw new RangeError(
      `super-graph node count ${k} exceeds ${MAX_SUPER_NODE_COUNT}; ` +
        `the numeric edge-key scheme would overflow Uint32. ` +
        `Reduce graph size or implement a BigInt-keyed fallback.`,
    );
  }

  // Step 2: superNodeOf for each original node.
  const superNodeOf = new Int32Array(n);
  for (let u = 0; u < n; u++) {
    superNodeOf[u] = oldToNew[refined.assignments[u] as number] as number;
  }

  // Step 3: aggregate edges.
  //
  // Self-loops on the super-graph collect:
  //   (a) within-community non-self edges in the original graph
  //   (b) self-loops in the original graph
  //
  // For each undirected edge (u, v, w) in the original (u ≤ v to count once):
  //   - If u == v (self-loop): selfLoopWeights[s_u] += w
  //   - If sup_u == sup_v (cross-node, same community): selfLoopWeights[s_u] += w
  //   - Else: edgeMap[(min(s_u,s_v), max(s_u,s_v))] += w
  //
  // edgeMap keyed by (lo × k + hi) — Uint32 safe given k ≤ 65535.
  const selfLoopWeights = new Float64Array(k);
  const edgeMap = new Map<number, number>();

  for (let u = 0; u < n; u++) {
    const sup_u = superNodeOf[u] as number;
    const start = graph.offsets[u] as number;
    const end = graph.offsets[u + 1] as number;
    for (let i = start; i < end; i++) {
      const v = graph.targets[i] as number;
      if (v < u) continue; // each undirected edge counted once via lower-id endpoint
      const w = graph.weights[i] as number;
      const sup_v = superNodeOf[v] as number;

      if (sup_u === sup_v) {
        selfLoopWeights[sup_u] = (selfLoopWeights[sup_u] as number) + w;
      } else {
        const lo = sup_u < sup_v ? sup_u : sup_v;
        const hi = sup_u < sup_v ? sup_v : sup_u;
        const key = lo * k + hi;
        edgeMap.set(key, (edgeMap.get(key) ?? 0) + w);
      }
    }
  }

  // Step 4: build edge list for the super-graph.
  //   self-loops first, then cross-edges. selfLoops:'allow' is required
  //   so Graph.fromEdgeList accepts the loop entries.
  const edges: Edge[] = [];
  for (let s = 0; s < k; s++) {
    const w = selfLoopWeights[s] as number;
    if (w > 0) edges.push([s, s, w]);
  }
  for (const [key, w] of edgeMap) {
    const lo = Math.floor(key / k);
    const hi = key % k;
    edges.push([lo, hi, w]);
  }

  // Step 5: build the super-graph. validate:false because we trust the
  // upstream invariants (fresh dense ids, non-negative weights, no
  // duplicates by construction).
  const superGraph = Graph.fromEdgeList(k, edges, {
    selfLoops: 'allow',
    validate: false,
  });

  // Step 6: build the super-partition.
  //
  // Each super-node s belongs to the parent community of its constituents.
  // We need a reverse mapping: super-node s → original refined community id.
  // Build it by inverting oldToNew.
  const refinedOfSuper = new Int32Array(k);
  for (let c = 0; c < refined.numCommunities; c++) {
    const newId = oldToNew[c] as number;
    if (newId !== -1) refinedOfSuper[newId] = c;
  }

  const newAssignments = new Int32Array(k);
  for (let s = 0; s < k; s++) {
    const refinedId = refinedOfSuper[s] as number;
    newAssignments[s] = parentOfRefined[refinedId] as number;
  }

  const newPartition = Partition.fromAssignments(superGraph, newAssignments);

  return {
    graph: superGraph,
    partition: newPartition,
    superNodeOf,
  };
}
