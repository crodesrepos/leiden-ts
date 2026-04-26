/**
 * Newman–Girvan modularity for an undirected weighted graph.
 *
 * Reference: Traag, Waltman & van Eck (2019) eq. 1; Newman & Girvan (2004).
 *
 *   Q = (1 / 2m) · Σᵤᵥ [Aᵤᵥ − γ · kᵤkᵥ / 2m] · δ(cᵤ, cᵥ)
 *
 * Equivalent per-community form (used here):
 *
 *   Q = Σ_c [ Σᵢₙ(c) / 2m  −  γ · (Σₜₒₜ(c) / 2m)² ]
 *
 * where
 *   Σᵢₙ(c)   = sum of edge weights with both endpoints in community c,
 *              with each undirected edge counted **twice** (once per
 *              direction), so a single intra-community edge of weight w
 *              contributes 2w. Self-loops contribute 2w.
 *   Σₜₒₜ(c)  = sum of node weights for nodes in community c
 *              (= Σᵢₙ(c) + edges from c to outside).
 *   2m       = totalWeight (sum of node weights).
 *
 * Numerical stability: the per-community contributions are accumulated
 * with Kahan compensation.
 *
 * @param graph     the input graph
 * @param partition Int32Array of length graph.n; partition[u] = community id
 *                  (community ids are arbitrary 32-bit signed integers; they
 *                  do not need to be contiguous or non-negative)
 * @param options   { resolution } — γ, default 1.0
 * @returns         modularity in [-0.5, 1)
 *
 * @throws RangeError if partition.length !== graph.n
 * @throws RangeError if resolution is not a finite number
 */

import type { Graph } from './graph.js';
import { KahanAccumulator } from './internal/kahan.js';
import type { ModularityOptions } from './types.js';

export function modularity(
  graph: Graph,
  partition: Int32Array,
  options?: ModularityOptions,
): number {
  if (partition.length !== graph.n) {
    throw new RangeError(
      `partition length ${partition.length} does not match graph.n ${graph.n}`,
    );
  }
  const resolution = options?.resolution ?? 1.0;
  if (!Number.isFinite(resolution)) {
    throw new RangeError(
      `resolution must be a finite number; got ${String(resolution)}`,
    );
  }

  const m2 = graph.totalWeight; // 2m
  if (m2 === 0) {
    // Empty graph (no edges): modularity is undefined in the strict formula
    // (division by zero). Convention here returns 0 for any partition,
    // which is consistent with networkx and graspologic.
    return 0;
  }

  // Per-community accumulators: internal weight (Σᵢₙ) and total weight (Σₜₒₜ).
  //
  // Communities ids may be any int32; we use Maps keyed by community id.
  // For modularity (computed once per call) the Map cost is acceptable.
  // The hot path that needs typed-array indexed accumulators is local
  // moving (M2), where community ids are guaranteed contiguous.
  const internalByCommunity = new Map<number, KahanAccumulator>();
  const totalByCommunity = new Map<number, KahanAccumulator>();

  const offsets = graph.offsets;
  const targets = graph.targets;
  const weights = graph.weights;
  const nodeWeights = graph.nodeWeights;
  const n = graph.n;

  // Walk every half-edge once, attributing weight to the appropriate
  // accumulator. Each non-self-loop edge contributes 2w to Σᵢₙ if both
  // endpoints share a community: the half-edge u→v contributes w, and
  // v→u contributes w again. Self-loops appear once with target == u
  // and contribute 2w to Σᵢₙ (per Newman convention).
  for (let u = 0; u < n; u++) {
    const cu = partition[u] as number;
    const start = offsets[u] as number;
    const end = offsets[u + 1] as number;
    for (let i = start; i < end; i++) {
      const v = targets[i] as number;
      const w = weights[i] as number;
      const cv = partition[v] as number;
      if (cu === cv) {
        let acc = internalByCommunity.get(cu);
        if (acc === undefined) {
          acc = new KahanAccumulator();
          internalByCommunity.set(cu, acc);
        }
        acc.add(w);
        // Self-loop counted twice to match the paper's A(u,u) = 2 convention.
        if (v === u) acc.add(w);
      }
    }

    // Σₜₒₜ(c) is the sum of nodeWeights over nodes in c.
    let tot = totalByCommunity.get(cu);
    if (tot === undefined) {
      tot = new KahanAccumulator();
      totalByCommunity.set(cu, tot);
    }
    tot.add(nodeWeights[u] as number);
  }

  // Q = Σ_c [ Σᵢₙ(c) / 2m  −  γ · (Σₜₒₜ(c) / 2m)² ]
  const q = new KahanAccumulator();
  const inv2m = 1 / m2;
  for (const [community, totAcc] of totalByCommunity) {
    const tot = totAcc.value;
    const intern = internalByCommunity.get(community)?.value ?? 0;
    const term = intern * inv2m - resolution * (tot * inv2m) * (tot * inv2m);
    q.add(term);
  }
  return q.value;
}
