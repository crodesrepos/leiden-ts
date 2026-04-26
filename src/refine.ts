/**
 * Refinement phase of the Leiden algorithm (Traag, Waltman & van Eck 2019,
 * Algorithm 3). The connectivity-preserving step that distinguishes
 * Leiden from Louvain.
 *
 * Algorithm:
 *   1. Start from a fresh singleton partition.
 *   2. For each parent community S of size > 1:
 *      a. Shuffle nodes in S using the seeded PRNG (per-community shuffle).
 *      b. For each node v (in shuffle order):
 *         - If v is no longer in a singleton refined community
 *           (singleton check), skip.
 *         - If v is not well-connected to S\{v} (paper §3.5), skip.
 *         - Walk v's CSR row, accumulating per-refined-community edge weight
 *           restricted to neighbors in parent S.
 *         - For each candidate sub-community c' ⊂ S that v has edges to,
 *           compute ΔQ via Modularity.delta.
 *         - Filter to candidates with ΔQ > tolerance.
 *         - Sample c' probabilistically with weight ∝ exp(ΔQ × β),
 *           where β = 1 / randomness. Default randomness = 0.001 matches
 *           graspologic; effectively greedy with random tie-break.
 *         - Move v to chosen c'.
 *   3. Build parentOfRefined[r] = parent community of any node in refined r.
 *
 * Output property (paper Theorem 1): every refined community induces a
 * connected subgraph of `graph`.
 *
 * Move selection design — see ADR-0009 (probabilistic Boltzmann)
 * Well-connectedness criterion — see ADR-0010
 */

import type { Graph } from './graph.js';
import { Partition } from './partition.js';
import { Modularity } from './internal/quality.js';
import { Xoshiro128, DEFAULT_SEED } from './prng.js';

/** Tunables for `refine`. */
export interface RefineOptions {
  /** Resolution parameter γ; default 1.0. */
  readonly resolution?: number;
  /** PRNG seed; default {@link DEFAULT_SEED}. */
  readonly seed?: number;
  /**
   * Boltzmann temperature for probabilistic move selection.
   * Lower → more deterministic; higher → more uniform.
   * Default 0.001 (matches graspologic). With β = 1/0.001 = 1000, only
   * candidates whose ΔQ is within ~0.001 of the max have non-trivial
   * sampling probability — effectively greedy with random tie-break.
   */
  readonly randomness?: number;
  /** Relative ΔQ threshold for accepting a merge. Default Number.EPSILON × 16. */
  readonly tolerance?: number;
}

/** Returned by `refine`. */
export interface RefineResult {
  /**
   * The refined partition. Each refined community is a subset of exactly
   * one parent community.
   */
  readonly refined: Partition;
  /**
   * parentOfRefined[r] = parent community id of refined community r.
   * Used by aggregation (M4) to build the super-graph.
   * Empty refined communities (size 0 after merges) have parent = -1.
   */
  readonly parentOfRefined: Int32Array;
  /** Total successful merges across all parent communities. */
  readonly merges: number;
}

export function refine(
  graph: Graph,
  parent: Partition,
  options?: RefineOptions,
): RefineResult {
  if (parent.graph !== graph) {
    throw new RangeError('parent partition does not belong to graph');
  }

  const resolution = options?.resolution ?? 1.0;
  const seed = options?.seed ?? DEFAULT_SEED;
  const randomness = options?.randomness ?? 0.001;
  const tolerance = options?.tolerance ?? Number.EPSILON * 16;

  if (!Number.isFinite(resolution)) {
    throw new RangeError(`resolution must be finite; got ${String(resolution)}`);
  }
  if (!Number.isFinite(randomness) || randomness <= 0) {
    throw new RangeError(
      `randomness must be a positive finite number; got ${String(randomness)}`,
    );
  }
  if (!Number.isFinite(tolerance) || tolerance < 0) {
    throw new RangeError(
      `tolerance must be a non-negative finite number; got ${String(tolerance)}`,
    );
  }

  const refined = Partition.singletons(graph);
  const quality = new Modularity(resolution);
  const prng = new Xoshiro128(seed);
  const m2 = graph.totalWeight;
  const beta = 1 / randomness;
  const n = graph.n;

  const nodesByParent = bucketize(parent);

  // Two scratch areas reused across all node visits:
  //   touchedBuf / touchedWeight / touchedFlag — the per-CSR-walk accumulators
  //   candidateBuf / candidateValue              — filtered candidates + ΔQ then weights
  const touchedBuf = new Uint32Array(n);
  const touchedWeight = new Float64Array(n);
  const touchedFlag = new Uint8Array(n);
  const candidateBuf = new Uint32Array(n);
  const candidateValue = new Float64Array(n);

  let totalMerges = 0;

  for (let S = 0; S < parent.numCommunities; S++) {
    if ((parent.communitySize[S] as number) <= 1) continue;
    const nodes = nodesByParent[S];
    if (nodes === undefined || nodes.length === 0) continue;
    prng.shuffleUint32(nodes);
    const sumS = parent.communityIncident[S] as number;

    for (let visitIdx = 0; visitIdx < nodes.length; visitIdx++) {
      const v = nodes[visitIdx] as number;
      const vRefined = refined.assignments[v] as number;

      // Singleton check (paper §3 / ADR-0011): v's refined community must
      // still be a singleton. If an earlier-visited node merged with v,
      // skip.
      if ((refined.communitySize[vRefined] as number) !== 1) continue;

      // CSR walk restricted to neighbors with parent == S.
      const start = graph.offsets[v] as number;
      const end = graph.offsets[v + 1] as number;

      let touchedCount = 0;
      let edgesToS = 0;
      let selfLoopWeight = 0;

      for (let i = start; i < end; i++) {
        const u = graph.targets[i] as number;
        const w = graph.weights[i] as number;
        if (u === v) {
          selfLoopWeight += w;
          continue;
        }
        if ((parent.assignments[u] as number) !== S) continue;
        edgesToS += w;
        const c = refined.assignments[u] as number;
        if (touchedFlag[c] === 0) {
          touchedBuf[touchedCount] = c;
          touchedWeight[c] = w;
          touchedFlag[c] = 1;
          touchedCount++;
        } else {
          touchedWeight[c] = (touchedWeight[c] as number) + w;
        }
      }

      // Well-connectedness of v to S\{v} (paper §3.5):
      //   E({v}, S\{v}) ≥ γ · k_v · (||S|| − k_v) / 2m
      const kv = graph.nodeWeights[v] as number;
      const wellConnectedThreshold = (resolution * kv * (sumS - kv)) / m2;

      if (edgesToS < wellConnectedThreshold) {
        // Not well-connected: skip. Reset scratch.
        for (let i = 0; i < touchedCount; i++) {
          const c = touchedBuf[i] as number;
          touchedFlag[c] = 0;
          touchedWeight[c] = 0;
        }
        continue;
      }

      // Compute ΔQ for each touched community; filter to ΔQ > tolerance.
      let candidateCount = 0;
      let bestDelta = 0;

      for (let i = 0; i < touchedCount; i++) {
        const cand = touchedBuf[i] as number;
        if (cand === vRefined) continue; // staying gives ΔQ=0
        const kuToTo = touchedWeight[cand] as number;
        const dq = quality.delta({
          graph,
          partition: refined,
          u: v,
          fromCommunity: vRefined,
          toCommunity: cand,
          kuToFrom: 0,
          kuToTo,
          selfLoopWeight,
        });
        if (dq > tolerance * Math.max(Math.abs(bestDelta), 1)) {
          candidateBuf[candidateCount] = cand;
          candidateValue[candidateCount] = dq;
          candidateCount++;
          if (dq > bestDelta) bestDelta = dq;
        }
      }

      // Reset touched scratch.
      for (let i = 0; i < touchedCount; i++) {
        const c = touchedBuf[i] as number;
        touchedFlag[c] = 0;
        touchedWeight[c] = 0;
      }

      if (candidateCount === 0) continue;

      // Probabilistic Boltzmann selection (ADR-0009).
      // weight_i = exp((ΔQ_i − bestΔQ) × β); largest weight = exp(0) = 1.
      let weightSum = 0;
      for (let i = 0; i < candidateCount; i++) {
        const dq = candidateValue[i] as number;
        const w = Math.exp((dq - bestDelta) * beta);
        candidateValue[i] = w;
        weightSum += w;
      }

      let pick = candidateBuf[0] as number;
      if (candidateCount > 1) {
        const u = prng.nextFloat() * weightSum;
        let acc = 0;
        for (let i = 0; i < candidateCount; i++) {
          acc += candidateValue[i] as number;
          if (acc >= u) {
            pick = candidateBuf[i] as number;
            break;
          }
        }
      }

      refined.move(v, pick);
      totalMerges++;
    }
  }

  // Build parentOfRefined: parent community id of any node in refined r,
  // or -1 if refined r is empty (all members merged into other communities
  // and none were ever assigned to r in the singleton init — impossible
  // for ids < numCommunities, but safe to assert).
  const parentOfRefined = new Int32Array(refined.numCommunities).fill(-1);
  for (let u = 0; u < n; u++) {
    const r = refined.assignments[u] as number;
    const cur = parentOfRefined[r] as number;
    if (cur === -1) {
      parentOfRefined[r] = parent.assignments[u] as number;
    }
  }

  return {
    refined,
    parentOfRefined,
    merges: totalMerges,
  };
}

// -------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------

function bucketize(parent: Partition): Array<Uint32Array | undefined> {
  const k = parent.numCommunities;
  const counts = new Uint32Array(k);
  for (let u = 0; u < parent.assignments.length; u++) {
    const c = parent.assignments[u] as number;
    counts[c] = (counts[c] as number) + 1;
  }
  const buckets = new Array<Uint32Array | undefined>(k);
  for (let c = 0; c < k; c++) {
    const ct = counts[c] as number;
    if (ct > 0) buckets[c] = new Uint32Array(ct);
  }
  const cursor = new Uint32Array(k);
  for (let u = 0; u < parent.assignments.length; u++) {
    const c = parent.assignments[u] as number;
    const bucket = buckets[c];
    if (bucket !== undefined) {
      bucket[cursor[c] as number] = u;
      cursor[c] = (cursor[c] as number) + 1;
    }
  }
  return buckets;
}
