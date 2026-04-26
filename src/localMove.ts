/**
 * Local-moving phase of the Leiden algorithm (Traag, Waltman & van Eck 2019,
 * Algorithm 2). M2 implementation: fixed-order pass over a seeded shuffle,
 * not the queue+re-enqueue variant from Traag 2015 — see ADR-0007.
 *
 * Iterate:
 *   For each node u in shuffled order:
 *     compute the neighboring community c* maximizing ΔQ(u → c*)
 *     if ΔQ > tolerance: move u to c*; mark "improved"
 *   if no node moved: converged.
 *
 * Tie-breaking when multiple communities give equal ΔQ:
 *   draw uniformly at random among the tied candidates via the seeded PRNG
 *   (ADR-0006). Deterministic given seed; unbiased across communities.
 *
 * Returns a `LocalMoveResult` with the converged partition, statistics,
 * and the final modularity. Throws `ConvergenceError` if `maxIterations`
 * is exceeded — silent non-convergence is forbidden (ADR's M2 risk
 * mitigation).
 */

import type { Graph } from './graph.js';
import { Partition } from './partition.js';
import { Modularity, type QualityFunction } from './internal/quality.js';
import { Xoshiro128, DEFAULT_SEED } from './prng.js';

/** Tunables for `localMove`. */
export interface LocalMoveOptions {
  /** Resolution parameter γ; default 1.0 (paper standard). */
  readonly resolution?: number;
  /** PRNG seed; default {@link DEFAULT_SEED} (= 42). */
  readonly seed?: number;
  /**
   * Minimum relative ΔQ for a move to be accepted.
   *
   *   reject if ΔQ / max(|Q|, 1) ≤ tolerance
   *
   * Default `Number.EPSILON × 16` (~ 3.5e-15) — scale-aware; rejects
   * only true rounding noise without leaving real improvements on the
   * table.
   */
  readonly tolerance?: number;
  /**
   * Maximum number of full passes over all nodes. Throws
   * {@link ConvergenceError} if exceeded. Default 100.
   *
   * Real graphs converge in 5–20 passes; 100 is a safety cap, not a
   * practical limit.
   */
  readonly maxIterations?: number;
  /**
   * Custom quality function. Defaults to modularity at the given
   * resolution. Public surface for this is M6+; it is internal in M2.
   */
  readonly quality?: QualityFunction;
}

/** Returned by `localMove`. */
export interface LocalMoveResult {
  /** Converged partition (also the input partition, mutated in-place). */
  readonly partition: Partition;
  /** Number of full passes executed before convergence. */
  readonly iterations: number;
  /** Final modularity Q at convergence. */
  readonly modularity: number;
  /** Total number of node moves accepted across all iterations. */
  readonly moves: number;
}

/** Thrown when `localMove` fails to converge within `maxIterations`. */
export class ConvergenceError extends Error {
  readonly iterations: number;
  readonly lastModularity: number;
  constructor(iterations: number, lastModularity: number) {
    super(
      `localMove did not converge after ${iterations} iterations ` +
        `(last Q=${lastModularity.toFixed(6)}); ` +
        `increase maxIterations or investigate the input graph`,
    );
    this.name = 'ConvergenceError';
    this.iterations = iterations;
    this.lastModularity = lastModularity;
  }
}

/**
 * Run the local-moving phase. Returns the converged partition.
 *
 * If `initial` is omitted, every node starts in its own community
 * (singletons — canonical Leiden initialization).
 *
 * The `initial` partition is mutated in place. If you need to preserve
 * it, pass `initial.copy()`.
 */
export function localMove(
  graph: Graph,
  initial?: Partition,
  options?: LocalMoveOptions,
): LocalMoveResult {
  const resolution = options?.resolution ?? 1.0;
  const seed = options?.seed ?? DEFAULT_SEED;
  const tolerance = options?.tolerance ?? Number.EPSILON * 16;
  const maxIterations = options?.maxIterations ?? 100;
  const quality: QualityFunction =
    options?.quality ?? new Modularity(resolution);

  if (!Number.isFinite(resolution)) {
    throw new RangeError(`resolution must be finite; got ${String(resolution)}`);
  }
  if (!Number.isFinite(tolerance) || tolerance < 0) {
    throw new RangeError(
      `tolerance must be a non-negative finite number; got ${String(tolerance)}`,
    );
  }
  if (!Number.isInteger(maxIterations) || maxIterations < 1) {
    throw new RangeError(
      `maxIterations must be a positive integer; got ${String(maxIterations)}`,
    );
  }

  const partition = initial ?? Partition.singletons(graph);

  // Initial visit order: shuffled identity permutation.
  const order = new Uint32Array(graph.n);
  for (let u = 0; u < graph.n; u++) order[u] = u;
  const prng = new Xoshiro128(seed);
  prng.shuffleUint32(order);

  // Scratch buffers reused across iterations to avoid allocation in the
  // hot loop:
  //   - neighborCommBuf: distinct neighboring community ids touched while scanning u
  //   - neighborCommWeight: kuToCommunity for each touched community
  //     (sparse — only the touched indices have meaningful values)
  //   - communityTouched: 1 if neighborCommWeight[c] was set this iteration
  //   - tiedBuf: scratch for tracking tied-best candidates during ΔQ scan
  const cap = Math.max(graph.n, 8);
  const neighborCommBuf = new Uint32Array(cap);
  const neighborCommWeight = new Float64Array(cap);
  const communityTouched = new Uint8Array(cap);
  const tiedBuf = new Uint32Array(cap);

  let totalMoves = 0;
  let iterations = 0;

  for (iterations = 1; iterations <= maxIterations; iterations++) {
    let movesThisPass = 0;

    for (let visitIdx = 0; visitIdx < order.length; visitIdx++) {
      const u = order[visitIdx] as number;
      const fromCommunity = partition.assignments[u] as number;

      // Walk u's CSR run once; collect:
      //   - kuToCommunity[c]: weighted edges from u to community c (excluding self-loop)
      //   - selfLoopWeight: u's self-loop weight (0 if no self-loop)
      const offsets = graph.offsets;
      const targets = graph.targets;
      const weights = graph.weights;
      const start = offsets[u] as number;
      const end = offsets[u + 1] as number;

      let touchedCount = 0;
      let selfLoopWeight = 0;
      // Always include u's current community as a candidate (the
      // "stay" option). Initialize its kuToFrom = 0; will be filled in
      // by the neighbor walk.
      neighborCommBuf[touchedCount] = fromCommunity;
      neighborCommWeight[fromCommunity] = 0;
      communityTouched[fromCommunity] = 1;
      touchedCount++;

      for (let i = start; i < end; i++) {
        const v = targets[i] as number;
        const w = weights[i] as number;
        if (v === u) {
          selfLoopWeight += w;
          continue;
        }
        const cv = partition.assignments[v] as number;
        if (communityTouched[cv] === 0) {
          neighborCommBuf[touchedCount] = cv;
          neighborCommWeight[cv] = w;
          communityTouched[cv] = 1;
          touchedCount++;
        } else {
          neighborCommWeight[cv] = (neighborCommWeight[cv] as number) + w;
        }
      }

      const kuToFrom = neighborCommWeight[fromCommunity] as number;

      // Find the community with maximum ΔQ. Track ties for unbiased
      // random tie-breaking (ADR-0006).
      let bestDelta = 0; // staying in fromCommunity has ΔQ = 0 by definition
      let tiedCount = 1;
      tiedBuf[0] = fromCommunity;

      for (let i = 0; i < touchedCount; i++) {
        const cand = neighborCommBuf[i] as number;
        if (cand === fromCommunity) continue;
        const kuToCand = neighborCommWeight[cand] as number;
        const dq = quality.delta({
          graph,
          partition,
          u,
          fromCommunity,
          toCommunity: cand,
          kuToFrom,
          kuToTo: kuToCand,
          selfLoopWeight,
        });
        if (dq > bestDelta + tolerance * Math.max(Math.abs(bestDelta), 1)) {
          bestDelta = dq;
          tiedCount = 1;
          tiedBuf[0] = cand;
        } else if (
          Math.abs(dq - bestDelta) <=
          tolerance * Math.max(Math.abs(bestDelta), 1)
        ) {
          tiedBuf[tiedCount] = cand;
          tiedCount++;
        }
      }

      // Reset the touched buffer ONLY at indices we actually set —
      // O(touchedCount), not O(numCommunities).
      for (let i = 0; i < touchedCount; i++) {
        const c = neighborCommBuf[i] as number;
        communityTouched[c] = 0;
        neighborCommWeight[c] = 0;
      }

      // Decide: move or stay.
      // Stay if best Δ is non-positive (within tolerance) — i.e. the
      // "stay" option (which has ΔQ = 0 and is in the tied set) is at
      // least as good as any alternative.
      if (bestDelta <= tolerance * Math.max(Math.abs(bestDelta), 1)) {
        continue; // no move
      }

      // Pick a tied candidate uniformly at random (excluding fromCommunity,
      // which has ΔQ = 0 and is only in tiedBuf when bestDelta = 0).
      // When bestDelta > 0 the tied set contains only candidates with that
      // Δ; fromCommunity is not among them.
      const pick =
        tiedCount === 1 ? (tiedBuf[0] as number) : (tiedBuf[prng.nextIntExclusive(tiedCount)] as number);

      partition.move(u, pick);
      movesThisPass++;
      totalMoves++;
    }

    if (movesThisPass === 0) break;
  }

  if (iterations > maxIterations) {
    throw new ConvergenceError(maxIterations, partition.modularity(resolution));
  }

  return {
    partition,
    iterations,
    modularity: partition.modularity(resolution),
    moves: totalMoves,
  };
}
