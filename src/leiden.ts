/**
 * Public Leiden algorithm — the multi-level loop that ties the M2/M3/M4
 * primitives together (Traag, Waltman & van Eck 2019, Algorithm 1).
 *
 * Algorithm sketch:
 *   P ← initialPartition or singletons(graph)
 *   loop until convergence (maxLevels safety cap):
 *     P ← localMove(currentGraph, P)        // Phase 1: local optimization
 *     if P is all singletons: break          // no further compression
 *     refined, parentOfRefined ← refine(...) // Phase 2: connectivity
 *     agg ← aggregate(currentGraph, refined, parentOfRefined)
 *     // project original-node assignments through this level's refinement
 *     update originalToCurrent[u] for all u
 *     currentGraph ← agg.graph
 *     P ← agg.partition
 *
 *   // build final partition over the original graph
 *   return Partition.fromAssignments(originalGraph, projection of P)
 *
 * Public API: graspologic-shaped option names (resolution, seed, randomness)
 * for drop-in compatibility with `graspologic.partition.leiden`'s call site.
 *
 * See ADR-0012 (multi-level termination) and ADR-0013 (API shape).
 */

import { Graph } from './graph.js';
import { Partition } from './partition.js';
import { localMove } from './localMove.js';
import { refine } from './refine.js';
import { aggregate } from './aggregate.js';
import { DEFAULT_SEED } from './prng.js';

/** Tunables for `leiden`. */
export interface LeidenOptions {
  /** Resolution γ; default 1.0 (matches graspologic). */
  readonly resolution?: number;
  /** PRNG seed; default DEFAULT_SEED. */
  readonly seed?: number;
  /**
   * Boltzmann temperature for refinement move selection;
   * default 0.001 (matches graspologic).
   */
  readonly randomness?: number;
  /** Move-acceptance tolerance; default Number.EPSILON × 16. */
  readonly tolerance?: number;
  /**
   * Safety cap on multi-level iterations; default 100.
   * Throws ConvergenceError if exceeded — silent non-convergence is
   * forbidden (per ADR-0007 / ADR-0012).
   */
  readonly maxLevels?: number;
  /**
   * Maximum localMove iterations per level; default 1000. ER graphs
   * without strong community structure converge slowly; community-rich
   * graphs converge in 5–20 inner iterations.
   */
  readonly maxLocalMoveIterations?: number;
  /**
   * Warm-start partition over the original graph. Defaults to singletons.
   * Useful for incremental clustering (re-cluster after graph edits).
   * The supplied partition is COPIED before use; the original is not mutated.
   */
  readonly initialPartition?: Partition;
}

export interface LeidenResult {
  /** Final partition over the original graph. */
  readonly partition: Partition;
  /** Final modularity Q at convergence. */
  readonly modularity: number;
  /** Number of multi-level iterations executed before convergence. */
  readonly levels: number;
  /** Total moves accepted across all localMove calls (diagnostic). */
  readonly totalMoves: number;
  /** Total merges accepted across all refine calls (diagnostic). */
  readonly totalMerges: number;
}

/** Thrown when leiden() exceeds maxLevels without converging. */
export class LeidenConvergenceError extends Error {
  readonly levels: number;
  readonly lastModularity: number;
  constructor(levels: number, lastModularity: number) {
    super(
      `leiden did not converge after ${levels} levels (last Q=${lastModularity.toFixed(6)}); ` +
        `increase maxLevels or investigate the input graph`,
    );
    this.name = 'LeidenConvergenceError';
    this.levels = levels;
    this.lastModularity = lastModularity;
  }
}

/**
 * Run the Leiden algorithm. Returns a partition over the original graph's
 * nodes (the assignments correspond to nodes 0..graph.n-1).
 */
export function leiden(graph: Graph, options?: LeidenOptions): LeidenResult {
  const resolution = options?.resolution ?? 1.0;
  const seed = options?.seed ?? DEFAULT_SEED;
  const randomness = options?.randomness ?? 0.001;
  const tolerance = options?.tolerance ?? Number.EPSILON * 16;
  const maxLevels = options?.maxLevels ?? 100;
  const maxLocalMoveIterations = options?.maxLocalMoveIterations ?? 1000;

  if (!Number.isFinite(resolution)) {
    throw new RangeError(`resolution must be finite; got ${String(resolution)}`);
  }
  if (!Number.isFinite(randomness) || randomness <= 0) {
    throw new RangeError(
      `randomness must be a positive finite number; got ${String(randomness)}`,
    );
  }
  if (!Number.isInteger(maxLevels) || maxLevels < 1) {
    throw new RangeError(
      `maxLevels must be a positive integer; got ${String(maxLevels)}`,
    );
  }
  if (options?.initialPartition !== undefined && options.initialPartition.graph !== graph) {
    throw new RangeError('initialPartition does not belong to graph');
  }

  // Initial partition over the original graph.
  let currentGraph: Graph = graph;
  let currentPartition: Partition =
    options?.initialPartition !== undefined
      ? options.initialPartition.copy()
      : Partition.singletons(graph);

  // originalToCurrent[u] = current super-node id of original node u.
  // Updated after each aggregation. At level 0, identity (each node is its
  // own super-node).
  const originalToCurrent = new Int32Array(graph.n);
  for (let u = 0; u < graph.n; u++) originalToCurrent[u] = u;

  // Stable per-level seed: derive distinct seeds for localMove and refine
  // at each level so the streams don't collide.
  let levelSeed = seed | 0;

  let totalMoves = 0;
  let totalMerges = 0;
  let levels = 0;

  for (levels = 0; levels < maxLevels; levels++) {
    // Phase 1: local moving.
    const lmResult = localMove(currentGraph, currentPartition, {
      resolution,
      seed: levelSeed,
      tolerance,
      maxIterations: maxLocalMoveIterations,
    });
    totalMoves += lmResult.moves;
    currentPartition = lmResult.partition;

    // Termination per Traag 2019 Algorithm 1: stop when the partition
    // stops changing. Two equivalent triggers:
    //   (a) localMove made no moves — partition is at a fixed point;
    //       refine+aggregate would produce a same-sized super-graph
    //       carrying the same partition, so the next level cannot
    //       improve. Without this check the loop runs to maxLevels on
    //       graphs where localMove finds an immediate local optimum
    //       at level ≥ 1 (observed on LFR-1k μ=0.3, seed 0xface).
    //   (b) every community is a singleton — no further compression
    //       possible. Subsumed by (a) but kept explicit for clarity.
    if (lmResult.moves === 0 || isAllSingletons(currentPartition)) {
      levels++; // count this level
      break;
    }

    // Phase 2: refinement.
    const rfResult = refine(currentGraph, currentPartition, {
      resolution,
      seed: levelSeed ^ 0x55555555, // distinct stream from localMove
      randomness,
      tolerance,
    });
    totalMerges += rfResult.merges;

    // Phase 3: aggregation. The super-partition's assignments ARE the
    // parent partition projected onto super-graph nodes; Aggregation
    // handles this for us via parentOfRefined.
    const agg = aggregate(currentGraph, rfResult.refined, rfResult.parentOfRefined);

    // Project original-node tracking through this level's refinement.
    // For each original node u, its NEW super-node id is the dense id
    // assigned by aggregate to its previous super-node's refined community.
    // We don't have direct access to that mapping for arbitrary current-graph
    // nodes; aggregate returned superNodeOf for original-of-currentGraph.
    // Since currentGraph at this iteration's start is what aggregate operated
    // on, agg.superNodeOf maps currentGraph nodes → new super-graph nodes.
    //
    // So: for each original node u, its current super-graph id is
    // agg.superNodeOf[ originalToCurrent[u] ].
    for (let u = 0; u < graph.n; u++) {
      const cur = originalToCurrent[u] as number;
      originalToCurrent[u] = agg.superNodeOf[cur] as number;
    }

    // Advance to the next level.
    currentGraph = agg.graph;
    currentPartition = agg.partition;

    // Different seed for next level (cheap mix).
    levelSeed = (Math.imul(levelSeed, 0x9e3779b9) ^ levels) | 0;
  }

  if (levels >= maxLevels) {
    // Compute final Q on whatever we have, then throw.
    const finalQ = projectAndScore(graph, currentPartition, originalToCurrent, resolution);
    throw new LeidenConvergenceError(levels, finalQ);
  }

  // Build the final partition over the original graph: each original node u
  // gets the community id its current super-node has in currentPartition.
  const finalAssignments = new Int32Array(graph.n);
  for (let u = 0; u < graph.n; u++) {
    finalAssignments[u] = currentPartition.assignments[originalToCurrent[u] as number] as number;
  }
  const finalPartition = Partition.fromAssignments(graph, finalAssignments);

  return {
    partition: finalPartition,
    modularity: finalPartition.modularity(resolution),
    levels,
    totalMoves,
    totalMerges,
  };
}

// -------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------

function isAllSingletons(partition: Partition): boolean {
  for (let c = 0; c < partition.numCommunities; c++) {
    if ((partition.communitySize[c] as number) >= 2) return false;
  }
  return true;
}

function projectAndScore(
  graph: Graph,
  currentPartition: Partition,
  originalToCurrent: Int32Array,
  resolution: number,
): number {
  const finalAssignments = new Int32Array(graph.n);
  for (let u = 0; u < graph.n; u++) {
    finalAssignments[u] = currentPartition.assignments[originalToCurrent[u] as number] as number;
  }
  const finalPartition = Partition.fromAssignments(graph, finalAssignments);
  return finalPartition.modularity(resolution);
}
