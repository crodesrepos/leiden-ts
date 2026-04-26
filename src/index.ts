/**
 * leiden-ts — pure-TypeScript implementation of the Leiden community
 * detection algorithm (Traag, Waltman & van Eck, 2019).
 *
 * Milestone 2 surface: Graph + modularity + Partition + localMove + PRNG.
 * The full `leiden(graph, opts)` function (refinement + aggregation)
 * lands in M3+.
 */

export { Graph } from './graph.js';
export { modularity } from './modularity.js';
export { Partition } from './partition.js';
export { localMove, ConvergenceError } from './localMove.js';
export type { LocalMoveOptions, LocalMoveResult } from './localMove.js';
export { refine } from './refine.js';
export type { RefineOptions, RefineResult } from './refine.js';
export { aggregate } from './aggregate.js';
export type { AggregateResult } from './aggregate.js';
export { leiden, LeidenConvergenceError } from './leiden.js';
export type { LeidenOptions, LeidenResult } from './leiden.js';
export { Xoshiro128, DEFAULT_SEED } from './prng.js';
export {
  GraphValidationError,
  type Edge,
  type GraphOptions,
  type GraphValidationCode,
  type ModularityOptions,
} from './types.js';
