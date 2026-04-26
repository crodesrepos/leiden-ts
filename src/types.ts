/**
 * Public types for leiden-ts.
 *
 * The library has a deliberately small public surface. Types in this file
 * are exported from the package barrel; everything else is internal.
 */

/**
 * An undirected edge.
 *
 *   [u, v]      — unit weight (1.0)
 *   [u, v, w]   — explicit weight w (must be finite and non-negative)
 *
 * Tuple form is chosen over an object form for allocation cost and to keep
 * the public API surface minimal. See ADR-0002.
 */
export type Edge = readonly [u: number, v: number, weight?: number];

/**
 * Options for {@link Graph.fromEdgeList}.
 */
export interface GraphOptions {
  /**
   * Currently always undirected. Reserved for a future directed-graph
   * variant; passing true is rejected at construction.
   */
  readonly directed?: false;

  /**
   * How to handle self-loops in the input edge list:
   *
   *   - `'reject'`   (default): throw `GraphValidationError` on any self-loop
   *   - `'allow'`:    keep self-loops; they contribute to nodeWeights and
   *                   modularity per Traag 2019 §3.4
   *   - `'collapse'`: silently drop self-loops at construction time
   *
   * See ADR-0003.
   */
  readonly selfLoops?: 'reject' | 'allow' | 'collapse';

  /**
   * When true (default), edge list is validated at construction time:
   *   - all node ids are in [0, n)
   *   - no duplicate undirected edges
   *   - all weights are finite and non-negative
   *   - self-loops are handled per the {@link selfLoops} setting
   *
   * Set to false ONLY if the input is already known to be valid (e.g.,
   * produced by a trusted upstream). Skipping validation is unsafe; an
   * invalid graph silently produces incorrect modularity values.
   */
  readonly validate?: boolean;
}

/**
 * Options for {@link modularity}.
 */
export interface ModularityOptions {
  /**
   * Resolution parameter γ. Default 1.0 (paper standard).
   *
   * Higher γ → smaller, denser communities.
   * Lower γ  → fewer, larger communities.
   *
   * Out-of-range or non-finite values throw RangeError.
   */
  readonly resolution?: number;
}

/**
 * Discriminator for {@link GraphValidationError}.
 */
export type GraphValidationCode =
  | 'NODE_OUT_OF_RANGE'
  | 'SELF_LOOP'
  | 'DUPLICATE_EDGE'
  | 'NEGATIVE_WEIGHT'
  | 'NOT_FINITE'
  | 'INVALID_NODE_COUNT'
  | 'DIRECTED_NOT_SUPPORTED';

/**
 * Thrown when an edge list violates {@link Graph} invariants.
 *
 * The {@link code} field is a stable string discriminator suitable for
 * programmatic handling. The {@link edge} field, when present, is the
 * specific edge that triggered the failure.
 */
export class GraphValidationError extends Error {
  readonly code: GraphValidationCode;
  readonly edge?: Edge;

  constructor(code: GraphValidationCode, message: string, edge?: Edge) {
    super(message);
    this.name = 'GraphValidationError';
    this.code = code;
    if (edge !== undefined) this.edge = edge;
  }
}
