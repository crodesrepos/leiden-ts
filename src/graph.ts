/**
 * Compressed Sparse Row (CSR) representation of an undirected, weighted graph.
 *
 *   offsets[u]      : index into targets/weights of u's first half-edge
 *   offsets[u + 1]  : one past the last
 *   targets[i]      : neighbor at half-edge i
 *   weights[i]      : weight at half-edge i
 *
 * Each undirected edge {u, v} appears as two half-edges: one in u's run
 * with target v, one in v's run with target u, sharing the same weight.
 *
 * Self-loops (when allowed) appear once per loop in their owning node's run
 * with target == owner.
 *
 * Each node's CSR run is sorted by target id (ascending). This makes the
 * graph object canonical — invariant under input edge-list reordering and
 * stable for downstream sums (paper-exact within float-64 precision).
 *
 * See ADR-0001 for the layout rationale and decision record.
 */

import {
  GraphValidationError,
  type Edge,
  type GraphOptions,
} from './types.js';

export class Graph {
  readonly n: number;
  readonly m: number;
  readonly offsets: Uint32Array;
  readonly targets: Uint32Array;
  readonly weights: Float64Array;
  readonly nodeWeights: Float64Array;
  readonly totalWeight: number;
  readonly hasSelfLoops: boolean;

  /**
   * Internal — use {@link Graph.fromEdgeList} or {@link Graph.fromCSR}.
   *
   * The constructor takes ownership of the typed arrays and does no
   * validation. Factories are responsible for producing consistent inputs.
   */
  private constructor(args: {
    n: number;
    m: number;
    offsets: Uint32Array;
    targets: Uint32Array;
    weights: Float64Array;
    nodeWeights: Float64Array;
    totalWeight: number;
    hasSelfLoops: boolean;
  }) {
    this.n = args.n;
    this.m = args.m;
    this.offsets = args.offsets;
    this.targets = args.targets;
    this.weights = args.weights;
    this.nodeWeights = args.nodeWeights;
    this.totalWeight = args.totalWeight;
    this.hasSelfLoops = args.hasSelfLoops;
  }

  /**
   * Construct a graph from an edge list.
   *
   * @param nodeCount  number of nodes; node ids are in [0, nodeCount)
   * @param edges      array of [u, v] or [u, v, weight] tuples
   * @param options    {@link GraphOptions}; selfLoops defaults to 'reject'
   *
   * @throws GraphValidationError on invalid input (when validate is true)
   */
  static fromEdgeList(
    nodeCount: number,
    edges: ReadonlyArray<Edge>,
    options: GraphOptions = {},
  ): Graph {
    const validate = options.validate ?? true;
    const selfLoopMode = options.selfLoops ?? 'reject';

    if ((options as { directed?: unknown }).directed === true) {
      throw new GraphValidationError(
        'DIRECTED_NOT_SUPPORTED',
        'Directed graphs are not supported in this version.',
      );
    }
    if (!Number.isInteger(nodeCount) || nodeCount < 0) {
      throw new GraphValidationError(
        'INVALID_NODE_COUNT',
        `nodeCount must be a non-negative integer; got ${String(nodeCount)}`,
      );
    }

    // -----------------------------------------------------------------
    // Pass 1: validate, filter self-loops if collapsing, count degrees.
    // -----------------------------------------------------------------
    const degree = new Uint32Array(nodeCount);
    let mUnique = 0;
    let hasSelfLoops = false;

    const seen = validate ? new Array<Set<number> | null>(nodeCount) : null;
    const kept: Edge[] = [];

    for (let i = 0; i < edges.length; i++) {
      const edge = edges[i] as Edge;
      const u = edge[0];
      const v = edge[1];
      const w = edge[2] ?? 1.0;

      if (validate) {
        if (!Number.isInteger(u) || u < 0 || u >= nodeCount) {
          throw new GraphValidationError(
            'NODE_OUT_OF_RANGE',
            `Edge endpoint u=${String(u)} out of range [0, ${nodeCount})`,
            edge,
          );
        }
        if (!Number.isInteger(v) || v < 0 || v >= nodeCount) {
          throw new GraphValidationError(
            'NODE_OUT_OF_RANGE',
            `Edge endpoint v=${String(v)} out of range [0, ${nodeCount})`,
            edge,
          );
        }
        if (!Number.isFinite(w)) {
          throw new GraphValidationError(
            'NOT_FINITE',
            `Edge weight ${String(w)} is not finite`,
            edge,
          );
        }
        if (w < 0) {
          throw new GraphValidationError(
            'NEGATIVE_WEIGHT',
            `Edge weight ${w} is negative`,
            edge,
          );
        }
      }

      const isSelfLoop = u === v;
      if (isSelfLoop) {
        if (selfLoopMode === 'reject') {
          if (validate) {
            throw new GraphValidationError(
              'SELF_LOOP',
              `Self-loop at node ${u}; pass selfLoops:'allow' or 'collapse' to permit`,
              edge,
            );
          }
          continue;
        }
        if (selfLoopMode === 'collapse') {
          continue;
        }
        hasSelfLoops = true;
      }

      if (validate && seen !== null) {
        const lo = u < v ? u : v;
        const hi = u < v ? v : u;
        let bucket = seen[lo];
        if (bucket === undefined || bucket === null) {
          bucket = new Set<number>();
          seen[lo] = bucket;
        }
        if (bucket.has(hi)) {
          throw new GraphValidationError(
            'DUPLICATE_EDGE',
            `Duplicate undirected edge (${u}, ${v})`,
            edge,
          );
        }
        bucket.add(hi);
      }

      kept.push([u, v, w]);
      degree[u] = (degree[u] as number) + 1;
      if (!isSelfLoop) degree[v] = (degree[v] as number) + 1;
      mUnique++;
    }

    // -----------------------------------------------------------------
    // Pass 2: build CSR offsets prefix-sum, then write edges.
    // -----------------------------------------------------------------
    const offsets = new Uint32Array(nodeCount + 1);
    let running = 0;
    for (let u = 0; u < nodeCount; u++) {
      offsets[u] = running;
      running += degree[u] as number;
    }
    offsets[nodeCount] = running;

    const halfEdgeCount = running;
    const targets = new Uint32Array(halfEdgeCount);
    const weights = new Float64Array(halfEdgeCount);
    const cursor = new Uint32Array(nodeCount);

    for (let i = 0; i < kept.length; i++) {
      const edge = kept[i] as Edge;
      const u = edge[0];
      const v = edge[1];
      const w = edge[2] as number;

      const uPos = (offsets[u] as number) + (cursor[u] as number);
      targets[uPos] = v;
      weights[uPos] = w;
      cursor[u] = (cursor[u] as number) + 1;

      if (u !== v) {
        const vPos = (offsets[v] as number) + (cursor[v] as number);
        targets[vPos] = u;
        weights[vPos] = w;
        cursor[v] = (cursor[v] as number) + 1;
      }
    }

    // Canonicalize each node's CSR run by target id. Makes the constructed
    // graph independent of input edge-list order — a property tested in
    // `test/properties.test.ts` (Cat-3). O(m log m) once at construction.
    sortRunsByTarget(offsets, targets, weights, nodeCount);

    // -----------------------------------------------------------------
    // Node weights and total weight.
    //
    // Newman convention (Traag 2019 §3.4): a self-loop on u with weight w
    // contributes 2w to nodeWeights[u]. CSR stores it once with target == u
    // and weight w; we add the second w during the sum below.
    // -----------------------------------------------------------------
    const nodeWeights = new Float64Array(nodeCount);
    for (let u = 0; u < nodeCount; u++) {
      const start = offsets[u] as number;
      const end = offsets[u + 1] as number;
      let sum = 0;
      for (let i = start; i < end; i++) {
        const w = weights[i] as number;
        sum += w;
        if ((targets[i] as number) === u) sum += w;
      }
      nodeWeights[u] = sum;
    }

    let totalWeight = 0;
    for (let u = 0; u < nodeCount; u++) totalWeight += nodeWeights[u] as number;

    return new Graph({
      n: nodeCount,
      m: mUnique,
      offsets,
      targets,
      weights,
      nodeWeights,
      totalWeight,
      hasSelfLoops,
    });
  }

  /**
   * Construct directly from prepared CSR arrays.
   *
   * Skips all validation. Caller asserts:
   *   - offsets is monotonically non-decreasing
   *   - offsets.length === n + 1
   *   - targets and weights have length === offsets[n]
   *   - undirected symmetry: every (u → v, w) has a matching (v → u, w)
   *     except when u === v (self-loops appear once)
   *   - all weights finite and non-negative
   *
   * The Graph takes ownership of the supplied typed arrays and does
   * not copy them. Mutating the inputs after this call is undefined
   * behaviour.
   */
  static fromCSR(args: {
    n: number;
    offsets: Uint32Array;
    targets: Uint32Array;
    weights: Float64Array;
  }): Graph {
    const { n, offsets, targets, weights } = args;

    const nodeWeights = new Float64Array(n);
    let m = 0;
    let hasSelfLoops = false;
    let totalWeight = 0;

    for (let u = 0; u < n; u++) {
      const start = offsets[u] as number;
      const end = offsets[u + 1] as number;
      let sum = 0;
      for (let i = start; i < end; i++) {
        const w = weights[i] as number;
        const t = targets[i] as number;
        sum += w;
        if (t === u) {
          sum += w;
          hasSelfLoops = true;
          m++;
        } else if (t > u) {
          m++;
        }
      }
      nodeWeights[u] = sum;
      totalWeight += sum;
    }

    return new Graph({
      n,
      m,
      offsets,
      targets,
      weights,
      nodeWeights,
      totalWeight,
      hasSelfLoops,
    });
  }

  /** Number of neighbors of u (unweighted; self-loops counted once). */
  degree(u: number): number {
    return (this.offsets[u + 1] as number) - (this.offsets[u] as number);
  }

  /** Sum of edge weights incident to u; alias for nodeWeights[u]. */
  weightedDegree(u: number): number {
    return this.nodeWeights[u] as number;
  }
}

/**
 * In-place insertion-sort of each node's CSR run by target id.
 * Insertion sort is fastest for small runs (typical graph degrees).
 * For very dense graphs (avg degree > ~30) a typed-array radix sort
 * would win; not worth optimizing for M1.
 */
function sortRunsByTarget(
  offsets: Uint32Array,
  targets: Uint32Array,
  weights: Float64Array,
  n: number,
): void {
  for (let u = 0; u < n; u++) {
    const start = offsets[u] as number;
    const end = offsets[u + 1] as number;
    for (let i = start + 1; i < end; i++) {
      const tKey = targets[i] as number;
      const wKey = weights[i] as number;
      let j = i - 1;
      while (j >= start && (targets[j] as number) > tKey) {
        targets[j + 1] = targets[j] as number;
        weights[j + 1] = weights[j] as number;
        j--;
      }
      targets[j + 1] = tKey;
      weights[j + 1] = wKey;
    }
  }
}
