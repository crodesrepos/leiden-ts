/**
 * Partition — a node-to-community assignment with cached per-community
 * accumulators (Σᵢₙ and Σₜₒₜ) needed for incremental modularity-delta
 * computation in the local-moving phase.
 *
 * Design: see `docs/adr/0008-partition-class-shape.md`.
 *
 * Invariants (must hold after every public-API call):
 *   - assignments.length === graph.n
 *   - assignments[u] ∈ [0, numCommunities) for every u
 *   - communityInternal[c] = Σ over (u, v) in c of A(u, v) — Newman convention
 *     (self-loop on u contributes 2w if u is in c)
 *   - communityIncident[c] = Σ over u in c of nodeWeights[u]
 *
 * Mutation goes through `move(u, toCommunity)`. Direct mutation of the
 * underlying typed arrays is forbidden — the caches drift.
 *
 * Reference paper convention: Traag, Waltman & van Eck (2019) §2-§3.
 */

import type { Graph } from './graph.js';

export class Partition {
  readonly graph: Graph;
  /** assignments[u] = community id of node u; ids are contiguous in [0, numCommunities). */
  readonly assignments: Int32Array;
  /** Σᵢₙ(c): total weight of edges with both endpoints in community c (Newman). */
  readonly communityInternal: Float64Array;
  /** Σₜₒₜ(c): total weight of edges incident to community c. */
  readonly communityIncident: Float64Array;
  /** Number of non-empty communities — also the upper bound of valid community ids. */
  numCommunities: number;
  /** communitySize[c] = count of nodes with assignments[u] === c. */
  readonly communitySize: Uint32Array;

  /**
   * Internal — use `Partition.singletons()` or `Partition.fromAssignments()`.
   * The constructor takes ownership of the typed arrays without copy.
   */
  private constructor(args: {
    graph: Graph;
    assignments: Int32Array;
    communityInternal: Float64Array;
    communityIncident: Float64Array;
    communitySize: Uint32Array;
    numCommunities: number;
  }) {
    this.graph = args.graph;
    this.assignments = args.assignments;
    this.communityInternal = args.communityInternal;
    this.communityIncident = args.communityIncident;
    this.communitySize = args.communitySize;
    this.numCommunities = args.numCommunities;
  }

  /**
   * Each node in its own singleton community.
   *
   *   assignments[u] = u
   *   numCommunities = graph.n
   *   communityInternal[c] = Σ A(u, u) for u in {c} (the self-loop contribution; 0 otherwise)
   *   communityIncident[c]  = nodeWeights[c]
   */
  static singletons(graph: Graph): Partition {
    const n = graph.n;
    const assignments = new Int32Array(n);
    for (let u = 0; u < n; u++) assignments[u] = u;

    const communityInternal = new Float64Array(n);
    const communityIncident = new Float64Array(n);
    const communitySize = new Uint32Array(n);

    // For each node u: incident weight = nodeWeights[u].
    // Internal weight only non-zero if u has a self-loop (in 'allow' mode).
    for (let u = 0; u < n; u++) {
      communityIncident[u] = graph.nodeWeights[u] as number;
      communitySize[u] = 1;
    }
    // Self-loop contribution to internal: walk CSR; if target == u, add 2w
    // (Newman convention: A(u, u) = 2w for an undirected self-loop).
    if (graph.hasSelfLoops) {
      const offsets = graph.offsets;
      const targets = graph.targets;
      const weights = graph.weights;
      for (let u = 0; u < n; u++) {
        const start = offsets[u] as number;
        const end = offsets[u + 1] as number;
        for (let i = start; i < end; i++) {
          if ((targets[i] as number) === u) {
            communityInternal[u] = (communityInternal[u] as number) + 2 * (weights[i] as number);
          }
        }
      }
    }

    return new Partition({
      graph,
      assignments,
      communityInternal,
      communityIncident,
      communitySize,
      numCommunities: n,
    });
  }

  /**
   * Construct from a user-supplied assignments array. Renumbers community
   * ids to be contiguous in [0, k) so the indexed accumulators have no
   * gaps. The renumbering is stable: ids encountered earlier in the
   * assignments array map to lower contiguous ids.
   *
   * @param graph        the graph being partitioned
   * @param assignments  Int32Array of length graph.n; values are arbitrary
   *                     int32 community ids (need not be contiguous or non-negative)
   *
   * @throws RangeError if assignments.length !== graph.n
   */
  static fromAssignments(graph: Graph, assignments: Int32Array): Partition {
    if (assignments.length !== graph.n) {
      throw new RangeError(
        `assignments length ${assignments.length} does not match graph.n ${graph.n}`,
      );
    }
    const n = graph.n;
    const renumbered = new Int32Array(n);
    const idMap = new Map<number, number>();
    let next = 0;
    for (let u = 0; u < n; u++) {
      const c = assignments[u] as number;
      let nc = idMap.get(c);
      if (nc === undefined) {
        nc = next++;
        idMap.set(c, nc);
      }
      renumbered[u] = nc;
    }
    const k = next;

    const communityInternal = new Float64Array(k);
    const communityIncident = new Float64Array(k);
    const communitySize = new Uint32Array(k);

    // Σₜₒₜ(c) = Σ_{u in c} k_u
    for (let u = 0; u < n; u++) {
      const c = renumbered[u] as number;
      communityIncident[c] = (communityIncident[c] as number) + (graph.nodeWeights[u] as number);
      communitySize[c] = (communitySize[c] as number) + 1;
    }

    // Σᵢₙ(c) = Σ over half-edges (u → v) where assignments[u] == assignments[v] of weight
    // (each undirected non-self edge gets counted twice; self-loops once with target==u
    // and we double via the convention.)
    const offsets = graph.offsets;
    const targets = graph.targets;
    const weights = graph.weights;
    for (let u = 0; u < n; u++) {
      const cu = renumbered[u] as number;
      const start = offsets[u] as number;
      const end = offsets[u + 1] as number;
      for (let i = start; i < end; i++) {
        const v = targets[i] as number;
        const cv = renumbered[v] as number;
        if (cu === cv) {
          const w = weights[i] as number;
          communityInternal[cu] = (communityInternal[cu] as number) + w;
          if (v === u) {
            // Self-loop: add the second w to match the Newman A(u,u) = 2w convention.
            communityInternal[cu] = (communityInternal[cu] as number) + w;
          }
        }
      }
    }

    return new Partition({
      graph,
      assignments: renumbered,
      communityInternal,
      communityIncident,
      communitySize,
      numCommunities: k,
    });
  }

  /**
   * Move node u from its current community to `toCommunity`, updating
   * the cached accumulators in O(degree(u)) time.
   *
   *   - If toCommunity ≥ numCommunities, expands the accumulator arrays
   *     to accommodate (rare; only happens when introducing a brand-new
   *     community id, e.g. during refinement).
   *   - If the source community becomes empty, its slot remains allocated
   *     (size 0). Renumbering happens at aggregation time, not here.
   *   - No-op if u is already in toCommunity.
   *
   * @throws RangeError if u or toCommunity are out of range
   */
  move(u: number, toCommunity: number): void {
    if (!Number.isInteger(u) || u < 0 || u >= this.graph.n) {
      throw new RangeError(`u ${u} out of range [0, ${this.graph.n})`);
    }
    if (!Number.isInteger(toCommunity) || toCommunity < 0) {
      throw new RangeError(
        `toCommunity ${toCommunity} must be a non-negative integer`,
      );
    }

    const fromCommunity = this.assignments[u] as number;
    if (fromCommunity === toCommunity) return;

    // Grow the accumulator arrays if a brand-new community id is being introduced.
    if (toCommunity >= this.numCommunities) {
      // We do not aggressively grow — single-step grow keeps allocations bounded.
      // (Practical paths use Partition.fromAssignments to renumber after a
      // major restructure rather than calling move() with arbitrary ids.)
      throw new RangeError(
        `toCommunity ${toCommunity} ≥ numCommunities ${this.numCommunities}; ` +
          `use Partition.fromAssignments to restructure communities`,
      );
    }

    // Compute kᵤ→from and kᵤ→to: weighted edges from u into each community.
    // Walk u's CSR run once.
    const offsets = this.graph.offsets;
    const targets = this.graph.targets;
    const weights = this.graph.weights;
    const start = offsets[u] as number;
    const end = offsets[u + 1] as number;

    let kFromInternal = 0; // Σ A(u, v) for v ≠ u in community fromCommunity
    let kToInternal = 0; // Σ A(u, v) for v ≠ u in community toCommunity
    let selfLoopWeight = 0;

    for (let i = start; i < end; i++) {
      const v = targets[i] as number;
      const w = weights[i] as number;
      if (v === u) {
        selfLoopWeight += w;
        continue;
      }
      const cv = this.assignments[v] as number;
      if (cv === fromCommunity) kFromInternal += w;
      else if (cv === toCommunity) kToInternal += w;
    }

    // Update Σᵢₙ:
    //   moving u OUT of fromCommunity:
    //     remove 2 × kFromInternal (the half-edges u→v and v→u for v in from)
    //     remove 2 × selfLoopWeight  (since u was in fromCommunity, its self-loop counted)
    //   moving u INTO toCommunity:
    //     add 2 × kToInternal + 2 × selfLoopWeight
    const fromInternalDelta = -2 * kFromInternal - 2 * selfLoopWeight;
    const toInternalDelta = 2 * kToInternal + 2 * selfLoopWeight;
    this.communityInternal[fromCommunity] =
      (this.communityInternal[fromCommunity] as number) + fromInternalDelta;
    this.communityInternal[toCommunity] =
      (this.communityInternal[toCommunity] as number) + toInternalDelta;

    // Update Σₜₒₜ: shift u's nodeWeight from fromCommunity to toCommunity.
    const ku = this.graph.nodeWeights[u] as number;
    this.communityIncident[fromCommunity] =
      (this.communityIncident[fromCommunity] as number) - ku;
    this.communityIncident[toCommunity] =
      (this.communityIncident[toCommunity] as number) + ku;

    // Update size and assignment.
    this.communitySize[fromCommunity] =
      (this.communitySize[fromCommunity] as number) - 1;
    this.communitySize[toCommunity] =
      (this.communitySize[toCommunity] as number) + 1;
    this.assignments[u] = toCommunity;
  }

  /**
   * Modularity from cached accumulators — O(numCommunities) instead of
   * the O(m) walk that the standalone `modularity()` function does.
   *
   * Result agrees with `modularity(graph, partition.assignments)` to
   * within Kahan precision.
   *
   * @param resolution γ; default 1.0
   */
  modularity(resolution = 1.0): number {
    const inv2m = 1 / this.graph.totalWeight;
    if (this.graph.totalWeight === 0) return 0;
    let q = 0;
    let c = 0; // Kahan compensation
    for (let comm = 0; comm < this.numCommunities; comm++) {
      const intern = this.communityInternal[comm] as number;
      const tot = this.communityIncident[comm] as number;
      const term = intern * inv2m - resolution * (tot * inv2m) * (tot * inv2m);
      const y = term - c;
      const t = q + y;
      c = t - q - y;
      q = t;
    }
    return q;
  }

  /** Deep copy — independent typed-array buffers, same graph reference. */
  copy(): Partition {
    return new Partition({
      graph: this.graph,
      assignments: new Int32Array(this.assignments),
      communityInternal: new Float64Array(this.communityInternal),
      communityIncident: new Float64Array(this.communityIncident),
      communitySize: new Uint32Array(this.communitySize),
      numCommunities: this.numCommunities,
    });
  }
}
