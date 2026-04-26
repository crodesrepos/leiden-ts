/**
 * Cat-3 (Property) tests using fast-check.
 *
 * Each test asserts an invariant that must hold for *every* valid input,
 * not just one. A failure here means the invariant is violated, which is
 * always a real bug — never relax the assertion to make a flaky test pass.
 *
 * See `docs/test-philosophy.md` for the full taxonomy.
 */

import { describe, expect, it } from 'vitest';
import fc from 'fast-check';

import { Graph } from '../src/graph.js';
import { modularity } from '../src/modularity.js';
import type { Edge } from '../src/types.js';

// -----------------------------------------------------------------------
// Generators
// -----------------------------------------------------------------------

/**
 * A small undirected weighted graph: 2..20 nodes, edges chosen from the
 * canonical (lo, hi) pair space with no duplicates and no self-loops.
 *
 * Returns the (n, edges, partitionSpaces) bundle so tests can derive
 * partitions whose community count is bounded by node count.
 */
const arbGraph = fc
  .integer({ min: 2, max: 20 })
  .chain((n) => {
    // All possible undirected edges (i, j) with i < j.
    const allPairs: [number, number][] = [];
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) allPairs.push([i, j]);
    }
    const maxEdges = allPairs.length;
    return fc.tuple(
      fc.constant(n),
      fc
        .array(fc.integer({ min: 0, max: maxEdges - 1 }), {
          minLength: 0,
          maxLength: Math.min(maxEdges, 60),
        })
        .map((indices) => {
          // dedupe and produce edge tuples with weight in [0.1, 5]
          const seen = new Set<number>();
          const out: [number, number, number][] = [];
          for (const idx of indices) {
            if (seen.has(idx)) continue;
            seen.add(idx);
            const pair = allPairs[idx]!;
            // Use a deterministic weight derived from the index so the
            // shrinker can reduce graphs without churning the weights.
            const w = 0.1 + ((idx * 13) % 50) / 10;
            out.push([pair[0], pair[1], w]);
          }
          return out;
        }),
    );
  })
  .map(([n, edges]) => ({ n, edges: edges as Edge[] }));

/**
 * A partition: assignments[i] ∈ [0, k) for some k ≤ n.
 */
const arbPartition = (n: number) =>
  fc.integer({ min: 1, max: n }).chain((k) =>
    fc
      .array(fc.integer({ min: 0, max: k - 1 }), {
        minLength: n,
        maxLength: n,
      })
      .map((arr) => Int32Array.from(arr)),
  );

// -----------------------------------------------------------------------
// Identities and bounds
// -----------------------------------------------------------------------

describe('Property — modularity bounds', () => {
  it('Q ∈ [-1, 1] for any (graph, partition) — strict bound is [-0.5, 1) but allow margin for resolution variations', () => {
    fc.assert(
      fc.property(arbGraph, (g) => {
        const graph = Graph.fromEdgeList(g.n, g.edges);
        return fc.assert(
          fc.property(arbPartition(g.n), (partition) => {
            const Q = modularity(graph, partition);
            // Tight paper bound is [-0.5, 1). Use slightly relaxed bound to
            // accommodate floating-point edge cases at exactly -0.5 / 1.
            return Q >= -0.5 - 1e-12 && Q < 1 + 1e-12;
          }),
        );
      }),
      { numRuns: 100, seed: 0xcaffe },
    );
  });
});

describe('Property — identity: all-one-community ⇒ Q = 0', () => {
  it('any graph with edges ⇒ Q for the all-one partition is exactly 0', () => {
    fc.assert(
      fc.property(arbGraph, (g) => {
        if (g.edges.length === 0) return true; // empty graph trivially Q=0
        const graph = Graph.fromEdgeList(g.n, g.edges);
        const partition = new Int32Array(g.n); // all zeros
        const Q = modularity(graph, partition);
        return Math.abs(Q) < 1e-12;
      }),
      { numRuns: 100, seed: 0xbeef },
    );
  });
});

describe('Property — identity: all-singletons ⇒ Q = -Σ(k/2m)²', () => {
  it('all-singletons partition matches the closed-form formula', () => {
    fc.assert(
      fc.property(arbGraph, (g) => {
        if (g.edges.length === 0) return true;
        const graph = Graph.fromEdgeList(g.n, g.edges);
        const partition = new Int32Array(g.n);
        for (let i = 0; i < g.n; i++) partition[i] = i;

        const Q = modularity(graph, partition);

        // closed form: -Σ_u (k_u / 2m)²
        let expected = 0;
        const inv2m = 1 / graph.totalWeight;
        for (let u = 0; u < graph.n; u++) {
          const r = (graph.nodeWeights[u] as number) * inv2m;
          expected -= r * r;
        }

        return Math.abs(Q - expected) < 1e-10;
      }),
      { numRuns: 100, seed: 0x1ee7 },
    );
  });
});

// -----------------------------------------------------------------------
// Invariances
// -----------------------------------------------------------------------

describe('Property — invariance: community-id renumbering', () => {
  it('renumbering community ids does not change Q', () => {
    fc.assert(
      fc.property(arbGraph, (g) => {
        if (g.edges.length === 0) return true;
        const graph = Graph.fromEdgeList(g.n, g.edges);
        return fc.assert(
          fc.property(
            arbPartition(g.n),
            fc.integer({ min: 1, max: 1_000_000 }),
            (partition, shift) => {
              const renumbered = new Int32Array(partition.length);
              for (let i = 0; i < partition.length; i++) {
                renumbered[i] = (partition[i] as number) + shift;
              }
              const Q1 = modularity(graph, partition);
              const Q2 = modularity(graph, renumbered);
              return Math.abs(Q1 - Q2) < 1e-12;
            },
          ),
        );
      }),
      { numRuns: 50, seed: 0xc0de },
    );
  });
});

describe('Property — invariance: edge-list order does not change anything', () => {
  it('shuffling the edge list produces an identical graph (CSR symmetry, weighted degrees, modularity)', () => {
    fc.assert(
      fc.property(arbGraph, (g) => {
        if (g.edges.length === 0) return true;
        const original = Graph.fromEdgeList(g.n, g.edges);

        // Deterministic shuffle by index sort with derived key.
        const shuffled: Edge[] = g.edges
          .map((e, i) => ({ e, key: ((i * 0x9e3779b1) >>> 0) % 1_000_003 }))
          .sort((a, b) => a.key - b.key)
          .map((x) => x.e);
        const reconstructed = Graph.fromEdgeList(g.n, shuffled);

        // Same totalWeight (cheap check).
        if (original.totalWeight !== reconstructed.totalWeight) return false;
        // Same per-node weighted degrees.
        for (let u = 0; u < g.n; u++) {
          if (
            Math.abs(original.weightedDegree(u) - reconstructed.weightedDegree(u)) > 1e-12
          ) {
            return false;
          }
        }

        // Same modularity on the all-singletons partition (a structural fingerprint).
        const partition = new Int32Array(g.n);
        for (let i = 0; i < g.n; i++) partition[i] = i;
        const Q1 = modularity(original, partition);
        const Q2 = modularity(reconstructed, partition);
        return Math.abs(Q1 - Q2) < 1e-10;
      }),
      { numRuns: 50, seed: 0x123456 },
    );
  });
});

describe('Property — γ-derivative identity (Reichardt-Bornholdt 2006)', () => {
  it('Q(γ_a) - Q(γ_b) = (γ_b - γ_a) · Σ_c (Σ_tot(c) / 2m)²', () => {
    // The dependence of Q on the resolution parameter is linear with slope
    // -Σ_c (Σ_tot(c)/2m)². This is a strong, derivable invariant; if the
    // resolution parameter is wired in incorrectly anywhere, this fails.
    fc.assert(
      fc.property(arbGraph, (g) => {
        if (g.edges.length === 0) return true;
        const graph = Graph.fromEdgeList(g.n, g.edges);
        return fc.assert(
          fc.property(
            arbPartition(g.n),
            fc.double({ min: 0.1, max: 5, noNaN: true }),
            fc.double({ min: 0.1, max: 5, noNaN: true }),
            (partition, gA, gB) => {
              if (Math.abs(gA - gB) < 1e-3) return true; // skip near-identical
              const Qa = modularity(graph, partition, { resolution: gA });
              const Qb = modularity(graph, partition, { resolution: gB });

              // Compute Σ_c (Σ_tot(c) / 2m)² directly from partition.
              const totByComm = new Map<number, number>();
              for (let u = 0; u < g.n; u++) {
                const c = partition[u] as number;
                totByComm.set(
                  c,
                  (totByComm.get(c) ?? 0) + (graph.nodeWeights[u] as number),
                );
              }
              const inv2m = 1 / graph.totalWeight;
              let sumSq = 0;
              for (const tot of totByComm.values()) {
                const r = tot * inv2m;
                sumSq += r * r;
              }
              const predicted = (gB - gA) * sumSq;
              return Math.abs(Qa - Qb - predicted) < 1e-10;
            },
          ),
        );
      }),
      { numRuns: 50, seed: 0xfacade },
    );
  });
});
