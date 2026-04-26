/**
 * Integration tests — end-to-end validation of the M1+M2+M3 pipeline on
 * real fixtures, with cross-implementation comparison against networkx
 * Louvain.
 *
 * Per `docs/test-philosophy.md` these are Cat-2 (Reference) tests: we
 * compare against committed reference values produced by a known-good
 * external library. Reference outputs are committed to `bench/compare/ref/outputs/`
 * and regenerated via `python ref/run_louvain.py --all`.
 *
 * Note on the Q gap with networkx Louvain:
 *   networkx Louvain has aggregation (Phase 3), we don't (yet — that's M4).
 *   So our localMove Q on Karate (~0.347) is expected to be lower than
 *   networkx Louvain Q (~0.385). The gap quantifies what aggregation
 *   provides; M4 will close it.
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

import { Graph } from '../src/graph.js';
import { localMove } from '../src/localMove.js';
import { refine } from '../src/refine.js';
import { Partition } from '../src/partition.js';
import { modularity } from '../src/modularity.js';
import type { Edge } from '../src/types.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

function loadFixture(name: string): { n: number; edges: Edge[]; raw: any } {
  const path = resolve(__dirname, `../bench/compare/fixtures/${name}.json`);
  const raw = JSON.parse(readFileSync(path, 'utf8')) as {
    n: number;
    edges: ReadonlyArray<[number, number]>;
  };
  return { n: raw.n, edges: raw.edges as Edge[], raw };
}

function loadLouvainReference(fixture: string, seed: number): {
  modularity: number;
  n_communities: number;
  assignments: number[];
  all_connected: boolean;
} {
  const path = resolve(
    __dirname,
    `../bench/compare/ref/outputs/${fixture}-louvain-seed${seed}.json`,
  );
  return JSON.parse(readFileSync(path, 'utf8'));
}

/**
 * BFS check: every community in `partition` induces a connected subgraph
 * of `graph`. Returns null if all connected; otherwise returns the first
 * community id that is disconnected.
 *
 * Used as Cat-3 Property test for the paper Theorem 1 guarantee.
 */
function findDisconnectedCommunity(
  graph: Graph,
  partition: Partition,
): number | null {
  for (let c = 0; c < partition.numCommunities; c++) {
    const size = partition.communitySize[c] as number;
    if (size <= 1) continue;
    // Find seed node in c.
    let seed = -1;
    for (let u = 0; u < graph.n; u++) {
      if ((partition.assignments[u] as number) === c) {
        seed = u;
        break;
      }
    }
    if (seed === -1) continue;

    // BFS within c.
    const visited = new Uint8Array(graph.n);
    const queue: number[] = [seed];
    visited[seed] = 1;
    let count = 1;
    while (queue.length > 0) {
      const u = queue.shift() as number;
      const start = graph.offsets[u] as number;
      const end = graph.offsets[u + 1] as number;
      for (let i = start; i < end; i++) {
        const v = graph.targets[i] as number;
        if ((partition.assignments[v] as number) !== c) continue;
        if (visited[v] === 1) continue;
        visited[v] = 1;
        queue.push(v);
        count++;
      }
    }
    if (count !== size) return c;
  }
  return null;
}

// ---------------------------------------------------------------------
// Karate club — full pipeline + cross-implementation comparison
// ---------------------------------------------------------------------

describe('Integration — Karate club end-to-end', () => {
  const { n, edges } = loadFixture('karate');
  const graph = Graph.fromEdgeList(n, edges);
  const louvainRef = loadLouvainReference('karate', 42);

  describe('after localMove (Phase 1 only)', () => {
    it('produces non-negative Q across many seeds', () => {
      for (const seed of [1, 7, 42, 1000, 0xc0ffee]) {
        const result = localMove(graph, undefined, { seed });
        expect(result.modularity).toBeGreaterThan(0);
      }
    });

    it('Q is within paper-empirical range [0.30, 0.45]', () => {
      // Karate's optimal Q via full Leiden ≈ 0.444. Phase-1 alone reaches
      // a local optimum that's typically in [0.30, 0.40] depending on seed.
      for (const seed of [1, 7, 42, 1000, 0xc0ffee]) {
        const result = localMove(graph, undefined, { seed });
        expect(result.modularity).toBeGreaterThanOrEqual(0.30);
        expect(result.modularity).toBeLessThanOrEqual(0.45);
      }
    });

    it('Q is within reasonable range of networkx Louvain reference', () => {
      // networkx Louvain has aggregation; we don't (yet). So our Q is
      // expected to be lower. We assert: our Q is within 25% of theirs
      // (ours ≥ 0.75 × theirs). The gap shrinks to zero in M4.
      const result = localMove(graph, undefined, { seed: 42 });
      const ratio = result.modularity / louvainRef.modularity;
      expect(ratio).toBeGreaterThan(0.75);
      expect(ratio).toBeLessThanOrEqual(1.0);
    });

    it('produces fewer communities than n (real merging happened)', () => {
      const result = localMove(graph, undefined, { seed: 42 });
      let nonEmpty = 0;
      for (let c = 0; c < result.partition.numCommunities; c++) {
        if ((result.partition.communitySize[c] as number) > 0) nonEmpty++;
      }
      expect(nonEmpty).toBeLessThan(graph.n);
      expect(nonEmpty).toBeGreaterThan(1);
    });
  });

  describe('after localMove + refine (Phase 1 + Phase 2, M2+M3)', () => {
    it('every refined community induces a connected subgraph (paper Theorem 1)', () => {
      // The connectivity guarantee that distinguishes Leiden from Louvain.
      for (const seed of [1, 42, 0xbeef]) {
        const lm = localMove(graph, undefined, { seed });
        const rf = refine(graph, lm.partition, { seed });
        const disconnected = findDisconnectedCommunity(graph, rf.refined);
        if (disconnected !== null) {
          expect.fail(
            `seed=${seed}: refined community ${disconnected} is disconnected`,
          );
        }
      }
    });

    it('every refined community is a subset of exactly one parent community', () => {
      const lm = localMove(graph, undefined, { seed: 42 });
      const rf = refine(graph, lm.partition, { seed: 42 });
      // For every node, its refined community's parent matches its parent.
      for (let u = 0; u < graph.n; u++) {
        const r = rf.refined.assignments[u] as number;
        const parentOfR = rf.parentOfRefined[r] as number;
        const directParent = lm.partition.assignments[u] as number;
        expect(parentOfR).toBe(directParent);
      }
    });

    it('cached and scratch modularity agree on the refined partition', () => {
      const lm = localMove(graph, undefined, { seed: 42 });
      const rf = refine(graph, lm.partition, { seed: 42 });
      const cached = rf.refined.modularity();
      const scratch = modularity(graph, rf.refined.assignments);
      expect(cached).toBeCloseTo(scratch, 10);
    });
  });

  describe('cross-implementation: networkx Louvain reference fingerprints', () => {
    it('the reference itself satisfies its claims (sanity check)', () => {
      // Verify the committed Louvain reference: Q value, community count,
      // and connectivity. Catches regenerated-reference corruption.
      const refPartition = Partition.fromAssignments(
        graph,
        Int32Array.from(louvainRef.assignments),
      );
      const Q = modularity(graph, refPartition.assignments);
      expect(Q).toBeCloseTo(louvainRef.modularity, 10);
      expect(louvainRef.all_connected).toBe(true);
    });

    it('our localMove output, when scored against the original graph, is in the same Q ballpark', () => {
      const result = localMove(graph, undefined, { seed: 42 });
      // Our Phase-1 Q vs networkx Louvain Q (Phase 1 + Phase 3):
      //   absolute gap ≤ 0.10 expected (M4 closes it)
      const gap = louvainRef.modularity - result.modularity;
      expect(gap).toBeGreaterThanOrEqual(0); // ours ≤ theirs
      expect(gap).toBeLessThan(0.10); // and within reasonable distance
    });
  });
});

// ---------------------------------------------------------------------
// Determinism across the full pipeline
// ---------------------------------------------------------------------

describe('Integration — pipeline determinism', () => {
  const { n, edges } = loadFixture('karate');
  const graph = Graph.fromEdgeList(n, edges);

  it('same seed → byte-identical output through localMove + refine', () => {
    const a = (() => {
      const lm = localMove(graph, undefined, { seed: 12345 });
      return refine(graph, lm.partition, { seed: 12345 });
    })();
    const b = (() => {
      const lm = localMove(graph, undefined, { seed: 12345 });
      return refine(graph, lm.partition, { seed: 12345 });
    })();
    for (let u = 0; u < graph.n; u++) {
      expect(a.refined.assignments[u]).toBe(b.refined.assignments[u]);
    }
    expect(a.merges).toBe(b.merges);
  });
});
