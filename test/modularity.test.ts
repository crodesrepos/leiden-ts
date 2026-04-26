import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

import { Graph } from '../src/graph.js';
import { modularity } from '../src/modularity.js';
import type { Edge } from '../src/types.js';
import { clique, twoTriangles } from './helpers.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

describe('modularity — basic identities', () => {
  it('Q = 0 on the empty graph for any partition', () => {
    const g = Graph.fromEdgeList(0, []);
    const empty = new Int32Array(0);
    expect(modularity(g, empty)).toBe(0);
  });

  it('Q = 0 when all nodes are in one community', () => {
    const g = Graph.fromEdgeList(3, [
      [0, 1],
      [1, 2],
    ]);
    const allOne = new Int32Array([0, 0, 0]);
    expect(modularity(g, allOne)).toBe(0);
  });

  it('Q ≈ 0.5 for two disjoint triangles partitioned correctly', () => {
    // Two K3 components, partitioned by component.
    // Σᵢₙ for each community = 6 (3 edges × 2 directions); Σₜₒₜ = 6.
    // 2m = 12. Q = Σ_c (Σᵢₙ/2m − (Σₜₒₜ/2m)²) = 2 × (6/12 − (6/12)²)
    //         = 2 × (0.5 − 0.25) = 0.5
    const g = twoTriangles();
    const partition = new Int32Array([0, 0, 0, 1, 1, 1]);
    expect(modularity(g, partition)).toBeCloseTo(0.5, 12);
  });

  it('Q for all-singletons on K_n is negative', () => {
    // Newman's identity: singleton partition gives Q = -Σ (kᵤ/2m)²
    // For K_n with unit edges: kᵤ = n-1, 2m = n(n-1).
    // So each term = ((n-1)/n(n-1))² = 1/n²; sum over n terms = 1/n.
    // Q = -1/n.
    for (const n of [3, 5, 10]) {
      const g = clique(n);
      const partition = new Int32Array(n);
      for (let i = 0; i < n; i++) partition[i] = i;
      const q = modularity(g, partition);
      expect(q).toBeCloseTo(-1 / n, 10);
    }
  });
});

describe('modularity — L1 cross-validation against networkx', () => {
  // Reads bench/compare/ref/outputs/karate-modularity.json (committed reference).
  // If our modularity() disagrees, our eq.(1) implementation is wrong.

  const fixturePath = resolve(
    __dirname,
    '../bench/compare/fixtures/karate.json',
  );
  const refPath = resolve(
    __dirname,
    '../bench/compare/ref/outputs/karate-modularity.json',
  );

  type KarateFixture = {
    n: number;
    edges: ReadonlyArray<[number, number]>;
    ground_truth_2_partition_assignments: number[];
  };

  type ModularityRef = {
    partitions: {
      ground_truth_2_partition: { Q: number; 'Q_resolution_2.0': number };
      all_singletons: { Q: number; 'Q_resolution_2.0': number };
      all_one_community: { Q: number };
    };
  };

  const karate = JSON.parse(readFileSync(fixturePath, 'utf8')) as KarateFixture;
  const ref = JSON.parse(readFileSync(refPath, 'utf8')) as ModularityRef;
  const graph = Graph.fromEdgeList(karate.n, karate.edges as Edge[]);

  it('matches networkx Q for the ground-truth 2-partition (resolution=1.0)', () => {
    const partition = Int32Array.from(
      karate.ground_truth_2_partition_assignments,
    );
    const Q = modularity(graph, partition);
    expect(Q).toBeCloseTo(ref.partitions.ground_truth_2_partition.Q, 10);
  });

  it('matches networkx Q for the ground-truth 2-partition at resolution=2.0', () => {
    const partition = Int32Array.from(
      karate.ground_truth_2_partition_assignments,
    );
    const Q = modularity(graph, partition, { resolution: 2.0 });
    expect(Q).toBeCloseTo(
      ref.partitions.ground_truth_2_partition['Q_resolution_2.0'],
      10,
    );
  });

  it('matches networkx Q for the all-singletons partition', () => {
    const partition = new Int32Array(karate.n);
    for (let i = 0; i < karate.n; i++) partition[i] = i;
    const Q = modularity(graph, partition);
    expect(Q).toBeCloseTo(ref.partitions.all_singletons.Q, 10);
  });

  it('matches networkx Q for the all-singletons partition at resolution=2.0', () => {
    const partition = new Int32Array(karate.n);
    for (let i = 0; i < karate.n; i++) partition[i] = i;
    const Q = modularity(graph, partition, { resolution: 2.0 });
    expect(Q).toBeCloseTo(ref.partitions.all_singletons['Q_resolution_2.0'], 10);
  });

  it('matches networkx Q for the all-one-community partition (= 0)', () => {
    const partition = new Int32Array(karate.n);
    const Q = modularity(graph, partition);
    expect(Q).toBeCloseTo(ref.partitions.all_one_community.Q, 12);
    expect(Q).toBe(0);
  });
});

describe('modularity — input validation', () => {
  const g = clique(3);

  it('throws RangeError when partition length mismatches graph.n', () => {
    expect(() => modularity(g, new Int32Array(2))).toThrowError(RangeError);
    expect(() => modularity(g, new Int32Array(4))).toThrowError(RangeError);
  });

  it('throws RangeError on non-finite resolution', () => {
    const partition = new Int32Array(3);
    expect(() =>
      modularity(g, partition, { resolution: Infinity }),
    ).toThrowError(RangeError);
    expect(() => modularity(g, partition, { resolution: NaN })).toThrowError(
      RangeError,
    );
  });

  it('accepts negative resolution (paper allows γ ≤ 0; produces unusual partitions)', () => {
    const partition = new Int32Array([0, 0, 0]);
    expect(() => modularity(g, partition, { resolution: -1 })).not.toThrow();
  });

  it('accepts non-contiguous community ids', () => {
    // Community ids of -5 and 1000 should produce the same Q as 0 and 1.
    const g2 = twoTriangles();
    const contiguous = new Int32Array([0, 0, 0, 1, 1, 1]);
    const sparse = new Int32Array([-5, -5, -5, 1000, 1000, 1000]);
    expect(modularity(g2, sparse)).toBeCloseTo(modularity(g2, contiguous), 12);
  });
});

describe('modularity — self-loop semantics (Newman/Traag §3.4)', () => {
  // Hand-derived expected value, paper convention:
  //   A(u, u) = 2w for an undirected self-loop of weight w.
  //   k_u    = Σ_j A(u, j)  (so a self-loop contributes 2w to k_u)
  //
  // Test graph (3 nodes):
  //   self-loop on 0 with weight 2
  //   edge 0-1 with weight 1
  //   edge 1-2 with weight 1
  //
  // Adjacency: A(0,0)=4, A(0,1)=A(1,0)=1, A(1,2)=A(2,1)=1, others 0.
  // Degrees: k_0 = 4+1 = 5,  k_1 = 1+1 = 2,  k_2 = 1.
  // 2m = Σ k = 8.
  //
  // Partition {0} | {1, 2} (community A = {0}, B = {1, 2}):
  //   Σᵢₙ(A) = A(0,0)             = 4
  //   Σᵢₙ(B) = A(1,1)+A(1,2)+A(2,1)+A(2,2) = 0+1+1+0 = 2
  //   Σₜₒₜ(A) = k_0 = 5
  //   Σₜₒₜ(B) = k_1 + k_2 = 3
  //   Q = (4/8 - (5/8)²) + (2/8 - (3/8)²)
  //     = (0.5 - 0.390625) + (0.25 - 0.140625)
  //     = 0.109375 + 0.109375
  //     = 0.21875

  const buildSelfLoopGraph = () =>
    Graph.fromEdgeList(
      3,
      [
        [0, 0, 2],
        [0, 1, 1],
        [1, 2, 1],
      ] as Edge[],
      { selfLoops: 'allow' },
    );

  it('weighted degree of self-loop node = 2w + Σ other weights (paper §3.4)', () => {
    const g = buildSelfLoopGraph();
    expect(g.weightedDegree(0)).toBe(5);
    expect(g.weightedDegree(1)).toBe(2);
    expect(g.weightedDegree(2)).toBe(1);
    expect(g.totalWeight).toBe(8);
  });

  it('Q = 0.21875 for partition {0} | {1,2} — derived from §3.4 formula', () => {
    const g = buildSelfLoopGraph();
    const partition = new Int32Array([0, 1, 1]);
    const Q = modularity(g, partition);
    expect(Q).toBeCloseTo(0.21875, 12);
  });

  it('Q for the all-one-community partition is exactly 0 even with self-loops', () => {
    // Identity that must hold regardless of self-loops: trivial partition Q=0.
    const g = buildSelfLoopGraph();
    const partition = new Int32Array([0, 0, 0]);
    expect(modularity(g, partition)).toBeCloseTo(0, 12);
  });

  it('Q for all-singletons accounts for the self-loop staying in its own community', () => {
    // With a self-loop on node 0, community {0} has Σᵢₙ = A(0,0) = 4
    // (the loop edge is in community 0 with both endpoints).
    //   contribution from {0}: 4/8 - (5/8)² = 32/64 - 25/64 = 7/64
    //   contribution from {1}: 0   - (2/8)² = -4/64
    //   contribution from {2}: 0   - (1/8)² = -1/64
    // Q = (7 - 4 - 1) / 64 = 2/64 = 0.03125
    //
    // Note: the no-self-loop closed form -Σ(k/2m)² does NOT apply here; that
    // simplification assumes Σᵢₙ(singleton) = 0 for every singleton, which
    // self-loops violate.
    const g = buildSelfLoopGraph();
    const partition = new Int32Array([0, 1, 2]);
    expect(modularity(g, partition)).toBeCloseTo(2 / 64, 12);
  });
});
