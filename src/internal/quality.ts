/**
 * Internal Quality function abstraction.
 *
 * In M2 we ship one implementation: Modularity (Newman & Girvan 2004 /
 * Reichardt-Bornholdt 2006 with γ resolution).
 *
 * The shape is designed so that adding CPM (Constant Potts Model) in
 * M6 is a drop-in addition without refactoring the local-moving loop.
 *
 * NOT exported from the package barrel. Promote to public surface when
 * additional quality functions (CPM, RBConfiguration, Surprise, custom)
 * ship.
 */

import type { Graph } from '../graph.js';
import type { Partition } from '../partition.js';

/**
 * A Quality interface specifies how to score a partition and how to
 * compute the *delta* in the score for a single-node move.
 *
 * The delta computation is the load-bearing performance primitive: it
 * is called once per neighboring community per node visited per pass,
 * which dominates the local-moving runtime.
 */
export interface QualityFunction {
  /** Scoring kind for diagnostic / logging. */
  readonly kind: 'modularity' | 'cpm';

  /** Resolution parameter γ; default 1.0. */
  readonly resolution: number;

  /**
   * Compute the quality of the current partition. O(numCommunities)
   * when the partition's caches are valid.
   */
  compute(graph: Graph, partition: Partition): number;

  /**
   * Compute the change in quality if node `u` were moved from its current
   * community to `toCommunity`. Used by local-moving's hot loop.
   *
   * Inputs the caller has already gathered (avoids duplicate CSR walks):
   *   - kuToFrom: Σ A(u, v) for v in u's current community, v ≠ u
   *   - kuToTo:   Σ A(u, v) for v in toCommunity, v ≠ u
   *   - selfLoopWeight: A(u, u) / 2 (the raw loop weight, not the doubled one)
   *
   * @returns ΔQ for the proposed move (positive = improvement)
   */
  delta(args: {
    graph: Graph;
    partition: Partition;
    u: number;
    fromCommunity: number;
    toCommunity: number;
    kuToFrom: number;
    kuToTo: number;
    selfLoopWeight: number;
  }): number;
}

/**
 * Modularity quality function (Newman-Girvan 2004 with R-B resolution).
 *
 *   Q = Σ_c [Σᵢₙ(c)/2m − γ (Σₜₒₜ(c)/2m)²]
 *
 * Move-delta derivation (paper §B): moving u from c₁ to c₂ where c₁ ≠ c₂
 *
 *   ΔQ = (1/2m) · [
 *     2·kᵤ→c₂ − 2·kᵤ→c₁
 *     + 2·A(u,u) (only if A(u,u) > 0; affects c₁ if u is in c₁, c₂ if in c₂)
 *   ]
 *   + γ · (1 / (2m)²) · [
 *     kᵤ · (Σₜₒₜ(c₁) − kᵤ)  // c₁ loses u; reduces (Σₜₒₜ(c₁) − kᵤ)² penalty
 *     − kᵤ · Σₜₒₜ(c₂)         // c₂ gains u; increases Σₜₒₜ(c₂) penalty
 *   ]
 *
 * Combined and simplified with the self-loop convention (A(u,u) = 2w):
 *
 *   ΔQ = (1/m) · (kuToTo + selfLoopWeight − kuToFrom − selfLoopWeight)
 *      + (γ / 2m²) · (kᵤ · (Σₜₒₜ(c₁) − kᵤ) − kᵤ · Σₜₒₜ(c₂))
 *
 * Which simplifies further: the self-loop terms cancel because moving u
 * out of c₁ removes one full A(u,u)=2w from Σᵢₙ(c₁), and into c₂ adds
 * one full 2w to Σᵢₙ(c₂). The "internal" change due to self-loops is
 * symmetric and the kuToFrom/kuToTo terms below cover only neighbor
 * (non-self) edges. We add the self-loop contribution explicitly.
 */
export class Modularity implements QualityFunction {
  readonly kind = 'modularity' as const;
  readonly resolution: number;

  constructor(resolution = 1.0) {
    if (!Number.isFinite(resolution)) {
      throw new RangeError(`resolution must be finite; got ${String(resolution)}`);
    }
    this.resolution = resolution;
  }

  compute(_graph: Graph, partition: Partition): number {
    return partition.modularity(this.resolution);
  }

  delta(args: {
    graph: Graph;
    partition: Partition;
    u: number;
    fromCommunity: number;
    toCommunity: number;
    kuToFrom: number;
    kuToTo: number;
    selfLoopWeight: number;
  }): number {
    const { graph, partition, u, fromCommunity, toCommunity, kuToFrom, kuToTo, selfLoopWeight } = args;

    if (fromCommunity === toCommunity) return 0;

    const m2 = graph.totalWeight; // 2m
    if (m2 === 0) return 0;

    const ku = graph.nodeWeights[u] as number;
    const totFrom = partition.communityIncident[fromCommunity] as number;
    const totTo = partition.communityIncident[toCommunity] as number;

    // Internal-weight delta:
    //   removing u from c₁: −2 · kuToFrom − 2 · selfLoopWeight
    //   adding   u to   c₂: +2 · kuToTo   + 2 · selfLoopWeight
    //   net contribution to Q from Σᵢₙ:
    //     ΔΣᵢₙ_total / 2m = (2·kuToTo − 2·kuToFrom + 2·sl_to − 2·sl_from) / 2m
    //   Self-loop appears in BOTH source and dest (always with u, since u is
    //   in exactly one community at a time). The change cancels out in the
    //   delta — algebraically: −2sl + 2sl = 0. So self-loops contribute
    //   nothing to ΔQ_internal. (Confirmed numerically by tests.)
    const internalDelta = (2 * (kuToTo - kuToFrom)) / m2;
    void selfLoopWeight; // confirmed to cancel; arg retained for symmetry / docs

    // Σₜₒₜ(c)² delta, divided by (2m)²:
    //   c₁ loses kᵤ → its Σₜₒₜ becomes (totFrom − kᵤ)
    //   c₂ gains kᵤ → its Σₜₒₜ becomes (totTo + kᵤ)
    //   ΔΣ((Σₜₒₜ/2m)²) = ((totFrom−kᵤ)/2m)² − (totFrom/2m)² + ((totTo+kᵤ)/2m)² − (totTo/2m)²
    //   Expanded:
    //     = (totFrom² − 2·totFrom·kᵤ + kᵤ² − totFrom²
    //        + totTo² + 2·totTo·kᵤ + kᵤ² − totTo²) / (2m)²
    //     = (2·kᵤ · (totTo − totFrom) + 2·kᵤ²) / (2m)²
    //     = 2·kᵤ · (totTo − totFrom + kᵤ) / (2m)²
    //
    //   Penalty term in Q is −γ·Σ(Σₜₒₜ/2m)², so:
    //     ΔQ_penalty = −γ · 2·kᵤ · (totTo − totFrom + kᵤ) / (2m)²
    const penaltyDelta =
      (-this.resolution * 2 * ku * (totTo - totFrom + ku)) / (m2 * m2);

    return internalDelta + penaltyDelta;
  }
}
