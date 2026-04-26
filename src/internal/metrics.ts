/**
 * Cluster-comparison metrics: NMI (Normalized Mutual Information) and
 * ARI (Adjusted Rand Index).
 *
 * Used for L2 cross-validation against external Leiden implementations
 * (graspologic, igraph). Both metrics are invariant to community-id
 * permutation: partitions {0,0,1,1} and {1,1,0,0} score 1.0.
 *
 * Pure TypeScript; no dependency on a stats library.
 *
 * References:
 *   Strehl, A. & Ghosh, J. (2002). Cluster ensembles. JMLR, 3, 583-617.
 *   Hubert, L. & Arabie, P. (1985). Comparing partitions. J. Classif., 2.
 */

/**
 * Build the contingency table N[i][j] = count of nodes in label-A community
 * i AND label-B community j. Sparse representation: returns Map keyed by
 * (i × kb + j).
 */
function contingencyTable(
  a: Int32Array,
  b: Int32Array,
): { table: Map<number, number>; ka: number; kb: number; n: number } {
  if (a.length !== b.length) {
    throw new RangeError(
      `partitions must be the same length; got a.length=${a.length}, b.length=${b.length}`,
    );
  }
  const n = a.length;
  // Renumber both arrays to dense [0, k) so we can size kb safely.
  const ka = denseRenumber(a);
  const kb = denseRenumber(b);
  const table = new Map<number, number>();
  for (let i = 0; i < n; i++) {
    const ai = a[i] as number;
    const bi = b[i] as number;
    const key = ai * kb + bi;
    table.set(key, (table.get(key) ?? 0) + 1);
  }
  return { table, ka, kb, n };
}

/**
 * In-place dense renumbering of a partition. Returns the number of distinct
 * labels (= largest dense label + 1). Stable order: labels are renumbered
 * in first-encounter order.
 */
function denseRenumber(p: Int32Array): number {
  const map = new Map<number, number>();
  let next = 0;
  for (let i = 0; i < p.length; i++) {
    const v = p[i] as number;
    let id = map.get(v);
    if (id === undefined) {
      id = next++;
      map.set(v, id);
    }
    p[i] = id;
  }
  return next;
}

/**
 * Mutual Information between two partitions, in nats.
 *
 *   I(A; B) = Σ_ij P(A=i, B=j) · log( P(A=i, B=j) / (P(A=i) P(B=j)) )
 */
function mutualInformation(
  table: Map<number, number>,
  ka: number,
  kb: number,
  n: number,
): number {
  // Marginal counts.
  const aCounts = new Float64Array(ka);
  const bCounts = new Float64Array(kb);
  for (const [key, count] of table) {
    const i = Math.floor(key / kb);
    const j = key % kb;
    aCounts[i] = (aCounts[i] as number) + count;
    bCounts[j] = (bCounts[j] as number) + count;
  }

  let mi = 0;
  for (const [key, count] of table) {
    if (count === 0) continue;
    const i = Math.floor(key / kb);
    const j = key % kb;
    const pij = count / n;
    const pi = (aCounts[i] as number) / n;
    const pj = (bCounts[j] as number) / n;
    if (pi === 0 || pj === 0) continue;
    mi += pij * Math.log(pij / (pi * pj));
  }
  return mi;
}

/**
 * Shannon entropy of a partition, in nats.
 *
 *   H(A) = − Σ_i P(A=i) · log( P(A=i) )
 */
function entropy(p: Int32Array): number {
  const counts = new Map<number, number>();
  for (let i = 0; i < p.length; i++) {
    const v = p[i] as number;
    counts.set(v, (counts.get(v) ?? 0) + 1);
  }
  const n = p.length;
  let h = 0;
  for (const c of counts.values()) {
    if (c === 0) continue;
    const px = c / n;
    h -= px * Math.log(px);
  }
  return h;
}

/**
 * Normalized Mutual Information between two partitions.
 *
 *   NMI(A, B) = I(A; B) / sqrt( H(A) · H(B) )
 *
 * Range: [0, 1]. NMI=1 when A and B are identical (modulo label permutation).
 * NMI=0 when A and B are statistically independent.
 *
 * Special cases (industry convention):
 *   - Both partitions trivial (all in one community): NMI = 1.
 *   - One trivial and one non-trivial: NMI = 0.
 */
export function nmi(a: Int32Array, b: Int32Array): number {
  if (a.length !== b.length) {
    throw new RangeError(
      `partitions must be the same length; got a.length=${a.length}, b.length=${b.length}`,
    );
  }
  if (a.length === 0) return 1;

  // Copy because contingencyTable's denseRenumber mutates.
  const aCopy = new Int32Array(a);
  const bCopy = new Int32Array(b);

  const ha = entropy(aCopy);
  const hb = entropy(bCopy);
  if (ha === 0 && hb === 0) return 1;
  if (ha === 0 || hb === 0) return 0;

  const { table, ka, kb, n } = contingencyTable(aCopy, bCopy);
  const mi = mutualInformation(table, ka, kb, n);
  return mi / Math.sqrt(ha * hb);
}

/**
 * Adjusted Rand Index between two partitions.
 *
 *   ARI = ( RI − E[RI] ) / ( max(RI) − E[RI] )
 *
 * Range: [-1, 1]. ARI=1 for identical partitions. ARI≈0 for random.
 * ARI corrects for chance, unlike raw Rand Index.
 *
 * Implementation follows Hubert & Arabie (1985) closed form:
 *
 *   sum_comb_C = Σ_ij C(n_ij, 2)              // pair-coincidence
 *   sum_comb_a = Σ_i  C(a_i, 2)               // pairs within rows
 *   sum_comb_b = Σ_j  C(b_j, 2)               // pairs within cols
 *   total      = C(n, 2)
 *
 *   ARI = ( sum_comb_C − sum_comb_a · sum_comb_b / total ) /
 *         ( 0.5 (sum_comb_a + sum_comb_b) − sum_comb_a · sum_comb_b / total )
 */
export function ari(a: Int32Array, b: Int32Array): number {
  if (a.length !== b.length) {
    throw new RangeError(
      `partitions must be the same length; got a.length=${a.length}, b.length=${b.length}`,
    );
  }
  const n = a.length;
  if (n < 2) return 1;

  const aCopy = new Int32Array(a);
  const bCopy = new Int32Array(b);

  const { table, ka, kb } = contingencyTable(aCopy, bCopy);

  // Marginal counts.
  const aCounts = new Float64Array(ka);
  const bCounts = new Float64Array(kb);
  for (const [key, count] of table) {
    const i = Math.floor(key / kb);
    const j = key % kb;
    aCounts[i] = (aCounts[i] as number) + count;
    bCounts[j] = (bCounts[j] as number) + count;
  }

  const choose2 = (x: number): number => (x * (x - 1)) / 2;

  let sumCombC = 0;
  for (const c of table.values()) sumCombC += choose2(c);
  let sumCombA = 0;
  for (let i = 0; i < ka; i++) sumCombA += choose2(aCounts[i] as number);
  let sumCombB = 0;
  for (let j = 0; j < kb; j++) sumCombB += choose2(bCounts[j] as number);

  const total = choose2(n);
  const expected = (sumCombA * sumCombB) / total;
  const numerator = sumCombC - expected;
  const denominator = 0.5 * (sumCombA + sumCombB) - expected;

  if (denominator === 0) {
    // Both partitions are trivial (single community); identical by definition.
    return numerator === 0 ? 1 : 0;
  }
  return numerator / denominator;
}
