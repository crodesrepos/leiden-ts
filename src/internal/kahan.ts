/**
 * Kahan compensated summation.
 *
 * Naïve summation of many small floating-point quantities loses precision
 * because each addition rounds the running sum to fit in a double.
 * Kahan tracks a "compensation" term that captures the discarded
 * low-order bits and re-injects them on the next addition.
 *
 * Cost: ~3× a naïve sum. Benefit: bounded error independent of n.
 *
 * Used by modularity computation, where a 10⁶-edge graph would otherwise
 * accumulate enough error to violate the [-0.5, 1) bound.
 */

/**
 * Compute Σ values using Kahan compensation. Single-pass.
 *
 * @param values  the array to sum (TypedArray or plain array of numbers)
 * @returns       the compensated sum
 */
export function kahanSum(values: ArrayLike<number>): number {
  let sum = 0;
  let c = 0;
  const n = values.length;
  for (let i = 0; i < n; i++) {
    const y = (values[i] as number) - c;
    const t = sum + y;
    c = t - sum - y;
    sum = t;
  }
  return sum;
}

/**
 * Mutable accumulator for incremental Kahan summation.
 *
 * Use when values arrive one at a time and you need the running total
 * (e.g., per-community contributions during modularity computation,
 * where iteration order is dictated by graph traversal, not array order).
 *
 * Cheap to construct; reuse across iterations by calling reset().
 */
export class KahanAccumulator {
  private _sum = 0;
  private _c = 0;

  /** Add a single value to the running total. */
  add(value: number): void {
    const y = value - this._c;
    const t = this._sum + y;
    this._c = t - this._sum - y;
    this._sum = t;
  }

  /** The current compensated sum. */
  get value(): number {
    return this._sum;
  }

  /** Reset to zero so the accumulator can be reused. */
  reset(): void {
    this._sum = 0;
    this._c = 0;
  }
}
