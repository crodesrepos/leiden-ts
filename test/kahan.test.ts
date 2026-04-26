import { describe, expect, it } from 'vitest';
import { kahanSum, KahanAccumulator } from '../src/internal/kahan.js';

describe('kahanSum', () => {
  it('matches naive sum for short, well-conditioned inputs', () => {
    const values = [1, 2, 3, 4, 5];
    expect(kahanSum(values)).toBe(15);
  });

  it('returns 0 for an empty input', () => {
    expect(kahanSum([])).toBe(0);
    expect(kahanSum(new Float64Array(0))).toBe(0);
  });

  it('handles a single value', () => {
    expect(kahanSum([42])).toBe(42);
  });

  it('beats naive sum on the classic 1.0 + 1e-20 × N drift case', () => {
    // Adding 1e-20 to 1.0 a billion times should yield 1.0 + 1e-11 ideally.
    // Naive double summation discards every increment, returning exactly 1.0.
    // Kahan recovers most of the precision.
    const N = 1_000_000;
    const values = new Float64Array(N + 1);
    values[0] = 1.0;
    for (let i = 1; i <= N; i++) values[i] = 1e-15;

    let naive = 0;
    for (let i = 0; i < values.length; i++) naive += values[i] as number;

    const kahan = kahanSum(values);
    const expected = 1 + N * 1e-15; // 1 + 1e-9

    // Naive is dominated by 1.0 and loses the small contributions.
    expect(Math.abs(naive - expected)).toBeGreaterThan(1e-12);
    // Kahan should be within rounding distance of expected.
    expect(Math.abs(kahan - expected)).toBeLessThan(1e-13);
  });

  it('works on Float64Array as well as plain Array', () => {
    const arr = [0.1, 0.2, 0.3, 0.4, 0.5];
    const typed = Float64Array.from(arr);
    expect(kahanSum(arr)).toBeCloseTo(1.5, 14);
    expect(kahanSum(typed)).toBe(kahanSum(arr));
  });

  it('handles negative values and signed cancellation in well-conditioned cases', () => {
    expect(kahanSum([1, -1, 1, -1, 1, -1])).toBe(0);
    // Kahan helps when many small values are added to a running sum, not
    // pure catastrophic cancellation between two near-equal large values
    // — that requires double-double or higher techniques (Knuth, Vol. 2).
    // Verify the well-conditioned case instead.
    expect(kahanSum([1e10, 1, -1e10])).toBeCloseTo(1, 10);
  });
});

describe('KahanAccumulator', () => {
  it('starts at zero', () => {
    const k = new KahanAccumulator();
    expect(k.value).toBe(0);
  });

  it('accumulates equivalently to kahanSum', () => {
    const values = [0.1, 0.2, 0.3, 0.4, 0.5, 1e-16, -1e-16];
    const k = new KahanAccumulator();
    for (const v of values) k.add(v);
    expect(k.value).toBe(kahanSum(values));
  });

  it('reset() returns the accumulator to zero', () => {
    const k = new KahanAccumulator();
    k.add(1);
    k.add(2);
    expect(k.value).toBe(3);
    k.reset();
    expect(k.value).toBe(0);
    k.add(5);
    expect(k.value).toBe(5);
  });

  it('preserves precision across many small additions', () => {
    const k = new KahanAccumulator();
    const N = 1_000_000;
    k.add(1.0);
    for (let i = 0; i < N; i++) k.add(1e-15);
    const expected = 1 + N * 1e-15;
    expect(Math.abs(k.value - expected)).toBeLessThan(1e-13);
  });
});
