/**
 * Tests for cluster-comparison metrics (NMI, ARI).
 *
 * Categories:
 *   Cat-1 Identity      — known closed-form values on small partitions
 *   Cat-3 Property      — invariance under label permutation; identity holds
 *   Cat-5 Contract      — input validation
 */

import { describe, expect, it } from 'vitest';
import fc from 'fast-check';

import { nmi, ari } from '../src/internal/metrics.js';

describe('NMI — Cat-1 Identity', () => {
  it('NMI(P, P) === 1 for any partition P', () => {
    const p = Int32Array.from([0, 0, 1, 1, 2, 2]);
    expect(nmi(p, p)).toBeCloseTo(1, 12);
  });

  it('NMI of permuted-id partition is 1 (label invariance)', () => {
    const a = Int32Array.from([0, 0, 1, 1, 2, 2]);
    const b = Int32Array.from([5, 5, 7, 7, 9, 9]); // same structure, different labels
    expect(nmi(a, b)).toBeCloseTo(1, 12);
  });

  it('NMI of independent partitions is 0', () => {
    // The deterministic counterpart: A puts each pair together, B puts them apart.
    // a = [0,0,1,1], b = [0,1,0,1] — random Rand Index pattern; MI close to 0.
    // For perfect independence on 4 nodes, MI is exactly 0.
    const a = Int32Array.from([0, 0, 1, 1]);
    const b = Int32Array.from([0, 1, 0, 1]);
    const v = nmi(a, b);
    expect(v).toBeLessThan(0.05);
    expect(v).toBeGreaterThanOrEqual(0);
  });

  it('NMI of all-same vs all-same trivial partitions === 1', () => {
    const a = Int32Array.from([0, 0, 0]);
    const b = Int32Array.from([5, 5, 5]);
    expect(nmi(a, b)).toBeCloseTo(1, 12);
  });

  it('NMI of all-same vs distinct labels === 0 (one is informative, the other is not)', () => {
    const a = Int32Array.from([0, 0, 0, 0]);
    const b = Int32Array.from([0, 1, 0, 1]);
    expect(nmi(a, b)).toBe(0);
  });

  it('NMI on empty partition is 1 (vacuously identical)', () => {
    const a = new Int32Array(0);
    const b = new Int32Array(0);
    expect(nmi(a, b)).toBe(1);
  });
});

describe('ARI — Cat-1 Identity', () => {
  it('ARI(P, P) === 1', () => {
    const p = Int32Array.from([0, 0, 1, 1, 2, 2]);
    expect(ari(p, p)).toBeCloseTo(1, 12);
  });

  it('ARI of permuted-id partition is 1', () => {
    const a = Int32Array.from([0, 0, 1, 1, 2, 2]);
    const b = Int32Array.from([7, 7, 3, 3, 5, 5]);
    expect(ari(a, b)).toBeCloseTo(1, 12);
  });

  it('ARI of trivial partitions is 1', () => {
    const a = Int32Array.from([0, 0, 0]);
    const b = Int32Array.from([5, 5, 5]);
    expect(ari(a, b)).toBeCloseTo(1, 12);
  });

  it('ARI bounded in [-1, 1] on small inputs', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 2, max: 20 }).chain((n) =>
          fc.tuple(
            fc.array(fc.integer({ min: 0, max: 4 }), {
              minLength: n,
              maxLength: n,
            }),
            fc.array(fc.integer({ min: 0, max: 4 }), {
              minLength: n,
              maxLength: n,
            }),
          ),
        ),
        ([a, b]) => {
          const v = ari(Int32Array.from(a), Int32Array.from(b));
          return v >= -1 - 1e-12 && v <= 1 + 1e-12;
        },
      ),
      { numRuns: 100, seed: 0xc0ffee },
    );
  });
});

describe('Property — agreement metrics behave consistently', () => {
  it('NMI ≥ 0 for any pair', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 2, max: 20 }).chain((n) =>
          fc.tuple(
            fc.array(fc.integer({ min: 0, max: 4 }), {
              minLength: n,
              maxLength: n,
            }),
            fc.array(fc.integer({ min: 0, max: 4 }), {
              minLength: n,
              maxLength: n,
            }),
          ),
        ),
        ([a, b]) => {
          const v = nmi(Int32Array.from(a), Int32Array.from(b));
          return v >= -1e-12 && v <= 1 + 1e-12;
        },
      ),
      { numRuns: 100, seed: 0xfade },
    );
  });

  it('NMI is symmetric: NMI(A, B) === NMI(B, A)', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 2, max: 15 }).chain((n) =>
          fc.tuple(
            fc.array(fc.integer({ min: 0, max: 3 }), {
              minLength: n,
              maxLength: n,
            }),
            fc.array(fc.integer({ min: 0, max: 3 }), {
              minLength: n,
              maxLength: n,
            }),
          ),
        ),
        ([a, b]) => {
          const ab = nmi(Int32Array.from(a), Int32Array.from(b));
          const ba = nmi(Int32Array.from(b), Int32Array.from(a));
          return Math.abs(ab - ba) < 1e-10;
        },
      ),
      { numRuns: 50, seed: 0xfeed },
    );
  });
});

describe('Cat-5 Contract', () => {
  it('NMI throws on length mismatch', () => {
    expect(() =>
      nmi(Int32Array.from([0, 1]), Int32Array.from([0, 1, 2])),
    ).toThrowError(RangeError);
  });

  it('ARI throws on length mismatch', () => {
    expect(() =>
      ari(Int32Array.from([0, 1]), Int32Array.from([0, 1, 2])),
    ).toThrowError(RangeError);
  });
});
