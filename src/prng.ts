/**
 * xoshiro128** — a fast, statistically strong, seeded pseudo-random
 * number generator suitable for randomized algorithms (NOT cryptography).
 *
 * Reference: Blackman, D. & Vigna, S. (2018). *Scrambled Linear
 * Pseudorandom Number Generators*. ACM Trans. Math. Softw.
 * https://prng.di.unimi.it/
 *
 * Design choice rationale: see `docs/adr/0005-prng-xoshiro128.md`.
 *
 * Key properties:
 *   - 128-bit state (4 × Uint32) — period 2¹²⁸ − 1
 *   - Passes all of TestU01's BigCrush
 *   - Output: 32-bit unsigned integer per call (`nextUint32()`)
 *   - Float in [0, 1):    `nextFloat()`         — uses upper 24 bits
 *   - Int in [0, n):       `nextIntExclusive(n)` — debiased rejection sampling
 *
 * All operations use Uint32 typed arrays and `Math.imul` for V8-friendly
 * 32-bit math. No BigInt; no allocation in hot path.
 */

const STATE_LEN = 4;

/**
 * Splitmix64-like seed expansion: takes a single 32-bit seed, expands
 * to 4 × 32-bit state. Avoids the "all-zero state" failure mode of
 * xoshiro by mixing the input through several rotations and xors.
 *
 * The expansion is deterministic: same seed → same state → same stream.
 */
function expandSeed(seed: number): Uint32Array {
  let z = (seed | 0) >>> 0;
  if (z === 0) z = 0x9e3779b9; // golden-ratio fallback for the all-zero pathology
  const state = new Uint32Array(STATE_LEN);
  for (let i = 0; i < STATE_LEN; i++) {
    z = (z + 0x9e3779b9) | 0;
    let v = z;
    v = Math.imul(v ^ (v >>> 16), 0x85ebca6b);
    v = Math.imul(v ^ (v >>> 13), 0xc2b2ae35);
    v = (v ^ (v >>> 16)) >>> 0;
    state[i] = v;
  }
  // Re-check: pathological all-zero state is impossible if the expansion
  // produces any non-zero word, which the constants above guarantee.
  return state;
}

/**
 * A seeded PRNG instance. Construct with a 32-bit integer seed; the
 * resulting stream is fully deterministic.
 */
export class Xoshiro128 {
  private readonly s: Uint32Array;

  constructor(seed: number) {
    if (!Number.isFinite(seed)) {
      throw new RangeError(
        `seed must be a finite number; got ${String(seed)}`,
      );
    }
    this.s = expandSeed(Math.trunc(seed));
  }

  /**
   * Advance the state and return a Uint32 in [0, 2³²).
   *
   * Implementation is the canonical xoshiro128** algorithm with the
   * `**` (star-star) scrambler:
   *   result = rotl(s[1] * 5, 7) * 9
   * State update mixes via xor + left rotations.
   */
  nextUint32(): number {
    const s = this.s;
    const s0 = s[0] as number;
    const s1 = s[1] as number;
    const s2 = s[2] as number;
    const s3 = s[3] as number;

    // Output (scrambler): rotl(s1 * 5, 7) * 9
    const r = Math.imul(s1, 5);
    const result = Math.imul(((r << 7) | (r >>> 25)) >>> 0, 9) >>> 0;

    // State update.
    const t = (s1 << 9) >>> 0;
    s[2] = (s2 ^ s0) >>> 0;
    s[3] = (s3 ^ s1) >>> 0;
    s[1] = ((s1 ^ (s[2] as number)) >>> 0) >>> 0;
    s[0] = ((s0 ^ (s[3] as number)) >>> 0) >>> 0;
    s[2] = ((s[2] as number) ^ t) >>> 0;
    s[3] = (((s[3] as number) << 11) | ((s[3] as number) >>> 21)) >>> 0;

    return result;
  }

  /**
   * Float64 in [0, 1). Uses the high 24 bits of one Uint32 draw — IEEE 754
   * has 53 bits of mantissa, but standard practice for [0,1) PRNGs uses
   * 24 or 53 bits. 24 is sufficient for shuffling and tie-breaking.
   *
   * For graph algorithms where bias matters, `nextIntExclusive(n)` is
   * the right primitive — do not derive integers from `nextFloat()` by
   * `Math.floor(nextFloat() * n)`, which biases for non-power-of-2 n.
   */
  nextFloat(): number {
    return (this.nextUint32() >>> 8) / 0x1000000; // 2^24
  }

  /**
   * Uniform integer in [0, n). Unbiased — uses rejection sampling on
   * the underlying Uint32 to eliminate modulo bias for non-power-of-2 n.
   *
   * @throws RangeError if n is not a positive integer ≤ 2³²
   */
  nextIntExclusive(n: number): number {
    if (!Number.isInteger(n) || n <= 0 || n > 0x100000000) {
      throw new RangeError(
        `n must be a positive integer in [1, 2^32]; got ${String(n)}`,
      );
    }
    if (n === 1) return 0;

    // Lemire's debiased multiplication-based bounded rejection.
    // For n ≤ 2³¹ this is exact; for n in (2³¹, 2³²] it falls back to
    // Daniel Lemire's full-precision multiplication via splitting.
    if (n <= 0x80000000) {
      // Most graphs have n far below 2³¹; fast path:
      const mask = n - 1;
      // Power-of-2 shortcut.
      if ((n & mask) === 0) {
        return this.nextUint32() & mask;
      }
      // Rejection sampling with threshold = 2³² mod n.
      const threshold = ((0x100000000 - n) % n) >>> 0;
      let x = this.nextUint32();
      while (x < threshold) x = this.nextUint32();
      return x % n;
    }
    // n in (2³¹, 2³²]: rejection on the rare upper region.
    while (true) {
      const x = this.nextUint32();
      if (x < n) return x;
    }
  }

  /**
   * Fisher–Yates shuffle in place over a Uint32Array. Single allocation:
   * none. Deterministic given seed.
   */
  shuffleUint32(arr: Uint32Array): void {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.nextIntExclusive(i + 1);
      const tmp = arr[i] as number;
      arr[i] = arr[j] as number;
      arr[j] = tmp;
    }
  }

  /**
   * Snapshot the internal state. Useful for tests that need to verify
   * deterministic stream identity.
   */
  snapshot(): Uint32Array {
    return new Uint32Array(this.s);
  }
}

/**
 * Convenience: construct a PRNG with default seed 42.
 * The number 42 is the canonical default in our test suite — keep it
 * stable so test fixtures don't churn when callers omit the seed.
 */
export const DEFAULT_SEED = 42;
