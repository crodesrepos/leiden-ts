/**
 * Benchmark helpers: timing primitives + percentile reporting.
 *
 * Uses `process.hrtime.bigint()` for nanosecond-precision monotonic time.
 * Discards the first N runs as warmup; reports p50/p90/p99 over the rest.
 */

export interface BenchOptions {
  warmup?: number;
  runs?: number;
}

export interface BenchResult {
  label: string;
  warmup: number;
  runs: number;
  median_ms: number;
  p50_ms: number;
  p90_ms: number;
  p99_ms: number;
  min_ms: number;
  max_ms: number;
  total_ms: number;
}

export function bench(
  label: string,
  fn: () => unknown,
  options: BenchOptions = {},
): BenchResult {
  const warmup = options.warmup ?? 10;
  const runs = options.runs ?? 100;

  // Warmup — discard timings.
  for (let i = 0; i < warmup; i++) fn();

  const samples = new Float64Array(runs);
  let total = 0;
  for (let i = 0; i < runs; i++) {
    const start = process.hrtime.bigint();
    fn();
    const end = process.hrtime.bigint();
    const ms = Number(end - start) / 1e6;
    samples[i] = ms;
    total += ms;
  }

  return {
    label,
    warmup,
    runs,
    median_ms: percentile(samples, 0.5),
    p50_ms: percentile(samples, 0.5),
    p90_ms: percentile(samples, 0.9),
    p99_ms: percentile(samples, 0.99),
    min_ms: minOf(samples),
    max_ms: maxOf(samples),
    total_ms: total,
  };
}

export function percentile(samples: Float64Array, p: number): number {
  if (samples.length === 0) return Number.NaN;
  const sorted = Float64Array.from(samples);
  // ascending
  for (let i = 1; i < sorted.length; i++) {
    const v = sorted[i] as number;
    let j = i - 1;
    while (j >= 0 && (sorted[j] as number) > v) {
      sorted[j + 1] = sorted[j] as number;
      j--;
    }
    sorted[j + 1] = v;
  }
  const idx = (sorted.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, sorted.length - 1);
  const frac = idx - lo;
  return (sorted[lo] as number) * (1 - frac) + (sorted[hi] as number) * frac;
}

function minOf(samples: Float64Array): number {
  let m = Infinity;
  for (let i = 0; i < samples.length; i++) {
    const v = samples[i] as number;
    if (v < m) m = v;
  }
  return m;
}

function maxOf(samples: Float64Array): number {
  let m = -Infinity;
  for (let i = 0; i < samples.length; i++) {
    const v = samples[i] as number;
    if (v > m) m = v;
  }
  return m;
}

export function formatResult(r: BenchResult): string {
  return [
    `${r.label.padEnd(40)}`,
    `runs=${String(r.runs).padStart(4)}`,
    `p50=${r.p50_ms.toFixed(3).padStart(8)}ms`,
    `p90=${r.p90_ms.toFixed(3).padStart(8)}ms`,
    `p99=${r.p99_ms.toFixed(3).padStart(8)}ms`,
    `min=${r.min_ms.toFixed(3).padStart(8)}ms`,
    `max=${r.max_ms.toFixed(3).padStart(8)}ms`,
  ].join('  ');
}

/**
 * Seeded mulberry32 PRNG. Used by benchmark fixtures that need to generate
 * graphs deterministically. NOT exported from the package; clustering will
 * have its own PRNG (chosen in a future ADR).
 */
export function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
