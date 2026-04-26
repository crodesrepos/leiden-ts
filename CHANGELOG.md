# Changelog

All notable changes to **leiden-ts** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — Initial public release

### Added

- Pure-TypeScript Leiden community detection (Traag, Waltman & van Eck, 2019), all three phases (local moving, refinement with the connectivity guarantee, aggregation), wrapped in a multi-level `leiden(graph, options)` public API.
- `Graph` — typed-array CSR store (`Uint32Array` offsets/targets, `Float64Array` weights). Construction normalizes node ids to `[0, n)`, sorts each node's adjacency by target id (canonical), and validates self-loops per the configured policy.
- `Partition` — assignments + cached `Σᵢₙ(c)` and `Σₜₒₜ(c)` accumulators, kept consistent through a single `move()` mutator. Allows incremental modularity-delta computation in O(degree) per pass.
- `modularity(graph, partition, gamma)` — Newman–Girvan modularity with configurable resolution, computed per-community with Kahan compensated summation.
- `leiden(graph, options)` — graspologic-shaped option names: `resolution`, `seed`, `randomness`, plus three additions (`tolerance`, `maxLevels`, `initialPartition` for warm-start).
- xoshiro128\*\* PRNG with debiased bounded sampling and Fisher–Yates shuffle. Deterministic given seed.
- Cross-validation harness comparing leiden-ts to `graspologic.partition.leiden` (graspologic 3.4.4) across 7 fixtures (Karate, Dolphins, Football, httpx, LFR-1k μ=0.3, LFR-1k μ=0.5, LFR-10k μ=0.3). Reference outputs committed under `bench/compare/ref/outputs/` so the test suite runs without a Python toolchain.
- One-sided parity gates: `Q_ours_median ≥ Q_grasp_median − 0.05` and `wallclock_ours_p50 ≤ wallclock_grasp_p50`. Anti-sarāb gate (truth-NMI parity) on Football and LFR-1k μ=0.3.
- Connectivity guarantee (Traag 2019 Theorem 1) verified by BFS on every output community in the test suite.
- Property tests (fast-check) for modularity bounds, γ-derivative, label-permutation invariance, and aggregation Q-preservation.
- Wall-clock benchmarks with absolute budget gates and parity gates against graspologic; runs under `npm run bench`.

### Performance

Median-of-10-seeds vs `graspologic.partition.leiden` on Apple Silicon, Node 24.7:

| fixture       | n      | m       | Q (ours)   | Q (grasp)  | speedup p50 |
|---            |---     |---      |---         |---         |---          |
| karate        | 34     | 78      | 0.4188     | 0.4193     | 3.7×        |
| dolphins      | 62     | 159     | 0.5201     | 0.5188     | 5.9×        |
| football      | 115    | 613     | 0.6043     | 0.6043     | 9.9×        |
| httpx         | 144    | 330     | 0.5085     | 0.5069     | 6.3×        |
| lfr-1k μ=0.3  | 1,000  | 5,422   | 0.5293     | 0.5307     | 4.5×        |
| lfr-1k μ=0.5  | 1,000  | 5,562   | 0.2991     | 0.2993     | 3.4×        |
| lfr-10k μ=0.3 | 10,000 | 643,259 | 0.4933     | 0.4933     | 4.6×        |

[Unreleased]: https://github.com/crodesrepos/leiden-ts/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/crodesrepos/leiden-ts/releases/tag/v0.1.0
