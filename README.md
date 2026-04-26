# leiden-ts

> Pure-TypeScript implementation of the Leiden community-detection algorithm (Traag, Waltman & van Eck, 2019). Zero runtime dependencies. Faster than [graspologic](https://github.com/microsoft/graspologic) (the canonical Python reference, Java backend) across the standard benchmark portfolio.

## Headline

Median over 10 seeds vs `graspologic.partition.leiden` (graspologic 3.4.4), Apple Silicon, Node 24.7. Reference outputs committed under `bench/compare/ref/outputs/`.

| fixture          | n      | m       | leiden-ts Q | graspologic Q | ΔQ (signed) | speedup (p50) |
|---               |---     |---      |---          |---            |---          |---            |
| karate           | 34     | 78      | 0.4188      | 0.4193        | -0.0005     | **3.7×**      |
| dolphins         | 62     | 159     | 0.5201      | 0.5188        | **+0.0013** | **5.9×**      |
| football         | 115    | 613     | 0.6043      | 0.6043        | +0.0000     | **9.9×**      |
| httpx (real)     | 144    | 330     | 0.5085      | 0.5069        | **+0.0017** | **6.3×**      |
| lfr-1k μ=0.3     | 1,000  | 5,422   | 0.5293      | 0.5307        | -0.0014     | **4.5×**      |
| lfr-1k μ=0.5     | 1,000  | 5,562   | 0.2991      | 0.2993        | -0.0002     | **3.4×**      |
| lfr-10k μ=0.3    | 10,000 | 643,259 | 0.4933      | 0.4933        | +0.0000     | **4.6×**      |

Bold ΔQ values mean leiden-ts produces a higher-modularity partition than graspologic. All seven fixtures clear the one-sided parity gates: `Q_ours ≥ Q_grasp − 0.05` and `wallclock_ours_p50 ≤ wallclock_grasp_p50`.

## Why this exists

There is currently no mature pure-JavaScript Leiden implementation:

- `graphology-communities-louvain` covers Louvain only — not Leiden, and Leiden's connectivity guarantee (Theorem 1 of the paper) is the load-bearing improvement Louvain lacks.
- `graspologic` (Python) wraps a Java backend.
- `leidenalg` (Python) wraps `libleidenalg` (C++) → `igraph`.
- `igraph` (C) compiles to a 3–5 MB WASM port that's awkward to ship.

A **correct, fast, audit-able pure-TypeScript Leiden** runs verbatim in Node, the browser, Deno, Bun, edge runtimes, and any embedded JS engine — no native modules, no WASM, no Python or JVM sidecar.

## Highlights

- **Pure TypeScript**, strict mode, ESM + CJS + `.d.ts`. Zero runtime dependencies.
- **Bundle size**: 16 KB ESM / 17 KB CJS / 24 KB types.
- **CSR typed-array graph store** (`Uint32Array` offsets + targets, `Float64Array` weights) for cache-friendly iteration and minimal GC pressure.
- **All three Leiden phases**: local moving, refinement (with the connectivity-preserving guarantee), and aggregation. Multi-level loop with paper-shape termination.
- **Connectivity guarantee verified**: every output community is internally connected (Theorem 1). BFS-asserted on every fixture in the test suite.
- **Deterministic** under fixed seed. xoshiro128\*\* PRNG; documented tie-breaking; documented convergence.
- **Drop-in API for graspologic**: `leiden(graph, { seed: 42, resolution: 1.0 })` returns a `Partition`. Option names match graspologic 1:1 (modulo `random_seed` → `seed`).
- **Warm-start support**: `initialPartition` lets callers resume from a prior partition for incremental re-clustering.

## Quickstart

```ts
import { Graph, leiden, modularity } from 'leiden-ts';

const graph = Graph.fromEdgeList(34, [
  [0, 1], [0, 2], [0, 3],
  // ...
]);

const result = leiden(graph, { seed: 42, resolution: 1.0 });

console.log(result.modularity);                  // 0.4188
console.log(result.partition.numCommunities);    // 4
console.log(Array.from(result.partition.assignments)); // [0, 0, 0, …]

// Or compute Q for an externally-supplied partition:
const q = modularity(graph, result.partition, 1.0);
```

## Cross-validation methodology

Quality and performance claims are reproducible and gated. The full methodology is in [ADR-0004](docs/adr/0004-benchmark-methodology.md). Highlights:

**Parity gates** (one-sided — we do not fail when leiden-ts beats graspologic):

| Gate | Shape | Threshold | Where |
|---|---|---|---|
| Quality | `Q_ours_median ≥ Q_grasp_median − 0.05` | ε = 0.05 | `test/cross-validation.test.ts` + `bench/cross-validation-report.ts` |
| Wall-clock | `wallclock_ours_p50 ≤ wallclock_grasp_p50 × 1.0` | parity-or-better | `bench/cross-validation-report.ts` (gated; exits non-zero on regression) |
| **Anti-sarāb** (truth-NMI parity, on fixtures with canonical truth) | `NMI_ours_vs_truth ≥ NMI_grasp_vs_truth − 0.05` | ε = 0.05 | `test/cross-validation.test.ts` (Football, LFR-1k μ=0.3) |
| Connectivity (Theorem 1) | every output community is internally connected | absolute | per-fixture, BFS-asserted |

**Diagnostic emit (NOT gates — printed but not gated):** NMI vs graspologic, ARI vs graspologic, n_communities agreement. Two correct implementations land at different equally-modular partitions whenever a node has tied ΔQ across communities — divergence at tied Q is not a regression.

**The sarāb partition** (سراب — Arabic for *desert mirage*) is the codebase's name for the failure mode where a partition has the right modularity at a distance but the wrong structure on contact with truth. Pure Q-parity is blind to it; the anti-sarāb gate is the instrument that catches it. See [ADR-0004](docs/adr/0004-benchmark-methodology.md) for the full definition.

## Comparison with peer implementations

| Project | Language | What it is | Wraps |
|---|---|---|---|
| **[graspologic.partition.leiden](https://github.com/microsoft/graspologic)** | Python | Microsoft/JHU's reference. The implementation we measure against. | Java backend |
| **[leidenalg](https://github.com/vtraag/leidenalg)** (Traag) | Python | The canonical reference; Traag is the paper's first author. | C++ (libleidenalg) → igraph |
| **[graphrs](https://docs.rs/graphrs)** | Rust | A general Rust graph crate with Leiden as one algorithm; supports modularity and CPM. | pure Rust |
| **[fa-leiden-cd](https://github.com/fixed-ai/fa-leiden-cd)** | Rust | Minimal-dependency Rust Leiden, parallel-capable. | none (pure) |
| **[graphology-communities-louvain](https://github.com/graphology/graphology/tree/master/src/communities-louvain)** | JavaScript | The closest existing JS option — but Louvain only, not Leiden, and uses non-typed arrays. | none |
| **leiden-ts** | TypeScript | This project. | none — typed-array CSR, no native modules, no WASM |

### Design choices that distinguish leiden-ts

| Dimension | graspologic | leidenalg | graphrs | **leiden-ts** |
|---|---|---|---|---|
| **Backend** | Java | C++ | pure Rust | **pure TypeScript** |
| **Graph type** | igraph-style | igraph CSR | custom | **typed-array CSR (canonical sort)** |
| **PRNG** | `java.util.Random` (LCG) | `std::mt19937` | `rand` crate | **xoshiro128\*\*** |
| **Tie-breaking** | undocumented (Java RNG order) | undocumented | undocumented | **explicitly documented: random via seeded PRNG** |
| **Convergence** | undocumented internal | tolerance-style | undocumented | **explicitly documented; both `lmResult.moves === 0` and `isAllSingletons` per Traag 2019 Algorithm 1** |
| **Quality fns** | modularity only | modularity + CPM + RB + Significance + Surprise | modularity + CPM | modularity (CPM and beyond planned) |
| **Public API shape** | `leiden(G, resolution, randomness, random_seed)` → dict | `find_partition(...)` → VertexPartition | `leiden(g, weighted, qf, ...)` → list | **graspologic-shaped** `leiden(graph, opts) → Partition` for drop-in replacement |
| **Warm-start** | not exposed | not exposed | not exposed | **`initialPartition` option** ✓ |

### What we deliberately copy

- **Match graspologic's API option names** (`resolution`, `seed`) so a reader of either codebase recognizes the parameters 1:1. This is the load-bearing compatibility decision; everything else is implementation freedom.
- **Hierarchical Leiden output** as in `leidenalg`'s multi-level partitions — for callers that want γ-resolution exploration.

### What we deliberately diverge on

- **Pure-TypeScript on typed arrays.** Every reference picks a different runtime; we own ours. See [ADR-0001](docs/adr/0001-pure-typescript-runtime.md) for the rationale.
- **Documented tie-breaking and convergence.** Every reference leaves these undocumented or hidden behind RNG side-effects. We make both explicit in ADRs and assert them in property tests.
- **Tighter cross-validation discipline.** [ADR-0004](docs/adr/0004-benchmark-methodology.md) defines the one-sided parity gates and the anti-sarāb gate; results trace to reproducible experiments under `bench/compare/`.
- **`Partition` as a class with cached community totals.** Other implementations return raw partition dicts/lists. We carry pre-computed `Σᵢₙ(c)` and `Σₜₒₜ(c)` accumulators so subsequent passes (refinement, aggregation, warm-start) start from O(1) state instead of O(m) recomputation.

### What we explicitly do not do

- **No igraph or petgraph dependency.** The `Graph` is CSR typed-array, owned. No graph-library lock-in for callers; no opaque C/Rust foreign code.
- **No WASM at runtime.**
- **No optional Java/Python bridge.** No "shell out to graspologic" mode.

## How to verify locally

```bash
git clone <this-repo>
cd leiden-ts
npm install
npm run test         # correctness + property tests + Q parity + anti-sarāb gates
npm run bench        # absolute performance budgets + parity gates (Q + wall-clock per fixture)
npm run bench:parity # the parity gates alone (cross-validation report)
npm run typecheck    # strict tsc --noEmit
npm run build        # produces dist/ (ESM + CJS + d.ts)
```

To regenerate graspologic reference outputs (optional — pinned values are committed):

```bash
cd bench/compare
python3 -m venv .venv && source .venv/bin/activate
pip install -r ref/requirements.txt
python ref/run_leiden.py --all --seed 42 --runs 30
```

## License

MIT — see [LICENSE](LICENSE).
