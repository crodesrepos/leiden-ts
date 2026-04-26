# Contributing to leiden-ts

Thanks for your interest in contributing. This document covers how to set up a development environment, what kinds of changes are likely to be accepted, and the conventions the project follows.

## Quick start

```bash
git clone https://github.com/crodesrepos/leiden-ts.git
cd leiden-ts
npm install
npm run typecheck
npm run test
npm run bench:parity
```

Node ≥ 20.10. No native modules; no WASM. The test suite runs offline using committed reference outputs under `bench/compare/ref/outputs/`.

## Reporting issues

Please file an issue at <https://github.com/crodesrepos/leiden-ts/issues>. For bug reports, the minimum useful info is:

- A minimal reproducer (graph as edge list, options used, observed output, expected output)
- Node version, OS, and architecture
- Whether the issue reproduces with the published `npm` package or only locally

For performance regressions, please include the output of `npm run bench:parity` showing the affected fixture(s).

## Pull requests

Before opening a PR for a non-trivial change, **please open an issue first** so we can align on scope. Small fixes (typos, doc improvements, narrow bug fixes) are fine to send directly.

### Required for any PR

1. **Tests pass**: `npm run test`
2. **Typecheck clean**: `npm run typecheck`
3. **Parity gates pass**: `npm run bench:parity`. The gates are one-sided; they fail only on regression.
4. **No new runtime dependencies.** The library is and will remain zero-runtime-dependency. If you have a strong case for a dep, raise it in an issue first.
5. **No native modules; no WASM at runtime.** The library targets pure JavaScript runtimes universally.

### Conventions

- **TypeScript strict mode.** No `any`, no `@ts-ignore`. Type assertions (`as`) only at well-defined boundaries (e.g. JSON parse).
- **Typed arrays for hot paths.** Inner loops over graph data must use `Uint32Array` / `Float64Array` views, not `Map` or `Object`. No allocation inside the inner pass of local-moving or refinement.
- **Documented determinism.** Anything that depends on a PRNG draws from the seeded `Xoshiro128`. No use of `Math.random()` outside test scaffolding.
- **Algorithm comments cite the paper.** Where Traag, Waltman & van Eck (2019) prescribes a specific behaviour, the code comment cites the section.

### Algorithmic changes

Changes that touch the local-moving, refinement, or aggregation phases require:

- A property-test demonstrating the invariant the change preserves (Q-preservation under aggregation, connectivity of output communities, etc.).
- Confirmation that the parity gates still pass on every fixture.
- A note in `CHANGELOG.md` under `[Unreleased]` describing the user-visible effect, if any.

If your change improves quality or performance against graspologic, the gates auto-pass (they're one-sided). Numbers in the README's parity table can be refreshed in the same PR.

## Releasing

Maintainers only. Releases use semantic versioning:

- **Patch** (0.1.0 → 0.1.1): bug fix, no API changes, parity numbers may improve.
- **Minor** (0.1.0 → 0.2.0): additive API surface (new option, new return field), no removals.
- **Major** (0.1.0 → 1.0.0): breaking API change, removed options, semantic shifts.

Process:

1. Update `CHANGELOG.md`: move `[Unreleased]` to a new dated version.
2. `npm version <patch|minor|major>` (creates the git tag).
3. `npm run typecheck && npm run test && npm run build`.
4. `npm publish` (publishConfig is already `public`).
5. `git push --follow-tags`.
