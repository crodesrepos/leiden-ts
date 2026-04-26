---
name: Bug report
about: Something doesn't work the way the README or types suggest.
title: ''
labels: bug
---

**What happened**

<!-- A short description of the unexpected behavior. -->

**Minimal reproduction**

```ts
import { Graph, leiden } from 'leiden-ts';

// smallest graph + options that reproduce the issue
```

**Expected vs actual**

- Expected: …
- Actual: …

**Environment**

- `leiden-ts` version:
- Node version (`node --version`):
- OS / arch:
- Installed via: `npm` / `pnpm` / `yarn` / git URL

**Performance regressions only**

If the issue is a performance regression, please attach the relevant lines from `npm run bench:parity` showing which fixture(s) regressed.
