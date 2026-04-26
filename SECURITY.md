# Security policy

## Supported versions

While leiden-ts is in 0.x, only the latest minor version receives security fixes. Once 1.0 ships, the policy will be revisited.

| Version  | Supported          |
|---       |---                 |
| 0.1.x    | :white_check_mark: |
| < 0.1.0  | :x:                |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for a security concern. Instead:

- Use GitHub's [private vulnerability reporting](https://github.com/crodesrepos/leiden-ts/security/advisories/new) on this repository, or
- Email <crankthesiren@gmail.com> with subject line `leiden-ts security: <short summary>`.

Include:

- A description of the vulnerability and its impact
- Steps to reproduce, or a proof-of-concept
- Any suggested mitigation

I will acknowledge receipt within 7 days and aim to ship a fix within 30 days for confirmed vulnerabilities. Public disclosure will be coordinated with the reporter.

## Scope

leiden-ts is a pure-TypeScript graph-clustering library with zero runtime dependencies. The realistic vulnerability surface is small but includes:

- **Algorithmic complexity attacks**: pathological inputs that cause unbounded iteration or memory growth. The `maxLevels` option and absolute budget gates in `bench/` are the first line of defense; reports of inputs that defeat both are in scope.
- **Numerical edge cases** that produce incorrect results (e.g. NaN propagation, integer overflow on edge counts) without throwing. In scope as correctness bugs even when not security-relevant.

Out of scope:

- Bugs that only manifest with adversarial use of `validate: false` (the option exists to skip validation; misuse is the caller's responsibility).
- General performance regressions (file as a regular issue, not a security advisory).
