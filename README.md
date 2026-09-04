# ak-partial-ramsey

Equation core, independent evaluator, and successor services for the **small-open
single-AK and AK-or-partial transition computation**, implementing blocks **G0**, **N0**,
and **N1** of specification CS011 version 0.4 (`draft`).

## What this repository is

CS011 sequences the calculation as `G0 -> N0 -> N1 -> N2 -> ... -> N7`. This repository
implements only the first three blocks:

| Block | Scope | Implemented here |
|---|---|---|
| G0 | reconciled single-AK interface and frozen test fixtures | yes |
| N0 | one solver-independent equation core plus a genuinely separate evaluator | yes |
| N1 | parameterized AK and fixed-task-share partial-automation successor services | yes |
| N2 onward | stationary continuation, local-theorem certificate, transition BVPs, value reconstruction, HJB verification, comparative statics | **no — deliberately out of scope** |

## What this repository is not

Nothing produced here is an equilibrium result, a Ramsey optimum, a calibration, or a
welfare result. The result-use ceiling for every output of these blocks is
**`exploratory_only`** in the sense of the project's error-handling and result-use policy.

Five CS011 activation gates remain open, so this repository deliberately does **not**
define any of the following, and will not silently supply a default for them:

1. the canonical inherited state `(S_0, M_0)`;
2. the economic parameter scenario;
3. the final synthetic anchor;
4. successor domains for a substantive run; or
5. economic materiality thresholds.

Generic solver APIs accept these objects as validated inputs. The only numerical fixtures
shipped are (a) the two-root AK fixture supplied verbatim by the single-AK theory note and
(b) fixtures whose names begin with `MANUFACTURED_`, which carry no economic
interpretation and exist solely to exercise code paths.

## Install and run

```bash
uv sync
uv run ak-partial-ramsey preflight
uv run ak-partial-ramsey successors --fixture single-ak-two-root
```

## Layout

```text
src/ak_partial_ramsey/
  errors.py        typed refusals: non-finite input, domain, branch, rank, nesting
  validation.py    finiteness and domain guards used by every public entry point
  params.py        immutable, validated, hashable parameter and domain schemas
  tolerances.py    solver tolerances, kept strictly separate from economic primitives
  primitives.py    production, installation, price, and factor-payment functions
  events.py        event maps and normalized <-> level balance-sheet transformations
  exposure.py      unmultiplied public and private portfolio equations and root services
  canonical.py     pre-arrival state, costate, control, and tax-identity equations
  recovery.py      government, private-owner, and foreign-residual recovery
  successors/
    ak.py          AK root enumeration, TVC/stability selection, values and envelopes
    partial.py     fixed-I_P stationary point, stable manifold, quadrature, BVP check
  evaluator.py     INDEPENDENT evaluator; imports no solver or residual module
  diagnostics.py   machine-readable diagnostic records
  reporting.py     JSON reports with source-manifest and configuration hashes
  fixtures.py      theory-supplied and clearly-named manufactured fixtures
  cli.py           `preflight` and `successors` stages
docs/
  source_manifest.yaml   SHA-256 of every authoritative input
  equation_crosswalk.md  equation-by-equation map from the theory notes to the code
  scale_map.md           units, timing, and scaling conventions
  validation_evidence.md recorded validation commands and results
```

## Numerical discipline

These rules are enforced in code and in tests, not left to convention.

- Every public numerical function rejects non-finite inputs and returns an explicit
  domain or branch failure rather than a NaN or a silently clipped value.
- The AK successor scans its declared root interval, enumerates **every** positive root,
  retains each with an explicit accept/reject reason, and selects by the strict
  productive-value transversality condition. The first root returned by a nonlinear
  solver is never selected.
- The partial-automation successor is the finite-task-share model with
  `0 < I_0 < I_P < 1`. Constructing an AK successor by sending `I_P -> 1` is refused.
- Exposure identification uses the unmultiplied first-order condition throughout. The
  code never divides by a mark payoff that may vanish; it refuses identification when the
  entire payoff vector is zero and preserves an unspanned mark when one component is zero.
- Physical and risk-neutral intensities are carried as distinct quantities.
- Branch selection, root rejection, and every domain boundary are surfaced in the returned
  diagnostics, never hidden inside a convenience function.

## Documentation

| File | Contents |
|---|---|
| [`docs/equation_crosswalk.md`](docs/equation_crosswalk.md) | every implemented equation mapped to its numbered equation in the theory packets, plus the three notation collisions between them |
| [`docs/scale_map.md`](docs/scale_map.md) | units, timing conventions, domains, transversality conditions, and the scaling this implementation applies |
| [`docs/source_manifest.yaml`](docs/source_manifest.yaml) | SHA-256 of every authoritative input, and the record of concurrent edits to the source workspace |
| [`docs/validation_evidence.md`](docs/validation_evidence.md) | validation commands and results, every enumerated root with its disposition, cross-route differences, and resource use |

## Provenance

`docs/source_manifest.yaml` records the SHA-256 hash of each authoritative input read
while writing this implementation. Every report emitted by the CLI carries the
source-manifest hash, the configuration hash, and the package and solver versions.

No substantive disagreement was found between the two theory packets on any equation,
unit, timing convention, or interface used by these blocks. Three notation collisions
between them are recorded in the crosswalk.
