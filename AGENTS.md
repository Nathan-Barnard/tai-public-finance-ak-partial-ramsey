# Implementation repository operating guide

This repository is bound to computational specification **CS011 version 0.4** (`draft`)
of the TAI public-finance research workspace. It implements blocks **G0**, **N0**, and
**N1** only. It is implementation code; it owns no economic claim, proof status, or
result registry. Those remain with their canonical owners in the research workspace.

## Non-waivable boundaries

1. **Scope.** Do not implement N2 (stationary continuation), N3 (local-theorem
   certificate), N4 (transition BVP), N5 (value reconstruction, HJB verification,
   second-order check), N6 (fallbacks), or N7 (comparative statics) in this repository
   without a new dispatch. A failed or absent predecessor block is never bypassed.
2. **Activation gates.** Five CS011 gates are open. Do not invent, default, or "pick a
   reasonable value" for the canonical inherited `(S_0, M_0)`, the economic parameter
   scenario, the final synthetic anchor, successor domains for a substantive run, or
   economic materiality thresholds. Accept them as validated inputs instead.
3. **Result use.** No output of this repository may be described as an equilibrium
   result, a Ramsey optimum, a calibration, or a welfare result. The ceiling is
   `exploratory_only`.
4. **Fixtures.** Tests may use only (a) numerical fixtures supplied verbatim by the
   theory notes or (b) fixtures named with a `MANUFACTURED_` prefix that carry no
   economic interpretation.
5. **Theory conflicts.** If the theory inputs disagree on an equation, unit, timing
   convention, or interface, stop and report file-and-line evidence. Do not resolve a
   substantive disagreement by choosing whichever formula is easier to code.

## Numerical rules enforced in code

- Reject non-finite inputs at every public entry point; raise a typed error rather than
  returning NaN or a clipped value (`errors.NonFiniteInputError`).
- Return explicit domain and branch failures (`errors.DomainError`,
  `errors.BranchFailure`, `errors.RankFailure`).
- Enumerate every positive AK root in the declared interval; retain rejected roots with
  reasons; select by the strict productive-value TVC. Never take the first solver root.
- Never construct the AK successor as a limit of the partial successor
  (`errors.NestingError` guards `I_P -> 1`).
- Use the unmultiplied exposure first-order condition. Never divide by a mark payoff.
  Refuse identification on a zero payoff vector; preserve an unspanned zero-payoff mark.
- Keep physical intensities `lambda_j` and risk-neutral intensities `lambda_j_star`
  distinct at every call site.
- Keep economic primitives (`params.py`) and solver tolerances (`tolerances.py`) in
  separate, separately hashed objects.
- The independent evaluator (`evaluator.py`) must not import `exposure`, `canonical`,
  `recovery`, or `successors`. A test enforces this by inspecting its import graph.
- A passing solver status is never acceptance. Acceptance requires the independent
  evaluator plus the structural gates.

## Failure dispositions

Follow the workspace policy `computation/error-handling-and-result-use.md`. Stop and
report exact evidence when: the theory inputs disagree; a unit, timing convention, or
event map is ambiguous; the AK interface cannot reproduce the supplied fixture; the
evaluator and the solver equations disagree; a root or branch changes under refinement
without a diagnosed fold; the partial stable-manifold routes materially disagree; a
non-finite value occurs; a test would require clipping an economic constraint; or
provenance or source hashes cannot be recorded.

Do not repair a failed test by loosening a tolerance without a quantity-specific error
argument. Structural failures cannot be averaged into a numerical tolerance.

## Validation

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ak-partial-ramsey preflight
uv run ak-partial-ramsey successors --fixture single-ak-two-root
```

No single smoke or test invocation may exceed 10 minutes (lane `L1_interactive`). No GPU,
paid compute, cloud compute, or distributed execution is authorized. Keep the checkout
and retained artifacts below 500 MiB; do not commit virtual environments, caches, large
meshes, or binary dumps.

## Provenance

`docs/source_manifest.yaml` records the SHA-256 of every authoritative input. If a source
hash changes, re-read that input before trusting any equation derived from it, and record
the change. Provenance that cannot be recorded is a stop condition, not a warning.
