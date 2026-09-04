# Validation evidence

Recorded for blocks G0, N0 and N1 of CS011. Every number below comes from the run
described here; nothing is quoted from an earlier state of the code.

Machine lane: **L1 interactive**. No single test or smoke invocation exceeded ten
minutes. No GPU, paid compute, cloud compute, or distributed execution was used.

**Result-use ceiling: `exploratory_only`.** Nothing recorded here is an equilibrium
result, a Ramsey optimum, a calibration, or a welfare result. A passing validation run is
not acceptance; the branch is ready for independent review, not accepted.

---

## 1. Environment and provenance

| Item | Value |
|---|---|
| Python | 3.13.5 |
| NumPy | 2.5.2 |
| SciPy | 1.18.1 |
| pytest | 9.1.1 |
| Ruff | 0.16.6 |
| Platform | macOS 15.6.1, arm64 |
| Package version | 0.1.0 |
| `uv.lock` SHA-256 | `abf3f4d29bd6cb401a4f1be5acec3d05335c85507d665f6189b45b8eb0ebac81` |
| `docs/source_manifest.yaml` SHA-256 | `57b1c1a7e630ef4f7968fe2fd3af0350457186f555b036066b88891860431ef9` |
| Tolerances digest | `e9bdecb11e3e28dff018f8e93f173936a3345833aa8d2b3542a3dec9d255c48c` |

Fixture parameter digests:

```
single-ak-two-root=5c3f90dd2e1bf2cc31f95d5ca43e921aed56930e8a5b6ef5bef10f60373a2080
manufactured-two-mark=056f49c2e65183f01184800b8b0ee7cc21ba9140c67d22787f45cef96c31a67e
manufactured-single-ak-support=0df2136683928f9ced864bbf68c61f02bba5262465c73f78ec69458b0cf26eb6
```

Source-input hashes are in [`source_manifest.yaml`](source_manifest.yaml), including the
record of the source workspace being under concurrent edit while this was written.

---

## 2. Commands run and results

```bash
rm -rf .venv
uv sync --frozen                                              # clean environment from the lockfile
uv run python -c "import ak_partial_ramsey"                   # editable install verified
uv run pytest -q                                              # 204 passed
uv run ruff check .                                           # All checks passed
uv run ruff format --check .                                  # 36 files already formatted
uv run ak-partial-ramsey preflight                            # PASS
uv run ak-partial-ramsey successors --fixture single-ak-two-root      # PASS
uv run ak-partial-ramsey successors --fixture manufactured-two-mark   # PASS
```

| Command | Result | Wall time |
|---|---|---|
| `uv sync --frozen` | success, editable install resolved to `src/ak_partial_ramsey` | 0.5 s |
| `pytest -q` | **204 passed, 0 failed** | 48.8 s |
| `ruff check .` | All checks passed | < 1 s |
| `ruff format --check .` | 36 files already formatted | < 1 s |
| `preflight` | PASS, 6/6 structural refusal checks | 1.06 s stage |
| `successors --fixture single-ak-two-root` | PASS | 0.007 s stage |
| `successors --fixture manufactured-two-mark` | PASS | 2.3 s stage |

Peak resident memory across CLI stages: **66.6 MB**.

Test distribution:

| File | Tests |
|---|---|
| `test_refusals.py` | 45 |
| `test_round_trips.py` | 30 |
| `test_partial_successor.py` | 24 |
| `test_exposure_exceptional_cases.py` | 21 |
| `test_evaluator_independence.py` | 20 |
| `test_cli_and_reporting.py` | 17 |
| `test_single_ak_reduction.py` | 13 |
| `test_branch_stability.py` | 11 |
| `test_g0_single_ak_fixture.py` | 10 |
| `test_nesting.py` | 8 |
| `test_mark_aggregation.py` | 5 |
| **Total** | **204** |

### Clean-checkout reproduction

Commit `02d22fb` (G0/N0) was checked out into a detached worktree and its 101 tests plus
`ruff check` were run there with no successor, assembly, reporting or CLI module present.
Both passed, confirming that the G0/N0 commit is self-consistent rather than only passing
against the final tip.

---

## 3. AK successor: roots, rejections, residuals

Theory-supplied fixture `single-ak-two-root`, from the single-AK packet's section
"4. Admissible and rejected scalar-root fixtures":
`(rho, r_F_bar, A_bar, varphi, delta) = (0.04, 0.035, 0.10, 2.5, 0.06)`.

Scalar coefficients: `a = 1.25`, `u = 0.2375`, `q_m = 1.2680749967907192`,
discriminant `a - exp(u) = -0.0180749968 < 0`, so exactly two positive roots.

**Every enumerated root, with its disposition:**

| `q_F` | Branch | Verdict | Reason | TVC margin `r_F - g` | `f` polynomial | `f` level form | Bracket width |
|---|---|---|---|---|---|---|---|
| 1.0600839471281598 | `lower_strict_tvc` | **accepted** | `accepted_strict_productive_value_tvc` | +0.071660759843 | −2.220e−16 | +6.939e−18 | 1.0e−14 |
| 1.4881237110913363 | `upper_tvc_violating` | rejected | `productive_value_tvc_violated` | −0.064006428852 | −2.220e−16 | +2.776e−17 | 1.0e−14 |

Agreement with the packet's stated table, to its nine quoted decimal places, at **both**
roots: `q_F`, `iota_F`, `g_F`, installation margin, `X_F`, `C_F^W`, TVC margin, and
full-BGP residual. The rejected root passes every current positivity and balance-sheet
check and fails only the forward-looking TVC — which is the packet's point in supplying
it.

**Cross-route agreement.** Lambert-W closed form gives
`q_L = 1.0600839471281591`, `q_H = 1.4881237110913388`; maximum disagreement with the
scanned-and-bracketed roots **2.44e−15**. The analytic discriminant predicts two roots and
two were found.

Recovered successor tax `tau_F = 0.0` exactly.

**Independent evaluator on the AK package** (7 residuals, all passing):

| Residual | Value | Scaled | Tolerance |
|---|---|---|---|
| `ak_price_equation_level_form` | +6.939e−18 | +6.939e−18 | 1e−09 |
| `ak_productive_wealth` | 0.0 | 0.0 | 1e−09 |
| `ak_value_by_quadrature` | 0.0 | 0.0 | 1e−09 |
| `ak_value_derivative_e` | −6.320e−09 | −3.186e−10 | 1e−07 |
| `ak_value_derivative_K` | +3.276e−09 | +1.558e−10 | 1e−07 |
| `ak_envelope_condition` | 0.0 | 0.0 | 1e−09 |
| `ak_recovered_successor_tax` | 0.0 | 0.0 | 1e−09 |

---

## 4. Partial successor: manifold, domains, cross-route differences

Manufactured configuration `manufactured-two-mark`. **No economic interpretation.**

Stationary point: `K_P* = 320.2402599858991`, `q_delta = 1.161834242728283`,
`U_P = 0.11701623801408596 > 0`. Field residuals at the point: `K'` −6.7e−15, `q'` +4.2e−17.

Linearisation: `nu_- = −0.10642222271120015`, `nu_+ = +0.15142222271120018`, spectral gap
0.1064. Exactly one stable and one unstable root. Characteristic-polynomial residual
against the packet's stated form: **−6.94e−18**. Rest-point slope gap against
`varphi q_delta nu_- / K_P*`: **9.21e−12**.

Certified domain `[213.49350665726607, 480.3603899788487]`, covering the full declared
interval; 3001 interpolation nodes.

**Cross-route differences:**

| Quantity | Value |
|---|---|
| Productive wealth: quadrature vs algebraic equation | 1.290e−07 |
| Productive wealth: quadrature vs present-value integral (3 held-out states) | 1.61e−08, 5.91e−09, 2.16e−09 |
| Manifold invariance residual (max over probes) | 7.310e−08 |
| `H_P'(K) − q_P(K)` (max over probes) | 1.580e−06 |
| **IVP vs independent BVP collocation (max)** | **1.676e−10** |
| Offset-size spread (offset halved four-fold) | 1.852e−09 |

**Node-by-node IVP/BVP comparison:**

| `K` | `q_P` (BVP) | `q_P` (IVP) | Difference | BVP nodes | BVP max residual |
|---|---|---|---|---|---|
| 266.8669 | 1.221050205844 | 1.221050206011 | −1.665e−10 | 2328 | 9.99e−11 |
| 346.9269 | 1.137614576729 | 1.137614576756 | −2.761e−11 | 2115 | 1.00e−10 |
| 426.9870 | 1.079389996297 | 1.079389996464 | +1.676e−10 | 2793 | 1.00e−10 |

The BVP route is seeded from the **linear** stable solution, not from the integrated
manifold, so it shares no numerical state with the IVP construction.

**Integration-direction checks** (forward time must contract to the rest point):

| Branch | `K` range | Steps | RHS evals | Contracts | Ratio | Linear prediction |
|---|---|---|---|---|---|---|
| upper | [320.2406, 480.3604] | 133 | 1982 | yes | 3.390e−02 | 3.930e−02 |
| lower | [213.4935, 320.2399] | 136 | 2027 | yes | 4.505e−02 | 3.930e−02 |

Departure from the linear prediction at the far ends is nonlinearity, not error: the
check starts well outside the linear regime. What it rules out is unstable-mode
contamination, which would show a ratio orders of magnitude larger or above one.

**Independent evaluator on the partial package** (5 residuals, all passing): manifold
invariance +6.63e−08 (scaled +5.19e−08), productive-wealth equation +1.04e−07 (scaled
+9.58e−11), and three present-value checks at scaled 1.84e−11, 6.17e−12, 2.09e−12.

---

## 5. Equation core at a probe state

**The state is an arbitrary manufactured probe, not an equilibrium point**, and its
implementation margins are not strict. The smallest margin is `debt_B = −1698.18`. That
is the expected and correct outcome: a strictly interior pre-arrival state is an
equilibrium object, and locating one requires blocks N2 and N4. What the stage verifies
is that a violated margin is reported as a signed distance and **never clipped**.

Independent evaluation through the level route (all passing):

| Residual | Value | Scaled |
|---|---|---|
| `event_map_level_normalized_round_trip` | +1.137e−13 | +8.433e−17 |
| `public_exposure_condition` | +9.317e−21 | +9.317e−21 |
| `successor_wealth_against_solver` | +1.137e−13 | +8.433e−17 |

---

## 6. Theory-supplied fixtures reproduced

Two, both verbatim from the packets:

1. **Single-AK two-root fixture** (single-AK packet §4) — every quantity at both roots,
   to the packet's nine quoted decimals, including the balance-sheet round trip
   (`psi = 0.3`, foreign equity −0.3, foreign safe −0.4, foreign net claim −0.7 = −n).
2. **Two-mark algebraic root fixture** (two-mark packet §10) — public exposure root
   `psi ≈ −4.94654`, `B ≈ 1.05346`, `(X_P, X_F) ≈ (9.50535, 5.52673)`,
   `(u_P, u_F) ≈ (0.00656119, −0.00131224)`, orthogonality residual **< 1e−17**, private
   share `pi ≈ −0.886306`, event solvency `(0.91137, 0.55685)`.

---

## 7. Mandatory test coverage

| # | Required test | Where |
|---|---|---|
| 1 | Supplied single-AK two-root fixture, including the rejected root | `test_g0_single_ak_fixture.py` |
| 2 | Exact reduction to single AK at `p_P = lambda_P_star = 0` | `test_single_ak_reduction.py` |
| 3 | Zero-intensity handling without division by zero | `test_exposure_exceptional_cases.py`, `test_single_ak_reduction.py` |
| 4 | Zero-payoff-vector refusal | `test_exposure_exceptional_cases.py` |
| 5 | One-zero-payoff-component through the unmultiplied FOC | `test_exposure_exceptional_cases.py` |
| 6 | Identical-successor aggregation | `test_mark_aggregation.py` |
| 7 | Partial-technology nesting only when all primitives align | `test_nesting.py` |
| 8 | Rejection of `I_P -> 1` as an AK implementation | `test_nesting.py` |
| 9 | Normalized-to-level event-map and budget round trips | `test_round_trips.py` |
| 10 | Successor value and envelope identities | `test_g0_single_ak_fixture.py`, `test_partial_successor.py` |
| 11 | Partial stable-manifold IVP vs BVP agreement | `test_partial_successor.py` |
| 12 | Non-finite input and missing-domain refusal | `test_refusals.py` |
| 13 | Branch labels stable under numerical refinement | `test_branch_stability.py` |
| 14 | Evaluator detects a corrupted equation or event map | `test_evaluator_independence.py` |

Test 14 is the load-bearing one. Structural independence is verified by parsing the
evaluator's import graph, and detection power by running each check against deliberately
corrupted prices, productive wealth, value constants, envelopes, exposure roots, budget
laws, reduction residuals, manifolds, and four distinct event-map corruptions (sign flip,
wrong denominator, omitted wealth, wrong timing). Every corruption is caught.

---

## 8. Resource use

| Item | Value |
|---|---|
| Machine lane | L1 interactive |
| Longest single invocation | 48.8 s (full test suite) |
| Peak resident memory | 66.6 MB |
| Tracked repository content | 472 KB |
| Working checkout excluding `.venv` | 1.7 MB |
| `.git` | 400 KB |
| Retained generated artifacts committed | none |

Well inside the 500 MiB cap and the ten-minute per-invocation ceiling. The `.venv`
(134 MB) is gitignored and not part of the checkout size that the cap governs.

---

## 9. Deviations and open items

No test was repaired by loosening a tolerance. Three numerical settings were changed
during development, each for a stated quantity-specific reason, and each recorded in the
commit that made it:

1. **Stable-manifold integration horizon.** First derived from the capital ratio; it
   should follow from the displacement amplification `exp(|nu_-| t)`. The first version
   covered 0.2% of the declared window. Corrected, not loosened.
2. **Manifold interpolant source.** Built from the integrator's step points, which are
   chosen for local error control and were three orders too sparse; switched to the dense
   output, which is accurate to the same tolerance everywhere between steps.
3. **Direction-check horizon.** Capped so the unstable mode can amplify at most
   hundredfold. A longer horizon reports divergence for a correct branch, because a point
   only numerically on the manifold carries a round-off-sized unstable component growing
   like `exp(nu_+ t)`.

One evaluator check was **removed and replaced**, not relaxed: a finite-difference
`H' = q` test that reduced term by term to the manifold-invariance residual already
checked, and so could not fail independently of it. It was replaced by the present-value
quadrature, which is a genuinely different route.

One implementation defect was found by a test and fixed: `fiscal_valuation_residual`
formed successor wealth before checking whether a mark was active, which made the
single-AK support restriction unreachable whenever the inactive mark's wealth was
nonpositive at the selected exposure.

No output is quarantined. No structural gate was traded against a numerical tolerance.
