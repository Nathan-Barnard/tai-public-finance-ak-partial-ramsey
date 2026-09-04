# Scale map: units, timing, and scaling conventions

Block G0 requires a scale map alongside the equation crosswalk. This file records the
units, timing conventions, and domains that the two theory packets agree on, and the
scaling this implementation applies.

## Units

Physical capital `K` is common across every regime and is continuous at the event. It is
the only physical quantity; everything else is a market value or a rate.

| Object | Unit |
|---|---|
| `K` | physical capital, common to the pre-arrival, `P` and `F` regimes |
| `q`, `q_P(K)`, `q_F` | installed-equity value **per physical unit** of capital |
| `e`, `psi`, `F`, `Theta`, `B`, `a`, `H_P`, `H_F`, `X_P`, `X_F` | market-value levels in the current-good numeraire |
| `C^W`, `C^O`, `T`, `Y`, `W`, `R K` | flows per unit time in the current-good numeraire |
| `rho`, `r_0_bar`, `r_P_bar`, `r_F_bar`, `g`, `delta` | rates per unit time |
| `lambda`, `lambda_j`, `lambda_j_star` | intensities per unit time |
| `iota` | installation rate per unit of capital, per unit time |
| `varphi` | inverse-curvature parameter of the log installation technology |
| `I_0`, `I_P` | task shares, dimensionless, in `(0, 1)` |
| `J_j` | dimensionless total-gain jump |
| `pi` | dimensionless owner equity wealth share |
| `tau` | dimensionless source-tax rate in `[0, 1]` |
| `mu_K` | marginal value per physical unit of capital |
| `mu_e` | marginal value per unit of external wealth; `mu_e = 1/C` |
| `V_j`, `H_*` derivatives | `V_{j,e}` is dimensionless per unit wealth; `V_{j,K} = q_j V_{j,e}` carries the price |

`q_j K` is installed-equity value, which is what makes `V_{j,K} = q_j V_{j,e}` an
envelope condition rather than a coincidence of scaling.

## Intensities are two distinct objects

Physical intensities `lambda_j = lambda p_j` govern the actual arrival process and enter
the worker's expected value. Risk-neutral intensities `lambda_j_star` come from the
**exogenous world stochastic discount factor** and enter asset pricing. They are carried
as separate fields in `MarkIntensities` and are never substituted for one another.

Their ratio is economically meaningful and appears in the results: on the single-mark
branch `V_{F,e}/mu_e = lambda_F_star/lambda`, and `1 + pi J_F = lambda/lambda_F_star`. The
government's fiscal marginal valuation is endogenous; the market SDF is not. Equality of
the two at the optimum does not make the fiscal costate a pricing kernel outside the
traded direction.

## Timing

- The event is **totally inaccessible**. Pre-arrival `q`, the tax, and positions are
  predictable.
- `K` and safe face-value positions are **continuous** across the event.
- The predictable pre-event equity position realises its gain at the **selected
  successor price, before any rebalancing**: `F_j^+ = F + Theta J_j`, equivalently
  `e_j^+ = e + psi J_j`.
- The rental-flow tax is a **predictable bounded flow with no event atom**. It therefore
  adds no contemporaneous jump direction and does not enlarge the payoff span.
- Continuation policies may jump after the event.

## Coordinates

```
e     = F - qK        public external-wealth coordinate
psi   = Theta - qK    public installed-equity exposure coordinate
B     = psi - e       gross public safe debt
n     = a + e         domestic net foreign assets
```

States are `x = (K, e)`, costates `m = (mu_K, mu_e)`, controls `v = (C, psi, q)`.

The costates are shooting variables, not inherited promises. Under the maintained
one-time-protection convention the surviving promise record is `M_0 = empty`; that
convention is an input to this repository, not something it decides.

## Domains

Enforced by construction, and reported as signed margins rather than clipped:

| Domain | Condition |
|---|---|
| installation | `iota > -1/varphi`, equivalently `q > 0` |
| task shares | `0 < I_0 < I_P < 1` |
| AK existence | `a < exp(u)` with `a = 1 + varphi A_bar`, `u = varphi(r_F_bar + delta)` |
| AK selection | `r_F_bar > g(q_F)`, strictly |
| partial existence | `U_P = r_P_bar q_delta + iota_delta > 0` |
| positive successor wealth | `X_j > 0` for every **active** mark |
| owner event solvency | `1 + pi J_j > 0` for every active mark |
| public debt | `B = psi - e >= 0` |
| transfers | `T = C^W - W_0(K) >= 0` |
| source tax | `0 <= tau <= 1` |
| owner wealth | `a > 0` |
| worker consumption | `C^W > 0` |

An **inactive** mark - both intensities exactly zero - imposes no positive-wealth
requirement, because it contributes no term to the Hamiltonian. This is what makes the
single-AK support restriction reachable through the two-mark code path.

## Transversality conditions

Carried separately, never netted against one another:

- productive-value TVC (AK): `r_F_bar > g(q_F)`, reported as `tvc_margin`;
- worker TVC: `e^{-rho t} V_{j,e} X_j -> 0`, immediate from `V_{j,e} X_j = 1/rho`;
- separate gross-position TVCs for public debt and public equity;
- owner and firm TVCs.

The full-allocation BGP condition `r_F_bar = rho + g_F` is **stronger** than the
world-priced continuation and is reported as `full_bgp_residual`, not imposed. Away from
that locus the world-priced root and the older resource-only formula differ.

## Scaling applied in this implementation

CS011 asks for order-one scaling at the stationary anchor, with unscaled economic
residuals also reported. Blocks G0/N0/N1 do not have a stationary anchor - that is block
N2 - so no anchor-based scaling is applied yet. Instead:

- every `Residual` in the evaluator carries both its raw `value` and a `scaled` value,
  together with the `scale` used, so the unscaled economic residual is always visible;
- scales are the natural magnitude of the quantity compared (`|r_F q_F|` for the price
  equation, `max|H_P|` for productive wealth, `max(|F|, |Theta|)` for the balance-sheet
  round trip), floored at one so that a near-zero quantity cannot manufacture a small
  scaled residual;
- root uncertainty is the converged bracket width, reported per root as
  `bracket_width`, not a solver-reported tolerance.

Anchor-relative scaling should be introduced with N2, when there is an anchor to scale
against.

## What is not scaled, and never should be

Structural gates are binary and lexicographically prior to any residual: root coverage,
branch identity, the strict TVC, rank, nesting, event-map correctness, provenance. A
small residual cannot buy off a failed structural gate, and a structural failure cannot
be averaged into a numerical tolerance.
