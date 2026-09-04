# Equation crosswalk

Block G0 of CS011 requires "the completed theory crosswalk as the solver-independent
table of primitives, units, timing, states, controls, successor outputs, domains, and
TVCs". This file is that table: it maps every equation implemented in this repository to
its numbered equation in the theory packets.

Two source packets are referenced:

| Short name | File |
|---|---|
| **single-AK** | `research-notes/single-ak-small-open-full-automation-solution.md` |
| **two-mark** | `research-notes/ak-or-partial-automation-small-open-ramsey-derivation.md` |

Equation tags such as `(AK.1)`, `(P.10)`, `(G0.15)`, `(EU.12)` are the packets' own.
Sections in the two-mark packet are cited by number.

---

## 1. Production, installation, prices, factor payments

Module: `ak_partial_ramsey.primitives`

| Code | Equation | Source |
|---|---|---|
| `omega_Z(tech)` | `Omega_Z(I) = (Z/(1-I))^(1-I) I^(-I)` | two-mark §1 |
| `task_output(K, tech)` | `Y_s(K) = Omega_Z(I_s) K^{I_s}` | two-mark §1 |
| `task_wage(K, tech)` | `W_s(K) = (1-I_s) Y_s(K)` | two-mark §1 |
| `task_rental(K, tech)` | `R_s(K) = I_s Y_s(K)/K` | two-mark §1 |
| `task_rental_derivative` | `R_s'(K) = -(1-I_s) R_s(K)/K` | implied; used in the partial linearisation |
| `ak_output(K, tech)` | `Y_F(K) = A_bar K` | single-AK frozen contract; two-mark §1 |
| `ak_wage(K, tech)` | `W_F = 0` | single-AK frozen contract |
| `ak_rental(K, tech)` | `R_F = A_bar` | single-AK frozen contract |
| `installation_function(z, inst)` | `Phi(z) = log(1 + varphi z)/varphi`, `z > -1/varphi` | single-AK frozen contract; `(EU.2)` |
| `installation_rate(q, inst)` | `iota(q) = (q-1)/varphi` | single-AK frozen contract; two-mark §1 |
| `capital_growth(q, inst)` | `g(q) = log(q)/varphi - delta` | single-AK frozen contract; two-mark §1 |
| `capital_growth_derivative(q, inst)` | `g_q = 1/(varphi q)` | two-mark §5 |
| `installation_rate_derivative(inst)` | `iota_q = 1/varphi` | two-mark §5 |
| `zero_growth_price(inst)` | `q_delta = exp(varphi delta)` | `(P.22)`; two-mark §2 |
| `installation_domain_margin(q, inst)` | `iota(q) + 1/varphi > 0` | single-AK scalar-root fixture table |

**Notation reconciliation.** The single-AK packet writes pre-arrival output as
`Y_0(K) = A_0 K^{I_0}` and notes that "the equivalent fixed-task-share level formula for
`A_0` is inherited from the source notebook". The two-mark packet gives that formula
explicitly. The two are the same function with

```
A_0 = Omega_Z(I_0).
```

This repository implements `Omega_Z` and does not carry a separate `A_0`. This is a
difference of explicitness between the packets, not a disagreement.

The identity `q g_q = iota_q = 1/varphi` is what makes the productive-wealth derivative
identity `H'(K) = q(K)` hold exactly. It is asserted in
`tests/test_partial_successor.py::test_wealth_derivative_identity`.

---

## 2. AK successor

Module: `ak_partial_ramsey.successors.ak`

| Code | Equation | Source |
|---|---|---|
| `ak_price_residual_level` | `r_F_bar q_F = A_bar - iota(q_F) + q_F g(q_F)` | `(AK.1)` |
| `ak_scalar_coefficients` | `a = 1 + varphi A_bar`, `u = varphi(r_F_bar + delta)` | `(AK.2)` |
| `ak_price_residual_polynomial` | `f(q) = q log q - (1+u) q + a` | `(AK.2)` |
| `ak_lambert_roots` | `q_L = -a/W_{-1}(-a e^{-(1+u)})`, `q_H = -a/W_0(...)` | `(AK.3)`, `(EU.4)` |
| `coefficients["discriminant"]` | two roots iff `a < exp(u)`; double root iff `a = exp(u)`; none iff `a > exp(u)` | `(EU.3)` and Proposition 1 |
| `AkRootCandidate.tvc_margin` | `r_F_bar - g(q_F) > 0`, equivalently `q < q_m = exp(u)` | `(AK.4)` |
| `AkSuccessor.H_F(K)` | `H_F(K) = q_F K` | `(AK.5)` |
| `AkSuccessor.X_F(K, e)` | `X_F = e + q_F K > 0` | `(AK.5)` |
| `AkSuccessor.C_F(K, e)` | `C_F^W = rho X_F` | `(AK.7)` |
| `AkSuccessor.V_F(K, e)` | `V_F = (1/rho) log(rho X_F) + (r_F_bar - rho)/rho^2` | `(AK.8)` |
| `AkSuccessor.V_F_e` | `V_{F,e} = 1/(rho X_F)` | `(AK.10)` |
| `AkSuccessor.V_F_K` | `V_{F,K} = q_F V_{F,e}` | `(AK.10)` |
| `AkSuccessor.full_bgp_residual` | `r_F_bar - rho - g_F`, zero only on the full-allocation BGP locus | `(AK.11)` |
| `AkSuccessor.recovered_tau_F` | `tau_F = 1 - [r_F_bar q_F + iota(q_F) - q_F g(q_F)]/A_bar = 0` | `(G0.16)` |

**Branch selection.** Only the lower root satisfies the strict productive-value TVC. The
double root has a zero TVC margin and the upper root a negative one; both are retained
with their rejection reason. Branch labels are the position relative to the analytic
minimiser `q_m`, never a sorted-list index.

---

## 3. Partial-automation successor

Module: `ak_partial_ramsey.successors.partial`. All from two-mark §2.

| Code | Equation |
|---|---|
| `partial_field(K, q)` | `K' = g(q) K`, `q' = [r_P_bar - g(q)] q - R_P(K) + iota(q)` |
| `PartialStationaryPoint.q_delta` | `q_delta = exp(varphi delta)` |
| `PartialStationaryPoint.U_P` | `U_P = r_P_bar q_delta + iota_delta` |
| `PartialStationaryPoint.K_star` | `K_P* = [I_P Omega_Z(I_P)/U_P]^{1/(1-I_P)}`, exists iff `U_P > 0` |
| `PartialLinearization.nu_pm` | `nu_pm = [r_P_bar +/- sqrt(r_P_bar^2 + 4(1-I_P) U_P/(varphi q_delta))]/2` |
| `characteristic_residual` | `nu^2 - r_P_bar nu - (1-I_P) U_P/(varphi q_delta) = 0` |
| `manifold_slope` | `q_P'(K_P*) = varphi q_delta nu_- / K_P*` |
| `PartialSuccessor.H_P_derivative` | `H_P'(K) = q_P(K)` |
| `PartialSuccessor.H_P_algebraic` | `r_P_bar H_P = Y_P - iota(q_P) K + q_P g(q_P) K` |
| `PartialSuccessor.H_anchor` | `H_P(K_P*) = [Y_P(K_P*) - iota_delta K_P*]/r_P_bar` |
| `check_partial_present_value` | `H_P(K_0) = int_0^inf e^{-r_P t}{Y_P(K_t) - iota[q_P(K_t)]K_t} dt` |
| `PartialSuccessor.X_P` | `X_P = e + H_P(K)` |
| `PartialSuccessor.C_P` | `C_P^W = rho X_P` |
| `PartialSuccessor.V_P` | `V_P = (1/rho) log(rho X_P) + (r_P_bar - rho)/rho^2` |
| `PartialSuccessor.V_P_e` / `V_P_K` | `V_{P,e} = 1/(rho X_P)`, `V_{P,K} = q_P(K) V_{P,e}` |

`refuse_ak_by_task_share_limit` implements the two-mark packet's "AK non-limit" row:
`I_P -> 1` is never used as the AK solver.

---

## 4. Event map and payoff geometry

Modules: `ak_partial_ramsey.events`, `ak_partial_ramsey.canonical`

| Code | Equation | Source |
|---|---|---|
| coordinates | `e = F - qK`, `psi = Theta - qK`, `B = psi - e` | `(G0.11)`; two-mark §4 |
| `jump_payoff` | `J_j(K,q) = [q_j(K) - q]/q` | `(G0.2)` region; two-mark §4 |
| `jump_payoff_dq` | `J_{j,q} = -(1 + J_j)/q` | two-mark §4 |
| `jump_payoff_dK` | `J_{j,K} = q_j'(K)/q`; zero for AK | two-mark §4 |
| `event_level` | `F_j^+ = F + Theta J_j` | `(G0.5)`; two-mark §4 |
| `event_normalized` | `e_j^+ = e + psi J_j` | `(G0.13)`; two-mark §4 |
| `owner_event` | `a_j^+ = a(1 + pi J_j)` | `(G0.5)`, `(G0.17)`; two-mark §8 |
| `national_net_foreign_assets` | `n = a + e` | `(G0.9)`; two-mark §8 |
| `national_event_jump` | `n_j^+ - n = (psi + pi a) J_j` | `(G0.10)`; two-mark §8 |
| `risk_neutral_drag` | `Lambda = sum_j lambda_j_star J_j` | two-mark §4 |

**Timing.** The event is totally inaccessible. `K` and safe face values are continuous
across it. The predictable pre-event equity position realises its gain at the selected
successor price **before** any rebalancing. `round_trip_diagnostics` checks that the
level and normalized statements of this agree numerically rather than assuming it.

---

## 5. Pre-arrival canonical system

Module: `ak_partial_ramsey.canonical`. All from two-mark §5, with the single-AK
specialisations from the single-AK packet's Step 2.

| Code | Equation | Source |
|---|---|---|
| `fiscal_valuation_residual` | `u_j = lambda_j V_{j,e} - lambda_j_star mu_e = lambda_j/(rho X_j) - lambda_j_star/C` | two-mark §5 |
| `public_exposure_residual` | `sum_j u_j J_j = 0` | two-mark §5 (the projection condition) |
| `public_exposure_residual_derivative` | `H_{psi psi} = -sum_j lambda_j J_j^2/(rho X_j^2) < 0` | two-mark §6 |
| `successor_wealth_interval` | `I_X = {psi : X_j > 0 for every active j}` | two-mark §6 |
| consumption FOC | `mu_e = 1/C` | two-mark §5; `(P.4)` |
| `total_valuation_residual` | `U = u_P + u_F` | two-mark §5 |
| `investment_wedge` | `D = mu_K - q mu_e = varphi psi U / K` | two-mark §5 |
| `consumption_growth` | `C'/C = r_0_bar + lambda_Sigma_star - rho - lambda + U/mu_e` | two-mark §5 |
| `state_K_dot` | `K' = g(q) K` | two-mark §5 |
| `state_e_dot` | `e' = r_0_bar e + Y_0(K) - iota(q)K - C - psi Lambda` | two-mark §5; `(P.2)`, `(G0.12)` |
| `costate_e_dot` | `mu_e' = (rho + lambda - r_0_bar) mu_e - sum_j lambda_j V_{j,e}` | two-mark §5 |
| `costate_K_dot` | `mu_K' = (rho+lambda-g) mu_K - mu_e[R_0 - iota] - sum_j lambda_j V_{j,K} - psi sum_j u_j J_{j,K}` | two-mark §5 |
| `control_map_residual` | `Gamma(z,v)` = (consumption, exposure, investment) conditions | two-mark §6 |
| `tax_identity_residual` | `mu_e tau_0 R_0 = (rho+lambda-g) D - D' - psi sum_j u_j J_{j,K}` | two-mark §7 |
| `private_portfolio_residual` | `sum_j lambda_j J_j/(1 + pi J_j) = Lambda` | two-mark §8 |

### Single-AK specialisations

| Code | Equation | Source |
|---|---|---|
| `single_ak_consumption_growth` | `C'/C = r_0_bar + lambda_F_star - lambda - rho` | `(P.9)` |
| `single_ak_q_dot` | `q' = [r_0_bar + lambda_F_star - g(q)]q - R_0(K) + iota(q) - lambda_F_star q_F` | `(P.10)` |
| `single_ak_productive_wealth_rhs` | `(r_0_bar + lambda_F_star) H_0 = Y_0 - iota(q)K + q g(q)K + lambda_F_star q_F K` | `(P.16)` |
| `single_ak_exposure_closed_form` | `psi = [(lambda/lambda_F_star) X_0 - e - q_F K]/J_F` | `(P.20)` |
| fiscal-to-market alignment | `lambda V_{F,e} = lambda_F_star mu_e` | `(P.6)`, `(P.7)` |
| investment condition | `mu_K = q mu_e` | `(P.8)` |
| zero pre-arrival tax | `tau_0 = 0` | `(P.12)` |
| owner portfolio | `1 + pi J_F = lambda/lambda_F_star` | `(R.2)` |

**The support restriction is algebraic, not a limit.** Setting `p_P = lambda_P_star = 0`
makes `lambda_P = 0`, so `u_P = 0` identically. With `J_F` nonzero the exposure condition
then forces `u_F = 0`, hence `U = D = 0`, and the saving, investment, capital-costate,
tax-recovery and portfolio equations collapse term by term onto the single-mark system.
This is the two-mark packet's own reconciliation record and is asserted in
`tests/test_single_ak_reduction.py`. It is **not** an `I_P` limit and claims no uniform
convergence of global policies.

---

## 6. Recovery

Module: `ak_partial_ramsey.recovery`

| Code | Equation | Source |
|---|---|---|
| `recover_positions` | `F = e + qK`, `Theta = psi + qK`, `B = psi - e` | `(G0.14)`; two-mark §8 |
| `recover_transfer` | `T = C^W - W_0(K)` | `(G0.14)`; two-mark §8 |
| `recover_source_tax` | `tau_0 = 1 - [(r_0_bar - g)q + iota - q' - q Lambda]/R_0(K)` | two-mark §8 |
| `recover_ak_source_tax` | `tau_0 = 1 - [r_0_bar q + iota - q' - qg - lambda_F_star(q_F - q)]/R_0(K)` | `(G0.15)` |
| `recover_ak_successor_tax` | `tau_F = 1 - [r_F_bar q_F + iota(q_F) - q_F g(q_F)]/A_bar` | `(G0.16)` |
| `recover_owner` | `C^O = rho a`, `a' = (r_0_bar - rho - pi Lambda) a`, `vartheta = pi a` | `(G0.17)`; two-mark §8 |
| `recover_foreign_residual` | `Theta_for = qK - Theta - vartheta`, `D_for = B - d^O` | `(G0.7)`; two-mark §8 |

The two tax arrangements are algebraically identical once `Lambda` collapses to
`lambda_F_star J_F`, but they are written independently and compared in
`tests/test_single_ak_reduction.py::test_zero_source_tax_is_recovered`.

**Margins are distances, not constraints.** `implementation_margins` returns
`B >= 0`, `T >= 0`, `0 <= tau <= 1`, `a > 0`, `C^W > 0`, `X_j > 0`, `1 + pi J_j > 0` and
the installation-domain margin as signed distances. Nothing is clipped. A nonpositive
margin means the candidate is not implementable on the smooth branch and must be
rejected or re-solved with the corresponding multiplier retained.

---

## 7. Notation collisions between the packets

Three symbols are reused across the two packets with different meanings. They are
recorded here because they are a genuine hazard for a reader moving between the notes,
and because the code deliberately gives each a distinct name.

| Symbol | Meaning in single-AK | Meaning in two-mark | Name in this code |
|---|---|---|---|
| `D` | `D = R_0 q_delta + (q_delta-1)/varphi - lambda_F_star q_F`, the scalar marginal-product target at `(EU.11)` | `D := mu_K - q mu_e`, the investment wedge, §5 | the two-mark object is `investment_wedge`; the single-AK scalar is local to `partial_stationary_point` as `U_P`'s analogue and is not exported |
| `R_0` | both the rental rate `R_0(K)` and the calligraphic `R_0 = r_0_bar + lambda_F_star` at `(EU.1)` | rental rate only | `task_rental(K, tech)` for the rental; the sum is written inline |
| `H` | productive wealth `H_0`, `H_F` | productive wealth `H_P`, `H_F`, and separately the Hamiltonian `\mathcal H` | productive wealth is `H_*`; the Hamiltonian appears only through `control_map_residual` |

These are notation collisions, not contradictions. **No substantive disagreement was
found between the two packets** on any equation, unit, timing convention, or interface
used by blocks G0, N0 or N1. The two-mark packet's own reconciliation record (its §3)
states the same conclusion, and this implementation reproduces the crosswalk it requires.
