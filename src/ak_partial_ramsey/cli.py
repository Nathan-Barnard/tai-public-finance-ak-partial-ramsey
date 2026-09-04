"""Command-line interface: ``preflight`` and ``successors``.

Each stage reads an immutable fixture, writes a distinct report, and exits nonzero on
any structural failure. A stage never overwrites a previous report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import BLOCKS_IMPLEMENTED, RESULT_USE_CEILING, SPEC_ID, SPEC_VERSION
from .errors import (
    ActivationGateError,
    AkPartialRamseyError,
    DomainError,
    NestingError,
    NonFiniteInputError,
    RankFailure,
)
from .evaluator import check_ak_successor, check_event_map_and_exposure
from .fixtures import FIXTURES, get_fixture
from .reporting import Timer, build_report, source_manifest_digest
from .reporting import write_report as _write_report
from .tolerances import DEFAULT_TOLERANCES, SolverTolerances

__all__ = ["main"]


# --- preflight ---------------------------------------------------------------------


def _refusal_checks() -> list[dict[str, Any]]:
    """Structural refusal checks (lane L0).

    Each asserts that a guard actually refuses rather than returning a plausible number.
    These are the checks whose failure would let a NaN, a clipped value, or a false
    nesting claim reach a result.
    """
    from .exposure import Mark, solve_public_exposure
    from .params import Installation, TaskTechnology
    from .primitives import capital_growth, installation_rate
    from .successors.partial import refuse_ak_by_task_share_limit

    checks: list[dict[str, Any]] = []

    def record(name: str, expected: type[BaseException], fn: Any, why: str) -> None:
        try:
            fn()
        except expected as exc:
            checks.append(
                {
                    "name": name,
                    "passed": True,
                    "why": why,
                    "refusal": type(exc).__name__,
                }
            )
        except Exception as exc:
            checks.append(
                {
                    "name": name,
                    "passed": False,
                    "why": why,
                    "refusal": f"wrong error type: {type(exc).__name__}: {exc}",
                }
            )
        else:
            checks.append({"name": name, "passed": False, "why": why, "refusal": "no refusal"})

    inst = Installation(varphi=2.5, delta=0.06)
    record(
        "non_finite_price_refused",
        NonFiniteInputError,
        lambda: capital_growth(float("nan"), inst),
        "a NaN price must raise rather than propagate",
    )
    record(
        "non_finite_infinite_price_refused",
        NonFiniteInputError,
        lambda: installation_rate(float("inf"), inst),
        "an infinite price must raise rather than propagate",
    )
    record(
        "nonpositive_price_refused",
        DomainError,
        lambda: capital_growth(0.0, inst),
        "a nonpositive price is outside the installation domain",
    )
    record(
        "task_share_domain_refused",
        DomainError,
        lambda: TaskTechnology(Z=1.0, I=1.0),
        "a task share must lie strictly inside (0, 1)",
    )
    record(
        "ak_by_task_share_limit_refused",
        NestingError,
        lambda: refuse_ak_by_task_share_limit(0.999999),
        "the AK successor is never built by sending I_P to one",
    )
    record(
        "zero_payoff_vector_refused",
        RankFailure,
        lambda: solve_public_exposure(
            (
                Mark("P", 0.012, 0.010, 0.0, 10.0),
                Mark("F", 0.008, 0.015, 0.0, 8.0),
            ),
            0.4,
            0.04,
            DEFAULT_TOLERANCES,
        ),
        "a zero payoff vector is a rank failure, not a conditioning problem",
    )
    return checks


def _gate_record() -> dict[str, Any]:
    """The five CS011 activation gates this repository refuses to close on its own."""
    return {
        "note": (
            "These remain open in CS011. This repository accepts each as a validated "
            "input and supplies no default for any of them."
        ),
        "open_gates": [
            "inherited state (S_0, M_0)",
            "economic parameter scenario",
            "final synthetic anchor",
            "successor domains for a substantive run",
            "economic materiality thresholds",
        ],
    }


def _cmd_preflight(args: argparse.Namespace) -> int:
    with Timer("preflight") as timer:
        checks = _refusal_checks()
        fixtures = {
            name: {
                "provenance": f.provenance,
                "source_locator": f.source_locator,
                "economic_interpretation": f.economic_interpretation,
                "params_digest": f.params.digest(),
                "has_partial_successor": f.params.has_partial_successor,
            }
            for name, f in FIXTURES.items()
        }
        manifest = source_manifest_digest()
        payload = {
            "refusal_checks": checks,
            "all_refusals_passed": all(c["passed"] for c in checks),
            "fixtures": fixtures,
            "activation_gates": _gate_record(),
            "source_manifest_available": manifest["available"],
        }
    report = build_report(
        stage="preflight",
        payload=payload,
        tolerances_digest=DEFAULT_TOLERANCES.digest(),
        wall_seconds=timer.elapsed,
    )
    ok = bool(payload["all_refusals_passed"]) and manifest["available"]

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        env = report["environment"]
        print(f"{SPEC_ID} v{SPEC_VERSION}  blocks {', '.join(BLOCKS_IMPLEMENTED)}")
        print(f"result-use ceiling: {RESULT_USE_CEILING}")
        print(
            f"python {env['python']}  numpy {env['numpy']}  scipy {env['scipy']}  {env['platform']}"
        )
        print(
            "source manifest: "
            + (
                f"{manifest['sha256'][:16]}...  ({manifest['bytes']} bytes)"
                if manifest["available"]
                else "MISSING - provenance cannot be recorded"
            )
        )
        print(f"tolerances digest: {DEFAULT_TOLERANCES.digest()[:16]}...")
        print("\nstructural refusal checks:")
        for c in checks:
            mark = "ok  " if c["passed"] else "FAIL"
            print(f"  [{mark}] {c['name']:38} {c['refusal']}")
        print("\nfixtures:")
        for name, meta in fixtures.items():
            print(f"  {name:34} {meta['provenance']:16} digest {meta['params_digest'][:16]}...")
        print("\nopen CS011 activation gates (no default supplied here):")
        for g in _gate_record()["open_gates"]:
            print(f"  - {g}")
        print(f"\npreflight: {'PASS' if ok else 'FAIL'}  ({timer.elapsed:.3f}s)")

    if args.out:
        _write_report(Path(args.out) / "preflight.json", report)
    return 0 if ok else 1


# --- successors --------------------------------------------------------------------


def _cmd_successors(args: argparse.Namespace) -> int:
    from .successors.ak import solve_ak_successor
    from .successors.partial import solve_partial_successor

    fixture = get_fixture(args.fixture)
    params = fixture.params
    tol = DEFAULT_TOLERANCES

    payload: dict[str, Any] = {
        "fixture": fixture.as_dict(),
        "tolerances": tol.as_dict(),
    }
    with Timer("successors") as timer:
        ak = solve_ak_successor(params, fixture.ak_root_interval, tol)
        payload["ak_successor"] = ak.as_dict()

        K = fixture.state.get("K", 1.0)
        e = fixture.state.get("e", 0.0)
        ak_eval = check_ak_successor(
            q_F=ak.q_F,
            K=K,
            e=e,
            V_F=ak.V_F(K, e),
            V_F_e=ak.V_F_e(K, e),
            V_F_K=ak.V_F_K(K, e),
            H_F=ak.H_F(K),
            rho=params.preferences.rho,
            rF_bar=params.rates.rF_bar,
            A_bar=params.ak_technology.A_bar,
            varphi=params.installation.varphi,
            delta=params.installation.delta,
            tolerance=tol.identity_atol,
            fd_step=tol.fd_step,
        )
        payload["ak_independent_evaluation"] = ak_eval.as_dict()

        partial = None
        if params.has_partial_successor and fixture.partial_capital_interval:
            partial = solve_partial_successor(params, fixture.partial_capital_interval, tol)
            payload["partial_successor"] = partial.as_dict()
            payload["partial_independent_evaluation"] = _evaluate_partial(partial, params, tol)

        if partial is not None and "q" in fixture.state:
            payload["state_evaluation"] = _evaluate_at_state(params, ak, partial, fixture, tol)

    reports = [ak_eval.as_dict()]
    if "partial_independent_evaluation" in payload:
        reports.append(payload["partial_independent_evaluation"])
    if "state_evaluation" in payload:
        reports.append(payload["state_evaluation"]["independent_evaluation"])
    ok = all(r["passed"] for r in reports)
    payload["all_independent_evaluations_passed"] = ok

    report = build_report(
        stage="successors",
        payload=payload,
        parameters_digest=params.digest(),
        tolerances_digest=tol.digest(),
        wall_seconds=timer.elapsed,
    )

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        _print_successors(fixture, ak, payload, timer.elapsed, ok)
    if args.out:
        _write_report(Path(args.out) / f"successors-{fixture.name}.json", report)
    return 0 if ok else 1


def _evaluate_partial(partial: Any, params: Any, tol: SolverTolerances) -> dict[str, Any]:
    import numpy as np

    from .evaluator import EvaluationReport, check_partial_present_value, check_partial_successor

    lo, hi = partial.certified_domain
    # Held-out nodes: interior points that are not the interpolation anchor.
    nodes = list(np.linspace(lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo), 15))
    report = check_partial_successor(
        K_nodes=nodes,
        q_P=[partial.q_P(float(K)) for K in nodes],
        q_P_prime=[partial.q_P_derivative(float(K)) for K in nodes],
        H_P=[partial.H_P(float(K)) for K in nodes],
        rho=params.preferences.rho,
        rP_bar=params.rates.rP_bar,
        Z=params.partial_technology.Z,
        I_P=params.partial_technology.I,
        varphi=params.installation.varphi,
        delta=params.installation.delta,
        tolerance=tol.cross_route_atol,
        fd_step=tol.fd_step,
    )
    # Third, genuinely independent route to productive wealth, at held-out states.
    pv = tuple(
        check_partial_present_value(
            K_0=float(K),
            price_graph=partial.q_P,
            H_reported=partial.H_P(float(K)),
            rP_bar=params.rates.rP_bar,
            Z=params.partial_technology.Z,
            I_P=params.partial_technology.I,
            varphi=params.installation.varphi,
            delta=params.installation.delta,
            K_star=partial.point.K_star,
            iota_delta=partial.point.iota_delta,
            tolerance=tol.cross_route_atol,
        )
        for K in (nodes[3], nodes[7], nodes[11])
    )
    return EvaluationReport(report.label, report.residuals + pv).as_dict()


def _evaluate_at_state(
    params: Any, ak: Any, partial: Any, fixture: Any, tol: SolverTolerances
) -> dict[str, Any]:
    from .assembly import evaluate_state

    K = fixture.state["K"]
    e = fixture.state["e"]
    q = fixture.state["q"]
    a = fixture.state.get("a", 1.0)
    C = params.preferences.rho * (e + partial.H_P(K))  # a positive consumption probe
    ev = evaluate_state(
        params,
        ak,
        partial,
        K=K,
        e=e,
        q=q,
        C=C,
        a=a,
        q_dot=0.0,
        tolerances=tol,
    )
    independent = check_event_map_and_exposure(
        e=e,
        psi=ev.exposure.psi,
        q=q,
        K=K,
        C=C,
        rho=params.preferences.rho,
        marks=[
            {
                "label": m.label,
                "lambda_physical": m.lambda_physical,
                "lambda_star": m.lambda_star,
                "q_successor": ev.mark_set.successor_prices[m.label],
                "H": ev.mark_set.successor_wealth_at_zero_exposure[m.label] - e,
                "X_reported": ev.exposure.successor_wealth[m.label],
            }
            for m in ev.mark_set.marks
        ],
        tolerance=tol.identity_atol,
    )
    return {
        "note": (
            "This is an arbitrary manufactured probe state, not an equilibrium point. "
            "q_dot is supplied as zero, so the recovered source tax is conditional on "
            "that probe value and is not a solved path quantity: the pre-arrival price "
            "path comes from the transition solve, which is block N4 and out of scope. "
            "The implementation margins below are therefore NOT expected to be strict. "
            "A strictly interior pre-arrival state is an equilibrium object; locating "
            "one needs the stationary continuation (N2) and the transition solve (N4). "
            "What this stage checks is that a violated margin is reported as a signed "
            "distance and never clipped back into the interior."
        ),
        "evaluation": ev.as_dict(),
        "independent_evaluation": independent.as_dict(),
    }


def _print_successors(
    fixture: Any, ak: Any, payload: dict[str, Any], elapsed: float, ok: bool
) -> None:
    print(f"fixture: {fixture.name}  ({fixture.provenance})")
    print(f"  {fixture.source_locator}")
    print(f"result-use ceiling: {RESULT_USE_CEILING}\n")
    print("AK successor")
    print(f"  selected q_F        {ak.q_F!r}   branch {ak.branch}")
    print(f"  iota_F              {ak.iota_F!r}")
    print(f"  g_F                 {ak.g_F!r}")
    print(f"  TVC margin r_F - g  {ak.tvc_margin!r}")
    print(f"  recovered tau_F     {ak.recovered_tau_F!r}")
    print("  enumerated roots:")
    for c in ak.candidates:
        state = "ACCEPTED" if c.accepted else "rejected"
        print(f"    q = {c.q!r:24} {state:9} {c.branch:28} {c.reason}")
        print(
            f"        tvc_margin {c.tvc_margin:+.12f}   "
            f"f_poly {c.residual_polynomial:+.3e}   f_level {c.residual_level:+.3e}"
        )
    ev = payload["ak_independent_evaluation"]
    print(
        f"  independent evaluation: {'PASS' if ev['passed'] else 'FAIL'} "
        f"({ev['n_residuals']} residuals, worst "
        f"{ev['worst_residual']['name']} = {ev['worst_residual']['scaled']:.3e})"
    )

    if "partial_successor" in payload:
        ps = payload["partial_successor"]
        pd = ps["diagnostics"]
        print("\nPartial successor (fixed I_P, finite task share)")
        print(f"  K_P*                {ps['stationary_point']['K_star']!r}")
        print(f"  q_delta             {ps['stationary_point']['q_delta']!r}")
        print(
            f"  eigenvalues         nu- {ps['linearization']['nu_minus']:+.9f}   "
            f"nu+ {ps['linearization']['nu_plus']:+.9f}"
        )
        print(f"  certified domain    {ps['certified_domain']}")
        for key in (
            "wealth_route_max_gap",
            "derivative_identity_max_gap",
            "manifold_invariance_max_residual",
            "max_ivp_bvp_difference",
            "max_offset_size_spread",
        ):
            print(f"  {key:34} {pd[key]:.4e}")
        pe = payload["partial_independent_evaluation"]
        print(
            f"  independent evaluation: {'PASS' if pe['passed'] else 'FAIL'} "
            f"(worst {pe['worst_residual']['name']} = "
            f"{pe['worst_residual']['scaled']:.3e})"
        )

    if "state_evaluation" in payload:
        se = payload["state_evaluation"]
        ev2 = se["evaluation"]
        print("\nEquation core at the fixture state")
        print(f"  psi                 {ev2['exposure']['psi']!r}")
        print(f"  orthogonality       {ev2['exposure']['orthogonality_residual']:.3e}")
        print(f"  U                   {ev2['total_valuation_residual_U']:.6e}")
        print(f"  D                   {ev2['investment_wedge_D']:.6e}")
        print(f"  private pi          {ev2['private_portfolio']['pi']!r}")
        m = ev2["implementation_margins"]
        print(
            f"  smallest margin     {m['smallest_margin_name']} = "
            f"{m['smallest_margin_value']:.6e}  (all strict: {m['all_strict']})"
        )
        if not m["all_strict"]:
            print(
                "    note: this is an arbitrary probe state, not an equilibrium point, "
                "so a non-strict\n"
                "    margin is expected. It is reported as a signed distance and not "
                "clipped. Locating a\n"
                "    strictly interior state needs blocks N2 and N4, which are out of "
                "scope here."
            )
        ie = se["independent_evaluation"]
        print(
            f"  independent evaluation: {'PASS' if ie['passed'] else 'FAIL'} "
            f"(worst {ie['worst_residual']['name']} = "
            f"{ie['worst_residual']['scaled']:.3e})"
        )

    print(f"\nsuccessors: {'PASS' if ok else 'FAIL'}  ({elapsed:.3f}s)")


# --- entry point -------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ak-partial-ramsey",
        description=(
            f"{SPEC_ID} v{SPEC_VERSION} blocks {', '.join(BLOCKS_IMPLEMENTED)}. "
            f"Result-use ceiling: {RESULT_USE_CEILING}. No output is an equilibrium "
            "result, a Ramsey optimum, a calibration, or a welfare result."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pre = sub.add_parser(
        "preflight",
        help="environment, provenance, structural refusal checks, fixture inventory",
    )
    pre.add_argument("--json", action="store_true", help="emit the full JSON report")
    pre.add_argument("--out", help="directory to write preflight.json into")
    pre.set_defaults(func=_cmd_preflight)

    succ = sub.add_parser(
        "successors",
        help="solve the AK and partial successor services and evaluate them independently",
    )
    succ.add_argument(
        "--fixture",
        default="single-ak-two-root",
        choices=sorted(FIXTURES),
        help="fixture to run",
    )
    succ.add_argument("--json", action="store_true", help="emit the full JSON report")
    succ.add_argument("--out", help="directory to write the report into")
    succ.set_defaults(func=_cmd_successors)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ActivationGateError as exc:
        print(f"refused (open CS011 activation gate): {exc.message}", file=sys.stderr)
        print(json.dumps(exc.as_dict(), indent=2, default=str), file=sys.stderr)
        return 3
    except AkPartialRamseyError as exc:
        print(f"refused ({exc.kind}): {exc.message}", file=sys.stderr)
        print(json.dumps(exc.as_dict(), indent=2, default=str), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
