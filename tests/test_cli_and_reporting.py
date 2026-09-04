"""CLI stages, provenance, and hashing.

CS011 requires every run to record the specification fingerprint, environment lock,
solver versions, parameter and configuration hashes, and timings. These tests assert the
report actually carries them, that hashes are stable and discriminating, and that a
stage never overwrites prior evidence.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from ak_partial_ramsey import RESULT_USE_CEILING, SPEC_ID, SPEC_VERSION
from ak_partial_ramsey.cli import main
from ak_partial_ramsey.hashing import canonical_json, digest_mapping
from ak_partial_ramsey.params import Preferences
from ak_partial_ramsey.reporting import build_report, write_report
from ak_partial_ramsey.tolerances import DEFAULT_TOLERANCES, SolverTolerances

# --- hashing ---------------------------------------------------------------------------


def test_float_hashing_round_trips_exactly():
    """repr-based serialisation round-trips floats; a fixed-precision format would not."""
    x = 0.1 + 0.2  # 0.30000000000000004
    y = 0.3
    assert x != y
    assert digest_mapping({"v": x}) != digest_mapping({"v": y})


def test_digest_is_stable_across_key_order():
    assert digest_mapping({"a": 1.0, "b": 2.0}) == digest_mapping({"b": 2.0, "a": 1.0})


def test_digest_discriminates_small_parameter_changes(two_mark_fixture):
    p = two_mark_fixture.params
    q = dataclasses.replace(p, preferences=Preferences(rho=0.04 + 1e-12))
    assert p.digest() != q.digest()


def test_digest_is_reproducible(two_mark_fixture):
    p = two_mark_fixture.params
    assert p.digest() == p.digest()
    assert canonical_json(p.as_dict()) == canonical_json(p.as_dict())


def test_tolerances_hash_separately_from_parameters(two_mark_fixture):
    """Economic primitives and solver tolerances are separate objects with separate hashes."""
    p = two_mark_fixture.params
    assert p.digest() != DEFAULT_TOLERANCES.digest()
    # A tolerance change must not look like a parameter change.
    other = SolverTolerances(root_scan_points=999)
    assert other.digest() != DEFAULT_TOLERANCES.digest()
    assert p.digest() == p.digest()
    # And no tolerance field leaks into the parameter serialisation.
    serialised = canonical_json(p.as_dict())
    for field in ("root_xtol", "ivp_rtol", "bvp_tol", "manifold_offset"):
        assert field not in serialised


def test_tolerances_reject_degenerate_settings():
    from ak_partial_ramsey.errors import DomainError

    with pytest.raises(ValueError, match="root_scan_points"):
        SolverTolerances(root_scan_points=2)
    with pytest.raises(ValueError, match="manifold_offset_ratio"):
        SolverTolerances(manifold_offset_ratio=1.0)
    with pytest.raises(DomainError):
        SolverTolerances(identity_atol=0.0)


# --- reports ----------------------------------------------------------------------------


def test_report_carries_full_provenance(two_mark_fixture):
    p = two_mark_fixture.params
    report = build_report(
        stage="test",
        payload={"x": 1},
        parameters_digest=p.digest(),
        tolerances_digest=DEFAULT_TOLERANCES.digest(),
        wall_seconds=1.5,
    )
    assert report["specification"]["id"] == SPEC_ID
    assert report["specification"]["version_implemented_against"] == SPEC_VERSION
    assert report["result_use_ceiling"] == RESULT_USE_CEILING == "exploratory_only"
    assert report["parameters_digest"] == p.digest()
    assert report["tolerances_digest"] == DEFAULT_TOLERANCES.digest()
    assert report["peak_memory_bytes"] > 0
    assert report["wall_seconds"] == 1.5
    env = report["environment"]
    for key in ("python", "numpy", "scipy", "platform", "package_version"):
        assert env[key]
    assert report["source_manifest"]["available"]
    assert len(report["source_manifest"]["sha256"]) == 64
    assert len(report["report_digest"]) == 64


def test_report_is_json_serialisable(two_mark_fixture):
    report = build_report(stage="test", payload={"x": [1.0, 2.0]})
    json.loads(json.dumps(report, default=str))


def test_reports_never_overwrite_prior_evidence(tmp_path):
    path = tmp_path / "run.json"
    write_report(path, build_report(stage="a", payload={}))
    with pytest.raises(FileExistsError) as exc:
        write_report(path, build_report(stage="b", payload={}))
    assert "never overwrite" in str(exc.value)


# --- CLI ----------------------------------------------------------------------------------


def test_preflight_exits_zero(capsys):
    assert main(["preflight"]) == 0
    out = capsys.readouterr().out
    assert "preflight: PASS" in out
    assert "exploratory_only" in out
    # The open activation gates are stated every run.
    assert "inherited state (S_0, M_0)" in out
    assert "economic materiality thresholds" in out


def test_preflight_json_report(capsys):
    assert main(["preflight", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["payload"]["all_refusals_passed"]
    assert len(report["payload"]["activation_gates"]["open_gates"]) == 5
    names = {c["name"] for c in report["payload"]["refusal_checks"]}
    assert "ak_by_task_share_limit_refused" in names
    assert "zero_payoff_vector_refused" in names


def test_successors_on_the_theory_fixture(capsys):
    assert main(["successors", "--fixture", "single-ak-two-root"]) == 0
    out = capsys.readouterr().out
    assert "ACCEPTED" in out and "rejected" in out
    assert "lower_strict_tvc" in out
    assert "upper_tvc_violating" in out
    assert "successors: PASS" in out


def test_successors_on_the_manufactured_two_mark_fixture(capsys):
    assert main(["successors", "--fixture", "manufactured-two-mark"]) == 0
    out = capsys.readouterr().out
    assert "Partial successor" in out
    assert "successors: PASS" in out


def test_successors_json_report_records_every_root(capsys):
    assert main(["successors", "--fixture", "single-ak-two-root", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    payload = report["payload"]
    candidates = payload["ak_successor"]["candidates"]
    assert len(candidates) == 2
    assert sum(c["accepted"] for c in candidates) == 1
    rejected = next(c for c in candidates if not c["accepted"])
    assert rejected["reason"] == "productive_value_tvc_violated"
    assert payload["all_independent_evaluations_passed"]
    assert payload["fixture"]["provenance"] == "theory_supplied"
    assert report["parameters_digest"]


def test_report_is_written_and_not_overwritten(tmp_path):
    assert main(["preflight", "--out", str(tmp_path)]) == 0
    written = tmp_path / "preflight.json"
    assert written.is_file()
    report = json.loads(written.read_text())
    assert report["stage"] == "preflight"
    with pytest.raises(FileExistsError):
        main(["preflight", "--out", str(tmp_path)])


def test_unknown_fixture_is_rejected():
    with pytest.raises(SystemExit):
        main(["successors", "--fixture", "does-not-exist"])


def test_cli_help_states_the_result_use_ceiling(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "exploratory_only" in capsys.readouterr().out
