"""Counterexamples for clipped singleton global claims and warm-only fits."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from CliPP2.api import fit_fixed_objective, prepare_problem
from CliPP2.config import resolve_fit_config
from CliPP2.core.fusion import solver
from CliPP2.core.fusion.graph import build_complete_uniform_graph
from CliPP2.io.tumor_txt import load_tumor_txt, write_tumor_txt


def _case(tmp_path, *, alt=10, major=1, purity=1, total=100):
    path = tmp_path / "clipping.tsv"
    write_tumor_txt(path, pd.DataFrame([
        dict(mutation_id=f"m{i}", sample_id="R1", alt_count=alt, ref_count=total-alt,
             count_observed=1, purity=purity, normal_cn=2, segment_id=f"s{i}",
             cn_state_id="clonal", cn_state_fraction=1, allele_a_cn=major, allele_b_cn=1)
        for i in range(2)
    ]))
    config = resolve_fit_config(device="cpu", dtype="float64", lambda_value=.1,
                                graph=build_complete_uniform_graph(2),
                                outer_max_iter=4, inner_max_iter=40)
    return load_tumor_txt(path), config


@pytest.mark.parametrize("lam", [0.0, .1, 100.0])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("objective_shape", ["auto", "generic_nonconvex"])
def test_supplied_lower_plateau_cannot_receive_false_global_certificate(tmp_path, lam, dtype, objective_shape):
    data, config = _case(tmp_path)
    config = replace(config, lambda_value=lam, runtime=replace(config.runtime, dtype=dtype),
                     solver=replace(config.solver, objective_shape=objective_shape))
    plateau = np.full((2, 1), config.eps)
    fit = fit_fixed_objective(data, config, phi_start=plateau)
    # Independent Bernoulli calculation; both feasible points have zero fusion.
    better = -2 * (10 * np.log(.1) + 90 * np.log(.9))
    assert not (fit.certificate.global_optimum and fit.objective.total > better + 1e-8)
    # The mandatory scalar-pilot comparison must also escape this known plateau.
    np.testing.assert_allclose(fit.objective.total, better, rtol=0, atol=1e-8)
    np.testing.assert_allclose(fit.phi, .2, rtol=0, atol=1e-6)
    assert fit.certificate.tolerance == .004
    assert not fit.certificate.global_optimum  # The whole-box proof still fails.


def test_global_gate_is_independent_of_pilot_escape(tmp_path, monkeypatch):
    data, config = _case(tmp_path)
    monkeypatch.setattr(solver, "_clipped_singleton_start_needs_pilot", lambda *a: False)
    fit = fit_fixed_objective(data, config, phi_start=np.full((2, 1), config.eps))
    assert fit.certificate.certified  # Stationary on a flat, strictly worse region.
    assert fit.objective.total > 276
    assert not fit.certificate.global_optimum
    assert fit.provenance.global_optimality_basis == "not_certified"


def test_zero_alt_clipped_singleton_has_valid_convex_global_authority(tmp_path):
    data, config = _case(tmp_path, alt=0)
    fit = fit_fixed_objective(data, config, phi_start=np.full((2, 1), config.eps))
    np.testing.assert_allclose(fit.objective.total, -200 * np.log1p(-config.eps), atol=1e-9)
    assert fit.certificate.global_optimum
    assert fit.provenance.global_optimality_basis == "convex_clipped_singleton_objective_plus_kkt_v1"


@pytest.mark.parametrize("objective_shape", ["auto", "generic_nonconvex"])
def test_warm_plateau_compares_scalar_pilot_with_defaults_disabled(tmp_path, objective_shape):
    data, config = _case(tmp_path)
    config = replace(config, solver=replace(config.solver, objective_shape=objective_shape))
    problem = prepare_problem(data, config)
    previous = solver.fit_prepared(problem, .1, config.solver)
    plateau = torch.full_like(previous.state.phi, config.eps)
    warm = replace(previous.state, phi=plateau,
                   warm_state=replace(previous.state.warm_state, phi=plateau))
    fit = solver.fit_prepared(problem, .2, config.solver, warm_state=warm,
                              include_default_starts=False)
    np.testing.assert_allclose(fit.phi, .2, atol=1e-6)
    assert not fit.certificate.global_optimum


def test_working_precision_clipping_keeps_pilot_when_source_is_just_outside(tmp_path):
    data, config = _case(tmp_path, purity=.9)
    config = replace(config, runtime=replace(config.runtime, dtype="float32"))
    problem = prepare_problem(data, config)
    phi = torch.full((2, 1), config.eps / .45, dtype=torch.float32)
    source_mass = problem.source_model.slope[..., 0] * phi.numpy()
    working_mass = problem.model.slope[..., 0] * phi
    assert np.all(source_mass > config.eps)
    assert bool((working_mass <= config.eps).all())
    assert solver._clipped_singleton_start_needs_pilot(problem, phi)
    fit = solver.fit_prepared(problem, .1, config.solver, phi_start=phi,
                              include_default_starts=False)
    np.testing.assert_allclose(fit.phi, .1 / .45, atol=1e-6)
    assert not fit.certificate.global_optimum


def test_nearby_pilot_across_clipping_threshold_is_not_deduplicated(tmp_path, monkeypatch):
    # Keep the CCF separation below the ordinary start-dedup tolerance while
    # leaving an objective improvement larger than the scalar-pilot tolerance.
    data, config = _case(tmp_path, alt=1, total=995100)
    problem = prepare_problem(data, config)
    plateau = torch.full((2, 1), 2 * config.eps, dtype=torch.float64)
    assert 0 < float(torch.max(problem.exact_pilot - plateau)) < 1e-8
    starts = []
    original = solver._fit_from_start
    def recording(problem, lambda_value, options, attempt):
        starts.append(attempt.phi)
        return original(problem, lambda_value, options, attempt)
    monkeypatch.setattr(solver, "_fit_from_start", recording)
    solver.fit_prepared(problem, .1, config.solver, phi_start=plateau,
                        include_default_starts=False)
    assert any(start is problem.exact_pilot for start in starts)


def test_warm_only_continuation_is_an_actual_start(tmp_path):
    data, config = _case(tmp_path, major=2)
    problem = prepare_problem(data, config)
    previous = solver.fit_prepared(problem, .1, config.solver)
    assert previous.state is not None
    fit = solver.fit_prepared(problem, .2, config.solver,
                              warm_state=previous.state, include_default_starts=False)
    assert np.isfinite(fit.objective.total)
    assert fit.provenance.lambda_value == .2
    assert fit.provenance.original_graph_hash == previous.provenance.original_graph_hash


@pytest.mark.parametrize("major", [1, 2])
def test_disabling_all_starts_is_an_immediate_configuration_error(tmp_path, monkeypatch, major):
    data, config = _case(tmp_path, major=major)
    problem = prepare_problem(data, config)
    monkeypatch.setattr(solver, "_fit_from_start", lambda *a, **k: pytest.fail("optimizer entered"))
    with pytest.raises(ValueError, match="No start supplied"):
        solver.fit_prepared(problem, .1, config.solver, include_default_starts=False)


@pytest.mark.parametrize("equal_phi", [False, True])
def test_explicit_start_does_not_silently_inherit_warm_primal(tmp_path, monkeypatch, equal_phi):
    data, config = _case(tmp_path, major=2)
    problem = prepare_problem(data, config)
    previous = solver.fit_prepared(problem, .1, config.solver)
    explicit = previous.state.phi.clone() if equal_phi else torch.full_like(previous.state.phi, .7)
    actual = []
    original = solver._fit_from_start
    def recording(problem, lambda_value, options, attempt):
        actual.append((attempt.phi, attempt.warm_state))
        return original(problem, lambda_value, options, attempt)
    monkeypatch.setattr(solver, "_fit_from_start", recording)
    solver.fit_prepared(problem, .2, config.solver, warm_state=previous.state,
                        phi_start=explicit, include_default_starts=False)
    assert any(state is previous.state and phi is previous.state.phi for phi, state in actual)
    assert any(state is None and phi is explicit for phi, state in actual)
