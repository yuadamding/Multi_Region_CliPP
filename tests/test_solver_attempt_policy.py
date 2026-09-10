"""Compact attempt interfaces preserve fixed-objective retry semantics."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from CliPP2.config import resolve_fit_config
from CliPP2.core.fusion import solver
from CliPP2.core.fusion.types import CompressedEdgeCertificate, KKTComponents
from test_solver_request import _prepared


@pytest.fixture(scope="module")
def solved():
    data, problem = _prepared()
    options = resolve_fit_config(device="cpu", dtype="float64",
                                 outer_max_iter=2, inner_max_iter=16).solver
    fit = solver.fit_prepared(problem, .1, options, phi_start=data.phi_init,
                              include_default_starts=False)
    return problem, options, fit


def test_dense_retry_keeps_objective_and_uses_unpaired_returned_primal(solved, monkeypatch):
    problem, options, fit = solved
    witness = CompressedEdgeCertificate(
        labels=torch.zeros(2, dtype=torch.int64), centers=fit.state.phi[:1],
        internal_edge_ids=torch.zeros(1, dtype=torch.int64),
        internal_dual=torch.zeros((1, 1), dtype=torch.float64),
        graph_hash=problem.graph_hash, gradient_scope="observed_objective",
    )
    incomplete = replace(fit, certificate=replace(
        fit.certificate, witness=witness, status="workset_incomplete", admissible=False,
    ))
    calls = []

    def run(context, lam, config, attempt):
        calls.append((context, lam, config, attempt))
        return incomplete if len(calls) == 1 else fit

    monkeypatch.setattr(solver, "_fit_from_start", run)
    result = solver.fit_prepared(problem, .1, options, warm_state=fit.state,
                                 include_default_starts=False)
    assert len(calls) == 2
    for context, lam, config, _ in calls:
        assert context is problem and config is options and lam == .1
    first, second = calls[0][3], calls[1][3]
    assert first.phi is fit.state.phi and first.warm_state is fit.state
    assert second.phi is incomplete.state.phi and second.warm_state is None
    assert second.source == "dense_retry"
    assert result.provenance.objective_key == fit.provenance.objective_key
    assert result.work == fit.work + incomplete.work
    assert "dense_current_device_after_compressed_not_certified" in result.certificate.fallback_reason


def test_precision_retry_reuses_source_state_and_single_outer_iteration(solved, monkeypatch):
    problem64, options, fit = solved
    problem32 = solver.promote_solver_context_dtype(problem64, dtype=torch.float32)
    working = replace(
        fit, phi=fit.phi.astype(np.float32),
        provenance=replace(fit.provenance, dtype="float32"),
        convergence=replace(fit.convergence, mm_consistency_violations=0),
        certificate=replace(
            fit.certificate, status="certified", admissible=False,
            directional_admissible=True, components=KKTComponents(.01, 0, 0, 0),
        ),
    )
    calls = []

    def run(context, lam, config, attempt):
        calls.append((context, lam, config, attempt))
        return working if len(calls) == 1 else fit

    monkeypatch.setattr(solver, "_fit_from_start", run)
    result = solver.fit_prepared(problem32, .1, options, phi_start=working.phi,
                                 include_default_starts=False)
    assert len(calls) == 2
    initial, promoted = calls
    assert initial[0] is problem32 and initial[2] is options
    context, lam, config, attempt = promoted
    assert context.runtime.dtype == torch.float64 and lam == .1
    assert context.problem.source_model is problem32.problem.source_model
    assert context.base_objective_key == problem32.base_objective_key
    assert context.data_fingerprint == problem32.data_fingerprint
    assert config == replace(options, outer_max_iter=1)
    assert options.outer_max_iter == 2
    assert attempt.warm_state is working.state
    assert attempt.phi.dtype == torch.float64 and attempt.source == "precision_polish"
    np.testing.assert_array_equal(attempt.phi.numpy(), working.phi)
    assert result.certificate.precision_polished
    assert result.certificate.tolerance == .004
    assert result.provenance.source_data_hash == fit.provenance.source_data_hash
