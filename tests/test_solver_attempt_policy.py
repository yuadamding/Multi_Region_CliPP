"""Compact attempt interfaces preserve fixed-objective retry semantics."""

from dataclasses import replace
import weakref

import numpy as np
import pytest
import torch

from CliPP2.config import resolve_fit_config
from CliPP2.core.fusion import solver
from CliPP2.core.fusion.types import (
    CompressedEdgeCertificate, DenseEdgeCertificate, KKTComponents, ObjectiveValue, SolverState,
)
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
    assert context.source_model is problem32.source_model
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


@pytest.mark.parametrize("num_starts", [4, 32])
def test_discarded_multistart_edge_states_die_before_next_solve(solved, monkeypatch, num_starts):
    problem, options, fit = solved
    seeds = tuple(torch.full_like(problem.exact_pilot, .1 + .8 * i / num_starts)
                  for i in range(num_starts))
    context = replace(problem, scalar_well_starts=seeds[:-1], pooled_start=seeds[-1],
                      _tensor_snapshot=())
    dual_references = []
    live_at_entry = []
    observed_order = []

    def run(prepared, lam, config, attempt):
        # Check before constructing the new state: only the incumbent may live.
        alive = sum(ref() is not None for ref in dual_references)
        live_at_entry.append(alive)
        assert alive <= 1
        index = len(dual_references)
        assert prepared is context and lam == .1 and config is options
        observed_order.append(attempt.phi)
        dual = torch.full((1, 1), float(index), dtype=torch.float64)
        dual_references.append(weakref.ref(dual))
        state = SolverState(phi=attempt.phi.clone(), dual=dual, previous_lambda=.1)
        witness = DenseEdgeCertificate(dual, prepared.graph_hash, "observed_objective")
        # Equal best objectives at starts 1 and 3 must keep the earlier start.
        objective = 1.0 if index in (1, 3) else 2.0
        return replace(fit, phi=state.phi.numpy(), state=state,
                       objective=ObjectiveValue(objective),
                       certificate=replace(fit.certificate, witness=witness))

    monkeypatch.setattr(solver, "_fit_from_start", run)
    result = solver.fit_prepared(context, .1, options)
    assert observed_order == list(seeds)
    assert live_at_entry == [0] + [1] * (num_starts - 1)
    assert [i for i, ref in enumerate(dual_references) if ref() is not None] == [1]
    np.testing.assert_array_equal(result.phi, seeds[1].numpy())
    assert result.objective.total == 1.0
    assert result.provenance.objective_key == fit.provenance.objective_key


def test_successful_cpu_fallback_releases_failed_runtime_traceback(solved, monkeypatch):
    problem, options, fit = solved
    # No CUDA allocation: exercise the resource-policy transition with CPU
    # fixtures while the failed runtime is represented by its device metadata.
    context = replace(problem, runtime=replace(problem.runtime, device=torch.device("cuda"),
                                               device_name="cuda"))
    deduplicate = solver._deduplicate_start_attempts
    monkeypatch.setattr(solver, "_deduplicate_start_attempts",
                        lambda attempts, **kwargs: deduplicate(
                            attempts, **(kwargs | {"runtime": problem.runtime}),
                        ))
    failed_state_refs = []
    calls = []

    def run(prepared, lam, config, attempt):
        calls.append(prepared.runtime.device.type)
        if len(calls) == 1:
            failed_edge_state = torch.ones((17, 2))
            failed_state_refs.append(weakref.ref(failed_edge_state))
            raise MemoryError("bounded injected allocation failure")
        assert calls == ["cuda", "cpu"]
        assert failed_state_refs[0]() is None
        assert prepared.source_model is problem.source_model
        assert prepared.graph_hash == problem.graph_hash
        return fit

    monkeypatch.setattr(solver, "_fit_from_start", run)
    result = solver.fit_prepared(context, .1, options, phi_start=problem.exact_pilot,
                                 include_default_starts=False)
    assert result.objective == fit.objective
    assert "dense_cpu_after_solver_resource_limit" in result.certificate.fallback_reason


@pytest.mark.parametrize("later_start_is_better", [False, True])
def test_multistart_fallback_retention_and_selected_precision_context(
    solved, monkeypatch, later_start_is_better,
):
    problem64, options, fit = solved
    problem32 = solver.promote_solver_context_dtype(problem64, dtype=torch.float32)
    seeds = tuple(torch.full_like(problem32.exact_pilot, value) for value in (.2, .4, .6))
    context = replace(
        problem32, scalar_well_starts=seeds[:2], pooled_start=seeds[2],
        runtime=replace(problem32.runtime, device=torch.device("cuda"), device_name="cuda"),
        _tensor_snapshot=(),
    )
    # Exercise scheduler-independent policy on CPU tensors; no CUDA operation.
    deduplicate = solver._deduplicate_start_attempts
    monkeypatch.setattr(solver, "_deduplicate_start_attempts",
                        lambda attempts, **kwargs: deduplicate(
                            attempts, **(kwargs | {"runtime": problem32.runtime}),
                        ))
    edge_refs = {}
    calls = []
    fallback_context = []
    polished_from = []

    def result(prepared, attempt, label, objective, *, compressed=False):
        dual = torch.ones((1, 1), dtype=attempt.phi.dtype)
        edge_refs[label] = weakref.ref(dual)
        state = SolverState(attempt.phi.clone(), dual, previous_lambda=.1)
        if compressed:
            witness = CompressedEdgeCertificate(
                labels=torch.zeros(2, dtype=torch.int64), centers=state.phi[:1],
                internal_edge_ids=torch.zeros(1, dtype=torch.int64), internal_dual=dual,
                graph_hash=prepared.graph_hash, gradient_scope="observed_objective",
            )
        else:
            witness = DenseEdgeCertificate(dual, prepared.graph_hash, "observed_objective")
        is_polish = attempt.source == "precision_polish"
        return replace(
            fit, phi=state.phi.numpy(), state=state, objective=ObjectiveValue(objective),
            provenance=replace(fit.provenance, dtype="float64" if is_polish else "float32"),
            convergence=replace(fit.convergence, mm_consistency_violations=0),
            certificate=replace(
                fit.certificate, witness=witness, admissible=is_polish,
                status="workset_incomplete" if compressed else "certified",
                directional_admissible=True,
                components=KKTComponents(0 if is_polish else .01, 0, 0, 0),
            ),
        )

    def run(prepared, lam, config, attempt):
        calls.append(attempt.source)
        assert lam == .1
        index = len(calls)
        if index == 1:
            assert prepared is context
            return result(prepared, attempt, "compressed", 5, compressed=True)
        if index == 2:
            assert attempt.source == "dense_retry" and prepared is context
            failed_edge_state = torch.ones((17, 2))
            edge_refs["failed_dense"] = weakref.ref(failed_edge_state)
            raise MemoryError("injected dense retry failure")
        if index == 3:
            assert attempt.source == "cpu_fallback" and prepared is not context
            assert prepared.runtime.device.type == "cpu"
            assert edge_refs["failed_dense"]() is None
            fallback_context.append(prepared)
            return result(prepared, attempt, "fallback", 2)
        assert edge_refs["compressed"]() is None
        assert edge_refs["failed_dense"]() is None
        if index == 4:
            assert prepared is context and edge_refs["fallback"]() is not None
            return result(prepared, attempt, "ordinary", 1 if later_start_is_better else 3)
        if index == 5:
            assert prepared is context
            assert (edge_refs["fallback"]() is None) == later_start_is_better
            assert (edge_refs["ordinary"]() is not None) == later_start_is_better
            return result(prepared, attempt, "last", 4)
        assert index == 6 and attempt.source == "precision_polish"
        assert edge_refs["last"]() is None
        assert config == replace(options, outer_max_iter=1)
        assert prepared.runtime.dtype == torch.float64
        expected_seed = seeds[1] if later_start_is_better else seeds[0]
        torch.testing.assert_close(attempt.phi, expected_seed.double(), rtol=0, atol=0)
        return result(prepared, attempt, "polished", .5 if later_start_is_better else 1.5)

    promote = solver._float64_context

    def precision_context(data, selected, **kwargs):
        expected = context if later_start_is_better else fallback_context[0]
        assert selected is expected
        assert selected.base_objective_key == context.base_objective_key
        polished_from.append(selected)
        return promote(data, selected, device=torch.device("cpu"))

    monkeypatch.setattr(solver, "_fit_from_start", run)
    monkeypatch.setattr(solver, "_float64_context", precision_context)
    final = solver.fit_prepared(context, .1, options)
    assert calls == ["scalar_well", "dense_retry", "cpu_fallback", "scalar_well",
                     "pooled", "precision_polish"]
    assert len(polished_from) == 1
    assert [name for name, ref in edge_refs.items() if ref() is not None] == ["polished"]
    assert final.certificate.precision_polished
    assert final.provenance.objective_key == fit.provenance.objective_key
    assert final.objective.total == (.5 if later_start_is_better else 1.5)
