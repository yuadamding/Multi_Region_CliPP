"""Prepared fusion contexts cannot silently replace the requested objective."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from CliPP2.config import resolve_fit_config
from CliPP2.core.fusion import solver
from CliPP2.core.fusion.graph import build_complete_uniform_graph
from test_integer_likelihood import EPS, integer_data


def _prepared(*, adaptive=False):
    data = integer_data(((2,), (3,)))
    graph = None if adaptive else build_complete_uniform_graph(2)
    context = solver.prepare_torch_problem(
        data, eps=EPS, tol=8e-4, inner_max_iter=16,
        graph=graph, exact_pilot=data.phi_init, pooled_start=data.phi_init,
        scalar_well_starts=(), device="cpu", dtype="float64",
    )
    return data, context


def _fit(data, context, **kwargs):
    options = resolve_fit_config(device="cpu", dtype="float64",
                                 outer_max_iter=2, inner_max_iter=16).solver
    return solver.fit_prepared(
        context, .1, options, phi_start=data.phi_init, **kwargs,
    )


def test_same_topology_different_graph_weights_rejected_before_solver(monkeypatch):
    data, context = _prepared()
    graph_b = replace(context.graph_spec, edge_w=context.graph_spec.edge_w * 2)
    monkeypatch.setattr(solver, "_fit_from_start", lambda *a, **k: pytest.fail("optimizer entered"))
    with pytest.raises(TypeError, match="graph"):
        _fit(data, context, graph=graph_b)
    with pytest.raises(ValueError, match="graph identity"):
        _fit(data, replace(context, graph_spec=graph_b))


@pytest.mark.parametrize("option", [
    {"adaptive_weight_gamma": 2}, {"adaptive_weight_floor": .01},
    {"adaptive_weight_baseline": 2}, {"data": None}, {"eps": None}, {"runtime": None},
])
def test_changed_adaptive_graph_recipe_rejected(monkeypatch, option):
    data, context = _prepared(adaptive=True)
    monkeypatch.setattr(solver, "_fit_from_start", lambda *a, **k: pytest.fail("optimizer entered"))
    with pytest.raises(TypeError):
        _fit(data, context, **option)


def test_equivalent_graph_can_reuse_context(monkeypatch):
    data, context = _prepared()
    def reached_optimizer(*args, **kwargs):
        raise RuntimeError("optimizer boundary reached")
    monkeypatch.setattr(solver, "_fit_from_start", reached_optimizer)
    with pytest.raises(RuntimeError, match="optimizer boundary"):
        _fit(data, context)


def test_corrupted_cached_graph_identity_is_rejected():
    data, context = _prepared()
    with pytest.raises(ValueError, match="identity is inconsistent"):
        _fit(data, replace(context, graph_hash="stale"))


def test_changed_source_data_cannot_reuse_prepared_identity():
    data, context = _prepared()
    changed = replace(data, alt_counts=np.zeros_like(data.alt_counts))
    with pytest.raises(ValueError, match="data fingerprint"):
        _fit(data, replace(context, source_data=changed))


def test_changed_epsilon_cannot_reuse_prepared_identity():
    data, context = _prepared()
    with pytest.raises(ValueError, match="likelihood or epsilon"):
        _fit(data, replace(context, problem=replace(context.problem, eps=.01)))


def test_changed_base_objective_is_rejected():
    data, context = _prepared()
    with pytest.raises(ValueError, match="objective identity"):
        _fit(data, replace(context, base_fusion_objective_hash="stale"))


def test_frozen_graph_arrays_cannot_be_made_writable():
    _, context = _prepared()
    for values in (context.graph_spec.edge_u, context.graph_spec.edge_v,
                   context.graph_spec.edge_w):
        with pytest.raises(ValueError):
            values.setflags(write=True)


def test_preparation_preserves_nondefault_epsilon_and_float64_source():
    import torch
    from CliPP2.core.objective import compile_observed_model
    data = integer_data(((2,), (3,)))
    context = solver.prepare_torch_problem(
        data, eps=.02, tol=8e-4, inner_max_iter=16,
        graph=build_complete_uniform_graph(2), device="cpu", dtype="float32",
    )
    assert context.problem.eps == .02
    assert context.problem.source_model is compile_observed_model(data, eps=.02)
    promoted = solver.promote_solver_context_dtype(context, dtype=torch.float64)
    assert promoted.problem.eps == .02
    assert promoted.problem.source_model is context.problem.source_model
    solver._validate_prepared_problem(promoted)


@pytest.mark.parametrize("target", ["weights", "counts", "lower", "pilot", "preconditioner"])
def test_prepared_runtime_tensor_mutation_is_rejected_before_optimization(monkeypatch, target):
    data, context = _prepared()
    tensors = {
        "weights": context.graph.weight,
        "counts": context.problem.observed_model.alt,
        "lower": context.lower,
        "pilot": context.exact_pilot,
        "preconditioner": context.graph.pdhg_tau_node,
    }
    tensors[target].add_(1)
    monkeypatch.setattr(solver, "_fit_from_start", lambda *a, **k: pytest.fail("optimizer entered"))
    with pytest.raises(ValueError, match="Prepared runtime tensor.*changed"):
        _fit(data, context)


def test_replacing_runtime_views_cannot_rebaseline_stale_identity():
    _, context = _prepared()
    with pytest.raises(ValueError, match="Prepared runtime tensor graph.weight changed"):
        replace(context, graph=replace(context.graph, weight=context.graph.weight * 10))
    model = context.problem.observed_model
    with pytest.raises(ValueError, match="Prepared runtime tensor model.alt changed"):
        replace(context, problem=replace(context.problem, observed_model=replace(model, alt=model.alt + 1)))


def test_runtime_precision_rebuild_is_bound_to_unchanged_host_sources():
    _, context = _prepared()
    promoted = solver.promote_solver_context_dtype(context, dtype=torch.float32)
    solver._validate_prepared_problem(promoted)
    restored = solver.promote_solver_context_dtype(promoted, dtype=torch.float64)
    solver._validate_prepared_problem(restored)
    assert restored.problem.source_model is context.problem.source_model
    assert restored.graph_hash == context.graph_hash
    torch.testing.assert_close(restored.graph.weight, context.graph.weight, rtol=0, atol=0)
    context.graph.weight.add_(1)
    with pytest.raises(ValueError, match="Prepared runtime tensor graph.weight changed"):
        solver.promote_solver_context_dtype(context, dtype=torch.float32)
