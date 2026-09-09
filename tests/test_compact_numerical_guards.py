"""Bounded scalar evidence and identity-safe resource fallback."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from CliPP2.config import resolve_fit_config
from CliPP2.core.fusion.solver import transfer_scalar_pilot_certificates
from CliPP2.core.objective import compile_observed_model
from CliPP2.core.scalar import (
    ScalarGlobalMinimumCertificate,
    ScalarProblem,
    _interval_lower_bound,
    certify_scalar_minimum,
    scalar_loss,
    scalar_problem_from_model,
)
from CliPP2.model_selection import proposals
from test_integer_likelihood import EPS, integer_data


@pytest.mark.parametrize("budget", [1, 2, 3, 7, 17, 23, 64])
def test_scalar_initial_partition_respects_even_tiny_budgets(budget):
    data = integer_data(((6,),))
    model = compile_observed_model(data, eps=EPS)
    problem = scalar_problem_from_model(
        model, np.array([0]), 0, lower=EPS, upper=1.0, eps=EPS
    )
    result = certify_scalar_minimum(
        problem, tolerance=1e-6, max_intervals=budget, hint=0.42
    )
    assert 1 <= result.intervals_evaluated <= budget
    assert problem.lower <= result.argmin <= problem.upper
    assert result.attained_value == pytest.approx(scalar_loss(problem, result.argmin))
    grid_min = scalar_loss(problem, np.linspace(EPS, 1.0, 10001)).min()
    assert result.global_lower_bound <= grid_min + 1e-10
    if result.globally_certified:
        assert result.optimality_gap <= 1e-6


@pytest.mark.parametrize("slope", [1.0, 3.0])
def test_coarse_intervals_keep_valid_bounds_across_clipping(slope):
    problem = ScalarProblem(
        alt=np.array([10.0]), nonalt=np.array([90.0]),
        observed=np.array([True]), slope=np.array([[slope]]),
        log_prior=np.array([[0.0]]), valid=np.array([[True]]),
        lower=EPS, upper=1.0, eps=EPS,
    )
    known_minimizer = 0.1 / slope
    exact_minimum = float(scalar_loss(problem, known_minimizer))
    bound = _interval_lower_bound(problem, EPS, 1.0)
    assert bound <= exact_minimum + 1e-12
    result = certify_scalar_minimum(problem, tolerance=1e-6, max_intervals=1)
    assert result.intervals_evaluated == 1
    assert result.global_lower_bound <= exact_minimum + 1e-12
    assert not result.globally_certified


def _context(*, certified=True, fingerprint="same", pilot=0.7, device="cpu"):
    # The transfer boundary only reads model identity and the exact pilot;
    # no GPU runtime or solver is needed for the controlled fallback test.
    context = SimpleNamespace()
    context.problem = SimpleNamespace(
        eps=EPS,
        observed_model=SimpleNamespace(source_fingerprint=fingerprint),
        source_model=SimpleNamespace(fingerprint=fingerprint),
    )
    context.exact_pilot = torch.tensor([[pilot]], dtype=torch.float64)
    context.pooled_start = context.exact_pilot
    context.scalar_well_starts = ()
    context.runtime = SimpleNamespace(
        device=torch.device(device), device_name=device, dtype=torch.float64
    )
    context.graph_spec = SimpleNamespace()
    context.scalar_pilot_certificates = (
        (ScalarGlobalMinimumCertificate(pilot, 1.0, 1.0, 0.0, True, "test", 1),)
        if certified else ()
    )
    return context


def test_transfer_rejects_changed_source_or_pilot(monkeypatch):
    # Dataclass replace normally carries all context state. Isolate the
    # helper with namespace copies instead of constructing an unrelated graph.
    monkeypatch.setattr("CliPP2.core.fusion.solver.replace", lambda obj, **kw: SimpleNamespace(**(vars(obj) | kw)))
    source = _context()
    for target in (_context(fingerprint="other"), _context(pilot=0.8)):
        with pytest.raises(ValueError, match="changed pilot/model"):
            transfer_scalar_pilot_certificates(source, target)


@pytest.mark.parametrize("change", ["eps", "lower", "prior", "missing_source"])
def test_transfer_rejects_changed_compiled_model_even_with_same_runtime_identity(change):
    model = compile_observed_model(integer_data(((4,),)), eps=EPS)
    source = _context(fingerprint=model.fingerprint)
    target = _context(fingerprint=model.fingerprint)
    source.problem.source_model = target.problem.source_model = model
    if change == "eps":
        setattr(target.problem, change, 0.1)
    elif change == "lower":
        target.problem.source_model = replace(model, lower=np.full(model.shape, 0.01))
    elif change == "prior":
        target.problem.source_model = replace(
            model, log_prior=np.log(np.array([[[0.1, 0.2, 0.3, 0.4]]]))
        )
    else:
        target.problem.source_model = None
    # A stale label on the runtime view cannot authorize a different source.
    assert source.problem.observed_model.source_fingerprint == (
        target.problem.observed_model.source_fingerprint
    )
    with pytest.raises(ValueError, match="changed pilot/model"):
        transfer_scalar_pilot_certificates(source, target)


def test_guided_cpu_fallback_preserves_scalar_certificates(monkeypatch):
    source = _context(device="cuda")
    target = _context(certified=False)
    built = []

    def fake_replace(obj, **updates):
        return SimpleNamespace(**(vars(obj) | updates))

    def fake_build(phi, labels, *, solver_context, **kwargs):
        built.append(solver_context)
        if len(built) == 1:
            raise MemoryError("controlled GPU allocation failure")
        return "guided"

    monkeypatch.setattr("CliPP2.core.fusion.solver.replace", fake_replace)
    monkeypatch.setattr(proposals, "build_guided_fusion_initialization", fake_build)
    monkeypatch.setattr(proposals, "prepare_torch_problem_with_resource_policy", lambda *a, **kw: target)
    options = resolve_fit_config(device="cpu")
    options = replace(options, runtime=replace(options.runtime, fallback="cpu_allowed"))
    guided, result, _ = proposals.build_guided_initialization_with_resource_policy(
        data=integer_data(((1,),)), guide_phi=np.array([[0.7]]),
        guide_labels=np.array([0]), solver_context=source, fit_options=options,
    )
    assert guided == "guided"
    assert result.scalar_pilot_certificates == source.scalar_pilot_certificates
    assert built[-1] is result
