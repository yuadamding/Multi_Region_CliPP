"""Independent free-energy/derivative checks with frozen EM responsibilities."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from CliPP2.core.fusion.torch_backend import resolve_runtime
from CliPP2.core.objective import (
    compile_observed_model, model_to_torch, observed_em_terms_torch, observed_terms_torch,
)
from test_integer_likelihood import integer_data


@pytest.mark.parametrize("epsilon", [1e-6, 0.1])
def test_frozen_responsibility_em_matches_independent_free_energy(epsilon):
    data = integer_data(((1, 2), (4, 6)), observed=np.array([[True, False], [True, True]]))
    model = compile_observed_model(data, eps=epsilon)
    tm = model_to_torch(model, resolve_runtime("cpu", dtype="float64"), eps=epsilon)
    phi = np.array([[epsilon / 2, 0.31], [0.67, 1.0]])
    weights = np.where(model.valid, np.arange(1, 7)[None, None, :], 0.0)
    weights /= weights.sum(axis=-1, keepdims=True)
    mass = model.slope * phi[..., None]
    probability = np.clip(mass, epsilon, 1 - epsilon)
    slope = np.where((mass > epsilon) & (mass < 1 - epsilon), model.slope, 0.0)
    complete = -(model.alt[..., None] * np.log(probability)
                 + model.nonalt[..., None] * np.log1p(-probability))
    complete -= np.where(model.valid, model.log_prior, 0.0)
    entropy = np.where(weights > 0, weights * np.log(np.maximum(weights, 1e-300)), 0.0)
    expected_loss = np.sum(weights * complete + entropy, axis=-1)
    expected_gradient = -np.sum(weights * slope * (
        model.alt[..., None] / probability - model.nonalt[..., None] / (1 - probability)
    ), axis=-1)
    expected_curvature = np.maximum(np.sum(weights * slope ** 2 * (
        model.alt[..., None] / probability ** 2
        + model.nonalt[..., None] / (1 - probability) ** 2
    ), axis=-1), 1e-8)
    terms = observed_em_terms_torch(
        tm, torch.tensor(phi), responsibilities=torch.tensor(weights), eps=epsilon,
    )
    for name, values in [
        ("loss", expected_loss), ("gradient", expected_gradient),
        ("hessian_upper", expected_curvature),
    ]:
        np.testing.assert_allclose(getattr(terms, name), np.where(model.observed, values, 0),
                                   rtol=1e-13, atol=1e-12)
    marginal = observed_terms_torch(tm, torch.tensor(phi), eps=epsilon)
    assert torch.all(terms.loss >= marginal.loss - 1e-12)


def test_em_touches_observed_loss_at_e_step_and_bounds_elsewhere():
    data = integer_data(((1,), (2,), (3,), (4,), (5,), (6,)))
    data = replace(data, alt_counts=np.arange(0, 120, 20)[:, None].astype(float))
    model = model_to_torch(compile_observed_model(data, eps=1e-6),
                           resolve_runtime("cpu", dtype="float64"), eps=1e-6)
    start = torch.full((6, 1), 0.6, dtype=torch.float64)
    observed = observed_terms_torch(model, start, eps=1e-6)
    for phi in (start, torch.full_like(start, 0.2), torch.full_like(start, 0.9)):
        surrogate = observed_em_terms_torch(
            model, phi, responsibilities=observed.posterior, eps=1e-6,
        )
        exact = observed_terms_torch(model, phi, eps=1e-6)
        assert torch.all(surrogate.loss >= exact.loss - 1e-12)
        if phi is start:
            torch.testing.assert_close(surrogate.loss, exact.loss, rtol=1e-13, atol=1e-12)
            torch.testing.assert_close(surrogate.gradient, exact.gradient, rtol=1e-13, atol=1e-12)
