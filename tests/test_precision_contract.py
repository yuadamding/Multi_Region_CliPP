"""Clipping must stay interior in the requested precision before any fit."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from CliPP2.config import RuntimeConfig, resolve_fit_config, validate_likelihood_precision
from CliPP2.core import objective
from CliPP2.core.fusion.torch_backend import resolve_runtime
from CliPP2.core.fusion.types import TorchRuntime


def _source(eps=1e-6):
    major = np.array([[1], [1], [2], [6], [2], [6]])
    return objective.compile_integer_observations(
        alt_counts=np.array([[100], [0], [0], [20], [30], [70]]),
        total_counts=np.array([[100], [100], [0], [100], [100], [100]]),
        count_observed=np.array([[True], [True], [True], [False], [True], [True]]),
        major_cn=major, scaling=1.0 / major,
        phi_upper=np.full((6, 1), 1.0 - eps), eps=eps,
    )


@pytest.mark.parametrize("dtype,eps", [
    ("float32", 1e-12), ("float64", 1e-20),
    ("float32", 0.5 - 1e-10),
])
def test_unrepresentable_endpoints_fail_at_config_and_model_boundaries(dtype, eps):
    with pytest.raises(ValueError, match="incompatible.*clipping endpoints"):
        resolve_fit_config(device="cpu", dtype=dtype, eps=eps)
    with pytest.raises(ValueError, match="incompatible.*clipping endpoints"):
        objective.model_to_torch(_source(), resolve_runtime("cpu", dtype=dtype), eps=eps)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "int32"])
def test_retired_dtypes_fail_without_cuda_or_fitting(dtype):
    with pytest.raises(ValueError, match="float32 or float64"):
        RuntimeConfig(dtype=dtype)
    with pytest.raises(ValueError, match="float32 or float64"):
        resolve_runtime("cuda", dtype=dtype)
    runtime = TorchRuntime(device=torch.device("cpu"), device_name="cpu", dtype=getattr(torch, dtype))
    with pytest.raises(ValueError, match="float32 or float64"):
        objective.model_to_torch(_source(), runtime, eps=1e-6)


@pytest.mark.parametrize("eps", [0, -1, 0.5, np.inf, np.nan])
def test_invalid_epsilon_fails_before_model_preparation(eps):
    with pytest.raises(ValueError, match="eps must be finite"):
        resolve_fit_config(device="cpu", eps=eps)


@pytest.mark.parametrize("dtype,half_gap", [("float32", 2.0**-25), ("float64", 2.0**-54)])
def test_epsilon_representation_boundary_at_one(dtype, half_gap):
    for rejected in (half_gap * (1 - 1e-6), half_gap):
        with pytest.raises(ValueError, match="incompatible"):
            validate_likelihood_precision(rejected, dtype)
    accepted = half_gap * (1 + 1e-6)
    assert validate_likelihood_precision(accepted, dtype) == accepted
    near_half = float(np.nextafter(np.dtype(dtype).type(0.5), np.dtype(dtype).type(0)))
    assert validate_likelihood_precision(near_half, dtype) == near_half


@pytest.mark.parametrize("dtype,eps", [
    ("float32", 1e-6), ("float32", 1e-7), ("float64", 1e-6), ("float64", 1e-12),
])
@pytest.mark.parametrize("boundary", ["lower", "upper", "interior"])
def test_endpoint_candidate_derivatives_and_masked_units_are_finite(dtype, eps, boundary):
    source = _source(eps)
    runtime = resolve_runtime("cpu", dtype=dtype)
    model = objective.model_to_torch(source, runtime, eps=eps)
    value = {"lower": eps, "upper": 1.0 - eps, "interior": 0.37}[boundary]
    phi = torch.full(model.shape, value, dtype=runtime.dtype)
    kernel = objective._emission_kernel_torch(model, phi, eps=eps)
    assert torch.all((kernel.probability > 0) & (kernel.probability < 1))
    candidate = objective._candidate_terms_torch(
        model.alt[..., None], model.nonalt[..., None], kernel.probability, kernel.slope,
    )
    for tensor in candidate:
        assert torch.isfinite(tensor).all()
    assert torch.all(candidate[1][~model.valid] == 0)
    assert torch.all(candidate[2][~model.valid] == 0)
    marginal = objective.observed_terms_torch(model, phi, eps=eps)
    em = objective.observed_em_terms_torch(
        model, phi, eps=eps, responsibilities=marginal.posterior,
    )
    for terms in (marginal, em):
        for field in ("loss", "gradient", "hessian_upper", "posterior"):
            assert torch.isfinite(getattr(terms, field)).all()
        assert torch.all(terms.posterior[~model.valid] == 0)
        assert terms.loss[3].item() == terms.gradient[3].item() == 0
        assert abs(terms.loss[2].item()) < 1e-6
    reference = objective.observed_terms_numpy(source, phi.double().numpy(), eps=eps)
    # Float32 rounds the upper probability, so compare its independent dtype
    # arithmetic exactly and the float64 likelihood within endpoint rounding.
    joint = (candidate[0] + model.log_prior).masked_fill(~model.valid, -torch.inf)
    expected = torch.where(model.observed, -torch.logsumexp(joint, -1), 0)
    torch.testing.assert_close(marginal.loss, expected, rtol=0, atol=0)
    if dtype == "float64":
        np.testing.assert_allclose(marginal.loss, reference.loss, rtol=1e-13, atol=1e-12)
    assert validate_likelihood_precision(eps, dtype) == eps


@pytest.mark.parametrize("dtype,counts", [("float32", 1e30), ("float64", 1e300)])
def test_interior_endpoints_do_not_admit_overflowing_candidate_arithmetic(dtype, counts):
    source = replace(_source(), alt=np.full((6, 1), counts))
    with pytest.raises(ValueError, match="Candidate arithmetic may overflow"):
        objective.model_to_torch(source, resolve_runtime("cpu", dtype=dtype), eps=1e-6)


def test_overflow_rejection_precedes_pilot_and_graph_construction(monkeypatch, tmp_path):
    from CliPP2.api import prepare_problem
    from CliPP2.core.fusion import solver
    from CliPP2.io.tumor_txt import load_tumor_txt
    from test_reference_pipeline import _input

    data = load_tumor_txt(_input(tmp_path, "cn2"))
    data = replace(data, alt_counts=np.full(data.alt_counts.shape, 1e30),
                   total_counts=np.full(data.total_counts.shape, 2e30))

    def not_reached(*args, **kwargs):
        pytest.fail("Precision rejection must precede pilot and graph preparation.")

    monkeypatch.setattr(solver, "compute_scalar_mutation_region_wells_torch", not_reached)
    monkeypatch.setattr(solver, "build_complete_adaptive_tensor_graph", not_reached)
    config = resolve_fit_config(device="cpu", dtype="float32")
    with pytest.raises(ValueError, match="Candidate arithmetic may overflow"):
        prepare_problem(data, config)


def test_promotion_reuses_exact_sources_and_never_adjusts_epsilon():
    source = _source(1e-12)
    with pytest.raises(ValueError, match="incompatible"):
        objective.model_to_torch(source, resolve_runtime("cpu", dtype="float32"), eps=1e-12)
    model = objective.model_to_torch(source, resolve_runtime("cpu", dtype="float64"), eps=1e-12)
    assert model.source_fingerprint == source.fingerprint
    for field in ("lower", "upper", "slope", "alt", "nonalt", "log_prior"):
        np.testing.assert_array_equal(getattr(model, field), getattr(source, field))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_likelihood_grid_omits_slope_and_mask_temporaries(monkeypatch, dtype):
    model = objective.model_to_torch(_source(), resolve_runtime("cpu", dtype=dtype), eps=1e-6)
    phi = torch.tensor([1e-6, 0.2, 0.5, 1 - 1e-6], dtype=model.alt.dtype).expand(6, 1, 4)
    expected = torch.stack([
        objective.observed_terms_torch(model, phi[..., i], eps=1e-6).loss for i in range(4)
    ], -1)
    original = objective._emission_kernel_torch
    seen = []

    def checked_kernel(*args, **kwargs):
        assert kwargs["derivatives"] is False
        kernel = original(*args, **kwargs)
        assert kernel.slope is None
        seen.append(kernel.probability.numel())
        return kernel

    monkeypatch.setattr(objective, "_emission_kernel_torch", checked_kernel)
    actual = objective.observed_loss_grid_torch(model, phi, eps=1e-6)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert seen == [6 * 1 * 4 * 6]
    # The shared kernel itself must not build derivative comparisons/where.
    def unexpected_where(*args, **kwargs):
        pytest.fail("Likelihood-only kernel constructed a clipped slope.")

    monkeypatch.setattr(torch, "where", unexpected_where)
    kernel = original(model, phi, eps=1e-6, derivatives=False)
    assert kernel.slope is None
