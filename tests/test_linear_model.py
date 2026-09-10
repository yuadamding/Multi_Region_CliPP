"""Linear-only kernels retain cc5a3d1 integer likelihood and clipping evidence."""

from dataclasses import fields, replace

import numpy as np
import pytest
import torch

from CliPP2.core.fusion.torch_backend import resolve_runtime
from CliPP2.core.objective import (
    ObservedModel,
    compile_observed_model,
    model_to_torch,
    observed_em_terms_torch,
    observed_internal_breakpoints_torch,
    observed_one_sided_gradients_torch,
    observed_terms_numpy,
    observed_terms_torch,
)
from CliPP2.core.scalar import (
    ScalarProblem,
    certify_scalar_minimum,
    scalar_breakpoints,
    scalar_loss_and_gradient,
    scalar_problem_from_model,
)
from test_integer_likelihood import EPS, integer_data


def _reference_model():
    count = np.arange(1, 7)[:, None]
    candidates = np.arange(1, 7)[None, None, :]
    valid = candidates <= count[..., None]
    alt = np.array([0, 12, 24, 48, 72, 100])[:, None]
    return ObservedModel(
        alt=alt, nonalt=100 - alt, observed=np.ones((6, 1), dtype=bool),
        lower=np.full((6, 1), EPS), upper=np.ones((6, 1)),
        slope=np.where(valid, candidates / (count[..., None] + 1), 0.0),
        log_prior=np.where(valid, -np.log(count[..., None]), -np.inf), valid=valid,
        model_id="clipp2_clonal_integer_multiplicity_mixture_v1",
    )


def test_integer_kernel_matches_pinned_cc5a3d1_numeric_reference():
    # Captured by executing the unmodified pinned source in an isolated process.
    # All six candidate counts are represented, including zero/all-alt reads.
    expected = {
        "loss": [28.76820724517809, 37.46316903742572, 56.9590404871155,
                 70.61206801553743, 62.91258664337332, 27.742879005661855],
        "gradient": [66.66666666666667, 3.9240248501762203, 10.065560097420098,
                     0.15367124454142939, -33.786031037697384, -111.1111111111111],
        "hessian_upper": [44.44444444444444, 88.11849368729091, 101.19501241300823,
                          144.94179607800606, 266.2422600177912, 123.45679012345678],
    }
    model = _reference_model()
    phi = np.array([0.5, 0.4, 0.6, 0.8, 0.75, 0.9])[:, None]
    numpy_terms = observed_terms_numpy(model, phi, eps=EPS)
    tm = model_to_torch(model, resolve_runtime("cpu", dtype="float64"), eps=EPS)
    tensor = torch.tensor(phi)
    torch_terms = observed_terms_torch(tm, tensor, eps=EPS)
    em_terms = observed_em_terms_torch(
        tm, tensor, responsibilities=torch_terms.posterior, eps=EPS,
    )
    for name, reference in expected.items():
        for terms in (numpy_terms, torch_terms, em_terms):
            np.testing.assert_allclose(getattr(terms, name), np.array(reference)[:, None],
                                       rtol=2e-13, atol=2e-12)


def test_scalar_certificates_match_pinned_cc5a3d1_reference():
    model = _reference_model()
    phi = [0.5, 0.4, 0.6, 0.8, 0.75, 0.9]
    # argmin, attained value, lower bound, certified, interval count.
    expected = [
        (1e-6, 0.00010000005000003333, 0.00010000005000003332, True, 18),
        (0.1888655190820694, 37.29640388772843, 37.2964038796115, True, 233),
        (0.3371588659667969, 56.0682174846848, 56.06821683896268, False, 256),
        (0.607227718212128, 70.56770196450717, 70.56770191625117, False, 255),
        (0.8653956805639267, 60.89281633124692, 60.89281632219062, True, 212),
        (1.0, 17.20682743987922, 17.206827439879216, True, 23),
    ]
    for i, (argmin, value, bound, certified, intervals) in enumerate(expected):
        problem = scalar_problem_from_model(
            model, np.array([i]), 0, lower=EPS, upper=1.0, eps=EPS,
        )
        result = certify_scalar_minimum(
            problem, tolerance=1e-8, max_intervals=256, hint=phi[i],
        )
        np.testing.assert_allclose(
            [result.argmin, result.attained_value, result.global_lower_bound],
            [argmin, value, bound], rtol=1e-13, atol=1e-12,
        )
        assert result.globally_certified is certified
        assert result.intervals_evaluated == intervals


@pytest.mark.parametrize("boundary", [0.1, 0.9])
def test_clipping_retains_exact_one_sided_derivatives(boundary):
    epsilon = 0.1
    model = ObservedModel(
        alt=np.array([[20.0]]), nonalt=np.array([[80.0]]),
        observed=np.array([[True]]), lower=np.array([[0.0]]), upper=np.array([[1.0]]),
        slope=np.array([[[1.0]]]), log_prior=np.array([[[0.0]]]),
        valid=np.array([[[True]]]), model_id="clipp2_clonal_integer_multiplicity_mixture_v1",
    )
    tm = model_to_torch(model, resolve_runtime("cpu", dtype="float64"), eps=epsilon)
    tensor = torch.tensor([[boundary]], dtype=torch.float64)
    left, right, kink = observed_one_sided_gradients_torch(tm, tensor, eps=epsilon)
    untruncated_gradient = -(20 / boundary - 80 / (1 - boundary))
    assert bool(kink.item())
    assert left.item() == pytest.approx(0.0 if boundary == epsilon else untruncated_gradient)
    assert right.item() == pytest.approx(untruncated_gradient if boundary == epsilon else 0.0)
    assert observed_terms_torch(tm, tensor, eps=epsilon).gradient.item() == 0.0
    points, valid = observed_internal_breakpoints_torch(tm, eps=epsilon)
    np.testing.assert_array_equal(points[valid], [epsilon, 1 - epsilon])
    problem = scalar_problem_from_model(
        model, np.array([0]), 0, lower=0.0, upper=1.0, eps=epsilon,
    )
    np.testing.assert_array_equal(scalar_breakpoints(problem), [0, epsilon, 1 - epsilon, 1])
    for phi in (np.nextafter(boundary, -np.inf), boundary,
                np.nextafter(boundary, np.inf)):
        loss, gradient = scalar_loss_and_gradient(problem, phi)
        terms = observed_terms_numpy(model, np.array([[phi]]), eps=epsilon)
        assert loss == pytest.approx(terms.loss.item())
        assert gradient == pytest.approx(terms.gradient.item())


def test_only_one_slope_array_is_stored_and_source_cache_is_eps_bound():
    for cls in (ObservedModel, ScalarProblem):
        names = {item.name for item in fields(cls)}
        assert "slope" in names
        assert not names.intersection({"first_scale", "second_scale", "switch", "legacy_major"})
    data = integer_data(((3,),))
    model = compile_observed_model(data, eps=EPS)
    assert compile_observed_model(data, eps=EPS) is model
    assert compile_observed_model(data, eps=0.01) is not model
    changed = replace(data, alt_counts=data.alt_counts + 1)
    assert compile_observed_model(changed, eps=EPS) is not model
    assert compile_observed_model(changed, eps=EPS).fingerprint != model.fingerprint
    for name in ("alt", "nonalt", "observed", "lower", "upper", "slope", "valid", "log_prior"):
        with pytest.raises(ValueError, match="WRITEABLE"):
            getattr(model, name).setflags(write=True)
    problem = scalar_problem_from_model(model, np.array([0]), 0, lower=EPS, upper=1.0, eps=EPS)
    with pytest.raises(ValueError, match="WRITEABLE"):
        problem.slope.setflags(write=True)
    with pytest.raises(TypeError, match="major_prior"):
        compile_observed_model(data, major_prior=0.7, eps=EPS)
