"""Independent numerical checks for the uniform categorical integer model."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.special import logsumexp

from CliPP2.core.objective import (
    compile_observed_model,
    model_to_torch,
    observed_em_terms_torch,
    observed_loss_grid_torch,
    observed_terms_numpy,
    observed_terms_torch,
)
from CliPP2.core.scalar import (
    partition_constrained_observed_refit,
    scalar_loss,
    scalar_problem_from_model,
)
from CliPP2.core.fusion.torch_backend import (
    resolve_runtime,
)
from CliPP2.io.data import TumorData, tumor_data_fingerprint


EPS = 1e-6


def integer_data(major=((1, 2), (4, 6)), *, observed=None, purity=0.7):
    major = np.asarray(major, dtype=np.float64)
    minor = np.minimum(major, 1.0)
    rho = np.full_like(major, purity)
    normal = np.full_like(major, 2.0)
    scaling = rho / ((1.0 - rho) * normal + rho * (major + minor))
    upper = np.clip(np.minimum(
        1.0, (1.0 - EPS) / np.clip(scaling * major, EPS, None)
    ), EPS, 1.0)
    return TumorData(
        tumor_id="integer-test",
        mutation_ids=[f"m{i}" for i in range(major.shape[0])],
        region_ids=[f"r{i}" for i in range(major.shape[1])],
        alt_counts=np.full_like(major, 24.0),
        total_counts=np.full_like(major, 100.0),
        purity=rho, major_cn=major, minor_cn=minor, normal_cn=normal,
        scaling=scaling,
        phi_upper=upper, phi_init=np.full_like(major, 0.5),
        count_observed=observed,
    )


def independent_candidates(data):
    candidates = np.arange(1, int(np.max(data.major_cn)) + 1)
    valid = candidates <= data.major_cn[..., None]
    return np.where(valid, candidates, 0), np.where(
        valid, -np.log(data.major_cn)[..., None], -np.inf,
    )


def enumeration(data, phi):
    copies, log_prior = independent_candidates(data)
    p = np.clip(data.scaling[..., None] * copies * phi[..., None],
                EPS, 1.0 - EPS)
    joint = log_prior + data.alt_counts[..., None] * np.log(p)
    joint += (data.total_counts - data.alt_counts)[..., None] * np.log1p(-p)
    loss = -logsumexp(joint, axis=-1)
    posterior = np.exp(joint + loss[..., None])
    if data.count_observed is not None:
        loss = np.where(data.count_observed, loss, 0.0)
        posterior = np.where(data.count_observed[..., None], posterior,
                             np.exp(log_prior))
    return loss, posterior


@pytest.mark.parametrize("major", [((1,),), ((6,),), ((1, 2), (4, 6))])
def test_numpy_torch_grid_scalar_and_em_use_all_candidates(major):
    data = integer_data(major)
    model = compile_observed_model(data, eps=EPS)
    runtime = resolve_runtime("cpu", dtype="float64")
    tm = model_to_torch(model, runtime, eps=EPS)
    phi = np.full(model.shape, 0.73)
    loss, posterior = enumeration(data, phi)
    nt = observed_terms_numpy(model, phi, eps=EPS)
    pt = torch.tensor(phi, dtype=torch.float64)
    tt = observed_terms_torch(tm, pt, eps=EPS)
    np.testing.assert_allclose(nt.loss, loss, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(nt.posterior, posterior, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(tt.loss.numpy(), loss, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(tt.posterior.numpy(), posterior, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(tt.gradient.numpy(), nt.gradient, atol=1e-11)
    em = observed_em_terms_torch(tm, pt, responsibilities=tt.posterior, eps=EPS)
    np.testing.assert_allclose(em.loss.numpy(), loss, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(em.gradient.numpy(), nt.gradient, atol=1e-11)
    grid = np.broadcast_to(np.array([0.17, 0.43, 0.73, 0.91]), (*model.shape, 4))
    actual = observed_loss_grid_torch(tm, torch.tensor(grid), eps=EPS).numpy()
    expected = np.stack([enumeration(data, grid[..., j])[0] for j in range(4)], -1)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    for row, col in np.ndindex(model.shape):
        problem = scalar_problem_from_model(model, np.array([row]), col,
                                            lower=EPS, upper=1.0, eps=EPS)
        np.testing.assert_allclose(scalar_loss(problem, grid[row, col]),
                                   expected[row, col], rtol=1e-13, atol=1e-13)


def test_gradient_finite_difference_and_torch_autograd():
    data = integer_data()
    model = compile_observed_model(data, eps=EPS)
    phi = np.array([[0.41, 0.62], [0.74, 0.31]])
    terms = observed_terms_numpy(model, phi, eps=EPS)
    step = 1e-6
    finite_difference = (enumeration(data, phi + step)[0]
                         - enumeration(data, phi - step)[0]) / (2 * step)
    np.testing.assert_allclose(terms.gradient, finite_difference, atol=2e-8, rtol=1e-7)
    tm = model_to_torch(model, resolve_runtime("cpu", dtype="float64"), eps=EPS)
    tensor = torch.tensor(phi, requires_grad=True)
    observed_terms_torch(tm, tensor, eps=EPS).loss.sum().backward()
    np.testing.assert_allclose(tensor.grad.numpy(), terms.gradient, atol=1e-11)


def test_marginalized_loss_is_not_a_hard_maximum():
    data = integer_data(((6,),))
    data = replace(data, alt_counts=np.full(data.alt_counts.shape, 2.0),
                   total_counts=np.full(data.total_counts.shape, 5.0))
    phi = np.array([[0.6]])
    model = compile_observed_model(data, eps=EPS)
    copies, log_prior = independent_candidates(data)
    probability = np.clip(data.scaling[..., None] * copies
                          * phi[..., None], EPS, 1 - EPS)
    joint = log_prior + 2 * np.log(probability)
    joint += 3 * np.log1p(-probability)
    marginal = observed_terms_numpy(model, phi, eps=EPS).loss
    assert np.all(-np.max(joint, axis=-1) - marginal > 0.5)


def test_missing_and_zero_depth_keep_prior_and_zero_loss():
    data = integer_data(observed=np.array([[False, True], [True, True]]))
    alt = data.alt_counts.copy()
    total = data.total_counts.copy()
    alt[1, 1] = total[1, 1] = 0
    data = replace(data, alt_counts=alt, total_counts=total)
    model = compile_observed_model(data, eps=EPS)
    terms = observed_terms_numpy(model, np.full(model.shape, 0.7), eps=EPS)
    expected_loss, expected_posterior = enumeration(data, np.full(model.shape, 0.7))
    np.testing.assert_allclose(terms.loss, expected_loss, atol=1e-13)
    np.testing.assert_allclose(terms.posterior, expected_posterior, atol=1e-13)
    assert terms.loss[0, 0] == terms.gradient[0, 0] == 0
    assert abs(terms.loss[1, 1]) < 1e-13


def test_major_at_most_two_matches_independent_integer_enumeration():
    data = integer_data(((1, 2), (2, 1)), observed=np.array([[True, True], [False, True]]))
    phi = np.array([[0.25, 0.75], [0.6, 0.9]])
    integer_terms = observed_terms_numpy(
        compile_observed_model(data, eps=EPS), phi, eps=EPS)
    expected_loss, expected_posterior = enumeration(data, phi)
    np.testing.assert_allclose(integer_terms.loss, expected_loss, atol=1e-12)
    np.testing.assert_allclose(integer_terms.posterior, expected_posterior, atol=1e-12)


def test_clonal_box_allowed_and_obsolete_model_families_are_rejected():
    data = integer_data(((6,),), purity=1.0)
    data = replace(data, minor_cn=np.zeros((1, 1)), scaling=np.full((1, 1), 1 / 6),
                   phi_upper=np.full((1, 1), 1 - EPS))
    model = compile_observed_model(data, eps=EPS)
    np.testing.assert_array_equal(model.upper, data.phi_upper)
    assert not hasattr(data, "path_likelihood")
    with pytest.raises(TypeError, match="path_likelihood"):
        replace(data, path_likelihood=None)


def test_compiler_rebuilds_candidates_after_cn_replacement():
    data = integer_data(((4,),))
    original = compile_observed_model(data, eps=EPS)
    changed = compile_observed_model(replace(data, major_cn=np.array([[3.0]])), eps=EPS)
    assert original.candidate_shape == (1, 1, 4)
    assert changed.candidate_shape == (1, 1, 3)
    np.testing.assert_array_equal(changed.valid, np.ones((1, 1, 3), dtype=bool))
    np.testing.assert_array_equal(changed.log_prior, np.full((1, 1, 3), -np.log(3.0)))


def test_candidate_model_identity_invalidates_tensor_and_refit_cache():
    data = integer_data(((4,),))
    changed = integer_data(((3,),))
    runtime = resolve_runtime("cpu", dtype="float64")
    from CliPP2.core.fusion.partition_starts import _resolve_partition_runtime
    tensors = model_to_torch(compile_observed_model(changed, eps=EPS), runtime, eps=EPS)
    assert tumor_data_fingerprint(data) != tumor_data_fingerprint(changed)
    assert compile_observed_model(data, eps=EPS).fingerprint != (
        compile_observed_model(changed, eps=EPS).fingerprint)
    with pytest.raises(ValueError, match="runtime source"):
        _resolve_partition_runtime(data=data, model=tensors, eps=EPS)
    stale = compile_observed_model(changed, eps=EPS)
    with pytest.raises(ValueError, match="tumor objective"):
        partition_constrained_observed_refit(
            data, np.array([0]), eps=EPS, tol=1e-5,
            max_iter=16, _model=stale,
        )
