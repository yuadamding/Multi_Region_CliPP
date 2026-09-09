"""Integer support is distinct, complete, uniform, and fingerprint-bound."""

from dataclasses import replace

import numpy as np
import pytest

from CliPP2.io.multiplicity import (
    CLONAL_INTEGER_GENERATOR_VERSION, CLONAL_INTEGER_MODEL_ID,
    CLONAL_INTEGER_PRIOR_MODE, build_clonal_integer_likelihood,
)


@pytest.mark.parametrize("major,minor", [(1, 1), (2, 0), (2, 2), (4, 2), (6, 0)])
def test_exact_candidate_support_and_uniform_normalized_prior(major, minor):
    spec = build_clonal_integer_likelihood(np.array([[major]]))
    assert spec.model_id == CLONAL_INTEGER_MODEL_ID
    assert spec.candidate_generator_version == CLONAL_INTEGER_GENERATOR_VERSION
    assert spec.prior_mode == CLONAL_INTEGER_PRIOR_MODE
    np.testing.assert_array_equal(spec.copies, np.arange(1, major + 1)[None, None])
    np.testing.assert_allclose(np.exp(spec.log_prior), 1.0 / major)
    np.testing.assert_allclose(np.exp(spec.log_prior).sum(axis=-1), 1.0)
    assert spec.valid.all()
    assert np.all(spec.copies[spec.valid] >= 1)


def test_padded_width_is_data_dependent_and_padding_never_has_prior_mass():
    spec = build_clonal_integer_likelihood(np.array([[1, 2], [4, 3]]))
    assert spec.copies.shape == (2, 2, 4)
    np.testing.assert_array_equal(spec.valid.sum(axis=-1), [[1, 2], [4, 3]])
    assert np.all(spec.copies[~spec.valid] == 0)
    assert np.all(spec.log_prior[~spec.valid] == -np.inf)
    assert not spec.copies.flags.writeable
    assert not spec.valid.flags.writeable


@pytest.mark.parametrize("major", [
    np.array([]), np.empty((0, 1)), np.array([2]), np.ones((1, 1, 1)),
    np.array([[np.nan]]), np.array([[np.inf]]), np.array([[2.1]]),
    np.array([[0]]), np.array([[-1]]), np.array([[7]]),
])
def test_rejects_invalid_and_unfiltered_major_cn(major):
    with pytest.raises(ValueError):
        build_clonal_integer_likelihood(major)


def test_prior_validation_does_not_silently_renormalize_or_duplicate():
    spec = build_clonal_integer_likelihood(np.array([[4]]))
    with pytest.raises(ValueError, match="fixed uniform priors"):
        replace(spec, log_prior=np.zeros_like(spec.log_prior))
