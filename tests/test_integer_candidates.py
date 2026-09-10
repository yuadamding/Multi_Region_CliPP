"""Integer support is distinct, complete, uniform, and fingerprint-bound."""

from dataclasses import replace

import numpy as np
import pytest

from CliPP2.io.multiplicity import (
    CLONAL_INTEGER_GENERATOR_VERSION, CLONAL_INTEGER_MODEL_ID,
    CLONAL_INTEGER_PRIOR_MODE,
)
from CliPP2.core.objective import _compile_integer_candidates, compile_observed_model
from test_integer_likelihood import integer_data


@pytest.mark.parametrize("major,minor", [(1, 1), (2, 0), (2, 2), (4, 2), (6, 0)])
def test_exact_candidate_support_and_uniform_normalized_prior(major, minor):
    data = integer_data(((major,),))
    model = compile_observed_model(data, eps=1e-6)
    assert model.model_id == CLONAL_INTEGER_MODEL_ID
    assert model.candidate_generator_version == CLONAL_INTEGER_GENERATOR_VERSION
    assert model.prior_mode == CLONAL_INTEGER_PRIOR_MODE
    np.testing.assert_allclose(model.slope / data.scaling[..., None],
                               np.arange(1, major + 1)[None, None])
    np.testing.assert_allclose(np.exp(model.log_prior), 1.0 / major)
    np.testing.assert_allclose(np.exp(model.log_prior).sum(axis=-1), 1.0)
    assert model.valid.all()
    assert np.all(model.slope[model.valid] > 0)


def test_padded_width_is_data_dependent_and_padding_never_has_prior_mass():
    model = compile_observed_model(integer_data(((1, 2), (4, 3))), eps=1e-6)
    assert model.path_shape == (2, 2, 4)
    np.testing.assert_array_equal(model.valid.sum(axis=-1), [[1, 2], [4, 3]])
    assert np.all(model.slope[~model.valid] == 0)
    assert np.all(model.log_prior[~model.valid] == -np.inf)
    for array in (model.slope, model.log_prior, model.valid):
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            array.setflags(write=True)


@pytest.mark.parametrize("major", [
    np.array([]), np.empty((0, 1)), np.array([2]), np.ones((1, 1, 1)),
    np.array([[np.nan]]), np.array([[np.inf]]), np.array([[2.1]]),
    np.array([[0]]), np.array([[-1]]), np.array([[7]]),
])
def test_rejects_invalid_and_unfiltered_major_cn(major):
    with pytest.raises(ValueError):
        _compile_integer_candidates(major, np.ones_like(major))


def test_prior_validation_does_not_silently_renormalize_or_duplicate():
    model = compile_observed_model(integer_data(((4,),)), eps=1e-6)
    with pytest.raises(ValueError, match="normalize"):
        replace(model, log_prior=np.zeros_like(model.log_prior))


def test_candidate_specification_class_is_removed():
    from CliPP2.io import multiplicity

    assert not hasattr(multiplicity, "IntegerMultiplicitySpec")
    assert not hasattr(multiplicity, "build_clonal_integer_likelihood")
