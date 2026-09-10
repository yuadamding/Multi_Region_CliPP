"""Returned raw estimates are immutable evidence, unlike solver work state."""

from dataclasses import replace

import numpy as np
import pytest

from CliPP2.config import resolve_fit_config
from CliPP2.core.fusion import solver
from CliPP2.io.data import tumor_data_fingerprint
from test_solver_request import _prepared


@pytest.fixture(scope="module")
def raw_fit():
    data, problem = _prepared()
    options = resolve_fit_config(device="cpu", dtype="float64",
                                 outer_max_iter=2, inner_max_iter=16).solver
    return data, problem, solver.fit_prepared(problem, .1, options,
                                            phi_start=data.phi_init)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_returned_raw_phi_rejects_mutation_and_preserves_dtype(raw_fit, dtype):
    _, _, fit = raw_fit
    result = replace(fit, phi=fit.phi.astype(dtype))
    assert result.phi.dtype == dtype
    with pytest.raises(ValueError, match="read-only"):
        result.phi[0, 0] = .5
    with pytest.raises(ValueError, match="WRITEABLE"):
        result.phi.setflags(write=True)


def test_modified_copy_does_not_change_raw_estimate_or_evidence(raw_fit):
    _, _, fit = raw_fit
    original = fit.phi.copy()
    modified = fit.phi.copy()
    modified[:] = .5
    np.testing.assert_array_equal(fit.phi, original)
    assert not np.shares_memory(modified, fit.phi)
    copied = replace(fit, phi=modified)
    modified[:] = .7
    np.testing.assert_array_equal(copied.phi, .5)
    assert copied.objective is fit.objective
    assert copied.certificate is fit.certificate
    assert copied.provenance is fit.provenance


def test_raw_fit_binds_the_immutable_source_data(raw_fit):
    data, problem, fit = raw_fit
    assert fit.provenance.source_data_hash == problem.data_fingerprint
    assert fit.provenance.source_data_hash == tumor_data_fingerprint(data)
