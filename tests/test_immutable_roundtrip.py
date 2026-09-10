"""Copying/saving immutable evidence must revalidate and freeze its arrays."""

import copy
from dataclasses import fields, replace
import pickle

import numpy as np
import pytest

from CliPP2.core.objective import has_proven_convex_observed_loss
from CliPP2.core.scalar import scalar_problem_from_model
from CliPP2.io.data import tumor_data_fingerprint
from CliPP2.model_selection.types import (
    DirectPartition, FusionPartition, PartitionRefitSummary,
)
from test_raw_integrity import raw_fit as raw_fit


def _roundtrip(value, mode):
    if mode == "deepcopy":
        return copy.deepcopy(value)
    options = {} if mode == "pickle_default" else {"protocol": 5}
    return pickle.loads(pickle.dumps(value, **options))


@pytest.fixture(scope="module")
def records(request):
    data, problem, fit = request.getfixturevalue("raw_fit")
    model = problem.problem.source_model
    epsilon = problem.problem.eps
    has_proven_convex_observed_loss(model, eps=epsilon)
    labels = np.zeros(data.num_mutations, dtype=np.int64)
    return dict(
        data=data, model=model, raw=fit, graph=problem.graph_spec,
        scalar=scalar_problem_from_model(
            model, np.arange(data.num_mutations), 0,
            lower=epsilon, upper=1.0, eps=epsilon,
        ),
        fusion=FusionPartition(
            labels, "roundtrip", False, "tolerance_defined_primal",
            mutation_ids=data.mutation_ids,
        ),
        direct=DirectPartition(labels, "roundtrip", "pilot_hessian_ward", data.mutation_ids),
        refit=PartitionRefitSummary(
            labels, "roundtrip", np.full(model.shape, .5),
            np.full((1, data.num_regions), .5), -1.0, True, False,
            tumor_data_fingerprint(data), epsilon,
        ),
    )


@pytest.mark.parametrize("mode", ["deepcopy", "pickle_default", "pickle5"])
@pytest.mark.parametrize("name", ["data", "model", "raw", "graph", "scalar", "fusion", "direct", "refit"])
def test_roundtrips_preserve_immutable_arrays_and_identity(records, mode, name):
    original = records[name]
    restored = _roundtrip(original, mode)
    assert restored is not original
    for item in fields(original):
        source = getattr(original, item.name)
        if not isinstance(source, np.ndarray):
            continue
        value = getattr(restored, item.name)
        np.testing.assert_array_equal(value, source)
        assert value.dtype == source.dtype
        with pytest.raises(ValueError, match="read-only"):
            value.flat[0] = value.flat[0]
        with pytest.raises(ValueError, match="WRITEABLE"):
            value.setflags(write=True)
    if name == "data":
        assert not restored._compiled_models
        assert tumor_data_fingerprint(restored) == tumor_data_fingerprint(original)
        changed = replace(restored, alt_counts=restored.alt_counts + 1)
        assert tumor_data_fingerprint(changed) != tumor_data_fingerprint(restored)
    elif name == "model":
        assert not restored._convexity
        assert restored.fingerprint == original.fingerprint
        assert restored.likelihood_fingerprint == original.likelihood_fingerprint
    elif name == "graph":
        assert restored.fingerprint == original.fingerprint


@pytest.mark.parametrize("mode", ["deepcopy", "pickle_default", "pickle5"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_raw_roundtrip_preserves_dtype_and_mutable_independent_workstate(records, mode, dtype):
    original = replace(records["raw"], phi=records["raw"].phi.astype(dtype))
    restored = _roundtrip(original, mode)
    original_work = original.state.phi.clone()
    restored.state.phi[0, 0] += .1
    np.testing.assert_array_equal(restored.phi, original.phi)
    assert restored.phi.dtype == dtype
    assert restored.objective == original.objective
    assert restored.provenance == original.provenance
    assert original.state.phi.equal(original_work)
