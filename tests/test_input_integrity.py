"""Public biological arrays must agree with the derived likelihood scaling."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest

from CliPP2.config import resolve_fit_config
from CliPP2.api import fit_fixed_objective, validate_public_tumor_data
from CliPP2.io.tumor_txt import load_tumor_txt, write_tumor_txt


@pytest.fixture
def loaded(tmp_path):
    rows = [dict(
        mutation_id=f"m{i}", sample_id=f"r{j}", alt_count=30, ref_count=70,
        count_observed=1, purity=0.8, normal_cn=2, segment_id=f"s{i}",
        cn_state_id="clonal", cn_state_fraction=1, allele_a_cn=4, allele_b_cn=1,
    ) for i in range(2) for j in range(2)]
    path = write_tumor_txt(tmp_path / "integrity.tsv", pd.DataFrame(rows))
    return load_tumor_txt(path)


@pytest.mark.parametrize("field,value", [
    ("purity", 0.5), ("minor_cn", 2.0), ("normal_cn", 3.0),
])
def test_changed_biology_cannot_keep_stale_scaling(loaded, monkeypatch, field, value):
    original = getattr(loaded, field)
    with pytest.raises(ValueError, match="read-only"):
        original[:] = value
    with pytest.raises(ValueError, match="WRITEABLE"):
        original.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        setattr(loaded, field, np.full(original.shape, value))
    changed = replace(loaded, **{field: np.full(original.shape, value)})
    monkeypatch.setattr(
        "CliPP2.api.fit_prepared",
        lambda **kwargs: pytest.fail("stale scaling must fail before optimization"),
    )
    with pytest.raises(ValueError, match="scaling is inconsistent"):
        fit_fixed_objective(changed, resolve_fit_config(device="cpu"))


def test_purity_one_to_half_reproduces_review_example(loaded):
    data = replace(loaded, purity=np.ones((2, 2)), scaling=np.full((2, 2), 0.2))
    config = resolve_fit_config(device="cpu")
    validate_public_tumor_data(data, config)
    data = replace(data, purity=np.full((2, 2), 0.5))
    assert np.all(data.scaling == 0.2)
    with pytest.raises(ValueError, match="scaling is inconsistent"):
        validate_public_tumor_data(data, config)


@pytest.mark.parametrize("field", [
    "alt_counts", "total_counts", "purity", "major_cn", "minor_cn", "normal_cn",
    "scaling", "phi_upper", "phi_init", "count_observed",
])
def test_broadcastable_but_wrong_shapes_fail_closed(loaded, field):
    invalid = replace(loaded, **{field: getattr(loaded, field)[:1]})
    with pytest.raises(ValueError, match=rf"{field}.*shape"):
        validate_public_tumor_data(invalid, resolve_fit_config(device="cpu"))


@pytest.mark.parametrize("field,value", [
    ("purity", 0), ("purity", 1.1), ("purity", np.nan),
    ("normal_cn", -1), ("normal_cn", np.inf),
    ("major_cn", 7), ("major_cn", 0), ("major_cn", 2.5),
    ("minor_cn", 5), ("minor_cn", -1), ("minor_cn", 0.5),
    ("alt_counts", -1), ("alt_counts", 101), ("alt_counts", 2.5),
    ("total_counts", -1), ("total_counts", 9.5),
    ("scaling", np.nan), ("phi_upper", np.inf), ("phi_init", np.nan),
])
def test_invalid_source_ranges_fail_closed(loaded, field, value):
    changed = np.full((2, 2), value, dtype=np.float64)
    with pytest.raises(ValueError):
        validate_public_tumor_data(
            replace(loaded, **{field: changed}), resolve_fit_config(device="cpu"),
        )


def test_nonconstant_region_purity_requires_reloading(loaded):
    purity = loaded.purity.copy()
    purity[0, 0] = 0.9
    changed = replace(loaded, purity=purity)
    with pytest.raises(ValueError, match="constant within each region"):
        validate_public_tumor_data(changed, resolve_fit_config(device="cpu"))


def test_observation_masks_cannot_be_numeric_truthiness(loaded):
    with pytest.raises(ValueError, match="Boolean array"):
        validate_public_tumor_data(
            replace(loaded, count_observed=np.ones((2, 2))),
            resolve_fit_config(device="cpu"),
        )


def test_explicit_consistent_reconstruction_is_accepted(loaded):
    purity = np.full((2, 2), 0.5)
    scaling = purity / ((1 - purity) * loaded.normal_cn
                       + purity * (loaded.major_cn + loaded.minor_cn))
    data = replace(loaded, purity=purity, scaling=scaling)
    validate_public_tumor_data(data, resolve_fit_config(device="cpu"))


def test_reconstructed_input_copies_caller_owned_arrays(loaded):
    purity = loaded.purity.copy()
    changed = replace(loaded, purity=purity)
    purity[:] = 0.5
    np.testing.assert_array_equal(changed.purity, loaded.purity)
    assert isinstance(changed.mutation_ids, tuple)
    assert isinstance(changed.region_ids, tuple)
