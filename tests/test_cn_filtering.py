"""Whole-mutation eligibility checks for the integer-CN workflow."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from CliPP2.io.data import tumor_data_fingerprint
from CliPP2.io.multiplicity import CLONAL_INTEGER_MODEL_ID
from CliPP2.io import tumor_txt


def _unit(mutation, sample, states, *, observed=True, purity=0.8):
    return [
        dict(
            mutation_id=mutation, sample_id=sample, segment_id=f"seg_{mutation}",
            alt_count=12 if observed else ".", ref_count=88 if observed else ".",
            count_observed=int(observed), purity=purity, normal_cn=2,
            cn_state_id=f"state{index}", cn_state_fraction=fraction,
            allele_a_cn=major, allele_b_cn=minor,
        )
        for index, (fraction, major, minor) in enumerate(states)
    ]


def _load(tmp_path, rows, **kwargs):
    path = tumor_txt.write_tumor_txt(
        tmp_path / "input.tsv", pd.DataFrame(rows), {"tumor_id": "test"}
    )
    return tumor_txt.load_tumor_txt(path, **kwargs)


@pytest.fixture
def cheap_initializer(monkeypatch):
    """Filtering tests isolate eligibility from numerical minimization."""
    monkeypatch.setattr(
        "CliPP2.core.fusion.starts.initialize_marginal_phi",
        lambda data, *, eps: np.asarray(data.phi_init).copy(),
    )


def test_excludes_whole_snv_for_any_region_and_records_both_reasons(
    tmp_path, cheap_initializer,
):
    rows = []
    for sample in ("R2", "R1"):
        rows += _unit("retained", sample, [(1.0, 2, 1)])
        states = [(0.5, 7, 1), (0.5, 3, 1)] if sample == "R2" else [(1.0, 1, 1)]
        rows += _unit("excluded", sample, states, observed=sample == "R1")
    data = _load(tmp_path, rows)
    assert data.mutation_ids == ("retained",)
    assert data.region_ids == ("R1", "R2")
    assert data.num_regions == 2
    report = data.cn_filter_report
    assert report.input_mutation_count == 2
    assert report.retained_mutation_count == 1
    assert report.excluded_mutation_ids == ("excluded",)
    assert [record.reason for record in report.records] == [
        tumor_txt.SUBCLONAL_CN_REGION, tumor_txt.MAJOR_CN_GT_6,
    ]
    assert all(record.sample_id == "R2" for record in report.records)
    assert all(record.n_distinct_cn_states == 2 for record in report.records)
    assert all(record.max_major_cn == 7 for record in report.records)
    with pytest.raises(FrozenInstanceError):
        report.policy_id = "changed"
    with pytest.raises(FrozenInstanceError):
        report.records[0].reason = "changed"


def test_duplicate_cn_pairs_aggregate_but_between_sample_states_may_differ(
    tmp_path, cheap_initializer,
):
    rows = _unit("m", "R1", [(0.2, 4, 2), (0.8, 4, 2)])
    rows += _unit("m", "R2", [(1.0, 2, 0)])
    data = _load(tmp_path, rows)
    np.testing.assert_array_equal(data.major_cn, [[4, 2]])
    np.testing.assert_array_equal(data.minor_cn, [[2, 0]])
    np.testing.assert_array_equal(data.total_counts, [[100, 100]])
    assert not data.cn_filter_report.records
    assert data.path_likelihood.model_id == CLONAL_INTEGER_MODEL_ID


def test_major_six_is_not_total_copy_six_and_no_fraction_threshold(
    tmp_path, cheap_initializer,
):
    rows = _unit("keep", "R1", [(1.0, 6, 6)])
    rows += _unit("high", "R1", [(1.0, 7, 0)])
    rows += _unit("tiny", "R1", [(1.0 - 1e-12, 2, 1), (1e-12, 3, 1)])
    data = _load(tmp_path, rows)
    assert data.mutation_ids == ("keep",)
    np.testing.assert_array_equal(data.major_cn + data.minor_cn, [[12]])
    assert data.cn_filter_report.excluded_mutation_ids == ("high", "tiny")


def test_all_excluded_has_dedicated_error_and_complete_report(tmp_path):
    rows = _unit("b", "R1", [(1.0, 7, 2)])
    rows += _unit("a", "R1", [(0.4, 2, 1), (0.6, 3, 1)])
    with pytest.raises(tumor_txt.NoEligibleSNVsError) as error:
        _load(tmp_path, rows)
    report = error.value.cn_filter_report
    assert error.value.tumor_id == "test"
    assert report.input_mutation_count == 2
    assert report.retained_mutation_count == 0
    assert report.excluded_mutation_ids == ("a", "b")


def test_retained_zero_copy_explicitly_unsupported(tmp_path):
    with pytest.raises(tumor_txt.UnsupportedTumorInputError, match=r"\(0, 0\)"):
        _load(tmp_path, _unit("deleted", "R1", [(1.0, 0, 0)]))


def test_filter_removes_unused_segments_and_preserves_numeric_identity(
    tmp_path, cheap_initializer,
):
    base = _unit("m", "R1", [(1.0, 4, 2)])
    original = _load(tmp_path, base)
    extended = base + _unit("excluded", "R1", [(0.5, 1, 1), (0.5, 3, 1)])
    filtered = _load(tmp_path, extended)
    assert tumor_data_fingerprint(original) == tumor_data_fingerprint(filtered)
    path = tmp_path / "input.tsv"
    metadata, frame = tumor_txt._read_text_table(path)
    validated = tumor_txt._validate_long_table(metadata, frame)
    retained, report = tumor_txt._filter_snv_cn(validated)
    assert list(retained.states_by_segment) == [("R1", "seg_m")]
    assert retained.sample_ids == validated.sample_ids
    assert retained.metadata == validated.metadata
    assert report.excluded_mutation_ids == ("excluded",)
    # Source data are not rewritten by filtering.
    assert "excluded" in path.read_text()


def test_excluded_near_equal_purity_cannot_change_retained_model(tmp_path):
    base = _unit("m", "R1", [(1.0, 4, 2)], purity=.7)
    original = _load(tmp_path, base)
    filtered = _load(tmp_path, base + _unit(
        "excluded", "R1", [(1.0, 7, 0)], purity=.7 - 5e-11,
    ))
    np.testing.assert_array_equal(original.purity, filtered.purity)
    np.testing.assert_array_equal(original.scaling, filtered.scaling)
    np.testing.assert_array_equal(original.phi_init, filtered.phi_init)
    assert tumor_data_fingerprint(original) == tumor_data_fingerprint(filtered)
    # Whole-input validation is not bypassed by subsequent exclusion.
    with pytest.raises(tumor_txt.TumorTxtError, match="purity must be constant"):
        _load(tmp_path, base + _unit(
            "excluded", "R1", [(1.0, 7, 0)], purity=.6,
        ))


@pytest.mark.parametrize("option", [
    {"dosage_prior_penalty": 0.0}, {"dosage_prior_penalty": 3.0},
    {"unsupported_policy": "mask"},
])
def test_obsolete_options_cannot_override_integer_model(tmp_path, option):
    with pytest.raises(TypeError, match="unexpected keyword"):
        _load(tmp_path, _unit("m", "R1", [(1.0, 2, 1)]), **option)


def test_malformed_original_input_fails_before_filtering(tmp_path):
    path = tmp_path / "invalid.tsv"
    frame = pd.DataFrame(_unit("excluded", "R1", [(0.7, 7, 1)]))
    frame.to_csv(path, sep="\t", index=False)
    with pytest.raises(tumor_txt.TumorTxtError, match="sum to one"):
        tumor_txt.load_tumor_txt(path)


def test_real_initializer_is_categorical_and_respects_configured_bounds(tmp_path):
    data = _load(tmp_path, _unit("m", "R1", [(1.0, 4, 0)], purity=1.0), eps=0.02)
    assert data.path_likelihood.model_id == CLONAL_INTEGER_MODEL_ID
    np.testing.assert_allclose(data.phi_upper, [[0.98]])
    assert np.all(data.phi_init >= 0.02)
    assert np.all(data.phi_init <= data.phi_upper)
