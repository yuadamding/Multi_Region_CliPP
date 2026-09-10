"""Reporting must reject mismatched biological rows and stored result evidence."""
from dataclasses import replace
from copy import deepcopy
import json
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from CliPP2 import reporting
from CliPP2.io.data import tumor_data_fingerprint
from CliPP2.io.tumor_txt import load_tumor_txt, write_tumor_txt
from CliPP2.config import resolve_fit_config
from CliPP2.model_selection.search import select_model
from CliPP2.model_selection.types import DirectPartition
from CliPP2.model_selection.proposals import pilot_matrix_hash
from test_integer_reporting import _data, _selection


@pytest.fixture
def fitted():
    data = _data(major=(4, 4), minor=(1, 1), alt=(45, 20), total=(100, 100))
    fields = ("alt_counts", "total_counts", "major_cn", "minor_cn", "normal_cn",
              "purity", "scaling", "phi_upper", "phi_init")
    data = replace(data, region_ids=("R1", "R2"),
                   **{name: np.repeat(getattr(data, name), 2, axis=1) for name in fields})
    return data, *_selection(data)


def _view(fitted, **kwargs):
    data, raw, partition, refit = fitted
    return reporting.AnalysisSerialization(
        kwargs.pop("data", data), raw_fit=kwargs.pop("raw_fit", raw),
        partition=kwargs.pop("partition", partition), refit=kwargs.pop("refit", refit), **kwargs,
    )


@pytest.mark.parametrize("field", ["mutation_ids", "region_ids", "alt_counts", "total_counts",
                                      "major_cn", "minor_cn", "normal_cn", "purity",
                                      "scaling", "phi_upper", "count_observed"])
def test_reporting_binds_ordered_ids_and_all_likelihood_inputs(fitted, field, tmp_path):
    data, raw, partition, refit = fitted
    if field.endswith("_ids"):
        value = tuple(reversed(getattr(data, field)))
    elif field == "count_observed":
        value = np.array([[True, False], [True, True]])
    elif field in ("phi_upper", "purity"):
        value = getattr(data, field) - .05
    else:
        value = getattr(data, field) + (1 if field.endswith("counts") or field.endswith("cn") else .01)
    changed = replace(data, **{field: value})
    with pytest.raises(ValueError, match="Reporting|clonal CN|scaling"):
        reporting.write_fit_outputs(outdir=tmp_path, data=changed, raw_fit=raw,
                                    partition=partition, refit=refit)
    manifest = json.loads((tmp_path / f"{data.tumor_id}_run_manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["analysis"] is None
    assert not list(tmp_path.glob("*.tsv"))


@pytest.mark.parametrize("field", ["likelihood_hash", "box_hash", "eps_hex"])
def test_reporting_checks_typed_base_objective_not_only_source_hash(fitted, field):
    raw = fitted[1]
    base = replace(raw.provenance.objective_key.base, **{field: "corrupt"})
    raw = replace(raw, provenance=replace(raw.provenance,
                  objective_key=replace(raw.provenance.objective_key, base=base)))
    with pytest.raises(ValueError, match="likelihood, box or epsilon"):
        _view(fitted, raw_fit=raw)


@pytest.mark.parametrize("target", ["raw", "refit", "refit_eps"])
def test_reporting_rejects_missing_or_mixed_source_evidence(fitted, target):
    if target == "raw":
        raw = replace(fitted[1], provenance=replace(fitted[1].provenance, source_data_hash=""))
        changes = {"raw_fit": raw}
    else:
        updates = {"likelihood_eps": 2e-6} if target == "refit_eps" else {"source_data_hash": "different-input"}
        changes = {"refit": replace(fitted[3], **updates)}
    with pytest.raises(ValueError, match="identity"):
        _view(fitted, **changes)


def test_reporting_rejects_replaced_box_even_if_input_hash_is_restamped(fitted):
    data = replace(fitted[0], phi_upper=fitted[0].phi_upper - .05)
    fingerprint = tumor_data_fingerprint(data)
    raw = replace(fitted[1], provenance=replace(fitted[1].provenance, source_data_hash=fingerprint))
    refit = replace(fitted[3], source_data_hash=fingerprint)
    with pytest.raises(ValueError, match="likelihood, box or epsilon"):
        _view(fitted, data=data, raw_fit=raw, refit=refit)


@pytest.mark.parametrize("field", ["labels", "phi", "cluster_centers"])
def test_selection_arrays_cannot_be_made_writable(fitted, field):
    values = getattr(fitted[3], field)
    with pytest.raises(ValueError):
        values.setflags(write=True)
    with pytest.raises(ValueError):
        values.flat[0] = .5
    copy = values.copy()
    copy.flat[0] += 1
    assert not np.array_equal(copy, values)
    with pytest.raises(ValueError):
        fitted[2].labels.setflags(write=True)


def test_standalone_final_phi_parent_is_verified(fitted):
    raw, partition = fitted[1:3]
    direct = DirectPartition(partition.labels, partition.signature,
        "final_phi_hessian_ward", partition.mutation_ids, parent_raw_candidate_id=4,
        parent_raw_lambda=raw.provenance.lambda_value,
        parent_raw_phi_hash=pilot_matrix_hash(raw.phi))
    analysis = _view(fitted, partition=direct)
    assert analysis.qualification["selected_raw_fit"] is None
    with pytest.raises(ValueError, match="parent-Phi"):
        _view(fitted, partition=replace(direct, parent_raw_phi_hash="corrupt"))


def test_table_consumers_do_not_repeat_partial_validation(fitted, monkeypatch, tmp_path):
    analysis = _view(fitted)
    expected = analysis.qualification
    analysis.qualification["refit"]["source_data_hash"] = "edited-copy"
    assert analysis.qualification == expected
    monkeypatch.setattr(reporting, "validate_partition_identity",
                        lambda *args: pytest.fail("already validated"))
    reporting.write_analysis_outputs(analysis, outdir=tmp_path)
    manifest = json.loads((tmp_path / f"{analysis.data.tumor_id}_run_manifest.json").read_text())
    assert manifest["analysis"] == expected
    assert manifest["status"] == "complete"


@pytest.fixture(scope="module")
def actual_analysis(tmp_path_factory):
    path = tmp_path_factory.mktemp("reporting-source") / "tumor.tsv"
    write_tumor_txt(path, pd.DataFrame([
        dict(mutation_id=f"m{i}", sample_id=f"R{r}", alt_count=alt, ref_count=100-alt,
             count_observed=1, purity=1, normal_cn=2, segment_id=f"s{i}",
             cn_state_id="clonal", cn_state_fraction=1, allele_a_cn=1, allele_b_cn=1)
        for i, alt in enumerate((10, 40)) for r in (1, 2)
    ]))
    data = load_tumor_txt(path)
    config = resolve_fit_config(device="cpu", dtype="float64")
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        result = select_model(data=data, fit_config=config, use_warm_starts=False)
    finally:
        torch.set_num_threads(previous)
    return reporting.AnalysisSerialization(data, path, config, result)


@pytest.mark.parametrize("field", ["mutation_ids", "region_ids", "alt_counts"])
def test_real_two_mutation_two_region_fit_cannot_be_relabelled(actual_analysis, field):
    data = actual_analysis.data
    assert not np.allclose(actual_analysis.refit.phi[0], actual_analysis.refit.phi[1])
    value = tuple(reversed(getattr(data, field))) if field.endswith("_ids") else data.alt_counts + 1
    with pytest.raises(ValueError, match="Reporting"):
        replace(actual_analysis, data=replace(data, **{field: value}))


@pytest.mark.parametrize("copy_kind", ["deepcopy", "pickle4", "pickle5"])
@pytest.mark.parametrize("standalone", [False, True])
def test_copied_analysis_revalidates_and_refreezes_arrays(actual_analysis, standalone, copy_kind):
    analysis = actual_analysis
    if standalone:
        analysis = reporting.AnalysisSerialization(analysis.data, raw_fit=analysis.raw_fit,
                    partition=analysis.partition, refit=analysis.refit)
    copied = (deepcopy(analysis) if copy_kind == "deepcopy" else
              pickle.loads(pickle.dumps(analysis, protocol=int(copy_kind[-1]))))
    assert copied.qualification == analysis.qualification
    assert copied.raw_fit.provenance.source_data_hash == tumor_data_fingerprint(copied.data)
    assert copied.refit.source_data_hash == tumor_data_fingerprint(copied.data)
    for array in (copied.raw_fit.phi, copied.partition.labels, copied.refit.phi,
                  copied.refit.cluster_centers, copied.data.alt_counts):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_cached_qualification_is_not_pickle_authority(actual_analysis):
    analysis = replace(actual_analysis)
    expected = analysis.qualification
    object.__setattr__(analysis, "_qualification_json", b'{"forged_cached_evidence": true}')
    assert deepcopy(analysis).qualification == expected
    assert pickle.loads(pickle.dumps(analysis)).qualification == expected
