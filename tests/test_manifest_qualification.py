"""Publication completion must not be confused with numerical qualification."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
import torch

from CliPP2 import api, reporting
from CliPP2.config import resolve_fit_config
from CliPP2.io.tumor_txt import load_tumor_txt, write_tumor_txt
from CliPP2.model_selection.search import select_model
from CliPP2.model_selection.types import DirectPartition, DirectPartitionCandidate, SelectedModel


@pytest.fixture(scope="module")
def analysis(tmp_path_factory):
    path = tmp_path_factory.mktemp("qualification") / "tumor.tsv"
    write_tumor_txt(path, pd.DataFrame([dict(
        mutation_id="m1", sample_id="R1", alt_count=20, ref_count=80,
        count_observed=1, purity=0.8, normal_cn=2, segment_id="s1",
        cn_state_id="clonal", cn_state_fraction=1, allele_a_cn=3, allele_b_cn=1,
    )]))
    config = resolve_fit_config(device="cpu", dtype="float64")
    data = load_tumor_txt(path)
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        result = select_model(data=data, fit_config=config, use_warm_starts=False)
    finally:
        torch.set_num_threads(previous)
    return reporting.AnalysisSerialization(data, path, config, result)


def _read(outdir):
    text = (outdir / "tumor_run_manifest.json").read_text()
    # Reject nonstandard NaN/Infinity tokens instead of Python's permissive
    # default JSON decoding; unknown numerical bounds must be null.
    return json.loads(text, parse_constant=lambda value: pytest.fail(value))


def test_complete_publication_retains_unresolved_search_and_refit(analysis, tmp_path):
    reporting.write_analysis_outputs(analysis, outdir=tmp_path)
    manifest = _read(tmp_path)
    qualification = manifest["analysis"]
    assert manifest["status"] == "complete"
    assert qualification["selection"]["status"] == "provisional_unresolved"
    assert not qualification["selection"]["optimum_resolved"]
    assert not qualification["selection"]["global_hybrid_optimum_certified"]
    assert qualification["selection"]["raw_lambda_path_resolved"]
    assert qualification["raw_reference"]["kkt_certified"]
    assert qualification["raw_reference"]["admissible"]
    assert not qualification["raw_reference"]["global_optimum_certified"]
    refit = qualification["refit"]
    assert refit["mode"] == "grid_local"
    assert refit["numerically_resolved"]
    assert not refit["global_optimum_certified"]
    assert refit["global_lower_bound"] is None
    assert refit["global_optimality_gap"] is None
    assert qualification["selected_partition"]["signature"] == analysis.partition.signature
    raw = analysis.raw_fit
    assert qualification["raw_reference"]["graph_hash"] == raw.provenance.original_graph_hash
    assert qualification["raw_reference"]["objective_hash"] == raw.provenance.certificate_problem_hash
    assert qualification["selected_raw_fit"]["objective"] == raw.objective.total
    assert len(manifest["files"]) == 4
    assert len(list(tmp_path.iterdir())) == 5


def test_direct_selection_does_not_inherit_raw_certification(analysis, tmp_path):
    selected = analysis.selected_candidate
    partition = DirectPartition(
        labels=selected.partition.labels, signature=selected.partition.signature,
        source="pilot_hessian_ward", mutation_ids=tuple(analysis.data.mutation_ids),
    )
    direct = DirectPartitionCandidate(
        partition=partition, refit=selected.refit, score=selected.score,
        eligible_for_selection=True, ineligibility_reason="none",
    )
    result = replace(analysis.selection_result, selected_model=SelectedModel(
        raw_reference=analysis.raw_reference, partition_candidate=direct,
    ), selected_lambda_representative=None, selected_kkt_residual=None)
    reporting.write_analysis_outputs(replace(analysis, selection_result=result), outdir=tmp_path)
    qualification = _read(tmp_path)["analysis"]
    assert qualification["selected_partition"]["family"] == "direct_partition"
    assert not qualification["selected_partition"]["raw_partition_certified"]
    assert qualification["selected_raw_fit"] is None
    assert qualification["raw_reference"]["kkt_certified"]
    assert qualification["raw_reference"]["admissible"]


def test_standalone_writer_does_not_invent_search_qualification(analysis, tmp_path):
    reporting.write_fit_outputs(
        outdir=tmp_path, data=analysis.data, raw_fit=analysis.raw_fit,
        partition=analysis.partition, refit=analysis.refit,
    )
    qualification = _read(tmp_path)["analysis"]
    assert qualification["raw_reference"]["kkt_certified"]
    search = qualification["selection"]
    assert search["status"] == "not_provided"
    assert all(value is None for key, value in search.items() if key != "status")


def test_raw_global_and_refit_global_are_not_combined(analysis, tmp_path):
    raw = replace(analysis.raw_fit, certificate=replace(analysis.raw_fit.certificate,
                  global_optimum=False))
    refit = replace(analysis.refit, global_optimum_certified=True,
                    global_lower_bound=-analysis.refit.loglik,
                    global_optimality_gap=0.0, global_certificate_method="fixture_bound")
    reporting.write_fit_outputs(outdir=tmp_path, data=analysis.data, raw_fit=raw,
                                partition=analysis.partition, refit=refit)
    qualification = _read(tmp_path)["analysis"]
    assert qualification["refit"]["global_optimum_certified"]
    assert not qualification["raw_reference"]["global_optimum_certified"]
    assert qualification["selection"]["optimum_resolved"] is None


def test_initial_and_prefit_failure_have_no_qualification(analysis, tmp_path, monkeypatch):
    def fail_before_fit(**kwargs):
        assert _read(tmp_path)["analysis"] is None
        assert _read(tmp_path)["status"] == "running"
        raise RuntimeError("before a qualified fit")

    monkeypatch.setattr("CliPP2.model_selection.search.select_model", fail_before_fit)
    with pytest.raises(RuntimeError, match="before a qualified fit"):
        api.process_tumor(analysis.input_file, tmp_path, fit_config=analysis.fit_config)
    manifest = _read(tmp_path)
    assert manifest["status"] == "failed"
    assert manifest["analysis"] is None


def test_failure_during_publication_can_preserve_validated_fit_evidence(analysis, tmp_path, monkeypatch):
    def fail_link(*args, **kwargs):
        raise OSError("publication fixture failure")

    monkeypatch.setattr(reporting.os, "link", fail_link)
    with pytest.raises(OSError, match="publication fixture failure"):
        reporting.write_analysis_outputs(analysis, outdir=tmp_path)
    manifest = _read(tmp_path)
    assert manifest["status"] == "failed"
    assert manifest["analysis"]["raw_reference"]["kkt_certified"]
    assert manifest["files"] == {}


def test_invalid_partition_never_acquires_qualification(analysis, tmp_path):
    bad = replace(analysis.refit, labels=np.array([9]))
    with pytest.raises(AssertionError, match="labels differ"):
        reporting.write_fit_outputs(outdir=tmp_path, data=analysis.data, raw_fit=analysis.raw_fit,
                                    partition=analysis.partition, refit=bad)
    assert _read(tmp_path)["status"] == "failed"
    assert _read(tmp_path)["analysis"] is None
