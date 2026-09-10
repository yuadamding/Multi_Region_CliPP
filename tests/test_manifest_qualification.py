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
    direct_analysis = replace(analysis, selection_result=result)
    reporting.write_analysis_outputs(direct_analysis, outdir=tmp_path)
    qualification = _read(tmp_path)["analysis"]
    assert qualification["selected_partition"]["family"] == "direct_partition"
    assert not qualification["selected_partition"]["raw_partition_certified"]
    assert qualification["selected_raw_fit"] is None
    assert qualification["raw_reference"]["kkt_certified"]
    assert qualification["raw_reference"]["admissible"]
    summary = reporting.analysis_summary(direct_analysis, elapsed_seconds=0)
    assert summary["summary_schema_version"] == 5
    assert all(value is None for key, value in summary.items() if key.startswith("selected_raw_"))
    assert summary["raw_reference_penalized_objective"] == qualification["raw_reference"]["objective"]
    assert summary["raw_reference_kkt_residual"] == qualification["raw_reference"]["kkt_residual"]
    assert summary["raw_reference_kkt_tolerance"] == qualification["raw_reference"]["kkt_tolerance"]
    assert summary["raw_reference_working_dtype"] == qualification["raw_reference"]["working_dtype"]
    assert summary["raw_reference_solve_tolerance"] == qualification["raw_reference"]["solve_tolerance"]
    assert summary["configured_raw_solver_primal_tol"] == direct_analysis.fit_config.solver.tolerance
    for key in ("selected_full_kkt_tolerance", "selected_working_dtype",
                "selected_certificate_audit_dtype", "selected_base_fusion_objective_hash",
                "selected_kkt_residual"):
        assert key not in summary


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


@pytest.mark.parametrize("field", ["value", "loglik", "penalty", "assignment_log_evidence"])
def test_analysis_boundary_rejects_corrupted_candidate_score(analysis, field):
    selected = analysis.selected_candidate
    score = replace(selected.score, **{field: getattr(selected.score, field) + 10})
    changed = replace(selected, score=score)
    result = replace(analysis.selection_result, selected_model=SelectedModel(
        raw_reference=analysis.raw_reference, partition_candidate=changed,
    ))
    with pytest.raises(AssertionError):
        replace(analysis, selection_result=result)


def test_summary_and_publication_use_one_validated_record(analysis, tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, "validate_candidate_identity",
                        lambda *args: pytest.fail("candidate already validated"))
    summary = reporting.analysis_summary(analysis, elapsed_seconds=0)
    reporting.write_analysis_outputs(analysis, outdir=tmp_path)
    qualification = _read(tmp_path)["analysis"]
    assert summary["selected_raw_penalized_objective"] == qualification["selected_raw_fit"]["objective"]
    assert summary["selected_raw_solve_tolerance"] == qualification["selected_raw_fit"]["solve_tolerance"]
    assert summary["selected_refit_numerically_resolved"] == qualification["refit"]["numerically_resolved"]


@pytest.mark.parametrize("field,value", [("n_eff", 0), ("degrees_of_freedom", 2)])
def test_analysis_binds_mathematically_reconstructible_score_dimensions(analysis, field, value):
    # log(max(n_eff, 1)) is zero here: both altered scores still reconstruct,
    # but neither describes the one-mutation, one-region source dataset.
    selected = replace(analysis.selected_candidate,
                       score=replace(analysis.score, **{field: value}))
    result = replace(analysis.selection_result, selected_model=SelectedModel(
        raw_reference=analysis.raw_reference, partition_candidate=selected,
    ))
    with pytest.raises(ValueError, match="score dimensions"):
        replace(analysis, selection_result=result)


def test_analysis_rejects_candidates_from_different_frozen_graphs(analysis):
    raw = analysis.raw_fit
    altered_key = replace(raw.provenance.objective_key,
                          base=replace(raw.provenance.objective_key.base, graph_hash="other-graph"))
    other = replace(raw, provenance=replace(raw.provenance, objective_key=altered_key))
    # Keep the singleton witness coherent with its changed graph; this would
    # otherwise remain a mathematically admissible standalone candidate.
    other = replace(other, certificate=replace(other.certificate,
                    witness=replace(other.certificate.witness, graph_hash="other-graph")))
    selected = replace(analysis.selected_candidate, raw_fit=other)
    result = replace(analysis.selection_result, selected_model=SelectedModel(
        raw_reference=analysis.raw_reference, partition_candidate=selected,
    ))
    with pytest.raises(ValueError, match="frozen base objective"):
        replace(analysis, selection_result=result)
