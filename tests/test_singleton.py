"""The whole-mutation filter may leave one valid, separable scalar problem."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from CliPP2.config import resolve_fit_config
from CliPP2.core.fusion import starts
from CliPP2.core.objective import compile_observed_model, observed_terms_numpy
from CliPP2.io.tumor_txt import load_tumor_txt, write_tumor_txt
from CliPP2.model_selection import search
from CliPP2.model_selection.candidates import validate_candidate_identity
from CliPP2.model_selection.scoring import raw_candidate_has_exact_fusion_certificate
from CliPP2.model_selection.types import is_zero_edge_singleton
from CliPP2.api import process_tumor


@pytest.fixture(autouse=True)
def one_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _input(tmp_path, *, filtered=False, regions=1):
    rows = []
    for index in range(regions):
        row = dict(
            mutation_id="kept", sample_id=f"r{index}", alt_count=20 + index * 5,
            ref_count=80 - index * 5, count_observed=1, purity=0.8, normal_cn=2,
            segment_id="s1", cn_state_id="clonal", cn_state_fraction=1,
            allele_a_cn=3 + index, allele_b_cn=1,
        )
        rows.append(row)
        if filtered:
            rows.append(dict(row, mutation_id="excluded", segment_id="s2", allele_a_cn=7))
    path = tmp_path / "singleton.tsv"
    write_tumor_txt(path, pd.DataFrame(rows))
    return path


def _config(**kwargs):
    return resolve_fit_config(device="cpu", dtype="float64", **kwargs)


@pytest.mark.parametrize("filtered,regions", [(False, 1), (True, 1), (False, 3), (True, 3)])
def test_original_and_filtered_singleton_skip_the_partition_lambda_ladder(
    tmp_path, monkeypatch, filtered, regions,
):
    def forbidden(**kwargs):
        pytest.fail("Singleton entered the partition/lambda search")

    monkeypatch.setattr(search, "_partition_guided_admm_selection", forbidden)
    path = _input(tmp_path, filtered=filtered, regions=regions)
    original = path.read_bytes()
    config = _config()
    data = load_tumor_txt(path, eps=config.eps)
    assert data.num_mutations == 1
    result = search.select_model(data=data, fit_config=config, use_warm_starts=True)
    candidate = result.selected_model.partition_candidate
    fit = candidate.raw_fit
    validate_candidate_identity(candidate)
    assert raw_candidate_has_exact_fusion_certificate(candidate)
    assert result.selection_method == "singleton_scalar_no_edges"
    assert result.num_candidates == 1
    assert result.selected_lambda_representative == 0.0
    assert result.raw_lambda_path_resolved
    assert not result.selection_boundary_unresolved
    assert not result.selection_hits_lower_boundary
    assert not result.selection_hits_upper_boundary
    assert candidate.partition.labels.tolist() == [0]
    assert candidate.refit.phi.shape == (1, regions)
    assert fit.provenance.inner_solver == "closed_form_projection"
    assert fit.convergence.stage_inner_iterations == 0
    assert len(fit.provenance.scalar_pilot_certificates) == regions
    assert is_zero_edge_singleton(fit)
    assert fit.certificate.audit_dtype == "float64"
    assert fit.certificate.tolerance == 0.004
    assert fit.certificate.components.residual <= 0.004
    model = compile_observed_model(data, eps=config.eps)
    expected = observed_terms_numpy(model, fit.phi, eps=config.eps).loss.sum()
    np.testing.assert_allclose(fit.objective.total, expected, atol=1e-10)
    assert path.read_bytes() == original


@pytest.mark.parametrize("filtered", [False, True])
def test_singleton_pipeline_publishes_integer_outputs(tmp_path, filtered):
    path = _input(tmp_path, filtered=filtered)
    destination = tmp_path / "output"
    summary = process_tumor(path, destination, fit_config=_config())
    assert summary["selected_n_clusters"] == 1
    assert summary["raw_reference_objective_certified"]
    assert summary["retained_mutation_count"] == 1
    assert summary["excluded_mutation_count"] == int(filtered)
    mutations = pd.read_csv(destination / "singleton_mutation_clusters.tsv", sep="\t")
    multiplicity = pd.read_csv(destination / "singleton_mutation_region_multiplicity.tsv", sep="\t")
    assert mutations.mutation_id.tolist() == ["kept"]
    assert multiplicity.mutation_id.tolist() == ["kept"]
    assert multiplicity.multiplicity_call.iloc[0] in (1, 2, 3)
    np.testing.assert_allclose(multiplicity.phi, mutations.phi_r0)


def test_unresolved_scalar_bound_is_not_upgraded_to_global_claim(tmp_path, monkeypatch):
    certify = starts.certify_scalar_minimum

    def exhausted(problem, **kwargs):
        return certify(problem, **dict(kwargs, max_intervals=1))

    monkeypatch.setattr(starts, "certify_scalar_minimum", exhausted)
    data = load_tumor_txt(_input(tmp_path))
    result = search.select_model(data=data, fit_config=_config(), use_warm_starts=False)
    fit = result.selected_model.raw_reference.raw_fit
    certificates = fit.provenance.scalar_pilot_certificates
    assert len(certificates) == 1
    assert not certificates[0].globally_certified
    assert certificates[0].optimality_gap > 0
    assert fit.certificate.certified  # Local audit and global enclosure differ.
    assert not fit.certificate.global_optimum
    assert not result.global_hybrid_optimum_certified
    assert not result.selection_optimum_resolved


def test_singleton_exception_does_not_admit_multimutation_zero_penalty(tmp_path):
    data = load_tumor_txt(_input(tmp_path))
    result = search.select_model(data=data, fit_config=_config(), use_warm_starts=False)
    candidate = result.selected_model.raw_reference
    invalid_fit = replace(candidate.raw_fit, phi=np.repeat(candidate.raw_fit.phi, 2, axis=0))
    assert not is_zero_edge_singleton(invalid_fit)
    assert not replace(candidate, raw_fit=invalid_fit).raw_objective_certified


def test_singleton_residual_gate_still_fails_closed(tmp_path, monkeypatch):
    fit_fixed_objective = search.fit_fixed_objective

    def failing_audit(*args, **kwargs):
        fit = fit_fixed_objective(*args, **kwargs)
        # Preserve the reported flags to ensure the shared admission predicate
        # independently enforces the measured residual, not just a success bit.
        return replace(fit, certificate=replace(
            fit.certificate,
            components=replace(fit.certificate.components, stationarity=0.0041),
        ))

    monkeypatch.setattr(search, "fit_fixed_objective", failing_audit)
    data = load_tumor_txt(_input(tmp_path))
    with pytest.raises(search.NoEligibleModelSelectionCandidatesError):
        search.select_model(data=data, fit_config=_config(), use_warm_starts=False)
