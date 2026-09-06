"""Dosage evidence, numerical boundaries, and public categorical likelihood."""

from dataclasses import replace
import csv

import numpy as np
import pytest
import torch

from CliPP2.core.objective import (
    compile_observed_model, model_to_torch, observed_terms_numpy, observed_terms_torch,
)
from CliPP2.core.fusion.torch_backend import resolve_runtime
from CliPP2.core.posterior import summarize_posterior_numpy
from CliPP2.io.tumor_txt import load_tumor_txt, write_tumor_txt


EPS = 1e-6


def rows(states, *, mutation="m0", **fields):
    return [dict(
        mutation_id=mutation, sample_id="r1", alt_count=25, ref_count=75,
        count_observed=1, purity=0.8, normal_cn=2, segment_id=mutation,
        cn_state_id=f"c{i}", cn_state_fraction=fraction,
        allele_a_cn=major, allele_b_cn=minor,
    ) | fields for i, (fraction, major, minor) in enumerate(states)]


def load(tmp_path, records):
    return load_tumor_txt(write_tumor_txt(tmp_path / "dosage.tsv", records))


def model(data):
    return compile_observed_model(data, major_prior=0.5, eps=EPS)


@pytest.mark.parametrize("phi", [0.49, 0.5, 0.50000001, 0.500000010000001, 0.7])
def test_positive_path_classes_partition_posterior_at_tolerance_edge(tmp_path, phi):
    source = model(load(tmp_path, rows([(0.5, 1, 0), (0.5, 2, 0)])))
    posterior = summarize_posterior_numpy(source, np.asarray([[phi]]), eps=EPS)
    single = posterior.single_copy_probability.item()
    amplified = posterior.amplified_mutant_copy_probability.item()
    assert single + amplified == pytest.approx(1.0, abs=1e-14)
    if phi <= 0.5:
        canonical = (source.first_copy == 1) & (source.second_copy == 1)
        assert single > posterior.path_probability[canonical].sum()


@pytest.mark.parametrize("case,fields,phi", [
    ("zero_depth", dict(alt_count=0, ref_count=0), 0.5),
    ("not_likelihood_included", dict(count_observed=0), 0.5),
    ("numerical_ccf_floor", dict(alt_count=0, ref_count=100), EPS),
])
def test_all_dosage_summaries_share_reportability(tmp_path, case, fields, phi):
    source = model(load(tmp_path, rows([(1.0, 2, 0)], **fields)))
    posterior = summarize_posterior_numpy(source, np.asarray([[phi]]), eps=EPS)
    for name in (
        "single_copy_probability", "amplified_mutant_copy_probability",
        "amplified_mutant_copy_call", "expected_multiplicity", "map_multiplicity",
        "expected_mutant_copy_mass", "map_mutant_copy_mass",
    ):
        assert np.isnan(getattr(posterior, name).item()), name
    assert not posterior.dosage_reportable.item()
    assert posterior.dosage_status.item() == case


def test_unidentified_ccf_suppresses_dosage(tmp_path):
    source = model(load(tmp_path, rows([(1.0, 3, 2)])))
    posterior = summarize_posterior_numpy(
        source, np.asarray([[0.5]]), eps=EPS, reportable=np.asarray([[False]]),
    )
    assert not posterior.dosage_reportable.item()
    assert posterior.dosage_status.item() == "ccf_not_reportable"
    assert np.isnan(posterior.amplified_mutant_copy_call.item())


def test_subunit_programmatic_paths_are_not_forced_into_single_copy_class(tmp_path):
    data = load(tmp_path, rows([(1.0, 1, 0)]))
    paths = replace(data.emission_paths, first_copy=np.asarray([[[0.5]]]),
                    second_copy=np.asarray([[[0.5]]]))
    posterior = summarize_posterior_numpy(
        model(replace(data, emission_paths=paths)), np.asarray([[0.5]]), eps=EPS,
    )
    assert posterior.single_copy_probability.item() == 0
    assert posterior.amplified_mutant_copy_probability.item() == 0


def test_prior_mass_exposes_prior_driven_single_copy_support(tmp_path):
    source = model(load(tmp_path, rows([(1.0, 2, 0)], alt_count=1, ref_count=1)))
    posterior = summarize_posterior_numpy(source, np.asarray([[0.5]]), eps=EPS)
    prior = 1 / (1 + np.exp(-3))
    expected = prior * 0.16 / (prior * 0.16 + (1 - prior) * 0.24)
    assert posterior.single_copy_prior_probability.item() == pytest.approx(prior)
    assert posterior.single_copy_probability.item() == pytest.approx(expected)
    assert expected < prior


def test_local_context_preserves_support_priors_and_likelihood(tmp_path):
    records = rows([(1.0, 3, 2)])
    alone = model(load(tmp_path, records))
    together = model(load(tmp_path, records + rows([(0.7, 2, 0), (0.3, 4, 0)], mutation="m1")))
    for field in ("first_copy", "second_copy", "switch", "log_prior", "first_scale"):
        np.testing.assert_array_equal(getattr(alone, field)[0, 0, alone.valid[0, 0]],
                                      getattr(together, field)[0, 0, together.valid[0, 0]])
    for phi in (0.2, 0.7, 1.0):
        a = observed_terms_numpy(alone, np.full(alone.shape, phi), eps=EPS)
        b = observed_terms_numpy(together, np.full(together.shape, phi), eps=EPS)
        np.testing.assert_allclose(a.loss[0], b.loss[0], rtol=0, atol=1e-12)


@pytest.mark.parametrize("device,dtype,atol", [
    ("cpu", "float64", 1e-10), ("cpu", "float32", 3e-5),
    ("cuda", "float64", 1e-10), ("cuda", "float32", 3e-5),
])
def test_categorical_likelihood_and_gradient_parity(tmp_path, device, dtype, atol):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    source = model(load(tmp_path, rows([(1.0, 3, 2)]) +
                        rows([(0.7, 2, 0), (0.3, 4, 0)], mutation="m1")))
    runtime = resolve_runtime(device, dtype=dtype)
    target = model_to_torch(source, runtime)
    for phi in (0.299, 0.301, 0.699, 0.701):
        value = np.full(source.shape, phi)
        expected = observed_terms_numpy(source, value, eps=EPS)
        actual = observed_terms_torch(
            target, torch.tensor(value, device=runtime.device, dtype=runtime.dtype), eps=EPS,
        )
        for field in ("loss", "gradient", "posterior"):
            np.testing.assert_allclose(getattr(actual, field).cpu().numpy(),
                                       getattr(expected, field), rtol=atol, atol=atol)


def test_subclonal_parent_likelihood_and_prior_hashes_are_preserved():
    from pathlib import Path

    source = model(load_tumor_txt(Path(__file__).resolve().parents[2] / "examples/exampleTumor1.tsv"))
    # Frozen before the clonal-loader migration: includes prior arrays and box.
    assert source.fingerprint == "483ed2ad3a8c5e2f414199a5f04b26674d0626edaad58eebe59fddc8711c329d"
    assert source.likelihood_fingerprint == "eca19a794f4efaa99da6d1454440ff29e291d8f2ed5baf644375b7d3d8e1fb8f"


@pytest.mark.parametrize("states,reason,fast", [
    ([(2, 0)], "none", True),
    ([(2, 0), (2, 1)], "heterogeneous_binary_priors", False),
    ([(3, 2)], "padded_path_count_not_two", False),
    ([(4, 2)], "padded_path_count_not_two", False),
])
def test_dispatch_diagnostic_preserves_shared_prior_gate(tmp_path, states, reason, fast):
    records = [row for i, (major, minor) in enumerate(states)
               for row in rows([(1.0, major, minor)], mutation=f"m{i}")]
    source = model(load(tmp_path, records))
    assert source.uses_binary_linear_mixture_fast_path is fast
    assert source.binary_linear_mixture_exclusion_reason == reason
    assert not source.has_internal_switches


def test_writer_blanks_dosage_for_zero_depth_and_lower_bound(tmp_path):
    from CliPP2.config import resolve_fit_config
    from CliPP2.core.bic import fixed_partition_bic
    from CliPP2.core.scalar import partition_constrained_observed_refit
    from CliPP2.model_selection.partitions import partition_signature
    from CliPP2.model_selection.types import (
        CandidateRecord, DirectPartition, DirectPartitionCandidate,
        SearchReport, SecondaryFallbackResult,
    )
    from CliPP2.reporting import AnalysisView, analysis_summary, write_analysis_outputs
    from validate_outputs_v4 import _validate_mutations

    records = rows([(1.0, 3, 2)], purity=1.0)
    records += rows([(1.0, 3, 2)], mutation="m1", purity=1.0, alt_count=0, ref_count=0)
    records += rows([(1.0, 1, 0)], mutation="m2", purity=1.0, alt_count=0, ref_count=100)
    data = load(tmp_path, records)
    labels = np.asarray([0, 0, 1])
    refit = partition_constrained_observed_refit(
        data, labels, major_prior=0.5, eps=EPS, tol=1e-5, max_iter=4096,
    )
    refit = replace(refit, partition_signature=partition_signature(labels, data.mutation_ids))
    partition = DirectPartition(labels=labels, signature=refit.partition_signature,
                                source="pilot_hessian_ward_cem", mutation_ids=data.mutation_ids)
    score = fixed_partition_bic(loglik=refit.loglik, num_clusters=2, data=data,
                                partition_signature=refit.partition_signature,
                                loglik_uncertainty=refit.global_optimality_gap)
    candidate = DirectPartitionCandidate(partition=partition, refit=refit, score=score,
                                         eligible_for_selection=True, ineligibility_reason="none")
    outcome = SecondaryFallbackResult(
        selected_partition=candidate, best_raw_attempt=None,
        reason="NoCertifiedRawReferenceError: reporting fixture",
        report=SearchReport(records=(CandidateRecord(candidate_id=0, candidate=candidate),),
                            selected_id=0, selection_method="hybrid-ward-cem-bic-v1",
                            adaptive_search_stop_reason="reporting_fixture", num_candidates_certified=0),
    )
    analysis = AnalysisView(data=data, input_file=tmp_path / "dosage.tsv",
                            fit_config=resolve_fit_config(computation_profile="fast", device="cpu", dtype="float64"),
                            selection_result=outcome)
    summary = analysis_summary(analysis, elapsed_seconds=0.0)
    outdir = tmp_path / "result"
    write_analysis_outputs(analysis, outdir=outdir, summary=summary)
    output = outdir / "dosage_mutations.tsv"
    assert _validate_mutations(output, summary) == 3
    with output.open(newline="") as handle:
        result = list(csv.DictReader(handle, delimiter="\t"))
    assert result[1]["phi_statistically_identified"] == "1"
    assert result[1]["dosage_status"] == "zero_depth"
    assert result[2]["dosage_status"] == "numerical_ccf_floor"
    for row in result[1:]:
        assert row["dosage_reportable"] == "0"
        assert row["single_copy_probability"] == row["amplified_mutant_copy_call"] == ""


def _validator_fixture(tmp_path):
    from validate_outputs_v4 import DOSAGE_COLUMNS

    record = dict.fromkeys(DOSAGE_COLUMNS, "") | dict(
        phi="0.5", cluster_label="1", count_available="1", likelihood_supported="1",
        likelihood_included="1", phi_statistically_identified="1", alt_count="1", ref_count="1",
        dosage_reportable="1", dosage_status="conditional_at_refit_ccf",
        single_copy_probability="0.9", single_copy_prior_probability="0.95",
        amplified_mutant_copy_probability="0.1", amplified_mutant_copy_call="0",
    )
    metadata = dict(dosage_reporting_policy_id="conditional_positive_depth_excess_mass_v2",
                    dosage_conditioning="selected_partition_refit_ccf", dosage_ccf_lower_bound=EPS,
                    dosage_ccf_floor_tolerance=1e-8, dosage_mass_tolerance=1e-8,
                    dosage_positive_path_family=True)
    return record, metadata


@pytest.mark.parametrize("changes", [
    {"single_copy_probability": "garbage"}, {"single_copy_probability": "nan"},
    {"single_copy_probability": "inf"}, {"single_copy_probability": "1.01"},
    {"single_copy_probability": "-0.01"}, {"single_copy_probability": ""},
    {"amplified_mutant_copy_probability": "0.3"}, {"amplified_mutant_copy_call": "1"},
    {"alt_count": "0", "ref_count": "0", "dosage_reportable": "0", "dosage_status": "zero_depth"},
    {"phi": str(EPS)}, {"likelihood_included": "0"},
    {"phi_statistically_identified": "0"}, {"dosage_status": "invented"},
])
def test_validator_rejects_invalid_dosage_values_and_evidence(tmp_path, changes):
    from validate_outputs_v4 import _validate_mutations

    record, metadata = _validator_fixture(tmp_path)
    path = tmp_path / "mutations.tsv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=record, delimiter="\t")
        writer.writeheader()
        writer.writerow(record)
    assert _validate_mutations(path, metadata) == 1
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=record, delimiter="\t")
        writer.writeheader()
        writer.writerow(record | changes)
    with pytest.raises(RuntimeError):
        _validate_mutations(path, metadata)
