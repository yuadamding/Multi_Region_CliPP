from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from CliPP2.core.objective import infer_integer_multiplicity_posterior_numpy
from CliPP2.core.objective import compile_observed_model, observed_terms_numpy, make_base_objective_key, make_lambda_objective_key
from CliPP2.core.fusion.types import (
    RawFit, ObjectiveValue, KKTComponents, CertificateResult, ConvergenceResult,
    FitProvenance, WorkCounters,
)
from CliPP2.io.data import CNFilterRecord, CNFilterReport, TumorData, tumor_data_fingerprint
from CliPP2.model_selection.partitions import _partition_signature
from CliPP2.model_selection.types import FusionPartition, PartitionRefitSummary
from CliPP2.reporting import (
    cn_filter_output_table,
    AnalysisSerialization, _mutation_region_output_table,
    write_cn_filter_output,
    write_fit_outputs,
)

EPS = 1e-6


def _data(major=(4,), minor=(1,), alt=(45,), total=(100,)) -> TumorData:
    major = np.asarray(major, dtype=float)[:, None]
    minor = np.asarray(minor, dtype=float)[:, None]
    alt = np.asarray(alt, dtype=float)[:, None]
    total = np.asarray(total, dtype=float)[:, None]
    scaling = 1.0 / (major + minor)
    upper = np.minimum(1.0, (1.0 - EPS) / (scaling * major))
    return TumorData(
        tumor_id="integer_reporting",
        mutation_ids=[f"m{i}" for i in range(len(major))],
        region_ids=["R1"],
        alt_counts=alt,
        total_counts=total,
        purity=np.ones_like(alt),
        major_cn=major,
        minor_cn=minor,
        normal_cn=np.full_like(alt, 2.0),
        scaling=scaling,
        phi_upper=upper,
        phi_init=np.minimum(0.5, upper),
    )


def _selection(data: TumorData, phi: float = 0.75):
    labels = np.zeros(data.num_mutations, dtype=np.int64)
    mutation_ids = tuple(data.mutation_ids)
    signature = _partition_signature(labels, mutation_ids)
    partition = FusionPartition(
        labels=labels,
        signature=signature,
        certified=True,
        source="verified_primal_equalities",
        mutation_ids=mutation_ids,
    )
    refit = PartitionRefitSummary(
        labels=labels,
        partition_signature=signature,
        phi=np.full(data.alt_counts.shape, phi),
        cluster_centers=np.full((1, data.num_regions), phi),
        loglik=-10.0,
        finite_candidate_found=True,
        global_optimum_certified=False,
        source_data_hash=tumor_data_fingerprint(data),
        likelihood_eps=EPS,
    )
    raw_phi = np.full(data.alt_counts.shape, 0.25)
    model = compile_observed_model(data, eps=EPS)
    raw = RawFit(
        phi=raw_phi, objective=ObjectiveValue(float(observed_terms_numpy(model, raw_phi, eps=EPS).loss.sum())),
        certificate=CertificateResult(
            components=KKTComponents(1, 0, 0, 0), certified=False, admissible=False,
            global_optimum=False, status="fixture_unqualified", tolerance=.004,
            scope="full_original_graph", gradient_scope="observed_objective",
            directional_admissible=False, witness=None, working_residual=1,
            working_dtype="float64", audit_dtype="float64", precision_polished=False,
            precision_polish_delta=0, residual_method="fixture", fallback_reason="none",
        ), convergence=ConvergenceResult(False, 0), work=WorkCounters(), state=None,
        provenance=FitProvenance(
            objective_key=make_lambda_objective_key(
                make_base_objective_key(model, graph_hash="fixture-graph", eps=EPS), lambda_value=.1),
            source_data_hash=tumor_data_fingerprint(data), device="cpu", dtype="float64",
            inner_solver="fixture", global_optimality_basis="not_certified", likelihood_eps=EPS,
        ),
    )
    return raw, partition, refit


def mutation_region_output_table(data, raw_fit, partition, refit, *, eps=None):
    return _mutation_region_output_table(AnalysisSerialization(
        data, raw_fit=raw_fit, partition=partition, refit=refit, eps=eps,
    ))


def test_intermediate_map_uses_full_posterior_not_endpoints_or_mean():
    data = _data()
    phi = np.array([[0.75]])
    result = infer_integer_multiplicity_posterior_numpy(data, phi, eps=EPS)
    p = np.array([0.15, 0.30, 0.45, 0.60])
    log_joint = -np.log(4.0) + 45 * np.log(p) + 55 * np.log1p(-p)
    expected = np.exp(log_joint - log_joint.max())
    expected /= expected.sum()
    np.testing.assert_allclose(result.posterior[0, 0], expected, atol=1e-14)
    assert result.multiplicity_call.dtype == np.int64
    assert result.multiplicity_call[0, 0] == 3
    assert result.map_probability[0, 0] == pytest.approx(expected[2])
    assert result.candidate_count[0, 0] == 4
    assert result.informative[0, 0]
    assert not np.isclose(np.dot(expected, np.arange(1, 5)), 3.0, atol=1e-8)


def test_missing_zero_depth_and_singleton_reporting():
    data = _data(
        major=(4, 4, 1, 1),
        minor=(1, 1, 1, 1),
        alt=(45, 0, 0, 10),
        total=(100, 0, 0, 100),
    )
    data = replace(data, count_observed=np.array([[False], [True], [True], [False]]))
    result = infer_integer_multiplicity_posterior_numpy(
        data, np.full((4, 1), 0.75), eps=EPS
    )
    np.testing.assert_array_equal(result.informative, False)
    np.testing.assert_array_equal(result.multiplicity_call, 1)
    np.testing.assert_allclose(result.posterior[:2, 0], 0.25)
    np.testing.assert_array_equal(result.posterior[2:, 0], [[1, 0, 0, 0]] * 2)
    table = mutation_region_output_table(data, *_selection(data))
    assert str(table.multiplicity_call.dtype) == "Int64"
    assert table.multiplicity_call.isna().tolist() == [True, True, False, False]
    assert table.multiplicity_call.iloc[2:].tolist() == [1, 1]
    assert table.multiplicity_candidates.tolist() == ["1,2,3,4", "1,2,3,4", "1", "1"]
    np.testing.assert_allclose(table.multiplicity_call_probability, [0.25, 0.25, 1, 1])


def test_output_uses_refit_phi_and_has_stable_six_probability_columns():
    data = _data()
    raw, partition, refit = _selection(data)
    assert infer_integer_multiplicity_posterior_numpy(
        data, raw.phi, eps=EPS
    ).multiplicity_call[0, 0] == 4
    table = mutation_region_output_table(data, raw, partition, refit, eps=EPS)
    assert table.loc[0, "phi"] == 0.75
    assert table.loc[0, "multiplicity_call"] == 3
    assert table.loc[0, "multiplicity_candidate_count"] == 4
    assert table.loc[0, "multiplicity_informative"]
    probability_columns = [f"multiplicity_p{value}" for value in range(1, 7)]
    np.testing.assert_allclose(table[probability_columns].sum(axis=1), 1.0)
    np.testing.assert_array_equal(table[["multiplicity_p5", "multiplicity_p6"]], 0.0)
    for forbidden in ("gamma_major", "major_call", "map_path", "pre_switch_path_probability"):
        assert forbidden not in table


def test_six_candidate_posterior_matches_canonical_evaluator():
    data = _data(major=(6, 2), minor=(0, 2), alt=(50, 30), total=(100, 100))
    phi = np.full((2, 1), 0.75)
    result = infer_integer_multiplicity_posterior_numpy(data, phi, eps=EPS)
    model = compile_observed_model(data, eps=EPS)
    expected = observed_terms_numpy(model, phi, eps=EPS).posterior
    np.testing.assert_array_equal(result.posterior, expected)
    np.testing.assert_array_equal(result.candidate_count[:, 0], [6, 2])
    np.testing.assert_array_equal(result.posterior[1, 0, 2:], 0)


@pytest.mark.parametrize("phi", [np.ones(1), np.array([[np.nan]]), np.array([[0.0]]), np.array([[1.01]])])
def test_posterior_rejects_invalid_phi(phi):
    with pytest.raises(ValueError, match="phi"):
        infer_integer_multiplicity_posterior_numpy(_data(), phi, eps=EPS)


def test_posterior_rejects_above_clonal_box():
    data = _data(major=(6,), minor=(0,))
    with pytest.raises(ValueError, match="bounds"):
        infer_integer_multiplicity_posterior_numpy(data, np.ones((1, 1)), eps=EPS)


@pytest.mark.parametrize("eps", [0.0, 0.5, float("nan")])
def test_posterior_rejects_invalid_eps(eps):
    with pytest.raises(ValueError, match="eps"):
        infer_integer_multiplicity_posterior_numpy(_data(), np.full((1, 1), 0.75), eps=eps)


def test_output_rejects_eps_mismatch():
    data = _data()
    with pytest.raises(ValueError, match="likelihood provenance"):
        mutation_region_output_table(data, *_selection(data), eps=EPS * 2)


def test_reporting_rejects_untyped_raw_evidence():
    data = _data()
    _, partition, refit = _selection(data)
    with pytest.raises(TypeError, match="typed RawFit"):
        mutation_region_output_table(data, object(), partition, refit)


def test_exclusion_output_preserves_both_reasons_and_empty_header(tmp_path):
    report = CNFilterReport(
        policy_id="test_policy",
        input_mutation_count=1,
        retained_mutation_count=0,
        excluded_mutation_ids=("excluded",),
        records=(
            CNFilterRecord("excluded", "R1", "s1", "SUBCLONAL_CN_REGION", 2, 7),
            CNFilterRecord("excluded", "R1", "s1", "MAJOR_CN_GT_6", 2, 7),
        ),
    )
    path = write_cn_filter_output(outdir=tmp_path, tumor_id="no_eligible", report=report)
    table = pd.read_csv(path, sep="\t")
    assert table.mutation_id.tolist() == ["excluded", "excluded"]
    assert table.reason.tolist() == ["SUBCLONAL_CN_REGION", "MAJOR_CN_GT_6"]
    empty_path = write_cn_filter_output(outdir=tmp_path, tumor_id="no_exclusions", report=None)
    empty = pd.read_csv(empty_path, sep="\t")
    assert empty.empty
    assert empty.columns.tolist() == table.columns.tolist()
    assert cn_filter_output_table("empty", replace(report, records=())).empty


def test_writer_retains_only_data_mutations_and_writes_exclusions(tmp_path):
    data = _data()
    data = replace(data, cn_filter_report=CNFilterReport(
        policy_id="test_policy",
        input_mutation_count=2,
        retained_mutation_count=1,
        excluded_mutation_ids=("excluded",),
        records=(CNFilterRecord("excluded", "R1", "s1", "MAJOR_CN_GT_6", 1, 7),),
    ))
    raw, partition, refit = _selection(data)
    write_fit_outputs(
        outdir=tmp_path, data=data, raw_fit=raw, partition=partition, refit=refit, eps=EPS
    )
    assert len(list(tmp_path.glob("*.tsv"))) == 4
    cluster = pd.read_csv(tmp_path / f"{data.tumor_id}_cluster_centers.tsv", sep="\t")
    assert cluster.cluster_size.sum() == 1
    for suffix in ("mutation_clusters", "mutation_region_multiplicity"):
        table = pd.read_csv(tmp_path / f"{data.tumor_id}_{suffix}.tsv", sep="\t")
        assert table.mutation_id.tolist() == ["m0"]
