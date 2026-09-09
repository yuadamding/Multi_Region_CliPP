"""Tiny CPU execution checks, not a CUDA or cohort release qualification."""

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import gammaln, logsumexp

from CliPP2.config import CertificateConfig, resolve_fit_config
from CliPP2.core.bic import fixed_partition_bic, fixed_partition_dirichlet_score
from CliPP2.core.fusion.graph import build_complete_uniform_graph
from CliPP2.core.fusion.multiplicity import infer_integer_multiplicity_posterior_numpy
from CliPP2.api import fit_fixed_objective
from CliPP2.core.objective import compile_observed_model, observed_terms_numpy
from CliPP2.core.scalar import partition_constrained_observed_refit
from CliPP2.io.data import tumor_data_fingerprint
from CliPP2.io.tumor_txt import load_tumor_txt, write_tumor_txt
from CliPP2.io.multiplicity import (
    CLONAL_INTEGER_MODEL_ID, CLONAL_INTEGER_GENERATOR_VERSION, CLONAL_INTEGER_PRIOR_MODE,
)
from CliPP2.api import process_tumor


def _enumerated_loss(data, phi, eps):
    spec = data.path_likelihood
    probability = np.clip(
        data.scaling[..., None] * spec.copies * phi[..., None], eps, 1.0 - eps,
    )
    joint = (
        spec.log_prior + data.alt_counts[..., None] * np.log(probability)
        + (data.total_counts - data.alt_counts)[..., None] * np.log1p(-probability)
    )
    return -logsumexp(joint, axis=-1)


@pytest.fixture(scope="module")
def smoke_fit(tmp_path_factory):
    path = tmp_path_factory.mktemp("integer_smoke") / "smoke.tsv"
    rows = [
        dict(
            mutation_id=f"m{index}", sample_id="R1", alt_count=alt,
            ref_count=60-alt, count_observed=1, purity=0.8, normal_cn=2,
            segment_id=f"s{index}", cn_state_id="clonal", cn_state_fraction=1,
            allele_a_cn=major, allele_b_cn=minor,
        )
        for index, (major, minor, alt) in enumerate([(3, 1, 18), (6, 2, 12), (3, 1, 30)])
    ]
    write_tumor_txt(path, pd.DataFrame(rows))
    data = load_tumor_txt(path)
    graph = build_complete_uniform_graph(data.num_mutations)
    config = resolve_fit_config(
        lambda_value=0.1, computation_profile="balanced", device="cpu",
        dtype="float64", graph=graph, outer_max_iter=20, inner_max_iter=80,
        certificate_max_iter=100, certificate_refinement_rounds=1,
    )
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        before = tumor_data_fingerprint(data)
        fit = fit_fixed_objective(data, config)
    finally:
        torch.set_num_threads(original_threads)
    return data, graph, config, fit, before


def test_actual_public_fit_preserves_marginal_objective_graph_and_gate(smoke_fit):
    data, graph, config, fit, before = smoke_fit
    assert np.isfinite(fit.phi).all()
    assert np.all(fit.phi >= config.eps)
    assert np.all(fit.phi <= data.phi_upper)
    assert tumor_data_fingerprint(data) == before
    assert fit.provenance.original_graph_hash == graph.fingerprint
    assert fit.provenance.lambda_value == config.lambda_value
    model = compile_observed_model(data, eps=config.eps)
    loss = _enumerated_loss(data, fit.phi, config.eps)
    np.testing.assert_allclose(
        observed_terms_numpy(model, fit.phi, eps=config.eps).loss, loss,
        rtol=1e-12, atol=1e-12,
    )
    penalty = config.lambda_value * np.sum(
        graph.edge_w * np.linalg.norm(fit.phi[graph.edge_u] - fit.phi[graph.edge_v], axis=1)
    )
    np.testing.assert_allclose(fit.objective.total, loss.sum() + penalty, rtol=1e-10)
    assert config.solver.tolerance == 8e-4
    assert CertificateConfig.admission_tolerance(config.solver.tolerance) == 0.004
    assert fit.certificate.tolerance == 0.004
    assert fit.certificate.schema_version == 2
    assert fit.certificate.audit_dtype == "float64"
    assert fit.certificate.residual_method == "componentwise_box_cone_backward_error_v1"
    if fit.certificate.certified:
        assert fit.certificate.components.residual <= 0.004
    print(
        "integer CPU smoke:", "objective=", fit.objective.total,
        "certified=", fit.certificate.certified,
        "residual=", fit.certificate.components.residual,
        "graph=", fit.provenance.original_graph_hash,
    )


def test_actual_fixed_partition_refit_marginal_loss_and_bic_reconstruct(smoke_fit):
    data, _graph, config, fit, _before = smoke_fit
    raw_phi = fit.phi.copy()
    labels = np.array([0, 0, 1], dtype=np.int64)
    refit = partition_constrained_observed_refit(
        data, labels, eps=config.eps, tol=1e-7, max_iter=128,
        scalar_mode="grid_local", scalar_grid_points=64, scalar_local_steps=3,
    )
    assert refit.finite_candidate_found
    np.testing.assert_array_equal(refit.labels, labels)
    np.testing.assert_array_equal(refit.phi[0], refit.phi[1])
    np.testing.assert_allclose(
        refit.loglik, -_enumerated_loss(data, refit.phi, config.eps).sum(), rtol=1e-12,
    )
    score = fixed_partition_bic(
        data=data, labels=labels, num_clusters=2, loglik=refit.loglik,
        partition_signature="smoke:001",
    )
    assert score.n_eff == 3
    assert score.degrees_of_freedom == 2
    np.testing.assert_allclose(score.value, -2 * refit.loglik + 2 * np.log(3))
    native_score = fixed_partition_dirichlet_score(
        data=data, labels=labels, num_clusters=2, loglik=refit.loglik,
        partition_signature="smoke:001", alpha=config.selection.dirichlet_alpha,
        code_weight=config.selection.dirichlet_code_weight,
    )
    alpha = config.selection.dirichlet_alpha
    log_assignment = (
        gammaln(2 * alpha) - gammaln(3 + 2 * alpha)
        + np.sum(gammaln(np.array([2, 1]) + alpha)) - 2 * gammaln(alpha)
        + gammaln(3)  # Two exchangeable blocks contribute their 2! labelings.
    )
    assert config.selection.score == native_score.name
    np.testing.assert_allclose(
        native_score.value, score.value - 2 * config.selection.dirichlet_code_weight * log_assignment,
    )
    posterior = infer_integer_multiplicity_posterior_numpy(data, refit.phi, eps=config.eps)
    assert np.all(posterior.multiplicity_call >= 1)
    assert np.all(posterior.multiplicity_call <= data.major_cn)
    np.testing.assert_array_equal(posterior.candidate_count, [[3], [6], [3]])
    np.testing.assert_array_equal(fit.phi, raw_phi)
    print(
        "integer refit smoke:", "loglik=", refit.loglik, "BIC=", score.value,
        "Dirichlet score=", native_score.value,
    )


def test_full_pipeline_identical_units_certifies_and_writes_filtered_outputs(tmp_path):
    input_path = tmp_path / "pipeline.tsv"
    rows = [
        dict(
            mutation_id=f"m{index}", sample_id="R1", alt_count=18, ref_count=42,
            count_observed=1, purity=0.8, normal_cn=2, segment_id=f"s{index}",
            cn_state_id="clonal", cn_state_fraction=1, allele_a_cn=major, allele_b_cn=1,
        )
        for index, major in enumerate([3, 3, 3, 7])
    ]
    write_tumor_txt(input_path, pd.DataFrame(rows))
    original_input = input_path.read_bytes()
    config = resolve_fit_config(
        computation_profile="balanced", device="cpu", dtype="float64",
        outer_max_iter=20, inner_max_iter=80, certificate_max_iter=100,
        certificate_refinement_rounds=1,
    )
    outdir = tmp_path / "outputs"
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        summary = process_tumor(input_path, outdir, fit_config=config)
    finally:
        torch.set_num_threads(original_threads)
    assert summary["summary_schema_version"] == 4
    assert summary["raw_reference_objective_certified"]
    assert summary["selected_n_clusters"] == 1
    assert summary["selected_full_kkt_tolerance"] == 0.004
    assert summary["input_mutation_count"] == 4
    assert summary["retained_mutation_count"] == 3
    assert summary["excluded_mutation_count"] == 1
    assert summary["multiplicity_model_id"] == CLONAL_INTEGER_MODEL_ID
    assert summary["multiplicity_candidate_generator_version"] == CLONAL_INTEGER_GENERATOR_VERSION
    assert summary["multiplicity_prior_mode"] == CLONAL_INTEGER_PRIOR_MODE
    assert summary["scalar_pilot_coordinate_count"] == 3
    assert 0 <= summary["scalar_pilot_certified_coordinate_count"] <= 3
    assert input_path.read_bytes() == original_input
    expected = {
        "pipeline_mutation_clusters.tsv", "pipeline_cluster_centers.tsv",
        "pipeline_mutation_region_multiplicity.tsv", "pipeline_excluded_mutations.tsv",
        "pipeline_run_manifest.json",
    }
    assert {path.name for path in outdir.iterdir()} == expected
    for suffix in ("mutation_clusters", "mutation_region_multiplicity"):
        table = pd.read_csv(outdir / f"pipeline_{suffix}.tsv", sep="\t")
        assert set(table.mutation_id) == {"m0", "m1", "m2"}
    excluded = pd.read_csv(outdir / "pipeline_excluded_mutations.tsv", sep="\t")
    assert excluded.mutation_id.tolist() == ["m3"]
    print(
        "integer full-pipeline CPU smoke:", "K=", summary["selected_n_clusters"],
        "raw_certified=", summary["raw_reference_objective_certified"],
        "pilot_certified=", summary["scalar_pilot_certified_coordinate_count"],
        "score=", summary["selection_score"],
    )
