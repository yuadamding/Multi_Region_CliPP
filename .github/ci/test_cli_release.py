"""End-to-end release contract for the compact CLI and four-file output."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch


SCHEMA_COLUMNS = (
    "mutation_id",
    "sample_id",
    "alt_count",
    "ref_count",
    "count_observed",
    "purity",
    "normal_cn",
    "segment_id",
    "cn_state_id",
    "cn_state_fraction",
    "allele_a_cn",
    "allele_b_cn",
)


def _write_smoke_input(path: Path, *, copy_state: tuple[int, int] = (1, 1)) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEMA_COLUMNS, delimiter="\t")
        writer.writeheader()
        for index, alt_count in enumerate((10, 11, 30, 31), start=1):
            writer.writerow(
                {
                    "mutation_id": f"m{index}",
                    "sample_id": "r1",
                    "alt_count": alt_count,
                    "ref_count": 100 - alt_count,
                    "count_observed": 1,
                    "purity": 0.6,
                    "normal_cn": 2,
                    "segment_id": f"s{index}",
                    "cn_state_id": "state1",
                    "cn_state_fraction": 1.0,
                    "allele_a_cn": copy_state[0],
                    "allele_b_cn": copy_state[1],
                }
            )


def _run_fit(
    input_file: Path,
    outdir: Path,
    config_file: Path,
    *options: str,
    failure_policy: str = "error",
    profile: str = "fast",
    device: str = "cpu",
) -> dict[str, object]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "CliPP2",
            "fit",
            "--input-file",
            str(input_file),
            "--outdir",
            str(outdir),
            "--profile",
            profile,
            "--device",
            device,
            "--failure-policy",
            failure_policy,
            "--config",
            str(config_file),
            *options,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return json.loads((outdir / "smoke_analysis.json").read_text())


def _validate_panel_output(
    outdir: Path,
    receipt: Path,
) -> dict[str, object]:
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("validate_outputs_v4.py")),
            "--outdir",
            str(outdir),
            "--tumor-id",
            "smoke",
            "--expected-mutations",
            "4",
            "--output",
            str(receipt),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(receipt.read_text(encoding="utf-8"))


def _assert_precision_provenance(summary: dict[str, object], requested_dtype: str) -> None:
    # Final raw state may be promoted; it is not the requested working dtype.
    assert summary["selected_working_dtype"] == requested_dtype
    assert summary["selected_certificate_audit_dtype"] == "float64"
    polished = summary["selected_precision_polish_applied"]
    assert isinstance(polished, bool)
    assert summary["dtype"] == ("float64" if polished else requested_dtype)


@pytest.mark.parametrize("requested,final,polished", [
    ("float32", "float32", False), ("float32", "float64", True),
    ("float64", "float64", False),
])
def test_precision_provenance_accepts_recorded_promotion(requested, final, polished):
    _assert_precision_provenance(dict(
        selected_working_dtype=requested, selected_certificate_audit_dtype="float64",
        selected_precision_polish_applied=polished, dtype=final,
    ), requested)


@pytest.mark.parametrize("changes", [
    {"dtype": "float64"}, {"selected_precision_polish_applied": True},
    {"selected_working_dtype": "float64"}, {"selected_certificate_audit_dtype": "float32"},
    {"selected_precision_polish_applied": "false"},
])
def test_precision_provenance_rejects_unexplained_dtype_changes(changes):
    summary = dict(selected_working_dtype="float32", selected_certificate_audit_dtype="float64",
                   selected_precision_polish_applied=False, dtype="float32")
    with pytest.raises(AssertionError):
        _assert_precision_provenance(summary | changes, "float32")


@pytest.mark.parametrize("copy_state", [(1, 1), (3, 2), (4, 2)])
@pytest.mark.parametrize("device,dtype,profile", [
    ("cpu", "float64", "fast"), ("cuda", "float32", "balanced"),
])
def test_cli_checkpoint_resume_and_four_file_output(tmp_path: Path, copy_state, device, dtype, profile) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    input_file = tmp_path / "smoke.tsv"
    config_file = tmp_path / "config.json"
    checkpoint = tmp_path / "checkpoint"
    _write_smoke_input(input_file, copy_state=copy_state)
    config_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fit": {
                    "dtype": dtype,
                    "max_direct_partition_candidates": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    first_dir = tmp_path / "first"
    resumed_dir = tmp_path / "resumed"
    first = _run_fit(
        input_file, first_dir, config_file, "--checkpoint", str(checkpoint), device=device, profile=profile,
    )
    resumed = _run_fit(
        input_file, resumed_dir, config_file, "--resume", str(checkpoint), device=device, profile=profile,
    )

    assert first["summary_schema_version"] == 13
    assert first["output_schema_version"] == 4
    assert first["emission_model_id"] == "clipp2_single_switch_path_mixture_v2"
    assert first["emission_model_version"] == "2"
    assert first["emission_valid_path_count_distribution"] == {str(copy_state[0]): 4}
    assert first["emission_scalar_well_dispatch"] == "generic_paths"
    assert first["emission_binary_exclusion_reason"] == "padded_path_count_not_two"
    assert first["emission_has_internal_switches"] is False
    assert first["dosage_reporting_policy_id"] == "conditional_positive_depth_excess_mass_v2"
    assert first["dosage_mass_tolerance"] == first["dosage_ccf_floor_tolerance"] == 1e-8
    assert first["dosage_conditioning"] == "selected_partition_refit_ccf"
    assert first["analysis_tier"] == "joint_certified"
    assert first["primary_estimator_available"] is True
    assert first["raw_reference_objective_certified"] is True
    assert first["selected_certificate_audit_dtype"] == "float64"
    assert first["selected_full_kkt_residual_method"] == "componentwise_box_cone_backward_error_v1"
    assert first["selected_full_kkt_tolerance"] == 5 * first["selected_raw_solver_primal_tol"]
    assert first["selection_score_name"] == "fixed_partition_bic"
    assert first["selection_policy_id"] == "hybrid-ward-cem-bic-v1"
    assert first["computation_profile"] == profile
    assert first["device"] == ("cuda:0" if device == "cuda" else "cpu")
    _assert_precision_provenance(first, dtype)
    _assert_precision_provenance(resumed, dtype)
    assert resumed["resumed_from_checkpoint"] is True
    for field in (
        "selected_partition_signature",
        "selected_n_clusters",
        "selection_score",
        "search_work_edge_pass_equivalents",
        "selection_pool_stop_reason",
        "observed_model_hash",
        "observed_likelihood_hash",
        "selected_raw_reference_objective_spec_hash",
        "selected_raw_reference_original_graph_hash",
        "dtype",
        "selected_working_dtype",
        "selected_certificate_audit_dtype",
        "selected_precision_polish_applied",
        "selected_precision_polish_max_abs_phi_delta",
    ):
        assert resumed[field] == first[field]

    assert (checkpoint / "manifest.json").is_file()
    assert any((checkpoint / "arrays").iterdir())
    with (first_dir / "smoke_mutations.tsv").open(newline="") as handle:
        mutations = list(csv.DictReader(handle, delimiter="\t"))
    for row in mutations:
        single = float(row["single_copy_probability"])
        amplified = float(row["amplified_mutant_copy_probability"])
        assert single + amplified == pytest.approx(1, abs=1e-12)
        assert row["dosage_status"] == "conditional_at_refit_ccf"
        if copy_state == (1, 1):
            assert single == 1.0
    assert (first_dir / "smoke_mutations.tsv").read_bytes() == (
        resumed_dir / "smoke_mutations.tsv"
    ).read_bytes()
    assert {path.name for path in first_dir.iterdir()} == {
        "smoke_analysis.json",
        "smoke_attempts.tsv",
        "smoke_clusters.tsv",
        "smoke_mutations.tsv",
    }


def test_resource_stop_remains_explicitly_unresolved(tmp_path: Path) -> None:
    input_file = tmp_path / "smoke.tsv"
    config_file = tmp_path / "config.json"
    _write_smoke_input(input_file)
    config_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fit": {
                    "dtype": "float64",
                    "max_tumor_edge_pass_equivalents": 1,
                    "max_direct_partition_candidates": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    analysis = _run_fit(
        input_file,
        tmp_path / "resource-stop",
        config_file,
        failure_policy="save-diagnostics",
    )
    assert analysis["search_stop_reason"] == "tumor_work_budget_reached"
    assert analysis["selection_optimum_resolved"] is False
    assert analysis["global_hybrid_optimum_certified"] is False


def test_panel_validator_separates_primary_and_conditional_status(tmp_path: Path) -> None:
    input_file = tmp_path / "smoke.tsv"
    config_file = tmp_path / "config.json"
    outdir = tmp_path / "result"
    _write_smoke_input(input_file)
    config_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fit": {
                    "dtype": "float64",
                    "max_direct_partition_candidates": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    _run_fit(input_file, outdir, config_file, profile="balanced")

    primary = _validate_panel_output(outdir, tmp_path / "primary.json")
    assert primary["execution_status"] == "completed"
    assert primary["artifact_status"] == "valid"
    assert primary["scientific_status"] == "primary_estimator_available"

    analysis_path = outdir / "smoke_analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis.update(
        {
            "analysis_tier": "conditional_partition_refit",
            "primary_estimator_available": False,
            "raw_reference_objective_certified": False,
            "selected_raw_reference_objective_spec_hash": "",
            "selected_raw_reference_original_graph_hash": "",
            "failure_reason": "NoCertifiedRawReferenceError: CI fixture",
        }
    )
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")

    conditional = _validate_panel_output(outdir, tmp_path / "conditional.json")
    assert conditional["execution_status"] == "completed"
    assert conditional["artifact_status"] == "valid"
    assert conditional["scientific_status"] == "no_certified_raw_reference"


def test_public_clonal_loader_keeps_single_copy_categorical_likelihood(tmp_path: Path) -> None:
    import numpy as np

    from CliPP2.core.objective import compile_observed_model
    from CliPP2.io.tumor_txt import load_tumor_txt, write_tumor_txt

    input_file = tmp_path / "clonal.tsv"
    _write_smoke_input(input_file)
    with input_file.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        row.update(allele_a_cn=3, allele_b_cn=2)
    write_tumor_txt(input_file, rows)
    data = load_tumor_txt(input_file)
    paths = data.emission_paths
    np.testing.assert_array_equal(paths.first_copy[0, 0], [1, 2, 3])
    weights = np.asarray([2, 2 * np.exp(-3), np.exp(-6)])
    np.testing.assert_allclose(np.exp(paths.log_prior[0, 0]), weights / weights.sum())
    assert not paths.major_prior_weighted
    model = compile_observed_model(data, major_prior=0.5, eps=1e-6)
    assert model.requires_generic_path_solver
