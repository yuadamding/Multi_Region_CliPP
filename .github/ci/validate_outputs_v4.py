#!/usr/bin/env python3
"""Fail-closed, analysis-tier-aware validation for one CliPP2 v0.4 result."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path


SUFFIXES = ("analysis.json", "clusters.tsv", "mutations.tsv", "attempts.tsv")
DOSAGE_COLUMNS = (
    "single_copy_probability", "single_copy_prior_probability",
    "amplified_mutant_copy_probability", "amplified_mutant_copy_call",
    "posterior_effective_multiplicity", "map_effective_multiplicity",
    "posterior_mutant_copy_mass", "map_mutant_copy_mass",
    "multiplicity_call", "multiplicity_estimated", "gamma_major", "major_call",
)


def _number(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be finite numeric data") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"{name} must be finite numeric data")
    return number


def _validate_mutations(path: Path, analysis: dict[str, object]) -> int:
    """Validate dosage evidence and blankness without loading inference code."""

    if analysis.get("dosage_reporting_policy_id") != "conditional_positive_depth_excess_mass_v2":
        raise RuntimeError("unsupported dosage reporting policy")
    if analysis.get("dosage_conditioning") != "selected_partition_refit_ccf":
        raise RuntimeError("dosage conditioning is missing")
    lower = _number(analysis.get("dosage_ccf_lower_bound"), "dosage_ccf_lower_bound")
    floor_tol = _number(analysis.get("dosage_ccf_floor_tolerance"), "dosage_ccf_floor_tolerance")
    mass_tol = _number(analysis.get("dosage_mass_tolerance"), "dosage_mass_tolerance")
    positive = analysis.get("dosage_positive_path_family")
    if not isinstance(positive, bool) or lower <= 0 or floor_tol < 0 or mass_tol < 0:
        raise RuntimeError("invalid dosage reporting policy values")
    required = {*DOSAGE_COLUMNS, "dosage_reportable", "dosage_status", "phi",
                "cluster_label", "count_available", "likelihood_supported",
                "likelihood_included", "phi_statistically_identified", "alt_count", "ref_count"}
    count = 0
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not required.issubset(reader.fieldnames or ()):
            raise RuntimeError(f"missing required dosage TSV columns: {sorted(required - set(reader.fieldnames or ())) }")
        for row in reader:
            count += 1
            for flag in ("count_available", "likelihood_supported", "likelihood_included",
                         "phi_statistically_identified", "dosage_reportable"):
                if row[flag] not in {"0", "1"}:
                    raise RuntimeError(f"{flag} must be 0 or 1")
            if row["likelihood_included"] == "1" and (
                row["count_available"] != "1" or row["likelihood_supported"] != "1"
            ):
                raise RuntimeError("likelihood inclusion contradicts count/support flags")
            if not row["cluster_label"]:
                status = "no_selected_partition"
            elif row["likelihood_included"] == "0":
                status = "not_likelihood_included"
            else:
                depth = _number(row["alt_count"], "alt_count") + _number(row["ref_count"], "ref_count")
                phi = _number(row["phi"], "phi")
                if depth <= 0:
                    status = "zero_depth"
                elif row["phi_statistically_identified"] == "0":
                    status = "ccf_not_reportable"
                elif phi <= lower + floor_tol:
                    status = "numerical_ccf_floor"
                else:
                    status = "conditional_at_refit_ccf"
            reportable = status == "conditional_at_refit_ccf"
            if row["dosage_status"] != status or row["dosage_reportable"] != str(int(reportable)):
                raise RuntimeError("dosage status disagrees with observation/CCF evidence")
            if not reportable:
                if any(row[name] != "" for name in DOSAGE_COLUMNS):
                    raise RuntimeError("unreportable dosage fields must be blank")
                continue
            for name in DOSAGE_COLUMNS:
                if row[name] != "":
                    value = _number(row[name], name)
                    if ("probability" in name or name == "gamma_major") and not 0 <= value <= 1:
                        raise RuntimeError(f"{name} must be in [0, 1]")
            single = _number(row["single_copy_probability"], "single_copy_probability")
            _number(row["single_copy_prior_probability"], "single_copy_prior_probability")
            amplified = _number(row["amplified_mutant_copy_probability"], "amplified_mutant_copy_probability")
            call = _number(row["amplified_mutant_copy_call"], "amplified_mutant_copy_call")
            if call != int(amplified >= 0.5):
                raise RuntimeError("amplified call disagrees with probability")
            if positive and not math.isclose(single + amplified, 1.0, rel_tol=0, abs_tol=1e-12):
                raise RuntimeError("positive-path dosage probabilities must sum to one")
    return count


def _count_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise RuntimeError(f"missing TSV header: {path}")
        return sum(1 for _ in reader)


def _require_hex64(analysis: dict[str, object], name: str) -> None:
    value = analysis.get(name)
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError(f"analysis identity is missing for {name}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise RuntimeError(f"analysis identity is not hexadecimal for {name}") from exc


def _require_empty(analysis: dict[str, object], name: str) -> None:
    if analysis.get(name) != "":
        raise RuntimeError(f"conditional analysis fabricates {name}")


def validate_outputs(
    *,
    outdir: Path,
    tumor_id: str,
    expected_mutations: int,
) -> dict[str, object]:
    """Validate four artifacts and return distinct operational/scientific status."""

    expected = {f"{tumor_id}_{suffix}" for suffix in SUFFIXES}
    actual = {
        path.name
        for path in outdir.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if actual != expected:
        raise RuntimeError(f"unexpected v0.4 result inventory: {sorted(actual)}")
    paths = {suffix: outdir / f"{tumor_id}_{suffix}" for suffix in SUFFIXES}
    if any(
        not path.is_file() or path.is_symlink() or path.stat().st_size == 0
        for path in paths.values()
    ):
        raise RuntimeError("a required v0.4 result is missing, linked, or empty")

    with paths["analysis.json"].open(encoding="utf-8") as handle:
        analysis = json.load(handle)
    required_equal = {
        "summary_schema_version": 13,
        "output_schema_version": 4,
        "tumor_id": tumor_id,
        "computation_profile": "balanced",
        "selection_policy_id": "hybrid-ward-cem-bic-v1",
    }
    for key, value in required_equal.items():
        if analysis.get(key) != value:
            raise RuntimeError(
                f"analysis contract mismatch for {key}: {analysis.get(key)!r}"
            )
    for key in (
        "emission_model_id", "emission_model_version",
        "emission_candidate_generator_version", "emission_prior_mode",
    ):
        if not isinstance(analysis.get(key), str) or not analysis[key]:
            raise RuntimeError(f"emission provenance is missing for {key}")
    if set(analysis.get("output_files", [])) != expected:
        raise RuntimeError("analysis output-file authority mismatch")
    if (
        not isinstance(analysis.get("selected_n_clusters"), int)
        or int(analysis["selected_n_clusters"]) < 1
        or not analysis.get("selected_partition_signature")
    ):
        raise RuntimeError("selected partition identity is missing")

    primary = analysis.get("primary_estimator_available")
    if not isinstance(primary, bool):
        raise RuntimeError("primary_estimator_available must be Boolean")
    selected_hashes = (
        "selected_raw_reference_objective_spec_hash",
        "selected_raw_reference_original_graph_hash",
    )
    if primary:
        if analysis.get("analysis_tier") != "joint_certified":
            raise RuntimeError("primary result is not joint_certified")
        if analysis.get("raw_reference_objective_certified") is not True:
            raise RuntimeError("primary result lacks a certified raw reference")
        if analysis.get("selected_refit_numerically_resolved") is not True:
            raise RuntimeError("primary result lacks a resolved partition refit")
        for name in selected_hashes:
            _require_hex64(analysis, name)
        scientific_status = "primary_estimator_available"
    else:
        if analysis.get("analysis_tier") != "conditional_partition_refit":
            raise RuntimeError("non-primary panel result is not a conditional refit")
        for name in selected_hashes:
            _require_empty(analysis, name)
        if int(analysis.get("num_raw_candidates", 0)) > 0:
            _require_hex64(analysis, "attempted_objective_spec_hash")
            _require_hex64(analysis, "attempted_original_graph_hash")
        reason = str(analysis.get("failure_reason", ""))
        if not reason.startswith("NoCertifiedRawReferenceError:"):
            raise RuntimeError("conditional result lacks raw-reference failure provenance")
        scientific_status = "no_certified_raw_reference"

    cluster_rows = _count_rows(paths["clusters.tsv"])
    mutation_rows = _validate_mutations(paths["mutations.tsv"], analysis)
    attempt_rows = _count_rows(paths["attempts.tsv"])
    if cluster_rows < 1 or mutation_rows != int(expected_mutations):
        raise RuntimeError(
            f"unexpected table rows: clusters={cluster_rows}, mutations={mutation_rows}, "
            f"attempts={attempt_rows}"
        )
    if int(analysis.get("num_raw_solver_attempts", 0)) > 0 and attempt_rows < 1:
        raise RuntimeError("raw solver attempts are absent from attempts.tsv")

    return {
        "tumor_id": tumor_id,
        "execution_status": "completed",
        "artifact_status": "valid",
        "scientific_status": scientific_status,
        "analysis_tier": analysis["analysis_tier"],
        "selection_status": analysis.get("selection_status"),
        "selected_n_clusters": analysis.get("selected_n_clusters"),
        "raw_reference_objective_certified": analysis.get(
            "raw_reference_objective_certified"
        ),
        "mutation_rows": mutation_rows,
        "cluster_rows": cluster_rows,
        "attempt_rows": attempt_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--tumor-id", required=True)
    parser.add_argument("--expected-mutations", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = validate_outputs(
        outdir=args.outdir,
        tumor_id=args.tumor_id,
        expected_mutations=args.expected_mutations,
    )
    descriptor = os.open(
        args.output,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o440,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
