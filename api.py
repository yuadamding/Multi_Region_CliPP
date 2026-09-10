"""Validated public input, preparation, fitting, and one-tumor orchestration."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .config import FitConfig, resolve_fit_config
from .io.data import TumorData
from .io.multiplicity import MAX_MAJOR_CN
from .io.tumor_txt import CN_FILTER_POLICY_ID, NoEligibleSNVsError, load_tumor_txt
from .core.fusion.solver import (
    fit_prepared, prepare_torch_problem_with_resource_policy,
)
from .core.fusion.types import RawFit, PreparedProblem
from .model_selection.candidates import validate_candidate_identity
from .model_selection.types import SearchCandidate
from .reporting import (
    RunPublication, AnalysisSerialization, analysis_summary, write_analysis_outputs,
    _file_hash,
)

def _validate_biological_arrays(data: TumorData) -> None:
    """Reject stale derived arrays and malformed programmatic replacements.

    Shape checks precede arithmetic so NumPy broadcasting cannot turn an
    inconsistent biological source into an apparently valid objective.
    """
    shape = (len(data.mutation_ids), len(data.region_ids))
    if 0 in shape or any(
        any(not isinstance(value, str) or not value.strip() for value in ids)
        or len(set(ids)) != len(ids)
        for ids in (data.mutation_ids, data.region_ids)
    ):
        raise ValueError("TumorData mutation/region identifiers must be nonempty and unique.")
    for name in (
        "alt_counts", "total_counts", "purity", "major_cn", "minor_cn",
        "normal_cn", "scaling", "phi_upper", "phi_init",
    ):
        values = np.asarray(getattr(data, name))
        if (
            not isinstance(getattr(data, name), np.ndarray)
            or values.shape != shape
            or values.dtype.kind not in "iuf"
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(f"TumorData.{name} must be a finite numeric array of shape {shape}.")
    for name in ("count_observed",):
        values = getattr(data, name)
        if values is None and name == "count_observed":
            continue
        values = np.asarray(values)
        if values.shape != shape or values.dtype.kind != "b":
            raise ValueError(f"TumorData.{name} must be a Boolean array of shape {shape}.")
    for name in ("alt_counts", "total_counts", "major_cn", "minor_cn"):
        values = getattr(data, name)
        if np.any(values < 0) or np.any(values != np.rint(values)):
            raise ValueError(f"TumorData.{name} must contain nonnegative integers.")
    if np.any(data.alt_counts > data.total_counts):
        raise ValueError("TumorData.alt_counts cannot exceed total_counts.")
    if np.any((data.purity <= 0.0) | (data.purity > 1.0)):
        raise ValueError("TumorData.purity must lie in (0, 1].")
    if not np.all(data.purity == data.purity[:1]):
        raise ValueError("TumorData.purity must be constant within each region.")
    if np.any(data.normal_cn < 0.0):
        raise ValueError("TumorData.normal_cn must be nonnegative.")
    if (
        np.any((data.major_cn < 1) | (data.major_cn > MAX_MAJOR_CN))
        or np.any(data.minor_cn > data.major_cn)
    ):
        raise ValueError("TumorData must satisfy 0 <= minor_cn <= major_cn <= 6 and major_cn >= 1.")
    expected_scaling = data.purity / (
        (1.0 - data.purity) * data.normal_cn
        + data.purity * (data.major_cn + data.minor_cn)
    )
    if not np.allclose(data.scaling, expected_scaling, rtol=1e-12, atol=0.0):
        raise ValueError(
            "TumorData.scaling is inconsistent with purity and copy number; "
            "reload the input after changing biological values."
        )


def validate_public_tumor_data(data: TumorData, config: FitConfig) -> None:
    """Require original-CN validation at the public fitting boundary.

    Retained CN arrays alone cannot establish original-input eligibility.
    """

    report = data.cn_filter_report
    if (
        report is None
        or report.policy_id != CN_FILTER_POLICY_ID
    ):
        raise ValueError(
            "The public fit requires clonal integer TumorData from "
            "load_tumor_txt, including its original-CN filtering report; "
            "legacy or unvalidated TumorData is not supported."
        )
    _validate_biological_arrays(data)
    excluded = set(report.excluded_mutation_ids)
    if (
        report.retained_mutation_count != data.num_mutations
        or report.input_mutation_count != data.num_mutations + len(excluded)
        or len(excluded) != len(report.excluded_mutation_ids)
        or excluded.intersection(data.mutation_ids)
    ):
        raise ValueError("CN filtering report is inconsistent with retained mutations.")
    epsilon = float(config.eps)
    expected_upper = np.clip(np.minimum(
        1.0, (1.0 - epsilon) / np.clip(data.scaling * data.major_cn, epsilon, None)
    ), epsilon, 1.0)
    if (
        not np.allclose(data.phi_upper, expected_upper, rtol=0.0, atol=1e-12)
        or not np.all(np.isfinite(data.phi_init))
        or np.any(data.phi_init < epsilon)
        or np.any(data.phi_init > expected_upper)
    ):
        raise ValueError(
            "Loaded CCF bounds/initialization do not match fit eps; "
            "reload the input with eps=config.eps."
        )


def prepare_problem(data: TumorData, options: FitConfig) -> PreparedProblem:
    """Validate inputs and freeze one likelihood, pilot, graph, and runtime."""
    validate_public_tumor_data(data, options)
    return prepare_torch_problem_with_resource_policy(data, options)


def fit_fixed_objective(
    data: TumorData,
    config: FitConfig,
    *,
    phi_start: np.ndarray | torch.Tensor | None = None,
) -> RawFit:
    """Prepare and fit one fixed lambda; reuse through fit_prepared explicitly."""
    return fit_prepared(
        prepare_problem(data, config), config.lambda_value, config.solver,
        phi_start=phi_start,
    )


FitResult = RawFit

def process_tumor_bundle(
    tumor_file: str | Path,
    outdir: str | Path,
    fit_config: FitConfig | None = None,
    use_warm_starts: bool = True,
    write_outputs: bool = True,
) -> tuple[dict[str, object], tuple[SearchCandidate, ...]]:
    """Fit one canonical tumor TSV file with the default workflow."""

    from .model_selection.search import select_model

    start_time = perf_counter()
    tumor_file = Path(tumor_file)
    outdir = Path(outdir)
    if not tumor_file.is_file():
        raise FileNotFoundError(f"Tumor input must be a file: {tumor_file}")
    if fit_config is None:
        fit_config = resolve_fit_config()
    input_sha256 = _file_hash(tumor_file)
    workflow = {"entrypoint": "process_tumor_bundle", "use_warm_starts": bool(use_warm_starts)}
    try:
        data = load_tumor_txt(
            tumor_file,
            eps=float(fit_config.eps),
        )
    except NoEligibleSNVsError as error:
        if write_outputs:
            publication = RunPublication(outdir, error.tumor_id, input_file=tumor_file,
                                         expected_input_sha256=input_sha256,
                                         fit_config=fit_config, workflow=workflow)
            try:
                publication.write_audit(error.cn_filter_report)
            finally:
                publication.fail(error)
        raise
    # Preserve the eligibility audit even when numerical fitting later fails.
    publication = (
        RunPublication(outdir, data.tumor_id, input_file=tumor_file,
                       expected_input_sha256=input_sha256,
                       fit_config=fit_config, workflow=workflow)
        if write_outputs else None
    )
    try:
        if publication is not None:
            publication.write_audit(data.cn_filter_report)
        selection_result = select_model(
            data=data, fit_config=fit_config, use_warm_starts=use_warm_starts,
        )
        analysis = AnalysisSerialization(
            data=data, input_file=tumor_file, fit_config=fit_config,
            selection_result=selection_result,
        )
        validate_candidate_identity(analysis.selected_candidate)
        validate_candidate_identity(analysis.raw_reference)
        summary = analysis_summary(analysis, elapsed_seconds=float(perf_counter() - start_time))
        if publication is not None:
            write_analysis_outputs(analysis, outdir=outdir, publication=publication)
            summary["run_id"] = publication.record["run_id"]
            summary["run_manifest"] = str(publication.path)
    except BaseException as error:
        if publication is not None:
            publication.fail(error)
        raise
    return summary, selection_result.search


def process_tumor(
    tumor_file: str | Path,
    outdir: str | Path,
    fit_config: FitConfig | None = None,
    use_warm_starts: bool = True,
    write_outputs: bool = True,
) -> dict[str, object]:
    """Fit one tumor TSV file."""

    summary, _ = process_tumor_bundle(
        tumor_file=tumor_file,
        outdir=outdir,
        fit_config=fit_config,
        use_warm_starts=use_warm_starts,
        write_outputs=write_outputs,
    )
    return summary
