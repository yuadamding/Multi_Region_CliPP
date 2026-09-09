from __future__ import annotations

from pathlib import Path
from time import perf_counter

from ..config import FitConfig, resolve_fit_config
from ..io.tumor_txt import NoEligibleSNVsError, load_tumor_txt
from ..model_selection.search import select_model
from ..model_selection.candidates import validate_candidate_identity
from ..model_selection.types import SearchCandidate
from .outputs import write_cn_filter_output
from .serialization import (
    AnalysisSerialization,
    analysis_summary,
    write_analysis_outputs,
)


def _preserve_input_file(tumor_file: Path, outdir: Path, tumor_id: str) -> None:
    """Reject output aliases of the original input before writing any table."""
    for suffix in (
        "mutation_clusters.tsv", "cluster_centers.tsv",
        "mutation_region_multiplicity.tsv", "excluded_mutations.tsv",
    ):
        destination = outdir / f"{tumor_id}_{suffix}"
        if destination.resolve() == tumor_file.resolve() or (
            destination.exists() and destination.samefile(tumor_file)
        ):
            raise ValueError(
                f"Output would overwrite original input: {destination}. "
                "Choose a separate output directory."
            )


def process_tumor_bundle(
    tumor_file: str | Path,
    outdir: str | Path,
    fit_config: FitConfig | None = None,
    use_warm_starts: bool = True,
    write_outputs: bool = True,
    unsupported_policy: str = "error",
    dosage_prior_penalty: float | None = None,
) -> tuple[dict[str, object], tuple[SearchCandidate, ...]]:
    """Fit one canonical tumor TSV file with the default workflow."""

    start_time = perf_counter()
    tumor_file = Path(tumor_file)
    outdir = Path(outdir)
    if not tumor_file.is_file():
        raise FileNotFoundError(f"Tumor input must be a file: {tumor_file}")
    if fit_config is None:
        fit_config = resolve_fit_config()
    fit_config.validate_integer_workflow()
    try:
        data = load_tumor_txt(
            tumor_file,
            eps=float(fit_config.eps),
            unsupported_policy=unsupported_policy,
            dosage_prior_penalty=dosage_prior_penalty,
        )
    except NoEligibleSNVsError as error:
        if write_outputs:
            _preserve_input_file(tumor_file, outdir, error.tumor_id)
            write_cn_filter_output(
                outdir=outdir, tumor_id=error.tumor_id,
                report=error.cn_filter_report,
            )
        raise
    # Preserve the eligibility audit even when numerical fitting later fails.
    if write_outputs:
        _preserve_input_file(tumor_file, outdir, data.tumor_id)
        write_cn_filter_output(
            outdir=outdir, tumor_id=data.tumor_id, report=data.cn_filter_report,
        )
    selection_result = select_model(
        data=data,
        fit_config=fit_config,
        use_warm_starts=use_warm_starts,
    )
    analysis = AnalysisSerialization(
        data=data,
        input_file=Path(tumor_file),
        fit_config=fit_config,
        selection_result=selection_result,
    )
    validate_candidate_identity(analysis.selected_candidate)
    validate_candidate_identity(analysis.raw_reference)
    summary = analysis_summary(
        analysis,
        elapsed_seconds=float(perf_counter() - start_time),
    )

    if write_outputs:
        write_analysis_outputs(
            analysis,
            outdir=outdir,
        )
    return summary, selection_result.search


def process_tumor(
    tumor_file: str | Path,
    outdir: str | Path,
    fit_config: FitConfig | None = None,
    use_warm_starts: bool = True,
    write_outputs: bool = True,
    unsupported_policy: str = "error",
    dosage_prior_penalty: float | None = None,
) -> dict[str, object]:
    """Fit one tumor TSV file."""

    summary, _ = process_tumor_bundle(
        tumor_file=tumor_file,
        outdir=outdir,
        fit_config=fit_config,
        use_warm_starts=use_warm_starts,
        write_outputs=write_outputs,
        unsupported_policy=unsupported_policy,
        dosage_prior_penalty=dosage_prior_penalty,
    )
    return summary
