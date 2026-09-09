"""Public fixed-objective fit boundary."""

from __future__ import annotations

import numpy as np
import torch

from ..config import FitConfig
from ..io.data import TumorData
from ..io.multiplicity import CLONAL_INTEGER_MODEL_ID
from ..io.tumor_txt import CN_FILTER_POLICY_ID
from .fusion.solver import fit_observed_data_pairwise_fusion
from .fusion.types import RawFit, SolverContext, SolverState


def validate_public_tumor_data(data: TumorData, config: FitConfig) -> None:
    """Require original-CN validation at the public fitting boundary.

    Dominant CN arrays alone cannot establish eligibility. Explicit legacy
    objects remain usable by the internal numerical compatibility routines.
    """
    config.validate_integer_workflow()
    spec = data.path_likelihood
    report = data.cn_filter_report
    if (
        spec is None
        or spec.model_id != CLONAL_INTEGER_MODEL_ID
        or report is None
        or report.policy_id != CN_FILTER_POLICY_ID
    ):
        raise ValueError(
            "The public fit requires clonal integer TumorData from "
            "load_tumor_txt, including its original-CN filtering report; "
            "legacy or unvalidated TumorData is not supported."
        )
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


def fit_fixed_objective(
    data: TumorData,
    config: FitConfig,
    phi_start: np.ndarray | torch.Tensor | None = None,
    exact_pilot: np.ndarray | torch.Tensor | None = None,
    pooled_start: np.ndarray | torch.Tensor | None = None,
    scalar_well_starts: list[np.ndarray | torch.Tensor] | None = None,
    start_mode: str = "full",
    append_default_nonconvex_starts: bool | None = None,
    runtime=None,
    torch_data=None,
    solver_context: SolverContext | None = None,
    solver_state: SolverState | None = None,
) -> RawFit:
    """Fit the immutable observed objective described by ``config``."""

    validate_public_tumor_data(data, config)
    solver = config.solver
    resources = solver.resources
    certificate = solver.certificate
    graph = config.graph
    return fit_observed_data_pairwise_fusion(
        data=data,
        lambda_value=float(config.lambda_value),
        major_prior=float(config.major_prior),
        eps=float(config.eps),
        outer_max_iter=max(int(solver.outer_max_iter), 1),
        inner_max_iter=max(int(solver.inner_max_iter), 16),
        tol=float(solver.tolerance),
        certification_tol=(
            float(solver.tolerance)
            if solver.certification_tolerance is None
            else float(solver.certification_tolerance)
        ),
        use_backward_error_progress=bool(solver.use_backward_error_progress),
        phi_start=phi_start,
        graph=graph.graph,
        adaptive_weight_gamma=float(graph.adaptive_weight_gamma),
        adaptive_weight_floor=float(graph.adaptive_weight_floor),
        adaptive_weight_baseline=float(graph.adaptive_weight_baseline),
        exact_pilot=exact_pilot,
        pooled_start=pooled_start,
        scalar_well_starts=scalar_well_starts,
        start_mode=str(start_mode),
        append_default_nonconvex_starts=append_default_nonconvex_starts,
        device=str(config.runtime.device),
        dtype=str(config.runtime.dtype),
        objective_shape=str(solver.objective_shape),
        workset_max_bytes=int(resources.workset_max_bytes),
        compressed_cache_max_bytes=int(resources.compressed_cache_max_bytes),
        dense_fallback_policy=str(config.runtime.fallback),
        workset_add_batch=int(resources.workset_add_batch),
        workset_max_expansions=int(resources.workset_max_expansions),
        certificate_max_iter=int(certificate.max_iter),
        certificate_refinement_rounds=int(certificate.refinement_rounds),
        certificate_column_tol_scale=float(certificate.column_tolerance_scale),
        runtime=runtime,
        torch_data=torch_data,
        solver_context=solver_context,
        solver_state=solver_state,
        verbose=bool(config.runtime.verbose),
    )


FitResult = RawFit

__all__ = ["FitConfig", "FitResult", "RawFit", "fit_fixed_objective"]
