from __future__ import annotations

from ..core.fusion.partition_starts import (
    PartitionCandidate,
    generate_likelihood_partition_starts,
    hessian_weighted_ward_label_sets_torch,
    observed_curvature_at_pilot_torch,
)
from ..config import (
    DIRICHLET_ALPHA,
    DIRICHLET_CODE_WEIGHT,
    FitConfig,
    LIKELIHOOD_PARTITION_K_MAX,
    PARTITION_CEM_MAX_ITER,
    PARTITION_GENERATION_REFIT_MAX_ITER,
    PARTITION_K_ANCHORS,
    PARTITION_MAX_CANDIDATES_PER_K,
)
from ..io.data import TumorData


def generate_partition_initializer_pool(
    *,
    data: TumorData,
    pilot_phi,
    fit_options: FitConfig,
    runtime,
    torch_data,
    rescore_candidates,
    curvature=None,
    declared_k_grid: tuple[int, ...] | None = None,
) -> tuple[PartitionCandidate, ...]:
    """Generate the deterministic Ward/CEM pool used to choose one guide.

    The chosen guide always supplies the initial solver state and lambda scale.
    Every profile builds adaptive weights from the zero-penalty likelihood
    pilot. Retained guide and final-Phi labels pass through the independent
    fixed-label Dirichlet score gate after the raw path terminates.
    """

    if declared_k_grid is None:
        sparse_k_grid = [
            int(value)
            for value in PARTITION_K_ANCHORS
            if 1 <= int(value) <= int(data.num_mutations)
        ]
        k_cap = min(int(LIKELIHOOD_PARTITION_K_MAX), int(data.num_mutations))
        if k_cap > 0 and k_cap not in sparse_k_grid:
            sparse_k_grid.append(k_cap)
        sparse_k_grid = sorted(set(sparse_k_grid))
    else:
        sparse_k_grid = sorted(
            {
                int(value)
                for value in declared_k_grid
                if 1 <= int(value) <= int(data.num_mutations)
            }
        )
    if curvature is None:
        curvature = observed_curvature_at_pilot_torch(
            data,
            pilot_phi,
            eps=float(fit_options.eps),
            torch_data=torch_data,
            device=runtime.device,
            dtype=runtime.dtype,
        )

    def generate(k_grid: list[int]) -> list[PartitionCandidate]:
        label_sets = hessian_weighted_ward_label_sets_torch(
            pilot_phi,
            curvature,
            K_grid=k_grid,
            device=runtime.device,
            dtype=runtime.dtype,
        )
        candidates = generate_likelihood_partition_starts(
            data,
            exact_pilot=pilot_phi,
            eps=float(fit_options.eps),
            K_grid=k_grid,
            max_candidates_per_K=PARTITION_MAX_CANDIDATES_PER_K,
            cem_max_iter=PARTITION_CEM_MAX_ITER,
            refit_max_iter=PARTITION_GENERATION_REFIT_MAX_ITER,
            tol=float(fit_options.solver.tolerance),
            curvature=curvature,
            label_sets=label_sets,
            torch_data=torch_data,
            device=runtime.device,
            dtype=runtime.dtype,
            use_torch=False,
            classification_weight_alpha=DIRICHLET_ALPHA,
            classification_code_weight=DIRICHLET_CODE_WEIGHT,
            allow_component_death=False,
            include_plain_ward=True,
            include_ward_cem=True,
        )
        return rescore_candidates(
            candidates,
            data=data,
        )

    return tuple(generate(sparse_k_grid))


__all__ = ["generate_partition_initializer_pool"]
