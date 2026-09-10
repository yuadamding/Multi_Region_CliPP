"""Single source of truth for model-selection score primitives.

This is a leaf module: it depends on immutable configuration and input data so that
both the ``core.fusion`` layer (partition refits / candidate BIC) and the
``model_selection`` / ``runners`` layers can import it *downward*. Keeping the
BIC arithmetic in one place avoids the correctness-drift risk of re-deriving
``-2*loglik + df*log(n)`` (and the observed-mutation_region count) in several modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, lgamma
from typing import Literal

import numpy as np

from ..io.data import TumorData
from ..config import DIRICHLET_ALPHA, DIRICHLET_CODE_WEIGHT


@dataclass(frozen=True, slots=True)
class SelectionScore:
    """Immutable value of the sole fixed-partition selection score."""

    name: Literal["fixed_partition_dirichlet_score"]
    value: float
    loglik: float
    penalty: float
    degrees_of_freedom: int
    n_eff: int
    partition_signature: str
    numerical_uncertainty: float = 0.0
    assignment_log_evidence: float = 0.0
    assignment_code_weight: float = 0.0
    assignment_dirichlet_alpha: float = DIRICHLET_ALPHA
    assignment_arithmetic_uncertainty: float = 0.0

    @property
    def assignment_penalty(self) -> float:
        return float(-2.0 * self.assignment_code_weight * self.assignment_log_evidence)

def effective_bic_mutation_region_count(data: TumorData) -> int:
    """Count positive-depth observed mutation-regions, floored at one for BIC."""
    mask = np.asarray(data.total_counts, dtype=np.float64) > 0.0
    if data.count_observed is not None:
        mask = mask & np.asarray(data.count_observed, dtype=bool)
    return max(int(np.sum(mask)), 1)


def _bic_terms(
    loglik: float, degrees_of_freedom: int, n_eff: int,
) -> tuple[float, float, float]:
    """BIC value, center penalty and arithmetic uncertainty, in that order."""

    penalty = float(degrees_of_freedom * np.log(max(int(n_eff), 1)))
    value = float(-2.0 * float(loglik) + penalty)
    arithmetic_uncertainty = 16.0 * np.finfo(np.float64).eps * (1.0 + abs(value))
    return value, penalty, float(arithmetic_uncertainty)


def cluster_sizes_from_labels(labels: np.ndarray) -> np.ndarray:
    """Return occupied-cluster sizes, invariant to the numeric label names."""
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("Partition labels must be a one-dimensional array.")
    if values.size == 0:
        raise ValueError("A partition must contain at least one mutation label.")
    if not np.issubdtype(values.dtype, np.integer):
        numeric = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.round(numeric)):
            raise ValueError("Partition labels must be finite integers.")
        values = numeric.astype(np.int64)
    _, counts = np.unique(values, return_counts=True)
    return counts.astype(np.int64, copy=False)


def _validated_cluster_sizes(cluster_sizes: np.ndarray) -> np.ndarray:
    raw = np.asarray(cluster_sizes)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("cluster_sizes must be a non-empty one-dimensional array.")
    try:
        values = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "cluster_sizes must contain finite positive integers."
        ) from exc
    if (
        not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
        or not np.all(values == np.round(values))
    ):
        raise ValueError("cluster_sizes must contain finite positive integers.")
    return values.astype(np.int64)


def _dirichlet_exact_partition_log_mass_and_uncertainty(
    cluster_sizes: np.ndarray,
    *,
    alpha: float,
) -> tuple[float, float]:
    """Integrated exact-partition log mass and a conservative arithmetic bound.

    A symmetric Dirichlet prior integrates the occupied-block proportions,
    summing over all K! equivalent block labelings. For alpha=1 the mass is
    K! (K-1)! prod_k n_k! / (n+K-1)!, not a classification-entropy term.
    """

    sizes = _validated_cluster_sizes(cluster_sizes)
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("Dirichlet alpha must be positive and finite.")
    num_clusters = int(sizes.size)
    num_mutations = int(np.sum(sizes))
    terms = [
        lgamma(float(num_clusters) * alpha),
        -lgamma(float(num_mutations) + float(num_clusters) * alpha),
    ]
    for size in sizes:
        terms.extend((lgamma(float(size) + alpha), -lgamma(alpha)))
    # Every block is exchangeable in the unanchored model, so integrate over
    # all K! equivalent component labelings.
    terms.append(lgamma(float(num_clusters) + 1.0))
    value = float(fsum(terms))
    magnitude = float(fsum(abs(term) for term in terms))
    arithmetic_uncertainty = float(
        32.0 * np.finfo(np.float64).eps * float(len(terms) + 1) * (1.0 + magnitude)
    )
    return value, arithmetic_uncertainty


def fixed_partition_dirichlet_score(
    *,
    loglik: float,
    num_clusters: int,
    data: TumorData,
    partition_signature: str,
    labels: np.ndarray,
    loglik_uncertainty: float = 0.0,
    alpha: float = DIRICHLET_ALPHA,
    code_weight: float = DIRICHLET_CODE_WEIGHT,
) -> SelectionScore:
    """Return BIC plus a Dirichlet-integrated exact-partition deviance.

    The BIC contribution retains nominal K*S center degrees of freedom and
    the count of observed positive-depth mutation-regions. The added term is
    invariant to arbitrary names of exchangeable blocks and is scaled by ``code_weight`` (0.7 by
    default). All K blocks are exchangeable. This is
    deliberately named a Dirichlet exact-partition score rather than
    posterior-entropy ICL.
    """

    labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    sizes = cluster_sizes_from_labels(labels_array)
    if int(sizes.size) != int(num_clusters):
        raise ValueError(
            "Dirichlet-score cluster count does not match the partition labels."
        )
    if int(np.sum(sizes)) != int(data.num_mutations):
        raise ValueError("Dirichlet-score labels must contain one value per mutation.")
    alpha = float(alpha)
    weight = float(code_weight)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("code_weight must be nonnegative and finite.")
    degrees_of_freedom = int(num_clusters) * int(data.num_regions)
    n_eff = effective_bic_mutation_region_count(data)
    bic, center_penalty, bic_arithmetic_uncertainty = _bic_terms(
        loglik, degrees_of_freedom, n_eff,
    )
    bic_uncertainty = float(
        2.0 * max(float(loglik_uncertainty), 0.0) + bic_arithmetic_uncertainty
    )
    log_evidence, log_evidence_uncertainty = (
        _dirichlet_exact_partition_log_mass_and_uncertainty(
            sizes,
            alpha=alpha,
        )
    )
    assignment_penalty = float(-2.0 * weight * log_evidence)
    assignment_arithmetic_uncertainty = float(
        2.0 * abs(weight) * log_evidence_uncertainty
    )
    score_value = float(bic + assignment_penalty)
    addition_uncertainty = float(
        16.0
        * np.finfo(np.float64).eps
        * (1.0 + abs(bic) + abs(assignment_penalty))
    )
    return SelectionScore(
        name="fixed_partition_dirichlet_score",
        value=score_value,
        loglik=float(loglik),
        penalty=float(center_penalty + assignment_penalty),
        degrees_of_freedom=degrees_of_freedom,
        n_eff=n_eff,
        partition_signature=str(partition_signature),
        numerical_uncertainty=float(
            bic_uncertainty
            + assignment_arithmetic_uncertainty
            + addition_uncertainty
        ),
        assignment_log_evidence=float(log_evidence),
        assignment_code_weight=weight,
        assignment_dirichlet_alpha=alpha,
        assignment_arithmetic_uncertainty=assignment_arithmetic_uncertainty,
    )


__all__ = [
    "cluster_sizes_from_labels",
    "effective_bic_mutation_region_count",
    "fixed_partition_dirichlet_score",
    "SelectionScore",
]
