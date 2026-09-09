"""Compile the clonal, uniform integer-multiplicity mixture."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import readonly_array

CLONAL_INTEGER_MODEL_ID = "clipp2_clonal_integer_multiplicity_mixture_v1"
CLONAL_INTEGER_GENERATOR_VERSION = "integer_1_to_major_cap6_v1"
CLONAL_INTEGER_PRIOR_MODE = "uniform_distinct_integer_v1"
MAX_MAJOR_CN = 6


@dataclass(frozen=True, slots=True)
class IntegerMultiplicitySpec:
    """One distinct linear integer candidate per allowed copy count."""
    copies: np.ndarray
    log_prior: np.ndarray
    valid: np.ndarray
    model_id: str = CLONAL_INTEGER_MODEL_ID
    model_version: str = "1"
    candidate_generator_version: str = CLONAL_INTEGER_GENERATOR_VERSION
    prior_mode: str = CLONAL_INTEGER_PRIOR_MODE

    def __post_init__(self) -> None:
        copies = readonly_array(self.copies, dtype=np.float64)
        valid = readonly_array(self.valid, dtype=bool)
        prior = readonly_array(self.log_prior, dtype=np.float64)
        if copies.ndim != 3 or 0 in copies.shape or not 1 <= copies.shape[-1] <= MAX_MAJOR_CN:
            raise ValueError("Integer candidate arrays require nonempty (M, S, K), K <= 6.")
        if valid.shape != copies.shape or prior.shape != copies.shape:
            raise ValueError("Integer candidate arrays must have identical shapes.")
        count = np.sum(valid, axis=-1)
        expected_valid = np.arange(1, copies.shape[-1] + 1) <= count[..., None]
        if np.any(count < 1) or not np.array_equal(valid, expected_valid):
            raise ValueError("Integer candidates require the complete ordered 1..major_cn range.")
        expected_copies = np.where(valid, np.arange(1, copies.shape[-1] + 1), 0)
        expected_prior = np.where(valid, -np.log(count)[..., None], -np.inf)
        if not np.array_equal(copies, expected_copies) or not np.array_equal(prior, expected_prior):
            raise ValueError("Integer candidates require distinct integers and fixed uniform priors.")
        if (self.model_id, self.model_version, self.candidate_generator_version, self.prior_mode) != (
            CLONAL_INTEGER_MODEL_ID, "1", CLONAL_INTEGER_GENERATOR_VERSION, CLONAL_INTEGER_PRIOR_MODE,
        ):
            raise ValueError("Unsupported integer multiplicity specification identity.")
        object.__setattr__(self, "copies", copies)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "log_prior", prior)

    def validate_observation_shape(self, shape: tuple[int, int]) -> None:
        if self.copies.shape[:2] != shape:
            raise ValueError(f"Integer candidate mutation-region shape must be {shape}.")


def build_clonal_integer_likelihood(major_cn: np.ndarray) -> IntegerMultiplicitySpec:
    """Compile one linear emission per distinct allowed integer, from 1 to A.

    CN eligibility must be established from the original local states before
    invoking this builder: dominant CN arrays cannot establish clonality.
    """
    major = np.asarray(major_cn, dtype=np.float64)
    if major.ndim != 2 or 0 in major.shape:
        raise ValueError("major_cn must have nonempty shape (M, S).")
    if not np.all(np.isfinite(major)):
        raise ValueError("major_cn must contain finite values.")
    rounded = np.rint(major)
    if not np.allclose(major, rounded, rtol=0.0, atol=1e-8):
        raise ValueError("Retained major_cn values must be integers.")
    if np.any((rounded < 1) | (rounded > MAX_MAJOR_CN)):
        raise ValueError(
            "Retained major_cn must lie in [1, 6]; "
            "apply CN filtering before candidate construction."
        )
    major_int = rounded.astype(np.int64)
    candidates = np.arange(1, int(major_int.max()) + 1, dtype=np.float64)
    valid = candidates <= major_int[..., None]
    copies = np.where(valid, candidates, 0.0)
    return IntegerMultiplicitySpec(
        model_id=CLONAL_INTEGER_MODEL_ID,
        model_version="1",
        candidate_generator_version=CLONAL_INTEGER_GENERATOR_VERSION,
        prior_mode=CLONAL_INTEGER_PRIOR_MODE,
        copies=copies,
        log_prior=np.where(
            valid, -np.log(major_int.astype(np.float64))[..., None], -np.inf
        ),
        valid=valid,
    )
