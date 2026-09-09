"""Compile the clonal, uniform integer-multiplicity mixture."""

from __future__ import annotations

import numpy as np

from .data import PathLikelihoodSpec

CLONAL_INTEGER_MODEL_ID = "clipp2_clonal_integer_multiplicity_mixture_v1"
CLONAL_INTEGER_GENERATOR_VERSION = "integer_1_to_major_cap6_v1"
CLONAL_INTEGER_PRIOR_MODE = "uniform_distinct_integer_v1"
MAX_MAJOR_CN = 6


def build_clonal_integer_likelihood(major_cn: np.ndarray) -> PathLikelihoodSpec:
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
    return PathLikelihoodSpec(
        model_id=CLONAL_INTEGER_MODEL_ID,
        model_version="1",
        candidate_generator_version=CLONAL_INTEGER_GENERATOR_VERSION,
        prior_mode=CLONAL_INTEGER_PRIOR_MODE,
        first_copy=copies,
        second_copy=copies,
        switch_fraction=np.where(valid, 1.0, 0.0),
        log_prior=np.where(
            valid, -np.log(major_int.astype(np.float64))[..., None], -np.inf
        ),
        valid=valid,
        legacy_major_indicator=None,
    )
