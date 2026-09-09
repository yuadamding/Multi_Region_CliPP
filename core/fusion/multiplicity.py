from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...io.data import TumorData
from ...io.multiplicity import CLONAL_INTEGER_MODEL_ID
from ..objective import compile_observed_model, observed_terms_numpy


@dataclass(frozen=True)
class IntegerMultiplicityPosterior:
    """Categorical multiplicity posterior conditional on the supplied CCF.

    ``multiplicity_call`` is the smallest MAP candidate, including prior-only
    ties. Reporting must hide uninformative calls with multiple candidates;
    a singleton remains structurally fixed at one.
    """

    posterior: np.ndarray
    multiplicity_call: np.ndarray
    map_probability: np.ndarray
    candidate_count: np.ndarray
    informative: np.ndarray


def infer_integer_multiplicity_posterior_numpy(
    data: TumorData,
    phi: np.ndarray,
    *,
    eps: float,
) -> IntegerMultiplicityPosterior:
    """Evaluate the fitted integer mixture without replacing marginalization."""

    spec = data.path_likelihood
    if spec is None or spec.model_id != CLONAL_INTEGER_MODEL_ID:
        raise ValueError("Expected the clonal integer multiplicity model.")
    phi_array = np.asarray(phi, dtype=np.float64)
    expected_shape = np.asarray(data.alt_counts).shape
    if phi_array.shape != expected_shape:
        raise ValueError(
            "phi must have the same mutation-region shape as the tumor data; "
            f"got {phi_array.shape}, expected {expected_shape}."
        )
    if not np.all(np.isfinite(phi_array)):
        raise ValueError("phi must contain only finite values.")
    model = compile_observed_model(data, eps=eps)
    if np.any((phi_array < model.lower) | (phi_array > model.upper)):
        raise ValueError("phi must lie inside the compiled CCF bounds.")

    posterior = np.asarray(
        observed_terms_numpy(model, phi_array, eps=eps).posterior,
        dtype=np.float64,
    )
    # The compiler validates the complete, increasingly ordered integer range.
    # np.argmax therefore selects the lowest candidate on an exact tie.
    map_index = np.argmax(np.where(spec.valid, posterior, -np.inf), axis=-1)
    calls = np.take_along_axis(
        spec.copies, map_index[..., None], axis=-1
    )[..., 0].astype(np.int64)
    probability = np.take_along_axis(
        posterior, map_index[..., None], axis=-1
    )[..., 0]
    return IntegerMultiplicityPosterior(
        posterior=posterior,
        multiplicity_call=calls,
        map_probability=probability,
        candidate_count=np.sum(spec.valid, axis=-1).astype(np.int64),
        informative=model.observed & ((model.alt + model.nonalt) > 0.0),
    )


__all__ = [
    "IntegerMultiplicityPosterior",
    "infer_integer_multiplicity_posterior_numpy",
]
