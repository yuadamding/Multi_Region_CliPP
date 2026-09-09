from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...io.data import TumorData
from ...io.multiplicity import CLONAL_INTEGER_MODEL_ID
from ..objective import compile_observed_model, observed_terms_numpy


@dataclass(frozen=True)
class MultiplicityPosterior:
    """Posterior multiplicity state at a fitted mutation-region CCF matrix."""

    gamma_major: np.ndarray
    major_call: np.ndarray
    multiplicity_call: np.ndarray
    estimation_mask: np.ndarray


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
    model = compile_observed_model(data, major_prior=0.5, eps=eps)
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
        spec.first_copy, map_index[..., None], axis=-1
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


def infer_multiplicity_posterior_numpy(
    data: TumorData,
    phi: np.ndarray,
    *,
    major_prior: float,
    eps: float,
) -> MultiplicityPosterior:
    """Infer the major/minor-copy posterior under the observed-count model.

    The calculation is the NumPy counterpart of
    ``mutation_region_terms_torch``. Entries outside the model's multiplicity
    estimation mask remain fixed to the available major-copy multiplicity.
    Entries whose counts are marked unobserved retain the configured prior.
    """

    phi_array = np.asarray(phi, dtype=np.float64)
    expected_shape = np.asarray(data.alt_counts).shape
    if phi_array.shape != expected_shape:
        raise ValueError(
            "phi must have the same mutation-region shape as the tumor data; "
            f"got {phi_array.shape}, expected {expected_shape}."
        )
    if not np.all(np.isfinite(phi_array)):
        raise ValueError("phi must contain only finite values.")

    prior = float(major_prior)
    if not np.isfinite(prior) or not (0.0 < prior < 1.0):
        raise ValueError("major_prior must lie strictly in (0, 1).")
    probability_eps = float(eps)
    if not np.isfinite(probability_eps) or not (0.0 < probability_eps < 0.5):
        raise ValueError("eps must lie strictly in (0, 0.5).")

    model = compile_observed_model(
        data,
        major_prior=prior,
        eps=probability_eps,
    )
    if model.legacy_major is None:
        raise ValueError(
            "Multiplicity reporting requires canonical legacy-major indicators."
        )
    posterior_major = observed_terms_numpy(
        model,
        phi_array,
        eps=probability_eps,
    ).legacy_major_probability
    estimation_mask = np.asarray(data.multiplicity_estimation_mask, dtype=bool)
    major_probability = np.asarray(posterior_major, dtype=np.float64)

    major_call = major_probability >= 0.5
    multiplicity_call = np.where(
        estimation_mask,
        np.where(major_call, data.major_cn, data.multiplicity_low),
        data.fixed_multiplicity,
    ).astype(np.float64, copy=False)
    return MultiplicityPosterior(
        gamma_major=major_probability,
        major_call=major_call.astype(bool, copy=False),
        multiplicity_call=multiplicity_call,
        estimation_mask=estimation_mask,
    )


__all__ = [
    "IntegerMultiplicityPosterior",
    "MultiplicityPosterior",
    "infer_integer_multiplicity_posterior_numpy",
    "infer_multiplicity_posterior_numpy",
]
