"""Convex global claims require the actual clipped loss, not candidate count."""

from dataclasses import replace

import numpy as np
import pytest

from CliPP2.core.objective import (
    ObservedModel, has_proven_convex_observed_loss, observed_terms_numpy,
)


EPS = 1e-6


def _model(*, slopes=(0.5,), alt=10.0, nonalt=90.0, lower=EPS, upper=1.0, observed=True):
    slopes = np.array(slopes, dtype=np.float64)[None, None, :]
    return ObservedModel(
        alt=np.array([[alt]]), nonalt=np.array([[nonalt]]),
        observed=np.array([[observed]]), lower=np.array([[lower]]), upper=np.array([[upper]]),
        slope=slopes, log_prior=np.full(slopes.shape, -np.log(slopes.shape[-1])),
        valid=np.ones(slopes.shape, dtype=bool), model_id="clipp2_clonal_integer_multiplicity_mixture_v1",
    )


def test_review_single_candidate_counterexample_is_not_convex():
    model = _model()
    assert not has_proven_convex_observed_loss(model, eps=EPS)
    plateau = observed_terms_numpy(model, np.array([[EPS]]), eps=EPS)
    better = observed_terms_numpy(model, np.array([[0.2]]), eps=EPS)
    assert plateau.gradient.item() == 0
    assert plateau.loss.item() > better.loss.item() + 105
    # Direct Jensen violation: midpoint loss exceeds the endpoint average.
    losses = [observed_terms_numpy(model, np.array([[phi]]), eps=EPS).loss.item()
              for phi in (EPS, 2 * EPS, 3 * EPS)]
    assert losses[1] > 0.5 * (losses[0] + losses[2])


@pytest.mark.parametrize("changes,expected", [
    ({"alt": 0, "nonalt": 100}, True),
    ({"alt": 1, "nonalt": 99}, False),
    ({"alt": 100, "nonalt": 0}, False),
    ({"alt": 1, "nonalt": 100_000_000}, True),
    ({"lower": 0.01, "upper": 0.8}, True),
    ({"lower": EPS, "upper": 1.5 * EPS}, True),
    ({"lower": EPS, "upper": 2 * EPS}, True),
    ({"lower": 2 * EPS, "upper": 0.8}, False),
    ({"lower": 4 * EPS, "upper": 0.8}, True),
    ({"slopes": (0.0,)}, True),
    ({"slopes": (0.25, 0.5)}, False),
    ({"slopes": (0.5, 0.5)}, False),
    ({"slopes": (0.25, 0.5), "observed": False}, True),
    ({"slopes": (0.25, 0.5), "alt": 0, "nonalt": 0}, True),
    ({"slopes": (0.25, 0.5), "lower": 0.2, "upper": 0.2}, True),
])
def test_lower_clip_and_constant_coordinate_proofs(changes, expected):
    assert has_proven_convex_observed_loss(_model(**changes), eps=EPS) is expected


@pytest.mark.parametrize("alt,nonalt,lower,upper,expected", [
    (0, 100, 0.2, 0.8, False),
    (10, 90, 0.2, 0.8, False),
    (100, 0, 0.2, 0.8, True),
    (100_000, 1, 0.2, 0.8, True),
    (0, 100, 0.45, 0.8, True),  # Entirely upper-clipped, including its endpoint.
    (0, 100, 0.2, 0.45, False),  # Flat-side kernel at a dangerous upper-box endpoint.
    (100, 0, 0.2, 0.45, True),
    (0, 100, 0.0, 1.0, False),  # Both clipping transitions occur inside the box.
    (100, 0, 0.0, 1.0, False),
])
def test_upper_clip_and_endpoint_proofs(alt, nonalt, lower, upper, expected):
    model = _model(slopes=(2.0,), alt=alt, nonalt=nonalt, lower=lower, upper=upper)
    assert has_proven_convex_observed_loss(model, eps=0.1) is expected


def test_one_unproven_coordinate_disqualifies_the_whole_observed_loss():
    good = _model(lower=0.1)
    model = replace(good, alt=np.full((2, 1), 10.0), nonalt=np.full((2, 1), 90.0),
                    observed=np.ones((2, 1), dtype=bool),
                    lower=np.array([[0.1], [EPS]]), upper=np.ones((2, 1)),
                    slope=np.full((2, 1, 1), 0.5), log_prior=np.zeros((2, 1, 1)),
                    valid=np.ones((2, 1, 1), dtype=bool))
    assert not has_proven_convex_observed_loss(model, eps=EPS)
    masked = replace(model, observed=np.array([[True], [False]]))
    assert has_proven_convex_observed_loss(masked, eps=EPS)


def test_rounding_at_a_derivative_sign_threshold_declines_proof():
    model = _model(alt=10, nonalt=90)
    # At eps=.1 the lower-side derivative is mathematically zero, but rounded
    # sign comparisons are intentionally conservative unless an exact zero
    # count makes the jump sign immediate.
    assert not has_proven_convex_observed_loss(model, eps=0.1)


@pytest.mark.parametrize("epsilon", [0.0, -1.0, 0.5, np.nan, np.inf])
def test_missing_source_or_invalid_clipping_cannot_prove_convexity(epsilon):
    assert not has_proven_convex_observed_loss(None, eps=EPS)
    assert not has_proven_convex_observed_loss(_model(), eps=epsilon)
