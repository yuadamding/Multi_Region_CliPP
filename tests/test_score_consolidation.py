"""Pinned bb726ad score/uncertainty parity, without retaining its score API."""

from dataclasses import asdict, replace
from math import fsum, lgamma

import numpy as np
import pytest

from CliPP2.core import bic
from CliPP2.io.data import TumorData, tumor_data_fingerprint
from CliPP2.model_selection.partitions import _partition_signature
from CliPP2.model_selection.scoring import select_candidate_records
from CliPP2.model_selection.types import (
    CandidateRecord, DirectPartition, DirectPartitionCandidate, PartitionRefitSummary,
)


def _data(n, regions, mode="full"):
    shape = (n, regions)
    total = np.full(shape, 20.0)
    observed = np.ones(shape, dtype=bool)
    if mode == "mixed":
        total.reshape(-1)[::4] = 0.0
        observed.reshape(-1)[1::5] = False
    elif mode == "zero":
        total[:] = 0.0
    return TumorData(
        tumor_id="score", mutation_ids=tuple(f"m{i}" for i in range(n)),
        region_ids=tuple(f"R{i}" for i in range(regions)),
        alt_counts=total / 2.0, total_counts=total, count_observed=observed,
        purity=np.ones(shape), major_cn=np.full(shape, 2.0), minor_cn=np.ones(shape),
        normal_cn=np.full(shape, 2.0), scaling=np.full(shape, 1.0 / 3.0),
        phi_upper=np.ones(shape), phi_init=np.full(shape, 0.5),
    )


def _reference(data, labels, loglik, gap=0.0, signature="golden"):
    """Independent two-stage arithmetic retained by bb726ad, including rounding."""
    sizes = np.unique(labels, return_counts=True)[1]
    k = len(sizes)
    n_eff = max(int(np.sum((data.total_counts > 0) & data.count_observed)), 1)
    df = k * data.num_regions
    center_penalty = float(df * np.log(n_eff))
    classic = float(-2.0 * loglik + center_penalty)
    epsilon = np.finfo(np.float64).eps
    base_uncertainty = float(2.0 * max(gap, 0.0) + 16.0 * epsilon * (1.0 + abs(classic)))
    terms = [lgamma(float(k)), -lgamma(float(sum(sizes)) + float(k))]
    for size in sizes:
        terms.extend((lgamma(float(size) + 1.0), -lgamma(1.0)))
    terms.append(lgamma(float(k) + 1.0))
    mass = float(fsum(terms))
    mass_uncertainty = float(
        32.0 * epsilon * float(len(terms) + 1)
        * (1.0 + float(fsum(abs(term) for term in terms)))
    )
    assignment = float(-2.0 * 0.7 * mass)
    assignment_uncertainty = float(2.0 * 0.7 * mass_uncertainty)
    addition_uncertainty = float(
        16.0 * epsilon * (1.0 + abs(classic) + abs(assignment))
    )
    return dict(
        name="fixed_partition_dirichlet_score", value=float(classic + assignment),
        loglik=float(loglik), penalty=float(center_penalty + assignment),
        degrees_of_freedom=df, n_eff=n_eff, partition_signature=signature,
        numerical_uncertainty=float(
            base_uncertainty + assignment_uncertainty + addition_uncertainty
        ),
        assignment_log_evidence=mass, assignment_code_weight=0.7,
        assignment_dirichlet_alpha=1.0,
        assignment_arithmetic_uncertainty=assignment_uncertainty,
    )


# Captured by executing core/bic.py from bb726ad3fb3737cca48e06cce66943d5fbba52b4.
# Rows store value, penalty, total uncertainty, allocation mass, allocation
# uncertainty, and n_eff. No fit or CUDA equivalence is inferred from these.
@pytest.mark.parametrize("sizes,regions,mode,loglik,gap,golden", [
    ([1], 1, "full", -0.125, 0.0,
     (0.25, 0.0, 6.856737400084966e-14, 0.0, 5.96855898038484e-14, 1)),
    ([2, 1], 1, "full", -123.456, 1e-8,
     (251.6176878342555, 4.705687834255496, 2.0002228859983336e-8,
      -1.791759469228055, 4.428152049089799e-13, 3)),
    ([3, 2, 1], 3, "mixed", -1000.125, 0.25,
     (2028.8619712813834, 28.61197128138336, 0.5000000000160438,
      -5.634789603169248, 1.6487563444833417e-12, 10)),
    ([9, 1], 2, "zero", -2.5, 0.0,
     (10.610266459325462, 5.610266459325463, 2.6089331392330024e-12,
      -4.007333185232474, 2.5463689046956353e-12, 1)),
    ([1] * 12, 6, "mixed", -98765.4321, 1e-3,
     (197819.74042703313, 288.8762270331226, 0.002000001430625427,
      -14.117153226228606, 2.5094741986070554e-11, 42)),
    ([4900, 70, 20, 10], 6, "mixed", -1e9, 0.125,
     (2000001051.95226, 1051.9522600537027, 0.2500142197704689,
      -583.4265794061597, 8.911173869908323e-9, 18000)),
])
def test_scores_and_all_uncertainties_match_pinned_golden(
    sizes, regions, mode, loglik, gap, golden,
):
    labels = np.repeat(np.arange(len(sizes)), sizes)
    data = _data(len(labels), regions, mode)
    score = bic.fixed_partition_dirichlet_score(
        data=data, labels=labels, num_clusters=len(sizes), loglik=loglik,
        partition_signature="golden", loglik_uncertainty=gap,
    )
    assert (
        score.value, score.penalty, score.numerical_uncertainty,
        score.assignment_log_evidence, score.assignment_arithmetic_uncertainty,
        score.n_eff,
    ) == golden
    assert asdict(score) == _reference(data, labels, loglik, gap)


@pytest.mark.parametrize("seed", range(24))
def test_score_primitives_preserve_randomized_values_and_ranking(seed):
    rng = np.random.default_rng(seed)
    n, regions = int(rng.integers(2, 200)), int(rng.integers(1, 7))
    data = _data(n, regions, "mixed")
    scores, references = [], []
    for k in range(1, min(n, 12) + 1):
        labels = np.arange(n) % k
        rng.shuffle(labels)
        loglik = -float(rng.uniform(0.0, 1e7))
        gap = float(rng.choice([-1.0, 0.0, 1e-9, 0.2]))
        score = bic.fixed_partition_dirichlet_score(
            data=data, labels=labels, num_clusters=k, loglik=loglik,
            partition_signature="golden", loglik_uncertainty=gap,
        )
        reference = _reference(data, labels, loglik, gap)
        assert asdict(score) == reference
        scores.append((score.value, score.numerical_uncertainty, k))
        references.append((reference["value"], reference["numerical_uncertainty"], k))
    assert sorted(scores) == sorted(references)


def _record(data, labels, loglik, gap, candidate_id):
    signature = _partition_signature(labels, data.mutation_ids)
    k = len(np.unique(labels))
    score = bic.fixed_partition_dirichlet_score(
        data=data, labels=labels, num_clusters=k, loglik=loglik,
        partition_signature=signature, loglik_uncertainty=gap,
    )
    refit = PartitionRefitSummary(
        labels=labels, partition_signature=signature, phi=np.full(data.alt_counts.shape, 0.5),
        cluster_centers=np.full((k, data.num_regions), 0.5), loglik=loglik,
        finite_candidate_found=True, global_optimum_certified=False,
        source_data_hash=tumor_data_fingerprint(data), likelihood_eps=1e-6,
    )
    partition = DirectPartition(
        labels, signature, "pilot_hessian_ward", data.mutation_ids,
    )
    return CandidateRecord(candidate_id, DirectPartitionCandidate(
        partition, refit, score, eligible_for_selection=True, ineligibility_reason="",
    ))


@pytest.mark.parametrize("offset", [-1e-7, 0.0, 1e-7])
def test_selection_interval_boundaries_match_reference_in_both_orders(offset):
    data = _data(6, 3, "mixed")
    first_labels, second_labels = np.zeros(6, dtype=int), np.arange(6) % 2
    first = _record(data, first_labels, -100.0, 0.25, 0)
    second_at_zero = _reference(data, second_labels, 0.0, 0.25)
    # Put the better nominal K=2 score at the uncertainty overlap boundary:
    # overlap must prefer K=1; separated intervals must retain K=2.
    target = first.score.value - 1.0 + offset
    loglik = (second_at_zero["value"] - target) / 2.0
    second = _record(data, second_labels, loglik, 0.25, 1)
    records = [first, second]
    reference_records = [replace(row, candidate=replace(
        row.candidate, score=bic.SelectionScore(**_reference(
            data, row.candidate.partition.labels, row.score.loglik, 0.25,
            row.partition_signature,
        )),
    )) for row in records]
    expected = select_candidate_records(reference_records).selected.candidate_id
    assert select_candidate_records(records).selected.candidate_id == expected
    assert select_candidate_records(list(reversed(records))).selected.candidate_id == expected
    if offset < 0.0:
        assert expected == 1
    elif offset > 0.0:
        assert expected == 0


def test_alternate_score_entry_points_are_retired():
    for name in (
        "fixed_partition_bic", "compute_classic_bic", "compute_bic_with_df",
        "compute_partition_dirichlet_score", "compute_dirichlet_exact_partition_log_mass",
        "effective_bic_depth_count", "PARTITION_DIRICHLET_SCORE_WEIGHT",
        "PARTITION_DIRICHLET_ALPHA",
    ):
        assert not hasattr(bic, name)
