"""Initialization seeds never replace the fitted categorical mixture."""

import numpy as np
import pandas as pd
import pytest
import torch

from CliPP2.core.fusion import starts
from CliPP2.core.fusion.graph import build_complete_adaptive_graph
from CliPP2.core.fusion.solver import (
    fit_prepared,
    objective_shape_for_data,
    prepare_torch_problem,
    promote_solver_context_dtype,
    has_multiplicity_ambiguity,
)
from CliPP2.core.fusion.torch_backend import resolve_runtime, to_torch_tumor_data
from CliPP2.core.objective import compile_observed_model, observed_terms_torch
from CliPP2.core.scalar import (
    certify_scalar_minimum,
    scalar_loss,
    scalar_problem_from_model,
)
from CliPP2.io.tumor_txt import load_tumor_txt, write_tumor_txt
from test_integer_likelihood import EPS, integer_data


def test_singleton_is_fixed_and_multiple_candidates_route_nonconvex():
    fixed = integer_data(((1,),))
    mixture = integer_data(((6,),))
    assert not has_multiplicity_ambiguity(fixed)
    assert objective_shape_for_data(fixed, "auto") == "unimodal"
    assert has_multiplicity_ambiguity(mixture)
    assert objective_shape_for_data(mixture, "unimodal") == "generic_nonconvex"


@pytest.mark.parametrize("candidates", [1, 2, 6])
def test_all_candidate_counts_share_scalar_numpy_and_torch_pilot(candidates):
    data = integer_data(((candidates,),))
    direct = starts.compute_scalar_mutation_region_wells(data, eps=EPS, tol=1e-7, max_iter=64)
    tensor = to_torch_tumor_data(data, resolve_runtime("cpu", dtype="float64"))
    converted = starts.compute_scalar_mutation_region_wells_torch(
        tensor, phi_init=data.phi_init, eps=EPS, tol=1e-7, max_iter=64,
    )
    for numpy_values, torch_values in zip(direct, converted):
        np.testing.assert_array_equal(numpy_values, torch_values.numpy())
    assert not hasattr(starts, "_binary_linear_model")


@pytest.mark.parametrize("candidates,expected_pilot,expected_weights", [
    (1, [1e-6, .5, 1., .5, 1., .5],
     [2.9999775002107483e-6, 1.4999872501151242e-6, 2.9999775002107483e-6,
      1.4999872501151242e-6, 1.499985750127874e-6, 2.999971500255748e-6,
      1.499985750127874, 2.999971500255748e-6, 1.499985750127874e-6,
      2.999971500255748e-6, 1.499985750127874, 1.499985750127874e-6,
      2.999971500255748e-6, 1.499985750127874e-6, 1.499985750127874e-6]),
    (2, [1e-6, .35194989414708155, .962500017361, .5, 1., .5],
     [.16191540474982202, .05920623981863613, .11397212323840043,
      .05698600463308161, .05698594764707698, .09333541257367138,
      .3849098743886757, .08793447780102749, .05698594764707698,
      .12321285515238617, 1.5196259741147602, .05698594764707698,
      .11397189529415395, .05698594764707698, .05698594764707698]),
    (6, [1e-6, .3011891772823886, .7173873770729129, .5, 1., .5],
     [.314052014971717, .13185189597150543, .18917788627815826,
      .0945888485500414, .09458875396119286, .2272685321772177,
      .4757726600002336, .13535673874274848, .09458875396119286,
      .4351161287965091, .3346940167835191, .09458875396119286,
      .18917750792238572, .09458875396119286, .09458875396119286]),
])
def test_certified_pilot_and_adaptive_graph_match_cc5a3d1(
    tmp_path, candidates, expected_pilot, expected_weights,
):
    # Captured independently from cc5a3d1ac28097c2b3005c1c2615930f0ab424de.
    # Boundaries, zero depth, and missing observations test pilot conventions,
    # not only attained likelihood. Retired two-candidate code rounded scalar
    # roots to12decimals; the shared routine uses14 (<6e-12 weight difference).
    rows = [dict(mutation_id=f"m{i}", sample_id="R1", alt_count=alt,
                 ref_count=total-alt, count_observed=observed, purity=.8, normal_cn=2,
                 segment_id=f"s{i}", cn_state_id="clonal", cn_state_fraction=1,
                 allele_a_cn=candidates, allele_b_cn=1)
            for i, (alt, total, observed) in enumerate([(0,100,1), (20,100,1), (55,100,1),
                                                      (0,0,1), (100,100,1), (10,100,0)])]
    path = tmp_path / "pilot.tsv"
    write_tumor_txt(path, pd.DataFrame(rows))
    data = load_tumor_txt(path)
    records = []
    primary, _, _ = starts.compute_scalar_mutation_region_wells(
        data, eps=EPS, tol=1e-5, max_iter=64, certificates=records,
    )
    np.testing.assert_allclose(primary[:, 0], expected_pilot, atol=1e-10, rtol=0)
    assert all(record.globally_certified for record in records)
    graph = build_complete_adaptive_graph(primary, count_observed=data.count_observed)
    np.testing.assert_allclose(graph.edge_w, expected_weights, atol=1e-10, rtol=0)


def test_candidate_start_bank_is_bounded_and_refines_full_marginal_loss():
    data = integer_data(((6,), (4,)))
    runtime = resolve_runtime("cpu", dtype="float64")
    td = to_torch_tumor_data(data, runtime)
    pilot = torch.tensor(data.phi_init)
    candidate_starts = starts._linear_candidate_starts_torch(td, pilot=pilot, eps=EPS)
    assert 1 < len(candidate_starts) <= 6
    bank = starts.compute_scalar_well_start_bank_torch(td, eps=EPS, exact_pilot=pilot)
    assert 1 <= len(bank) <= 7
    torch.testing.assert_close(bank[0], pilot)
    for index in range(6):
        scale = td.observed_model.slope[..., index]
        valid = td.observed_model.valid[..., index]
        seed = torch.where(valid, ((td.alt + 0.5) / (td.total + 1.0))
                           / torch.clamp(scale, min=1e-100), pilot)
        seed = torch.clamp(seed, min=EPS, max=1.0)
        seed_loss = observed_terms_torch(td.observed_model, seed, eps=EPS).loss
        # Deduplication can merge candidates. At least one retained matrix
        # reaches a loss no worse than each original initialization seed.
        assert any(bool(torch.all(observed_terms_torch(td.observed_model, s,
                       eps=EPS).loss <= seed_loss + 1e-10)) for s in candidate_starts)
    assert all(bool(torch.all((s >= EPS) & (s <= td.phi_upper))) for s in bank)


def test_scalar_certificate_metadata_propagates_without_grid_global_claim():
    data = integer_data(((4,),))
    model = compile_observed_model(data, eps=EPS)
    records = []
    primary, _, _ = starts.compute_scalar_mutation_region_wells(
        data, eps=EPS, tol=1e-5, max_iter=64,
        certificates=records)
    assert len(records) == 1
    result = records[0]
    assert result.method == "interval_binomial_mixture_bound_v1"
    problem = scalar_problem_from_model(model, np.array([0]), 0,
                                       lower=EPS, upper=1.0, eps=EPS)
    assert primary[0, 0] == result.argmin
    np.testing.assert_allclose(scalar_loss(problem, result.argmin), result.attained_value)
    grid_best = np.min(scalar_loss(problem, np.linspace(EPS, 1.0, 10001)))
    assert result.global_lower_bound <= grid_best + 1e-10
    if result.globally_certified:
        assert result.optimality_gap <= 1e-6
    else:
        assert result.optimality_gap > 1e-6
    context = prepare_torch_problem(data, eps=EPS, tol=1e-5,
                                    inner_max_iter=32, device="cpu", dtype="float64",
                                    defer_graph=True)
    assert len(context.scalar_pilot_certificates) == 1
    assert context.scalar_pilot_certificates[0].argmin == context.exact_pilot[0, 0]
    promoted = promote_solver_context_dtype(context, dtype=torch.float32)
    assert promoted.scalar_pilot_certificates == context.scalar_pilot_certificates
    replaced = promote_solver_context_dtype(context, dtype=torch.float32,
                                            start_override=np.array([[0.8]]))
    assert replaced.scalar_pilot_certificates == ()


def test_interval_budget_does_not_discard_unexplored_half(monkeypatch):
    # Simulated interval bounds isolate the budget accounting: initial 16
    # intervals plus one child used to lose the other half when max=17.
    from CliPP2.core import scalar
    data = integer_data(((4,),))
    model = compile_observed_model(data, eps=EPS)
    problem = scalar_problem_from_model(model, np.array([0]), 0,
                                       lower=0.1, upper=0.9, eps=EPS)
    monkeypatch.setattr(scalar, "scalar_breakpoints", lambda p: np.array([p.lower, p.upper]))
    monkeypatch.setattr(scalar, "scalar_loss", lambda p, beta: 1.0)
    calls = []
    def bound(p, left, right):
        calls.append((left, right))
        # Only the first initial interval is unresolved. Its left child has
        # bound 1, while its discarded right child would retain bound 0.
        if left < 0.15 and right - left > 0.03:
            return 0.0
        if 0.12 < left < 0.15:
            return 0.0
        return 1.0
    monkeypatch.setattr(scalar, "_interval_lower_bound", bound)
    result = certify_scalar_minimum(problem, tolerance=1e-6, max_intervals=17)
    assert not result.globally_certified
    assert result.global_lower_bound == 0.0
    assert result.intervals_evaluated == 16
    assert len(calls) == 16


def test_tiny_fixed_emission_fit_retains_scalar_pilot_evidence():
    from CliPP2.config import resolve_fit_config
    data = integer_data(((1,),))
    context = prepare_torch_problem(
        data, eps=EPS, inner_max_iter=16, tol=1e-4,
        device="cpu", dtype="float64",
    )
    options = resolve_fit_config(device="cpu", dtype="float64",
                                 outer_max_iter=2, inner_max_iter=16, tol=1e-4)
    result = fit_prepared(context, 0.0, options.solver)
    certificates = result.provenance.scalar_pilot_certificates
    assert len(certificates) == 1
    assert np.isfinite(certificates[0].attained_value)
    assert certificates[0].optimality_gap >= 0.0
    assert certificates[0].method == "interval_binomial_mixture_bound_v1"
