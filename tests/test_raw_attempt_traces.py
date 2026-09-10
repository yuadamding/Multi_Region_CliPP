"""Discarded raw starts retain scalar diagnostics, never numerical witnesses."""

from dataclasses import FrozenInstanceError, asdict, fields, is_dataclass, replace
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch

from CliPP2.config import resolve_fit_config
from CliPP2.core.bic import fixed_partition_dirichlet_score
from CliPP2.core.fusion.types import (
    ConvergenceResult, DenseEdgeCertificate, KKTComponents, ObjectiveValue, SolverState,
    WorkCounters,
)
from CliPP2.model_selection import search
from CliPP2.model_selection.proposals import RawStartAttempt, select_raw_start_attempt
from CliPP2.model_selection.types import CandidateRecord, CandidateTrace, RawFusionCandidate
from test_integer_reporting import _data, _selection


def _fit():
    data = _data(major=(3, 2), minor=(1, 1), alt=(30, 25), total=(100, 100))
    fit, partition, refit = _selection(data)
    fit = replace(
        fit, objective=ObjectiveValue(123.0), work=WorkCounters(7),
        certificate=replace(
            fit.certificate, components=KKTComponents(.01, .02, .03, .04),
            status="workset_incomplete", working_dtype="float32", audit_dtype="float64",
            precision_polished=True, fallback_reason="dense_cpu_after_solver_resource_limit",
        ),
        convergence=ConvergenceResult(
            converged=False, mm_consistency_violations=2,
            stage_outer_iterations=3, stage_outer_max_iter=4,
            stage_inner_iterations=5, stage_inner_max_iter=6, stage_inner_solve_calls=7,
            stop_reason="outer_iteration_limit", progress_residual_method="componentwise",
            solve_tolerance=5e-5, legacy_stop_kkt_residual=.2,
            componentwise_stop_kkt_residual=.04,
            accepted_full_steps=1, accepted_damped_steps=2, rejected_outer_steps=3,
        ),
    )
    return data, fit, partition, refit


def _trace(fit, source="cold", *, promotion="applied"):
    return search._raw_attempt_trace(
        RawStartAttempt(fit, source, .2, 3, False, promotion),
        search_round=2, search_phase="solver_recovery",
        outer_max_iter=40, inner_max_iter=60, certificate_max_iter=80,
    )


def _record(data, fit, partition, refit, traces):
    score = fixed_partition_dirichlet_score(
        data=data, labels=partition.labels, num_clusters=partition.n_clusters,
        loglik=refit.loglik, partition_signature=partition.signature,
    )
    return CandidateRecord(0, RawFusionCandidate(
        fit, partition, refit, score, False, "raw_not_certified",
    ), CandidateTrace(2, "solver_recovery", "cold", .2, 3, traces))


def _assert_scalar_record(value):
    if is_dataclass(value):
        for item in fields(value):
            _assert_scalar_record(getattr(value, item.name))
    else:
        assert type(value) in (int, float, bool, str), type(value)


def test_trace_releases_fit_state_and_certificate_witness():
    _, fit, _, _ = _fit()
    dual = torch.ones((1, 1))
    witness = DenseEdgeCertificate(dual, fit.provenance.original_graph_hash, "observed_objective")
    state = SolverState(torch.tensor(fit.phi.copy()), dual, .1, certificate=witness)
    fit = replace(fit, state=state, certificate=replace(fit.certificate, witness=witness))
    refs = [weakref.ref(item) for item in (fit.phi, state.phi, dual)]
    trace = _trace(fit)
    del fit, state, dual, witness
    assert all(ref() is None for ref in refs)
    _assert_scalar_record(trace)
    assert "fit" not in {item.name for item in fields(trace)}
    assert trace.objective == 123.0 and trace.work.full_certificate_audit_passes == 7
    assert trace.kkt_components == KKTComponents(.01, .02, .03, .04)
    with pytest.raises(FrozenInstanceError):
        trace.source = "changed"


def test_failure_fields_and_compact_token_match_bb726ad():
    data, fit, partition, refit = _fit()
    trace = _trace(fit)
    record = _record(data, fit, partition, refit, (trace,))
    diagnostics = search._raw_reference_failure_diagnostics([record])
    # Scalar fields below and the exact token were captured from the pinned
    # bb726ad diagnostic representation. New diagnostics add no legacy fallback.
    expected_best = dict(
        search_round=2, search_phase="solver_recovery", lambda_value=.1, source="cold",
        kkt_residual=.04, kkt_tolerance=.004, dominant_kkt_component="box",
        outer_max_iter=40, inner_max_iter=60, certificate_max_iter=80,
        working_dtype="float32", audit_dtype="float64", precision_polished=True,
        promotion_status="applied", stage_outer_iterations=3, stage_outer_max_iter=4,
        stage_inner_iterations=5, stage_inner_max_iter=6, stage_inner_solve_calls=7,
        stop_reason="outer_iteration_limit", progress_residual_method="componentwise",
        solve_tolerance=5e-5, legacy_stop_kkt_residual=.2,
        componentwise_stop_kkt_residual=.04, accepted_full_steps=1,
        accepted_damped_steps=2, rejected_outer_steps=3,
        fallback_reason="dense_cpu_after_solver_resource_limit",
    )
    best = asdict(diagnostics.best_raw_attempt)
    assert {key: best[key] for key in expected_best} == expected_best
    assert diagnostics.best_raw_attempt is trace
    assert diagnostics.raw_candidate_count == diagnostics.raw_solver_attempt_count == 1
    assert diagnostics.raw_certified_count == 0
    assert diagnostics.partition_certified_count == 1
    assert diagnostics.mm_violation_min == 2 and diagnostics.mm_violating_count == 1
    assert diagnostics.certificate_status_counts == (("workset_incomplete", 1),)
    assert diagnostics.promotion_status_counts == (("applied", 1),)
    assert diagnostics.attempt_summaries == (
        "r2:solver_recovery:cold@0.1|kkt=0.04/0.004|dtype=float32/float64"
        "|prom=applied|polish=1|progress=componentwise|solve_tol=5e-05"
        "|outer=3/4|inner=5/7x6|stop=outer_iteration_limit|stop_kkt=0.2/0.04"
        "|steps=1/2/3|fallback=dense_cpu_after_solver_resource_limit",
    )
    error = search.NoCertifiedRawReferenceError(
        tumor_id=data.tumor_id, records=[record], adaptive_search_stop_reason="budget",
    )
    assert "raw_solver_attempts=1" in str(error)
    assert "'stage_outer': '3/4'" in str(error)
    assert diagnostics.attempt_summaries[0] in str(error)


def test_trace_fields_are_typed_without_retired_result_fallbacks():
    _, fit, _, _ = _fit()
    with pytest.raises(TypeError, match="current typed RawFit"):
        _trace(SimpleNamespace(certificate=fit.certificate, convergence=fit.convergence))
    legacy = replace(fit, convergence=SimpleNamespace(iterations=10, mm_consistency_violations=0))
    with pytest.raises(AttributeError, match="stage_outer_iterations"):
        _trace(legacy)
    assert not hasattr(search, "BestRawAttemptDiagnostics")


def test_best_trace_stable_ties_nonfinite_components_and_absent_start_details():
    data, fit, partition, refit = _fit()
    first = _trace(fit, "first")
    second = _trace(fit, "second")
    record = _record(data, fit, partition, refit, (first, second))
    assert search._raw_reference_failure_diagnostics([record]).best_raw_attempt is first
    bad = _trace(replace(fit, certificate=replace(
        fit.certificate, components=KKTComponents(np.inf, np.inf, np.inf, np.inf),
    )))
    assert bad.dominant_kkt_component == "unknown"
    unresolved = search._raw_reference_failure_diagnostics([
        replace(record, trace=replace(record.trace, raw_attempts=(bad,))),
    ])
    assert unresolved.best_raw_attempt is None and np.isnan(unresolved.min_kkt_residual)
    fallback = search._raw_attempts(replace(record, trace=replace(record.trace, raw_attempts=())))
    assert len(fallback) == 1 and fallback[0].outer_max_iter == 0
    assert fallback[0].stage_outer_max_iter == 4
    _assert_scalar_record(fallback[0])


@pytest.mark.parametrize("seed", range(12))
def test_pairwise_streaming_start_selection_matches_historical_full_list(seed):
    _, base, _, _ = _fit()
    rng = np.random.default_rng(seed)
    attempts = []
    for index in range(24):
        certified = bool(rng.integers(0, 2)) if seed % 2 else False
        fit = replace(base, objective=ObjectiveValue(float(rng.choice([1, 1 + 1e-9, 2, np.inf]))),
                      certificate=replace(base.certificate,
                          components=KKTComponents(float(rng.choice([.01, .1, np.inf])), 0, 0, 0),
                          certified=certified, admissible=certified))
        attempts.append(RawStartAttempt(fit, f"s{index % 4}", .1, 0, certified))
    historical = select_raw_start_attempt(attempts)
    streaming = attempts[0]
    for attempt in attempts[1:]:
        streaming = select_raw_start_attempt([streaming, attempt])
    assert streaming is historical


@pytest.mark.parametrize("bank_size", [4, 32])
def test_search_discards_side_starts_but_preserves_recovery_and_bracket_states(
    monkeypatch, bank_size,
):
    data, template, _, _ = _fit()
    options = resolve_fit_config(device="cpu", dtype="float64")
    dual_refs = []
    calls = []
    observations = []
    markers = {}

    class Controller:
        stop_reason = "controlled_trace_test"
        round = -1

        def __init__(self, **kwargs):
            pass

        def propose(self):
            self.round += 1
            if self.round == 3:
                return None
            return SimpleNamespace(
                lambda_value=1.0 if self.round < 2 else 2.0,
                phase="solver_recovery" if self.round == 1 else "initial",
                retry_number=1 if self.round == 1 else 0,
                warm_start_lambda=1.0 if self.round else None,
                alternate_start_lambda=None,
            )

        def observe(self, observation):
            observations.append(observation)
            # One selected record per round plus the exact same-lambda KKT
            # recovery winner. No other completed start remains reachable.
            assert sum(ref() is not None for ref in dual_refs) <= self.round + 2

    def fake_fit(context, lam, solver_options, *, phi_start, include_default_starts, warm_state):
        round_index = len(observations)
        local_index = sum(round_ == round_index for round_, _ in calls)
        assert sum(ref() is not None for ref in dual_refs) <= round_index + 2
        if local_index == 0 and round_index in (1, 2):
            assert warm_state is not None
            expected = markers["recovery" if round_index == 1 else "selected"]
            torch.testing.assert_close(warm_state.phi, expected, rtol=0, atol=0)
        calls.append((round_index, local_index))
        phi = torch.full(data.alt_counts.shape, .2 + .001 * len(calls), dtype=torch.float64)
        dual = torch.zeros((1, 1), dtype=torch.float64)
        dual_refs.append(weakref.ref(dual))
        if round_index == 0 and local_index in (1, 2):
            markers["selected" if local_index == 1 else "recovery"] = phi.clone()
        witness = DenseEdgeCertificate(dual, context.graph_hash, "observed_objective")
        state = SolverState(phi, dual, lam, certificate=witness,
                            objective_spec_hash=context.objective_spec_hash)
        return replace(
            template, phi=phi.numpy(), state=state,
            objective=ObjectiveValue(1.0 if local_index == 1 else 2.0),
            certificate=replace(template.certificate, witness=witness,
                precision_polished=False, working_dtype="float64",
                components=KKTComponents(.01 if local_index == 2 else .04, 0, 0, 0)),
            provenance=replace(template.provenance, source_data_hash=context.data_fingerprint,
                objective_key=replace(template.provenance.objective_key,
                    base=context.base_objective_key, lambda_hex=float(lam).hex())),
        )

    monkeypatch.setattr(search, "OnlineLambdaController", Controller)
    monkeypatch.setattr(search, "fit_prepared", fake_fit)
    monkeypatch.setattr(search, "_escape_path_breakpoint_retry_state",
                        lambda state, **kwargs: (state, 0))
    monkeypatch.setattr(search, "_explicit_path_default_start_specs", lambda **kwargs: tuple(
        (f"bank_{i}", 0.0, None, np.full(data.alt_counts.shape, .1 + .7 * i / bank_size))
        for i in range(bank_size)
    ))
    monkeypatch.setattr(search, "_assemble_selection_result", lambda **kwargs: kwargs["result_entries"])
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        records = search._partition_guided_admm_selection(
            data=data, fit_options=options, use_warm_starts=True,
        )
    finally:
        torch.set_num_threads(previous_threads)
    raw = [row for row in records if row.family == "raw_fusion"]
    assert len(raw) == len(observations) == 3
    assert [obs.lambda_value for obs in observations] == [1.0, 1.0, 2.0]
    assert [row.candidate.raw_fit.objective.total for row in raw] == [1.0] * 3
    assert sum(len(row.trace.raw_attempts) for row in raw) == len(calls)
    assert len(calls) >= 3 * bank_size
    for row in raw:
        for trace in row.trace.raw_attempts:
            _assert_scalar_record(trace)
    # The returned historical candidates remain intact; controller-local
    # recovery state no longer lives after search completes.
    assert sum(ref() is not None for ref in dual_refs) == len(raw)
