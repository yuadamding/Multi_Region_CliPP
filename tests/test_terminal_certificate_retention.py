"""Terminal refinement keeps evidence, not a history of obsolete witnesses."""

from dataclasses import replace
import weakref

import numpy as np
import pytest
import torch

from CliPP2.config import resolve_fit_config
from CliPP2.core.fusion import solver
from CliPP2.core.fusion.certificates import CertificateAttempt, CertificateGradient
from CliPP2.core.fusion.types import DenseEdgeCertificate, KKTDiagnostics, WorkCounters
from test_solver_request import _prepared


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_terminal_four_plus_final_pass_streams_work_and_releases_witnesses(monkeypatch, dtype):
    data, context = _prepared()
    if dtype == "float32":
        context = solver.prepare_torch_problem(
            data, eps=context.eps, tol=8e-4, inner_max_iter=16,
            graph=context.graph_spec, exact_pilot=data.phi_init,
            pooled_start=data.phi_init, scalar_well_starts=(), device="cpu", dtype=dtype,
        )
    options = resolve_fit_config(
        device="cpu", dtype=dtype, outer_max_iter=1, inner_max_iter=16,
        certificate_max_iter=512,
    ).solver
    references, live_before = [], []
    gradient_calls = 0
    audit_calls = []

    def alternating_gradient(model, phi, **kwargs):
        nonlocal gradient_calls
        gradient_calls += 1
        return CertificateGradient(
            value=torch.full_like(phi, .1 * gradient_calls),
            scope="observed_objective", directional_admissible=True,
            at_breakpoint=torch.ones_like(phi, dtype=torch.bool),
        )

    def recorded_certificate(*, problem, phi, gradient, **kwargs):
        terminal = kwargs.get("max_iter") == 512
        if terminal:
            live_before.append(sum(reference() is not None for reference in references))
        index = len(references) + 1 if terminal else 0
        dual = torch.full(
            (problem.graph.weight.numel(), phi.shape[1]), float(index),
            dtype=phi.dtype, device=phi.device,
        )
        if terminal:
            references.append(weakref.ref(dual))
        return CertificateAttempt(
            DenseEdgeCertificate(dual, problem.graph_hash, gradient.scope),
            KKTDiagnostics(1., 1., 1., 0., 1., 1., 1., 1., 1.),
            "refined_fused_edge_dual", WorkCounters(index),
        )

    monkeypatch.setattr(solver, "build_certificate_gradient", alternating_gradient)
    monkeypatch.setattr(solver, "certify", recorded_certificate)

    def final_audit(**kwargs):
        # The optional fifth pass must also release its predecessor BEFORE
        # promotion/auditing allocates more working storage, not just on return.
        assert [reference() is not None for reference in references] == [False] * 4 + [True]
        audit_calls.append(True)
        return KKTDiagnostics(1., 1., 1., 0., 1., 1., 1., 1., 1.), "observed_objective", True, 123.

    monkeypatch.setattr(solver, "_terminal_backward_error_audit_float64", final_audit)
    fit = solver._fit_from_start(
        context, .1, options, solver._StartAttempt(data.phi_init, None, "retention_fixture"),
    )
    assert len(references) == 5  # Four alternating passes and the unchanged final pass.
    assert live_before == [0, 1, 1, 1, 1]
    assert fit.work.full_certificate_audit_passes == 1 + 2 + 3 + 4 + 5 + len(audit_calls)
    assert len(audit_calls) == int(dtype == "float32")
    assert [reference() is not None for reference in references] == [False] * 4 + [True]
    torch.testing.assert_close(fit.certificate.witness.dual, torch.full((1, 1), 5., dtype=context.runtime.dtype))
    assert fit.certificate.components.residual == 1.
    assert not fit.certificate.certified


@pytest.mark.parametrize("residual, accepted", [
    (0., True), (.004, True), (.00401, False),
    (float("inf"), False), (float("nan"), False), (-.1, False),
])
def test_typed_admission_keeps_componentwise_gate(residual, accepted):
    diagnostics = KKTDiagnostics(0., 0., 0., 0., 0.)
    assert not solver._backward_error_kkt_within_gate(diagnostics, certification_tol=8e-4)
    diagnostics = replace(diagnostics, backward_error_kkt_residual=residual)
    assert solver._backward_error_kkt_within_gate(diagnostics, certification_tol=8e-4) is accepted
    # A favorable legacy progress residual cannot override componentwise rejection.
    assert diagnostics.kkt_residual == 0.
    if np.isnan(residual):
        assert np.isnan(diagnostics.backward_error_kkt_residual)


def test_mapping_diagnostics_are_not_a_second_internal_interface():
    with pytest.raises(AttributeError):
        solver._backward_error_kkt_within_gate(
            {"backward_error_kkt_residual": 0.}, certification_tol=8e-4,
        )
