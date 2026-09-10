"""Derived diagnostics reject invalid components without conflicting identities."""

from dataclasses import asdict, fields, replace
import math

import numpy as np
import pytest
import torch

from CliPP2.core.fusion import torch_backend as backend
from CliPP2.core.fusion.solver import _backward_error_kkt_within_gate
from CliPP2.core.fusion.types import FitProvenance, KKTComponents, KKTDiagnostics, _residual_max
from CliPP2.core.objective import BaseObjectiveKey, LambdaObjectiveKey


INVALID = (float("nan"), float("inf"), -float("inf"), -.1)
LEGACY = ("stationarity_residual", "edge_subgradient_residual", "dual_ball_residual")
BACKWARD = ("backward_error_stationarity_residual", "backward_error_edge_subgradient_residual",
            "backward_error_dual_ball_residual")


def _diagnostics():
    return KKTDiagnostics(.001, .002, .003, .0005, .0001, .0002, .0003)


def test_residual_max_empty_and_valid_values():
    assert _residual_max() == 0.
    assert _residual_max(np.float32(0), .2, 0, .1) == .2
    assert _residual_max(-0.) == 0.


@pytest.mark.parametrize("invalid", INVALID)
@pytest.mark.parametrize("position", range(4))
def test_terminal_component_max_cannot_hide_invalid_values(position, invalid):
    values = [.001, .002, .003, .0005]
    values[position] = invalid
    components = KKTComponents(*values)
    assert components.residual == math.inf
    assert _residual_max(*values) == math.inf
    recorded = getattr(components, fields(components)[position].name)
    assert np.isnan(recorded) if np.isnan(invalid) else recorded == invalid


@pytest.mark.parametrize("invalid", INVALID)
@pytest.mark.parametrize("component", LEGACY + BACKWARD + ("box_residual",))
def test_each_diagnostic_total_uses_its_own_fail_closed_components(component, invalid):
    diagnostics = replace(_diagnostics(), **{component: invalid})
    if component in LEGACY or component == "box_residual":
        assert diagnostics.kkt_residual == math.inf
    else:
        assert diagnostics.kkt_residual == .003
    if component in BACKWARD or component == "box_residual":
        assert diagnostics.backward_error_kkt_residual == math.inf
        assert not _backward_error_kkt_within_gate(diagnostics, certification_tol=8e-4)
    else:
        assert diagnostics.backward_error_kkt_residual == .0005
    recorded = getattr(diagnostics, component)
    assert np.isnan(recorded) if np.isnan(invalid) else recorded == invalid


def test_totals_are_derived_and_flattened_valid_values_are_unchanged():
    diagnostics = _diagnostics()
    assert diagnostics.kkt_residual == .003
    assert diagnostics.backward_error_kkt_residual == .0005
    assert _backward_error_kkt_within_gate(diagnostics, certification_tol=8e-4)
    stored = asdict(diagnostics)
    assert "kkt_residual" not in stored and "backward_error_kkt_residual" not in stored
    flattened = stored | {"kkt_residual": diagnostics.kkt_residual,
                          "backward_error_kkt_residual": diagnostics.backward_error_kkt_residual}
    assert flattened == dict(stationarity_residual=.001, edge_subgradient_residual=.002,
        dual_ball_residual=.003, box_residual=.0005, kkt_residual=.003,
        backward_error_stationarity_residual=.0001, backward_error_edge_subgradient_residual=.0002,
        backward_error_dual_ball_residual=.0003, backward_error_kkt_residual=.0005)
    for name in ("kkt_residual", "backward_error_kkt_residual"):
        with pytest.raises(TypeError, match=name):
            replace(diagnostics, **{name: 0.})
    assert KKTDiagnostics(0., 0., 0., 0.).backward_error_kkt_residual == math.inf


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("invalid", INVALID)
@pytest.mark.parametrize("component", ["max_edge_residual", "max_ball_residual",
                                        "max_scaled_edge_residual", "max_scaled_ball_residual"])
def test_backend_assembly_preserves_invalid_edge_components(dtype, component, invalid):
    kwargs = dict(phi=torch.full((2, 1), .5, dtype=dtype),
        grad_smooth=torch.zeros((2, 1), dtype=dtype), adj=torch.zeros((2, 1), dtype=dtype),
        lower=torch.zeros((2, 1), dtype=dtype), upper=torch.ones((2, 1), dtype=dtype), atol=1e-6,
        max_edge_residual=.002, max_ball_residual=.001, max_radius=1.,
        max_scaled_edge_residual=.002, max_scaled_ball_residual=.001)
    kwargs[component] = torch.tensor(invalid, dtype=dtype)
    diagnostics = backend.graph_fusion_kkt_diagnostics_from_components_torch(**kwargs)
    if component.startswith("max_scaled"):
        assert diagnostics.backward_error_kkt_residual == math.inf
        assert not _backward_error_kkt_within_gate(diagnostics, certification_tol=8e-4)
    else:
        assert diagnostics.kkt_residual == math.inf


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("radius", INVALID + (-1.,))
@pytest.mark.parametrize("scaled", [False, True])
def test_invalid_radius_cannot_normalize_a_violation_to_zero(dtype, radius, scaled):
    phi = torch.full((2, 1), .5, dtype=dtype)
    diagnostics = backend.graph_fusion_kkt_diagnostics_from_components_torch(
        phi=phi, grad_smooth=torch.zeros_like(phi), adj=torch.zeros_like(phi),
        lower=torch.zeros_like(phi), upper=torch.ones_like(phi), atol=1e-6,
        max_edge_residual=.1, max_ball_residual=0., max_radius=radius,
        max_scaled_edge_residual=0. if scaled else None,
        max_scaled_ball_residual=0. if scaled else None,
    )
    assert diagnostics.kkt_residual == diagnostics.backward_error_kkt_residual == math.inf
    assert not _backward_error_kkt_within_gate(diagnostics, certification_tol=8e-4)


@pytest.mark.parametrize("epsilon", [1e-6, 2e-6, np.nextafter(1e-6, np.inf), 1e-12])
def test_provenance_epsilon_has_one_exact_identity(epsilon):
    base = BaseObjectiveKey("likelihood", "graph", "box", float(epsilon).hex())
    provenance = FitProvenance(LambdaObjectiveKey(base, .1.hex()), "data", "cpu", "float64",
                               "fixture", "not_certified")
    assert provenance.likelihood_eps.hex() == float(epsilon).hex()
    assert "likelihood_eps" not in {item.name for item in fields(provenance)}
    with pytest.raises(TypeError, match="likelihood_eps"):
        replace(provenance, likelihood_eps=.1)
    changed_base = replace(base, eps_hex=float(2 * epsilon).hex())
    changed = replace(provenance, objective_key=replace(provenance.objective_key, base=changed_base))
    assert changed.likelihood_eps == 2 * epsilon
    assert changed.objective_spec_hash != provenance.objective_spec_hash


def test_malformed_epsilon_identity_fails_instead_of_using_a_duplicate_value():
    base = BaseObjectiveKey("likelihood", "graph", "box", "not-hex")
    provenance = FitProvenance(LambdaObjectiveKey(base, .1.hex()), "data", "cpu", "float64",
                               "fixture", "not_certified")
    with pytest.raises(ValueError):
        _ = provenance.likelihood_eps
