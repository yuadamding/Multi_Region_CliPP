"""Guided starts cross the host boundary only after numerical diagnostics."""

from dataclasses import asdict

import numpy as np
import pytest
import torch

from CliPP2.core.fusion.graph import build_complete_uniform_graph
from CliPP2.core.fusion.solver import prepare_torch_problem
from CliPP2.model_selection.guided_fusion import build_guided_fusion_initialization
from test_integer_reporting import _data


@pytest.mark.parametrize("dtype,expected_lambda,expected_kkt,flow_balance,floor", [
    ("float32", 5.753207206726074, .020790545752470677,
     1.9073486328125e-6, 3.856043804262299e-6),
    ("float64", 5.753207693597483, .020790541478628972,
     3.552713678800501e-15, 7.182443061964168e-15),
])
def test_guided_host_state_preserves_pinned_72f76a8_values(
    monkeypatch, dtype, expected_lambda, expected_kkt, flow_balance, floor,
):
    data = _data(major=(3, 2), minor=(1, 1), alt=(30, 25), total=(100, 100))
    context = prepare_torch_problem(
        data, eps=1e-6, tol=8e-4, inner_max_iter=16,
        graph=build_complete_uniform_graph(2), exact_pilot=data.phi_init,
        pooled_start=data.phi_init, scalar_well_starts=(), device="cpu", dtype=dtype,
    )
    host_transfers = []
    original_cpu = torch.Tensor.cpu

    def transfer(tensor, *args, **kwargs):
        result = original_cpu(tensor, *args, **kwargs)
        host_transfers.append(result)
        return result

    monkeypatch.setattr(torch.Tensor, "cpu", transfer)
    result = build_guided_fusion_initialization(
        np.full(data.alt_counts.shape, .3), np.zeros(2, dtype=np.int64),
        solver_context=context,
    )
    state = result.solver_state
    for tensor in (state.phi, state.dual):
        assert any(tensor is transferred for transferred in host_transfers)
        assert tensor.device.type == "cpu" and tensor.dtype == getattr(torch, dtype)
        assert not tensor.requires_grad
    assert state.warm_state is None and state.certificate is None
    assert state.objective_spec_hash == context.objective_spec_hash
    assert state.previous_lambda == result.lambda_value == expected_lambda
    torch.testing.assert_close(state.phi, torch.full((2, 1), .3, dtype=getattr(torch, dtype)),
                               rtol=0, atol=0)
    torch.testing.assert_close(state.dual, torch.tensor([[expected_lambda]], dtype=getattr(torch, dtype)),
                               rtol=0, atol=0)
    # Captured independently from commit 72f76a8, before the state transfer and
    # mapping-to-KKTDiagnostics migration. No fit or CUDA qualification implied.
    assert asdict(result.diagnostics) == {
        "lambda_value": expected_lambda,
        "required_lambda_without_between_edges": expected_lambda,
        "numerical_lambda_floor": floor, "capacity_iterations": 1,
        "capacity_converged": True, "capacity_status": "exact_dual_capacity",
        "num_mutations": 2, "num_regions": 1, "num_clusters": 1,
        "within_edge_count": 1, "between_edge_count": 0,
        "zero_separation_between_edge_count": 0,
        "gradient_source": "observed_likelihood", "guide_adjustment_max_abs": 0.0,
        "max_within_cluster_deviation": 0.0,
        "block_flow_balance_max_abs": flow_balance,
        "max_dual_ball_ratio": 1.0, "max_within_dual_ball_ratio": 1.0,
        "max_between_dual_ball_ratio": 0.0,
        "kkt_residual": expected_kkt, "stationarity_residual": expected_kkt,
        "edge_subgradient_residual": 0.0, "dual_ball_residual": 0.0,
        "box_residual": 0.0, "num_exact_lower_active_coordinates": 0,
        "num_exact_upper_active_coordinates": 0, "num_exact_frozen_coordinates": 0,
    }
