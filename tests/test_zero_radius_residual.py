"""Zero-radius edges and invalid chunk diagnostics must never certify by NaN."""

from dataclasses import asdict
import math
import os

import pytest
import torch

from CliPP2.core.fusion import certificates, torch_backend as backend
from CliPP2.core.fusion.types import CompressedEdgeCertificate, TensorFusionGraph


def _problem(dtype, device="cpu", *, fused=False):
    phi = torch.tensor([[.2], [.2], [.2 if fused else .8]], dtype=dtype, device=device)
    edges = torch.triu_indices(3, 3, 1, device=device)
    weight = torch.tensor([0., 1., 1.], dtype=dtype, device=device)
    dual = torch.zeros((3, 1), dtype=dtype, device=device)
    if fused:
        dual[1] = 2.
    grad = -backend.graph_adjoint_edges(dual, edge_u=edges[0], edge_v=edges[1], num_nodes=3)
    kwargs = dict(phi=phi, grad_smooth=grad, lower=torch.zeros_like(phi),
                  upper=torch.ones_like(phi), lambda_value=1., atol=1e-6)
    return kwargs, edges, weight, dual


def _dense(kwargs, edges, weight, dual, chunk_edges):
    return backend.graph_fusion_kkt_residual_from_grad_torch(
        **kwargs, edge_u=edges[0], edge_v=edges[1], edge_w=weight, dual_kkt=dual,
        edge_work_bytes=chunk_edges * kwargs["phi"].element_size())


def _compressed(kwargs, edges, weight, dual):
    phi = kwargs["phi"]
    graph = TensorFusionGraph(edges, weight, torch.full_like(weight, 2.),
                              torch.ones_like(weight), 3, True, "zero_radius_fixture")
    certificate = CompressedEdgeCertificate(
        torch.zeros(3, dtype=torch.long, device=phi.device), phi[:1].clone(),
        torch.arange(3, device=phi.device), dual, "fixture", "observed_objective")
    return certificates._compressed_graph_fusion_kkt(
        **kwargs, certificate=certificate, graph=graph, graph_hash="fixture")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("dual_present", [False, True])
def test_zero_radius_fused_edge_is_exactly_zero(dtype, dual_present):
    diff = torch.zeros((1, 2), dtype=dtype)
    actual = backend.edge_kkt_maxima_from_diff_torch(
        diff=diff, dual=torch.zeros_like(diff) if dual_present else None,
        radius=torch.zeros(1, dtype=dtype))
    assert all(value.item() == 0. for value in actual)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("chunk_edges", [1, 2, 3])
@pytest.mark.parametrize("dual_present", [False, True])
def test_zero_weight_edge_does_not_hide_positive_edge_violation(dtype, chunk_edges, dual_present):
    kwargs, edges, weight, dual = _problem(dtype)
    diag = _dense(kwargs, edges, weight, dual if dual_present else None, chunk_edges)
    assert all(math.isfinite(value) for value in asdict(diag).values())
    assert diag.kkt_residual == pytest.approx(.3, abs=1e-7)
    assert diag.backward_error_kkt_residual == pytest.approx(.6, abs=1e-7)
    # The original U is not an optimum of this quadratic-plus-fusion problem.
    phi = kwargs["phi"]
    trial = phi + phi.new_tensor([[.01], [.01], [-.01]])
    def objective(x):
        return (.5 * (x - phi).square().sum()
            + (weight * torch.linalg.vector_norm(x[edges[0]] - x[edges[1]], dim=1)).sum())

    assert objective(trial) < objective(phi)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("chunk_edges", [1, 2])
def test_compressed_zero_radius_and_positive_violation_match_dense(monkeypatch, dtype, chunk_edges):
    kwargs, edges, weight, dual = _problem(dtype, fused=True)
    monkeypatch.setattr(certificates, "_compressed_edge_chunk_size", lambda **_: chunk_edges)
    compressed = _compressed(kwargs, edges, weight, dual)
    dense = _dense(kwargs, edges, weight, dual, chunk_edges)
    assert asdict(compressed) == pytest.approx(asdict(dense))
    assert compressed.kkt_residual == .5
    assert compressed.backward_error_kkt_residual == 1.


@pytest.mark.parametrize("route", ["dense", "compressed"])
@pytest.mark.parametrize("position", range(3))
@pytest.mark.parametrize("component", range(5))
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), -.1])
def test_chunk_aggregation_preserves_nonfinite_values(monkeypatch, route, position, component, invalid):
    kwargs, edges, weight, dual = _problem(torch.float64, fused=True)
    original = backend.edge_kkt_maxima_from_diff_torch
    calls = 0

    def injected(**values):
        nonlocal calls
        result = list(original(**values))
        if calls == position:
            result[component] = values["diff"].new_tensor(invalid)
        calls += 1
        return tuple(result)

    if route == "dense":
        monkeypatch.setattr(backend, "edge_kkt_maxima_from_diff_torch", injected)
        diag = _dense(kwargs, edges, weight, dual, 1)
    else:
        monkeypatch.setattr(certificates, "_compressed_edge_chunk_size", lambda **_: 1)
        monkeypatch.setattr(certificates, "edge_kkt_maxima_from_diff_torch", injected)
        diag = _compressed(kwargs, edges, weight, dual)
    assert calls == 3
    if component <= 2:
        assert diag.kkt_residual == math.inf
    if component >= 2:
        assert diag.backward_error_kkt_residual == math.inf


def test_omitted_edge_scan_cannot_hide_nan():
    kwargs, edges, weight, _ = _problem(torch.float64, fused=True)
    graph = TensorFusionGraph(edges, weight, torch.full_like(weight, 2.),
                              torch.ones_like(weight), 3, True, "fixture")
    residual = kwargs["phi"].clone()
    residual[0] = float("nan")
    maximum, _ = certificates._scan_omitted_internal_edges(
        residual=residual, labels=torch.zeros(3, dtype=torch.long),
        support_ids=torch.empty(0, dtype=torch.long), graph=graph, scale=1., add_batch=1)
    assert maximum == math.inf


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), -.1])
@pytest.mark.parametrize("first_invalid", [False, True])
def test_device_residual_maximum_rejects_invalidity_in_either_position(invalid, first_invalid):
    first, second = torch.tensor(0.), torch.tensor(invalid)
    if first_invalid:
        first, second = second, first
    result = backend._residual_maximum_torch(first, second).item()
    assert not math.isfinite(result)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("component", range(5))
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), -.1])
def test_streamed_alm_cannot_stop_on_invalid_chunk_diagnostics(monkeypatch, dtype, component, invalid):
    runtime = backend.resolve_runtime("cpu", dtype=dtype)
    kwargs, edges, weight, _ = _problem(runtime.dtype, fused=True)
    original = backend.edge_kkt_maxima_from_diff_torch

    def injected(**values):
        result = list(original(**values))
        result[component] = values["diff"].new_tensor(invalid)
        return tuple(result)

    monkeypatch.setattr(backend, "edge_kkt_maxima_from_diff_torch", injected)
    result = backend.solve_majorized_subproblem_alm_torch(
        runtime=runtime, num_mutations=3, U=kwargs["phi"], h=torch.ones_like(kwargs["phi"]),
        lower=kwargs["lower"], upper=kwargs["upper"], lambda_value=1.,
        edge_u=edges[0], edge_v=edges[1], edge_w=weight, tol=1e-6,
        max_iter=10, phi_start=kwargs["phi"], dual_start=None, edge_work_bytes=1,
        kkt_check_every=1, use_backward_error_stopping=component >= 2,
    )
    assert result[2] == 10 and not result[3]
    diag = result[4]
    if component <= 2:
        assert diag.kkt_residual == math.inf
    if component >= 2:
        assert diag.backward_error_kkt_residual == math.inf


@pytest.mark.skipif(os.environ.get("CLIPP2_TEST_CUDA") != "1",
                   reason="Enable only for explicit LSF CUDA qualification.")
def test_cuda_working_and_terminal_zero_radius_audits():
    assert torch.cuda.is_available()
    kwargs, edges, weight, dual = _problem(torch.float32, "cuda")
    working = _dense(kwargs, edges, weight, dual, 1)
    terminal_kwargs = {key: value.double() if torch.is_tensor(value) else value
                       for key, value in kwargs.items()}
    terminal = _dense(terminal_kwargs, edges, weight.double(), dual, 1)
    assert working.backward_error_kkt_residual == pytest.approx(.6, abs=1e-7)
    assert terminal.backward_error_kkt_residual == pytest.approx(.6, abs=1e-7)
    assert all(math.isfinite(value) for value in asdict(working).values())
    assert all(math.isfinite(value) for value in asdict(terminal).values())
