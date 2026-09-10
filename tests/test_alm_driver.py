"""One ALM controller retains full/chunked arithmetic and actual-dual meaning."""
from itertools import product
import os

import numpy as np
import pytest
import torch

from CliPP2.core.fusion import solver, torch_backend as backend
from CliPP2.core.fusion.types import KKTDiagnostics


def _problem(dtype="float64", *, chunk_edges=100, regions=2, device="cpu"):
    runtime = backend.resolve_runtime(device, dtype=dtype)
    U = torch.tensor([[.1, .9], [.2, .8], [.55, .45], [.7, .3], [.9, .1]],
                     dtype=runtime.dtype, device=runtime.device)[:, :regions]
    h = torch.full_like(U, 4.)
    edge_u, edge_v = torch.triu_indices(5, 5, 1, device=runtime.device)
    return dict(
        runtime=runtime, num_mutations=5, U=U, h=h,
        lower=torch.zeros_like(U), upper=torch.ones_like(U),
        lambda_value=1.7, edge_u=edge_u, edge_v=edge_v,
        edge_w=torch.linspace(.13, .67, 10, dtype=runtime.dtype, device=runtime.device),
        tol=1e-7, max_iter=64, phi_start=U, dual_start=None,
        edge_work_bytes=chunk_edges * regions * U.element_size(), kkt_check_every=3,
    )


def _box_oracle(matrix, rhs, lower, upper):
    """Independent exhaustive active-set solution of a tiny strictly convex QP."""
    for pattern in product((0, 1, 2), repeat=len(rhs)):
        pattern = np.asarray(pattern)
        free, lower_active, upper_active = pattern == 0, pattern == 1, pattern == 2
        value = np.where(lower_active, lower, upper)
        value[free] = np.linalg.solve(
            matrix[np.ix_(free, free)],
            rhs[free] - matrix[np.ix_(free, ~free)] @ value[~free],
        )
        if np.any(value < lower - 1e-12) or np.any(value > upper + 1e-12):
            continue
        gradient = matrix @ value - rhs
        if (np.all(gradient[lower_active] >= -1e-12)
            and np.all(gradient[upper_active] <= 1e-12)
            and np.all(np.abs(gradient[free]) < 1e-12)):
            return value
    raise AssertionError("No feasible stationary point found by independent box oracle")


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("chunk_edges", [1, 100])
def test_two_complete_edge_phases_match_independent_box_admm(monkeypatch, dtype, chunk_edges):
    kwargs = _problem(dtype, chunk_edges=chunk_edges, regions=1)
    runtime = kwargs["runtime"]
    kwargs.update(
        num_mutations=3, U=torch.tensor([[-.2], [.45], [1.2]], dtype=runtime.dtype),
        h=torch.tensor([[2.], [5.], [3.]], dtype=runtime.dtype),
        lower=torch.tensor([[.1], [.2], [.3]], dtype=runtime.dtype),
        upper=torch.tensor([[.6], [.65], [.9]], dtype=runtime.dtype),
        phi_start=torch.tensor([[.2], [.4], [.8]], dtype=runtime.dtype),
        edge_u=torch.tensor([0, 0, 1]), edge_v=torch.tensor([1, 2, 2]),
        edge_w=torch.tensor([.3, .7, 1.1], dtype=runtime.dtype),
        lambda_value=.7, max_iter=10, tol=1e-12,
    )
    original_box = backend._complete_graph_isotropic_box_qp_torch
    node_steps = []

    def box(**values):
        result = original_box(**values)
        node_steps.append((values["q"].clone(), result.clone(), float(values["rho"])))
        return result

    monkeypatch.setattr(backend, "_complete_graph_isotropic_box_qp_torch", box)
    phi, actual_dual, iterations, _, diagnostics = backend.solve_majorized_subproblem_alm_torch(**kwargs)
    assert iterations == len(node_steps) == 10
    assert isinstance(diagnostics, KKTDiagnostics)
    # Reproduce every node update using a separately written dense incidence
    # formulation and exhaustive active-set box solution, never solver helpers.
    incidence = np.array([[1., -1., 0.], [1., 0., -1.], [0., 1., -1.]])
    U, h = kwargs["U"].numpy().ravel().astype(float), kwargs["h"].numpy().ravel().astype(float)
    lower = kwargs["lower"].numpy().ravel().astype(float)
    upper = kwargs["upper"].numpy().ravel().astype(float)
    current = kwargs["phi_start"].numpy().ravel().astype(float)
    radius = .7 * kwargs["edge_w"].numpy().astype(float)
    scaled_dual = np.zeros(3)
    rho = 3.
    matrix = np.diag(h) + rho * incidence.T @ incidence
    tolerance = 8e-7 if dtype == "float32" else 2e-12
    for actual_q, actual_phi, used_rho in node_steps:
        z_argument = incidence @ current + scaled_dual
        z = z_argument * np.maximum(1 - radius / rho / np.maximum(np.abs(z_argument), 1e-12), 0)
        q = incidence.T @ (z - scaled_dual)
        current = _box_oracle(matrix, h * U + rho * q, lower, upper)
        scaled_dual += incidence @ current - z
        assert used_rho == rho
        np.testing.assert_allclose(actual_q.numpy().ravel(), q, rtol=0, atol=tolerance)
        np.testing.assert_allclose(actual_phi.numpy().ravel(), current, rtol=0, atol=tolerance)
    np.testing.assert_allclose(phi.numpy().ravel(), current, rtol=0, atol=tolerance)
    np.testing.assert_allclose(actual_dual.numpy().ravel(), rho * scaled_dual, rtol=0, atol=tolerance)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("chunk_edges", [1, 100])
def test_scaled_and_actual_warm_starts_keep_the_same_multiplier(dtype, chunk_edges):
    kwargs = _problem(dtype, chunk_edges=chunk_edges)
    initial_scaled = torch.linspace(-.01, .01, 20, dtype=kwargs["runtime"].dtype).reshape(10, 2)
    initial_actual = 4. * initial_scaled
    kwargs.update(dual_start=initial_scaled, dual_start_is_actual=False)
    scaled_result = backend.solve_majorized_subproblem_alm_torch(**kwargs)
    kwargs.update(dual_start=initial_actual, dual_start_is_actual=True)
    actual_result = backend.solve_majorized_subproblem_alm_torch(**kwargs)
    assert len(actual_result) == len(scaled_result) == 5
    for actual, expected in zip(actual_result[:2], scaled_result[:2]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual_result[2:] == scaled_result[2:]
    torch.testing.assert_close(initial_actual, 4. * initial_scaled, rtol=0, atol=0)
    # Carry only actual y into another inner problem whose internal rho differs.
    next_actual = backend.project_dual_ball(actual_result[1], 1.7 * kwargs["edge_w"])
    kwargs["h"] = torch.full_like(kwargs["h"], 8.)
    kwargs.update(phi_start=actual_result[0], dual_start=next_actual, dual_start_is_actual=True)
    continued_actual = backend.solve_majorized_subproblem_alm_torch(**kwargs)
    kwargs.update(dual_start=next_actual / 8., dual_start_is_actual=False)
    continued_scaled = backend.solve_majorized_subproblem_alm_torch(**kwargs)
    for actual, expected in zip(continued_actual[:2], continued_scaled[:2]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert continued_actual[2:] == continued_scaled[2:]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("chunk_edges", [1, 100])
def test_spectral_rho_changes_preserve_actual_dual_and_pinned_schedule(monkeypatch, dtype, chunk_edges):
    kwargs = _problem(dtype, chunk_edges=chunk_edges)
    kwargs.update(h=torch.full_like(kwargs["h"], .02), lambda_value=20., spectral_rho=True)
    rho_sequence, rescalings = [], []
    original_box = backend._complete_graph_isotropic_box_qp_torch
    original_mul, original_imul = torch.Tensor.__mul__, torch.Tensor.mul_

    def box(**values):
        rho_sequence.append(float(values["rho"]))
        return original_box(**values)

    def record(original, tensor, factor):
        rescaling = isinstance(factor, float) and factor in (.5, 2.) and tensor.shape == (10, 2)
        before = tensor.clone() if rescaling else None
        result = original(tensor, factor)
        if rescaling:
            old_rho = rho_sequence[-1]
            rescalings.append((old_rho * before, (old_rho / factor) * result))
        return result

    monkeypatch.setattr(backend, "_complete_graph_isotropic_box_qp_torch", box)
    monkeypatch.setattr(torch.Tensor, "__mul__", lambda value, factor: record(original_mul, value, factor))
    monkeypatch.setattr(torch.Tensor, "mul_", lambda value, factor: record(original_imul, value, factor))
    counters = {}
    result = backend.solve_majorized_subproblem_alm_torch(**kwargs, diagnostics_out=counters)
    assert rescalings
    for actual_before, actual_after in rescalings:
        torch.testing.assert_close(actual_before, actual_after, rtol=0, atol=0)
    changes = [(i + 1, rho) for i, rho in enumerate(rho_sequence) if i == 0 or rho != rho_sequence[i - 1]]
    assert changes[0][0] == 1 and changes[1][0] == 11
    assert changes[1][1] == 2 * changes[0][1]
    if dtype == "float64":
        # Captured from the corresponding pinned aa799344 routes before merge.
        assert changes == [(1, .004), (11, .008)]
        assert result[2:4] == (18, True)
        assert counters == {"inner_kkt_audits": 6, "inner_stationarity_checks": 6}
        np.testing.assert_allclose(result[0].numpy(), [[.489999942100029, .510000057899971],
            [.48999995698438353, .5100000430156165], [.4900000090796244, .5099999909203757],
            [.49000003140615617, .5099999685938439], [.49000006117486516, .5099999388251348]],
            rtol=0, atol=2e-16)


@pytest.mark.parametrize("budget,expected_full_calls", [(1, 0), (16, 20)])
def test_sub_edge_work_budget_retains_streamed_route(monkeypatch, budget, expected_full_calls):
    kwargs = _problem()
    for name in ("U", "h", "lower", "upper", "phi_start"):
        kwargs[name] = kwargs[name][:2]
    kwargs.update(num_mutations=2, edge_u=torch.tensor([0]), edge_v=torch.tensor([1]),
                  edge_w=torch.ones(1, dtype=torch.float64), edge_work_bytes=budget,
                  max_iter=10, tol=1e-12, kkt_check_every=1)
    calls = []
    original = backend.graph_adjoint_edges

    def adjoint(*args, **options):
        calls.append(options.get("prefer_cpu_bincount"))
        return original(*args, **options)

    monkeypatch.setattr(backend, "graph_adjoint_edges", adjoint)
    result = backend.solve_majorized_subproblem_alm_torch(**kwargs)
    assert result[2] == 10
    # One edge still selects the streamed route if its work exceeds the byte
    # budget, even though the effective minimum chunk also contains one edge.
    assert len(calls) == expected_full_calls
    assert all(calls)


@pytest.mark.parametrize("no_edges", [False, True])
def test_closed_form_and_invalid_budget_contract(no_edges):
    kwargs = _problem()
    if no_edges:
        for name in ("edge_u", "edge_v", "edge_w"):
            kwargs[name] = kwargs[name][:0]
    else:
        kwargs["lambda_value"] = 0.
    phi, dual, iterations, converged, diagnostics = backend.solve_majorized_subproblem_alm_torch(**kwargs)
    assert dual.shape == (0, 2) and iterations == 0 and converged
    assert isinstance(diagnostics, KKTDiagnostics)
    torch.testing.assert_close(phi, kwargs["U"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="edge work budget"):
        backend.solve_majorized_subproblem_alm_torch(**(kwargs | {"edge_work_bytes": 0}))


def test_outer_inner_boundary_reuses_one_actual_dual(monkeypatch):
    kwargs = _problem()
    marker = torch.zeros((10, 2), dtype=torch.float64)
    diag = KKTDiagnostics(0., 0., 0., 0., 0., 0., 0.)
    def inner(**values):
        return values["phi_start"], marker, 9, True, diag
    monkeypatch.setattr(solver, "solve_majorized_subproblem_alm_torch", inner)
    result = solver._solve_inner_subproblem(
        use_alm=True, runtime=kwargs["runtime"], num_mutations=5, U=kwargs["U"], h=kwargs["h"],
        lower=kwargs["lower"], upper=kwargs["upper"], lambda_value=1.7,
        edge_u=kwargs["edge_u"], edge_v=kwargs["edge_v"], edge_w=kwargs["edge_w"],
        degree_bound=4, tol=1e-7, inner_max_iter=10, phi=kwargs["phi_start"], dual=None,
        dual_start_is_actual=True, spectral_rho=False, use_backward_error_stopping=False,
        pdhg_tau_node=None, backend_name="test", graph_hash="frozen",
    )
    assert result.warm_state.dual is result.surrogate_certificate.dual is marker
    assert result.surrogate_kkt is diag and result.iterations == 9


@pytest.mark.skipif(os.environ.get("CLIPP2_TEST_CUDA") != "1",
                    reason="Explicit commit-pinned Seadragon LSF CUDA qualification only.")
@pytest.mark.parametrize("dtype,regions", [("float32", 1), ("float64", 2)])
@pytest.mark.parametrize("chunk_edges", [1, 100])
def test_cuda_keeps_original_adjoint_and_compiled_box_routes(monkeypatch, dtype, regions, chunk_edges):
    assert torch.cuda.is_available(), "Explicit CUDA qualification requires CUDA."
    kwargs = _problem(dtype, chunk_edges=chunk_edges, regions=regions, device="cuda")
    kwargs.update(max_iter=10, tol=1e-12)
    adjoints, compiled = [], []
    original_adjoint = backend.graph_adjoint_edges
    original_compiled = backend._complete_graph_isotropic_box_qp_cuda
    def adjoint(*args, **options):
        adjoints.append(True)
        return original_adjoint(*args, **options)
    def box(*args):
        compiled.append(True)
        return original_compiled(*args)
    monkeypatch.setattr(backend, "graph_adjoint_edges", adjoint)
    monkeypatch.setattr(backend, "_complete_graph_isotropic_box_qp_cuda", box)
    result = backend.solve_majorized_subproblem_alm_torch(**kwargs)
    assert bool(adjoints) is (chunk_edges == 100)
    assert len(compiled) == (result[2] if dtype == "float64" and regions > 1 else 0)
    assert torch.isfinite(result[0]).all() and torch.isfinite(result[1]).all()
