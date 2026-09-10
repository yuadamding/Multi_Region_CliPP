"""Persistent fit evidence is relocated intact, including independent witnesses."""

from dataclasses import fields, is_dataclass, replace
import os
import weakref

import numpy as np
import pytest
import torch

from CliPP2.core.fusion.types import (
    CompressedEdgeCertificate, DenseEdgeCertificate, DenseWarmState,
    PrimalOnlyWarmState, SolverState,
)
from CliPP2.model_selection.proposals import offload_raw_fit_to_cpu
from test_integer_reporting import _data, _selection


def _persistent_fit(kind, *, state_present=True, device="cpu", dtype=torch.float64):
    fit, _, _ = _selection(_data(major=(3, 2), minor=(1, 1), alt=(30, 25), total=(100, 100)))
    phi = torch.tensor(fit.phi.copy(), dtype=dtype, device=device)
    dual = torch.tensor([[.125]], dtype=dtype, device=device)
    graph_hash = fit.provenance.original_graph_hash
    if kind == "dense":
        witness = DenseEdgeCertificate(dual, graph_hash, "observed_objective")
        warm = DenseWarmState(phi, dual.detach(), .1, graph_hash)
    else:
        labels = torch.tensor([0, 0], device=device)
        witness = CompressedEdgeCertificate(
            labels, phi[:1], torch.tensor([0], device=device), dual,
            graph_hash, "observed_objective",
        )
        warm = PrimalOnlyWarmState(phi, labels.detach(), witness)
    state = SolverState(
        phi, dual.detach(), .1, warm_state=warm, certificate=witness,
        objective_spec_hash=fit.provenance.objective_key.base.fingerprint,
    ) if state_present else None
    return replace(
        fit, state=state, certificate=replace(fit.certificate, witness=witness),
        provenance=replace(fit.provenance, device=device, dtype=str(dtype).removeprefix("torch.")),
    )


def _payloads(value):
    if torch.is_tensor(value):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from _payloads(getattr(value, field.name))


def _assert_preserved(original, moved):
    assert type(original) is type(moved)
    if torch.is_tensor(original):
        assert moved.device.type == "cpu"
        assert original.dtype == moved.dtype and original.shape == moved.shape
        assert not moved.requires_grad
        torch.testing.assert_close(original.detach().cpu(), moved, rtol=0, atol=0)
    elif isinstance(original, np.ndarray):
        np.testing.assert_array_equal(original, moved)
        assert original.dtype == moved.dtype
    elif is_dataclass(original):
        for field in fields(original):
            _assert_preserved(getattr(original, field.name), getattr(moved, field.name))
    elif isinstance(original, float) and np.isnan(original):
        assert np.isnan(moved)
    else:
        assert original == moved


@pytest.mark.parametrize("kind", ["dense", "compressed"])
@pytest.mark.parametrize("state_present", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_whole_fit_offload_preserves_evidence_and_aliases(kind, state_present, dtype):
    fit = _persistent_fit(kind, state_present=state_present, dtype=dtype)
    moved = offload_raw_fit_to_cpu(fit)
    _assert_preserved(fit, moved)
    witness = moved.certificate.witness
    assert witness is not None
    if not state_present:
        assert moved.state is None
        return
    if kind == "dense":
        assert moved.state.dual is witness.dual
        assert moved.state.certificate.dual is witness.dual
        assert moved.state.warm_state.dual is witness.dual
    else:
        assert moved.state.dual is witness.internal_dual
        assert moved.state.certificate.internal_dual is witness.internal_dual
        assert moved.state.warm_state.certificate_hint.internal_dual is witness.internal_dual
        assert moved.state.warm_state.structure_hint is witness.labels
    assert moved.state.phi is moved.state.warm_state.phi


def test_absent_state_and_witness_remain_absent():
    fit, _, _ = _selection(_data())
    _assert_preserved(fit, offload_raw_fit_to_cpu(fit))


@pytest.mark.parametrize("kind", ["dense", "compressed"])
def test_shared_transfer_memo_releases_original_payloads(monkeypatch, kind):
    # Force distinct CPU buffers to exercise copy ownership without pretending
    # that this is a CUDA allocator/VRAM measurement.
    original_to = torch.Tensor.to

    def copying_to(tensor, *args, **kwargs):
        result = original_to(tensor, *args, **kwargs)
        return result.clone() if kwargs.get("device") == "cpu" else result

    monkeypatch.setattr(torch.Tensor, "to", copying_to)
    fit = _persistent_fit(kind)
    references = [weakref.ref(tensor) for tensor in _payloads(fit)]
    pointers = {tensor.untyped_storage().data_ptr() for tensor in _payloads(fit)}
    moved = offload_raw_fit_to_cpu(fit)
    _assert_preserved(fit, moved)
    assert pointers.isdisjoint(tensor.untyped_storage().data_ptr() for tensor in _payloads(moved))
    del fit
    assert all(reference() is None for reference in references)


@pytest.mark.skipif(
    os.environ.get("CLIPP2_TEST_CUDA") != "1",
    reason="Enable CLIPP2_TEST_CUDA=1 only for explicit LSF CUDA qualification.",
)
@pytest.mark.parametrize("kind", ["dense", "compressed"])
@pytest.mark.parametrize("state_present", [False, True])
def test_cuda_persistent_history_is_entirely_host_backed(kind, state_present):
    assert torch.cuda.is_available(), "Explicit CUDA qualification requires a CUDA runtime."
    history = []
    original_payloads = []
    for dtype in (torch.float32, torch.float64):
        fit = _persistent_fit(kind, state_present=state_present, device="cuda", dtype=dtype)
        original_payloads.extend(weakref.ref(tensor) for tensor in _payloads(fit))
        moved = offload_raw_fit_to_cpu(fit)
        _assert_preserved(fit, moved)
        assert moved.provenance.device == "cuda"
        history.append(moved)
        del fit
    torch.cuda.synchronize()
    assert all(reference() is None for reference in original_payloads)
    assert all(tensor.device.type == "cpu" for fit in history for tensor in _payloads(fit))
