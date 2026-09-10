"""Search changes numerical effort through one configuration owner."""
from dataclasses import replace

import pytest

from CliPP2.config import resolve_fit_config
from CliPP2.model_selection.proposals import solver_retry_fit_options
from test_integer_likelihood import integer_data


@pytest.mark.parametrize("profile", ["balanced", "strict", "fast"])
def test_retry_options_preserve_fixed_objective_and_refit_contract(profile):
    data = integer_data(((2,), (6,)))
    options = resolve_fit_config(computation_profile=profile, device="cpu", dtype="float64")
    assert solver_retry_fit_options(data, options, retry_number=0,
                                    certification_recovery=False) is options
    retry = solver_retry_fit_options(data, options, retry_number=2,
                                    certification_recovery=False)
    expected_solver = replace(options.solver,
        outer_max_iter=3 * options.solver.outer_max_iter,
        inner_max_iter=3 * options.solver.inner_max_iter)
    assert retry == replace(options, solver=expected_solver)
    recovery = solver_retry_fit_options(data, options, retry_number=9,
                                       certification_recovery=True)
    assert recovery.selection is options.selection
    assert recovery.graph is options.graph
    assert recovery.eps == options.eps
    assert recovery.runtime is options.runtime
    assert recovery.solver.outer_max_iter == max(options.solver.outer_max_iter * 24, 144)
    assert recovery.solver.inner_max_iter == max(options.solver.inner_max_iter * 6, 150)
    assert recovery.solver.tolerance == min(options.solver.tolerance, 5e-5)
    assert recovery.solver.certification_tolerance == (
        options.solver.tolerance if options.solver.certification_tolerance is None
        else options.solver.certification_tolerance)
    assert recovery.solver.objective_shape == "generic_nonconvex"
