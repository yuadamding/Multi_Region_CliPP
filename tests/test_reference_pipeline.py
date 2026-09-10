"""CPU integration goldens from cc5a3d1ac28097c2b3005c1c2615930f0ab424de.

Captured in ml1 with balanced settings, float64, one Torch/BLAS thread, and
the exact synthetic inputs below. This is numerical regression evidence, not
CUDA/cohort qualification. The heterogeneous CN6 full hybrid run was also
paired manually; its slower search is not repeated in the default test suite.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from CliPP2.api import fit_fixed_objective, process_tumor_bundle
from CliPP2.config import resolve_fit_config
from CliPP2.core.fusion.graph import build_complete_uniform_graph
from CliPP2.core.objective import infer_integer_multiplicity_posterior_numpy
from CliPP2.io.tumor_txt import load_tumor_txt, write_tumor_txt
from CliPP2.model_selection.partitions import extract_certified_fusion_partition
from CliPP2.model_selection import search as search_module


UNITS = {
    "cn1": [(1, 1, 12), (1, 1, 13), (1, 1, 35)],
    "cn2": [(2, 1, 10), (2, 1, 11), (2, 1, 30), (2, 1, 31)],
    "cn6": [(6, 1, 7), (6, 1, 8), (6, 2, 22), (6, 2, 23), (6, 3, 40), (6, 3, 41)],
    "cn6_equal": [(6, 1, 24)] * 6,
}
# Objective and raw CCFs at lambda 0.1 on the complete uniform graph.
FIXED = {
    "cn1": (124.43988946279813, [0.3759919793375991, 0.40625452745043833, 1.0]),
    "cn2": (171.09238563718532, [0.23710199127021275, 0.2553241777264417,
                                0.6560691580626586, 0.677301402584129]),
    "cn6": (261.903082797596, [0.14882691568856868, 0.1672066703737848,
                              0.46045050777387714, 0.47780792653113324,
                              0.8570271542644695, 0.8732818451423033]),
    "cn6_equal": (299.68838908555114, [0.43792433412772525] * 6),
}
GRAPH_HASH = {
    3: "c30da6d6444efa71f6f292a6f691244c9a84247140942931e2d72f9a6f3a7723",
    4: "afb0c4972c42f34dd2c22b99302186dde3138341d664f07ce351dbbd094ac0ff",
    6: "3fd6766b9b161e2b1e3929e8389f14f4df94f61c9642cdfd26d097df10997529",
}
# Immutable selected labels, public fixed-label CCFs, and Dirichlet score.
SELECTED = {
    "cn1": ([0, 0, 1], [0.3908736249999999, 0.3908736249999999, 1.0], 253.50799665644035),
    "cn2": ([0] * 4, [0.6091273749999999] * 4, 346.9828901250838),
    "cn6_equal": ([0] * 6, [0.4384926249999999] * 6, 601.1686536330919),
}
POSTERIOR = {
    "cn1": [[1.0]] * 3,
    "cn2": [
        [0.9999343937281712, 6.560627182886569e-05],
        [0.9998337761054343, 0.0001662238945656684],
        [0.0001279777943267617, 0.9998720222056732],
        [5.05098955748974e-05, 0.999949490104425],
    ],
    "cn6_equal": [[7.152719852642593e-11, 3.311739778345575e-05,
                    0.012030127706894827, 0.1950991278682821,
                    0.4847358009597703, 0.30810182599574204]] * 6,
}
TABLE_SCHEMAS = {
    "cluster_centers.tsv": ["tumor_id", "cluster_label", "cluster_size", "phi_R1"],
    "mutation_clusters.tsv": ["tumor_id", "mutation_id", "cluster_label", "phi_R1"],
    "excluded_mutations.tsv": ["tumor_id", "mutation_id", "sample_id", "segment_id",
                               "reason", "n_distinct_cn_states", "max_major_cn"],
    "mutation_region_multiplicity.tsv": [
        "tumor_id", "mutation_id", "region_id", "cluster_label", "phi", "major_cn", "minor_cn",
        "multiplicity_candidates", "multiplicity_candidate_count", "multiplicity_call",
        "multiplicity_call_probability", "multiplicity_informative",
        *[f"multiplicity_p{m}" for m in range(1, 7)],
    ],
}


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _input(tmp_path, name):
    path = tmp_path / f"{name}.tsv"
    write_tumor_txt(path, pd.DataFrame([
        dict(mutation_id=f"m{index}", sample_id="R1", alt_count=alt, ref_count=80-alt,
             count_observed=1, purity=0.8, normal_cn=2, segment_id=f"s{index}",
             cn_state_id="clonal", cn_state_fraction=1, allele_a_cn=major, allele_b_cn=minor)
        for index, (major, minor, alt) in enumerate(UNITS[name])
    ]))
    return path


def _config(**kwargs):
    return resolve_fit_config(
        computation_profile="balanced", device="cpu", dtype="float64",
        outer_max_iter=20, inner_max_iter=80, certificate_max_iter=100,
        certificate_refinement_rounds=1, **kwargs,
    )


@pytest.mark.parametrize("name", UNITS)
def test_fixed_lambda_matches_reference_objective_phi_labels_and_graph(tmp_path, name):
    data = load_tumor_txt(_input(tmp_path, name))
    graph = build_complete_uniform_graph(data.num_mutations)
    config = _config(lambda_value=0.1, graph=graph)
    fit = fit_fixed_objective(data, config)
    objective, phi = FIXED[name]
    np.testing.assert_allclose(fit.objective.total, objective, rtol=0, atol=2e-10)
    np.testing.assert_allclose(fit.phi[:, 0], phi, rtol=0, atol=1e-8)
    assert fit.provenance.original_graph_hash == GRAPH_HASH[data.num_mutations]
    partition = extract_certified_fusion_partition(
        fit, graph=graph, tolerance=config.selection.partition_tolerance,
        mutation_ids=tuple(data.mutation_ids),
    )
    expected = [0] * data.num_mutations if name == "cn6_equal" else list(range(data.num_mutations))
    np.testing.assert_array_equal(partition.labels, expected)
    assert fit.certificate.certified
    assert fit.certificate.audit_dtype == "float64"
    assert fit.certificate.tolerance == 0.004
    assert fit.certificate.components.residual <= 0.004


@pytest.mark.parametrize("name", SELECTED)
def test_hybrid_matches_reference_labels_ccfs_score_posterior_and_tsv_schemas(tmp_path, name, monkeypatch):
    path = _input(tmp_path, name)
    data = load_tumor_txt(path)
    output = tmp_path / "output"
    build_graph = search_module._build_partition_guided_graph_with_resource_policy
    captured = []

    def capture_graph(*args, **kwargs):
        graph, tensor, tau = build_graph(*args, **kwargs)
        captured.append((graph, tau))
        return graph, tensor, tau

    monkeypatch.setattr(search_module, "_build_partition_guided_graph_with_resource_policy", capture_graph)
    summary, search = process_tumor_bundle(path, output, fit_config=_config())
    assert len(captured) == 1
    if name == "cn2":
        graph, tau = captured[0]
        np.testing.assert_array_equal(graph.edge_u, [0, 0, 0, 1, 1, 2])
        np.testing.assert_array_equal(graph.edge_v, [1, 2, 3, 2, 3, 3])
        # Baseline/current weights differ by at most 9.3e-13 after scalar
        # initialization contraction. Therefore their exact hashes differ;
        # freeze each runtime graph, but compare cross-version weights directly.
        np.testing.assert_allclose(graph.edge_w, [
            0.8362834738886106, 0.08207003220914953, 0.0780132805538162,
            0.08589143490846102, 0.08145830455135218, 0.8362834738886106,
        ], rtol=0, atol=1e-10)
        np.testing.assert_allclose(tau, 0.04122141582553202, rtol=0, atol=1e-10)
    candidate = next(item.candidate for item in search if item.selected)
    labels, phi, score = SELECTED[name]
    np.testing.assert_array_equal(candidate.partition.labels, labels)
    np.testing.assert_allclose(candidate.refit.phi[:, 0], phi, rtol=0, atol=1e-10)
    np.testing.assert_allclose(candidate.score.value, score, rtol=0, atol=2e-10)
    assert summary["selected_candidate_family"] == "raw_fusion"
    assert summary["raw_reference_objective_certified"]
    phases = {item.trace.search_phase for item in search}
    assert {"pilot_direct_partition_pool", "final_phi_direct_partition_pool"} <= phases
    posterior = infer_integer_multiplicity_posterior_numpy(data, candidate.refit.phi, eps=1e-6)
    expected = np.asarray(POSTERIOR[name])
    np.testing.assert_allclose(posterior.posterior[:, 0], expected, rtol=1e-9, atol=2e-11)
    np.testing.assert_array_equal(posterior.multiplicity_call[:, 0], 1 + expected.argmax(axis=-1))
    schemas = {p.name.removeprefix(name+"_"): pd.read_csv(p, sep="\t", nrows=0).columns.tolist()
               for p in output.glob("*.tsv")}
    assert schemas == TABLE_SCHEMAS
    # The four statistical tables are unchanged; run identity is now persisted
    # separately and is not part of the old TSV contract.
    assert (output / f"{name}_run_manifest.json").is_file()
