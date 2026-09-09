"""Clonal-CN simulation matches whole-mutation eligibility without filtering truth."""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from CliPP2.io.multiplicity import MAX_MAJOR_CN
from CliPP2.io.tumor_txt import CN_FILTER_POLICY_ID, load_tumor_txt
from tools.simulation import (
    CopyNumberEvolutionConfig,
    TumorSimulationConfig,
    simulate_tumor,
)
from tools.simulation.config import GENERATOR_VERSION, OUTPUT_SCHEMA_VERSION
from tools.simulation.evolution import (
    CNAEvent,
    simulate_branch_cna_events,
    simulate_joint_snv_cna_evolution,
)


def test_trunk_sampler_keeps_repeated_gains_without_descendant_events():
    events = simulate_branch_cna_events(
        np.array([-1, 0, 1]),
        CopyNumberEvolutionConfig(n_segments=2, cna_event_rate=20),
        random_state=3,
    )
    assert len(events) > 2
    assert all(event.clone_id == 0 and event.parent_clone_id == -1 for event in events)
    assert len({event.event_id for event in events}) == len(events)
    assert (
        simulate_branch_cna_events(
            np.array([-1]), CopyNumberEvolutionConfig(cna_event_rate=0), random_state=3
        )
        == []
    )


def test_trunk_gain_timing_can_generate_multiplicity_six_and_postgain_one():
    evolution = simulate_joint_snv_cna_evolution(
        parent=np.array([-1, 0, 1]),
        mutation_origin_clone=np.array([0, 0, 1]),
        mutation_segment=np.zeros(3, dtype=int),
        branch_cna_events=[CNAEvent(i, 0, -1, (i + 1) / 10, (0,), 0) for i in range(8)],
        n_segments=1,
        max_allele_cn=MAX_MAJOR_CN,
        mutation_branch_time=np.array([0.0, 0.95, 0.5]),
        mutation_origin_allele=np.zeros(3, dtype=int),
        random_state=4,
    )
    np.testing.assert_array_equal(evolution.clone_allele_cn, [[[6, 1]]] * 3)
    np.testing.assert_array_equal(
        evolution.mutation_dosage_numeric, [[6, 6, 6], [1, 1, 1], [0, 1, 1]]
    )
    assert len(evolution.cna_event_history) == 5


@pytest.mark.parametrize("clone_count,cna_rate", [(3, 1.5), (3, 0), (1, 20)])
def test_generated_bundle_retains_exact_size_and_integer_truth(
    tmp_path, clone_count, cna_rate
):
    directory = simulate_tumor(
        TumorSimulationConfig(
            out_dir=tmp_path,
            tumor_id="matched",
            mutation_count=60,
            clone_count=clone_count,
            copy_number=CopyNumberEvolutionConfig(
                n_segments=3, cna_event_rate=cna_rate
            ),
        )
    )
    data = load_tumor_txt(directory / "matched.clipp2.txt")
    assert data.num_mutations == 60
    assert data.cn_filter_report.input_mutation_count == 60
    assert data.cn_filter_report.retained_mutation_count == 60
    assert not data.cn_filter_report.excluded_mutation_ids
    canonical = pd.read_csv(directory / "matched.clipp2.txt", sep="\t", comment="#")
    assert len(canonical) == 120
    assert canonical.groupby(["mutation_id", "sample_id"]).size().eq(1).all()
    assert canonical.allele_a_cn.between(1, MAX_MAJOR_CN).all()
    np.testing.assert_allclose(canonical.cn_state_fraction, 1, atol=1e-8)
    truth = pd.read_csv(directory / "truth_mutation_sample.tsv", sep="\t")
    truth["sample_id"] = "region" + (truth.sample_id + 1).astype(str)
    paired = canonical.merge(
        truth, on=["mutation_id", "sample_id"], validate="one_to_one"
    )
    assert len(paired) == len(canonical)
    assert paired.multiplicity.between(1, paired.allele_a_cn).all()
    np.testing.assert_allclose(
        paired.effective_multiplicity, paired.multiplicity, atol=1e-8
    )
    np.testing.assert_allclose(
        paired.mutant_copy_mass, paired.ccf * paired.multiplicity
    )
    total = paired.allele_a_cn + paired.allele_b_cn
    denominator = paired.purity * total + (1 - paired.purity) * paired.normal_cn
    np.testing.assert_allclose(
        paired.expected_vaf,
        paired.purity * paired.ccf * paired.multiplicity / denominator,
    )
    if cna_rate == 0:
        assert paired.allele_a_cn.eq(1).all()
        assert paired.multiplicity.eq(1).all()
    if cna_rate == 20:
        assert paired.allele_a_cn.eq(MAX_MAJOR_CN).all()
    manifest = json.loads((directory / "scenario_manifest.json").read_text())
    assert manifest["generator_version"] == GENERATOR_VERSION
    assert manifest["output_schema_version"] == OUTPUT_SCHEMA_VERSION
    assert manifest["intended_factors"]["cn_filter_policy_id"] == CN_FILTER_POLICY_ID
    assert manifest["realized_factors"]["retained_mutation_count"] == 60
    assert manifest["realized_factors"]["excluded_mutation_count"] == 0
    assert manifest["rejection_counts"]["copy_number"] == 0


def test_seed_reproducibility_and_depth_independent_truth(tmp_path):
    config = TumorSimulationConfig(
        out_dir=tmp_path / "first", mutation_count=60, tumor_id="matched"
    )
    first = simulate_tumor(config)
    repeated = simulate_tumor(replace(config, out_dir=tmp_path / "repeated"))
    deep = simulate_tumor(replace(config, out_dir=tmp_path / "deep", mean_depth=200))
    for path in first.rglob("truth*"):
        relative = path.relative_to(first)
        assert path.read_bytes() == (repeated / relative).read_bytes()
        assert path.read_bytes() == (deep / relative).read_bytes()
    assert (first / "matched.clipp2.txt").read_bytes() == (
        repeated / "matched.clipp2.txt"
    ).read_bytes()
    assert (first / "matched.clipp2.txt").read_bytes() != (
        deep / "matched.clipp2.txt"
    ).read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        simulate_tumor(config)


@pytest.mark.parametrize(
    "name,value",
    [
        ("max_allele_cn", 7),
        ("max_allele_cn", 0),
        ("max_allele_cn", 2.5),
        ("max_allele_cn", np.inf),
        ("max_allele_cn", np.nan),
        ("max_allele_cn", True),
        ("n_segments", 2.5),
        ("segment_size_bp", 0),
        ("cna_event_rate", -1),
        ("cna_event_rate", np.nan),
        ("cna_event_rate", np.inf),
        ("mean_cna_span_segments", 0.5),
        ("mean_cna_span_segments", np.nan),
    ],
)
def test_invalid_cn_config_fails_before_creating_output(tmp_path, name, value):
    with pytest.raises(ValueError, match=name):
        simulate_tumor(
            TumorSimulationConfig(
                out_dir=tmp_path,
                copy_number=replace(CopyNumberEvolutionConfig(), **{name: value}),
            )
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name", ["clone_count", "mutation_count", "seed"])
def test_noninteger_dimensions_are_not_silently_truncated(tmp_path, name):
    with pytest.raises(ValueError, match=name):
        simulate_tumor(replace(TumorSimulationConfig(out_dir=tmp_path), **{name: 2.5}))
    assert not list(tmp_path.iterdir())
