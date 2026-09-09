"""Generated benchmark truth must survive the public whole-mutation filter."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from CliPP2.cli import parse_args
from CliPP2.io.tumor_txt import load_tumor_txt, write_tumor_txt
from tools.simulation import (
    CopyNumberEvolutionConfig,
    TumorSimulationConfig,
    simulate_tumor,
    validate_generated_tumor_directory,
)
from tools.simulation.cli import build_parser, tumor_simulation_config_from_args


@pytest.fixture
def bundle(tmp_path):
    return simulate_tumor(TumorSimulationConfig(
        out_dir=tmp_path, tumor_id="checked", mutation_count=30,
        clone_count=2, seed=17,
        copy_number=CopyNumberEvolutionConfig(n_segments=3, cna_event_rate=3),
    ))


def test_bundle_roundtrips_with_versioned_manifest(bundle):
    validate_generated_tumor_directory(bundle)
    data = load_tumor_txt(bundle / "checked.clipp2.txt")
    assert data.num_mutations == 30
    assert not data.cn_filter_report.excluded_mutation_ids


@pytest.mark.parametrize("reason", ["high_major", "subclonal"])
def test_bundle_rejects_one_region_triggering_whole_mutation_exclusion(bundle, reason):
    path = bundle / "checked.clipp2.txt"
    table = pd.read_csv(path, sep="\t", comment="#")
    table.loc[0, "segment_id"] = 999
    if reason == "high_major":
        table.loc[0, "allele_a_cn"] = 7
    else:
        table.loc[0, ["cn_state_fraction", "allele_a_cn", "allele_b_cn"]] = [.5, 2, 1]
        extra = table.iloc[[0]].copy()
        extra["cn_state_id"] = "second"
        extra["allele_a_cn"] = 3
        table = pd.concat([table, extra], ignore_index=True)
    write_tumor_txt(path, table)
    assert load_tumor_txt(path).num_mutations == 29
    with pytest.raises(ValueError, match="retain every mutation"):
        validate_generated_tumor_directory(bundle)


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra_id", "noninteger", "vaf"])
def test_bundle_rejects_inconsistent_mutation_sample_truth(bundle, change):
    path = bundle / "truth_mutation_sample.tsv"
    table = pd.read_csv(path, sep="\t")
    if change == "missing":
        table = table.iloc[1:]
    elif change == "duplicate":
        table = pd.concat([table.iloc[1:], table.iloc[[1]]], ignore_index=True)
    elif change == "extra_id":
        table.loc[0, "mutation_id"] = "not_in_input"
    elif change == "noninteger":
        table["multiplicity"] = table.multiplicity.astype(float)
        table.loc[0, "multiplicity"] = 1.5
    else:
        table.loc[0, "expected_vaf"] += .1
    table.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="truth|Truth"):
        validate_generated_tumor_directory(bundle)


def test_bundle_rejects_missing_clustering_truth(bundle):
    path = bundle / "truth.txt"
    pd.read_csv(path, sep="\t").iloc[1:].to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="Mutation truth"):
        validate_generated_tumor_directory(bundle)


def test_bundle_rejects_clone_ccf_mismatch(bundle):
    path = bundle / "truth_clone_sample.txt"
    table = pd.read_csv(path, sep="\t")
    table.loc[0, "ccf"] -= .1
    table.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="acquisition-clone"):
        validate_generated_tumor_directory(bundle)


def test_bundle_rejects_manifest_count_mismatch(bundle):
    path = bundle / "scenario_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["intended_factors"]["mutation_count"] += 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        validate_generated_tumor_directory(bundle)


def test_source_only_simulation_cli_uses_clonal_cn_defaults():
    config = tumor_simulation_config_from_args(build_parser().parse_args([]))
    assert config.copy_number.max_allele_cn == 6
    assert not hasattr(config.copy_number, "min_two_state_snv_fraction")
    assert "Trunk gain rate" in build_parser().format_help()
    with pytest.raises(SystemExit):
        parse_args(["simulate"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--min-two-state-snv-fraction", "0.6"])


def test_source_tree_simulation_command(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(repository.parent), PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(
        [sys.executable, "-m", "tools.simulation", "--out-dir", str(tmp_path),
         "--tumor-id", "cli_smoke", "--mutation-count", "15",
         "--clone-count", "1", "--region-count", "1", "--seed", "8"],
        env=env, cwd=repository, capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    data = load_tumor_txt(tmp_path / "cli_smoke" / "cli_smoke.clipp2.txt")
    assert data.num_mutations == 15
    assert not data.cn_filter_report.excluded_mutation_ids
