"""No-clobber and completion-last regression tests for persisted run identity."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from CliPP2._version import __version__
from CliPP2.config import resolve_fit_config
from CliPP2 import reporting as outputs, api as pipeline
from CliPP2.reporting import RunPublication, cn_filter_output_table
from test_integer_reporting import _data, _selection


def _manifest(directory: Path, tumor: str = "tumor") -> dict:
    return json.loads((directory / f"{tumor}_run_manifest.json").read_text())


def _tables() -> dict[str, pd.DataFrame]:
    return {
        "mutation_clusters": pd.DataFrame({"cluster_label": [1]}),
        "cluster_centers": pd.DataFrame({"phi": [0.5]}),
        "mutation_region_multiplicity": pd.DataFrame({"multiplicity_call": [1]}),
        "excluded_mutations": cn_filter_output_table("tumor", None),
    }


def _input(path: Path, major: int = 2) -> Path:
    pd.DataFrame([
        dict(mutation_id=f"m{index}", sample_id="R1", alt_count=18, ref_count=42,
             count_observed=1, purity=.8, normal_cn=2, segment_id=f"s{index}",
             cn_state_id="clonal", cn_state_fraction=1, allele_a_cn=major, allele_b_cn=1)
        for index in range(3)
    ]).to_csv(path, sep="\t", index=False)
    return path


def test_input_changed_while_loading_is_not_bound_to_old_data(tmp_path, monkeypatch):
    path = _input(tmp_path / "tumor.tsv")
    original_loader = pipeline.load_tumor_txt
    def changed_loader(*args, **kwargs):
        data = original_loader(*args, **kwargs)
        _input(path, major=3)
        return data
    monkeypatch.setattr(pipeline, "load_tumor_txt", changed_loader)
    monkeypatch.setattr("CliPP2.model_selection.search.select_model",
                        lambda **kw: pytest.fail("must reject before selection"))
    with pytest.raises(ValueError, match="Input changed while loading"):
        pipeline.process_tumor(path, tmp_path / "output",
                               fit_config=resolve_fit_config(device="cpu"))
    assert not (tmp_path / "output").exists()


def test_manifest_binds_config_source_input_and_completed_tables(tmp_path):
    path = _input(tmp_path / "tumor.tsv")
    config = resolve_fit_config(device="cpu")
    outdir = tmp_path / "out"
    run = RunPublication(outdir, "tumor", input_file=path, fit_config=config,
                         workflow={"use_warm_starts": False})
    assert _manifest(outdir)["status"] == "running"
    run.write_audit(None)
    run.publish(_tables())
    manifest = _manifest(outdir)
    assert manifest["status"] == "complete"
    assert len(manifest["run_id"]) == 32
    assert manifest["software_version"] == __version__
    assert manifest["source"] == outputs._source_identity()
    if manifest["source"]["commit"] is not None:
        assert len(manifest["source"]["commit"]) == 40
    assert len(manifest["source"]["python_source_sha256"]) == 64
    assert manifest["input"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest["config"]["runtime"]["device"] == "cpu"
    assert manifest["config_sha256"] == hashlib.sha256(outputs._json_bytes(manifest["config"])).hexdigest()
    assert manifest["workflow_sha256"] == hashlib.sha256(outputs._json_bytes(manifest["workflow"])).hexdigest()
    assert len(manifest["files"]) == 4
    for name, identity in manifest["files"].items():
        content = (outdir / name).read_bytes()
        assert identity == {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}
    assert {p.name for p in outdir.iterdir()} == {*manifest["files"], "tumor_run_manifest.json"}


@pytest.mark.parametrize("all_excluded", [False, True])
def test_completed_run_cannot_be_mixed_with_a_failed_retry(tmp_path, monkeypatch, all_excluded):
    input_path = _input(tmp_path / "tumor.tsv")
    outdir = tmp_path / "out"
    config = resolve_fit_config(device="cpu", dtype="float64", outer_max_iter=20,
                                inner_max_iter=80, certificate_max_iter=100,
                                certificate_refinement_rounds=1)
    pipeline.process_tumor(input_path, outdir, fit_config=config)
    originals = {p.name: p.read_bytes() for p in outdir.iterdir()}
    assert _manifest(outdir)["status"] == "complete"
    _input(input_path, major=7 if all_excluded else 3)
    monkeypatch.setattr("CliPP2.model_selection.search.select_model", lambda **kwargs: pytest.fail("rerun must reject before optimization"))
    with pytest.raises(FileExistsError, match="Existing tumor outputs"):
        pipeline.process_tumor(input_path, outdir, fit_config=config)
    assert {p.name: p.read_bytes() for p in outdir.iterdir()} == originals


@pytest.mark.parametrize("suffix", ["mutation_clusters.tsv", "run_summary.tsv", "run_manifest.json", "old_notes.txt"])
def test_legacy_or_partial_output_namespace_is_preserved(tmp_path, suffix):
    original = tmp_path / f"tumor_{suffix}"
    original.write_bytes(b"previous analysis")
    with pytest.raises(FileExistsError, match="Existing tumor outputs"):
        RunPublication(tmp_path, "tumor")
    assert list(tmp_path.iterdir()) == [original]
    assert original.read_bytes() == b"previous analysis"


def test_fit_failure_records_audit_and_never_completion(tmp_path, monkeypatch):
    path = _input(tmp_path / "tumor.tsv")
    outdir = tmp_path / "out"

    def fail(**kwargs):
        assert _manifest(outdir)["status"] == "running"
        assert (outdir / "tumor_excluded_mutations.tsv").is_file()
        raise RuntimeError("deliberate scientific failure")

    monkeypatch.setattr("CliPP2.model_selection.search.select_model", fail)
    with pytest.raises(RuntimeError, match="deliberate scientific failure"):
        pipeline.process_tumor(path, outdir)
    manifest = _manifest(outdir)
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "RuntimeError"
    assert set(manifest["files"]) == {"tumor_excluded_mutations.tsv"}
    assert len(list(outdir.iterdir())) == 2


def test_partial_publication_is_failed_and_never_overwrites(tmp_path, monkeypatch):
    data = _data()
    link = outputs.os.link
    calls = []

    def fail_on_second_link(source, destination):
        assert _manifest(tmp_path, data.tumor_id)["status"] == "running"
        calls.append(destination)
        if len(calls) == 2:
            destination.write_bytes(b"concurrent protected content")
        link(source, destination)

    monkeypatch.setattr(outputs.os, "link", fail_on_second_link)
    with pytest.raises(FileExistsError):
        outputs.write_fit_outputs(outdir=tmp_path, data=data,
                                  raw_fit=_selection(data)[0], partition=_selection(data)[1],
                                  refit=_selection(data)[2])
    assert calls[1].read_bytes() == b"concurrent protected content"
    assert _manifest(tmp_path, data.tumor_id)["status"] == "failed"
    assert calls[0].name in _manifest(tmp_path, data.tumor_id)["files"]
    assert not any(p.name.startswith(".clipp2-") for p in tmp_path.iterdir())


def test_staging_failure_publishes_no_fit_tables(tmp_path, monkeypatch):
    run = RunPublication(tmp_path, "tumor")
    run.write_audit(None)
    original = (tmp_path / "tumor_excluded_mutations.tsv").read_bytes()
    to_csv = pd.DataFrame.to_csv

    def fail(table, path, *args, **kwargs):
        if Path(path).name.endswith("cluster_centers.tsv"):
            raise OSError("disk failure")
        return to_csv(table, path, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail)
    with pytest.raises(OSError, match="disk failure") as error:
        run.publish(_tables())
    run.fail(error.value)
    assert _manifest(tmp_path)["status"] == "failed"
    assert (tmp_path / "tumor_excluded_mutations.tsv").read_bytes() == original
    assert len(list(tmp_path.iterdir())) == 2


@pytest.mark.parametrize("tamper", ["audit", "input"])
def test_changed_input_or_audit_cannot_be_marked_complete(tmp_path, tamper):
    input_file = _input(tmp_path / "input.tsv")
    outdir = tmp_path / "out"
    run = RunPublication(outdir, "tumor", input_file=input_file)
    run.write_audit(None)
    target = input_file if tamper == "input" else outdir / "tumor_excluded_mutations.tsv"
    target.write_text("changed")
    with pytest.raises(ValueError, match="changed") as error:
        run.publish(_tables())
    run.fail(error.value)
    assert _manifest(outdir)["status"] == "failed"
    assert target.read_text() == "changed"


def test_changed_config_changes_hash_without_changing_source(tmp_path):
    config = resolve_fit_config(device="cpu")
    first = RunPublication(tmp_path / "a", "tumor", fit_config=config)
    second = RunPublication(tmp_path / "b", "tumor", fit_config=replace(config, eps=config.eps * 2))
    assert first.record["config_sha256"] != second.record["config_sha256"]
    assert first.record["source"] == second.record["source"]
    assert first.record["run_id"] != second.record["run_id"]


def test_source_identity_does_not_infer_a_wheel_commit_from_parent_git(tmp_path, monkeypatch):
    package = tmp_path / "installed" / "CliPP2"
    runner = package / "runners"
    runner.mkdir(parents=True)
    module = runner / "outputs.py"
    module.write_text("# fixture\n")
    monkeypatch.setattr(outputs, "__file__", str(module))
    monkeypatch.setattr(outputs.subprocess, "check_output", lambda *a, **kw: pytest.fail("no package git metadata"))
    identity = outputs._source_identity()
    assert identity["commit"] is None
    assert identity["dirty"] is None
    assert len(identity["python_source_sha256"]) == 64


def test_standalone_exclusion_writer_never_clobbers(tmp_path):
    path = outputs.write_cn_filter_output(outdir=tmp_path, tumor_id="tumor", report=None)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        outputs.write_cn_filter_output(outdir=tmp_path, tumor_id="tumor", report=None)
    assert path.read_bytes() == original
