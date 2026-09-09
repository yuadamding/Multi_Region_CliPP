"""Public imports preserve their identities without loading unused backends."""

import os
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import venv
from zipfile import ZipFile

import pytest


def _fresh_python(source: str) -> None:
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    completed = subprocess.run(
        [sys.executable, "-c", source],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    ("statement", "forbidden"),
    [
        ("import CliPP2", ("numpy", "pandas", "torch")),
        ("from CliPP2 import __version__", ("numpy", "pandas", "torch")),
        ("import CliPP2.core", ("numpy", "pandas", "torch")),
        ("import CliPP2.config", ("numpy", "pandas", "torch")),
        ("from CliPP2 import FitConfig", ("numpy", "pandas", "torch")),
        ("from CliPP2 import TumorData", ("pandas", "torch")),
        ("from CliPP2 import load_tumor_txt", ("torch",)),
    ],
)
def test_import_does_not_load_unused_backend(statement, forbidden):
    _fresh_python(
        f"import sys\n{statement}\n"
        f"assert not (set({forbidden!r}) & sys.modules.keys())\n"
    )


def test_public_exports_retain_names_identity_and_discovery():
    _fresh_python(
        """
from importlib import import_module
import CliPP2
expected = {
    'FitConfig': 'CliPP2.config',
    'FitResult': 'CliPP2.api',
    'TumorData': 'CliPP2.io.data',
    'fit_fixed_objective': 'CliPP2.api',
    'prepare_problem': 'CliPP2.api',
    'PreparedProblem': 'CliPP2.core.fusion.types',
    'fit_prepared': 'CliPP2.core.fusion.solver',
    'process_tumor': 'CliPP2.api',
    'load_tumor_txt': 'CliPP2.io.tumor_txt',
    'resolve_fit_config': 'CliPP2.config',
}
assert set(CliPP2.__all__) == set(expected) | {'__version__'}
assert set(CliPP2.__all__) <= set(dir(CliPP2))
namespace = {}
exec('from CliPP2 import *', namespace)
for name, module in expected.items():
    actual = getattr(CliPP2, name)
    assert actual is getattr(import_module(module), name)
    assert actual is namespace[name] is CliPP2.__dict__[name]
assert not hasattr(CliPP2, 'nonexistent_public_operation')
"""
    )


def test_core_graph_export_retains_identity_and_discovery():
    _fresh_python(
        """
import CliPP2.core as core
assert core.__all__ == ['PairwiseFusionGraph']
assert 'PairwiseFusionGraph' in dir(core)
namespace = {}
exec('from CliPP2.core import *', namespace)
from CliPP2.core.fusion.types import PairwiseFusionGraph
assert core.PairwiseFusionGraph is PairwiseFusionGraph
assert namespace['PairwiseFusionGraph'] is PairwiseFusionGraph
assert core.__dict__['PairwiseFusionGraph'] is PairwiseFusionGraph
assert not hasattr(core, 'nonexistent_public_operation')
"""
    )


def test_built_wheel_installs_without_checkout_shadow_and_completes_cpu_fit(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    # Build a source copy: no generated build/ or egg-info dirties the checkout.
    source = tmp_path / "source"
    shutil.copytree(repository, source, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", ".pytest_cache", ".ruff_cache", "*.egg-info", "build", "dist",
    ))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    built = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
         "--wheel-dir", str(tmp_path / "dist"), str(source)],
        env=env, cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    wheel, = (tmp_path / "dist").glob("*.whl")
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        assert all(not name.startswith(("CliPP2/tests/", "CliPP2/tools/", "CliPP2/simulation/", "CliPP2/runners/")) for name in names)
        assert "CliPP2/api.py" in names and "CliPP2/reporting.py" in names
        assert "CliPP2/setup.py" not in names
        identity = json.loads(archive.read("CliPP2/_build_source.json"))
        digest = hashlib.sha256()
        for name in sorted(name for name in names if name.startswith("CliPP2/") and name.endswith(".py")):
            digest.update(name.removeprefix("CliPP2/").encode() + b"\0")
            digest.update(hashlib.sha256(archive.read(name)).digest())
        assert identity["python_source_sha256"] == digest.hexdigest()
        assert identity["commit"] is None  # This source copy intentionally has no Git metadata.
    environment = tmp_path / "environment"
    # Numerical dependencies are inherited from the qualified test interpreter;
    # the CliPP2 distribution itself is installed freshly into the isolated venv.
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
    python = environment / "bin" / "python"
    installed = subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", "--ignore-installed", str(wheel)],
        env=env, cwd=tmp_path, capture_output=True, text=True, timeout=90,
    )
    assert installed.returncode == 0, installed.stdout + installed.stderr
    script = r'''
from pathlib import Path
import json
import sys
import CliPP2
from CliPP2.api import process_tumor
from CliPP2.config import resolve_fit_config
assert Path(CliPP2.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert CliPP2.__version__ == "0.5.0"
path = Path("wheel.tsv")
path.write_text("mutation_id\tsample_id\talt_count\tref_count\tcount_observed\tpurity\tnormal_cn\tsegment_id\tcn_state_id\tcn_state_fraction\tallele_a_cn\tallele_b_cn\n"
                "keep\tR1\t20\t80\t1\t0.8\t2\ts1\tclonal\t1\t2\t1\n"
                "drop\tR1\t20\t80\t1\t0.8\t2\ts2\tclonal\t1\t7\t0\n")
config = resolve_fit_config(device="cpu", dtype="float64", outer_max_iter=20, inner_max_iter=80)
summary = process_tumor(path, Path("results"), fit_config=config)
assert summary["selected_n_clusters"] == 1
assert summary["retained_mutation_count"] == 1
assert summary["excluded_mutation_count"] == 1
assert summary["raw_reference_objective_certified"]
manifest = json.loads(Path("results/wheel_run_manifest.json").read_text())
assert manifest["status"] == "complete"
assert len(manifest["files"]) == 4
print(CliPP2.__file__)
'''
    smoke = subprocess.run(
        [str(python), "-I", "-c", script], env=env, cwd=tmp_path,
        capture_output=True, text=True, timeout=90,
    )
    assert smoke.returncode == 0, smoke.stdout + smoke.stderr


def test_installed_build_identity_requires_matching_python_bytes(tmp_path, monkeypatch):
    from CliPP2 import reporting
    module = tmp_path / "reporting.py"
    module.write_text("# installed source\n")
    monkeypatch.setattr(reporting, "__file__", str(module))
    monkeypatch.setattr(reporting.subprocess, "check_output", lambda *a, **kw: pytest.fail("no Git checkout"))
    original = reporting._source_identity()
    build = dict(original, schema_version=1, commit="a" * 40, dirty=False)
    metadata = tmp_path / "_build_source.json"
    metadata.write_text(json.dumps(build))
    assert reporting._source_identity()["commit"] == "a" * 40
    module.write_text("# modified after installation\n")
    assert reporting._source_identity()["commit"] is None
    for malformed in ("null", "[]", "not JSON"):
        metadata.write_text(malformed)
        assert reporting._source_identity()["commit"] is None
