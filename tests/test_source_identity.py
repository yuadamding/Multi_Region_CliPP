"""Build/runtime identity uses one portable inventory and hashing convention."""
import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath
import subprocess
import sys

import pytest

from CliPP2._source import (
    canonical_source_name,
    fingerprint_source_entries,
    source_fingerprint,
    source_inventory,
)


def test_posix_and_windows_inventory_names_and_order_have_identical_hashes():
    names = ["core/objective.py", "a.py", "a/nested.py", "Z.py", "z.py", "io/é.py"]
    posix = [(PurePosixPath(name), name.encode("utf-8")) for name in names]
    windows = [(PureWindowsPath(name), name.encode("utf-8")) for name in reversed(names)]
    assert [canonical_source_name(path) for path, _ in posix] == names
    assert [canonical_source_name(path) for path, _ in windows] == names[::-1]
    assert fingerprint_source_entries(posix) == fingerprint_source_entries(windows)
    # Explicitly exercise ordering that differs under host-native PurePath
    # sorting (a.py vs a/nested.py, and Windows case-insensitive Z.py/z.py).
    expected = hashlib.sha256()
    for name in sorted(names):
        expected.update(name.encode("utf-8") + b"\0")
        expected.update(hashlib.sha256(name.encode("utf-8")).digest())
    assert fingerprint_source_entries(posix) == expected.hexdigest()


@pytest.mark.parametrize("path", [
    PurePosixPath("/core/model.py"), PureWindowsPath("C:/core/model.py"),
    PureWindowsPath("C:core/model.py"), PurePosixPath("../model.py"),
    PureWindowsPath("core/../model.py"), PurePosixPath("."),
])
def test_source_names_must_be_unambiguous_relative_paths(path):
    with pytest.raises(ValueError, match="relative paths"):
        canonical_source_name(path)


def test_duplicate_names_are_rejected_after_canonicalization():
    with pytest.raises(ValueError, match="duplicate canonical"):
        fingerprint_source_entries([
            (PureWindowsPath("core/objective.py"), b"first"),
            (PurePosixPath("core/objective.py"), b"second"),
        ])


def test_inventory_is_sorted_and_excludes_only_non_runtime_python(tmp_path):
    included = {"__init__.py": b"# package", "a.py": b"# A", "a/nested.py": b"# nested",
                "core/solver.py": b"# solver", "io/é.py": b"# unicode"}
    excluded = ["setup.py", "core/setup.py", "tests/test.py", "tools/probe.py",
                "build/lib/stale.py", "__pycache__/generated.py", ".git/hook.py",
                "core/__pycache__/code.py", "_build_source.json", "README.md", "core/UPPER.PY"]
    for name, content in [*reversed(list(included.items())), *((name, b"ignored") for name in excluded)]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    # A directory ending in .py is not a source file.
    (tmp_path / "directory.py").mkdir()
    assert [name for name, _ in source_inventory(tmp_path)] == sorted(included)
    expected = fingerprint_source_entries((PurePosixPath(name), data) for name, data in included.items())
    assert source_fingerprint(tmp_path) == expected
    for name in excluded:
        (tmp_path / name).write_bytes(b"changed non-source")
    assert source_fingerprint(tmp_path) == expected
    (tmp_path / "core/solver.py").write_bytes(b"modified solver")
    assert source_fingerprint(tmp_path) != expected


def test_source_hash_depends_on_filename_and_bytes_not_creation_order(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for directory, names in [(first, ["A.py", "b.py"]), (second, ["b.py", "A.py"])]:
        for name in names:
            (directory / name).write_bytes(b"identical bytes")
    assert source_fingerprint(first) == source_fingerprint(second)
    (second / "b.py").rename(second / "renamed.py")
    assert source_fingerprint(first) != source_fingerprint(second)


def test_build_hash_helper_loads_without_importing_the_package_or_numerical_stack():
    helper = Path(__file__).resolve().parents[1] / "_source.py"
    script = (
        "import runpy, sys\n"
        f"helper = runpy.run_path({str(helper)!r})\n"
        "assert callable(helper['source_fingerprint'])\n"
        "assert not ({'CliPP2', 'numpy', 'pandas', 'torch'} & sys.modules.keys())\n"
    )
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", script],
                               capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stdout + completed.stderr
