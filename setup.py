"""Embed honest source identity in wheels without importing the inference stack."""
import hashlib
import json
from pathlib import Path
import subprocess

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def find_package_modules(self, package, package_dir):
        return [module for module in super().find_package_modules(package, package_dir)
                if module[1] != "setup"]

    def run(self):
        source = Path(__file__).resolve().parent
        commit = dirty = None
        if (source / ".git").exists():
            try:
                commit = subprocess.check_output(
                    ["git", "-C", str(source), "rev-parse", "--verify", "HEAD"],
                    text=True, stderr=subprocess.DEVNULL, timeout=5,
                ).strip()
                dirty = bool(subprocess.check_output(
                    ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=normal"],
                    text=True, stderr=subprocess.DEVNULL, timeout=5,
                ).strip())
            except (OSError, subprocess.SubprocessError):
                commit = dirty = None
        super().run()
        package = Path(self.build_lib) / "CliPP2"
        digest = hashlib.sha256()
        for path in sorted(package.rglob("*.py")):
            digest.update(path.relative_to(package).as_posix().encode() + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        record = {"schema_version": 1, "commit": commit, "dirty": dirty,
                  "python_source_sha256": digest.hexdigest()}
        (package / "_build_source.json").write_text(json.dumps(record, sort_keys=True) + "\n")


setup(cmdclass={"build_py": BuildPy})
