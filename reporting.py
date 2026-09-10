from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from uuid import uuid4
import warnings

import numpy as np
import pandas as pd

from .core.fusion.multiplicity import (
    infer_integer_multiplicity_posterior_numpy,
)
from ._version import __version__
from ._source import source_fingerprint
from .config import FitConfig
from .core.fusion.types import RawFit
from .io.data import CNFilterReport, TumorData
from .io.multiplicity import CLONAL_INTEGER_MODEL_ID, MAX_MAJOR_CN
from .model_selection.partitions import _partition_signature
from .model_selection.types import (
    BICSelectionResult,
    RawFusionCandidate,
    SelectablePartitionCandidate,
    SelectionScore,
    DirectPartition,
    FusionPartition,
    PartitionRefitSummary,
)

SelectedPartition = FusionPartition | DirectPartition
OUTPUT_SUFFIXES = (
    "mutation_clusters.tsv", "cluster_centers.tsv",
    "mutation_region_multiplicity.tsv", "excluded_mutations.tsv",
    "run_manifest.json",
)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _source_identity() -> dict[str, object]:
    """Hash installed Python sources; never mistake an enclosing repo for ours."""
    root = Path(__file__).resolve().parent
    source_hash = source_fingerprint(root)
    commit = dirty = None
    if (root / ".git").exists():
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
                text=True, stderr=subprocess.DEVNULL, timeout=5,
            ).strip()
            dirty = bool(subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
                text=True, stderr=subprocess.DEVNULL, timeout=5,
            ).strip())
        except (OSError, subprocess.SubprocessError):
            commit = dirty = None
    else:
        # A wheel identity is valid only while its installed Python bytes
        # still match the package actually fingerprinted by the build hook.
        try:
            built = json.loads((root / "_build_source.json").read_text())
            if not isinstance(built, dict):
                raise ValueError("Build source identity must be a JSON object.")
            revision = built.get("commit")
            if (built.get("schema_version") == 1
                and built.get("python_source_sha256") == source_hash
                and isinstance(revision, str) and len(revision) == 40
                and all(character in "0123456789abcdef" for character in revision)
                and isinstance(built.get("dirty"), bool)):
                commit, dirty = revision, built["dirty"]
        except (OSError, ValueError, TypeError):
            pass
    return {"commit": commit, "dirty": dirty, "python_source_sha256": source_hash}


class RunPublication:
    """Exclusive tumor namespace with a completion-last, hash-bound manifest.

    A failed or interrupted attempt must use a new output directory to retry.
    Existing files are never replaced or removed, including partial attempts.
    """

    def __init__(
        self, outdir: Path, tumor_id: str, *, input_file: Path | None = None,
        expected_input_sha256: str | None = None,
        fit_config: FitConfig | None = None, workflow: dict[str, object] | None = None,
    ) -> None:
        self.outdir, self.tumor_id = Path(outdir), str(tumor_id)
        if not self.tumor_id or Path(self.tumor_id).name != self.tumor_id or self.tumor_id in {".", ".."}:
            raise ValueError("tumor_id must be a nonempty filename component.")
        if input_file is not None:
            for suffix in OUTPUT_SUFFIXES:
                destination = self.outdir / f"{self.tumor_id}_{suffix}"
                if destination.resolve() == input_file.resolve() or (
                    destination.exists() and destination.samefile(input_file)
                ):
                    raise ValueError(f"Output would overwrite original input: {destination}. Choose a separate output directory.")
        input_sha256 = None if input_file is None else _file_hash(input_file)
        if expected_input_sha256 is not None and input_sha256 != expected_input_sha256:
            raise ValueError("Input changed while loading; no output was published.")
        self.outdir.mkdir(parents=True, exist_ok=True)
        conflicts = sorted(path.name for path in self.outdir.iterdir()
                           if path.name.startswith(f"{self.tumor_id}_"))
        if conflicts:
            raise FileExistsError(f"Existing tumor outputs must not be overwritten: {', '.join(conflicts)}. Choose a new output directory.")
        config = None
        if fit_config is not None:
            config = asdict(replace(fit_config, graph=replace(fit_config.graph, graph=None)))
            graph = fit_config.graph.graph
            config["graph"]["graph"] = None if graph is None else {"fingerprint": graph.fingerprint}
        workflow = {} if workflow is None else workflow
        self.path = self.outdir / f"{self.tumor_id}_run_manifest.json"
        self.record = {
            "schema_version": 1, "run_id": uuid4().hex, "tumor_id": self.tumor_id,
            "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
            "software_version": __version__, "source": _source_identity(),
            "input": None if input_file is None else {
                "path": str(input_file.resolve()), "sha256": input_sha256,
            },
            "config": config, "config_sha256": None if config is None else hashlib.sha256(_json_bytes(config)).hexdigest(),
            "workflow": workflow, "workflow_sha256": hashlib.sha256(_json_bytes(workflow)).hexdigest(),
            "files": {}, "analysis": None,
        }
        # The manifest is also the exclusive claim: two concurrent starts
        # cannot both acquire the same namespace after their directory check.
        with self.path.open("xb") as stream:
            stream.write(_json_bytes(self.record))
            stream.flush()
            os.fsync(stream.fileno())

    def _save(self) -> None:
        current = json.loads(self.path.read_text())
        if current.get("run_id") != self.record["run_id"] or current.get("status") != "running":
            raise RuntimeError("Run manifest ownership or running status changed.")
        with tempfile.NamedTemporaryFile(dir=self.outdir, prefix=".clipp2-manifest-", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(_json_bytes(self.record))
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def _remember(self, path: Path) -> None:
        self.record["files"][path.name] = {"sha256": _file_hash(path), "size_bytes": path.stat().st_size}

    def write_audit(self, report: CNFilterReport | None) -> None:
        path = write_cn_filter_output(outdir=self.outdir, tumor_id=self.tumor_id, report=report)
        self._remember(path)
        self._save()

    def fail(self, error: BaseException) -> None:
        self.record.update(status="failed", finished_at=datetime.now(timezone.utc).isoformat(),
                           error={"type": type(error).__name__, "message": str(error)})
        try:
            self._save()
        except Exception as manifest_error:
            # A disk/full permission failure must not hide the fit exception.
            # The previous running manifest still cannot be read as complete.
            warnings.warn(f"Could not persist failed run status: {manifest_error}", RuntimeWarning)

    def publish(self, tables: dict[str, pd.DataFrame], *, analysis: dict[str, object] | None = None) -> None:
        if {f"{suffix}.tsv" for suffix in tables} != set(OUTPUT_SUFFIXES[:-1]):
            raise ValueError("Publication requires all four analysis tables.")
        # Qualification is supplied only after identity-valid table generation.
        # It describes the fit even if file publication subsequently fails.
        _json_bytes(analysis)
        self.record["analysis"] = analysis
        with tempfile.TemporaryDirectory(dir=self.outdir, prefix=".clipp2-tables-") as staging:
            paths = []
            for suffix, table in tables.items():
                path = Path(staging) / f"{self.tumor_id}_{suffix}.tsv"
                table.to_csv(path, sep="\t", index=False)
                paths.append(path)
            # Validate the early audit before publishing any fit table.
            for path in paths:
                existing = self.record["files"].get(path.name)
                if existing is not None and (
                    _file_hash(path) != existing["sha256"]
                    or _file_hash(self.outdir / path.name) != existing["sha256"]
                ):
                    raise ValueError("Early exclusion audit changed before publication.")
            for path in paths:
                if path.name not in self.record["files"]:
                    destination = self.outdir / path.name
                    os.link(path, destination)  # Atomic publication, never clobber.
                    self._remember(destination)
            for name, identity in self.record["files"].items():
                if _file_hash(self.outdir / name) != identity["sha256"]:
                    raise ValueError("Published output changed before completion.")
            expected = {self.path.name, *self.record["files"]}
            actual = {path.name for path in self.outdir.iterdir()
                      if path.name.startswith(f"{self.tumor_id}_")}
            if actual != expected:
                raise ValueError("Unexpected tumor outputs appeared during publication.")
            if self.record["input"] is not None:
                if _file_hash(Path(self.record["input"]["path"])) != self.record["input"]["sha256"]:
                    raise ValueError("Input changed during the run.")
            self.record.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat())
            self._save()


def _validated_profile(
    data: TumorData,
    values: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    profile = np.asarray(values, dtype=np.float64)
    expected = (int(data.num_mutations), int(data.num_regions))
    if profile.shape != expected:
        raise ValueError(f"{name} has shape {profile.shape}; expected {expected}.")
    if not np.all(np.isfinite(profile)):
        raise ValueError(f"{name} must contain only finite values.")
    return profile


def _validate_identity(
    raw_fit: RawFit,
    partition: SelectedPartition,
    refit: PartitionRefitSummary,
) -> np.ndarray:
    raw_phi = np.asarray(raw_fit.phi, dtype=np.float64)
    labels = np.asarray(partition.labels, dtype=np.int64)
    if labels.shape != (raw_phi.shape[0],):
        raise AssertionError("Selected partition does not match raw fit mutations.")
    if not np.array_equal(labels, np.asarray(refit.labels, dtype=np.int64)):
        raise AssertionError("Selected partition and fixed refit labels differ.")
    if partition.signature != _partition_signature(
        labels,
        partition.mutation_ids if partition.mutation_ids else None,
    ):
        raise AssertionError("Selected partition signature does not match its labels.")
    if partition.signature != refit.partition_signature:
        raise AssertionError("Selected partition and fixed refit signatures differ.")
    if isinstance(partition, FusionPartition) and not partition.certified:
        raise AssertionError("Refusing to serialize an uncertified raw partition.")
    return labels


def mutation_output_table(
    data: TumorData,
    raw_fit: RawFit,
    partition: SelectedPartition,
    refit: PartitionRefitSummary,
) -> pd.DataFrame:
    labels = _validate_identity(raw_fit, partition, refit)
    refit_phi = _validated_profile(data, refit.phi, name="refit.phi")
    table = pd.DataFrame(
        {
            "tumor_id": np.repeat(data.tumor_id, data.num_mutations),
            "mutation_id": data.mutation_ids,
            "cluster_label": labels + 1,
        }
    )
    for column, region_id in enumerate(data.region_ids):
        region = str(region_id)
        # The selected fixed-partition refit is the authoritative reported CCF.
        # Keep the compact v0.2.1-style public name requested by downstream
        # consumers; raw-fusion diagnostics remain in the audit tables.
        table[f"phi_{region}"] = refit_phi[:, column]
    return table


def cluster_output_table(
    data: TumorData,
    raw_fit: RawFit,
    partition: SelectedPartition,
    refit: PartitionRefitSummary,
) -> pd.DataFrame:
    labels = _validate_identity(raw_fit, partition, refit)
    centers = np.asarray(refit.cluster_centers, dtype=np.float64)
    expected = (int(partition.n_clusters), int(data.num_regions))
    if centers.shape != expected or not np.all(np.isfinite(centers)):
        raise ValueError(f"refit.cluster_centers must have shape {expected}.")
    sizes = np.bincount(labels, minlength=int(partition.n_clusters))
    table = pd.DataFrame(
        {
            "tumor_id": np.repeat(data.tumor_id, partition.n_clusters),
            "cluster_label": np.arange(1, partition.n_clusters + 1, dtype=int),
            "cluster_size": sizes,
        }
    )
    for column, region_id in enumerate(data.region_ids):
        region = str(region_id)
        table[f"phi_{region}"] = centers[:, column]
    return table


def _add_integer_multiplicity(
    table: pd.DataFrame,
    *,
    data: TumorData,
    phi: np.ndarray,
    eps: float,
) -> None:
    posterior = infer_integer_multiplicity_posterior_numpy(data, phi, eps=eps)
    count = posterior.candidate_count.reshape(-1)
    table["multiplicity_candidates"] = [
        ",".join(str(candidate) for candidate in range(1, int(size) + 1))
        for size in count
    ]
    table["multiplicity_candidate_count"] = count
    calls = pd.array(posterior.multiplicity_call.reshape(-1), dtype="Int64")
    informative = posterior.informative.reshape(-1)
    calls[(~informative) & (count > 1)] = pd.NA
    table["multiplicity_call"] = calls
    table["multiplicity_call_probability"] = posterior.map_probability.reshape(-1)
    table["multiplicity_informative"] = informative
    for candidate in range(1, MAX_MAJOR_CN + 1):
        table[f"multiplicity_p{candidate}"] = (
            posterior.posterior[..., candidate - 1].reshape(-1)
            if candidate <= posterior.posterior.shape[-1]
            else 0.0
        )


def mutation_region_output_table(
    data: TumorData,
    raw_fit: RawFit,
    partition: SelectedPartition,
    refit: PartitionRefitSummary,
    *,
    eps: float | None = None,
) -> pd.DataFrame:
    labels = _validate_identity(raw_fit, partition, refit)
    refit_phi = _validated_profile(data, refit.phi, name="refit.phi")
    mutation_ids = np.repeat(
        np.asarray(data.mutation_ids, dtype=object), data.num_regions
    )
    region_ids = np.tile(
        np.asarray([str(x) for x in data.region_ids], dtype=object),
        data.num_mutations,
    )
    table = pd.DataFrame(
        {
            "tumor_id": np.repeat(data.tumor_id, mutation_ids.shape[0]),
            "mutation_id": mutation_ids,
            "region_id": region_ids,
            "cluster_label": np.repeat(labels + 1, data.num_regions),
            "phi": refit_phi.reshape(-1),
            "major_cn": data.major_cn.reshape(-1),
            "minor_cn": data.minor_cn.reshape(-1),
        }
    )
    likelihood_eps = float(raw_fit.provenance.likelihood_eps)
    if eps is not None and float(eps) != likelihood_eps:
        raise ValueError("Reporting eps must match the fitted likelihood provenance.")
    eps = likelihood_eps
    spec = data.path_likelihood
    if spec is None or spec.model_id != CLONAL_INTEGER_MODEL_ID:
        raise ValueError("Reporting supports only the clonal integer multiplicity model.")
    _add_integer_multiplicity(table, data=data, phi=refit_phi, eps=eps)
    return table


def cn_filter_output_table(
    tumor_id: str,
    report: CNFilterReport | None,
) -> pd.DataFrame:
    """One row per triggering mutation-region-reason, also when none remain."""

    fields = (
        "mutation_id",
        "sample_id",
        "segment_id",
        "reason",
        "n_distinct_cn_states",
        "max_major_cn",
    )
    return pd.DataFrame(
        [
            {"tumor_id": tumor_id, **{name: getattr(record, name) for name in fields}}
            for record in (() if report is None else report.records)
        ],
        columns=("tumor_id", *fields),
    )


def write_cn_filter_output(
    *,
    outdir: Path,
    tumor_id: str,
    report: CNFilterReport | None,
) -> Path:
    """Write exclusion provenance independently of a successful fit."""

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{tumor_id}_excluded_mutations.tsv"
    with path.open("x") as stream:
        cn_filter_output_table(tumor_id, report).to_csv(stream, sep="\t", index=False)
    return path


def write_fit_outputs(
    *,
    outdir: Path,
    data: TumorData,
    raw_fit: RawFit,
    partition: SelectedPartition,
    refit: PartitionRefitSummary,
    eps: float | None = None,
    publication: RunPublication | None = None,
) -> None:
    """Purely serialize one already selected, identity-validated model."""

    own_publication = publication is None
    if publication is None:
        publication = RunPublication(outdir, data.tumor_id, workflow={"entrypoint": "write_fit_outputs", "eps": eps})
    if publication.outdir.resolve() != Path(outdir).resolve() or publication.tumor_id != data.tumor_id:
        raise ValueError("Publication does not belong to this tumor and output directory.")
    try:
        _write_fit_tables(data, raw_fit, partition, refit, eps, publication)
    except BaseException as error:
        if own_publication:
            publication.fail(error)
        raise


def _write_fit_tables(
    data: TumorData, raw_fit: RawFit, partition: SelectedPartition,
    refit: PartitionRefitSummary, eps: float | None, publication: RunPublication,
    *, selection_result: BICSelectionResult | None = None,
) -> None:
    tables = {
        "mutation_clusters": mutation_output_table(data, raw_fit, partition, refit),
        "cluster_centers": cluster_output_table(data, raw_fit, partition, refit),
        "mutation_region_multiplicity": mutation_region_output_table(
            data,
            raw_fit,
            partition,
            refit,
            eps=eps,
        ),
        "excluded_mutations": cn_filter_output_table(
            data.tumor_id, data.cn_filter_report
        ),
    }
    publication.publish(tables, analysis=_manifest_qualification(
        raw_fit, partition, refit, selection_result=selection_result,
    ))


SUMMARY_SCHEMA_VERSION = 4


def input_model_summary(data: TumorData) -> dict[str, object]:
    """Separate eligibility provenance from the retained numerical model."""
    report = data.cn_filter_report
    spec = data.path_likelihood
    records = () if report is None else report.records
    return {
        "input_mutation_count": data.num_mutations if report is None else report.input_mutation_count,
        "retained_mutation_count": data.num_mutations,
        "excluded_mutation_count": 0 if report is None else len(report.excluded_mutation_ids),
        "excluded_subclonal_cn_mutation_count": len({
            record.mutation_id for record in records if record.reason == "SUBCLONAL_CN_REGION"
        }),
        "excluded_major_cn_gt6_mutation_count": len({
            record.mutation_id for record in records if record.reason == "MAJOR_CN_GT_6"
        }),
        "cn_filter_policy_id": None if report is None else report.policy_id,
        "multiplicity_model_id": None if spec is None else spec.model_id,
        "multiplicity_candidate_generator_version": None if spec is None else spec.candidate_generator_version,
        "multiplicity_prior_mode": None if spec is None else spec.prior_mode,
    }


def _array_fingerprint(values: np.ndarray, *, dtype: np.dtype) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _finite_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _raw_qualification(fit: RawFit) -> dict[str, object] | None:
    """Persist raw evidence; missing typed evidence is explicitly unknown."""
    if not isinstance(fit, RawFit):
        return None
    certificate, provenance = fit.certificate, fit.provenance
    return {
        "kkt_certified": bool(certificate.certified),
        "admissible": bool(certificate.admissible),
        "global_optimum_certified": bool(certificate.global_optimum),
        "global_optimality_basis": str(provenance.global_optimality_basis),
        "certificate_status": str(certificate.status),
        "certificate_schema_version": int(certificate.schema_version),
        "kkt_residual": _finite_number(certificate.components.residual),
        "kkt_tolerance": _finite_number(certificate.tolerance),
        "residual_method": str(certificate.residual_method),
        "audit_dtype": str(certificate.audit_dtype),
        "objective": _finite_number(fit.objective.total),
        "lambda": _finite_number(provenance.lambda_value),
        "objective_hash": str(provenance.certificate_problem_hash),
        "base_objective_hash": str(provenance.base_fusion_objective_hash),
        "graph_hash": str(provenance.original_graph_hash),
        "phi_hash": _array_fingerprint(fit.phi, dtype=np.dtype(np.float64)),
    }


def _manifest_qualification(
    raw_fit: RawFit,
    partition: SelectedPartition,
    refit: PartitionRefitSummary,
    *, selection_result: BICSelectionResult | None,
) -> dict[str, object]:
    """Keep publication, raw admission, fixed-label refit, and search distinct."""
    _validate_identity(raw_fit, partition, refit)
    selected_raw = raw_fit if isinstance(partition, FusionPartition) else None
    selected_score = None
    if selection_result is not None:
        from .model_selection.candidates import validate_candidate_identity

        selected = selection_result.selected_model.partition_candidate
        reference = selection_result.selected_model.raw_reference
        validate_candidate_identity(selected)
        validate_candidate_identity(reference)
        if selected.partition is not partition or selected.refit is not refit or reference.raw_fit is not raw_fit:
            raise ValueError("Manifest qualification does not match the serialized selection.")
        selected_raw = selected.raw_fit if isinstance(selected, RawFusionCandidate) else None
        selected_score = {"name": str(selected.score.name), "value": _finite_number(selected.score.value)}
    direct = isinstance(partition, DirectPartition)
    return {
        "raw_reference": _raw_qualification(raw_fit),
        "selected_raw_fit": None if selected_raw is None else _raw_qualification(selected_raw),
        "selected_partition": {
            "family": "direct_partition" if direct else "raw_fusion",
            "source": str(partition.source), "signature": str(partition.signature),
            "labels_hash": _array_fingerprint(partition.labels, dtype=np.dtype(np.int64)),
            "n_clusters": int(partition.n_clusters),
            "raw_partition_certified": bool(not direct and partition.certified),
            "parent_raw_candidate_id": (int(partition.parent_raw_candidate_id)
                                        if direct and partition.parent_raw_candidate_id is not None else None),
            "parent_raw_lambda": (_finite_number(partition.parent_raw_lambda)
                                  if direct and partition.parent_raw_lambda is not None else None),
            "parent_raw_phi_hash": (partition.parent_raw_phi_hash or None) if direct else None,
        },
        "refit": {
            "finite_candidate_found": bool(refit.finite_candidate_found),
            "numerically_resolved": bool(refit.refit_numerically_resolved),
            "global_optimum_certified": bool(refit.global_optimum_certified),
            "mode": str(refit.refit_mode),
            "certificate_method": str(refit.global_certificate_method),
            "global_lower_bound": _finite_number(refit.global_lower_bound),
            "global_optimality_gap": _finite_number(refit.global_optimality_gap),
            "loglik": _finite_number(refit.loglik),
            "phi_hash": _array_fingerprint(refit.phi, dtype=np.dtype(np.float64)),
        },
        "selection": {
            "status": ("not_provided" if selection_result is None else
                       "resolved" if selection_result.selection_optimum_resolved else "provisional_unresolved"),
            "optimum_resolved": None if selection_result is None else bool(selection_result.selection_optimum_resolved),
            "boundary_unresolved": None if selection_result is None else bool(selection_result.selection_boundary_unresolved),
            "raw_lambda_path_resolved": None if selection_result is None else bool(selection_result.raw_lambda_path_resolved),
            "global_hybrid_optimum_certified": None if selection_result is None else bool(selection_result.global_hybrid_optimum_certified),
            "stop_reason": None if selection_result is None else str(selection_result.adaptive_search_stop_reason),
            "method": None if selection_result is None else str(selection_result.selection_method),
            "score": selected_score,
        },
    }


@dataclass(frozen=True, slots=True)
class AnalysisSerialization:
    """One normalized, output-ready view of a completed selection result.

    Everything downstream consumes this normalized typed boundary.
    """

    data: TumorData
    input_file: Path
    fit_config: FitConfig
    selection_result: BICSelectionResult

    @property
    def selected_candidate(self) -> SelectablePartitionCandidate:
        return self.selection_result.selected_model.partition_candidate

    @property
    def raw_reference(self) -> RawFusionCandidate:
        return self.selection_result.selected_model.raw_reference

    @property
    def partition_parent_raw(self) -> RawFusionCandidate | None:
        return self.selection_result.selected_model.partition_parent_raw

    @property
    def raw_fit(self) -> RawFit:
        return self.raw_reference.raw_fit

    @property
    def partition(self) -> FusionPartition | DirectPartition:
        return self.selected_candidate.partition

    @property
    def refit(self) -> PartitionRefitSummary:
        return self.selected_candidate.refit

    @property
    def score(self) -> SelectionScore:
        return self.selected_candidate.score


def analysis_summary(
    analysis: AnalysisSerialization,
    *,
    elapsed_seconds: float,
) -> dict[str, object]:
    """Serialize schema-v4 diagnostics without changing estimator state."""

    data = analysis.data
    fit_config = analysis.fit_config
    profile = fit_config.computation_profile
    result = analysis.selection_result
    selected_model = result.selected_model
    raw_reference = analysis.raw_reference
    raw_fit = analysis.raw_fit
    parent_raw = analysis.partition_parent_raw
    partition = analysis.partition
    refit = analysis.refit
    score = analysis.score
    selected_lambda = result.selected_lambda_representative
    optimum_resolved = bool(result.selection_optimum_resolved)
    selected_partition_certified = bool(
        isinstance(partition, FusionPartition) and partition.certified
    )
    exactness = raw_fit.certificate
    scalar_pilots = raw_fit.provenance.scalar_pilot_certificates

    return {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "tumor_id": data.tumor_id,
        "input_file": str(analysis.input_file),
        **input_model_summary(data),
        "scalar_pilot_coordinate_count": len(scalar_pilots),
        "scalar_pilot_certified_coordinate_count": sum(
            item.globally_certified for item in scalar_pilots
        ),
        "scalar_pilot_all_coordinates_certified": (
            all(item.globally_certified for item in scalar_pilots)
            if scalar_pilots else None
        ),
        "scalar_pilot_attained_value": (
            sum(item.attained_value for item in scalar_pilots)
            if scalar_pilots else None
        ),
        "scalar_pilot_optimality_gap": (
            sum(item.optimality_gap for item in scalar_pilots)
            if scalar_pilots else None
        ),
        "computation_profile": str(profile.name),
        "selection_status": (
            "resolved" if optimum_resolved else "provisional_unresolved"
        ),
        "selection_contract_id": str(fit_config.selection.contract_id),
        "selection_optimum_resolved": optimum_resolved,
        "selection_boundary_unresolved": bool(result.selection_boundary_unresolved),
        "selection_hits_lower_boundary": bool(result.selection_hits_lower_boundary),
        "selection_hits_upper_boundary": bool(result.selection_hits_upper_boundary),
        "selected_lambda": (
            None if selected_lambda is None else float(selected_lambda)
        ),
        "raw_reference_lambda": float(
            raw_fit.provenance.lambda_value
        ),
        "raw_reference_objective_certified": bool(
            raw_reference.raw_objective_certified
        ),
        "selected_candidate_family": str(selected_model.selected_candidate_family),
        "selected_partition_source": str(partition.source),
        "selected_partition_parent_lambda": (
            float(partition.parent_raw_lambda)
            if isinstance(partition, DirectPartition)
            and partition.parent_raw_lambda is not None
            else None
        ),
        "selected_partition_parent_phi_hash": (
            str(partition.parent_raw_phi_hash)
            if isinstance(partition, DirectPartition) and parent_raw is not None
            else ""
        ),
        "selected_partition_parent_signature": (
            str(parent_raw.partition.signature) if parent_raw is not None else ""
        ),
        "selected_n_clusters": int(partition.n_clusters),
        "selected_partition_signature": str(partition.signature),
        "selected_partition_certified": selected_partition_certified,
        "selected_labels_hash": _array_fingerprint(
            partition.labels,
            dtype=np.dtype(np.int64),
        ),
        "raw_reference_phi_hash": _array_fingerprint(
            raw_fit.phi,
            dtype=np.dtype(np.float64),
        ),
        "selected_fixed_partition_refit_centers_hash": _array_fingerprint(
            refit.cluster_centers,
            dtype=np.dtype(np.float64),
        ),
        "selection_score_name": str(score.name),
        "selection_score": float(score.value),
        "selection_score_numerical_uncertainty": float(score.numerical_uncertainty),
        "selection_loglik": float(score.loglik),
        "selection_df": int(score.degrees_of_freedom),
        "selection_penalty": float(score.penalty),
        "selection_n_eff": int(score.n_eff),
        "selection_assignment_log_evidence": float(score.assignment_log_evidence),
        "selection_assignment_code_weight": float(score.assignment_code_weight),
        "selection_assignment_penalty": float(score.assignment_penalty),
        "selection_assignment_dirichlet_alpha": float(
            score.assignment_dirichlet_alpha
        ),
        "selected_raw_penalized_objective": float(raw_fit.objective.total),
        "selected_refit_numerically_resolved": bool(refit.refit_numerically_resolved),
        "selected_refit_global_optimum_certified": bool(
            refit.global_optimum_certified
        ),
        "selected_refit_global_optimality_gap": float(refit.global_optimality_gap),
        "selected_refit_global_lower_bound": float(refit.global_lower_bound),
        "selected_refit_global_certificate_method": str(
            refit.global_certificate_method
        ),
        "selected_raw_solver_primal_tol": float(fit_config.solver.tolerance),
        "selected_full_kkt_tolerance": float(raw_fit.certificate.tolerance),
        "selected_full_kkt_residual_method": str(
            exactness.residual_method
        ),
        "selected_working_precision_kkt_residual": float(
            raw_fit.certificate.working_residual
        ),
        "selected_working_dtype": str(
            raw_fit.certificate.working_dtype
        ),
        "selected_certificate_audit_dtype": str(
            raw_fit.certificate.audit_dtype
        ),
        "selected_precision_polish_applied": bool(
            raw_fit.certificate.precision_polished
        ),
        "selected_precision_polish_max_abs_phi_delta": float(
            raw_fit.certificate.precision_polish_delta
        ),
        "selected_base_fusion_objective_hash": str(
            raw_fit.provenance.base_fusion_objective_hash
        ),
        "selected_original_graph_hash": str(raw_fit.provenance.original_graph_hash),
        "selection_method": str(result.selection_method),
        "num_candidates": int(result.num_candidates),
        "num_candidates_certified": int(result.num_candidates_certified),
        "ward_candidate_pool_complete": bool(result.ward_candidate_pool_complete),
        "raw_lambda_path_complete": bool(result.raw_lambda_path_resolved),
        "global_hybrid_optimum_certified": bool(
            result.global_hybrid_optimum_certified
        ),
        "selected_kkt_residual": (
            None
            if result.selected_kkt_residual is None
            else float(result.selected_kkt_residual)
        ),
        "search_stop_reason": str(result.adaptive_search_stop_reason),
        "device": str(raw_fit.provenance.device),
        "dtype": str(raw_fit.provenance.dtype),
        "elapsed_seconds": float(elapsed_seconds),
        "software_version": __version__,
    }


def write_analysis_outputs(
    analysis: AnalysisSerialization,
    *,
    outdir: Path,
    publication: RunPublication | None = None,
) -> None:
    """Write the retained-mutation tables and original-CN exclusion audit."""
    own_publication = publication is None
    if publication is None:
        publication = RunPublication(outdir, analysis.data.tumor_id,
                                     input_file=analysis.input_file, fit_config=analysis.fit_config)
    if publication.outdir.resolve() != Path(outdir).resolve() or publication.tumor_id != analysis.data.tumor_id:
        raise ValueError("Publication does not belong to this tumor and output directory.")
    try:
        _write_fit_tables(
            analysis.data, analysis.raw_fit, analysis.partition, analysis.refit,
            float(analysis.fit_config.eps), publication,
            selection_result=analysis.selection_result,
        )
    except BaseException as error:
        if own_publication:
            publication.fail(error)
        raise


__all__ = [
    "AnalysisSerialization",
    "RunPublication",
    "SUMMARY_SCHEMA_VERSION",
    "analysis_summary",
    "write_analysis_outputs",
    "cluster_output_table",
    "cn_filter_output_table",
    "mutation_output_table",
    "mutation_region_output_table",
    "write_cn_filter_output",
    "write_fit_outputs",
]
