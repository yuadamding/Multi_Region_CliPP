"""Public workflow integration for the fa52ecf integer-mixture revision."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from CliPP2.cli import parse_args
from CliPP2.config import resolve_fit_config
from CliPP2.api import fit_fixed_objective, validate_public_tumor_data
from CliPP2.core.objective import compile_observed_model, observed_terms_numpy
from CliPP2.core.scalar import partition_constrained_observed_refit
from CliPP2.io.tumor_txt import NoEligibleSNVsError, load_tumor_txt
from CliPP2 import api as pipeline
from CliPP2.reporting import SUMMARY_SCHEMA_VERSION, input_model_summary


def _write(path: Path, *, exclude_only: bool = False) -> Path:
    rows = []
    if not exclude_only:
        for mutation, major, alt in [('keep1', 4, 45), ('keep2', 2, 25)]:
            rows.append(dict(mutation_id=mutation, sample_id='s1', alt_count=alt,
                ref_count=100-alt, count_observed=1, purity=1.0, normal_cn=2,
                segment_id=mutation, cn_state_id='c1', cn_state_fraction=1.0,
                allele_a_cn=major, allele_b_cn=1))
    for state, major, fraction in [('c1', 7, .6), ('c2', 2, .4)]:
        rows.append(dict(mutation_id='drop', sample_id='s1', alt_count=20,
            ref_count=80, count_observed=1, purity=1.0, normal_cn=2,
            segment_id='drop', cn_state_id=state, cn_state_fraction=fraction,
            allele_a_cn=major, allele_b_cn=0))
    pd.DataFrame(rows).to_csv(path, sep='\t', index=False)
    return path


@pytest.mark.parametrize('write_outputs', [False, True])
def test_no_eligible_snv_stops_before_solver_with_optional_audit(tmp_path, monkeypatch, write_outputs):
    path = _write(tmp_path / 'empty.tsv', exclude_only=True)
    original = path.read_bytes()
    monkeypatch.setattr('CliPP2.model_selection.search.select_model', lambda **kw: pytest.fail('solver must not run'))
    outdir = tmp_path / 'out'
    with pytest.raises(NoEligibleSNVsError) as failure:
        pipeline.process_tumor(path, outdir, write_outputs=write_outputs)
    assert failure.value.cn_filter_report.excluded_mutation_ids == ('drop',)
    audit = outdir / 'empty_excluded_mutations.tsv'
    assert audit.exists() is write_outputs
    if write_outputs:
        table = pd.read_csv(audit, sep='\t')
        assert set(table.reason) == {'SUBCLONAL_CN_REGION', 'MAJOR_CN_GT_6'}
        assert len(list(outdir.iterdir())) == 2
    assert path.read_bytes() == original


def test_pipeline_filters_and_resolves_eps_before_search(tmp_path, monkeypatch):
    path = _write(tmp_path / 'tumor.tsv')
    config = resolve_fit_config(device='cpu', eps=.02)
    outdir = tmp_path / 'out'

    def stop_at_search(*, data, fit_config, **kwargs):
        assert data.mutation_ids == ('keep1', 'keep2')
        assert data.phi_init.min() >= config.eps
        assert fit_config is config
        validate_public_tumor_data(data, config)
        assert (outdir / 'tumor_excluded_mutations.tsv').exists()
        raise RuntimeError('test search boundary reached')

    monkeypatch.setattr('CliPP2.model_selection.search.select_model', stop_at_search)
    with pytest.raises(RuntimeError, match='test search boundary reached'):
        pipeline.process_tumor(path, outdir, fit_config=config)


def test_filter_summary_counts_overlap_without_double_counting(tmp_path):
    data = load_tumor_txt(_write(tmp_path / 'summary.tsv'))
    summary = input_model_summary(data)
    assert SUMMARY_SCHEMA_VERSION == 5
    assert summary['input_mutation_count'] == 3
    assert summary['retained_mutation_count'] == 2
    assert summary['excluded_mutation_count'] == 1
    assert summary['excluded_subclonal_cn_mutation_count'] == 1
    assert summary['excluded_major_cn_gt6_mutation_count'] == 1
    assert summary['multiplicity_prior_mode'] == 'uniform_distinct_integer_v1'


@pytest.mark.parametrize('option', [
    ['--major-prior', '.5'], ['--major-prior', '.7'], ['--major-prior', 'nan'],
    ['--dosage-prior-penalty', '0'], ['--dosage-prior-penalty', '3'],
    ['--unsupported-policy', 'error'], ['--unsupported-policy', 'mask'],
])
def test_cli_rejects_obsolete_model_overrides(option):
    with pytest.raises(SystemExit) as failure:
        parse_args(['fit', '--input-file', 'unused.tsv', *option])
    assert failure.value.code == 2


def test_public_fit_rejects_unvalidated_or_legacy_objects(tmp_path, monkeypatch):
    data = load_tumor_txt(_write(tmp_path / 'public.tsv'))
    config = resolve_fit_config(device='cpu')
    monkeypatch.setattr('CliPP2.api.fit_prepared',
                        lambda **kw: pytest.fail('must reject before solver'))
    with pytest.raises(ValueError, match='legacy or unvalidated'):
        fit_fixed_objective(replace(data, cn_filter_report=None), config)
    with pytest.raises(TypeError, match='path_likelihood'):
        replace(data, path_likelihood=None)
    with pytest.raises(ValueError, match='inconsistent'):
        fit_fixed_objective(replace(data, cn_filter_report=replace(
            data.cn_filter_report, retained_mutation_count=12)), config)


def test_public_fit_rejects_loader_fit_eps_mismatch(tmp_path, monkeypatch):
    path = _write(tmp_path / 'eps.tsv')
    table = pd.read_csv(path, sep='\t')
    table['allele_b_cn'] = 0
    table.to_csv(path, sep='\t', index=False)
    data = load_tumor_txt(path, eps=1e-6)
    config = resolve_fit_config(device='cpu', eps=.02)
    monkeypatch.setattr('CliPP2.api.fit_prepared',
                        lambda **kw: pytest.fail('must reject before solver'))
    with pytest.raises(ValueError, match='reload the input'):
        fit_fixed_objective(data, config)
    validate_public_tumor_data(load_tumor_txt(path, eps=.02), config)


@pytest.mark.parametrize('exclude_only', [False, True])
@pytest.mark.parametrize('suffix', [
    'excluded_mutations.tsv', 'mutation_clusters.tsv',
    'cluster_centers.tsv', 'mutation_region_multiplicity.tsv',
])
def test_output_cannot_overwrite_input(tmp_path, monkeypatch, exclude_only, suffix):
    path = _write(tmp_path / f'protected_{suffix}', exclude_only=exclude_only)
    path.write_text('##tumor_id=protected\n' + path.read_text())
    original = path.read_bytes()
    monkeypatch.setattr('CliPP2.model_selection.search.select_model', lambda **kw: pytest.fail('must reject first'))
    with pytest.raises(ValueError, match='overwrite original input'):
        pipeline.process_tumor(path, tmp_path)
    assert path.read_bytes() == original
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize('link_type', ['symlink', 'hardlink'])
def test_output_aliases_cannot_overwrite_input(tmp_path, link_type):
    path = _write(tmp_path / 'protected.tsv')
    original = path.read_bytes()
    destination = tmp_path / 'protected_excluded_mutations.tsv'
    if link_type == 'symlink':
        destination.symlink_to(path)
    else:
        destination.hardlink_to(path)
    with pytest.raises(ValueError, match='overwrite original input'):
        pipeline.process_tumor(path, tmp_path)
    assert path.read_bytes() == original


def test_fixed_partition_refit_uses_marginal_integer_likelihood(tmp_path):
    data = load_tumor_txt(_write(tmp_path / 'refit.tsv'))
    labels = np.zeros(data.num_mutations, dtype=np.int64)
    refit = partition_constrained_observed_refit(data, labels,
        eps=1e-6, tol=1e-4, max_iter=2048)
    model = compile_observed_model(data, eps=1e-6)
    np.testing.assert_array_equal(refit.labels, labels)
    assert refit.phi.shape == (2, 1)
    assert np.ptp(refit.phi[:, 0]) == 0
    terms = observed_terms_numpy(model, refit.phi, eps=1e-6)
    np.testing.assert_allclose(refit.loglik, -np.sum(terms.loss), rtol=1e-12)
