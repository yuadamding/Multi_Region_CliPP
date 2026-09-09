# CliPP2

CliPP2 estimates mutation cancer-cell fractions (CCFs), clusters SNVs, and
infers mutant-copy multiplicity from single- or multi-region tumor sequencing
data with observed-data pairwise fusion.

## Install

```bash
pip install .
```

## Input

The public input is one tab-delimited file per tumor. See [`examples/exampleTumor1.tsv`](examples/exampleTumor1.tsv).

The loader validates the complete input, aggregates identical local CN states,
then excludes an entire mutation across all samples if **any** sample has more
than one distinct positive-fraction CN state or major CN greater than 6. It
does not mask individual observations or change the input file. Different
clonal states between samples are allowed; `(6,6)` is retained. A retained
`(0,0)` state is unsupported. If all mutations are excluded, fitting stops
with `NoEligibleSNVsError` and, when outputs are enabled, an exclusion audit.

Each retained mutation–sample unit has the distinct multiplicities
`1, …, major_cn`, each with equal prior probability. Fitting and fixed-label
refitting retain the **marginalized** binomial likelihood (log-sum-exp over
candidates), not hard multiplicity selection. The denominator remains
`(1-purity)*normal_cn + purity*(major_cn+minor_cn)`. No anchor or CCF-one block
is imposed; the fusion penalty and numerical admission thresholds are unchanged.

The final integer call is the posterior mode **conditional on the reported
fixed-partition CCF**. Exact ties choose the smallest candidate. Missing or
zero-depth observations have no informative call: multi-candidate calls are
missing, while a singleton is structurally fixed at one. Candidate expansion
does not resolve every multiplicity/CCF ambiguity.

## Fit

Fit on CUDA (the bundled mixed-CN example retains 101 of its 300 mutations):

```bash
clipp2 fit \
  --input-file examples/exampleTumor1.tsv \
  --outdir exampleTumor1_results
```


Use `--device cpu` on a CPU-only machine. Run `clipp2 fit --help` for profile,
solver, resource, selection-score, and partition-tolerance controls.

The new integer workflow has fixed uniform priors. `--dosage-prior-penalty`
and `--unsupported-policy mask` are rejected; nondefault `--major-prior` is
also rejected. The fixed major-CN cutoff is not a runtime option. Public
programmatic fits require data from `load_tumor_txt` with its CN-filter report;
dominant-CN arrays alone cannot establish original CN eligibility.

This revision is based on v0.3.4 (`fa52ecf`) and changes the likelihood's
candidate support and prior. Previous binary-mixture benchmark results must
not be attributed to this revised estimator. The existing partition-selection
score and its penalty settings are unchanged and need fresh calibration on
simulation; expanded candidates are not counted as extra continuous parameters.

## Outputs

A fit writes four tables into `--outdir`, prefixed with the tumor id (the input
file stem unless a `##tumor_id` metadata line overrides it):

| File | One row per | Main fields |
| --- | --- | --- |
| `{tumor_id}_mutation_clusters.tsv` | mutation | selected cluster and final fixed-partition CCF per region |
| `{tumor_id}_cluster_centers.tsv` | selected cluster | size and final CCF per region |
| `{tumor_id}_mutation_region_multiplicity.tsv` | retained mutation × region | final CCF, CN, integer MAP call and conditional candidate probabilities |
| `{tumor_id}_excluded_mutations.tsv` | triggering mutation × sample × reason | segment, exclusion reason, distinct CN-state count, maximum major CN |

Integer output fields include `multiplicity_candidates`,
`multiplicity_candidate_count`, nullable integer `multiplicity_call`,
`multiplicity_call_probability`, `multiplicity_informative`, and
`multiplicity_p1` through `multiplicity_p6` (invalid candidates have zero mass).
Binary major/low and occupancy-switch summaries are not emitted for the new
model. Exclusion reason counts can overlap; the schema-v4 stdout summary
separately records unique input, retained, and excluded mutation counts plus
the filtering policy, candidate generator, model ID, and prior mode.

## Simulation

Generate a matched, exact-size tumor:

```bash
clipp2 simulate --out-dir simulations --tumor-id simulatedTumor1 \
  --mutation-count 300 --clone-count 3 --region-count 2 --seed 1
```

The simulator uses **clonal trunk gains**, inherited unchanged by every
descendant clone, with major CN at most 6. Repeated gains are allowed; SNV
acquisition timing and physical-copy inheritance determine integer truth
multiplicity. Descendant SNVs arise after the shared gains and have multiplicity
one. Evolutionary multiplicities are not sampled from the inference model's
uniform candidate prior.

All requested mutations must survive the loader's whole-mutation CN filter.
Validation rejects exclusions or mismatched truth; it never silently drops
truth rows. `--cna-event-rate` controls trunk gains; the obsolete two-state
quota and descendant-CN controls are removed. The bundle includes canonical
input, tree/CN/CCF truth, integer `truth_mutation_sample.tsv:multiplicity`, and
a schema-6 manifest with generator/source identity, hashes and retained counts.
`effective_multiplicity` remains as a truth cross-check and equals that integer.

This is a narrower, matched benchmark—not a simulator of subclonal CN,
deletions/LOH, or different clonal CN between regions. Those eligibility and
boundary cases require separate tests. Generator v6 results must not be pooled
with the former two-state v5 simulations. Evaluate multiplicity primarily on
`major_cn != minor_cn` rows using pooled exact-class macro-F1, also reporting
micro/weighted/per-class F1 and eligible row count.
