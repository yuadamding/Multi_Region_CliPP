# Multi-Region CliPP: README

**Authors**: Yu Ding  
**Date**: 10/29/2024  
**Email**: [yding1995@gmail.com](mailto:yding1995@gmail.com), [yding4@mdanderson.org](mailto:yding4@mdanderson.org)

This repository provides **multi-region CliPP (multi-region Clonal structure identification through penalizing pairwise differences)** analysis in Python. It uses augmented Lagrangian / ADMM methods to infer the subclonal structure of tumors by integrating mutation (SNV) counts, copy numbers, and purity across multiple samples or regions.

---

## Overview

1. **Preprocessing**: The `preprocess_for_clipp2` function implements an R-style filtering pipeline (similar to "clipp1"):
   - Filters out invalid chromosome indices or read counts.
   - Matches each SNV to its copy number segment.
   - Calculates multiplicities.
   - Performs piecewise-linear approximations of \(\theta\).
   - Removes outlier SNVs based on \(\phi\).

2. **Building CliPP2 Input**: The `build_cliPP2_input` function converts the preprocessed SNVs into a format that the core ADMM solver (`CliPP2`) expects (i.e., columns for mutation ID, alt/ref counts, CN values, purity, etc.).

3. **ADMM Solver**:  
   - `CliPP2` is defined as a Ray remote function for parallel or distributed execution.
   - It handles pairwise constraints among SNVs, iteratively updates primal (`p`) and dual (`y`), and uses `v` for penalizing differences between SNVs (or clustering).
   - Key subroutines include `a_mat_generator`, `get_loglikelihood`, `update_v_SCAD`, and so on.

4. **`run_preproc_and_CliPP2`**: An end-to-end pipeline that
   1. Preprocesses SNVs + CN data,
   2. Builds the required input DataFrame,
   3. Runs `CliPP2` for a list of `gamma` values.

---

## Code Contents

1. **Imports**  
   Uses `numpy`, `pandas`, `scipy`, `torch` for linear algebra, and `ray` for parallel computations.

2. **Core Utility Functions**

   - **Sigmoid and its derivatives** (`sigmoid`, `sigmoid_derivative`, `sigmoid_second_derivative`)  
   - **`a_mat_generator`** & **`a_trans_a_mat_generator_quick`**: Create linear constraints for pairwise SNV differences in the ADMM updates.  
   - **`matmul_by_torch`, `mat_inverse_by_torch`, `sum_by_torch`**: Wrapper functions that apply PyTorch operations (and return NumPy arrays).  
   - **`theta`, `linear_evaluate`, `linear_approximate`**: For piecewise approximation of the \(\theta\) function in the R code.  
   - **CN-related** (e.g., `get_major_cn`, `get_c_mat`) and read-count–related (e.g., `get_read_mat`) utility routines that reshape or compute essential matrices.

3. **`CliPP2(...)`**  
   A Ray remote function implementing the ADMM solver. Its main steps are:
   - Construct pairwise combinations among SNVs (`combinations_2`).
   - Compute read matrices, copy number matrices, etc.
   - Initialize `p`, `v`, and `y`.
   - Iteratively update `p` (in `update_p`), then update `v` (`update_v_SCAD` or `update_v`), and finally update dual variable `y`.
   - Perform optional cluster refinement and merging steps (e.g., reassigning small clusters, computing centroid \(\phi\), merging clusters with similar scalar values).

   Returns a dictionary with final cluster labels, cell prevalence matrix, and purity.

4. **Preprocessing** (`preprocess_for_clipp2`)  
   - Cleans up SNVs, matches to CN segments, calculates multiplicities.
   - Applies a piecewise-linear approximation filter on \(\theta\).
   - Excludes SNVs based on out-of-range \(\phi\).
   - Returns a dictionary with final arrays: `minor_read`, `total_read`, `coef`, etc.

5. **`build_cliPP2_input(...)`**  
   Converts preprocessed SNV data into a DataFrame with columns required by the ADMM solver (`mutation`, `alt_counts`, `ref_counts`, `major_cn`, `minor_cn`, `normal_cn`, `tumour_purity`). Also returns `n, m`.

6. **`run_preproc_and_CliPP2(...)`**  
   A high-level function to:
   1. Preprocess SNVs via `preprocess_for_clipp2`.
   2. Build the ADMM-compatible DataFrame via `build_cliPP2_input`.
   3. Initialize Ray, run `CliPP2` across multiple `gamma` values, and collect final results.

---

## Usage

### Installation

```bash
pip install numpy pandas scipy torch ray
```
## Example
```
import pandas as pd
from clipp2_module import run_preproc_and_CliPP2

snv_df = pd.read_csv("my_snv_input.tsv", sep="\t")  # must have columns chromosome_index, position, alt_count, ref_count, ...
cn_df  = pd.read_csv("my_cnv_input.tsv", sep="\t")  # must have columns [chr, start, end, minor_cn, major_cn, total_cn]
purity = 0.6

results = run_preproc_and_CliPP2(
    snv_df, cn_df,
    purity=purity,
    sample_id="SampleA",
    gamma_list=[0.01, 0.1, 1],  # tries multiple gamma values
    rho=0.8, omega=1.0
)

{
  'phi'   : 2D array of shape (n, m),
  'label' : array of shape (n,) with cluster labels,
  'purity': float
  ...
}
```

