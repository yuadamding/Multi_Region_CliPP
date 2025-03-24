# Multi-Region CliPP

**Authors**: Yu Ding  
**Date**: 10/29/2024  
**Email**: [yding1995@gmail.com](mailto:yding1995@gmail.com), [yding4@mdanderson.org](mailto:yding4@mdanderson.org)

This repository provides **multi-region CliPP (Clonal structure identification through penalizing pairwise differences)** analysis in Python. It uses augmented Lagrangian / ADMM methods to infer the subclonal structure of tumors by integrating mutation (SNV) counts, copy numbers, and purity across multiple samples or regions.

---

## Directory Structure
clipp2\
├── init.py\
├── math_utils.py\
├── matrix_ops.py\
├── df_accessors.py\
├── approx_utils.py\
├── admm.py\
└── pipeline.py

1. **math_utils.py**  
   Contains core mathematical functions such as logistic/sigmoid transformations, their derivatives, inverse, clipping arrays, etc.

2. **matrix_ops.py**  
   Matrix-building utilities (e.g., Kronecker products for pairwise differences) and PyTorch-based linear algebra helpers.

3. **df_accessors.py**  
   Functions that extract or compute matrices and arrays (copy numbers, read counts, purity) from a DataFrame.

4. **approx_utils.py**  
   Piecewise-linear approximations of logistic-like functions (e.g., \(\theta(w)\)), plus helper logic for segment-based approximations.

5. **admm.py**  
   ADMM-related routines (objective function, gradient computation, SCAD-based updates, etc.) crucial for the CliPP2 solver.

6. **pipeline.py**  
   High-level functions that orchestrate the entire process:
   - `CliPP2(...)` for the main ADMM solver
   - `preprocess_for_clipp2(...)` for R-style data filtering
   - `build_cliPP2_input(...)` to structure SNVs for the solver
   - `run_preproc_and_CliPP2(...)` as an end-to-end pipeline

---

## Overview

- **Preprocessing**:  
  The function `preprocess_for_clipp2(...)` provides R-style filters (removing invalid SNVs, matching CN segments, computing multiplicities, etc.). It also performs piecewise-linear approximations and outlier filtering.

- **Building CliPP2 Input**:  
  After preprocessing, `build_cliPP2_input(...)` converts the SNV data into a DataFrame that the core ADMM solver (`CliPP2`) expects: columns for mutation indices, alt/ref counts, copy numbers, and purity.

- **ADMM Solver**:  
  `CliPP2(...)` uses the pairwise difference constraints among SNVs, constructs and updates auxiliary variables (`v`), dual variables (`y`), and the main logit-scale parameters (`p`). It supports SCAD-based shrinkage (`update_v_SCAD`) for grouping/clustering SNVs with small differences.

- **Full Pipeline**:  
  `run_preproc_and_CliPP2(...)` ties everything together. It:
  1. Runs preprocessing on SNVs + copy-number data.
  2. Builds the solver-ready DataFrame.
  3. Invokes `CliPP2` for a list of `gamma` values in parallel (via Ray).

---

## How to Use

1. **Installation**  
Ensure that `numpy`, `pandas`, `scipy`, `torch`, and `ray` are installed:
```bash
pip install numpy pandas scipy torch ray
```

Then place the clipp2/ directory in your Python path (for example, in your project folder).

2.	**Imports**
```bash
from clipp2.pipeline import run_preproc_and_CliPP2
```
Or 
```bash
import clipp2.pipeline as pipeline
```

3.	**Usage Example**
```bash
import pandas as pd
from clipp2.pipeline import run_preproc_and_CliPP2

snv_df = pd.read_csv("my_snv_input.tsv", sep="\t")
cn_df  = pd.read_csv("my_cnv_input.tsv", sep="\t")
purity = 0.6

results = run_preproc_and_CliPP2(
    snv_df=snv_df,
    cn_df=cn_df,
    purity=purity,
    sample_id="SampleA",
    gamma_list=[0.01, 0.1, 1.0],
    rho=0.8,
    omega=1.0,
    max_iteration=1000,
    precision=1e-2,
    control_large=5,
    valid_snvs_threshold=0,
    diff_cutoff=0.1
)

# 'results' is a list of outputs, one per gamma in gamma_list:
# [
#   {'phi': (n x m array), 'label': (n, ), 'purity': float, 'p': ...},
#   ...
# ]
for i, gamma_val in enumerate([0.01, 0.1, 1.0]):
    res = results[i]
    print(f"Gamma={gamma_val}, clusters={res['label']}, purity={res['purity']}")
```
---
**Notes**

For multi-region scenarios (m>1), each SNV should be replicated across rows in the input DataFrame, and preprocess_for_clipp2(...) plus the solver can be adapted accordingly.

The ADMM approach will produce grouped or clustered SNVs when SCAD or L1 penalties are applied. Fine-tune gamma and rho for desired regularization strength or convergence properties.

---
**Contact**

For questions, suggestions, or issues, please email:\
Yu Ding (yding1995@gmail.com, yding4@mdanderson.org)