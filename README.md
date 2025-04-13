
# SCAD-Penalized ADMM for Multi-region Subclone Reconstruction

This project implements a research-oriented pipeline to perform SCAD-penalized ADMM for subclone reconstruction. It supports both single-sample \((M=1)\) and multi-sample \((M>1)\) scenarios. The code is structured to load data from multiple directories (each representing a “region” or “sample”), build logistic-scale parameters, and then run an ADMM loop with SCAD-based thresholding to cluster SNVs. Finally, it merges clusters if needed and outputs the final group assignments.

## Table of Contents
1. [Introduction](#introduction)  
2. [Directory Layout](#directory-layout)  
3. [Dependencies](#dependencies)  
4. [Running the Code](#running-the-code)  
5. [Script Overview](#script-overview)  
6. [Contact](#contact)

---

## Introduction

**Goal:**  
Subclone reconstruction by grouping single nucleotide variants (SNVs) that share a similar frequency pattern, either in single-sample or multi-sample contexts. The penalty is a group-SCAD that encourages merging of pairwise differences in the logistic-scale parameters.

**Key Idea:**  
1. **Logistic Transform**: We parametrize each SNV’s frequency in a logit scale.  
2. **SCAD Penalty**: We apply a piecewise, nonconvex SCAD approach on the pairwise differences.  
3. **CliPP2**: The Alternating Direction Method of Multipliers separates the likelihood (IRLS approximation) from the nonconvex penalty (for detailed formulations, you need to check **CliPP2.pdf**).  

---

## Directory Layout

```
YOUR_PROJECT/
  ├── input_files/
  │   ├── regionA/
  │   │    ├── r.txt
  │   │    ├── n.txt
  │   │    ├── minor.txt
  │   │    ├── total.txt
  │   │    ├── purity_ploidy.txt
  │   │    ├── coef.txt
  │   └── regionB/ 
  │        └── ...
  ├── preprocess_result/
  ├── input_files/
  │   ├── core.py
  │   ├── preprocess.R
  ├── README.md
  └── ...
```

- **`input_files/`**: Example or raw input data with one subdirectory per region (`regionA`, `regionB`, etc.).  
- **`preprocess_result/`**: (Optional) where you might store processed data.  
- **`scad_admm.py`**: The main script containing your SCAD-based ADMM code.  
- **`README.md`**: This documentation file.

---

## Dependencies

- **Python 3.7+**
- **NumPy** (for arrays, linear algebra)
- **SciPy** (for sparse matrices, linear solvers)
- **`scipy.sparse`** and **`scipy.sparse.linalg`** (for building/solving large, sparse systems)
- **`subprocess`, `os`, etc.** for directory management (optional)

Install packages:

```bash
pip install numpy scipy
```

---

## Running the Code

1. **Organize Data**:  
   Make sure each “region” subdirectory has `r.txt, n.txt, minor.txt, total.txt, purity_ploidy.txt, coef.txt`. All regions must have the same number of SNVs (rows).

2. **Load and Prepare**:  
   The function `group_all_regions_for_ADMM(root_dir)` reads from all subdirectories in `root_dir` and returns stacked arrays `(r, n, minor, total, purity, ploidy, coef_list, wcut)`.

3. **Call the CliPP2**:  
   In your script or an interactive session:

   ```python
   from clipp2.core import *

   # Step 1: gather data
   root_dir = "input_files"
   (r, n, minor, total,
    purity, ploidy,
    coef_list,
    wcut) = group_all_regions_for_ADMM(root_dir)

   # Step 2: run CliPP2
   result = clipp2(
       r, n, minor, total,
       purity, ploidy,
       coef_list,
       wcut=wcut,
       alpha=1.0, gamma=3.7, rho=0.95, precision=0.01,
       Run_limit=1000, control_large=5, Lambda=0.01,
       post_th=0.05, least_diff=0.01
   )

   # result is a dict with final 'phi' and 'label'
   print("Final logistic-scale subclone means:", result['phi'])
   print("Cluster assignments:", result['label'])
   ```

4. **Results**:  
   - `result['phi']`: The final logistic-scale estimates (or cluster means) for each SNV.  
   - `result['label']`: Cluster labels for each SNV, indicating subclone membership.

---

## Script Overview

Below is a brief outline of the key functions in `scad_admm.py` (or whichever script name you use):

1. **`sort_by_2norm(x)`**  
   Sorts the rows of `x` by L2 norm.

2. **`find_min_row_by_2norm(x)`**  
   Identifies the row in `x` with the smallest L2 norm.

3. **`group_all_regions_for_ADMM(root_dir)`**  
   - Searches each subdirectory in `root_dir`.  
   - Reads `r.txt, n.txt, minor.txt, total.txt, purity_ploidy.txt, coef.txt`.  
   - Stacks horizontally into shape `(No_mutation, M)`.  
   - Returns `(r, n, minor, total, purity_list, ploidy_list, coef_list, wcut)`.

4. **`soft_threshold_group(...)` & `SCAD_group_threshold(...)`**  
   Core piecewise definitions for group SCAD thresholding in the ADMM updates.

5. **`build_DELTA_multiM(No_mutation, M)`**  
   Builds a sparse operator that, when multiplied by flattened `w`, yields pairwise differences `(w_i - w_j)` for every `(i<j)` and every coordinate in `[0..M-1]`.

6. **`initialize_w(...)`**  
   Creates an initial guess for logistic-scale parameters with copy-number/purity adjustments.

7. **`reshape_eta_to_2D(...)`**  
   Initializes `eta` by flattening the initial `w_new` and extracting upper-triangle differences.

8. **`diff_mat(...)`**  
   Builds a “signed” difference matrix for multi-sample data:  
   - L2 norm across all M coordinates,  
   - sign determined by the difference in the first coordinate.

9. **`clipp2(...)`**  
   - The main ADMM logic:
     1. IRLS expansions => build `A_array, B_array` => flatten.  
     2. Solve `(B^T B + alpha * DELTA^T DELTA)*w = linear`.  
     3. SCAD threshold => update `eta`.  
     4. Repeat until residual < `precision` or `Run_limit`.  
   - Post-processing: merges clusters if 2-norm difference < `least_diff`.  
   - Returns `{'phi': phi_res, 'label': class_label}`.

---

## Contact

- **Author: Yu Ding, Ph.D. / Wenyi Wang's Lab / MD Anderson Cancer Center**  
- **Date: Oct 2024**
- **Contact: yding4@mdanderson.org, yding1995@gmail.com**  
- Contributions, bug reports, and feature requests are welcome.  

Feel free to adapt this pipeline to your own subclone modeling or SCAD-penalized tasks.  