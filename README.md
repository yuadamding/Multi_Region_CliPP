<<<<<<< HEAD
# SCAD-Penalized ADMM for Multi-region Subclonal Reconstruction

This project implements a **CUDA-accelerated**, research-oriented pipeline to perform **SCAD-penalized ADMM** for subclonal reconstruction. It supports both single-sample (M=1) and multi-sample (M>1) scenarios.

The newest version removes the need to build large dense matrices on CPU, instead using **iterative Conjugate Gradient** (CG) on the GPU to solve the ADMM subproblems efficiently, relying on **vectorized** or **sparse** operations in PyTorch.

See [CliPP2.pdf](/CliPP2.pdf) and [gpu_implementation.pdf](/gpu_implementation.pdf) for theoretical details.

---
=======
# SCAD-Penalized ADMM for Multi-region subclonal Reconstruction

This project implements a **CUDA-accelerated**, research-oriented pipeline to perform **SCAD-penalized ADMM** for subclonal reconstruction. It supports both single-sample(M=1) and multi-sample(M>1) scenarios. The newest version removes the need to build large dense matrices on CPU, instead using **iterative Conjugate Gradient** (CG) on the GPU to solve the ADMM subproblems efficiently, relying on **vectorized** or **sparse** operations in PyTorch. See [CliPP2.pdf](/CliPP2.pdf) and [gpu_implementation.pdf](/gpu_implementation.pdf) for details.
>>>>>>> 880d373a1d0cdce96c236b4194ad360014ebaaaf

## Table of Contents

1. [Introduction](#introduction)  
2. [Directory Layout](#directory-layout)  
3. [Dependencies](#dependencies)  
4. [Running the CUDA ADMM Pipeline](#running-the-cuda-admm-pipeline)  
5. [General Multi-sample CLiPP2 Pipeline](#general-multi-sample-clipp2-pipeline)  
6. [Script Overviews](#script-overviews)  
7. [Contact](#contact)

---

## Introduction

**Goal:**  
<<<<<<< HEAD
Subclonal reconstruction by grouping single nucleotide variants (SNVs) that share similar frequency patterns, in either a single-sample or multi-sample context. The penalty is a group-SCAD that encourages merging of pairwise differences in logistic-scale parameters.
=======
subclonal reconstruction by grouping single nucleotide variants (SNVs) that share similar frequency patterns, in either a single-sample or multi-sample context. The penalty is a group-SCAD that encourages merging of pairwise differences in logistic-scale parameters.
>>>>>>> 880d373a1d0cdce96c236b4194ad360014ebaaaf

**Key CUDA Implementation Highlights:**
1. GPU (CUDA) acceleration via PyTorch tensors and sparse operations.  
2. Iterative Conjugate Gradient solves the linear subproblem without forming a dense Hessian.  
3. No Python `for` loops in main numeric kernels—pairwise differences via sparse multiplication.  
4. Scales to large mutation × sample sizes due to low memory footprint.

---

## Directory Layout

```
<<<<<<< HEAD
MULTI_REGION_CLIPP2/         # root directory
├── clipp2/                  # core library
│   ├── core.py              # SCAD-ADMM implementation
│   ├── preprocess.py        # data I/O & preprocessing
│   └──...
├── input_files/             # sample subdirectories
│   ├── region1/             # e.g. region1/
│   │    ├── snv.txt
│   │    ├── cna.txt
│   │    ├── purity.txt
│   └── region2/ ...
├── preprocess_result/       # (optional) processed data
├── README.md                # this documentation
└── gpu_implementation.pdf   # GPU design details
```

=======
MUITI_REGION_CLIPP/
  ├── clipp2/
  │   ├── core.py/
  │   ├── core_cuda.py/
  │   ├── preprocess.py/
  │   └── ...
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
  ├── README.md
  └── ...
```

- **`input_files/`**: Example or raw input data with one subdirectory per region (`regionA`, `regionB`, etc.).  
- **`preprocess_result/`**: (Optional) where you might store processed data.  
- **`README.md`**: This documentation file, which you are reading now.

>>>>>>> 880d373a1d0cdce96c236b4194ad360014ebaaaf
---

## Dependencies

<<<<<<< HEAD
- Python 3.7+  
- NumPy  
- SciPy  
- PyTorch (>=1.10; PyTorch 2.0+ for `torch.sparse.linalg.cg`)  
=======
- **Python 3.7+**
- **NumPy**  
- **SciPy**  
- **PyTorch (>=1.10)** or ideally **PyTorch 2.0+** for advanced CUDA/sparse ops
  - If you want to use the **built-in** `torch.sparse.linalg.cg`, you need **PyTorch ≥2.0** (experimental).
  - Otherwise, a **manual CG** loop is provided.
>>>>>>> 880d373a1d0cdce96c236b4194ad360014ebaaaf

Install example:
```bash
pip install numpy scipy torch
```

---


## General Multi-sample CliPP2 Pipeline

A higher-level script `CliPP2_pipeline.py` wraps pre-/post-processing, loci alignment, and clustering end-to-end:

```bash
#!/usr/bin/env python3
python CliPP2_pipeline.py \
  --input_dir input_files \
  --output_root output \
  --Lambda 0.1 \
  --subsample_rate 1.0 \
  --device cuda
```

**Steps performed:**
1. Scans `input_files/` for sample subdirs containing `*.snv.txt`, `*.cna.txt`, and `*.purity.txt`.  
2. Processes each region into DataFrames and aligns loci across samples.  
3. Exports and preprocesses per-region SNV/CNA/purity into intermediate files.  
4. Reads matrices `r, n, minor, total, purity, coef_list` via `group_all_regions_for_ADMM()`.  
5. (Optional) Subsampling at rate `--subsample_rate` for large M.  
6. Runs GPU‐accelerated `clipp2()` ADMM with SCAD penalty.  
7. Computes raw CP (`phi_hat`) values, merges cluster labels, and builds a combined result table.  
8. Outputs `result.txt` with columns:
   - `chromosome_index`, `position`, `region`,  
   - `label` (cluster ID),  
   - `phi` (ADMM estimate),  
   - `phi_hat` (raw CCF),  
   - `dropped` (0/1).  
9. Cleans up temporary work files.

<<<<<<< HEAD
Results are saved in `output/result.txt` and `output/intermediate/all_loci.txt`.
=======
   root_dir = "input_files"
   r, n, minor, total, purity, coef_list, wcut, drop_rows = group_all_regions_for_ADMM(root_dir)

   # Optionally pick a GPU device
   result = clipp2(
       r, n, minor, total,
       purity,
       coef_list,
       wcut=wcut,
       alpha=0.8, gamma=3.7, rho=1.02, precision=0.01,
       Run_limit=1e4, control_large=5, Lambda=0.01,
       post_th=0.05, least_diff=0.01,
       device='cuda'    # GPU acceleration
   )

   print("Final subclonal logistic-scale means (phi):", result['phi'])
   print("Cluster assignments (label):", result['label'])
   ```

   - Set `device='cuda'` if you have a GPU and want to accelerate.  
   - The code now uses **vectorized** sparse operations to compute pairwise differences.  
   - The main linear system `(B^T B + alpha * Delta^T Delta) * w = \alpha Delta^T (\eta_old - \tau_old) - B^T A` is solved **iteratively** with a **manual Conjugate Gradient** approach, **no** Python `for` loops.

4. **Results**  
   - **`result['phi']`**: The final logistic-scale estimates (subclonal means) for each SNV in shape `(No_mutation, M)`.  
   - **`result['label']`**: The integer cluster labels for each SNV, indicating subclonal membership.
>>>>>>> 880d373a1d0cdce96c236b4194ad360014ebaaaf

---

## Script Overviews

- **`clipp2/core.py`**: Core SCAD‐ADMM routines and sparse CG solver.  
- **`clipp2/preprocess.py`**: I/O, data alignment, and export helpers.  
- **`CliPP2_pipeline.py`**: End‐to‐end multi-sample orchestration with logging.

---
## Time Consumption
<img src="time_comparison_clipp2_pyclone.png" width="700" height="500">


## Contact

- **Authors**: Yu Ding, Ph.D. (Wenyi Wang Lab, MD Anderson)  
- **Email**: yding4@mdanderson.org, yding1995@gmail.com  
- **Date**: April 2025  

<<<<<<< HEAD
For issues, feature requests, or contributions, please open a GitHub issue or contact the authors directly.
=======
For questions, bug reports, or contributions, please reach out or open an issue. Feel free to adapt this pipeline to your subclonal modeling or SCAD-penalized tasks!
>>>>>>> 880d373a1d0cdce96c236b4194ad360014ebaaaf

