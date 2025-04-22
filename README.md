# SCAD-Penalized ADMM for Multi-region Subclonal Reconstruction

This project implements a **CUDA-accelerated**, research-oriented pipeline to perform **SCAD-penalized ADMM** for subclonal reconstruction. It supports both single-sample (M=1) and multi-sample (M>1) scenarios.

The newest version removes the need to build large dense matrices on CPU, instead using **iterative Conjugate Gradient** (CG) on the GPU to solve the ADMM subproblems efficiently, relying on **vectorized** or **sparse** operations in PyTorch.

See [CliPP2.pdf](/CliPP2.pdf) and [gpu_implementation.pdf](/gpu_implementation.pdf) for theoretical details.

---

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
Subclonal reconstruction by grouping single nucleotide variants (SNVs) that share similar frequency patterns, in either a single-sample or multi-sample context. The penalty is a group-SCAD that encourages merging of pairwise differences in logistic-scale parameters.

**Key CUDA Implementation Highlights:**
1. GPU (CUDA) acceleration via PyTorch tensors and sparse operations.  
2. Iterative Conjugate Gradient solves the linear subproblem without forming a dense Hessian.  
3. No Python `for` loops in main numeric kernels—pairwise differences via sparse multiplication.  
4. Scales to large mutation × sample sizes due to low memory footprint.

---

## Directory Layout

```
MULTI_REGION_CLIPP2/         # root directory
├── clipp2/                  # core library
│   ├── core.py              # SCAD-ADMM implementation
│   ├── preprocess.py        # data I/O & preprocessing
│   └── scad_admm.py         # alternate entrypoint
├── input_files/             # sample subdirectories
│   ├── region1/             # e.g. region1/
│   │    ├── snv.txt
│   │    ├── cna.txt
│   │    ├── purity.txt
│   └── region2/ ...
├── README.md                # this documentation
├── CliPP2.pdf               # problem formulation and algorithm details
└── gpu_implementation.pdf   # GPU design details
```

---

## Dependencies

- Python 3.7+  
- NumPy  
- SciPy  
- PyTorch (>=1.10; PyTorch 2.0+ for `torch.sparse.linalg.cg`)  

Install example:
```bash
pip install numpy scipy torch
```

---
## General Multi-sample CLiPP2 Pipeline

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

Results are saved in `output/result.txt` and `output/intermediate/all_loci.txt`.

---

## Script Overviews

- **`clipp2/core.py`**: Core SCAD‐ADMM routines and sparse CG solver.  
- **`clipp2/preprocess.py`**: I/O, data alignment, and export helpers.  
- **`scad_admm.py`**: Alternate entrypoint for direct ADMM on GPU.  
- **`CliPP2_pipeline.py`**: End‐to‐end multi-sample orchestration with logging.

---

## Contact

- **Authors**: Yu Ding, Ph.D. (Wenyi Wang Lab, MD Anderson)  
- **Email**: yding4@mdanderson.org, yding1995@gmail.com  
- **Date**: April 2025  

For issues, feature requests, or contributions, please open a GitHub issue or contact the authors directly.

