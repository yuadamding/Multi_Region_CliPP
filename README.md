# Multi-Region CliPP: README

This repository provides a Python implementation of multi-region **CliPP** (Clonal Proliferation and Prevalence) analysis. It includes data structures and classes to represent individual SNVs (`snv`) and collections of SNVs (`snvs`), along with methods to estimate cellular prevalence, copy numbers, and perform clustering.

---

## Overview

Clonal structure inference in cancer often requires integrating **mutation reads** (both reference and alternative alleles), **copy number calls**, and **tumor purity estimates** across multiple **spatial regions** or **time points**. The goal is to model **cellular prevalence** of each SNV under certain constraints (e.g., piecewise approximation, ADMM-based optimization).

This project:

1. Defines classes `snv` and `snvs` for storing and manipulating multi-region data.
2. Provides methods to vectorize and combine per-SNV data for use in augmented Lagrangian or ADMM algorithms.
3. Calculates likelihood values, read proportions, copy numbers, and other parameters used in cluster or subclonal structure inference.

---

## The Basic Classes

### **Class: `snv`**

The `snv` class contains all information (observed and inferred) about **one** single-nucleotide variant across multiple regions.

| Attribute                | Type       | Description                                                                                                     |
|--------------------------|-----------|-----------------------------------------------------------------------------------------------------------------|
| **name**                 | `str`      | Name or identifier for this SNV.                                                                                |
| **num_regions**          | `int`      | Number of regions (samples) that contain this SNV.                                                              |
| **reads**                | `np.array` | An array of length = `num_regions`, holding *alternate-allele* read counts for each region.                     |
| **total_reads**          | `np.array` | An array of length = `num_regions`, holding total reads (ref + alt) for each region.                            |
| **rho**                  | `np.array` | An array of length = `num_regions`, holding tumor purity values per region.                                     |
| **tumor_cn**             | `np.array` | An array of length = `num_regions`, holding *tumor* total copy number per region.                               |
| **normal_cn**            | `np.array` | An array of length = `num_regions`, holding *normal* copy number (commonly 2) per region.                       |
| **major_cn**             | `np.array` | An array of length = `num_regions`, holding *major-allele* tumor CN per region.                                 |
| **minor_cn**             | `np.array` | An array of length = `num_regions`, holding *minor-allele* tumor CN per region.                                 |
| **cp**                   | `np.array` | An array of length = `num_regions`, representing the *cellular prevalence* of the SNV in each region.           |
| **prop**                 | `np.array` | An array of length = `num_regions`, representing the *expected Binomial proportion* per region.                 |
| **specific_copy_number** | `np.array` | An array of length = `num_regions`, representing the *SNV-specific copy number* (sometimes called multiplicity). |
| **likelihood**           | `np.array` | An array of length = `num_regions`, storing the *log-likelihood* contributions in each region.                  |
| **mapped_cp**            | `np.array` | An array of length = `num_regions`, mapping `cp` onto the real line (e.g., via logit) to remove [0,1] bounds.   |
| **map_method**           | `str`      | The method used to transform `cp` → `mapped_cp` (typically the inverse of a sigmoid or Beta CDF).               |

### **Class: `snvs`**

The `snvs` class manages a collection of `snv` objects, aggregating per-SNV information and storing global parameters for ADMM/augmented Lagrangian inference.

| Attribute               | Type        | Description                                                                                                                                          |
|-------------------------|------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| **num_snvs**           | `int`       | Number of SNVs contained in `snv_lst`.                                                                                                              |
| **num_regions**        | `int`       | Number of regions (samples) each SNV is observed in.                                                                                                 |
| **likelihood**         | `float`     | Total log-likelihood across all SNVs and all regions.                                                                                                |
| **snv_lst**            | `list`      | A list of `snv` objects; each entry stores raw observations and inferred parameters for one SNV.                                                    |
| **p**                  | `np.array`  | A flattened vector (length = `num_snvs * num_regions`), representing mapped SNV prevalences across all SNVs and regions.                             |
| **combination**        | `list`      | A list of two-element sets (pairs) representing all possible `(SNV_i, SNV_j)` combinations used for pairwise constraints in clustering/inference.   |
| **v**                  | `np.array`  | A vector (length = `len(combination) * num_regions`) used in the ADMM or augmented Lagrangian updates (often related to cluster penalization).       |
| **y**                  | `np.array`  | A vector (same shape as `v`), typically storing *dual variables* in the ADMM updates.                                                                |
| **gamma**              | `float`     | A hyperparameter for controlling the regularization strength (e.g., penalizing cluster differences).                                                 |
| **omega**              | `np.array`  | A vector (length = `len(combination)`), specifying penalty weights for each pair in `combination`.                                                  |
| **paris_mapping**       | `dict`      | Maps a pair (SNV_i, SNV_j) to a linear index, enabling fast indexing into `v` or `y`.                                                                |
| **paris_mapping_inverse** | `dict`   | Inverse of `paris_mapping`: given an index, identifies the corresponding `(SNV_i, SNV_j)` pair.                                                      |

---

## Installation & Dependencies

- **Python >= 3.8**
- **NumPy** for numerical arrays
- **Pandas** for DataFrame support
- **SciPy** for optimization and sparse matrix structures
- **PyTorch** (optional) if you wish to leverage GPU-based linear algebra in the ADMM steps
- **Ray** (optional) if you wish to distribute or parallelize the ADMM routines

Example installation:

```bash
pip install numpy pandas scipy torch ray