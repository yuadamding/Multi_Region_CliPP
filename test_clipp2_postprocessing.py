#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-case lambda selector for BECS-style post-processed Ward clustering.

What this script does
---------------------
Given a folder of TSVs named like "lambda_<value>.tsv" (each containing
per-mutation cellular prevalence phi across regions/samples), this script:

1) Parses λ from each filename, scales it by SCALE_LAMBDA, and fits
   Ward clustering with a λ-penalized cut:
      K_opt(λ) = argmin_K  [ cumulative Ward SSE up to K ] + λ (K - 1)

2) Applies two rounds of post-processing merges on the phi matrix:
      A. Grow undersized clusters until >= least_mut mutations;
      B. Merge the closest centroid pair if their distance < least_diff.
   It counts how many merges happened in (A) and (B).

3) Applies two cross-sample consolidation heuristics with counters:
      - handle_superclusters (prevalence > purity-normalized 1 across samples)
      - handle_small_clones  (consolidate when clonal fraction is small)

4) Computes model selection metrics for each λ:
      - Log-likelihood using read-count tensors built from the patient input.
      - BIC = -2 loglik + k * log(N)
      - Cluster size entropy  (base 2)
      - Founder distance      (min L2 distance of centroids to "founder" = 1,
                               offset by per-sample purity vector ρ)
      - BECS score = 15 * BIC/N + 100 * entropy/log2(20) + 300 * fd / sqrt(M)

5) Selects the best λ by minimizing BECS (tie-breaker: BIC, then entropy,
   then founder distance, then fewer clusters).

6) Saves:
      - all_results.csv                      (one row per λ)
      - selected_lambda.txt                  (just the winning λ file)
      - selected_phi_clean.tsv               (post-processed φ for winner)
      - selected_labels.csv                  (final labels for winner)
      - selected_prevalence_matrix.csv       (K×M mean φ by cluster×sample)
      - best_lambda_summary.json             (metrics and postprocessing counts)

Assumptions
-----------
- You have an "input_dir" that contains the per-sample input needed by
  clipp2.preprocess.combine_patient_samples / build_tensor_arrays.
  This is how we obtain the read-count tensors (r, n, minor, total) and purity.
- Each lambda_*.tsv has columns: ['chromosome_index','position','region','phi']
  with region names matching the "samples" from the tensors.

Usage
-----
python lambda_selector_real.py \
  --phi_dir /path/to/patient/phi_outputs \
  --input_dir /path/to/patient/input \
  --save_dir /path/to/save/results \
  [--scale_lambda 10.0] \
  [--least_mut_frac 0.05] \
  [--least_diff_raw 0.02] \
  [--max_post_merges 100] \
  [--use_depth_weight] \
  [--export_arrays]

Notes
-----
- If your purity array is two-dimensional (per-mutation-by-sample), we gracefully
  handle it. For founder_distance we need a per-sample vector ρ; in that case,
  we convert to per-sample purity by column-wise mean of the purity array.
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import re
import sys
from math import log, sqrt
from typing import Dict, List, Optional, Tuple
import shutil

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy

# ──────────────────────────────────────────────────────────────────────
# Configuration (overridable by CLI)
# ──────────────────────────────────────────────────────────────────────
SCALE_LAMBDA      = 10.0
LEAST_MUT_FRAC    = 0.05   # fraction of N if <1; absolute if >=1
LEAST_DIFF_RAW    = 0.02   # raw per-sample gap, scaled by √m
MAX_POST_MERGES   = 100
USE_DEPTH_WEIGHT  = False
EXPORT_BIG_ARRAYS = False

# ──────────────────────────────────────────────────────────────────────
# λ parser
# ──────────────────────────────────────────────────────────────────────
_LAMBDA_RE = re.compile(r'lambda_([0-9eE.+-]+)\.tsv$')
def _parse_lambda(fname: str) -> float:
    m = _LAMBDA_RE.search(fname)
    if not m:
        raise ValueError(f"Cannot parse λ from {fname}")
    lam = float(m.group(1))
    if lam < 0:
        raise ValueError("λ must be non-negative")
    return SCALE_LAMBDA * lam

# ──────────────────────────────────────────────────────────────────────
# Helpers: reindex and prevalence
# ──────────────────────────────────────────────────────────────────────
def _reindex(phi: np.ndarray, labels: np.ndarray):
    """
    Return (labels0-based, centroids, counts, uniq_labels)
    """
    uniq, inv = np.unique(labels, return_inverse=True)
    k, m = len(uniq), phi.shape[1]
    cent = np.zeros((k, m), dtype=phi.dtype)
    cnt  = np.zeros(k,    dtype=np.int64)
    for i, u in enumerate(uniq):
        mask = inv == i
        cent[i] = phi[mask].mean(0)
        cnt[i]  = mask.sum()
    return inv, cent, cnt, uniq

def _cluster_prevalence(phi_clean: np.ndarray,
                        labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
      cp_mat … shape (K, M) – mean CP per (cluster, sample)
      uniq   … length-K     – cluster IDs corresponding to rows
    """
    uniq = np.unique(labels)
    K, M = len(uniq), phi_clean.shape[1]
    cp   = np.zeros((K, M), dtype=phi_clean.dtype)
    for i, k in enumerate(uniq):
        cp[i] = phi_clean[labels == k].mean(0)
    return cp, uniq

def _merge_two_clusters(labels: np.ndarray,
                        cid_keep: int,
                        cid_merge: int) -> np.ndarray:
    """
    Relabel cid_merge → cid_keep and make the label set contiguous.
    """
    labels = labels.copy()
    labels[labels == cid_merge] = cid_keep
    _, inv = np.unique(labels, return_inverse=True)
    return inv

# ──────────────────────────────────────────────────────────────────────
# Post-processing steps (A) and (B) with counters
# ──────────────────────────────────────────────────────────────────────
def _merge_one_small(phi, labels, cent, cnt, uniq, least_mut):
    small_idx = np.where(cnt < least_mut)[0]
    if small_idx.size == 0:
        return False, labels
    cid = small_idx[0]
    mask_src = labels == cid
    other_ids = np.delete(np.arange(len(uniq)), cid)
    other_cent = cent[other_ids]
    dmat = np.linalg.norm(phi[mask_src][:, None, :] - other_cent[None, :, :],
                          axis=2)
    nearest = dmat.argmin(axis=1)
    labels[mask_src] = other_ids[nearest]
    return True, labels

def _merge_one_close(phi, labels, cent, cnt, uniq, least_diff):
    if len(cent) <= 1:
        return False, labels
    dist = squareform(pdist(cent, 'euclidean'))
    np.fill_diagonal(dist, np.inf)
    min_val = dist.min()
    if min_val >= least_diff:
        return False, labels
    i, j = np.unravel_index(dist.argmin(), dist.shape)
    cid = i if cnt[i] < cnt[j] else j
    mask_src = labels == cid
    other_ids = np.delete(np.arange(len(uniq)), cid)
    other_cent = cent[other_ids]
    dmat = np.linalg.norm(phi[mask_src][:, None, :] - other_cent[None, :, :],
                          axis=2)
    nearest = dmat.argmin(axis=1)
    labels[mask_src] = other_ids[nearest]
    return True, labels

def _postprocess_clusters(phi, labels, least_mut, least_diff):
    """
    While-loop that applies A then B, exactly ONE cluster dissolved per iteration.
    Returns: phi_clean, labels, merges_small, merges_close
    """
    merges_small = 0
    merges_close = 0
    merges = 0
    while merges < MAX_POST_MERGES:
        labels, cent, cnt, uniq = _reindex(phi, labels)

        changed, labels = _merge_one_small(phi, labels, cent, cnt, uniq, least_mut)
        if changed:
            merges += 1
            merges_small += 1
            continue

        changed, labels = _merge_one_close(phi, labels, cent, cnt, uniq, least_diff)
        if changed:
            merges += 1
            merges_close += 1
            continue

        break

    labels, cent, _, _ = _reindex(phi, labels)
    return cent[labels].astype(np.float32), labels.astype(np.int64), merges_small, merges_close

# ──────────────────────────────────────────────────────────────────────
# Cross-sample consolidations with counters
# ──────────────────────────────────────────────────────────────────────
def handle_superclusters(phi_clean: np.ndarray,
                         labels: np.ndarray,
                         purity: np.ndarray,
                         max_clonal_frac: float = 0.40) -> Tuple[np.ndarray, int]:
    """
    Multi-sample super-cluster consolidation.
    Returns (labels, merges_super)
    """
    merges = 0
    while True:
        cp_mat, uniq = _cluster_prevalence(phi_clean, labels)
        # normalize by purity; purity may be (N,M) or (M,)
        if purity.ndim == 2:
            rho = purity.mean(axis=0)  # per-sample vector
        else:
            rho = purity
        cp_mat = cp_mat / rho  # normalize CP by purity
        K, M = cp_mat.shape

        if not (cp_mat > 1.0).any():
            break
        if K <= 2:
            break
        d2_one = np.linalg.norm(cp_mat - 1.0, axis=1)      # L2 to 𝟙
        clonal_idx = d2_one.argmin()
        clonal_frac = np.bincount(labels)[clonal_idx] / labels.size
        if clonal_frac > max_clonal_frac:
            break

        cp_mean = np.linalg.norm(cp_mat, axis=1)
        top2 = np.argsort(cp_mean)[-2:]
        cid_keep  = int(uniq[top2[1]])
        cid_merge = int(uniq[top2[0]])
        labels = _merge_two_clusters(labels, cid_keep, cid_merge)
        merges += 1

    return labels, merges

def handle_small_clones(phi_clean: np.ndarray,
                        labels: np.ndarray,
                        max_clonal_frac: float = 0.15) -> Tuple[np.ndarray, int]:
    """
    Multi-sample small-clone consolidation.
    Returns (labels, merges_smallc)
    """
    merges = 0
    while True:
        cp_mat, uniq = _cluster_prevalence(phi_clean, labels)
        K = cp_mat.shape[0]
        if K <= 2:
            break

        d2_one = np.linalg.norm(cp_mat - 1.0, axis=1)      # L2 to 𝟙
        clonal_idx = d2_one.argmin()
        clonal_frac = np.bincount(labels)[clonal_idx] / labels.size
        if clonal_frac > max_clonal_frac:
            break

        cp_mean = np.linalg.norm(cp_mat, axis=1)
        top2 = np.argsort(cp_mean)[-2:]
        cid_keep  = int(uniq[top2[1]])
        cid_merge = int(uniq[top2[0]])
        labels = _merge_two_clusters(labels, cid_keep, cid_merge)
        merges += 1

    return labels, merges

# ──────────────────────────────────────────────────────────────────────
# Ward + penalty + post-processing
# ──────────────────────────────────────────────────────────────────────
def ward_penalised(phi: np.ndarray,
                   lambda_pen: float,
                   weights: Optional[np.ndarray] = None):
    """
    1. Build Ward dendrogram; choose K_opt(λ)
    2. Apply small & close-cluster merging.
    Returns: phi_clean, labels, K_raw, merges_small, merges_close
    """
    n, m = phi.shape
    if n < 2:
        return phi.copy(), np.zeros(n, np.int64), 1, 0, 0

    if weights is not None:
        if weights.shape != (n,):
            raise ValueError("weights must be (n,) vector")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")

    X = phi if weights is None else phi * np.sqrt(weights)[:, None]
    Z = linkage(X, method='ward', metric='euclidean')

    delta = Z[:, 2].astype(np.float64)
    cum_sse = np.concatenate([[0.0], np.cumsum(delta)])
    K_vals = np.arange(n, 0, -1, dtype=np.int32)
    obj = cum_sse + lambda_pen * (K_vals - 1)
    K_opt = min(int(K_vals[obj.argmin()]), 20)

    labels = fcluster(Z, t=K_opt, criterion='maxclust') - 1
    K_raw = len(np.unique(labels))

    least_mut = int(LEAST_MUT_FRAC*n) if LEAST_MUT_FRAC < 1 else int(LEAST_MUT_FRAC)
    least_mut = max(1, least_mut)
    least_diff = LEAST_DIFF_RAW * sqrt(m)

    phi_clean, labels, m_small, m_close = _postprocess_clusters(phi, labels,
                                                                least_mut, least_diff)
    return phi_clean, labels, K_raw, m_small, m_close

# ──────────────────────────────────────────────────────────────────────
# Likelihood and metrics
# ──────────────────────────────────────────────────────────────────────
def loglik_of_phi(phi_clean: np.ndarray,
                  df_phi_index: pd.MultiIndex,
                  pairs_df: pd.DataFrame,
                  tensors: Dict[str, np.ndarray],
                  eps: float = 1e-12) -> float:
    full_index = pairs_df.index
    locs = full_index.get_indexer(df_phi_index)
    keep = locs >= 0
    if keep.sum() == 0:
        return -np.inf

    phi_clean = phi_clean[keep]
    locs      = locs[keep]

    r, n = tensors['r'][locs], tensors['n'][locs]
    minor, total = tensors['minor'][locs], tensors['total'][locs]
    purity = tensors['purity'][locs] if tensors['purity'].ndim == 2 else tensors['purity']

    theta = (phi_clean * minor) / (2 * (1 - purity) + purity * total)
    theta = np.clip(theta, eps, 1 - eps)
    return float((r * np.log(theta) + (n - r) * np.log1p(-theta)).sum())

def safe_entropy(labels: np.ndarray) -> float:
    counts = np.bincount(labels)
    if counts.size == 1:
        return 0.0
    return float(entropy(counts, base=2))

def founder_distance(phi_clean: np.ndarray, labels: np.ndarray, purity: np.ndarray) -> float:
    cent = np.vstack([phi_clean[labels == k].mean(0) for k in np.unique(labels)])
    # founder assumed at CCF = 1 across samples; offset by per-sample purity ρ
    if purity.ndim == 2:
        rho = purity.mean(axis=0)
    else:
        rho = purity
    return float(np.min(np.linalg.norm(cent - rho, axis=1)))

# ──────────────────────────────────────────────────────────────────────
# Single-λ evaluation (NO TRUTH NEEDED)
# ──────────────────────────────────────────────────────────────────────
def evaluate_single_lambda(tsv_file: str,
                           samples: List[str],
                           pairs_df: pd.DataFrame,
                           tensors: Dict[str, np.ndarray]) -> Optional[Dict]:

    try:
        df_phi = (pd.read_csv(tsv_file, sep='\t')
                    .pivot_table(index=['chromosome_index', 'position'],
                                 columns='region', values='phi')
                    .reindex(columns=samples)
                    .dropna(how='all'))
        if df_phi.empty:
            return None

        phi = df_phi.to_numpy(np.float32)
        N, M = phi.shape
        if N == 0:
            return None

        lam_pen = _parse_lambda(os.path.basename(tsv_file))

        # Weights option (per-mutation coverage) for Ward
        weights = None
        if USE_DEPTH_WEIGHT:
            pair_idx = pairs_df.index  # MultiIndex (chrom, pos)
            locs = pair_idx.get_indexer(df_phi.index)
            if (locs < 0).any():
                raise ValueError("φ rows not all present in pairs_df")
            weights = tensors['n'][locs]

        # Ward + post-processing
        phi_c, labels, K_raw, m_small, m_close = ward_penalised(phi, lam_pen, weights)

        # Likelihood & metrics BEFORE cross-sample consolidation
        loglik = loglik_of_phi(phi_c, df_phi.index, pairs_df, tensors)
        k_act_pre  = len(np.unique(labels))
        bic_pre    = -2.0*loglik if N <= 1 else -2.0*loglik + k_act_pre*log(N)
        entr_pre   = safe_entropy(labels)
        fdst_pre   = founder_distance(phi_c, labels, tensors["purity"])

        # Cross-sample consolidations
        labels_sc, m_super = handle_superclusters(phi_c, labels, tensors["purity"])
        labels_sc, m_smallc = handle_small_clones(phi_c, labels_sc)

        # Recompute metrics AFTER consolidations
        k_act  = len(np.unique(labels_sc))
        bic    = -2.0*loglik if N <= 1 else -2.0*loglik + k_act*log(N)
        entr   = safe_entropy(labels_sc)
        fdst   = founder_distance(phi_c, labels_sc, tensors["purity"])

        # BECS meta-score (weights tuned empirically in prior sims)
        becs = 15 * bic / max(N, 1) + 100 * entr / np.log2(20) + 300 * fdst / (M ** 0.5)

        out = dict(
            source_lambda_file=os.path.basename(tsv_file),
            lambda_pen=float(lam_pen),
            N=int(N),
            M=int(M),

            # Raw Ward cut and merges A/B
            K_raw=int(K_raw),
            merges_small=int(m_small),
            merges_close=int(m_close),

            # After consolidations
            K_final=int(k_act),
            merges_super=int(m_super),
            merges_smallclones=int(m_smallc),

            # Scores (pre vs post for transparency)
            log_likelihood=float(loglik),
            bic_pre=float(bic_pre),
            entropy_pre=float(entr_pre),
            founder_distance_pre=float(fdst_pre),

            bic_score=float(bic),
            entropy=float(entr),
            founder_distance=float(fdst),
            bec_score=float(becs),
        )

        if EXPORT_BIG_ARRAYS:
            out.update(phi_clean=phi_c.tolist(), labels=labels_sc.tolist())

        # Attach arrays we always need later for the selected model
        out["_phi_clean"] = phi_c          # cleaned per-mutation φ (N×M)
        out["_labels"]    = labels_sc      # final labels (length N)
        out["_phi_index"] = df_phi.index   # MultiIndex (chromosome_index, position)
        out["_phi_hat"]   = phi 
        
        return out

    except Exception as e:
        print(f"[WARNING] Failed {os.path.basename(tsv_file)}: {e}", file=sys.stderr)
        return None

# ──────────────────────────────────────────────────────────────────────
# Driver for a *single* real-case directory of lambda_*.tsv
# ──────────────────────────────────────────────────────────────────────
def process_all_lambdas_real(phi_dir: str, input_dir: str) -> Tuple[List[Dict], List[str], pd.DataFrame, Dict[str, np.ndarray]]:
    # tensors & samples from the real-case patient input
    from clipp2.preprocess import combine_patient_samples, add_to_df, build_tensor_arrays
    df_s = add_to_df(combine_patient_samples(input_dir))
    dic  = build_tensor_arrays(df_s)
    pairs_df = dic["pairs"].set_index(['chromosome_index', 'position'])
    tensors  = {k: dic[k] for k in ("r", "n", "minor", "total")}
    tensors["purity"] = dic["pur_arr"]
    samples  = dic["samples"]

    lambda_files = sorted(glob.glob(os.path.join(phi_dir, "lambda_*.tsv")))
    if not lambda_files:
        raise FileNotFoundError(f"No lambda_*.tsv found under {phi_dir}")
    print(f"[INFO] {len(lambda_files)} lambda files detected.")

    results = []
    for tsv in lambda_files:
        res = evaluate_single_lambda(tsv, samples, pairs_df, tensors)
        if res is not None:
            results.append(res)

    return results, samples, pairs_df, tensors

# ──────────────────────────────────────────────────────────────────────
# Selection / Reporting
# ──────────────────────────────────────────────────────────────────────
def _select_best(results: List[Dict]) -> Dict:
    if not results:
        raise RuntimeError("No successful results to select from.")
    # rank keys in order
    def _key(r):
        return (r["bec_score"], r["bic_score"], r["entropy"],
                r["founder_distance"], r["K_final"])
    best = sorted(results, key=_key)[0]
    return best

def _save_selection(best: Dict, samples: List[str], save_dir: str, phi_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    # Save selected lambda filename
    with open(os.path.join(save_dir, "selected_lambda.txt"), "w") as f:
        f.write(str(best["source_lambda_file"]) + "\n")
    shutil.copy(os.path.join(phi_dir, str(best["source_lambda_file"])), save_dir)

    # Final arrays
    phi_c   = best.pop("_phi_clean")
    labels  = best.pop("_labels").astype(int)
    idx     = best.pop("_phi_index")     # MultiIndex
    phi_hat = best.pop("_phi_hat")       # original φ (same shape as phi_c)

    # Save phi_clean as TSV (unchanged)
    df_phi_c = pd.DataFrame(phi_c, columns=samples)
    df_phi_c.to_csv(os.path.join(save_dir, "selected_phi_clean.tsv"),
                    sep="\t", index_label="row_id")

    # Build labels table with chrom,pos,cluster + JSON arrays for φ
    idx_df = idx.to_frame(index=False).reset_index(drop=True)
    idx_df["cluster"]    = labels

    # JSON-encode per-row vectors so each row carries its full M-sample profile
    phi_clean_json = [json.dumps([float(v) for v in row]) for row in phi_c]
    phi_hat_json   = [json.dumps([float(v) for v in row]) for row in phi_hat]
    idx_df["phi_clean"]  = phi_clean_json
    idx_df["phi_hat"]    = phi_hat_json

    # Ensure column order
    idx_df = idx_df[["chromosome_index", "position", "cluster", "phi_clean", "phi_hat"]]
    idx_df.to_csv(os.path.join(save_dir, "selected_labels.csv"), index=False)

    # Save prevalence matrix (K×M)
    cp_mat, uniq = _cluster_prevalence(phi_c, labels)
    counts = np.array([(labels == k).sum() for k in uniq], dtype=int)
    df_cp = pd.DataFrame(cp_mat, columns=samples)
    # Put cluster_id and n_mut up front
    df_cp.insert(0, "cluster_id", uniq.astype(int))
    df_cp.insert(1, "n_mut", counts.astype(int))

    df_cp.to_csv(os.path.join(save_dir, "selected_prevalence_matrix.csv"), index=False)

    # Summary JSON (metrics & postprocessing counts)
    with open(os.path.join(save_dir, "best_lambda_summary.json"), "w") as f:
        json.dump(best, f, indent=2)

def _save_all_results(results: List[Dict], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    # Prepare a tidy table and drop in-memory arrays
    rows = []
    for r in results:
        r = dict(r)  # shallow copy
        r.pop("_phi_clean", None)
        r.pop("_labels", None)
        rows.append(r)
    cols = ['source_lambda_file', 'lambda_pen', 'N', 'M',
            'K_raw', 'merges_small', 'merges_close',
            'K_final', 'merges_super', 'merges_smallclones',
            'log_likelihood', 'bic_pre', 'entropy_pre', 'founder_distance_pre',
            'bic_score', 'entropy', 'founder_distance', 'bec_score']
    df = pd.DataFrame(rows)
    # ensure consistent column order if all keys present
    if set(cols).issubset(df.columns):
        df = df[cols]
    df.to_csv(os.path.join(save_dir, "all_results.csv"), index=False)
    return df

# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main(phi_dir, input_dir, save_dir):
    global SCALE_LAMBDA, LEAST_MUT_FRAC, LEAST_DIFF_RAW, MAX_POST_MERGES
    global USE_DEPTH_WEIGHT, EXPORT_BIG_ARRAYS

    print("[INFO] Starting real-case λ selection…")
    results, samples, pairs_df, tensors = process_all_lambdas_real(phi_dir, input_dir)
    if not results:
        print("[ERROR] No successful λ evaluations.", file=sys.stderr)
        sys.exit(2)

    df_all = _save_all_results(results, save_dir)
    best = _select_best(results)
    _save_selection(best, samples, save_dir, phi_dir)

    print("\n[INFO] Selection complete.")
    print(f"[INFO] Best λ file: {best['source_lambda_file']}")
    print(f"[INFO] Results table: {os.path.join(save_dir, 'all_results.csv')}")
    print(f"[INFO] Summary JSON: {os.path.join(save_dir, 'best_lambda_summary.json')}")
    print(f"[INFO] Final φ/labels in: {save_dir}")

if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Select best λ from real-case lambda_*.tsv folder.")
    p.add_argument("--phi_dir", required=True, help="Folder containing lambda_*.tsv")
    p.add_argument("--input_dir", required=True, help="Patient input folder for tensors")
    p.add_argument("--save_dir", required=True, help="Output folder")
    p.add_argument("--scale_lambda", type=float, default=SCALE_LAMBDA)
    p.add_argument("--least_mut_frac", type=float, default=LEAST_MUT_FRAC)
    p.add_argument("--least_diff_raw", type=float, default=LEAST_DIFF_RAW)
    p.add_argument("--max_post_merges", type=int, default=MAX_POST_MERGES)
    p.add_argument("--use_depth_weight", action="store_true", default=False)
    p.add_argument("--export_arrays", action="store_true", default=False,
                   help="Also include arrays per-λ inside all_results.json rows (very large).")
    args = p.parse_args()

    SCALE_LAMBDA     = args.scale_lambda
    LEAST_MUT_FRAC   = args.least_mut_frac
    LEAST_DIFF_RAW   = args.least_diff_raw
    MAX_POST_MERGES  = args.max_post_merges
    USE_DEPTH_WEIGHT = args.use_depth_weight
    EXPORT_BIG_ARRAYS= args.export_arrays
    main(args.phi_dir, args.input_dir, args.save_dir)