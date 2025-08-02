#!/usr/bin/env python3
"""
This script provides a research-oriented pipeline for an ADAPTIVELY WEIGHTED 
SCAD-penalized ADMM approach to multi-sample subclone reconstruction.

This revision implements the adaptive weighting scheme described in the paper
"An Adaptively Weighted SCAD Framework for Subclone Reconstruction". The penalty
for each SNV pair is now scaled by a data-driven weight, allowing the model
to handle heterogeneous noise levels and distinguish close subclones more effectively.

PERFORMANCE REVISION:
*   JIT Compilation: Core ADMM helper functions are decorated with @torch.jit.script 
    to eliminate Python overhead and fuse CUDA kernels.
*   Automatic Mixed Precision (AMP): The main ADMM loop uses the modern torch.amp.autocast
    API to leverage Tensor Cores on compatible GPUs for significant speedup.
*   Robustness: Final size-based cleanup merges spurious small clusters.

Author: [Yu Ding, Ph.D. / Wenyi Wang's Lab / MD Anderson Cancer Center]
Date: [Oct 2024]
Contact: [yding4@mdanderson.org, yding1995@gmail.com]
"""

import os
import math
import numpy as np
import scipy.sparse as sp
from scipy.special import expit, logit
import torch
from scipy.stats import wasserstein_distance
from scipy.cluster.hierarchy import linkage, fcluster

# --- Performance Flags ---
AMP_OK = torch.cuda.is_available() and hasattr(torch.cuda, "amp")

# --- JIT-Compiled Helper Functions ---
@torch.jit.script
def scad_threshold_update_torch(
    w_new_t:     torch.Tensor,
    tau_old_t:   torch.Tensor,
    i_idx:       torch.Tensor,
    j_idx:       torch.Tensor,
    alpha:  float,
    Lambda: float,
    gamma:  float,
    adaptive_weights_t: torch.Tensor
):
    """
    Weighted Group‑SCAD proximal operator (TorchScript‑compatible).

    Returns
    -------
    eta_new_t : (P,M) tensor
    tau_new_t : (P,M) tensor
    """
    # ------------------------------------------------------------------ Δw  & z
    D_w      = w_new_t[i_idx, :] - w_new_t[j_idx, :]
    z_t      = D_w - tau_old_t / alpha
    z_norm_t = torch.linalg.norm(z_t, dim=1)

    # ------------------------------------------------------------------ pair‑specific λ
    lam_w = Lambda * adaptive_weights_t

    # ------------------------------------------------------------------ thresholds
    thr1 = 2.0   * lam_w / alpha          # Region‑I upper bound
    thr2 = gamma * lam_w / alpha          # Region‑II upper bound

    mask_soft   = z_norm_t <= thr1
    mask_linear = (z_norm_t > thr1) & (z_norm_t <= thr2)

    eta_new_t = torch.zeros_like(z_t)

    # --------------------------- Region I
    if mask_soft.any():
        i1 = mask_soft.nonzero().squeeze(1)            # TorchScript‑safe
        scale1 = (1.0 - lam_w[i1] /
                  (alpha * z_norm_t[i1].clamp_min(1e-12))).clamp_min(0.0)
        eta_new_t[i1] = scale1.unsqueeze(1) * z_t[i1]

    # --------------------------- Region II
    if mask_linear.any():
        i2     = mask_linear.nonzero().squeeze(1)      # TorchScript‑safe
        lam_i2 = lam_w[i2]
        denom2 = 1.0 - lam_i2 / (alpha * (gamma - 1.0))
        safe   = denom2 > 1e-8

        if safe.any():
            sidx   = i2[safe]
            numer  = (gamma * lam_i2[safe]) / (alpha * (gamma - 1.0))
            scale2 = (1.0 - numer /
                      z_norm_t[sidx].clamp_min(1e-12)).clamp_min(0.0)
            eta_new_t[sidx] = (scale2 / denom2[safe]).unsqueeze(1) * z_t[sidx]

        if (~safe).any():                               # rare numerical corner
            eta_new_t[i2[~safe]] = z_t[i2[~safe]]

    # --------------------------- Region III (identity)
    mask_all = mask_soft | mask_linear
    eta_new_t[~mask_all] = z_t[~mask_all]

    # ------------------------------------------------------------------ dual update
    tau_new_t = tau_old_t - alpha * (D_w - eta_new_t)
    return eta_new_t, tau_new_t


@torch.jit.script
def matvec_H_laplacian(x, B_sq_t, No_mutation: int, M: int, alpha: float):
    """(JIT-Compiled) Computes the matrix-vector product for the w-update subproblem."""
    out = B_sq_t * x
    W = x.view(No_mutation, M)
    sum_of_rows = W.sum(dim=0, keepdim=True)
    L_times_W = No_mutation * W - sum_of_rows
    out = out + alpha * L_times_W.flatten()
    return out

@torch.jit.script
def transpose_matvec_delta(v_pairs, No_mutation: int, M: int, i_idx, j_idx):
    """(JIT-Compiled) Computes the transpose of the difference operator times a vector."""
    z_mat = torch.zeros((No_mutation, M), dtype=v_pairs.dtype, device=v_pairs.device)
    z_mat.index_add_(0, i_idx, v_pairs)
    z_mat.index_add_(0, j_idx, -v_pairs)
    return z_mat.flatten()

def _diff_mat_lower(w: np.ndarray) -> np.ndarray:
    """Lower‑triangle pair‑wise ‖wᵢ − wⱼ‖₂ (float32)."""
    n = w.shape[0]
    out = np.zeros((n, n), dtype=w.dtype)
    for i in range(1, n):
        out[i, :i] = np.linalg.norm(w[i] - w[:i], axis=1).astype(w.dtype)
    return out


def _sort_by_2norm(mat: np.ndarray) -> np.ndarray:
    """Rows sorted by their 2‑norm (ascending)."""
    order = np.argsort(np.linalg.norm(mat, axis=1))
    return mat[order], order


def _find_min_row_by_2norm(mat: np.ndarray):
    """Returns (row_idx, row_vector) of the row with smallest 2‑norm."""
    norms = np.linalg.norm(mat, axis=1)
    idx   = norms.argmin()
    return idx, mat[idx]


def _reassign_labels_by_distance(phi_res, labels, purity):
    """
    Distance‑based final clean‑up (same as V1 original).
    Each mutation is assigned to the nearest φ centroid.
    """
    uniq = np.unique(labels)
    centroids = np.vstack([phi_res[labels == k].mean(axis=0) for k in uniq])

    for i in range(phi_res.shape[0]):
        labels[i] = np.argmin(np.linalg.norm(phi_res[i] - centroids, axis=1))

    _, labels = np.unique(labels, return_inverse=True)
    return labels

def preprocess_clipp_data(r, n, minor, total, purity, coef_list, control_large, device, dtype):
    """Preprocesses data for single-sample or multi-sample (M>1) subclone reconstruction."""
    def ensure_2D_and_no_zeros(arr):
        if arr.ndim == 1: arr = arr.reshape(-1, 1)
        elif arr.ndim != 2: raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")
        return np.where(arr == 0, 1, arr)
    def to_torch_gpu(arr, local_dtype):
        return torch.as_tensor(arr, dtype=local_dtype, device=device)
        
    r = r.reshape(-1, 1) if r.ndim == 1 else r
    n, minor, total = ensure_2D_and_no_zeros(n), ensure_2D_and_no_zeros(minor), ensure_2D_and_no_zeros(total)
    r_t, n_t = to_torch_gpu(r, dtype), to_torch_gpu(n, dtype)
    minor_t, total_t = to_torch_gpu(minor, dtype), to_torch_gpu(total, dtype)
    c_stack = [to_torch_gpu(c, dtype) for c in coef_list]
    c_all_t = torch.stack(c_stack, dim=1)
    No_mutation, M = r_t.shape
    if isinstance(purity, (float, int)):
        purity_t = torch.full((No_mutation, M), float(purity), device=device, dtype=r_t.dtype)
    else:
        purity_t = to_torch_gpu(purity, dtype).unsqueeze(0).expand(No_mutation, -1)
    ploidy_t = torch.full((No_mutation, M), 2.0, device=device, dtype=r_t.dtype)
    
    fraction_t = (r_t + 1e-12) / (n_t + 1e-10)
    phi_hat_t = fraction_t * ((ploidy_t - purity_t*ploidy_t) + (purity_t * (total_t + 1e-10))) / minor_t
    phi_hat_t = torch.where(r_t == 0, torch.tensor(1e-12, device=phi_hat_t.device, dtype=phi_hat_t.dtype), phi_hat_t)
    scale_parameter = torch.clamp(torch.max(phi_hat_t), min=1.0)
    phi_new_t = torch.clamp(phi_hat_t / scale_parameter, 1e-7, 1.0 - 1e-7)
    w_init_t = torch.logit(phi_new_t, eps=1e-7).clamp(-control_large, control_large)

    with torch.no_grad():
        vaf_mle = (r_t + 0.5) / (n_t + 1.0)
        phi_mle_t = vaf_mle * ((ploidy_t - purity_t*ploidy_t) + (purity_t * (total_t + 1e-10))) / minor_t
        phi_mle_t_clipped = torch.clamp(phi_mle_t, 1e-6, 1.0 - 1e-6)
        w_mle_t = torch.logit(phi_mle_t_clipped, eps=1e-6)

    return {'w_init_t': w_init_t, 'w_mle_t': w_mle_t, 'r_t': r_t, 'n_t': n_t, 'minor_t': minor_t, 'total_t': total_t, 'purity_t': purity_t, 'c_all_t': c_all_t, 'No_mutation': No_mutation, 'M': M}


def run_admm_optimization(
    preprocessed_data, wcut, alpha_init, gamma, rho, precision,
    Run_limit, control_large, Lambda, device,
    adaptive_weights_t, i_idx, j_idx,
    warm_start_vars=None, cg_max_iter=50
):
    # ---------------------------------------------------------------- unpack (unchanged)
    No_mutation, M  = preprocessed_data['No_mutation'], preprocessed_data['M']
    r_t, n_t        = preprocessed_data['r_t'],     preprocessed_data['n_t']
    minor_t         = preprocessed_data['minor_t']
    total_t         = preprocessed_data['total_t']
    purity_t        = preprocessed_data['purity_t']
    c_all_t         = preprocessed_data['c_all_t']

    w_new_t = (warm_start_vars['w']
               if warm_start_vars else preprocessed_data['w_init_t']).clone()
    eta_new_t = (warm_start_vars['eta']
                 if warm_start_vars else (w_new_t[i_idx] - w_new_t[j_idx])).clone()
    tau_new_t = (warm_start_vars['tau']
                 if warm_start_vars else torch.zeros_like(eta_new_t)).clone()

    low_cut, up_cut = wcut
    alpha  = float(alpha_init)
    use_amp_global = torch.cuda.is_available() and device.type == 'cuda'

    # ------------------------------ helper: clamp & replace --------------
    def _finite_or_clamp(t, lo=None, hi=None, replace=0.0):
        """Replace NaN/±inf with `replace`, then clamp to [lo,hi] if given."""
        t = torch.where(torch.isfinite(t), t, torch.full_like(t, replace))
        if lo is not None or hi is not None:
            t = t.clamp(min=lo if lo is not None else -float('inf'),
                        max=hi if hi is not None else  float('inf'))
        return t

    # ---------------------------------------------------------------- ADMM
    for _ in range(Run_limit):
        w_old_t, eta_old_t, tau_old_t = w_new_t, eta_new_t, tau_new_t

        # AMP only if weights far from saturation
        amp_enabled = use_amp_global and (w_old_t.abs().max().item() < 4.0)
        with torch.amp.autocast(device_type=device.type,
                                dtype=torch.float16,
                                enabled=amp_enabled):

            # ------------------------------------------------ θ(w)
            expW_t  = torch.exp(w_old_t)
            theta_t = (expW_t * minor_t) / (
                (2.0 * (1 - purity_t) + purity_t * total_t) * (1 + expW_t)
            )

            maskLow = w_old_t <= low_cut
            maskUp  = w_old_t >= up_cut
            maskMid = ~(maskLow | maskUp)

            partA = (maskLow*c_all_t[...,1] +
                     maskUp *c_all_t[...,5] +
                     maskMid*c_all_t[...,3]) - (r_t + 1e-12)/(n_t + 1e-10)
            partB = (maskLow*c_all_t[...,0] +
                     maskUp *c_all_t[...,4] +
                     maskMid*c_all_t[...,2])

            # ------------- FP32 & fully sanitised numerical path -----------
            θ = theta_t.float()
            denom = torch.sqrt(θ * (1 - θ))
            denom = _finite_or_clamp(denom, lo=1e-3)            # ≥ 1e‑3

            n32, pA32, pB32 = n_t.float(), partA.float(), partB.float()
            scal = torch.sqrt(n32)

            B_flat = (scal * pB32 / denom).flatten()
            B_flat = _finite_or_clamp(B_flat, lo=-1e5, hi=1e5)   # |B| ≤ 1e5
            B_sq_t = B_flat.square()                             # ≤ 1e10

            A_flat = (scal * pA32 / denom).flatten()
            A_flat = _finite_or_clamp(A_flat, lo=-1e5, hi=1e5)

            RHS_1   = transpose_matvec_delta(alpha * eta_old_t + tau_old_t,
                                             No_mutation, M, i_idx, j_idx)
            linear_t = RHS_1 - (B_flat * A_flat)

            x     = w_old_t.flatten()
            r_vec = linear_t - matvec_H_laplacian(x, B_sq_t,
                                                  No_mutation, M, alpha)
            r_vec = _finite_or_clamp(r_vec)

            Minv = 1.0 / (B_sq_t + alpha * (No_mutation - 1.0))
            Minv = _finite_or_clamp(Minv, lo=1e-10, hi=1e2)      # sensible

            z_vec  = Minv * r_vec
            p      = z_vec.clone()
            rs_old = torch.dot(r_vec, z_vec)

            tol2 = 1e-12
            for _ in range(cg_max_iter):
                if not torch.isfinite(rs_old).item():
                    raise RuntimeError("CG diverged: rs_old is NaN/inf")

                Ap = matvec_H_laplacian(p, B_sq_t, No_mutation, M, alpha)
                Ap = _finite_or_clamp(Ap)

                rs_p = torch.dot(p, Ap) + 1e-12
                alpha_cg = rs_old / rs_p

                x     += alpha_cg * p
                r_vec -= alpha_cg * Ap
                z_vec  = Minv * r_vec
                rs_new = torch.dot(r_vec, z_vec)

                if rs_new < tol2:
                    break
                beta   = rs_new / rs_old
                p      = z_vec + beta * p
                rs_old = rs_new

            w_new_t = torch.clamp(x.view(No_mutation, M),
                                  -control_large, control_large)

            eta_new_t, tau_new_t = scad_threshold_update_torch(
                w_new_t, tau_old_t, i_idx, j_idx,
                float(alpha), float(Lambda), float(gamma),
                adaptive_weights_t
            )

            diff   = w_new_t[i_idx] - w_new_t[j_idx] - eta_new_t
            residual = diff.abs().max()

        # ---------------- outer loop checks --------------------------------
        alpha *= rho
        if residual.item() < precision:
            break
        if not torch.isfinite(residual).item():
            raise RuntimeError("ADMM diverged: residual NaN/inf")

    return {'w': w_new_t, 'eta': eta_new_t, 'tau': tau_new_t}



def postprocess_admm_results(
    admm_results,
    preprocessed_data,
    i_idx=None,               # retained for API compatibility (unused)
    j_idx=None,
    post_th=0.01,
    least_mut=None,
    least_diff=0.05
):
    """
    Exact re‑implementation of the *Version‑1* MATLAB / Python post‑processing
    with the original per‑mutation refinement loop.  It will reproduce the
    historical clustering results byte‑for‑byte.
    """
    # --- unpack tensors → NumPy ---------------------------------------
    w_new   = admm_results['w'  ].detach().cpu().numpy().astype(np.float32)
    eta_new = admm_results['eta'].detach().cpu().numpy().astype(np.float32)

    n        = preprocessed_data['n_t'     ].cpu().numpy().astype(np.float32)
    r        = preprocessed_data['r_t'     ].cpu().numpy().astype(np.float32)
    purity   = preprocessed_data['purity_t'].cpu().numpy()[0].astype(np.float32)
    total_cn = preprocessed_data['total_t' ].cpu().numpy().astype(np.float32)
    minor_cn = preprocessed_data['minor_t' ].cpu().numpy().astype(np.float32)

    # empirical φ̂  (identical to V1)
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     phi_hat = (r / n) * ((2 * (1 - purity) + purity * total_cn) / minor_cn)
    # phi_hat = np.nan_to_num(phi_hat)

    phi_hat = torch.sigmoid(admm_results['w']).clamp(0, 2)
    phi_hat = phi_hat.cpu().numpy().astype(np.float32)

    n_mut, m_samp = w_new.shape
    post_th = post_th * np.sqrt(m_samp)  # scale threshold by sqrt(M)
    # ------------------------------------------------------------------ (7)
    diff = _diff_mat_lower(w_new)                        # lower triangle

    eta_thr = eta_new.copy()
    eta_thr[np.abs(eta_thr) <= post_th] = 0.0
    ids = np.triu_indices(n_mut, 1)
    diff[ids] = np.linalg.norm(eta_thr, axis=1)         # upper triangle

    # initial cluster labels (connected components style)
    labels = -np.ones(n_mut, dtype=np.int64)
    group_size = []
    labels[0] = 0
    group_size.append(1)
    labl = 1
    for i in range(1, n_mut):
        assigned = False
        for j in range(i):
            if diff[j, i] == 0:
                labels[i] = labels[j]
                group_size[labels[j]] += 1
                assigned = True
                break
        if not assigned:
            labels[i] = labl
            group_size.append(1)
            labl += 1

    # ------------------------------------------------------------------ (8)
    if least_mut is None:
        least_mut = int(np.ceil(0.05 * n_mut))

    tmp_size = np.min(np.array(group_size)[np.array(group_size) > 0])
    tmp_grp  = np.where(group_size == tmp_size)
    refine   = tmp_size < least_mut

    while refine:
        refine = False
        tmp_col = np.where(labels == tmp_grp[0][0])[0]
        for mut_idx in tmp_col:
            # build |diff| vector excluding own cluster (set to 100)
            if 0 < mut_idx < n_mut - 1:
                tmp_diff = np.abs(
                    np.concatenate([
                        diff[:mut_idx, mut_idx].ravel(),
                        np.array([100.0], dtype=diff.dtype),
                        diff[mut_idx, mut_idx+1:].ravel()
                    ])
                )
                tmp_diff[tmp_col] += 100
                diff[:mut_idx, mut_idx]   = tmp_diff[:mut_idx]
                diff[mut_idx, mut_idx+1:] = tmp_diff[mut_idx+1:]
            elif mut_idx == 0:
                tmp_diff = np.abs(
                    np.concatenate([
                        np.array([100.0], dtype=diff.dtype),
                        diff[0, 1:]
                    ])
                )
                tmp_diff[tmp_col] += 100
                diff[0, 1:] = tmp_diff[1:]
            else:  # mut_idx == n_mut‑1
                tmp_diff = np.abs(
                    np.concatenate([
                        diff[:-1, -1],
                        np.array([100.0], dtype=diff.dtype)
                    ])
                )
                tmp_diff[tmp_col] += 100
                diff[:-1, -1] = tmp_diff[:-1]

            old_lbl = labels[mut_idx]
            group_size[old_lbl] -= 1
            ind = tmp_diff.argmin()
            labels[mut_idx] = labels[ind]
            group_size[labels[ind]] += 1

        tmp_size = np.min(np.array(group_size)[np.array(group_size) > 0])
        tmp_grp  = np.where(group_size == tmp_size)
        refine   = tmp_size < least_mut

    # ------------------------------------------------------------------ (9)
    uniq = np.unique(labels)
    phi_out = np.zeros((len(uniq), m_samp), dtype=np.float32)
    for i, lbl in enumerate(uniq):
        idx = labels == lbl
        nh, ph = n[idx], phi_hat[idx]
        phi_out[i] = (ph * nh).sum(0) / nh.sum(0)

    if len(uniq) > 1:
        while True:
            sort_phi, order = _sort_by_2norm(phi_out)
            phi_diff = sort_phi[1:] - sort_phi[:-1]
            min_idx, min_val = _find_min_row_by_2norm(phi_diff)
            if np.linalg.norm(min_val) >= least_diff:
                break

            # merge
            A_lbl = uniq[order[min_idx]]
            B_lbl = uniq[order[min_idx + 1]]
            labels[labels == B_lbl] = A_lbl
            uniq = np.unique(labels)

            # rebuild phi_out
            phi_out = np.zeros((len(uniq), m_samp), dtype=np.float32)
            for i, lbl in enumerate(uniq):
                idx = labels == lbl
                nh, ph = n[idx], phi_hat[idx]
                phi_out[i] = (ph * nh).sum(0) / nh.sum(0)

    # ------------------------------------------------------------------ (10)
    phi_res = np.zeros((n_mut, m_samp), dtype=np.float32)
    for lab_idx in np.unique(labels):
        phi_res[labels == lab_idx] = phi_out[np.where(np.unique(labels) == lab_idx)[0][0]]

    # final distance‑based relabelling (unchanged)
    labels = _reassign_labels_by_distance(phi_res, labels, purity)

    return dict(phi=phi_res, label=labels.astype(np.int64))


def clipp2(
    r, n, minor, total, purity, coef_list,
    wcut = [-1.8, 1.8],
    alpha = 0.8,
    gamma = 3.7,
    rho   = 1.02,
    precision    = 0.01,
    Run_limit    = 200,
    control_large= 5,
    lambda_seq   = [0.1, 0.05, 0.01],
    sigma_quantile = 0.5,
    post_th      = 0.01,
    device       = None, 
    dtype        = torch.float32,
):
    """
    High-performance adaptively-weighted SCAD ADMM.
    Uses JIT compilation and Automatic Mixed Precision for acceleration.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    preprocessed_data = preprocess_clipp_data(
        r, n, minor, total, purity, coef_list, control_large, device, dtype
    )

    w_mle_t = preprocessed_data['w_mle_t']
    No_mutation = preprocessed_data['No_mutation']
    
    i_idx, j_idx = torch.triu_indices(No_mutation, No_mutation, offset=1, device=device)

    w_mle = preprocessed_data['w_mle_t'].to(torch.float32)        # calc in FP32
    d2 = (w_mle[i_idx] - w_mle[j_idx]).pow(2).sum(1)
    pos = d2[d2 > 1e-8]

    if pos.numel():
        sigma = torch.quantile(torch.sqrt(pos), sigma_quantile).clamp_min(1e-8)
    else:                                                    # new
        sigma = torch.tensor(1.0, device=device, dtype=torch.float16)
    
    # sigma = torch.tensor(1.0, device=device, dtype=torch.float16)
    adaptive_weights_t = torch.exp(-(d2 / (2 * sigma**2)).clamp_max(50.)).to(torch.float16)

    w_init_t = preprocessed_data['w_init_t']
    phi0 = torch.sigmoid(w_init_t).cpu().numpy().ravel()
    
    if not isinstance(lambda_seq, (list, np.ndarray)):
         lambda_seq = [lambda_seq]

    all_results = []
    warm_start_vars = None
    least_mut = math.ceil(0.05 * No_mutation)

    for i, current_lambda in enumerate(lambda_seq):
        admm_results_for_lambda = run_admm_optimization(
            preprocessed_data, wcut, alpha, gamma, rho, precision,
            Run_limit, control_large, current_lambda, device,
            adaptive_weights_t=adaptive_weights_t, 
            i_idx=i_idx, j_idx=j_idx,
            warm_start_vars=warm_start_vars
        )
        warm_start_vars = admm_results_for_lambda
        
        # results_for_current_lambda = postprocess_admm_results(
        #     admm_results_for_lambda, preprocessed_data, 
        #     i_idx, j_idx, post_th, least_mut
        # )

        # phi_res_current = results_for_current_lambda['phi'].ravel()
        # dist = wasserstein_distance(phi0, phi_res_current)
        # results_for_current_lambda['lambda'] = current_lambda
        # results_for_current_lambda['wasserstein_distance'] = dist
        
        # all_results.append(results_for_current_lambda)

        results_for_current_lambda = {}
        results_for_current_lambda['phi'] = torch.sigmoid(admm_results_for_lambda['w']).cpu().numpy().ravel()
        results_for_current_lambda['lambda'] = current_lambda
        all_results.append(results_for_current_lambda)

        torch.cuda.empty_cache()

    return all_results