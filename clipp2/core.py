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


def preprocess_clipp_data(r, n, minor, total, purity, coef_list, control_large, device, dtype):
    """Preprocesses data for single-sample or multi-sample (M>1) subclone reconstruction."""
    def ensure_2D_and_no_zeros(arr):
        if arr.ndim == 1: arr = arr.reshape(-1, 1)
        elif arr.ndim != 2: raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")
        return np.where(arr == 0, 1, arr)
    def to_torch_gpu(arr, local_dtype):
        return torch.as_tensor(arr, dtype=torch.float32, device=device)
        
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
    No_mutation, M = preprocessed_data['No_mutation'], preprocessed_data['M']
    w_new_t = (warm_start_vars['w'] if warm_start_vars else preprocessed_data['w_init_t']).clone()
    eta_new_t = (warm_start_vars['eta'] if warm_start_vars else w_new_t[i_idx, :] - w_new_t[j_idx, :]).clone()
    tau_new_t = (warm_start_vars['tau'] if warm_start_vars else torch.zeros_like(eta_new_t)).clone()
    
    r_t, n_t, minor_t, total_t, purity_t, c_all_t = (
        preprocessed_data['r_t'], preprocessed_data['n_t'], preprocessed_data['minor_t'],
        preprocessed_data['total_t'], preprocessed_data['purity_t'], preprocessed_data['c_all_t']
    )
    
    alpha = alpha_init
    low_cut, up_cut = wcut
    
    for k_iter in range(Run_limit):
        w_old_t, eta_old_t, tau_old_t = w_new_t, eta_new_t, tau_new_t
        
        # --- Start of high-performance computation block ---
        # FIX: Use modern torch.amp.autocast syntax
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(AMP_OK and device.type=='cuda')):
            expW_t = torch.exp(w_old_t)
            theta_t = (expW_t * minor_t) / ((2.0 * (1 - purity_t) + purity_t * total_t) * (1 + expW_t))
            
            maskLow, maskUp = w_old_t <= low_cut, w_old_t >= up_cut
            maskMid = ~maskLow & ~maskUp
            
            partA = (maskLow*c_all_t[...,1] + maskUp*c_all_t[...,5] + maskMid*c_all_t[...,3]) - (r_t+1e-12)/(n_t+1e-10)
            partB = (maskLow*c_all_t[...,0] + maskUp*c_all_t[...,4] + maskMid*c_all_t[...,2])
            
            A_flat = (torch.sqrt(n_t) * partA / torch.sqrt(theta_t * (1 - theta_t) + 1e-20)).flatten()
            B_flat = (torch.sqrt(n_t) * partB / torch.sqrt(theta_t * (1 - theta_t) + 1e-20)).flatten()
            B_sq_t = B_flat**2

            RHS_1 = transpose_matvec_delta(alpha * eta_old_t + tau_old_t, No_mutation, M, i_idx, j_idx)
            linear_t = RHS_1 - (B_flat * A_flat)
            
            x = w_old_t.flatten()
            r_vec = linear_t - matvec_H_laplacian(x, B_sq_t, No_mutation, M, alpha)
            
            Minv = 1.0 / (B_sq_t + alpha * (No_mutation - 1.0))
            z_vec = Minv * r_vec
            p, rs_old = z_vec.clone(), torch.dot(r_vec, z_vec)

            max_cg = 120              # keep safety cap
            tol2   = 1e-12            # squared tolerance
            for _ in range(max_cg):
                Ap       = matvec_H_laplacian(p, B_sq_t, No_mutation, M, alpha)
                rs_p     = torch.dot(p, Ap) + 1e-12      # fused reduction
                alpha_cg = rs_old / rs_p

                # in‑place fused updates (avoids 3 temporaries)
                x     = x + alpha_cg * p
                r_vec = r_vec - alpha_cg * Ap

                z_vec  = Minv * r_vec
                rs_new = torch.dot(r_vec, z_vec)
                if rs_new < tol2:
                    break
                beta = rs_new / rs_old
                p    = z_vec + beta * p
                rs_old = rs_new
                
            w_new_t = torch.clamp(x.view(No_mutation, M), -control_large, control_large)
            
            eta_new_t, tau_new_t = scad_threshold_update_torch(
                w_new_t, tau_old_t, i_idx, j_idx, alpha, Lambda, gamma, adaptive_weights_t
            )
            
            residual = torch.max(torch.abs(w_new_t[i_idx, :] - w_new_t[j_idx, :] - eta_new_t)).float()
        # --- End of high-performance block ---

        alpha *= rho
        if residual.item() < precision: break
        if torch.isnan(residual) or torch.isinf(residual): break

    return {'w': w_new_t, 'eta': eta_new_t, 'tau': tau_new_t}


@torch.jit.script
def _scatter_mean(src, index, K: int):
    out  = torch.zeros((K, src.reshape(src.size(0), -1).size(1)),
                       dtype=src.dtype, device=src.device)
    cnt  = torch.zeros((K,), dtype=torch.int32, device=src.device)
    out.index_add_(0, index, src)
    cnt.index_add_(0, index, torch.ones_like(index, dtype=torch.int32))
    return out / cnt.clamp_min(1).view(-1, 1).to(out.dtype)

def _log_ic(lbl_np, w, prep_data):
    lbl = torch.as_tensor(lbl_np, device=w.device, dtype=torch.long)
    K = int(lbl.max().item()) + 1
    
    phi_mean_by_cluster = _scatter_mean(torch.sigmoid(w), lbl, K)
    phi = phi_mean_by_cluster[lbl]

    r, n, mn, tt, pur = (
        prep_data['r_t'], prep_data['n_t'], prep_data['minor_t'],
        prep_data['total_t'], prep_data['purity_t']
    )
    
    denom = 2 * (1 - pur) + pur * tt
    pp = (phi * mn / denom).clamp(1e-15, 1 - 1e-15)
    logL = (r * pp.log() + (n - r) * (1 - pp).log()).sum().item()
    
    No_mutation, M = prep_data['No_mutation'], prep_data['M']
    kpar = K * M
    aic = -2 * logL + 2 * kpar
    bic = -2 * logL + kpar * math.log(No_mutation * M)
    return aic, bic, phi.cpu().numpy()

def postprocess_admm_results(admm_results, preprocessed_data, i_idx, j_idx, post_th, least_mut):
    w, eta = admm_results['w'], admm_results['eta']
    No_mutation, dev = preprocessed_data['No_mutation'], w.device
    n_samples = preprocessed_data['M']
    post_th = post_th * n_samples
    norm = torch.linalg.norm(eta, dim=1)
    keep = (norm <= post_th).nonzero().squeeze(1)
    parents = torch.arange(No_mutation, dtype=torch.int64,  device=dev)
    
    if keep.numel():
        # iterate log₂(N) times to converge
        for _ in range(15):
            pa = parents[i_idx[keep]]
            pb = parents[j_idx[keep]]
            new_parent = torch.minimum(pa, pb)
            old_parent = torch.maximum(pa, pb).to(torch.int64)
            parents.scatter_(0, old_parent, new_parent.to(torch.int64))
            parents = parents[parents]
    
    _, lbl = torch.unique(parents, return_inverse=True)
    K0 = int(lbl.max().item()) + 1

    if K0 > 1 and K0 > least_mut:
        centroids = _scatter_mean(w, lbl, K0).cpu().numpy()
        centroids = np.nan_to_num(centroids)
        Z = linkage(centroids, method='ward')
        
        _, bic0, _ = _log_ic(lbl.cpu().numpy(), w, preprocessed_data)
        best_k, best_bic = K0, bic0
        
        for k_test in range(1, K0):
            grp = fcluster(Z, k_test, 'maxclust') - 1
            _, bic_test, _ = _log_ic(grp[lbl.cpu().numpy()], w, preprocessed_data)
            if bic_test < best_bic: best_k, best_bic = k_test, bic_test
        
        if best_k < K0:
            final_grp = fcluster(Z, best_k, 'maxclust') - 1
            # Note: fcluster returns int32, which sets the dtype of lbl
            lbl = torch.as_tensor(final_grp, device=dev)[lbl.cpu()]

    while True:
        K_current = int(lbl.max().item()) + 1
        if K_current <= 1: break
        cluster_sizes = torch.bincount(lbl, minlength=K_current)
        small_clusters = (cluster_sizes < least_mut) & (cluster_sizes > 0)
        if not small_clusters.any(): break
        
        cluster_to_merge_idx = small_clusters.nonzero().squeeze(1)[0]
        centroids = _scatter_mean(w, lbl, K_current)
        distances = torch.linalg.norm(centroids - centroids[cluster_to_merge_idx], dim=1)
        distances[cluster_to_merge_idx] = float('inf')
        target_cluster_idx = torch.argmin(distances)
        
        # FIX: Cast source dtype to match destination dtype
        lbl[lbl == cluster_to_merge_idx] = target_cluster_idx.to(lbl.dtype)
        
        _, lbl = torch.unique(lbl, return_inverse=True)

    K_final = int(lbl.max().item()) + 1
    if K_final > 1:
        phi_centroids = torch.sigmoid(_scatter_mean(w, lbl, K_final))
        pur_ref = preprocessed_data['purity_t'][0]
        dist_to_ref = torch.linalg.norm(phi_centroids - pur_ref, dim=1).cpu().numpy()
        order = np.argsort(dist_to_ref)
        remap = np.empty(K_final, dtype=np.int32)
        remap[order] = np.arange(K_final)
        lbl_np = remap[lbl.cpu().numpy()]
    else:
        lbl_np = lbl.cpu().numpy()

    aic, bic, phi = _log_ic(lbl_np, w, preprocessed_data)
    return dict(phi=phi, label=lbl_np, aic=aic, bic=bic)


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
    post_th      = 0.001,
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
        
        
        results_for_current_lambda = postprocess_admm_results(
            admm_results_for_lambda, preprocessed_data, 
            i_idx, j_idx, post_th, least_mut
        )

        phi_res_current = results_for_current_lambda['phi'].ravel()
        dist = wasserstein_distance(phi0, phi_res_current)
        results_for_current_lambda['lambda'] = current_lambda
        results_for_current_lambda['wasserstein_distance'] = dist
        
        all_results.append(results_for_current_lambda)
        torch.cuda.empty_cache()

    return all_results