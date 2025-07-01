"""
This script provides a research-oriented pipeline for a SCAD-penalized ADMM approach 
to multi-region subclone reconstruction in single-sample or multi-sample (M>1) scenarios. 

Main steps:
1) Preprocess data: Load files, validate shapes, and initialize parameters.
2) Run ADMM: Execute the core optimization loop with SCAD-based thresholding
   over a sequence of Lambda values with warm starts.
3) Postprocess results: For each Lambda, merge clusters and calculate the 
   Wasserstein distance between the result and the unpenalized estimate.
4) Return the final results along with the sequence of distances.

Author: [Yu Ding, Ph.D. / Wenyi Wang's Lab / MD Anderson Cancer Center]
Date: [Oct 2024]
Contact: [yding4@mdanderson.org, yding1995@gmail.com]
"""

from __future__ import annotations
import numpy as np
import torch
from scipy.stats import wasserstein_distance

torch.set_grad_enabled(False)                      
device_default = 'cuda' if torch.cuda.is_available() else 'cpu'

def _ensure_2d(a: np.ndarray) -> np.ndarray:
    """Guarantee 2-D view without copying when possible and avoid zeros."""
    if a.ndim == 1:
        a = a.reshape(-1, 1)          
    elif a.ndim != 2:
        raise ValueError(f"Expected 1- or 2-D array, got shape {a.shape}")
    return np.where(a == 0, 1, a)

def _to_tensor(a: np.ndarray, dtype: torch.dtype,
               device: str = device_default, pin: bool = False) -> torch.Tensor:
    t = torch.as_tensor(a, dtype=dtype, device=device)
    if pin and t.device.type == 'cpu':     # optional branch
        t = t.pin_memory()
    return t

def scad_threshold_update_torch(w_new_t: torch.Tensor,
                                tau_old_t: torch.Tensor,
                                i_idx: torch.Tensor,
                                j_idx: torch.Tensor,
                                alpha: float,
                                lam  : float,
                                gamma: float):
    """Vectorised, branch-free SCAD update."""
    D_w   = w_new_t[i_idx] - w_new_t[j_idx]              # pairwise diffs
    delt  = D_w - (tau_old_t / alpha)
    norm  = torch.norm(delt, dim=1)

    lam_over_alpha = lam / alpha
    gamma_lam      = gamma * lam
    denom2         = max(1.0 - 1.0/((gamma - 1)*alpha), 1e-12)

    # shrinkage factor s(norm) in closed form
    s = torch.where(
            norm <= lam_over_alpha,                                 0.0,
            torch.where(norm <= lam + lam_over_alpha,
                        1.0 - lam_over_alpha / norm,
                        torch.where(norm <= gamma_lam,
                                    (1.0 / denom2) *
                                    (1.0 - (gamma_lam/((gamma-1)*alpha)) / norm),
                                    1.0)
                       )
        )
    s.clamp_(min=0.0)
    eta_new_t = delt.mul_(s.unsqueeze(1))                # same memory as delt
    tau_old_t.sub_(alpha * (D_w - eta_new_t))            # in-place
    return eta_new_t, tau_old_t                          # tau_old_t is new now

def matvec_H_laplacian(x: torch.Tensor,
                       B_sq_t: torch.Tensor,
                       No_mutation: int,
                       M          : int,
                       alpha      : float):
    """H · x where H = diag(B²) + α L  (L=graph Laplacian on mutations)."""
    out = B_sq_t * x
    W   = out.view(No_mutation, M)                      # re-use `out`
    out.add_(alpha * (No_mutation * W - W.sum(0, keepdim=True)).flatten())
    return out

def transpose_matvec_delta(v_pairs: torch.Tensor,
                           No_mutation: int,
                           M          : int,
                           i_idx      : torch.Tensor,
                           j_idx      : torch.Tensor):
    """Sparse incidence-matrix transpose multiply."""
    z = torch.zeros((No_mutation, M), dtype=v_pairs.dtype, device=v_pairs.device)
    z.index_add_(0, i_idx,  v_pairs)
    z.index_add_(0, j_idx, -v_pairs)
    return z.flatten()

def sort_by_2norm(x):
    row_norms = np.linalg.norm(x, axis=1)
    sort_idx = np.argsort(row_norms)
    return x[sort_idx, :]

def reassign_labels_by_distance(a, b, ref, tol=1e-8):
    uniq, first_idx, inv = np.unique(b, return_index=True, return_inverse=True)
    reps = a[first_idx]
    diff = np.abs(a - reps[inv])
    if np.max(diff) > tol:
        bad = np.argmax(np.max(diff, axis=1))
        raise ValueError(f"Row {bad} differs from its rep by {np.max(diff):.3g}")
    dists = np.linalg.norm(reps - ref, axis=1)
    order = np.argsort(dists)
    new_label = np.empty_like(order)
    new_label[order] = np.arange(len(order))
    return new_label[inv]

def find_min_row_by_2norm(x):
    row_norms = np.linalg.norm(x, axis=1)
    min_index = np.argmin(row_norms)
    return min_index, x[min_index, :]

def diff_mat(w_new):
    No_mutation, M = w_new.shape
    mag = np.zeros((No_mutation, No_mutation), dtype=w_new.dtype)
    for i in range(No_mutation):
        mag[i, :] = np.linalg.norm(w_new - w_new[i, :], axis=1)
    first_coord_diff = w_new[None, :, 0] - w_new[:, None, 0]
    sign_mat = np.sign(first_coord_diff)
    diff_signed = -sign_mat * mag
    np.fill_diagonal(diff_signed, 0.0)
    i_idx, j_idx = np.triu_indices(No_mutation, k=1)
    diff_signed[j_idx, i_idx] = -diff_signed[i_idx, j_idx]
    return diff_signed


def preprocess_clipp_data(r, n, minor, total, purity,
                          coef_list, control_large,
                          *,
                          device: str   = device_default,
                          dtype : torch.dtype = torch.float32):
    """Convert inputs → torch tensors & initial estimates."""
    r, n, minor, total = map(_ensure_2d, (r, n, minor, total))

    r_t, n_t     = _to_tensor(r,     dtype, device), _to_tensor(n,     dtype, device)
    minor_t      = _to_tensor(minor, dtype, device)
    total_t      = _to_tensor(total, dtype, device)
    coef_t       = torch.stack([_to_tensor(c, dtype, device) for c in coef_list], dim=2)
    No_mut, M    = r_t.shape

    if isinstance(purity, (float, int)):
        purity_t = torch.full((No_mut, M), float(purity), device=device, dtype=dtype)
    else:
        purity_t = _to_tensor(purity, dtype, device).unsqueeze(0).expand(No_mut, -1)

    # initial φ̂
    frac      = (r_t + 1e-12) / (n_t + 1e-10)
    denominator = (2.0 * (1 - purity_t) + purity_t * total_t)
    phi_hat_t = frac * denominator / minor_t
    phi_hat_t = torch.clamp(phi_hat_t, min=1e-12)

    # scaled log-odds initial w
    scale_param = torch.clamp(phi_hat_t.max(), min=1.0).to(dtype)
    p = (phi_hat_t / scale_param).clamp_(1e-12, 1 - 1e-12)
    w_init_t = torch.log(p) - torch.log1p(-p)
    w_init_t.clamp_(-control_large, control_large)

    return dict(w_init_t = w_init_t, r_t = r_t, n_t = n_t, minor_t = minor_t,
                total_t = total_t, purity_t = purity_t, c_all_t = coef_t,
                phi_hat_t = phi_hat_t, No_mutation = No_mut, M = M,
                original_n = n, original_r = r, original_total = total,
                original_minor = minor)

def run_admm_optimization(pre, wcut, alpha_init, gamma, rho, precision,
                          Run_limit, control_large, lam, *,
                          device: str,
                          warm_start: dict | None = None):

    No_mut, M = pre['No_mutation'], pre['M']
    # cache pair indices once per run
    PAIRS_IDX = torch.triu_indices(No_mut, No_mut, offset=1, device=device)
    i_idx, j_idx = PAIRS_IDX

    if warm_start is None:
        w_new  = pre['w_init_t'].clone()
        eta    = w_new[i_idx] - w_new[j_idx]
        tau    = torch.zeros_like(eta)
    else:
        w_new, eta, tau = (warm_start[k].clone() for k in ('w', 'eta', 'tau'))

    low_cut, up_cut = wcut
    alpha = alpha_init
    residual, k_iter = 1e6, 0

    with torch.inference_mode():
        while k_iter < Run_limit and residual > precision:
            k_iter += 1
            w_old = w_new.clone()

            # ---------- (A)  IRLS  -------------------------------------------------- #
            expW    = torch.exp(w_old)
            theta   = (expW*pre['minor_t']) / ((2.0*(1-pre['purity_t']) +
                      pre['purity_t']*pre['total_t']) * (1+expW))

            maskL   = w_old <= low_cut
            maskH   = w_old >= up_cut
            maskM   = ~(maskL | maskH)

            partA   = (maskL*pre['c_all_t'][...,1] +
                       maskH*pre['c_all_t'][...,5] +
                       maskM*pre['c_all_t'][...,3]) - \
                      (pre['r_t']+1e-12)/(pre['n_t']+1e-10)
            partB   = (maskL*pre['c_all_t'][...,0] +
                       maskH*pre['c_all_t'][...,4] +
                       maskM*pre['c_all_t'][...,2])

            sqrtN   = torch.sqrt(pre['n_t'] + 1e-10)
            denom   = torch.sqrt(theta*(1-theta) + 1e-20)
            A_flat  = (sqrtN * partA / denom).flatten()
            B_flat  = (sqrtN * partB / denom).flatten()
            B_sq    = B_flat.square()

            # ---------- (B)  Conjugate-Gradient solve ------------------------------ #
            RHS     = transpose_matvec_delta(alpha*eta + tau,
                                             No_mut, M, i_idx, j_idx)
            lin_rhs = RHS - B_flat*A_flat
            x       = w_old.flatten()
            r_vec   = lin_rhs - matvec_H_laplacian(x, B_sq, No_mut, M, alpha)
            p       = r_vec.clone()
            rs_old  = torch.dot(r_vec, r_vec)

            for _ in range(200):
                Ap       = matvec_H_laplacian(p, B_sq, No_mut, M, alpha)
                alpha_cg = rs_old / (torch.dot(p, Ap) + 1e-12)
                x.add_(p, alpha=alpha_cg)
                r_vec.sub_(Ap, alpha=alpha_cg)
                rs_new   = torch.dot(r_vec, r_vec)
                if rs_new.sqrt() < 1e-6:
                    break
                p.mul_(rs_new/rs_old).add_(r_vec)
                rs_old = rs_new

            w_new = x.view(No_mut, M).clamp_(-control_large, control_large)

            # ---------- (C)  SCAD threshold  --------------------------------------- #
            eta, tau = scad_threshold_update_torch(w_new, tau, i_idx, j_idx,
                                                   alpha, lam, gamma)
            alpha  *= rho

            # ---------- (D)  residual --------------------------------------------- #
            residual = (w_new[i_idx] - w_new[j_idx] - eta).abs().max().item()

    return dict(w=w_new, eta=eta, tau=tau)

def postprocess_admm_results(admm_results, preprocessed_data, purity, post_th, least_diff):
    # ... (implementation from previous answer)
    w_final_np, eta_final_np = admm_results['w'].detach().cpu().numpy(), admm_results['eta'].detach().cpu().numpy()
    phi_hat = preprocessed_data['phi_hat_t'].detach().cpu().numpy()
    No_mutation, M = preprocessed_data['No_mutation'], preprocessed_data['M']
    n, r, total, minor = (preprocessed_data['original_n'], preprocessed_data['original_r'], preprocessed_data['original_total'], preprocessed_data['original_minor'])
    diff = diff_mat(w_final_np)
    ids = np.triu_indices(diff.shape[1], 1)
    eta_final_np[np.abs(eta_final_np) <= post_th] = 0
    diff[ids] = np.linalg.norm(eta_final_np, axis=1)
    class_label, labl, group_size = -np.ones(No_mutation, dtype=int), 1, [1]
    class_label[0] = 0
    for i in range(1, No_mutation):
        assigned = False
        for j in range(i):
            if diff[j, i] == 0:
                class_label[i] = class_label[j]; group_size[class_label[j]] += 1; assigned = True; break
        if not assigned:
            class_label[i] = labl; labl += 1; group_size.append(1)
    least_mut = np.ceil(0.05 * No_mutation)
    gs_array = np.array(group_size)
    if np.any(gs_array > 0):
        tmp_size = np.min(gs_array[gs_array > 0])
        refine = tmp_size < least_mut
        count = 0
        while refine:
            if count > 50: break
            count += 1
            refine = False
            tmp_grp = np.where(gs_array == tmp_size)[0]
            tmp_col = np.where(class_label == tmp_grp[0])[0]
            for mut_idx in tmp_col:
                if (mut_idx != 0) and (mut_idx != No_mutation - 1): tmp_diff = np.abs(np.concatenate([diff[:mut_idx, mut_idx].ravel(), [100], diff[mut_idx, mut_idx+1:].ravel()]))
                elif mut_idx == 0: tmp_diff = np.concatenate([[100], diff[0, 1:No_mutation]])
                else: tmp_diff = np.concatenate([diff[:No_mutation-1, No_mutation-1], [100]])
                tmp_diff[tmp_col] += 100
                if (mut_idx != 0) and (mut_idx != No_mutation - 1):
                    diff[:mut_idx, mut_idx] = tmp_diff[:mut_idx]
                    diff[mut_idx, mut_idx+1:] = tmp_diff[mut_idx+1:]
                elif mut_idx == 0: diff[0, 1:No_mutation] = tmp_diff[1:]
                else: diff[:No_mutation-1, No_mutation-1] = tmp_diff[:-1]
                old_lbl = class_label[mut_idx]
                gs_array[old_lbl] -= 1
                class_label[mut_idx] = class_label[tmp_diff.argmin()]
                gs_array[class_label[mut_idx]] += 1
            if np.any(gs_array > 0):
                tmp_size = np.min(gs_array[gs_array > 0])
                if tmp_size < least_mut: refine = True
    labels = np.unique(class_label)
    phi_out = np.zeros((len(labels), M))
    for i, lbl in enumerate(labels):
        cluster_idx = (class_label == lbl)
        class_label[cluster_idx] = i
        nh, ph = n[cluster_idx, :], phi_hat[cluster_idx, :]
        phi_out[i, :] = np.sum(ph * nh, axis=0) / np.sum(nh, axis=0)
    if len(labels) > 1:
        sort_phi = sort_by_2norm(phi_out)
        phi_diff = sort_phi[1:, :] - sort_phi[:-1, :]
        min_ind, min_val = find_min_row_by_2norm(phi_diff)
        count = 0
        while np.linalg.norm(min_val) < least_diff:
            if count > 50: break
            count += 1
            orig_label_A, orig_label_B = np.where(np.all(phi_out == sort_phi[min_ind], axis=1))[0][0], np.where(np.all(phi_out == sort_phi[min_ind + 1], axis=1))[0][0]
            class_label[class_label == orig_label_A] = orig_label_B
            labels = np.unique(class_label)
            phi_out = np.zeros((len(labels), M))
            for i, lbl in enumerate(labels):
                idx = (class_label == lbl)
                class_label[idx] = i
                nh, ph = n[idx, :], phi_hat[idx, :]
                phi_out[i, :] = np.sum(ph * nh, axis=0) / np.sum(nh, axis=0)
            if len(labels) == 1: break
            sort_phi = sort_by_2norm(phi_out)
            phi_diff = sort_phi[1:, :] - sort_phi[:-1, :]
            min_ind, min_val = find_min_row_by_2norm(phi_diff)
    phi_res = np.zeros((No_mutation, M))
    for lab_idx in range(phi_out.shape[0]): phi_res[class_label == lab_idx, :] = phi_out[lab_idx, :]
    purity_arr = np.array([purity]) if isinstance(purity, (int, float)) else purity
    final_labels = reassign_labels_by_distance(phi_res, class_label, purity_arr)
    phi_clip = np.clip(phi_res, 1e-15, 1 - 1e-15)
    denominator = 2.0 * (1.0 - purity_arr[None, :]) + purity_arr[None, :] * total
    pp_matrix = phi_clip * minor / denominator
    pp_matrix = pp_matrix.clip(1e-15, 1 - 1e-15)  
    logL_matrix = r * np.log(pp_matrix) + (n - r) * np.log(1 - pp_matrix)
    logL = np.sum(logL_matrix)
    N, K_clusters = No_mutation * M, len(np.unique(final_labels))
    k_params = K_clusters * M
    AIC, BIC = -2.0 * logL + 2.0 * k_params, -2.0 * logL + k_params * np.log(N)
    return {'phi': phi_res, 'label': final_labels, 'aic': AIC, 'bic': BIC}

# =============================================================================
# Main Orchestrator Function (Modified for Wasserstein Distance)
# =============================================================================

def clipp2(r, n, minor, total, purity, coef_list,
           *,
           wcut            = (-1.8, 1.8),
           alpha           = 0.8,
           gamma           = 3.7,
           rho             = 1.02,
           precision       = 0.01,
           Run_limit       = 200,
           control_large   = 5,
           lambda_seq      = (0.1, 0.05, 0.01),
           post_th         = 0.001,
           least_diff      = 0.01,
           device          = device_default,
           dtype           = torch.float32):

    pre = preprocess_clipp_data(r, n, minor, total, purity,
                                coef_list, control_large,
                                device=device, dtype=dtype)
    phi_unpen = pre['phi_hat_t'].cpu().numpy().ravel()

    if isinstance(lambda_seq, (float, int)):
        lambda_seq = (lambda_seq,)

    all_results  = []
    warm_start   = None
    for lam in lambda_seq:
        admm_out   = run_admm_optimization(pre, wcut, alpha, gamma, rho,
                                           precision, Run_limit,
                                           control_large, lam, device=device,
                                           warm_start=warm_start)
        warm_start = admm_out                                    # warm start
        res        = postprocess_admm_results(admm_out, pre,
                                              purity, post_th, least_diff)
        res['lambda']               = lam
        res['wasserstein_distance'] = wasserstein_distance(phi_unpen,
                                                           res['phi'].ravel())
        all_results.append(res)

    return all_results
    