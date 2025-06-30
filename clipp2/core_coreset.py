"""
This module provides a modular and robust ADMM (Alternating Direction Method 
of Multipliers) framework for solving penalized likelihood problems and a
post-processing pipeline to refine and evaluate the results, as inspired by
the CliPP2 model.

The framework is architected for clarity, reusability, and performance, featuring
a generic core ADMM solver applicable to both the full problem (operating on S
SNVs) and a coreset approximation (operating on K centroids).

Public API:
- `ADMMConfig`: A dataclass for managing all solver hyperparameters.
- `solve_coreset_path`: Solves the coreset problem for a sequence of Lambda
  values using a warm-start strategy.
- `solve_original_problem`: Solves the full problem for all S SNVs.
- `reconstruct_full_solution_admm`: Performs Phase 2 reconstruction.
- `post_process_solution`: Applies refinement and clustering to a final
  solution matrix to derive final labels and model selection scores.
- `validate_inputs`: A utility function to check the integrity of input data.
- `self_test`: An internal verification function for framework correctness.

Author: [Yu Ding, Ph.D. / Wenyi Wang's Lab / MD Anderson Cancer Center]
Date: [Finalized with Post-Processing]
Contact: [yding4@mdanderson.org, yding1995@gmail.com]
"""

import torch
from typing import Dict, Any, Optional, List, Callable
import copy
from dataclasses import dataclass, field
import numpy as np
from math import comb

# ===================================================================
# SECTION 1: CONFIGURATION CLASS
# ===================================================================

@dataclass
class ADMMConfig:
    """A dataclass for managing all ADMM solver hyperparameters."""
    Lambda: float = 0.01; gamma: float = 3.7; alpha: float = 0.8
    rho: float = 1.02; precision: float = 1e-3; Run_limit: int = 200
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    wcut: List[float] = field(default_factory=lambda: [-1.8, 1.8])
    control_large: float = 5.0

# ===================================================================
# SECTION 2: LOW-LEVEL HELPERS & OPERATOR BUILDERS (Private)
# ===================================================================

# --- PyTorch-based helpers for ADMM ---

def _calculate_irls_terms(w_t, r_t, n_t, purity_t, total_t, minor_t, c_all_t, wcut):
    expP_t = torch.exp(w_t)
    denom_theta = (2.0 * (1 - purity_t) + purity_t * total_t) * (1 + expP_t)
    theta_t = torch.clamp((expP_t * minor_t) / denom_theta, 1e-9, 1 - 1e-9)
    maskLow_t, maskUp_t = (w_t <= wcut[0]), (w_t >= wcut[1])
    maskMid_t = ~(maskLow_t | maskUp_t)
    partA_full_t = (maskLow_t*c_all_t[...,1] + maskUp_t*c_all_t[...,5] + maskMid_t*c_all_t[...,3]) - (r_t / n_t.clamp(min=1))
    partB_full_t = (maskLow_t*c_all_t[...,0] + maskUp_t*c_all_t[...,4] + maskMid_t*c_all_t[...,2])
    sqrt_n_t = torch.sqrt(n_t.clamp(min=1))
    denom2_t = torch.sqrt(theta_t * (1 - theta_t)).clamp(min=1e-6)
    a_flat = ((sqrt_n_t * partA_full_t) / denom2_t).flatten()
    B_flat = ((sqrt_n_t * partB_full_t) / denom2_t).flatten()
    return a_flat, B_flat

def _weighted_scad_threshold_update(c_new_t, tau_old_t, Delta_coo, alpha, Lambda, gamma, penalty_weights):
    if penalty_weights is None:
        penalty_weights = torch.ones(tau_old_t.shape[0], 1, device=c_new_t.device)
    c_new_flat = c_new_t.reshape(-1)
    D_c_flat = torch.sparse.mm(Delta_coo, c_new_flat.unsqueeze(1)).squeeze(1)
    D_c = D_c_flat.view_as(tau_old_t)
    d_kl = D_c + (1.0 / alpha) * tau_old_t
    d_kl_norm = torch.norm(d_kl, dim=1).clamp(min=1e-12)
    weights = penalty_weights.squeeze()
    lam_w, gamma_lam_w = Lambda * weights, gamma * Lambda * weights
    lam_w_over_alpha = lam_w / alpha
    eta_new_t = torch.zeros_like(d_kl)
    mask1 = d_kl_norm <= lam_w_over_alpha
    mask2 = (d_kl_norm > lam_w_over_alpha) & (d_kl_norm <= lam_w + lam_w_over_alpha)
    mask3 = (d_kl_norm > lam_w + lam_w_over_alpha) & (d_kl_norm <= gamma_lam_w)
    mask4 = d_kl_norm > gamma_lam_w
    i1 = mask1.nonzero(as_tuple=True)[0]
    if i1.numel() > 0: eta_new_t[i1] = 0.0
    i2 = mask2.nonzero(as_tuple=True)[0]
    if i2.numel() > 0:
        scale = 1.0 - (lam_w_over_alpha[i2] / d_kl_norm[i2])
        eta_new_t[i2] = scale.unsqueeze(1) * d_kl[i2]
    i3 = mask3.nonzero(as_tuple=True)[0]
    if i3.numel() > 0:
        scale_factor = 1.0 / (1.0 - 1.0 / ((gamma - 1.0) * alpha))
        threshold = (gamma_lam_w[i3] / ((gamma - 1.0) * alpha)) / d_kl_norm[i3]
        scale = torch.clamp(1.0 - threshold, min=0.0)
        eta_new_t[i3] = scale_factor * scale.unsqueeze(1) * d_kl[i3]
    i4 = mask4.nonzero(as_tuple=True)[0]
    if i4.numel() > 0: eta_new_t[i4] = d_kl[i4]
    tau_new_t = tau_old_t + alpha * (D_c - eta_new_t)
    return eta_new_t, tau_new_t

def _build_delta_operator(num_items, M, device):
    idx1, idx2 = torch.triu_indices(num_items, num_items, offset=1, device=device)
    No_pairs = idx1.size(0)
    if No_pairs == 0:
        empty_sparse = torch.sparse_coo_tensor(torch.zeros((2, 0), dtype=torch.long, device=device), torch.zeros(0, device=device), (0, num_items * M))
        empty_square = torch.sparse_coo_tensor(torch.zeros((2, 0), dtype=torch.long, device=device), torch.zeros(0, device=device), (num_items * M, num_items * M))
        return empty_sparse, empty_square
    row_idx = (torch.arange(No_pairs, device=device).unsqueeze(1) * M + torch.arange(M, device=device)).flatten().repeat(2)
    col1_flat, col2_flat = (idx1.unsqueeze(1) * M + torch.arange(M, device=device)).flatten(), (idx2.unsqueeze(1) * M + torch.arange(M, device=device)).flatten()
    vals = torch.cat([torch.ones_like(col1_flat, dtype=torch.float32), -torch.ones_like(col2_flat, dtype=torch.float32)])
    indices = torch.stack([row_idx, torch.cat([col1_flat, col2_flat])])
    Delta_coo = torch.sparse_coo_tensor(indices, vals, (No_pairs * M, num_items * M), device=device).coalesce()
    DTD = torch.sparse.mm(Delta_coo.transpose(0, 1), Delta_coo)
    return Delta_coo, DTD

def _build_z_operator(S, K, cluster_assign, device):
    indices = torch.stack([torch.arange(S, device=device), cluster_assign.to(device)])
    values = torch.ones(S, device=device)
    Z_sparse = torch.sparse_coo_tensor(indices, values, (S, K), device=device)
    return Z_sparse, Z_sparse.transpose(0, 1).coalesce()

# --- NumPy-based helpers for Post-Processing ---

def _synthesize_eta_from_p(p_final_np: np.ndarray) -> np.ndarray:
    """Calculates the pairwise differences to create an eta matrix using vectorized operations."""
    S, M = p_final_np.shape
    all_diffs = p_final_np[:, None, :] - p_final_np[None, :, :]
    i_indices, j_indices = np.triu_indices(S, k=1)
    return all_diffs[i_indices, j_indices]

def _diff_mat(w_new: np.ndarray) -> np.ndarray:
    No_mutation, M = w_new.shape
    diff_vec = w_new[:, None, :] - w_new[None, :, :]
    mag = np.sqrt(np.sum(diff_vec**2, axis=2))
    # This sign logic is specific and needs to be preserved
    first_coord_diff = w_new[None, :, 0] - w_new[:, None, 0]
    sign_mat = np.sign(first_coord_diff)
    diff_signed = -sign_mat * mag
    np.fill_diagonal(diff_signed, 0)
    i, j = np.triu_indices(No_mutation, k=1)
    diff_signed[j, i] = -diff_signed[i, j]
    return diff_signed

def _sort_by_2norm(x: np.ndarray) -> np.ndarray:
    return x[np.argsort(np.linalg.norm(x, axis=1))]

def _find_min_row_by_2norm(x: np.ndarray) -> (int, np.ndarray):
    if x.shape[0] == 0: return -1, np.array([])
    row_norms = np.linalg.norm(x, axis=1)
    min_index = np.argmin(row_norms)
    return min_index, x[min_index, :]

def _reassign_labels_by_distance(a: np.ndarray, b: np.ndarray, ref: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    uniq_labels, first_idx, inv_map = np.unique(b, return_index=True, return_inverse=True)
    reps = a[first_idx]
    if np.max(np.abs(a - reps[inv_map])) > tol:
        # In a library, raising an error is good. For this context, a warning might be sufficient.
        print(f"Warning: A point differs from its cluster representative. Reassignment might be imperfect.")
    # Ensure ref is 1D for distance calculation
    ref_1d = ref.flatten()
    dists = np.linalg.norm(reps - ref_1d, axis=1)
    order = np.argsort(dists)
    new_label_map = np.empty_like(order)
    new_label_map[order] = np.arange(len(order))
    return new_label_map[inv_map]

def _post_process_solution(
    p_final: np.ndarray,
    eta_final: np.ndarray,
    raw_data_np: dict,
    post_th: float,
    least_diff: float,
    ebic_gamma: float
) -> dict:
    """
    A direct implementation of the clipp2 post-processing logic.
    CORRECTED: Fixes the infinite loop in small cluster refinement.
    """
    print("  -> Starting full clipp2 post-processing...")
    S, M = p_final.shape
    r, n = raw_data_np['r_t'], raw_data_np['n_t']
    purity, total, minor = raw_data_np['purity_t'], raw_data_np['total_t'], raw_data_np['minor_t']
    
    with np.errstate(divide='ignore', invalid='ignore'):
        fraction = r / n.clip(min=1)
        phi_hat = fraction * ((2.0 - 2.0*purity) + (purity * total)) / minor.clip(min=1e-9)
        phi_hat = np.nan_to_num(phi_hat)

    # --- Step 7: Build difference matrix and initial clusters ---
    diff = _diff_mat(p_final)
    ids = np.triu_indices(S, k=1)
    eta_final_norms = np.linalg.norm(eta_final, axis=1)
    diff[ids] = eta_final_norms
    diff[ids[1], ids[0]] = eta_final_norms
    
    class_label = -np.ones(S, dtype=int)
    current_label = 0
    for i in range(S):
        if class_label[i] == -1:
            class_label[i] = current_label
            for j in range(i + 1, S):
                if diff[i, j] <= post_th:
                    class_label[j] = current_label
            current_label += 1

    # --- Step 8: Refine small clusters (Corrected Logic) ---
    least_mut = np.ceil(0.05 * S)
    max_refine_iters = S 
    for iter_count in range(max_refine_iters):
        unique_labels, counts = np.unique(class_label, return_counts=True)
        
        if not counts.any() or np.min(counts) >= least_mut:
            break

        # Find the label of the smallest cluster
        smallest_cluster_label = unique_labels[np.argmin(counts)]
        
        # Get indices of mutations to be moved
        mut_indices_to_move = np.where(class_label == smallest_cluster_label)[0]
        
        # *** THE FIX: Explicitly define the set of valid targets ***
        # A valid target is any SNV NOT in the cluster being dissolved.
        valid_target_mask = (class_label != smallest_cluster_label)
        
        # If there are no other clusters to merge into, we must stop.
        if not np.any(valid_target_mask):
            break

        for mut_idx in mut_indices_to_move:
            # Get distances from this SNV to all other SNVs
            tmp_diff_vec = diff[mut_idx, :].copy()
            
            # Invalidate distances to all SNVs in the small cluster itself
            # and any other invalid targets by setting them to infinity.
            tmp_diff_vec[~valid_target_mask] = np.inf
            
            # Find the closest SNV in a valid, different cluster
            new_owner_idx = np.argmin(tmp_diff_vec)
            
            # Re-assign the label
            class_label[mut_idx] = class_label[new_owner_idx]

    # --- Step 9 & 10: Compute final cluster means and merge by distance ---
    # (The rest of the function is unchanged as its logic is sound)
    unique_labels, class_label = np.unique(class_label, return_inverse=True)
    while True:
        labels = np.unique(class_label)
        if len(labels) <= 1: break
        phi_out = np.zeros((len(labels), M))
        for i, lbl_val in enumerate(labels):
            idx = (class_label == lbl_val)
            nh, ph = n[idx, :], phi_hat[idx, :]
            phi_out[i, :] = np.sum(ph * nh, axis=0) / np.sum(nh, axis=0).clip(min=1)
        sort_phi = _sort_by_2norm(phi_out)
        phi_diff = sort_phi[1:, :] - sort_phi[:-1, :]
        min_ind, min_val = _find_min_row_by_2norm(phi_diff)
        if min_ind != -1 and np.linalg.norm(min_val) < least_diff:
            vec_a, vec_b = sort_phi[min_ind], sort_phi[min_ind + 1]
            labelA_idx = np.where(np.all(np.isclose(phi_out, vec_a), axis=1))[0][0]
            labelB_idx = np.where(np.all(np.isclose(phi_out, vec_b), axis=1))[0][0]
            labelA, labelB = labels[labelA_idx], labels[labelB_idx]
            class_label[class_label == labelA] = labelB
            unique_labels, class_label = np.unique(class_label, return_inverse=True)
        else:
            break
            
    # --- Finalize phi and labels ---
    phi_res, final_labels = np.zeros((S, M)), np.zeros_like(class_label)
    unique_labels = np.unique(class_label)
    for i, lbl_val in enumerate(unique_labels):
        cluster_idx = (class_label == lbl_val)
        nh, ph = n[cluster_idx, :], phi_hat[cluster_idx, :]
        final_phi_vec = np.sum(ph * nh, axis=0) / np.sum(nh, axis=0).clip(min=1)
        phi_res[cluster_idx, :] = final_phi_vec
        final_labels[cluster_idx] = i
    purity_ref_vector = purity[0, :]
    final_labels = _reassign_labels_by_distance(phi_res, final_labels, purity_ref_vector)

    # --- Step 11: AIC/BIC Calculation ---
    phi_clip = np.clip(phi_res, 1e-15, 1 - 1e-15)
    denominator = (2.0*(1.0 - purity) + purity*total).clip(min=1e-9)
    pp_matrix = np.clip(phi_clip * minor / denominator, 1e-9, 1-1e-9)
    logL_matrix = r * np.log(pp_matrix) + (n - r) * np.log(1 - pp_matrix)
    logL = np.sum(logL_matrix)
    N_data, K_clusters = S * M, len(np.unique(final_labels))
    k_params = K_clusters * M
    AIC = -2.0 * logL + 2.0 * k_params
    BIC = -2.0 * logL + k_params * np.log(N_data)
    try:
        log_omega = np.log(float(comb(N_data, k_params))) if 0 <= k_params <= N_data else 0
    except (ValueError, OverflowError):
        log_omega = 0
    ebic = BIC + 2.0 * log_omega * ebic_gamma
    
    return {'phi_final': phi_res, 'labels': final_labels, 'aic': AIC, 'bic': BIC, 'ebic': ebic}

# ===================================================================
# SECTION 3: CORE ADMM SOLVERS (Private)
# ===================================================================

def _run_admm_core(w_init_t, H_func, linear_t, Delta_coo, eta_init_t, tau_init_t, config, penalty_weights=None):
    w_new_t, eta_new_t, tau_new_t = w_init_t.clone(), eta_init_t.clone(), tau_init_t.clone()
    alpha = config.alpha
    print(f"  -> Entering generic ADMM core (Lambda={config.Lambda:.2e})...")
    for k_iter in range(1, config.Run_limit + 1):
        w_old_t, eta_old_t, tau_old_t = w_new_t.clone(), eta_new_t.clone(), tau_new_t.clone()
        RHS_admm = torch.sparse.mm(Delta_coo.transpose(0, 1), (alpha * eta_old_t - tau_old_t).flatten().unsqueeze(1)).squeeze(1)
        RHS_total = RHS_admm + linear_t
        x = w_old_t.flatten()
        r_vec = RHS_total - H_func(x, alpha); p = r_vec.clone(); rs_old = torch.dot(r_vec, r_vec)
        for _ in range(100):
            Ap = H_func(p, alpha); alpha_cg = rs_old / (torch.dot(p, Ap) + 1e-12)
            x += alpha_cg * p; r_vec -= alpha_cg * Ap; rs_new = torch.dot(r_vec, r_vec)
            if rs_new.sqrt() < 1e-6: break
            p = r_vec + (rs_new / (rs_old + 1e-12)) * p; rs_old = rs_new
        w_new_t = torch.clamp(x.view_as(w_init_t), -config.control_large, config.control_large)
        eta_new_t, tau_new_t = _weighted_scad_threshold_update(w_new_t, tau_old_t, Delta_coo, alpha, config.Lambda, config.gamma, penalty_weights)
        alpha *= config.rho
        residual = torch.max(torch.abs(torch.sparse.mm(Delta_coo, w_new_t.flatten().unsqueeze(1)).squeeze(1).view_as(eta_new_t) - eta_new_t)).item()
        if k_iter % 20 == 0: print(f"    Iter: {k_iter}, Residual: {residual:.6f}")
        if residual < config.precision or torch.isnan(torch.tensor(residual)): break
    print(f"  -> ADMM core finished after {k_iter} iterations.")
    return {'w_final_t': w_new_t, 'eta_final_t': eta_new_t, 'tau_final_t': tau_new_t}

def _solve_original_problem(init_data, config):
    print("Setting up Original Problem...")
    device = config.device
    r_t, n_t, purity_t, total_t, minor_t, c_all_t = (init_data['r_t'].to(device), init_data['n_t'].to(device), init_data['purity_t'].to(device), init_data['total_t'].to(device), init_data['minor_t'].to(device), init_data['c_all_t'].to(device))
    S, M = r_t.shape; w_init_t = init_data['p_hat_mle'].to(device)
    Delta_coo, DTD = _build_delta_operator(S, M, device)
    a_S_flat, B_S_flat = _calculate_irls_terms(w_init_t, r_t, n_t, purity_t, total_t, minor_t, c_all_t, config.wcut)
    def H_func_original(w_vec, alpha):
        return (B_S_flat**2) * w_vec + alpha * torch.sparse.mm(DTD, w_vec.unsqueeze(-1)).squeeze(-1)
    linear_t = -(B_S_flat * a_S_flat)
    s_idx, l_idx = torch.triu_indices(S, S, offset=1, device=device)
    eta_init_t = w_init_t[s_idx, :] - w_init_t[l_idx, :]
    tau_init_t = torch.zeros_like(eta_init_t, device=device)
    return _run_admm_core(w_init_t, H_func_original, linear_t, Delta_coo, eta_init_t, tau_init_t, config)

def _solve_coreset_problem(init_data, cluster_assign, cluster_size, config):
    print("Setting up Coreset Problem...")
    device = config.device
    c_init_t = init_data['w_init_t'].to(device)
    r_t, n_t, purity_t, total_t, minor_t, c_all_t = (init_data['r_t'].to(device), init_data['n_t'].to(device), init_data['purity_t'].to(device), init_data['total_t'].to(device), init_data['minor_t'].to(device), init_data['c_all_t'].to(device))
    K, M = c_init_t.shape; S = r_t.shape[0]
    cluster_assign, cluster_size = cluster_assign.to(device), cluster_size.to(device)
    Delta_coo, DTD = _build_delta_operator(K, M, device)
    Z_sparse, Z_sparse_T = _build_z_operator(S, K, cluster_assign, device)
    k_idx, l_idx = torch.triu_indices(K, K, offset=1, device=device)
    penalty_weights = (cluster_size[k_idx] * cluster_size[l_idx]).view(-1, 1)
    p_expanded_t = torch.matmul(Z_sparse, c_init_t)
    a_S_flat, B_S_flat = _calculate_irls_terms(p_expanded_t, r_t, n_t, purity_t, total_t, minor_t, c_all_t, config.wcut)
    def H_func_coreset(c_vec, alpha):
        p_vec = torch.matmul(Z_sparse, c_vec.view(K, M)).flatten()
        return torch.matmul(Z_sparse_T, ((B_S_flat**2) * p_vec).view(S, M)).flatten() + alpha * torch.sparse.mm(DTD, c_vec.unsqueeze(-1)).squeeze(-1)
    linear_t = torch.matmul(Z_sparse_T, (-(B_S_flat * a_S_flat)).view(S, M)).flatten()
    if 'eta_init_t' in init_data and 'tau_init_t' in init_data:
        eta_init_t, tau_init_t = init_data['eta_init_t'].to(device), init_data['tau_init_t'].to(device)
    else:
        eta_init_t = c_init_t[k_idx, :] - c_init_t[l_idx, :]
        tau_init_t = torch.zeros_like(eta_init_t, device=device)
    return _run_admm_core(c_init_t, H_func_coreset, linear_t, Delta_coo, eta_init_t, tau_init_t, config, penalty_weights)

def _reconstruct_full_solution_admm(coreset_results, init_data, config, cluster_size):
    """Performs Phase 2 using a dedicated, vectorized ADMM consensus algorithm."""
    print("\n--- Starting Phase 2: Full Solution Reconstruction (ADMM Consensus) ---")
    device = config.device
    c_anchors = coreset_results['w_final_t'].to(device)
    r_t, n_t, purity_t, total_t, minor_t, c_all_t = (
        init_data['r_t'].to(device), init_data['n_t'].to(device), init_data['purity_t'].to(device),
        init_data['total_t'].to(device), init_data['minor_t'].to(device), init_data['c_all_t'].to(device)
    )
    S, M = r_t.shape; K = c_anchors.shape[0]
    p_t = init_data['p_hat_mle'].to(device).clone()
    z_t = p_t.unsqueeze(1).expand(-1, K, -1)
    u_t = torch.zeros_like(z_t, device=device)
    penalty_weights = cluster_size.to(device).unsqueeze(0).expand(S, -1)
    alpha = config.alpha
    print(f"  -> Refining {S} SNVs in parallel...")
    for i in range(1, config.Run_limit + 1):
        p_old_t = p_t.clone()
        a_flat, B_flat = _calculate_irls_terms(p_t, r_t, n_t, purity_t, total_t, minor_t, c_all_t, config.wcut)
        B, a = B_flat.view(S, M), a_flat.view(S, M)
        sum_z_u = torch.sum(z_t - u_t, dim=1)
        RHS = alpha * sum_z_u - (B * a)
        Hessian_diag = B**2 + alpha * K
        p_t = torch.clamp(RHS / Hessian_diag, -config.control_large, config.control_large)
        p_expanded = p_t.unsqueeze(1).expand(-1, K, -1)
        d_ik = p_expanded + u_t
        z_t = _scad_proximal_operator_phase2(d_ik, c_anchors, penalty_weights, alpha, config)
        u_t += p_expanded - z_t
        change = torch.max(torch.abs(p_t - p_old_t))
        if i % 20 == 0: print(f"    Iter: {i}, Max Change in p: {change:.6f}")
        if change < config.precision: break
    print(f"--- Full Solution Reconstruction Finished after {i} iterations ---")
    return {'p_recon_t': p_t}

def _scad_proximal_operator_phase2(d_ik, c_anchors, penalty_weights, alpha, config):
    """Helper for the z-update in Phase 2."""
    d_flat = (d_ik - c_anchors.unsqueeze(0)).reshape(-1, d_ik.shape[-1])
    lam_w = (config.Lambda * penalty_weights).flatten()
    gamma_lam_w = (config.gamma * lam_w)
    z_update_flat = _scad_proximal_operator_core(d_flat, lam_w, gamma_lam_w, alpha, config.gamma)
    return (c_anchors.unsqueeze(0) + z_update_flat.reshape_as(d_ik))

def _scad_proximal_operator_core(d, lam, gamma_lam, alpha, gamma):
    """Core logic for the SCAD proximal operator on a flattened vector."""
    d_norm = torch.norm(d, dim=1).clamp(min=1e-12)
    lam_over_alpha = lam / alpha
    eta_new_t = torch.zeros_like(d)
    mask1 = d_norm <= lam + lam / alpha
    mask2 = (d_norm > lam + lam / alpha) & (d_norm <= gamma_lam)
    mask3 = d_norm > gamma_lam
    i1 = mask1.nonzero(as_tuple=True)[0]
    if i1.numel() > 0:
        scale = torch.clamp(1.0 - (lam_over_alpha[i1] / d_norm[i1]), min=0.0)
        eta_new_t[i1] = scale.unsqueeze(1) * d[i1]
    i2 = mask2.nonzero(as_tuple=True)[0]
    if i2.numel() > 0:
        num = (alpha * (gamma - 1.0) * d[i2]) - (gamma_lam[i2] / d_norm[i2]).unsqueeze(1) * d[i2]
        den = alpha * (gamma - 1.0) - 1.0
        eta_new_t[i2] = num / den
    i3 = mask3.nonzero(as_tuple=True)[0]
    if i3.numel() > 0: eta_new_t[i3] = d[i3]
    return eta_new_t

# ===================================================================
# SECTION 4: PUBLIC API (Continued)
# ===================================================================

def solve_coreset_path(init_data, cluster_assign, cluster_size, lambda_sequence, config):
    """Solves the coreset problem efficiently for a sequence of Lambda values using a warm-start strategy."""
    validate_inputs(init_data, cluster_assign=cluster_assign, cluster_size=cluster_size, is_coreset=True)
    lambda_sequence.sort()
    print(f"--- Starting Warm-Start ADMM Path for {len(lambda_sequence)} Lambda values ---")
    path_results = {}
    current_init_data = copy.deepcopy(init_data)
    for i, lam in enumerate(lambda_sequence):
        print(f"\nProcessing Path (Step {i+1}/{len(lambda_sequence)}): Lambda = {lam:.2e}")
        current_config = copy.deepcopy(config)
        current_config.Lambda = lam
        results = _solve_coreset_problem(current_init_data, cluster_assign, cluster_size, current_config)
        path_results[lam] = results
        current_init_data['w_init_t'] = results['w_final_t']
        current_init_data['eta_init_t'] = results['eta_final_t']
        current_init_data['tau_init_t'] = torch.zeros_like(results['eta_final_t'])
    return path_results

def solve_original_problem(init_data: Dict, config: ADMMConfig) -> Dict:
    """Sets up and solves the original S-mutation problem."""
    validate_inputs(init_data, is_coreset=False)
    return _solve_original_problem(init_data, config)

def solve_coreset_problem(init_data: Dict, cluster_assign: torch.Tensor, cluster_size: torch.Tensor, config: ADMMConfig) -> Dict:
    """Sets up and solves a single instance of the K-centroid coreset problem."""
    validate_inputs(init_data, cluster_assign=cluster_assign, cluster_size=cluster_size, is_coreset=True)
    return _solve_coreset_problem(init_data, cluster_assign, cluster_size, config)

def reconstruct_full_solution_admm(coreset_results: Dict, init_data: Dict, config: ADMMConfig, cluster_size: torch.Tensor) -> Dict:
    """Performs Phase 2 reconstruction using a dedicated ADMM consensus algorithm."""
    validate_inputs(init_data, is_coreset=False) # Phase 2 uses the full original data
    return _reconstruct_full_solution_admm(coreset_results, init_data, config, cluster_size)

def post_process_solution(
    p_final: np.ndarray,
    eta_final: None,
    raw_data_np: dict,
    post_th: float = 0.001,
    least_diff: float = 0.01,
    ebic_gamma: float = 13.5
) -> dict:
    """
    Applies the full clipp2 post-processing logic to a final solution matrix.

    This function serves as the public API wrapper for the detailed internal
    post-processing implementation. It calculates necessary preliminary
    values like phi_hat and then calls the internal logic.

    Args:
        p_final: The final (S, M) solution matrix from ADMM.
        raw_data_np: Dictionary with original data as NumPy arrays
                     (must contain 'r_t', 'n_t', 'purity_t', 'total_t', 'minor_t').
        post_th: Threshold for zeroing out small eta values.
        least_diff: Distance threshold for final cluster merging.
        ebic_gamma: Gamma parameter for eBIC calculation.

    Returns:
        A dictionary with final refined results, including 'phi_final', 'labels',
        'aic', 'bic', and 'ebic'.
    """
    # 1. Calculate phi_hat, which is required for post-processing but not for the ADMM solver itself.
    r, n = raw_data_np['r_t'], raw_data_np['n_t']
    purity, total, minor = raw_data_np['purity_t'], raw_data_np['total_t'], raw_data_np['minor_t']
    if eta_final is None:
        eta_final = _synthesize_eta_from_p(p_final)
    with np.errstate(divide='ignore', invalid='ignore'):
        fraction = r / n.clip(min=1)
        phi_hat = fraction * ((2.0 - 2.0*purity) + (purity * total)) / minor.clip(min=1e-9)
        phi_hat = np.nan_to_num(phi_hat)
        
    # Add phi_hat to the dictionary to pass it to the internal function
    data_for_pp = raw_data_np.copy()
    data_for_pp['phi_hat'] = phi_hat
    
    # 2. Call the internal implementation with the complete data.
    return _post_process_solution(
        p_final=p_final,
        eta_final=eta_final,
        raw_data_np=data_for_pp,
        post_th=post_th,
        least_diff=least_diff,
        ebic_gamma=ebic_gamma
    )

def validate_inputs(init_data, cluster_assign=None, cluster_size=None, is_coreset=False):
    """Performs sanity checks on the input data."""
    print("Validating inputs...")
    required_keys = ['r_t', 'n_t', 'purity_t', 'total_t', 'minor_t', 'c_all_t', 'p_hat_mle']
    if is_coreset: required_keys.append('w_init_t')
    for key in required_keys:
        if key not in init_data: raise ValueError(f"Input 'init_data' is missing key: '{key}'")
    S, M = init_data['r_t'].shape
    shape_map = {'n_t': (S, M), 'purity_t': (S, M), 'total_t': (S, M), 'minor_t': (S, M), 'c_all_t': (S, M, 6), 'p_hat_mle': (S, M)}
    for key, expected_shape in shape_map.items():
        if init_data[key].shape != expected_shape: raise ValueError(f"Shape mismatch for '{key}': Expected {expected_shape}, got {init_data[key].shape}")
    if torch.any(init_data['n_t'] < init_data['r_t']): raise ValueError("'r_t' cannot be greater than 'n_t'.")
    if torch.any((init_data['purity_t'] < 0) | (init_data['purity_t'] > 1)): raise ValueError("'purity_t' must be in [0, 1].")
    if is_coreset:
        if cluster_assign is None or cluster_size is None: raise ValueError("Coreset problem requires 'cluster_assign' and 'cluster_size'.")
        K = init_data['w_init_t'].shape[0]
        if cluster_assign.shape != (S,): raise ValueError(f"Shape mismatch for 'cluster_assign': Expected ({S},), got {cluster_assign.shape}")
        if cluster_size.shape != (K,): raise ValueError(f"Shape mismatch for 'cluster_size': Expected ({K},), got {cluster_size.shape}")
        if K > 0 and cluster_assign.max() >= K: raise ValueError(f"Invalid 'cluster_assign': index {cluster_assign.max()} out of bounds for K={K}.")
        if int(cluster_size.sum()) != S: raise ValueError(f"Inconsistent cluster sizes: sum of 'cluster_size' ({int(cluster_size.sum())}) does not equal S ({S}).")
    print("Input validation successful.")
    
def self_test():
    """Runs a self-test of the framework using the S=K verification case."""
    print("--- Running Framework Self-Test (S=K Consistency Check) ---")
    S = 10; K = 10; M = 2; K_true = 2; device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    torch.manual_seed(42); np.random.seed(42)
    true_centers = (torch.rand(K_true, M, dtype=torch.float32, device=device)-0.5)*5.0
    true_assignments = torch.randint(0, K_true, (S,), device=device)
    true_snv_params = true_centers[true_assignments] + torch.randn(S, M, dtype=torch.float32, device=device)*0.1
    n_t = torch.randint(80, 200, (S, M), dtype=torch.float32, device=device)
    minor_t = torch.ones(S, M, dtype=torch.float32, device=device); total_t = torch.full((S, M), 2.0, dtype=torch.float32, device=device); purity_t = torch.full((S, M), 0.8, dtype=torch.float32, device=device)
    w_pts = torch.tensor([-4.0, -1.8, 1.8, 4.0], device=device)
    expw = torch.exp(w_pts); onep = 1 + expw
    theta_den = onep.view(4, 1, 1) * (2.0*(1 - purity_t) + total_t * purity_t)
    act = (expw.view(4, 1, 1) * minor_t) / theta_den.clamp(min=1e-6)
    a1 = (act[1] - act[0]) / (w_pts[1] - w_pts[0]); b1 = act[0] - a1 * w_pts[0]
    a2 = (act[2] - act[1]) / (w_pts[2] - w_pts[1]); b2 = act[1] - a2 * w_pts[1]
    a3 = (act[3] - act[2]) / (w_pts[3] - w_pts[2]); b3 = act[2] - a3 * w_pts[2]
    c_all_t = torch.stack([a1, b1, a2, b2, a3, b3], dim=2)
    expP_t_true = torch.exp(true_snv_params)
    denom_true = (2.0 * (1 - purity_t) + purity_t * total_t) * (1 + expP_t_true)
    true_theta_t = torch.clamp((expP_t_true * minor_t) / denom_true, 1e-9, 1 - 1e-9)
    r_t = torch.distributions.Binomial(total_count=n_t, probs=true_theta_t).sample()
    observed_vaf = r_t / n_t.clamp(min=1)
    adjustment_factor = (2.0 * (1 - purity_t) + purity_t * total_t)
    numerator = observed_vaf * adjustment_factor
    p_hat_mle = torch.log(numerator / (minor_t - numerator).clamp(min=1e-6)).clamp(-10, 10)
    raw_data = {'r_t': r_t, 'n_t': n_t, 'minor_t': minor_t, 'total_t': total_t, 'purity_t': purity_t, 'c_all_t': c_all_t, 'p_hat_mle': p_hat_mle}
    cluster_assign_t = torch.arange(S)
    raw_data['w_init_t'] = p_hat_mle.clone()
    cluster_size_t = torch.ones(S)
    config = ADMMConfig(Lambda=5e-3, precision=1e-4, Run_limit=50, device=device)
    results_coreset = _solve_coreset_problem(raw_data, cluster_assign_t, cluster_size_t, config)
    results_original = _solve_original_problem(raw_data, config)
    distance = torch.norm(results_coreset['w_final_t'] - results_original['w_final_t']).item()
    print("\n--- SELF-TEST COMPLETE ---")
    print(f"L2 Distance between Original and Coreset (S=K) solutions: {distance:.6g}")
    if distance < 1e-4: print("✅ SUCCESS: The framework is consistent and working correctly.")
    else: print("❌ FAILURE: A significant discrepancy exists between the problem formulations.")
    print("--------------------------")