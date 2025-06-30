"""
This script provides a research-oriented pipeline for a SCAD-penalized ADMM approach
to multi-region subclone reconstruction in single-sample or multi-sample (M>1) scenarios.

The model, CliPP2, is a statistical framework for identifying subclonal structures by
grouping Single Nucleotide Variants (SNVs) that share common evolutionary patterns across
multiple tumor samples. It achieves this by minimizing a penalized negative log-likelihood
objective function, where a Group SCAD penalty is applied to the pairwise differences
of logistic-scale cellular prevalence vectors ($\vec{p}_i$) for each SNV.

The objective function is:
    min_{P} { -l(P) + sum_{i < i'} SCAD_lambda(||p_i - p_{i'}||_2) }

This non-convex problem is solved using the Alternating Direction Method of Multipliers (ADMM),
which involves iteratively solving for the prevalence matrix P and auxiliary variables.

Main Pipeline Steps:
1) Preprocess Data: Load input files (read counts, copy number, etc.), validate
   data shapes, compute initial unpenalized estimates, and move data to the
   specified compute device (CPU/GPU).
2) Run ADMM Optimization: Execute the core ADMM optimization loop. This involves
   iteratively updating a quadratic approximation of the likelihood (IRLS step),
   solving a linear system for the primary variables `P` using the Conjugate
   Gradient method, and applying a group SCAD thresholding operator for the
   auxiliary variables. This process is run over a sequence of regularization
   parameters (Lambda), using warm starts to enhance convergence.
3) Postprocess Results: For each Lambda, the raw output is processed to form
   distinct clusters. This involves merging small or proximal clusters based on
   pre-defined heuristics. Model selection criteria (AIC, BIC) are computed.
4) Calculate Quality Metrics: The Wasserstein distance between the penalized
   result and the initial unpenalized estimate is calculated for each Lambda to
   quantify the effect of regularization.
5) Return Results: The final output is a list of dictionaries, with each dictionary
   containing the comprehensive results (phi, labels, AIC, BIC, distance) for
   one Lambda value.

Author: [Yu Ding, Ph.D. / Wenyi Wang's Lab / MD Anderson Cancer Center]
Date: [Oct 2024]
Contact: [yding4@mdanderson.org, yding1995@gmail.com]
"""

import os
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.special import expit, logit
import torch
from scipy.stats import wasserstein_distance
from typing import List, Dict, Any, Tuple, Union, Optional

# =============================================================================
# I. CORE ALGORITHMIC COMPONENTS (ADMM UPDATES & LINEAR ALGEBRA)
# =============================================================================

def scad_threshold_update_torch(
    w_new_t: torch.Tensor,
    tau_old_t: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    alpha: float,
    Lambda: float,
    gamma: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs the eta-update and tau-update steps of the ADMM algorithm.

    This function implements the group SCAD thresholding operator, which is the
    proximal operator for the SCAD penalty. It solves the subproblem for the
    auxiliary variable `eta` and then updates the dual variable `tau`.

    Ref: Supplementary Information, Section 2.2 (eta-update).

    The update for `eta` is:
    eta_new = argmin_{eta} { SCAD_lambda(||eta||_2) + (alpha/2) * ||eta - delta||_2^2 }
    where delta = (p_i - p_j) + (1/alpha) * tau_old.

    The variable `tau` in this code corresponds to the negative scaled dual variable,
    i.e., tau = -alpha * u. The update rule used here,
    tau_new = tau_old - alpha * (residual), is equivalent to the standard scaled
    dual update u_new = u_old + residual.

    Parameters
    ----------
    w_new_t : torch.Tensor
        The updated logistic-scale parameter matrix P (S x M). Corresponds to `p^(k+1)`.
    tau_old_t : torch.Tensor
        The dual variable from the previous iteration. Corresponds to `tau^(k)`.
    i_idx, j_idx : torch.Tensor
        Indices for the upper triangle of the pairwise difference matrix.
    alpha : float
        The augmented Lagrangian penalty parameter.
    Lambda : float
        The SCAD regularization parameter (lambda).
    gamma : float
        The SCAD shape parameter (a > 2, typically 3.7).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - eta_new_t: The updated auxiliary variable. Corresponds to `eta^(k+1)`.
        - tau_new_t: The updated dual variable. Corresponds to `tau^(k+1)`.
    """
    # Calculate the argument for the proximal operator
    D_w = w_new_t[i_idx, :] - w_new_t[j_idx, :]
    delt_t = D_w - (1.0 / alpha) * tau_old_t  # This is `delta` in the math derivation
    delt_norm_t = torch.norm(delt_t, dim=1)

    # Pre-calculate constants for efficiency
    lam_over_alpha = Lambda / alpha
    gamma_lam = gamma * Lambda

    # Apply the three-part SCAD thresholding rule based on the norm of `delta`
    mask2 = (delt_norm_t > lam_over_alpha) & (delt_norm_t <= (Lambda + lam_over_alpha))
    mask3 = (delt_norm_t > (Lambda + lam_over_alpha)) & (delt_norm_t <= gamma_lam)
    mask4 = (delt_norm_t > gamma_lam)

    eta_new_t = torch.zeros_like(delt_t)

    # Part 1: Soft-thresholding region
    if mask2.any():
        i2 = mask2.nonzero(as_tuple=True)[0]
        scale2 = torch.clamp(1.0 - (lam_over_alpha / delt_norm_t[i2]), min=0.0)
        eta_new_t[i2] = scale2.unsqueeze(1) * delt_t[i2]

    # Part 2: Intermediate region (moves solution towards `delta`)
    if mask3.any():
        i3 = mask3.nonzero(as_tuple=True)[0]
        denom2 = 1.0 - 1.0/((gamma - 1)*alpha)
        if denom2 > 0:
            scale3 = (1.0 / denom2) * torch.clamp(1.0 - ((gamma_lam / ((gamma - 1)*alpha)) / delt_norm_t[i3]), min=0.0)
            eta_new_t[i3] = scale3.unsqueeze(1) * delt_t[i3]
        else:
            eta_new_t[i3] = delt_t[i3] # Fallback if denominator is non-positive

    # Part 3: Unpenalized region (solution is `delta`)
    if mask4.any():
        i4 = mask4.nonzero(as_tuple=True)[0]
        eta_new_t[i4] = delt_t[i4]

    # Dual variable update (tau-update)
    # This corresponds to tau^(k+1) <- tau^(k) + alpha * (Delta*p^(k+1) - eta^(k+1))
    # with a sign convention difference.
    tau_new_t = tau_old_t - alpha * (D_w - eta_new_t)

    return eta_new_t, tau_new_t

def matvec_H_laplacian(x: torch.Tensor, B_sq_t: torch.Tensor, No_mutation: int, M: int, alpha: float) -> torch.Tensor:
    """
    Computes the matrix-vector product for the Conjugate Gradient solver.

    This function calculates `y = Hx` where `H = B^T*B + alpha * Delta^T*Delta`.
    `B` is the diagonal matrix from the quadratic approximation of the likelihood,
    and `Delta^T*Delta` is the graph Laplacian matrix. This operation is a key
    part of the p-update step.

    Ref: Supplementary Information, Section 3 (Conjugate gradient method).

    Parameters
    ----------
    x : torch.Tensor
        The vector to be multiplied by the matrix H (flattened, size SM).
    B_sq_t : torch.Tensor
        The squared diagonal elements of the matrix B (flattened, size SM).
    No_mutation : int
        Number of SNVs (S).
    M : int
        Number of samples.
    alpha : float
        The augmented Lagrangian penalty parameter.

    Returns
    -------
    torch.Tensor
        The result of the matrix-vector product `Hx`.
    """
    # Part 1: B^T*B * x (element-wise product since B is diagonal)
    out = B_sq_t * x

    # Part 2: alpha * Delta^T*Delta * x (the graph Laplacian part)
    # Reshape x into the matrix W (S x M) to compute the Laplacian action.
    W = x.view(No_mutation, M)
    # The action of the graph Laplacian L = Delta^T*Delta on W is S*W - sum(W),
    # where the sum is broadcasted across rows.
    sum_of_rows = W.sum(dim=0, keepdim=True)
    L_times_W = No_mutation * W - sum_of_rows
    out.add_(L_times_W.flatten(), alpha=alpha)

    return out

def transpose_matvec_delta(v_pairs: torch.Tensor, No_mutation: int, M: int, i_idx: torch.Tensor, j_idx: torch.Tensor) -> torch.Tensor:
    """
    Computes the transpose matrix-vector product for the operator Delta.

    This function calculates `z = Delta^T * v`, where `Delta` is the pairwise
    difference operator. This is used to construct the right-hand side of the
    linear system in the p-update.

    Ref: Supplementary Information, Section 2.1, Eq. (7).

    Parameters
    ----------
    v_pairs : torch.Tensor
        The vector representing values on the edges of the SNV graph.
    No_mutation : int
        Number of SNVs (S).
    M : int
        Number of samples.
    i_idx, j_idx : torch.Tensor
        Indices for the pairs defining the Delta operator.

    Returns
    -------
    torch.Tensor
        The result of the product `Delta^T * v`, a flattened vector of size SM.
    """
    z_mat = torch.zeros((No_mutation, M), dtype=v_pairs.dtype, device=v_pairs.device)
    # The action of Delta^T is to aggregate the differences back to the nodes.
    # For a pair (i,j), v_ij contributes positively to node i and negatively to node j.
    z_mat.index_add_(0, i_idx, v_pairs)
    z_mat.index_add_(0, j_idx, -v_pairs)
    return z_mat.flatten()

def run_admm_optimization(
    preprocessed_data: Dict[str, Any],
    wcut: List[float],
    alpha_init: float,
    gamma: float,
    rho: float,
    precision: float,
    Run_limit: int,
    control_large: float,
    Lambda: float,
    device: str,
    warm_start_vars: Optional[Dict[str, torch.Tensor]] = None
) -> Dict[str, torch.Tensor]:
    """
    Executes the core ADMM optimization loop for a single Lambda value.

    This function implements the full ADMM algorithm by iteratively updating
    the quadratic approximation of the likelihood, solving for the primary
    variable `w` (P matrix), and updating the auxiliary (`eta`) and dual (`tau`)
    variables until convergence.

    Ref: Supplementary Information, Section 2 (Computational structure using ADMM).

    Parameters
    ----------
    preprocessed_data : dict
        A dictionary containing all necessary preprocessed data tensors.
    wcut : list
        Cutoff points for the piecewise-linear approximation of the logit function.
    alpha_init : float
        Initial value for the ADMM penalty parameter `alpha`.
    gamma : float
        The SCAD shape parameter.
    rho : float
        The multiplicative factor to increase `alpha` in each iteration.
    precision : float
        The convergence tolerance for the primal residual.
    Run_limit : int
        Maximum number of ADMM iterations.
    control_large : float
        Value to clamp the logistic-scale parameters `w` to prevent divergence.
    Lambda : float
        The SCAD regularization parameter for this run.
    device : str
        The compute device ('cuda' or 'cpu').
    warm_start_vars : dict, optional
        A dictionary with 'w', 'eta', 'tau' from a previous run to use as a
        warm start. Defaults to None.

    Returns
    -------
    dict
        A dictionary containing the converged variables: 'w', 'eta', 'tau'.
    """
    # --- Step 0: Unpack Data and Initialize ---
    w_init_t = preprocessed_data['w_init_t']
    r_t, n_t, minor_t, total_t, purity_t, c_all_t = (
        preprocessed_data['r_t'], preprocessed_data['n_t'], preprocessed_data['minor_t'],
        preprocessed_data['total_t'], preprocessed_data['purity_t'], preprocessed_data['c_all_t']
    )
    No_mutation, M = preprocessed_data['No_mutation'], preprocessed_data['M']

    # Indices for all unique pairs of mutations (i, j) where i < j
    i_idx, j_idx = torch.triu_indices(No_mutation, No_mutation, offset=1, device=device)

    # Initialize from warm start if provided, otherwise initialize from scratch
    if warm_start_vars:
        w_new_t = warm_start_vars['w'].clone()
        eta_new_t = warm_start_vars['eta'].clone()
        tau_new_t = warm_start_vars['tau'].clone()
    else:  # First run, no warm start available
        w_new_t = w_init_t.clone()
        eta_new_t = w_new_t[i_idx, :] - w_new_t[j_idx, :]
        tau_new_t = torch.ones_like(eta_new_t)

    # --- Main ADMM Iteration Loop ---
    residual, k_iter = 1e6, 0
    low_cut, up_cut = wcut
    alpha = alpha_init  # Use a fresh alpha for each Lambda run

    while (k_iter < Run_limit) and (residual > precision):
        k_iter += 1
        w_old_t, eta_old_t, tau_old_t = w_new_t.clone(), eta_new_t.clone(), tau_new_t.clone()

        # --- (A) IRLS: Update Quadratic Approximation of Likelihood ---
        expW_t = torch.exp(w_old_t)
        # Calculate theta_ij based on the current `w` (p)
        theta_t = (expW_t * minor_t) / ((2.0 * (1 - purity_t) + purity_t * total_t) * (1 + expW_t))
        # Use piecewise-linear surrogate coefficients for the logistic function
        maskLow, maskUp, maskMid = (w_old_t <= low_cut), (w_old_t >= up_cut), ~(w_old_t <= low_cut) & ~(w_old_t >= up_cut)
        # partA corresponds to (v_ij - r_ij/n_ij) in the derivation
        partA = (maskLow*c_all_t[...,1] + maskUp*c_all_t[...,5] + maskMid*c_all_t[...,3]) - (r_t+1e-12)/(n_t+1e-10)
        # partB corresponds to u_ij in the derivation
        partB = (maskLow*c_all_t[...,0] + maskUp*c_all_t[...,4] + maskMid*c_all_t[...,2])
        # Construct the vectorized `a` and diagonal `B` for the quadratic form: -1/2 * ||B*p + a||^2
        A_flat = (torch.sqrt(n_t + 1e-10) * partA / torch.sqrt(theta_t * (1 - theta_t) + 1e-20)).flatten()
        B_flat = (torch.sqrt(n_t + 1e-10) * partB / torch.sqrt(theta_t * (1 - theta_t) + 1e-20)).flatten()
        B_sq_t = B_flat**2

        # --- (B) P-Update: Solve Linear System via Conjugate Gradient ---
        # Construct the right-hand side of the linear system: alpha*Delta^T*(eta - (1/alpha)*tau) - B^T*a
        RHS_1 = transpose_matvec_delta(alpha * eta_old_t + tau_old_t, No_mutation, M, i_idx, j_idx)
        linear_t = RHS_1 - (B_flat * A_flat) # Note: B is diagonal, so B^T*a = B*a
        # Solve (B^T*B + alpha*Delta^T*Delta) * x = linear_t for x
        x = w_old_t.flatten()
        r_vec = linear_t - matvec_H_laplacian(x, B_sq_t, No_mutation, M, alpha)
        p, rs_old = r_vec.clone(), torch.dot(r_vec, r_vec)
        for _ in range(200): # CG inner loop
            Ap = matvec_H_laplacian(p, B_sq_t, No_mutation, M, alpha)
            alpha_cg = rs_old / (torch.dot(p, Ap) + 1e-12)
            x.add_(p, alpha=alpha_cg); r_vec.sub_(Ap, alpha=alpha_cg)
            rs_new = torch.dot(r_vec, r_vec)
            if rs_new.sqrt() < 1e-6: break
            p = r_vec + (rs_new / rs_old) * p
            rs_old = rs_new
        w_new_t = torch.clamp(x.view(No_mutation, M), -control_large, control_large)

        # --- (C) Eta/Tau-Update: SCAD Thresholding and Dual Update ---
        eta_new_t, tau_new_t = scad_threshold_update_torch(w_new_t, tau_old_t, i_idx, j_idx, alpha, Lambda, gamma)
        alpha *= rho # Update the ADMM penalty parameter

        # --- (D) Convergence Check ---
        # Calculate the primal residual: max(||Delta*p^(k+1) - eta^(k+1)||)
        D_w2 = w_new_t[i_idx, :] - w_new_t[j_idx, :]
        residual = torch.max(torch.abs(D_w2 - eta_new_t)).item()
        if np.isnan(residual):
            print("Warning: Residual is NaN. Terminating optimization.")
            break

    return {'w': w_new_t, 'eta': eta_new_t, 'tau': tau_new_t}


# =============================================================================
# II. PRE- AND POST-PROCESSING UTILITY FUNCTIONS
# =============================================================================

def sort_by_2norm(x: np.ndarray) -> np.ndarray:
    """Sorts matrix rows by their L2 norm."""
    row_norms = np.linalg.norm(x, axis=1)
    sort_idx = np.argsort(row_norms)
    return x[sort_idx, :]

def reassign_labels_by_distance(a: np.ndarray, b: np.ndarray, ref: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Re-assigns cluster labels based on the L2 distance of cluster centers to a reference vector."""
    uniq, first_idx, inv = np.unique(b, return_index=True, return_inverse=True)
    reps = a[first_idx]
    diff = np.abs(a - reps[inv])
    if np.max(diff) > tol:
        bad = np.argmax(np.max(diff, axis=1))
        raise ValueError(f"Row {bad} differs from its representative by {np.max(diff):.3g}")
    dists = np.linalg.norm(reps - ref, axis=1)
    order = np.argsort(dists)
    new_label = np.empty_like(order)
    new_label[order] = np.arange(len(order))
    return new_label[inv]

def find_min_row_by_2norm(x: np.ndarray) -> Tuple[int, np.ndarray]:
    """Finds the row with the minimum L2 norm in a matrix."""
    row_norms = np.linalg.norm(x, axis=1)
    min_index = np.argmin(row_norms)
    return min_index, x[min_index, :]

def diff_mat(w_new: np.ndarray) -> np.ndarray:
    """Computes a signed pairwise distance matrix based on L2 norm."""
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

def preprocess_clipp_data(
    r: np.ndarray, n: np.ndarray, minor: np.ndarray, total: np.ndarray,
    purity: Union[float, np.ndarray], coef_list: List, control_large: float,
    device: str, dtype: torch.dtype
) -> Dict[str, Any]:
    """
    Prepares and transforms input data for the ADMM optimization.

    This function handles data loading, validation, calculation of initial
    cellular prevalence estimates (phi_hat), and conversion to torch tensors
    on the specified device.

    The cellular prevalence `phi` is estimated from the variant allele fraction (VAF)
    using the formula:
    phi = VAF * [(ploidy - purity*ploidy) + purity*total_cn] / minor_cn

    This `phi` is then transformed to the logit scale `w` to initialize the
    optimization.

    Parameters are described in the main `clipp2` function.

    Returns
    -------
    dict
        A dictionary containing all necessary preprocessed data as torch tensors,
        along with original numpy arrays for post-processing.
    """
    def ensure_2D_and_no_zeros(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1: arr = arr.reshape(-1, 1)
        elif arr.ndim != 2: raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")
        return np.where(arr == 0, 1, arr)

    def to_torch_gpu(arr: np.ndarray, local_dtype: torch.dtype) -> torch.Tensor:
        arr = arr.astype(np.float32)
        if str(local_dtype) == 'float8':
            return torch.as_tensor(arr, dtype=torch.float8_e4m3fn, device=device)
        elif str(local_dtype) == 'float16':
            return torch.as_tensor(arr, dtype=torch.float16, device=device)
        else:
            return torch.as_tensor(arr, dtype=torch.float32, device=device)

    # Ensure inputs are 2D and move to torch tensors on the correct device
    r = r.reshape(-1, 1) if r.ndim == 1 else r
    n, minor, total = ensure_2D_and_no_zeros(n), ensure_2D_and_no_zeros(minor), ensure_2D_and_no_zeros(total)
    r_t, n_t = to_torch_gpu(r, dtype), to_torch_gpu(n, dtype)
    minor_t, total_t = to_torch_gpu(minor, dtype), to_torch_gpu(total, dtype)
    c_stack = [to_torch_gpu(c, dtype) for c in coef_list]
    c_all_t = torch.stack(c_stack, dim=1)
    No_mutation, M = r_t.shape

    # Handle purity (can be a scalar or an array per sample)
    if isinstance(purity, (float, int)):
        purity_t = torch.full((No_mutation, M), float(purity), device=device, dtype=r_t.dtype)
    else:
        purity_t = to_torch_gpu(purity, dtype).unsqueeze(0).expand(No_mutation, -1)

    # Calculate initial unpenalized cellular prevalence (phi_hat)
    ploidy_t = torch.full((No_mutation, M), 2.0, device=device, dtype=r_t.dtype)
    fraction_t = (r_t + 1e-12) / (n_t + 1e-10) # VAF
    phi_hat_t = fraction_t * ((ploidy_t - purity_t*ploidy_t) + (purity_t * (total_t + 1e-10))) / minor_t
    # Handle cases where read count is zero
    phi_hat_t = torch.where(r_t == 0, torch.tensor(1e-12, device=phi_hat_t.device, dtype=phi_hat_t.dtype), phi_hat_t)

    # Normalize and transform to logit scale for initialization
    scale_parameter = torch.clamp(torch.max(phi_hat_t), min=1.0)
    phi_new_t = phi_hat_t / scale_parameter
    w_init_t = torch.clamp(torch.log(phi_new_t / (1 - phi_new_t)), -control_large, control_large)

    return {
        'w_init_t': w_init_t, 'r_t': r_t, 'n_t': n_t, 'minor_t': minor_t,
        'total_t': total_t, 'purity_t': purity_t, 'c_all_t': c_all_t,
        'phi_hat_t': phi_hat_t, 'No_mutation': No_mutation, 'M': M,
        'original_n': n, 'original_r': r, 'original_total': total, 'original_minor': minor
    }

def postprocess_admm_results(
    admm_results: Dict[str, torch.Tensor],
    preprocessed_data: Dict[str, Any],
    purity: Union[float, np.ndarray],
    post_th: float,
    least_diff: float
) -> Dict[str, Any]:
    """
    Post-processes the raw ADMM output to generate final subclonal structures.

    This involves a series of heuristic steps:
    1. Initial clustering based on the penalized differences (`eta`).
    2. Iterative merging of small clusters into their nearest neighbors.
    3. Calculation of cluster centroid cellular prevalences (`phi`).
    4. Iterative merging of phenotypically similar clusters based on `least_diff`.
    5. Calculation of model fitness scores (Log-Likelihood, AIC, BIC).

    Parameters
    ----------
    admm_results : dict
        The dictionary returned by `run_admm_optimization`.
    preprocessed_data : dict
        The dictionary returned by `preprocess_clipp_data`.
    purity : float or ndarray
        The tumor purity value(s).
    post_th : float
        Threshold below which `eta` norms are considered zero for clustering.
    least_diff : float
        The minimum L2 distance between cluster centroids to remain separate.

    Returns
    -------
    dict
        A dictionary containing the final results:
        - 'phi': The (S x M) matrix of cellular prevalences for each SNV.
        - 'label': The cluster assignment for each SNV.
        - 'aic': The Akaike Information Criterion for the model.
        - 'bic': The Bayesian Information Criterion for the model.
    """
    # --- Step 1: Initial Clustering ---
    w_final_np = admm_results['w'].detach().cpu().numpy()
    eta_final_np = admm_results['eta'].detach().cpu().numpy()
    phi_hat = preprocessed_data['phi_hat_t'].detach().cpu().numpy()
    No_mutation, M = preprocessed_data['No_mutation'], preprocessed_data['M']
    n, r, total, minor = (
        preprocessed_data['original_n'], preprocessed_data['original_r'],
        preprocessed_data['original_total'], preprocessed_data['original_minor']
    )
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
                class_label[i] = class_label[j]
                group_size[class_label[j]] += 1
                assigned = True
                break
        if not assigned:
            class_label[i] = labl
            labl += 1
            group_size.append(1)

    # --- Step 2: Refine Clusters by Merging Small Groups ---
    least_mut = np.ceil(0.05 * No_mutation)
    gs_array = np.array(group_size)
    if np.any(gs_array > 0):
        tmp_size = np.min(gs_array[gs_array > 0])
        refine = tmp_size < least_mut
        while refine:
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

    # --- Step 3: Compute Cluster Centroids (Phi) ---
    labels = np.unique(class_label)
    phi_out = np.zeros((len(labels), M))
    for i, lbl in enumerate(labels):
        cluster_idx = (class_label == lbl)
        class_label[cluster_idx] = i # Re-index labels to be contiguous 0, 1, 2...
        nh, ph = n[cluster_idx, :], phi_hat[cluster_idx, :]
        phi_out[i, :] = np.sum(ph * nh, axis=0) / np.sum(nh, axis=0)

    # --- Step 4: Refine Clusters by Merging Proximal Centroids ---
    if len(labels) > 1:
        sort_phi = sort_by_2norm(phi_out)
        phi_diff = sort_phi[1:, :] - sort_phi[:-1, :]
        min_ind, min_val = find_min_row_by_2norm(phi_diff)
        while np.linalg.norm(min_val) < least_diff:
            # Find original labels corresponding to the two closest clusters
            orig_label_A = np.where(np.all(phi_out == sort_phi[min_ind], axis=1))[0][0]
            orig_label_B = np.where(np.all(phi_out == sort_phi[min_ind + 1], axis=1))[0][0]
            # Merge cluster A into B
            class_label[class_label == orig_label_A] = orig_label_B
            # Recalculate centroids
            labels = np.unique(class_label)
            phi_out = np.zeros((len(labels), M))
            for i, lbl in enumerate(labels):
                idx = (class_label == lbl)
                class_label[idx] = i
                nh, ph = n[idx, :], phi_hat[idx, :]
                phi_out[i, :] = np.sum(ph * nh, axis=0) / np.sum(nh, axis=0)
            if len(labels) == 1: break
            # Find next closest pair
            sort_phi = sort_by_2norm(phi_out)
            phi_diff = sort_phi[1:, :] - sort_phi[:-1, :]
            min_ind, min_val = find_min_row_by_2norm(phi_diff)

    # --- Step 5: Finalize Outputs and Compute Model Fitness ---
    phi_res = np.zeros((No_mutation, M))
    for lab_idx in range(phi_out.shape[0]):
        phi_res[class_label == lab_idx, :] = phi_out[lab_idx, :]
    purity_arr = np.array([purity]) if isinstance(purity, (int, float)) else purity
    # Reassign final labels for consistent ordering (e.g., 0 for smallest norm cluster)
    final_labels = reassign_labels_by_distance(phi_res, class_label, purity_arr)

    # Calculate Log-Likelihood, AIC, and BIC
    phi_clip = np.clip(phi_res, 1e-15, 1 - 1e-15)
    denominator = 2.0 * (1.0 - purity_arr[None, :]) + purity_arr[None, :] * total
    # Calculate expected VAF (theta) from the final phi
    pp_matrix = phi_clip * minor / denominator
    pp_matrix = pp_matrix.clip(1e-15, 1 - 1e-15)
    logL_matrix = r * np.log(pp_matrix) + (n - r) * np.log(1 - pp_matrix)
    logL = np.sum(logL_matrix)

    N, K_clusters = No_mutation * M, len(np.unique(final_labels))
    k_params = K_clusters * M # Number of estimated parameters
    AIC = -2.0 * logL + 2.0 * k_params
    BIC = -2.0 * logL + k_params * np.log(N)

    return {'phi': phi_res, 'label': final_labels, 'aic': AIC, 'bic': BIC}


# =============================================================================
# III. MAIN PIPELINE ORCHESTRATOR
# =============================================================================

def clipp2(
    r: np.ndarray,
    n: np.ndarray,
    minor: np.ndarray,
    total: np.ndarray,
    purity: Union[float, np.ndarray],
    coef_list: List,
    wcut: List[float] = [-1.8, 1.8],
    alpha: float = 0.8,
    gamma: float = 3.7,
    rho: float = 1.02,
    precision: float = 0.01,
    Run_limit: int = 200,
    control_large: float = 5,
    lambda_seq: Union[List[float], float] = [0.1, 0.05, 0.01],
    post_th: float = 0.001,
    least_diff: float = 0.01,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    dtype: torch.dtype = torch.float32,
) -> List[Dict[str, Any]]:
    """
    Main entry point for the CliPP2 subclone reconstruction pipeline.

    This function orchestrates the entire process, from data preprocessing to
    running the ADMM optimization over a sequence of regularization parameters
    (lambda) and post-processing the results for each.

    Parameters
    ----------
    r : np.ndarray
        Matrix (S x M) of variant read counts for S SNVs in M samples.
    n : np.ndarray
        Matrix (S x M) of total read counts.
    minor : np.ndarray
        Matrix (S x M) of minor allele copy numbers.
    total : np.ndarray
        Matrix (S x M) of total copy numbers.
    purity : float or np.ndarray
        Tumor purity for each sample. Can be a single float or an array of size M.
    coef_list : list
        List of pre-computed coefficients for the piecewise-linear approximation
        of the logistic function.
    wcut : list, optional
        Cutoff points for the piecewise-linear approximation. Defaults to [-1.8, 1.8].
    alpha : float, optional
        Initial ADMM penalty parameter. Defaults to 0.8.
    gamma : float, optional
        SCAD shape parameter. Defaults to 3.7.
    rho : float, optional
        Multiplicative factor for increasing `alpha` during ADMM. Defaults to 1.02.
    precision : float, optional
        Convergence tolerance for ADMM. Defaults to 0.01.
    Run_limit : int, optional
        Maximum number of ADMM iterations. Defaults to 200.
    control_large : float, optional
        Clamping value for logit-scale parameters. Defaults to 5.
    lambda_seq : list of floats or float, optional
        A sequence of lambda values to solve for. The solver iterates through
        the values, using the result of one run as a warm start for the next.
        Defaults to [0.1, 0.05, 0.01].
    post_th : float, optional
        Threshold for post-processing clustering. Defaults to 0.001.
    least_diff : float, optional
        Threshold for merging clusters based on centroid distance. Defaults to 0.01.
    device : str, optional
        Compute device ('cuda' or 'cpu'). Defaults to 'cuda' if available.
    dtype : torch.dtype, optional
        The torch data type to use for computation. Defaults to torch.float32.

    Returns
    -------
    list[dict]
        A list where each element is a dictionary containing the complete
        results for a single lambda value from the input sequence. Each dict
        includes 'phi', 'label', 'aic', 'bic', 'lambda', and 'wasserstein_distance'.
    """
    # --- Step 1: Preprocess data (done only once) ---
    preprocessed_data = preprocess_clipp_data(
        r, n, minor, total, purity, coef_list, control_large, device, dtype
    )
    # The unpenalized estimate is used as a baseline for distance calculation
    phi_hat_unpenalized = preprocessed_data['phi_hat_t'].detach().cpu().numpy().flatten()

    # Ensure lambda_seq is a list for iteration
    if not isinstance(lambda_seq, (list, np.ndarray)):
         lambda_seq = [lambda_seq]

    all_results = []
    warm_start_vars = None # Initialize warm start to None for the first run

    # --- Main Loop: Iterate over the sequence of Lambda values ---
    for i, current_lambda in enumerate(lambda_seq):
        # --- Step 2: Run ADMM optimization for the current Lambda ---
        # The output of the previous run is used as a warm start for the current run.
        admm_results_for_lambda = run_admm_optimization(
            preprocessed_data, wcut, alpha, gamma, rho, precision,
            Run_limit, control_large, current_lambda, device,
            warm_start_vars=warm_start_vars
        )

        # Use the results of this run as the warm start for the next lambda
        warm_start_vars = admm_results_for_lambda

        # --- Step 3: Post-process results for this Lambda ---
        results_for_current_lambda = postprocess_admm_results(
            admm_results_for_lambda, preprocessed_data, purity, post_th, least_diff
        )

        # --- Step 4: Calculate distance and finalize result dictionary ---
        phi_res_current = results_for_current_lambda['phi'].flatten()
        dist = wasserstein_distance(phi_hat_unpenalized, phi_res_current)

        results_for_current_lambda['lambda'] = current_lambda
        results_for_current_lambda['wasserstein_distance'] = dist

        all_results.append(results_for_current_lambda)

    return all_results