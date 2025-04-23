#!/usr/bin/env python3

"""
This script provides a research-oriented pipeline for a SCAD-penalized ADMM approach 
to multi-region subclone reconstruction in single-sample or multi-sample (M>1) scenarios. 

Main steps:
1) Load data files from multiple "regions" (directories), each representing 
   a sample/region for ADMM.  (See `group_all_regions_for_ADMM`.)
2) Initialize logistic-scale parameters `w_new`.
3) Build difference operators and run an ADMM loop with SCAD-based thresholding.
4) Merge clusters in a final post-processing step if they are too close.

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

def scad_threshold_update_torch(
    w_new_t,        # shape (No_mutation, M)
    tau_old_t,      # shape (No_pairs, M)
    Delta_coo,      # shape ((No_pairs*M), (No_mutation*M)), sparse
    alpha, Lambda, gamma
):
    """
    GPU-based SCAD threshold update using PyTorch, fully vectorized.
    No Python for-loops. 
    """
    # ------------------------------------------------------------------
    # 1) Compute all pairwise differences D_w = w_i - w_j
    #    Because Delta_coo * w_new_flat = [ (w[i]-w[j])_m ] stacked
    # ------------------------------------------------------------------
    w_new_flat = w_new_t.reshape(-1)  # shape => (No_mutation*M,)
    # shape => ((No_pairs*M),)
    D_w_flat = torch.sparse.mm(Delta_coo, w_new_flat.unsqueeze(1)).squeeze(1)
    # Reshape to (No_pairs, M)
    # We know the first dimension is No_pairs*M
    # So we can do a simple view if we track them carefully:
    # D_w_flat[ k*M : k*M+M ] are pair k, across M
    # => .view(No_pairs, M)
    # But we must ensure correct ordering from the way we built Delta_coo.
    D_w = D_w_flat.view(tau_old_t.shape[0], tau_old_t.shape[1])  # shape => (No_pairs, M)

    # 2) delt_ij = D_w - (1/alpha)*tau_old_t
    delt_t = D_w - (1.0/alpha)*tau_old_t

    # 3) Norm of each row
    delt_norm_t = torch.norm(delt_t, dim=1)  # shape (No_pairs,)

    lam_over_alpha = Lambda / alpha
    gamma_lam      = gamma  * Lambda

    # 4) Piecewise region masks
    mask1 = (delt_norm_t <= lam_over_alpha)
    mask2 = (delt_norm_t > lam_over_alpha) & (delt_norm_t <= (Lambda + lam_over_alpha))
    mask3 = (delt_norm_t > (Lambda + lam_over_alpha)) & (delt_norm_t <= gamma_lam)
    mask4 = (delt_norm_t > gamma_lam)

    # 5) Create eta_new in one shot
    eta_new_t = torch.zeros_like(delt_t)

    # region 1 => zero
    eta_new_t[mask1] = 0.0

    # region 2 => group soft threshold
    i2 = mask2.nonzero(as_tuple=True)[0]  # shape (#region2,)
    scale2 = 1.0 - (lam_over_alpha / delt_norm_t[i2])
    scale2 = torch.clamp(scale2, min=0.0)
    eta_new_t[i2] = scale2.unsqueeze(1) * delt_t[i2]

    # region 3 => SCAD mid region
    i3 = mask3.nonzero(as_tuple=True)[0]  # shape (#region3,)
    T_mid  = gamma_lam / ((gamma - 1)*alpha)
    denom2 = 1.0 - 1.0/((gamma - 1)*alpha)
    if denom2 <= 0:
        # fallback => no shrink
        eta_new_t[i3] = delt_t[i3]
    else:
        scale_factor_ = 1.0 / denom2
        scale3 = 1.0 - (T_mid / delt_norm_t[i3])
        scale3 = torch.clamp(scale3, min=0.0)
        st_vec = scale3.unsqueeze(1) * delt_t[i3]
        eta_new_t[i3] = scale_factor_ * st_vec

    # region 4 => no shrink
    i4 = mask4.nonzero(as_tuple=True)[0]
    eta_new_t[i4] = delt_t[i4]

    # 6) Update tau
    diff_2D_t = D_w - eta_new_t
    tau_new_t = tau_old_t - alpha*diff_2D_t

    return eta_new_t, tau_new_t

def sort_by_2norm(x):
    """
    Sort the rows of array x by their Euclidean (L2) norm.

    Parameters
    ----------
    x : np.ndarray
        Shape (N, M).  Each row x[i,:] is a vector in R^M.

    Returns
    -------
    x_sorted : np.ndarray, shape (N, M)
        The same rows as x, but ordered by ascending L2 norm.

    Notes
    -----
    1) We compute the L2-norm of each row => row_norms[i] = ||x[i,:]||_2.
    2) We get an argsort over these norms and reorder the rows accordingly.
    3) We return only x_sorted.  If you also need the sorted norms, you can extend the code.
    """

    # 1) Compute the L2-norm of each row
    row_norms = np.linalg.norm(x, axis=1)  # shape (N,)

    # 2) Obtain an ordering that sorts by row_norms
    sort_idx = np.argsort(row_norms)       # shape (N,)

    # 3) Reorder x by that index
    x_sorted = x[sort_idx, :]
    return x_sorted

import numpy as np

def reassign_labels_by_distance(a: np.ndarray,
                                b: np.ndarray,
                                ref: np.ndarray,
                                tol: float = 1e-8) -> np.ndarray:
    """
    Parameters
    ----------
    a   : np.ndarray, shape (n, m)
          Data matrix.
    b   : np.ndarray, shape (n,)
          Original integer labels.
    ref : np.ndarray, shape (m,)
          Reference vector to compute distances against.
    tol : float
          Tolerance for checking identical rows.
          
    Returns
    -------
    new_b : np.ndarray, shape (n,)
            Integer labels 0..(k-1) in order of increasing
            distance from cluster‐rep to `ref`.
    """
    # 1) find unique labels and map each row to its cluster‐index
    uniq, first_idx, inv = np.unique(b,
                                     return_index=True,
                                     return_inverse=True)
    # representative row per cluster
    reps = a[first_idx]           # shape (k, m)
    
    # 2) verify identical within each cluster
    #    broadcast reps[inv] back to shape (n, m)
    diff = np.abs(a - reps[inv])
    if np.max(diff) > tol:
        bad = np.argmax(np.max(diff, axis=1))
        raise ValueError(f"Row {bad} differs from its rep by {np.max(diff):.3g}")
    
    # 3) compute distances of each rep to ref
    #    shape (k,)
    dists = np.linalg.norm(reps - ref, axis=1)
    
    # 4) sort clusters by increasing distance
    order = np.argsort(dists)    # indices into uniq/reps
    
    # 5) build mapping cluster‐idx → new label
    new_label = np.empty_like(order)
    new_label[order] = np.arange(len(order))
    
    # 6) relabel all rows
    return new_label[inv]


def find_min_row_by_2norm(x):
    """
    Find the single row in 'x' that has the minimal L2 norm.

    Parameters
    ----------
    x : np.ndarray, shape (N, M)
        Each row x[i,:] is a vector in R^M.

    Returns
    -------
    min_index : int
        The index of the row with the smallest L2 norm.
    min_row : np.ndarray of shape (M,)
        The row itself, x[min_index,:].

    Notes
    -----
    1) We compute row_norms[i] = || x[i,:] ||_2 
    2) argmin => minimal row index => min_index
    3) Return that row and its index
    """

    # 1) Compute each row's L2-norm (Euclidean norm)
    row_norms = np.linalg.norm(x, axis=1)  # shape (N,)

    # 2) Find the index of the minimal norm
    min_index = np.argmin(row_norms)       # an integer

    # 3) Extract the corresponding row
    min_row = x[min_index, :]

    return min_index, min_row

def group_all_regions_for_ADMM(root_dir):
    """
    Search all subdirectories under root_dir. Each subdirectory is one region (one sample).
    Load r.txt, n.txt, minor.txt, total.txt, purity_ploidy.txt, coef.txt,
    and combine them into the multi-sample arrays for run_clipp2_ADMM.

    We assume:
      - Each file is shape (No_mutation,) or (No_mutation,1).
      - purity_ploidy.txt has a single scalar => the region's purity. We'll fix ploidy=2.0.
      - coef.txt is shape (No_mutation, 6) for each region, stored in a list of length M.
      - All regions have the same No_mutation => we can stack horizontally.
      - We define wcut = np.array([-0.18, 1.8]) globally.

    Returns
    -------
    r_final     : np.ndarray, shape (No_mutation_filtered, M)
    n_final     : np.ndarray, shape (No_mutation_filtered, M)
    minor_final : np.ndarray, shape (No_mutation_filtered, M)
    total_final : np.ndarray, shape (No_mutation_filtered, M)
    purity_list : list of floats, length M
    coef_list   : list of np.ndarray, length M, each shape (No_mutation_filtered, 6)
    wcut        : np.ndarray of shape (2,)
    drop_rows   : np.ndarray of int, the row indices that were dropped
    """

    # 1) Find all subdirectories => each subdir is one region => M subdirs total.
    subdirs = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]
    subdirs.sort()  # for stable ordering

    if not subdirs:
        raise ValueError(f"No subdirectories found in {root_dir}. Each region must be in its own subdir.")

    # Prepare lists to accumulate region data
    r_list      = []
    n_list      = []
    minor_list  = []
    total_list  = []
    purity_list = []
    coef_list   = []  # length M, each is (No_mutation, 6)

    # 2) Loop over each region subdir
    for region_name in subdirs:
        region_path = os.path.join(root_dir, region_name)
        if not os.path.isdir(region_path):
            continue  # skip non-directory

        # Construct file paths
        r_file     = os.path.join(region_path, "r.txt")
        n_file     = os.path.join(region_path, "n.txt")
        minor_file = os.path.join(region_path, "minor.txt")
        total_file = os.path.join(region_path, "total.txt")
        purity_file= os.path.join(region_path, "purity_ploidy.txt")
        coef_file  = os.path.join(region_path, "coef.txt")

        # Read each file (adjust delimiter if needed)
        r_data     = np.genfromtxt(r_file,     delimiter="\t")
        n_data     = np.genfromtxt(n_file,     delimiter="\t")
        minor_data = np.genfromtxt(minor_file, delimiter="\t")
        total_data = np.genfromtxt(total_file, delimiter="\t")

        # Single scalar in purity_ploidy.txt => the region's purity
        purity_val = np.genfromtxt(purity_file, delimiter="\t")
        if purity_val.ndim != 0:
            raise ValueError(
                f"purity_ploidy.txt in {region_name} must be a single scalar. "
                f"Got shape {purity_val.shape}."
            )
        purity_val = float(purity_val)

        # Read coef => shape (No_mutation, 6)
        coef_data  = np.genfromtxt(coef_file, delimiter="\t")
        if coef_data.ndim != 2 or coef_data.shape[1] != 6:
            raise ValueError(
                f"coef.txt in {region_name} must be 2D with 6 columns. Got shape {coef_data.shape}."
            )

        # Flatten any (No_mutation,1) arrays to 1D
        def flatten_1d(arr):
            if arr.ndim == 2 and arr.shape[1] == 1:
                return arr[:,0]
            elif arr.ndim == 1:
                return arr
            else:
                raise ValueError(
                    f"Expected shape (No_mutation,) or (No_mutation,1). Got {arr.shape}"
                )

        r_data     = flatten_1d(r_data)
        n_data     = flatten_1d(n_data)
        minor_data = flatten_1d(minor_data)
        total_data = flatten_1d(total_data)

        # Append to lists
        r_list.append(r_data)
        n_list.append(n_data)
        minor_list.append(minor_data)
        total_list.append(total_data)
        purity_list.append(purity_val)
        coef_list.append(coef_data)

        print(f"Loaded region '{region_name}': "
              f"r.shape={r_data.shape}, "
              f"coef.shape={coef_data.shape}, purity={purity_val}")

    # 3) Check consistency of No_mutation
    M = len(subdirs)
    No_mutation = len(r_list[0])
    for arr in r_list[1:]:
        if len(arr) != No_mutation:
            raise ValueError("Inconsistent No_mutation across subdirs. "
                             "All regions must have the same number of SNVs.")

    # 4) Convert each list => 2D arrays (No_mutation, M)
    r_final     = np.column_stack(r_list)       # shape (No_mutation, M)
    n_final     = np.column_stack(n_list)
    minor_final = np.column_stack(minor_list)
    total_final = np.column_stack(total_list)

    # 5) A single global wcut
    wcut = np.array([-0.18, 1.8], dtype=float)

    print("\n=== Summary of grouped data before dropping rows ===")
    print(f"Found M={M} regions. r shape= {r_final.shape}, n= {n_final.shape}")
    print(f"minor= {minor_final.shape}, total= {total_final.shape}")
    print(f"coef_list length= {len(coef_list)} (each is (No_mutation,6))")
    print(f"wcut= {wcut}")

    # ----------------------------------------------------------
    # (A) Stack coef_list into a single 3D array: c_all shape = (No_mutation, M, 6)
    #     c_all[i,m,:] is the 6 coefficients for SNV i in region m.
    c_all = np.stack(coef_list, axis=1)  # shape (No_mutation, M, 6)

    # (B) Identify which rows are all zeros in each array:
    zero_r      = np.all(r_final     == 0, axis=1)  # shape = (No_mutation,)
    zero_n      = np.all(n_final     == 0, axis=1)
    zero_minor  = np.all(minor_final == 0, axis=1)
    zero_total  = np.all(total_final == 0, axis=1)

    # For coef: row i is "all 0" if c_all[i,:,:] is 0 for every region & col
    zero_coef   = np.all(c_all == 0, axis=(1,2))     # shape = (No_mutation,)

    # (C) Combine the masks => these rows should be dropped
    drop_mask = (zero_r | zero_n | zero_minor | zero_total | zero_coef)
    drop_rows = np.where(drop_mask)[0]  # actual indices

    # (D) Build a "keep" mask
    keep_mask = ~drop_mask

    # (E) Filter out those rows from each final array
    r_final     = r_final[    keep_mask, :]
    n_final     = n_final[    keep_mask, :]
    minor_final = minor_final[keep_mask, :]
    total_final = total_final[keep_mask, :]

    # Also remove them from c_all along axis=0 => shape becomes (No_mutation_kept, M, 6)
    c_all       = c_all[keep_mask, :, :]

    # (F) Re-split c_all into a new list of shape (M) each => (No_mutation_kept, 6)
    new_coef_list = []
    for m_idx in range(M):
        # c_all[:, m_idx, :] shape => (No_mutation_kept, 6)
        new_coef_list.append(c_all[:, m_idx, :])

    coef_list = new_coef_list

    print(f"\nDropped {len(drop_rows)} rows that were all-zero in r/n/minor/total/coef.")
    if len(drop_rows) > 0:
        print(f"Indices of dropped rows: {drop_rows}")

    print("\n=== Summary of grouped data after dropping rows ===")
    print(f"r shape= {r_final.shape}, n= {n_final.shape}")
    print(f"minor= {minor_final.shape}, total= {total_final.shape}")
    print(f"coef_list length= {len(coef_list)}, each => shape ({r_final.shape[0]}, 6)")

    return (
        r_final,
        n_final,
        minor_final,
        total_final,
        purity_list,
        coef_list,
        wcut,
        drop_rows  
    )

def soft_threshold_group(vec, threshold):
    """
    Group soft-thresholding operator for a vector in R^M.
    This shrinks vec toward 0 by threshold in L2 norm,
    or sets it to 0 if ||vec|| < threshold.
    """
    norm_vec = np.linalg.norm(vec)
    if norm_vec == 0:
        return np.zeros_like(vec)
    scale = max(0.0, 1 - threshold / norm_vec)
    return scale * vec

def SCAD_group_threshold(delta, lam, alpha, gamma):
    """
    Apply the group-SCAD threshold rule to the vector delta in R^M.

    Parameters
    ----------
    delta : np.ndarray of shape (M,)
        The difference vector to be thresholded in L2-norm.
    lam   : float
        The SCAD lambda.
    alpha : float
        The ADMM penalty parameter.
    gamma : float
        The SCAD gamma (> 2).

    Returns
    -------
    eta : np.ndarray of shape (M,) after group SCAD thresholding.

    Notes
    -----
    - D = np.linalg.norm(delta).
    - We compare D with lam/alpha, lam + lam/alpha, gamma*lam to decide piecewise region.
    - We apply group soft-threshold or full shrink / SCAD plateau logic.
    """
    D = np.linalg.norm(delta)
    lam_over_alpha = lam / alpha
    gamma_lam = gamma * lam

    # (1) If D <= lam/alpha => full shrink to 0
    if D <= lam_over_alpha:
        return np.zeros_like(delta)
    # (2) If lam/alpha < D <= lam + lam/alpha => standard group soft threshold
    elif D <= lam + lam_over_alpha:
        return soft_threshold_group(delta, lam_over_alpha)
    # (3) Middle SCAD region
    elif D <= gamma_lam:
        T_mid = gamma_lam / ((gamma - 1)*alpha)
        scale_factor = 1.0 / (1.0 - 1.0/((gamma - 1)*alpha))
        st_vec = soft_threshold_group(delta, T_mid)
        return scale_factor * st_vec
    else:
        # (4) If D > gamma_lam => no shrink
        return delta

def build_DELTA_multiM(No_mutation, M):
    """
    Construct the block-structured difference operator DELTA for M>1 case.
    DELTA will map w in R^{No_mutation * M} -> the stacked pairwise differences in R^{(#pairs) * M}.

    Returns
    -------
    DELTA : scipy.sparse.csr_matrix
        shape ((No_pairs*M), (No_mutation*M))
        #pairs = No_mutation*(No_mutation - 1)//2
    pair_indices : list of (i,j) pairs with i<j
    """
    row_idx = []
    col_idx = []
    vals    = []

    pair_indices = []
    No_pairs = No_mutation*(No_mutation-1)//2

    pair_count = 0
    for i in range(No_mutation-1):
        for j in range(i+1, No_mutation):
            pair_indices.append((i,j))
            for m in range(M):
                row_i = pair_count*M + m
                # +1 for w_i[m]
                col_i = i*M + m
                row_idx.append(row_i)
                col_idx.append(col_i)
                vals.append(1.0)
                # -1 for w_j[m]
                col_j = j*M + m
                row_idx.append(row_i)
                col_idx.append(col_j)
                vals.append(-1.0)
            pair_count += 1

    data = np.array(vals, dtype=float)
    rows = np.array(row_idx, dtype=int)
    cols = np.array(col_idx, dtype=int)

    total_rows = No_pairs * M
    total_cols = No_mutation * M
    DELTA = sp.coo_matrix((data, (rows, cols)), shape=(total_rows, total_cols))
    return DELTA.tocsr(), pair_indices

def initialize_w(r, n, ploidy, purity, total, minor, control_large=4.0):
    """
    Initialize w (logistic scale) from input read counts and copy number/purity.

    r, n : shape (No_mutation, M) => variant reads, total reads
    ploidy, purity, total, minor : can be broadcast or shape (No_mutation, M)
    control_large : clamp boundary for logistic transform

    Returns
    -------
    w_init : shape (No_mutation, M)

    Steps
    -----
    1) Basic fraction => (r + eps)/(n + 2*eps).
    2) Multiply by ( (ploidy - purity*ploidy + purity*total)/minor )
    3) Clip logistic scale in [-control_large, control_large].
    """

    eps_small = 1e-5
    theta_hat = (r + eps_small) / (n + 2*eps_small)  # avoid 0/1
    phi_hat = theta_hat * ((ploidy - purity*ploidy + purity*total)/minor)

    scale_parameter = max(1.0, np.max(phi_hat))
    phi_new = phi_hat / scale_parameter

    lower_bound = expit(-control_large)
    upper_bound = expit(control_large)
    phi_new_clamped = np.clip(phi_new, lower_bound, upper_bound)

    w_init = logit(phi_new_clamped)
    w_init_clamped = np.clip(w_init, -control_large, control_large)
    return w_init_clamped

def soft_threshold_group(vec, threshold):
    """
    Group soft-thresholding operator for a vector in R^M.
    This shrinks vec toward 0 by threshold in L2 norm, 
    or sets it to 0 if ||vec|| < threshold.
    """
    norm_vec = np.linalg.norm(vec)
    if norm_vec == 0:
        return vec * 0.0
    scale = max(0.0, 1 - threshold / norm_vec)
    return scale * vec

def SCAD_group_threshold(delta, lam, alpha, gamma):
    """
    Apply the group-SCAD threshold rule to the vector delta in R^M.
    """
    D = np.linalg.norm(delta)
    lam_over_alpha = lam / alpha
    gamma_lam      = gamma * lam
    if D <= lam_over_alpha:
        return np.zeros_like(delta)
    elif D <= lam + lam_over_alpha:
        return soft_threshold_group(delta, lam_over_alpha)
    elif D <= gamma_lam:
        T_mid = (gamma_lam / ((gamma - 1)*alpha))
        scale_factor = 1.0 / (1.0 - 1.0 / ((gamma - 1)*alpha))
        st_vec = soft_threshold_group(delta, T_mid)
        return scale_factor * st_vec
    else:
        return delta

def build_DELTA_multiM(No_mutation, M):
    """
    Re-declared for consistency 
    (Note: function was declared above, repeated here presumably by mistake in your code).
    """
    row_idx = []
    col_idx = []
    vals    = []
    pair_indices = []
    No_pairs = No_mutation*(No_mutation-1)//2
    pair_count = 0
    for i in range(No_mutation-1):
        for j in range(i+1, No_mutation):
            pair_indices.append((i,j))
            for m in range(M):
                row_i = pair_count*M + m
                col_i = i*M + m
                row_idx.append(row_i)
                col_idx.append(col_i)
                vals.append(+1.0)
                col_j = j*M + m
                row_idx.append(row_i)
                col_idx.append(col_j)
                vals.append(-1.0)
            pair_count += 1

    data = np.array(vals)
    rows = np.array(row_idx)
    cols = np.array(col_idx)
    total_rows = No_pairs * M
    total_cols = No_mutation * M
    DELTA = sp.coo_matrix((data, (rows, cols)),
                          shape=(total_rows, total_cols))
    return DELTA.tocsr(), pair_indices

def initialize_w(r, n, ploidy, purity, total, minor, No_mutation, M, control_large):
    """
    Re-declared for consistency 
    (Note: function was also declared above, repeated here presumably by mistake).
    """
    theta_hat = r / n
    phi_hat = theta_hat * ((ploidy - purity*ploidy + purity*total)/minor)
    scale_parameter = max(1, np.max(phi_hat))
    phi_new = phi_hat / scale_parameter
    lower_bound = expit(-control_large)
    upper_bound = expit(control_large)
    phi_new = np.clip(phi_new, lower_bound, upper_bound)
    w_init = logit(phi_new)
    w_init = np.clip(w_init, -control_large, control_large)
    return w_init

def reshape_eta_to_2D(w_new, No_mutation, M):
    """
    (Reiterated from above code, repeated here presumably by mistake).
    """
    w_flat = w_new.ravel()
    NM = No_mutation*M
    if w_flat.size != NM:
        raise ValueError("Mismatch: w_flat.size != No_mutation*M")
    diff = np.subtract.outer(w_flat, w_flat)
    ids  = np.triu_indices(NM, k=1)
    eta_new_1d = diff[ids]
    pair2idx = {}
    for one_d_index, (p_val, q_val) in enumerate(zip(ids[0], ids[1])):
        pair2idx[(p_val, q_val)] = one_d_index
    No_pairs = No_mutation*(No_mutation-1)//2
    eta_2d = np.zeros((No_pairs, M), dtype=w_new.dtype)
    pair_idx = 0
    for i in range(No_mutation-1):
        for j in range(i+1, No_mutation):
            for m in range(M):
                p = i*M + m
                q = j*M + m
                if p < q:
                    idx_in_1d = pair2idx[(p, q)]
                    val = eta_new_1d[idx_in_1d]
                else:
                    idx_in_1d = pair2idx[(q, p)]
                    val = -eta_new_1d[idx_in_1d]
                eta_2d[pair_idx, m] = val
            pair_idx += 1
    return eta_2d

def diff_mat(w_new):
    """
    Build a 'signed' distance matrix for M>1 by combining L2 norm with sign from the first coordinate,
    ensuring antisymmetry.  Negative sign is used as in the snippet.
    """
    No_mutation, M = w_new.shape
    diff_vec = w_new[:, None, :] - w_new[None, :, :]
    mag = np.sqrt(np.sum(diff_vec**2, axis=2))  # shape (No_mutation, No_mutation)
    first_coord_diff = w_new[None, :, 0] - w_new[:, None, 0]
    sign_mat = np.sign(first_coord_diff)
    diff_signed = -sign_mat * mag
    for i in range(No_mutation):
        diff_signed[i, i] = 0.0
    i_idx, j_idx = np.triu_indices(No_mutation, k=1)
    diff_signed[j_idx, i_idx] = -diff_signed[i_idx, j_idx]
    return(diff_signed)


def matvec_H(x, B_sq, DTD, alpha):
    # x shape => (NM,)
    # 1) Multiply by diagonal B^2
    out = B_sq * x
    # 2) Multiply by alpha*(Delta^T Delta)
    #    DTD is a sparse matrix => shape (NM, NM)
    out += alpha * torch.sparse.mm(DTD, x.unsqueeze(-1)).squeeze(-1)
    return out


def clipp2(
    r, n, minor, total,
    purity,
    coef_list,
    wcut = [-1.8, 1.8],
    alpha=0.8,
    gamma=3.7,
    rho=1.02,
    precision= 0.01,
    Run_limit=20,
    control_large=5,
    Lambda=0.01,
    post_th=0.05,
    least_diff=0.01,
    device='cuda', 
    dtype = torch.float32,
):
    """
    Perform the ADMM + SCAD approach for multi-sample subclone reconstruction,
    plus final cluster assignment.

    Steps:
    1) Convert r,n,minor,total => shape (No_mutation,M). Broadcast purity,ploidy => same shape.
    2) Initialize w => logistic transform in [-control_large, control_large].
    3) Build difference operator (DELTA).
    4) Initialize eta, tau, diff => run ADMM:
       - IRLS expansions => build A,B => flatten => solve (B^T B + alpha Delta^T Delta) w = ...
       - group SCAD threshold => update eta,tau
       - residual check => repeat
    5) Post-processing:
       - if ||eta|| <= post_th => set to 0 => same cluster
       - refine small clusters => combine
       - compute phi_out => cluster means
       - combine clusters if 2norm difference < least_diff
    6) Return final assignment: 
       { 'phi': shape(No_mutation,M), 'label': shape(No_mutation,) }

    Returns
    -------
    results : dict with 
       'phi'   => shape (No_mutation, M)
       'label' => shape (No_mutation,)

    This merges single-sample (M=1) approach with an extension for M>1 
    using the L2 norm across coordinates + sign from first coordinate.
    """

    # -------- 1) Ensure shape (No_mutation, M) for all inputs --------
    def ensure_2D_and_no_zeros(arr):
        """
        - If arr is 1D, reshape to (No_mutation, 1).
        - If arr is 2D, keep shape.
        - Then replace any zeros with 1.
        """
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim != 2:
            raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")
        # Replace zeros with 1
        # This is vectorized, no Python loops:
        arr = np.where(arr == 0, 1, arr)
        return arr
    
    r     = ensure_2D_and_no_zeros(r)
    n     = ensure_2D_and_no_zeros(n)
    minor = ensure_2D_and_no_zeros(minor)
    total = ensure_2D_and_no_zeros(total)

    def to_torch_gpu(arr, dtype):
        arr = arr.astype(np.float32)
        if dtype== 'float8':
            return torch.as_tensor(arr, dtype=torch.float8, device=device)
        elif dtype== 'float16':
            return torch.as_tensor(arr, dtype=torch.float16, device=device)
        else:
            return torch.as_tensor(arr, dtype=torch.float32, device=device)
        
    r_t     = to_torch_gpu(r, dtype)
    n_t     = to_torch_gpu(n, dtype)
    minor_t = to_torch_gpu(minor, dtype)
    total_t = to_torch_gpu(total, dtype)

    # Build c_all => shape(No_mutation, M, 6)
    c_stack = []
    for c in coef_list:
        # Each c => shape(No_mutation, 6)
        c_stack.append(to_torch_gpu(c, dtype))
    c_all_t = torch.stack(c_stack, dim=1)  # shape => (No_mutation, M, 6)

    No_mutation, M = r_t.shape

    # 1) Prepare purity => shape (No_mutation, M)
    if isinstance(purity, (float, int)):
        purity_t = torch.full((No_mutation, M), float(purity), device=device)
    else:
        purity_vec = to_torch_gpu(purity, dtype)  # shape (M,)
        purity_t   = purity_vec.unsqueeze(0).expand(No_mutation, -1)

    ploidy_t = torch.full((No_mutation, M), 2.0, device=device)

    # 2) Initialize w_new from logistic bounding
    fraction_t = r_t / (n_t + 1e-12)
    phi_hat_t  = fraction_t * ((ploidy_t - purity_t*ploidy_t) + (purity_t*total_t)) / (minor_t + 1e-12)

    scale_parameter = torch.clamp(torch.max(phi_hat_t), min=1.0)
    phi_new_t = phi_hat_t / scale_parameter

    low_b = torch.sigmoid(torch.tensor(-control_large, device=device))
    up_b  = torch.sigmoid(torch.tensor( control_large, device=device))
    phi_new_t = torch.clamp(phi_new_t, low_b, up_b)

    w_init_t = torch.log(phi_new_t) - torch.log(1 - phi_new_t + 1e-12)
    w_init_t = torch.clamp(w_init_t, -control_large, control_large)
    w_new_t  = w_init_t.clone()

    # 3) Build the sparse DELTA operator (COO)
    #    For i<j pairs, each row in the big (No_pairs*M) row-block has +1 at (i,m) and -1 at (j,m).
    #    We'll do this *without* for-loops using advanced indexing.
    i_idx_np, j_idx_np = torch.triu_indices(No_mutation, No_mutation, offset=1)
    # Move them to 'device'
    i_idx_np = i_idx_np.to(device)
    j_idx_np = j_idx_np.to(device)

    No_pairs = i_idx_np.size(0)

    # Then proceed to build row/col indices for the sparse Delta operator:
    pair_range = torch.arange(No_pairs, device=device)
    m_range    = torch.arange(M, device=device)

    # row_idx for the +1 block => shape (No_pairs, M)
    row_plus_2D = pair_range.unsqueeze(1)*M + m_range  # broadcast
    # row_idx for the -1 block => same row => row_plus_2D
    row_combined = torch.cat([row_plus_2D, row_plus_2D], dim=1)  # shape => (No_pairs, 2*M)

    # col_idx for +1 => i_idx_np*M + m
    col_plus_2D = i_idx_np.unsqueeze(1)*M + m_range
    # col_idx for -1 => j_idx_np*M + m
    col_minus_2D = j_idx_np.unsqueeze(1)*M + m_range
    col_combined = torch.cat([col_plus_2D, col_minus_2D], dim=1)

    row_idx = row_combined.reshape(-1)
    col_idx = col_combined.reshape(-1)

    # data => +1 then -1
    plus_block  = torch.ones_like(col_plus_2D, dtype=torch.float32)
    minus_block = -torch.ones_like(col_minus_2D, dtype=torch.float32)
    vals_2D = torch.cat([plus_block, minus_block], dim=1)
    vals = vals_2D.reshape(-1)

    total_rows = No_pairs*M
    total_cols = No_mutation*M
    Delta_coo = torch.sparse_coo_tensor(
        indices=torch.stack([row_idx, col_idx], dim=0),
        values=vals,
        size=(total_rows, total_cols),
        device=device
    ).coalesce()

    # 4) Initialize eta, tau
    #    We can get the difference directly: D_w => shape(No_pairs, M) = w[i_idx]-w[j_idx]
    #    We'll re-use the code from scad_threshold_update for consistency.
    #    For initialization, just do:
    eta_new_t = (w_new_t[i_idx_np, :] - w_new_t[j_idx_np, :])
    tau_new_t = torch.ones_like(eta_new_t, device=device)

    # 5) ADMM iteration
    residual = 1e6
    k_iter   = 0

    low_cut, up_cut = wcut

    while k_iter < Run_limit and residual > precision:

        # Might also stop if we get nan
        # (we do a small check below)
        k_iter += 1

        w_old_t  = w_new_t.clone()
        eta_old_t = eta_new_t.clone()
        tau_old_t = tau_new_t.clone()

        # =========== (A) IRLS expansions (vectorized) ===========
        expW_t = torch.exp(w_old_t)
        denom_ = 2.0 + expW_t*total_t
        denom_ = torch.clamp(denom_, min=1e-12)
        theta_t = (expW_t * minor_t)/denom_

        maskLow_t = (w_old_t <= low_cut)
        maskUp_t  = (w_old_t >= up_cut)
        maskMid_t = ~(maskLow_t | maskUp_t)

        # c_all_t => shape(No_mutation, M, 6)
        partA_full_t = (
            maskLow_t * c_all_t[...,1]
          + maskUp_t  * c_all_t[...,5]
          + maskMid_t * c_all_t[...,3]
        ) - (r_t/(n_t+1e-12))

        partB_full_t = (
            maskLow_t * c_all_t[...,0]
          + maskUp_t  * c_all_t[...,4]
          + maskMid_t * c_all_t[...,2]
        )

        sqrt_n_t = torch.sqrt(n_t)
        denom2_t = torch.sqrt(theta_t*(1-theta_t) + 1e-12)

        A_array_t = (sqrt_n_t * partA_full_t)/denom2_t
        B_array_t = (sqrt_n_t * partB_full_t)/denom2_t

        A_flat_t = A_array_t.flatten()  # shape => (No_mutation*M,)
        B_flat_t = B_array_t.flatten()

        # ---------------- (B) Build system & solve with manual CG ----------------

        # 1) Get dimensions and build diag(B^2):
        NM = B_flat_t.shape[0]      # = (No_mutation * M)
        B_sq_t = B_flat_t**2        # shape => (NM,)

        # 2) Compute sparse Delta^T Delta => shape (NM, NM)
        Delta_t = Delta_coo.transpose(0, 1)  # shape => (NM, No_pairs*M)
        DTD = torch.sparse.mm(Delta_t, Delta_coo)  # shape => (NM, NM), still sparse

        # 3) Define a function that multiplies H = (diag(B^2) + alpha * (Delta^T Delta)) by x
        def matvec_H(x):
            # x shape => (NM,)
            # (A) multiply by diag(B_sq_t)
            out = B_sq_t * x
            # (B) add alpha * (DTD @ x)
            out += alpha * torch.sparse.mm(DTD, x.unsqueeze(-1)).squeeze(-1)
            return out

        # 4) Build the right-hand side: linear_t
        big_eta_tau_t = alpha*eta_old_t + tau_old_t   # shape => (No_pairs, M)
        big_eta_tau_flat_t = big_eta_tau_t.flatten()  # shape => (No_pairs*M,)

        RHS_1 = torch.sparse.mm(Delta_t, big_eta_tau_flat_t.unsqueeze(1)).squeeze(1)
        linear_t = RHS_1 - (B_flat_t * A_flat_t)      # shape => (NM,)

        # 5) Manual Conjugate Gradient solve
        #    x0: initial guess => use the old solution w_old_t.
        x = w_old_t.flatten().clone()
        r = linear_t - matvec_H(x)
        p = r.clone()
        rs_old = torch.dot(r, r)

        max_cg_iter = 500
        tol = 1e-6
        iter_cg = 0

        while True:
            # Ap = H * p
            Ap = matvec_H(p)
            denom_ = torch.dot(p, Ap) + 1e-12
            alpha_cg = rs_old / denom_
            
            x = x + alpha_cg * p
            r = r - alpha_cg * Ap
            rs_new = torch.dot(r, r)

            iter_cg += 1
            # Check stopping criteria
            if rs_new.sqrt() < tol or iter_cg >= max_cg_iter:
                break

            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

        # x now holds the solution for (diag(B^2) + alpha * DTD) * x = linear_t
        w_new_flat_t = x
        w_new_t = w_new_flat_t.view(No_mutation, M)

        # Finally, clamp in [-control_large, control_large]
        w_new_t = torch.clamp(w_new_t, -control_large, control_large)


        # Finally, clamp in [-control_large, control_large]
        w_new_t = torch.clamp(w_new_t, -control_large, control_large)

        # =========== (C) SCAD threshold ===========  
        eta_new_t, tau_new_t = scad_threshold_update_torch(
            w_new_t, tau_old_t, Delta_coo, alpha, Lambda, gamma
        )

        # scale alpha
        alpha *= rho

        # =========== (D) residual check ===========
        # residual = max_{i<j,m} | (w[i]-w[j]) - eta[k]| 
        # We'll do it again with the Delta approach
        # D_w = Delta_coo w_new_flat
        w_new_flat2 = w_new_t.flatten()
        D_w_flat2   = torch.sparse.mm(Delta_coo, w_new_flat2.unsqueeze(1)).squeeze(1)
        # reshape to (No_pairs, M)
        D_w2 = D_w_flat2.view(No_pairs, M)
        diff_2D_t = D_w2 - eta_new_t
        residual_val_t = torch.max(torch.abs(diff_2D_t))
        residual = float(residual_val_t.item())

        if torch.isnan(residual_val_t):
            break

        print(f"Iter={k_iter}, alpha={alpha:.4f}, residual={residual:.6g}")

    print("\nADMM finished.\n")
    
    w_new = w_new_t.detach().cpu().numpy()
    eta_new = eta_new_t.detach().cpu().numpy()
    phi_hat = phi_hat_t.detach().cpu().numpy()
    diff = diff_mat(w_new)
    ids = np.triu_indices(diff.shape[1], 1)
    eta_new[np.where(np.abs(eta_new) <= post_th)] = 0
    diff[ids] = np.linalg.norm(eta_new, axis=1)
    class_label = -np.ones(No_mutation)
    class_label[0] = 0
    group_size = [1]
    labl = 1

    for i in range(1, No_mutation):
        for j in range(i):
            if diff[j, i] == 0:
                class_label[i] = class_label[j]
                group_size[int(class_label[j])] += 1
                break
        if class_label[i] == -1:
            class_label[i] = labl
            labl += 1
            group_size.append(1)

    # quality control
    least_mut = np.ceil(0.05 * No_mutation)
    tmp_size = np.min(np.array(group_size)[np.array(group_size) > 0])
    tmp_grp = np.where(group_size == tmp_size)
    refine = False
    if tmp_size < least_mut:
        refine = True

    while refine:
        refine = False
        tmp_col = np.where(class_label == tmp_grp[0][0])[0]
        for i in range(len(tmp_col)):
            if tmp_col[i] != 0 and tmp_col[i] != No_mutation - 1:
                tmp_diff = np.abs(np.append(np.append(diff[0:tmp_col[i], tmp_col[i]].T.ravel(), 100),
                                            diff[tmp_col[i], (tmp_col[i] + 1):No_mutation].ravel()))
                tmp_diff[tmp_col] += 100
                diff[0:tmp_col[i], tmp_col[i]] = tmp_diff[0:tmp_col[i]]
                diff[tmp_col[i], (tmp_col[i] + 1):No_mutation] = tmp_diff[(tmp_col[i] + 1):No_mutation]
            elif tmp_col[i] == 0:
                tmp_diff = np.append(100, diff[0, 1:No_mutation])
                tmp_diff[tmp_col] += 100
                diff[0, 1:No_mutation] = tmp_diff[1:No_mutation]
            else:
                tmp_diff = np.append(diff[0:(No_mutation - 1), No_mutation - 1], 100)
                tmp_diff[tmp_col] += 100
                diff[0:(No_mutation - 1), No_mutation - 1] = tmp_diff[0:(No_mutation - 1)]
            ind = tmp_diff.argmin()
            group_size[class_label.astype(np.int64, copy=False)[tmp_col[i]]] -= 1
            class_label[tmp_col[i]] = class_label[ind]
            group_size[class_label.astype(np.int64, copy=False)[tmp_col[i]]] += 1
        tmp_size = np.min(np.array(group_size)[np.array(group_size) > 0])
        tmp_grp = np.where(group_size == tmp_size)
        refine = False
        if tmp_size < least_mut:
            refine = True

    labels = np.unique(class_label)
    phi_out = np.zeros((len(labels), M))
    for i in range(len(labels)):
        ind = np.where(class_label == labels[i])[0]
        class_label[ind] = i
        phi_out[i, :] = np.sum(phi_hat[ind, : ] * n[ind, : ], axis=0) / np.sum(n[ind, : ], axis=0)

    if len(labels) > 1:
        sort_phi = sort_by_2norm(phi_out)
        phi_diff = sort_phi[1:, :] - sort_phi[:-1, :]
        min_ind, min_val = find_min_row_by_2norm(phi_diff)
        while np.linalg.norm(min_val) < least_diff:
            combine_ind = np.where(phi_out == sort_phi[min_ind, ])[0]
            combine_to_ind = np.where(phi_out == sort_phi[min_ind + 1, ])[0]
            class_label[class_label == combine_ind] = combine_to_ind
            labels = np.unique(class_label)
            phi_out = np.zeros((len(labels), M))
            for i in range(len(labels)):
                ind = np.where(class_label == labels[i])[0]
                class_label[ind] = i
                phi_out[i] = np.sum(phi_hat[ind, : ] * n[ind, : ], axis=0) / np.sum(n[ind, : ], axis=0)
            if len(labels) == 1:
                break
            else:
                sort_phi = sort_by_2norm(phi_out)
                phi_diff = sort_phi[1:, :] - sort_phi[:-1, :]
                min_ind, min_val = find_min_row_by_2norm(phi_diff)
    phi_res = np.zeros((No_mutation, M))
    for lab in range(np.shape(phi_out)[0]):
        phi_res[class_label == lab, ] = phi_out[lab, ]

    class_label = reassign_labels_by_distance(phi_res, class_label, purity)
    
    return {'phi': phi_res, 'label': class_label}