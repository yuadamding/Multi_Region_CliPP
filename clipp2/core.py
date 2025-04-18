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

def scad_threshold_update(
    w_new,             # shape (N, M)
    tau_old,           # shape (No_pairs, M)
    DELTA,             # shape (No_pairs*M, N*M)
    alpha, Lambda, gamma
):
    """
    Vectorized update of eta, tau in the group-SCAD ADMM steps.

    Returns
    -------
    eta_new : (No_pairs, M)
    tau_new : (No_pairs, M)
    """

    # 1) All differences D_w = (w_i - w_j), shape (No_pairs, M)
    w_new_flat = w_new.ravel()                      # (N*M,)
    D_w_flat   = DELTA.dot(w_new_flat)              # (No_pairs*M,)
    D_w        = D_w_flat.reshape((-1, w_new.shape[1]))  # (No_pairs, M)

    # 2) Compute delt_ij = D_w - (1/alpha)*tau_old
    delt = D_w - (1.0/alpha)*tau_old

    # 3) Norm of each row
    delt_norm = np.linalg.norm(delt, axis=1)  # (No_pairs,)

    # 4) Piecewise group SCAD
    lam_over_alpha = Lambda / alpha
    gamma_lam      = gamma  * Lambda

    mask1 = (delt_norm <= lam_over_alpha)
    mask2 = (delt_norm > lam_over_alpha) & (delt_norm <= Lambda + lam_over_alpha)
    mask3 = (delt_norm > (Lambda + lam_over_alpha)) & (delt_norm <= gamma_lam)
    mask4 = (delt_norm >  gamma_lam)

    eta_new = np.zeros_like(delt)

    # region 1 => zero
    eta_new[mask1] = 0.0

    # region 2 => group soft threshold
    i2     = np.where(mask2)[0]
    scale2 = 1.0 - (lam_over_alpha / delt_norm[i2])
    scale2 = np.clip(scale2, 0.0, None)  # no negatives
    eta_new[i2] = scale2[:,None]*delt[i2]

    # region 3 => SCAD mid region
    i3 = np.where(mask3)[0]
    T_mid  = gamma_lam / ((gamma - 1)*alpha)
    denom2 = 1.0 - 1.0/((gamma - 1)*alpha)
    if denom2 <= 0:
        # fallback => no shrink
        eta_new[i3] = delt[i3]
    else:
        scale_factor_ = 1.0 / denom2
        scale3 = 1.0 - (T_mid / delt_norm[i3])
        scale3 = np.clip(scale3, 0.0, None)
        st_vec = scale3[:,None]*delt[i3]
        eta_new[i3] = scale_factor_*st_vec

    # region 4 => no shrink
    eta_new[mask4] = delt[mask4]

    # 5) Now update tau in bulk
    diff_2D = D_w - eta_new
    tau_new = tau_old - alpha*diff_2D

    return eta_new, tau_new



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
    least_diff=0.01
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

    No_mutation, M = r.shape

    # -------- 2) Broadcast purity => shape (No_mutation, M).  Fix ploidy=2.0. --------
    if np.isscalar(purity):
        purity_arr = np.full((No_mutation, M), float(purity))
    else:
        purity = np.asarray(purity, dtype=float)
        if purity.ndim == 1 and purity.shape[0] == M:
            purity_arr = np.broadcast_to(purity.reshape(1,M), (No_mutation, M))
        else:
            raise ValueError(f"purity must be scalar or shape (M,). Got shape {purity.shape}.")
    ploidy_arr = np.full((No_mutation, M), 2.0)

    # -------- 3) Compute w_new, using single-sample style bounding logic --------
    # fraction = (r + eps)/(n + 2eps), then scale by copy/purity factor, clip in [expit(-cl), expit(cl)]
    fraction = (r ) / (n )
    phi_hat  = fraction * ( (ploidy_arr - purity_arr*ploidy_arr) + (purity_arr*total) ) / minor

    scale_parameter = max(1.0, np.max(phi_hat))
    phi_new = phi_hat / scale_parameter

    low_b = expit(-control_large)
    up_b  = expit(control_large)
    phi_new = np.clip(phi_new, low_b, up_b)  # shape (No_mutation, M)

    w_init = logit(phi_new)
    w_init = np.clip(w_init, -control_large, control_large)
    w_new  = w_init.copy()

    # -------- 4) Build the DELTA operator with NO Python for-loops --------
    # We want pairs (i<j).  We'll use triu_indices:
    i_idx, j_idx = np.triu_indices(No_mutation, k=1)     # shape=(No_pairs,)
    No_pairs = i_idx.size                                # # of i<j pairs

    # For each pair k, we have M rows in the final big matrix.
    # row indices => (k*M + m) for m in [0..M-1].
    # We'll build them in bulk.
    row_block = np.arange(No_pairs)[:,None]*M + np.arange(M)   # shape=(No_pairs, M)
    # We want 2 columns per row block => one for +1 at col i, one for -1 at col j,
    # so the total rows = 2 * (No_pairs * M).  But the row index is the same for i and j.
    # => We'll horizontally stack row_block with itself, then flatten.
    row_block_2 = np.hstack((row_block, row_block))            # shape=(No_pairs, 2*M)
    row_idx = row_block_2.ravel()                              # shape=(2*M*No_pairs,)

    # For column indices => i_idx[k], j_idx[k] => i_idx[k]*M + m
    col_block_i = i_idx[:,None]*M + np.arange(M)   # shape=(No_pairs, M)
    col_block_j = j_idx[:,None]*M + np.arange(M)   # shape=(No_pairs, M)
    col_block_2 = np.hstack((col_block_i, col_block_j))  # shape=(No_pairs, 2*M)
    col_idx = col_block_2.ravel()

    # For the data => +1 for i, -1 for j
    plus_ones  = np.ones_like(col_block_i)
    minus_ones = -np.ones_like(col_block_j)
    data_block = np.hstack((plus_ones, minus_ones))    # shape=(No_pairs, 2*M)
    vals = data_block.ravel()                          # shape=(2*M*No_pairs,)

    total_rows = No_pairs * M
    total_cols = No_mutation * M
    DELTA = sp.coo_matrix((vals, (row_idx, col_idx)),
                          shape=(total_rows, total_cols)).tocsr()

    # -------- 5) Initialize eta, tau, residual, iteration k with NO loops --------
    # We can get the initial pairwise differences: w_new[i_idx,:] - w_new[j_idx,:]
    # shape => (No_pairs, M)
    eta_new = w_new[i_idx, :] - w_new[j_idx, :]

    # tau_new => just ones
    tau_new = np.ones_like(eta_new)

    # Large initial residual
    residual = 1e6
    k = 0

    c_all = np.stack(coef_list, axis=1)
    i_idx, j_idx = np.triu_indices(No_mutation, k=1)  
    No_pairs = i_idx.size
    # 6) ADMM loop
    while True:
            
        if k > 10 and (k > Run_limit or residual < precision):
            break
        elif k > 10 and np.isnan(residual):
            break
        
        k += 1
        w_old  = w_new.copy()
        eta_old = eta_new.copy()
        tau_old = tau_new.copy()

        # =========================
        # (A) IRLS expansions in bulk
        # =========================

        # 1) theta = (e^w * minor) / (2 + e^w * total)
        expW   = np.exp(w_old)
        denom_ = 2.0 + (expW*total)
        # avoid zero denominators
        denom_ = np.where(denom_==0, 1e-12, denom_)
        theta  = (expW*minor)/denom_    # shape => (No_mutation, M)

        # 2) Build partA, partB fully vectorized
        #    shape => (No_mutation, M)
        #    using c_all => shape(No_mutation, M, 6)
        low_cut, up_cut = wcut
        maskLow =  (w_old <= low_cut)
        maskUp  =  (w_old >= up_cut)
        maskMid = ~(maskLow | maskUp)

        # partA_full = (branch on c_all[...,1], c_all[...,3], c_all[...,5]) 
        #              minus (r/n)
        partA_full = (
            (maskLow * c_all[...,1])
          + (maskUp  * c_all[...,5])
          + (maskMid * c_all[...,3])
        ) - (r / n)

        # partB_full similarly from c_all[...,0], c_all[...,2], c_all[...,4]
        partB_full = (
            (maskLow * c_all[...,0])
          + (maskUp  * c_all[...,4])
          + (maskMid * c_all[...,2])
        )

        # 3) A_array, B_array => multiply by sqrt(n) / sqrt(theta*(1-theta))
        sqrt_n = np.sqrt(n)
        # avoid zero => add small epsilon
        denom2 = np.sqrt(theta*(1 - theta) + 1e-12)

        A_array = (sqrt_n * partA_full) / denom2
        B_array = (sqrt_n * partB_full) / denom2

        # 4) Flatten them for the linear system
        A_flat = A_array.ravel()    # shape => (No_mutation*M,)
        B_flat = B_array.ravel()

        # =========================
        # (B) Form the linear system => (B^T B + alpha Delta^T Delta) w = ...
        # =========================
        # big_eta_tau => alpha*eta_old + tau_old
        big_eta_tau = alpha*eta_old + tau_old
        big_eta_tau_flat = big_eta_tau.ravel()

        linear_1 = DELTA.transpose().dot(big_eta_tau_flat)
        # linear_2 = B_flat * A_flat => elementwise, shape => (No_mutation*M,)
        linear_2 = B_flat * A_flat
        # final 'linear'
        linear   = linear_1 - linear_2

        # B^T B => diag of B_flat^2
        B_sq = B_flat**2
        Bmat = sp.diags(B_sq, 0, shape=(No_mutation*M, No_mutation*M))
        H    = Bmat + alpha*(DELTA.transpose().dot(DELTA))

        # Solve
        w_new_flat = spsolve(H, linear)
        w_new = w_new_flat.reshape(No_mutation, M)
        # clip
        np.clip(w_new, -control_large, control_large, out=w_new)

        # =========================
        # (C) SCAD threshold => eta_new, tau_new (already vectorized)
        # =========================
        eta_new, tau_new = scad_threshold_update(
            w_new, tau_old, DELTA, alpha, Lambda, gamma
        )

        # scale alpha
        alpha *= rho

        # =========================
        # (D) residual check in bulk (no for-loops)
        # =========================
        #  Residual = max_{pair} max_{m} | (w_new[i,:]-w_new[j,:]) - eta_new[k,:] |
        #
        # i_idx, j_idx => shape (No_pairs,)
        # => compute w_new[i_idx,:] - w_new[j_idx,:], shape => (No_pairs, M)
        # subtract eta_new => shape => (No_pairs, M)
        diff_2D = (w_new[i_idx,:] - w_new[j_idx,:]) - eta_new
        # compute max absolute difference
        residual = np.max(np.abs(diff_2D))

        print(f"Iteration={k}, alpha={alpha:.4g}, residual={residual:.6g}")

    # End while

    print("\nADMM finished.\n")
    
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

    return {'phi': phi_res, 'label': class_label}
