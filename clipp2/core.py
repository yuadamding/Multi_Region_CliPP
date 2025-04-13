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
    r      : np.ndarray, shape (No_mutation, M)
    n      : np.ndarray, shape (No_mutation, M)
    minor  : np.ndarray, shape (No_mutation, M)
    total  : np.ndarray, shape (No_mutation, M)
    purity : list of floats, length M
    ploidy : list of floats, length M
    coef_list : list of np.ndarray, length M, each shape (No_mutation, 6)
    wcut   : np.ndarray of shape (2,), e.g. [-0.18, 1.8]

    Example
    -------
    # Suppose we have run_clipp2_ADMM imported
    r, n, minor, total, purity, ploidy, coef_list, wcut = group_all_regions_for_ADMM("preprocess_result")
    w_new, eta_new, tau_new = run_clipp2_ADMM(
        r, n, minor, total,
        purity, ploidy,
        coef_list, wcut,
        alpha=1.0, rho=0.95, ...
    )

    Process
    -------
    1) subdirs => list of region directories
    2) For each region, read relevant txt files
    3) Flatten to 1D if needed, store in list
    4) Stack all => shape (No_mutation, M)
    5) return plus wcut
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
    r_list     = []
    n_list     = []
    minor_list = []
    total_list = []
    purity_list = []
    ploidy_list = []
    coef_list   = []

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

        # Read each file
        # (If your files use a different delimiter, adapt below)
        r_data     = np.genfromtxt(r_file, delimiter="\t")
        n_data     = np.genfromtxt(n_file, delimiter="\t")
        minor_data = np.genfromtxt(minor_file, delimiter="\t")
        total_data = np.genfromtxt(total_file, delimiter="\t")

        # We treat the single scalar in purity_ploidy.txt as 'purity'
        purity_val = np.genfromtxt(purity_file, delimiter="\t")
        if purity_val.ndim != 0:
            raise ValueError(f"purity_ploidy.txt in {region_name} must be a single scalar. Got shape {purity_val.shape}.")
        purity_val = float(purity_val)
        # Hardcode ploidy = 2.0, or adapt if you have a separate file/logic
        ploidy_val = 2.0

        # Read coef => shape (No_mutation, 6).  If your code uses 6 columns:
        coef_data  = np.genfromtxt(coef_file, delimiter="\t")
        if coef_data.ndim != 2:
            raise ValueError(f"coef.txt in {region_name} must be 2D. Found shape {coef_data.shape}.")

        # Ensure each r_data, n_data, etc. is 1D => shape (No_mutation,)
        # or 2D => (No_mutation,1). We'll flatten to 1D for stacking
        def flatten_1d(arr):
            if arr.ndim == 2 and arr.shape[1] == 1:
                return arr[:,0]
            elif arr.ndim == 1:
                return arr
            else:
                raise ValueError(f"Expected shape (No_mutation,) or (No_mutation,1). Got {arr.shape}")

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
        ploidy_list.append(ploidy_val)
        coef_list.append(coef_data)

        print(f"Loaded region '{region_name}': r.shape={r_data.shape}, coef.shape={coef_data.shape}, purity={purity_val}, ploidy={ploidy_val}")

    # 3) Now we have M items in each list, each shape => (No_mutation,).  Let's check consistent No_mutation
    No_mutation = len(r_list[0])
    for arr in r_list[1:]:
        if len(arr) != No_mutation:
            raise ValueError("Inconsistent No_mutation across subdirs. All regions must have the same number of SNVs.")

    # 4) Convert each list of length M => a 2D array of shape (No_mutation, M)
    r_final     = np.column_stack(r_list)
    n_final     = np.column_stack(n_list)
    minor_final = np.column_stack(minor_list)
    total_final = np.column_stack(total_list)

    # 5) We define a single global wcut => [-0.18, 1.8]
    wcut = np.array([-0.18, 1.8], dtype=float)

    print("\n=== Summary of grouped data ===")
    print(f"Found M={len(subdirs)} regions. Final r shape= {r_final.shape}, n= {n_final.shape}")
    print(f"minor= {minor_final.shape}, total= {total_final.shape}")
    print(f"purity_list= {purity_list}, ploidy_list= {ploidy_list}")
    print(f"coef_list length= {len(coef_list)} (each is (No_mutation,6) typically)")
    print(f"wcut= {wcut}\n")

    return (r_final, n_final, 
            minor_final, total_final,
            purity_list, ploidy_list,
            coef_list,
            wcut)

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

def ensure_2D_column(arr):
    """
    Re-declared for consistency, ensuring shape (No_mutation, M).
    """
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    elif arr.ndim == 2:
        return arr
    else:
        raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")

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

def clipp2_all_in_one(
    r, n, minor, total,
    purity, ploidy,
    coef_list,
    wcut = [-1.8, 1.8],
    alpha=0.8,
    gamma=3.7,
    rho=1.02,
    precision= 0.01,
    Run_limit=1e4,
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

    # 1) Ensure shape (No_mutation, M) for input
    r = ensure_2D_column(r)
    n = ensure_2D_column(n)
    minor = ensure_2D_column(minor)
    total = ensure_2D_column(total)

    # 2) Basic shape, broadcast purity/ploidy
    No_mutation, M = r.shape
    ploidy_arr = np.broadcast_to(np.array(ploidy, dtype=float).reshape((1,M)), (No_mutation, M))
    purity_arr = np.broadcast_to(np.array(purity, dtype=float).reshape((1,M)), (No_mutation, M))

    # 3) Initialize w
    eps_small = 1e-5
    theta_hat = (r + eps_small)/(n + 2*eps_small)
    phi_hat = theta_hat * ((ploidy_arr - purity_arr*ploidy_arr) + (purity_arr*total)) / minor
    scale_parameter = max(1.0, np.max(phi_hat))
    phi_new = phi_hat / scale_parameter

    low_b = expit(-control_large)
    up_b  = expit(control_large)
    phi_new = np.clip(phi_new, low_b, up_b)
    w_init = logit(phi_new)
    w_init = np.clip(w_init, -control_large, control_large)
    w_new = w_init.copy()

    # 4) Build DELTA operator
    row_idx = []
    col_idx = []
    vals = []
    pair_indices = []
    No_pairs = No_mutation*(No_mutation-1)//2
    pair_count = 0
    for i in range(No_mutation-1):
        for j in range(i+1, No_mutation):
            pair_indices.append((i,j))
            for m_idx in range(M):
                row_i = pair_count*M + m_idx
                col_i = i*M + m_idx
                row_idx.append(row_i)
                col_idx.append(col_i)
                vals.append(1.0)
                col_j = j*M + m_idx
                row_idx.append(row_i)
                col_idx.append(col_j)
                vals.append(-1.0)
            pair_count += 1

    data_ = np.array(vals, dtype=float)
    rows_ = np.array(row_idx, dtype=int)
    cols_ = np.array(col_idx, dtype=int)
    total_rows = No_pairs*M
    total_cols = No_mutation*M
    DELTA = sp.coo_matrix((data_, (rows_, cols_)),
                          shape=(total_rows, total_cols)).tocsr()

    # 5) Initialize eta, tau
    eta_new = reshape_eta_to_2D(w_new, No_mutation, M)
    tau_new = np.ones((No_pairs, M), dtype=float)
    diff = diff_mat(w_new)
    residual = 1e6
    k = 0

    # 6) ADMM loop
    while True:
            
        if k > 10 and (k > Run_limit or residual < precision):
            break
        
        k += 1
        w_old  = w_new.copy()
        eta_old = eta_new.copy()
        tau_old = tau_new.copy()

        # --- IRLS (A)
        expW = np.exp(w_old)
        denom_ = (2.0 + expW*total)
        denom_[denom_ == 0] = 1e-12
        theta = (expW*minor)/denom_

        # Build A,B
        A_array = np.zeros((No_mutation, M), dtype=float)
        B_array = np.zeros((No_mutation, M), dtype=float)
        for m_idx in range(M):
            w_m = w_old[:, m_idx]
            c_m = coef_list[m_idx]  # shape (No_mutation, 6)
            low_cut, up_cut = wcut[0], wcut[1]
            partA = (
                ((w_m <= low_cut)*c_m[:,1]) +
                ((w_m >= up_cut)*c_m[:,5]) +
                (((w_m>low_cut)&(w_m<up_cut))*c_m[:,3])
                - (r[:,m_idx]/n[:,m_idx])
            )
            partB = (
                ((w_m <= low_cut)*c_m[:,0]) +
                ((w_m >= up_cut)*c_m[:,4]) +
                (((w_m>low_cut)&(w_m<up_cut))*c_m[:,2])
            )
            sqrt_n_m = np.sqrt(n[:,m_idx])
            denom_m  = np.sqrt(theta[:,m_idx]*(1-theta[:,m_idx]) + 1e-12)
            A_array[:,m_idx] = (sqrt_n_m*partA)/denom_m
            B_array[:,m_idx] = (sqrt_n_m*partB)/denom_m

        A_flat = A_array.ravel()
        B_flat = B_array.ravel()
        big_eta_tau = alpha*eta_old + tau_old
        big_eta_tau_flat = big_eta_tau.ravel()
        linear_1 = DELTA.transpose().dot(big_eta_tau_flat)
        linear_2 = B_flat*A_flat
        linear = linear_1 - linear_2

        # --- Solve system (B^T B + alpha Delta^T Delta) w = linear
        B_sq = B_flat**2
        Bmat = sp.diags(B_sq,0,shape=(No_mutation*M, No_mutation*M))
        H = Bmat + alpha*(DELTA.transpose().dot(DELTA))
        w_new_flat = spsolve(H.tocsr(), linear)
        w_new = w_new_flat.reshape((No_mutation, M))
        np.clip(w_new, -control_large, control_large, out=w_new)
        diff = diff_mat(w_new)
        
        # --- group-SCAD threshold on eta
        for idx_p, (i,j) in enumerate(pair_indices):
            delt_ij = (w_new[i,:] - w_new[j,:]) - (1.0/alpha)*tau_old[idx_p,:]
            norm_d = np.linalg.norm(delt_ij)
            lam_over_alpha = Lambda/alpha
            gamma_lam      = gamma*Lambda
            if norm_d <= lam_over_alpha:
                eta_new[idx_p,:] = 0.0
            elif norm_d <= (Lambda + lam_over_alpha):
                scale_ = max(0.0, 1 - lam_over_alpha/norm_d)
                eta_new[idx_p,:] = scale_*delt_ij
            elif norm_d <= gamma_lam:
                T_mid = gamma_lam/((gamma-1)*alpha)
                denom2= 1.0 - 1.0/((gamma-1)*alpha)
                if denom2<=0:
                    eta_new[idx_p,:] = delt_ij
                else:
                    scale_factor_=1.0/denom2
                    scale_=max(0.0, 1 - T_mid/norm_d)
                    st_vec_=scale_*delt_ij
                    eta_new[idx_p,:]=scale_factor_*st_vec_
            else:
                eta_new[idx_p,:] = delt_ij

        # --- update tau
        for idx_p, (i,j) in enumerate(pair_indices):
            diff_ij = (w_new[i,:]-w_new[j,:]) - eta_new[idx_p,:]
            tau_new[idx_p,:] = tau_old[idx_p,:] - alpha*diff_ij

        alpha *= rho

        # --- residual
        max_diff = 0.0
        for idx_p, (i,j) in enumerate(pair_indices):
            tmp_ = (w_new[i,:] - w_new[j,:]) - eta_new[idx_p,:]
            local_ = np.max(np.abs(tmp_))
            if local_>max_diff:
                max_diff = local_
        residual = max_diff

        print(f"\rIteration {k}, residual={residual:0.5g}, alpha={alpha:0.5g}", end="")

    print("\nADMM finished.\n")
    
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
