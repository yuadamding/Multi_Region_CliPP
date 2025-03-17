import numpy as np
import pandas as pd

def preprocess_for_clipp2(snv_df, cn_df, purity, sample_id="unknown_sample",
                          valid_snvs_threshold=0, diff_cutoff=0.1):
    """
    Python version of the 'clipp1'-style preprocessing for multi-region CliPP2.
    Returns in-memory arrays/data for direct use by CliPP2.
    
    Parameters
    ----------
    snv_df : pd.DataFrame
        Must have columns: ["chromosome_index", "position", "ref_count", "alt_count"].
        One row per SNV (or per SNV-region if you are combining multiple regions).
    cn_df : pd.DataFrame
        Must have columns: [chr, start, end, minor_cn, major_cn, total_cn].
        Typically one row per CN segment.
    purity : float
        Sample-level tumor purity.
    sample_id : str
        Used for logging or error messages.
    valid_snvs_threshold : int
        Minimum required SNV count after filtering. Raises ValueError if below this.
    diff_cutoff : float
        Maximum piecewise-linear approximation error allowed for \(\theta\).

    Returns
    -------
    result_dict : dict
        {
            "snv_df_final": final filtered SNV DataFrame,
            "minor_read": 1D array of alt counts,
            "total_read": 1D array of alt+ref counts,
            "minor_count": 1D array of final minor counts,
            "total_count": 1D array of final total CN,
            "coef": 2D array of piecewise-linear coefficients (one row per SNV),
            "cutbeta": 2D array of the cut-points (one row per SNV),
            "excluded_SNVs": list of strings describing dropped SNVs,
            "purity": float
        }
    """
    ############################
    # Some small helper functions
    ############################
    def combine_reasons(chroms, positions, indices, reason):
        return [
            f"{chroms[i]}\t{positions[i]}\t{reason}"
            for i in indices
        ]
    
    def theta_func(w, bv, cv, cn, pur):
        """
        R-style: theta = (exp(w)*bv) / ((1+exp(w))*cn*(1-pur) + (1+exp(w))*cv*pur)
        """
        num = np.exp(w) * bv
        den = ((1 + np.exp(w)) * cn * (1 - pur) + (1 + np.exp(w)) * cv * pur)
        return num / den

    def linear_evaluate(x, a, b):
        return a*x + b

    def linear_approximate(bv, cv, cn, pur, diag_plot=False):
        """
        Piecewise-linear approximation of theta(w), enumerating breakpoints.
        """
        w_vals = np.arange(-5, 5.01, 0.1)
        actual = theta_func(w_vals, bv, cv, cn, pur)
        
        best_diff = float('inf')
        best_cuts = None
        best_coef = None

        for i in range(1, len(w_vals)-2):
            for j in range(i+1, len(w_vals)-1):
                w1, w2 = w_vals[i], w_vals[j]
                # 3 segments: [0..i], [i+1..j], [j+1..end]
                
                # seg1 slope/intercept
                b1 = (actual[i] - actual[0]) / (w1 - w_vals[0])
                a1 = actual[0] - w_vals[0]*b1
                # seg2 slope/intercept
                b2 = (actual[j] - actual[i]) / (w2 - w1)
                a2 = actual[i] - w1*b2
                # seg3 slope/intercept
                b3 = (actual[-1] - actual[j]) / (w_vals[-1] - w2)
                a3 = actual[-1] - w_vals[-1]*b3

                approx1 = linear_evaluate(w_vals[:i+1], b1, a1)
                approx2 = linear_evaluate(w_vals[i+1:j+1], b2, a2)
                approx3 = linear_evaluate(w_vals[j+1:], b3, a3)
                approx_val = np.concatenate((approx1, approx2, approx3))
                
                diff = np.max(np.abs(actual - approx_val))
                if diff < best_diff:
                    best_diff = diff
                    best_cuts = [w1, w2]
                    best_coef = [b1, a1, b2, a2, b3, a3]

        return {
            "w_cut": best_cuts,
            "diff": best_diff,
            "coef": best_coef
        }

    ############################
    # Start actual logic
    ############################
    dropped_reasons = []

    # 1) Filter out invalid chromosome
    mask_chr = snv_df["chromosome_index"].notna()
    drop_inds = np.where(~mask_chr)[0]
    if len(drop_inds) > 0:
        dropped_reasons += combine_reasons(
            snv_df["chromosome_index"].values,
            snv_df["position"].values,
            drop_inds,
            "Missing/invalid chromosome index"
        )
    snv_df = snv_df[mask_chr].reset_index(drop=True)
    if len(snv_df) < valid_snvs_threshold:
        raise ValueError(f"{sample_id}: only {len(snv_df)} SNVs left after invalid chromosome removal; need >= {valid_snvs_threshold}.")

    # 2) Negative or invalid read counts
    alt_arr = snv_df["alt_count"].values
    ref_arr = snv_df["ref_count"].values
    tot_arr = alt_arr + ref_arr
    mask_reads = (alt_arr >= 0) & (tot_arr >= 0)
    drop_inds = np.where(~mask_reads)[0]
    if len(drop_inds) > 0:
        dropped_reasons += combine_reasons(
            snv_df["chromosome_index"].values,
            snv_df["position"].values,
            drop_inds,
            "Negative or invalid read counts"
        )
    snv_df = snv_df[mask_reads].reset_index(drop=True)
    if len(snv_df) < valid_snvs_threshold:
        raise ValueError(f"{sample_id}: only {len(snv_df)} with valid read counts; need >= {valid_snvs_threshold}.")

    # Recalc
    alt_arr = snv_df["alt_count"].values
    ref_arr = snv_df["ref_count"].values
    tot_arr = alt_arr + ref_arr

    # 3) Match CN segment
    cn_df_ = cn_df.dropna(subset=["minor_cn"]).reset_index(drop=True)
    if len(cn_df_) == 0:
        raise ValueError(f"{sample_id}: no valid CN rows after dropping NA minor_cn.")

    def match_cn(schr, spos):
        row_ids = cn_df_.index[
            (cn_df_.iloc[:, 0] == schr) &
            (cn_df_.iloc[:, 1] <= spos) &
            (cn_df_.iloc[:, 2] >= spos)
        ]
        return row_ids[0] if len(row_ids)>0 else -1

    matched_idx = [match_cn(snv_df.loc[i,"chromosome_index"], snv_df.loc[i,"position"])
                   for i in range(len(snv_df))]
    matched_idx = np.array(matched_idx)

    mask_cn = (matched_idx >= 0)
    drop_inds = np.where(~mask_cn)[0]
    if len(drop_inds) > 0:
        dropped_reasons += combine_reasons(
            snv_df["chromosome_index"].values,
            snv_df["position"].values,
            drop_inds,
            "No matching CN segment"
        )
    snv_df = snv_df[mask_cn].reset_index(drop=True)
    matched_idx = matched_idx[mask_cn]
    if len(snv_df) < valid_snvs_threshold:
        raise ValueError(f"{sample_id}: only {len(snv_df)} with valid CN match; need >= {valid_snvs_threshold}.")

    alt_arr = snv_df["alt_count"].values
    ref_arr = snv_df["ref_count"].values
    tot_arr = alt_arr + ref_arr

    tot_cn   = cn_df_["total_cn"].values[matched_idx]
    minor_cn = cn_df_["minor_cn"].values[matched_idx]
    major_cn = cn_df_["major_cn"].values[matched_idx]
    minor_lim = np.maximum(minor_cn, major_cn)

    # 4) Multiplicity
    mult = np.round(
        (alt_arr / tot_arr / purity) * (tot_cn * purity + (1 - purity)*2)
    ).astype(int)
    # minor_count = np.minimum(mult, minor_lim)
    minor_count = np.minimum(mult, minor_lim)
    minor_count[minor_count == 0] = 1

    mask_valid_mult = (minor_count>0) & (tot_cn>0)
    drop_inds = np.where(~mask_valid_mult)[0]
    if len(drop_inds) > 0:
        dropped_reasons += combine_reasons(
            snv_df["chromosome_index"].values,
            snv_df["position"].values,
            drop_inds,
            "Invalid multiplicities (<=0)"
        )
    snv_df = snv_df[mask_valid_mult].reset_index(drop=True)
    minor_count = minor_count[mask_valid_mult]
    tot_cn      = tot_cn[mask_valid_mult]

    if len(snv_df) < valid_snvs_threshold:
        raise ValueError(f"{sample_id}: only {len(snv_df)} remain after multiplicity filtering; need >= {valid_snvs_threshold}.")

    # 5) Piecewise approximation of theta
    #    For each SNV => key (cn=2, total_cn, minor_count)
    def combo_key(tcount, mcount):
        return f"{tcount}_{mcount}"

    cache_good = {}
    cache_bad  = set()
    sample_coef = np.zeros((len(snv_df),6))
    sample_cut  = np.zeros((len(snv_df),2))
    sample_diff = np.zeros(len(snv_df))

    for i in range(len(snv_df)):
        key = combo_key(tot_cn[i], minor_count[i])
        if key in cache_good:
            # load
            sample_coef[i,:] = cache_good[key]["coef"]
            sample_cut[i,:]  = cache_good[key]["cut"]
            sample_diff[i]   = cache_good[key]["diff"]
        elif key in cache_bad:
            sample_diff[i] = 999
        else:
            # new approximation
            res = linear_approximate(
                bv = minor_count[i], 
                cv = tot_cn[i],
                cn = 2,
                pur = purity
            )
            if res["diff"] <= diff_cutoff:
                sample_coef[i,:] = res["coef"]
                sample_cut[i,:]  = res["w_cut"]
                sample_diff[i]   = res["diff"]
                cache_good[key]  = {
                    "coef": res["coef"],
                    "cut":  res["w_cut"],
                    "diff": res["diff"]
                }
            else:
                sample_diff[i] = 999
                cache_bad.add(key)

    mask_approx = (sample_diff <= diff_cutoff)
    drop_inds = np.where(~mask_approx)[0]
    if len(drop_inds)>0:
        dropped_reasons += combine_reasons(
            snv_df["chromosome_index"].values,
            snv_df["position"].values,
            drop_inds,
            "Piecewise approx error>cutoff"
        )
    snv_df = snv_df[mask_approx].reset_index(drop=True)
    sample_coef = sample_coef[mask_approx,:]
    sample_cut  = sample_cut[mask_approx,:]
    minor_count = minor_count[mask_approx]
    tot_cn      = tot_cn[mask_approx]

    if len(snv_df) < valid_snvs_threshold:
        raise ValueError(f"{sample_id}: only {len(snv_df)} remain after approx filter; need >= {valid_snvs_threshold}.")

    # Recalc alt/ref after final drop
    alt_arr = snv_df["alt_count"].values
    ref_arr = snv_df["ref_count"].values
    tot_arr = alt_arr + ref_arr

    # 6) Evaluate phi
    #   phi = 2 / ( minor_count/(alt_arr/tot_arr) - tot_cn + 2 )
    fraction = alt_arr / tot_arr
    fraction[fraction<=0] = 1e-12
    phi = 2.0 / ((minor_count/fraction) - tot_cn + 2.0)

    # keep phi <=1.5 and >0
    mask_phi = (phi>0) & (phi<=1.5)
    drop_inds = np.where(~mask_phi)[0]
    if len(drop_inds)>0:
        dropped_reasons += combine_reasons(
            snv_df["chromosome_index"].values,
            snv_df["position"].values,
            drop_inds,
            "Phi out of range (<=0 or>1.5)"
        )
    snv_df = snv_df[mask_phi].reset_index(drop=True)
    sample_coef = sample_coef[mask_phi,:]
    sample_cut  = sample_cut[mask_phi,:]
    minor_count = minor_count[mask_phi]
    tot_cn      = tot_cn[mask_phi]

    if len(snv_df) < valid_snvs_threshold:
        raise ValueError(f"{sample_id}: only {len(snv_df)} remain after phi filter; need >= {valid_snvs_threshold}.")

    # Final results
    result_dict = {
        "snv_df_final": snv_df,
        "minor_read": alt_arr[mask_phi],
        "total_read": (alt_arr+ref_arr)[mask_phi],
        "minor_count": minor_count,
        "total_count": tot_cn,
        "coef": sample_coef,
        "cutbeta": sample_cut,
        "excluded_SNVs": dropped_reasons,
        "purity": purity
    }
    return result_dict

def build_cliPP2_input(preproc_res):
    """
    Convert the dictionary from preprocess_for_clipp2(...) 
    into a DataFrame that matches the columns needed by CliPP2.
    
    For multi-region data, you would expand this by replicating each SNV 
    across multiple 'regions' or 'samples', each with distinct alt_counts, etc.
    
    Returns
    -------
    df_for_cliPP2 : pd.DataFrame
       Columns: [mutation, alt_counts, ref_counts, major_cn, minor_cn, normal_cn, tumour_purity]
    n : int
       Number of unique mutations
    m : int
       Number of samples (here 1 if single region)
    """

    snv_df_final = preproc_res["snv_df_final"]
    alt_counts   = preproc_res["minor_read"]
    tot_counts   = preproc_res["total_read"]  # alt+ref
    minor_count  = preproc_res["minor_count"] # from CN step
    total_count  = preproc_res["total_count"]
    purity       = preproc_res["purity"]
    
    n_snv = len(snv_df_final)  # one row per SNV if single region
    # For multi-region, you'd have n_snv * n_regions
    # Here we do single region => m=1
    df_for_cliPP2 = pd.DataFrame({
        "mutation": np.arange(n_snv),  # 0..n_snv-1
        "alt_counts": alt_counts,
        "ref_counts": (tot_counts - alt_counts),
        # We can define minor_cn, major_cn from (minor_count, total_count-minor_count)
        "minor_cn": minor_count,
        "major_cn": (total_count - minor_count),
        "normal_cn": 2,            # typical assumption
        "tumour_purity": purity    # same purity repeated
    })
    n = len(np.unique(df_for_cliPP2.mutation))
    m = 1
    return df_for_cliPP2, n, m