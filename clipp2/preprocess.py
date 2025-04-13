import os
import sys
import numpy as np
import pandas as pd


def process_files(snvfile, cnafile, purityfile):
    """
    Process SNV, CNA, and purity files into a DataFrame with computed CCF.
    
    Parameters:
      - snvfile: path to the SNV file.
      - cnafile: path to the CNA file.
      - purityfile: path to the purity file.
    
    Returns:
      A pandas DataFrame containing the following columns:
        'mutation_chrom', 'mutation', 'position', 'region', 'ref_counts',
        'alt_counts', 'major_cn', 'minor_cn', 'normal_cn', 'tumour_purity', 'ccf'
    """
    # ---------------------------
    # Load data from files.
    dfsnv = pd.read_csv(snvfile, sep='\t')
    dfcna = pd.read_csv(cnafile, sep='\t')
    with open(purityfile, 'r') as file:
        purity = float(file.read().strip())

    # ---------------------------
    # Check that required columns exist in the input files.
    required_snv_cols = {'chromosome_index', 'position', 'ref_count', 'alt_count'}
    required_cna_cols = {'chr', 'startpos', 'endpos', 'nMajor', 'nMinor'}
    missing_snv = required_snv_cols - set(dfsnv.columns)
    missing_cna = required_cna_cols - set(dfcna.columns)
    assert not missing_snv, f"Missing required columns in SNV file: {missing_snv}"
    assert not missing_cna, f"Missing required columns in CNA file: {missing_cna}"

    # Ensure purity is within a plausible range.
    assert 0 <= purity <= 1, f"Tumour purity must be between 0 and 1, but got {purity}"
    
    print(f"Loaded SNV data from {snvfile} with shape {dfsnv.shape}")
    print(f"Loaded CNA data from {cnafile} with shape {dfcna.shape}")
    print(f"Purity: {purity}")

    # ---------------------------
    # Initialize copy number lists.
    major_cn = []
    minor_cn = []
    start = []
    end = []
    # Iterate over SNV rows and match each row in the CNA DataFrame by chromosome and position range.
    for idx, row in dfsnv.iterrows():
        matching_rows = dfcna[
            (dfcna['chr'] == str(row['chromosome_index'])) & 
            (dfcna['startpos'] <= row['position']) &
            (dfcna['endpos'] >= row['position'])
        ]
        if not matching_rows.empty:
            major_cn.append(matching_rows['nMajor'].iloc[0])
            minor_cn.append(matching_rows['nMinor'].iloc[0])
            start.append(matching_rows['startpos'].iloc[0])
            end.append(matching_rows['endpos'].iloc[0])
        else:
            major_cn.append(1)
            minor_cn.append(1)
            start.append(row['position'] - 1)
            end.append(row['position'] + 1)
    
    # Validate that the copy number lists have the same length as SNV rows.
    assert len(major_cn) == dfsnv.shape[0], "Length of major_cn does not match the number of SNV rows"
    assert len(minor_cn) == dfsnv.shape[0], "Length of minor_cn does not match the number of SNV rows"
    
    # ---------------------------
    # Construct the DataFrame.
    df = pd.DataFrame({
        'mutation_chrom': dfsnv['chromosome_index'],
        'mutation': dfsnv['position'],  # Preserve the mutation location if needed.
        'position': dfsnv['position'],
        'ref_counts': dfsnv['ref_count'],
        'alt_counts': dfsnv['alt_count'],
        'major_cn': major_cn,
        'minor_cn': minor_cn,
        'normal_cn': [2] * len(major_cn),
        'tumour_purity': [purity] * len(major_cn),
        'startpos': start,
        'endpos': end
    })
    
    # Check that the constructed DataFrame contains all required columns.
    required_df_cols = {'mutation_chrom', 'position', 'ref_counts', 'alt_counts',
                        'major_cn', 'minor_cn', 'normal_cn', 'tumour_purity'}
    missing_df_cols = required_df_cols - set(df.columns)
    assert not missing_df_cols, f"Missing required columns in constructed DataFrame: {missing_df_cols}"
    print(f"Constructed df with shape {df.shape}")
    
    # Additional check: Randomly validate matching in the CNA file.
    sample_idx = np.random.choice(df.index, size=min(5, df.shape[0]), replace=False)
    for idx in sample_idx:
        row = df.loc[idx]
        matching_rows = dfcna[
            (dfcna['chr'] == str(row['mutation_chrom'])) & 
            (dfcna['startpos'] <= row['position']) &
            (dfcna['endpos'] >= row['position'])
        ]
        if not matching_rows.empty:
            expected_major = matching_rows['nMajor'].iloc[0]
            expected_minor = matching_rows['nMinor'].iloc[0]
            if row['major_cn'] != expected_major or row['minor_cn'] != expected_minor:
                print(f"Warning: Mismatch in row {idx}. Expected major: {expected_major}, minor: {expected_minor}; got major: {row['major_cn']}, minor: {row['minor_cn']}")
        else:
            if row['major_cn'] != 1 or row['minor_cn'] != 1:
                print(f"Warning: No match in CNA for row {idx} but found major_cn={row['major_cn']} and minor_cn={row['minor_cn']}. Expected defaults 1 and 1.")
    
    # ---------------------------
    return df


def insert_distinct_rows_multi(df_list):
    """
    Given a list of M DataFrames (df_list), each DataFrame has columns including:
      ['mutation_chrom', 'mutation', 'position', 'ref_counts', 'alt_counts',
       'major_cn', 'minor_cn', 'normal_cn', 'tumour_purity', 'startpos', 'endpos']
    We want to ensure that each DataFrame has *all* the rows (by key) 
    that appear in any DataFrame in df_list.

    Steps:
      1) Identify the key columns: ['mutation_chrom','mutation'].
      2) Gather the union of *all* (key) rows across all DataFrames.
      3) For each DataFrame, find which key rows are missing 
         and create new rows with default columns:
            major_cn      = 1
            minor_cn      = 1
            normal_cn     = 2
            startpos      = 1
            endpos        = 1
            tumour_purity = (the DataFrame's own tumour_purity 
                             => we take the first row's value as a default)
      4) Insert those new rows into the DataFrame. 
         This ensures each DataFrame ends up with all keys 
         from the union of keys in df_list.

    Returns
    -------
    new_list : list of pd.DataFrame
        The updated DataFrames after insertion of missing rows.
    """

    # 1) Define the key columns
    key_cols = ['mutation_chrom', 'position']

    # 2) Gather union of all keys across all DataFrames (only once)
    all_key_tuples = set()
    for df in df_list:
        # For each row, collect the tuple (mutation_chrom, mutation)
        all_key_tuples.update(zip(df['mutation_chrom'], df['position']))

    # Convert that set into a list we can iterate over
    all_keys_list = list(all_key_tuples)

    new_list = []

    # 3) For each DataFrame, identify which keys are missing and insert them
    for df in df_list:
        # If the DataFrame is empty, pick a fallback purity
        if df.empty:
            df_purity = 0.5
        else:
            # pick the first row's tumour_purity as this DF's default
            df_purity = df['tumour_purity'].iloc[0]

        # Build a set of existing keys in this DataFrame
        existing_keys = set(zip(df['mutation_chrom'], df['position']))

        # Find keys in all_keys_list that are not present in the current DF
        missing_keys = [k for k in all_keys_list if k not in existing_keys]

        if len(missing_keys) == 0:
            # No missing keys => no change
            new_list.append(df)
            continue
        print(len(missing_keys))
        # Build a DataFrame for the missing keys
        to_insert = pd.DataFrame(missing_keys, columns=['mutation_chrom','position'])

        # Provide default values for required columns
        to_insert['major_cn']      = 1
        to_insert['minor_cn']      = 1
        to_insert['normal_cn']     = 2
        to_insert['tumour_purity'] = df_purity
        to_insert['startpos']      = 1
        to_insert['endpos']        = 1

        # If you want defaults for alt_counts, ref_counts, etc., do it here:
        # Example:
        to_insert['ref_counts'] = 1000
        to_insert['alt_counts'] = 1

        # For columns that exist in df but are still missing in to_insert,
        # fill them with e.g. np.nan or some default:
        needed_cols = set(df.columns) - set(to_insert.columns)
        for c in needed_cols:
            to_insert[c] = np.nan

        # Reorder columns to match the original DataFrame's columns
        to_insert = to_insert[df.columns]

        # Append the missing rows to the original DataFrame
        df_out = pd.concat([df, to_insert], ignore_index=True)
        df_out['total_cn'] = df_out['major_cn'] + df_out['minor_cn']
        new_list.append(df_out)

    return new_list

def export_snv_cna_and_purity(df, dir, snv_path, cna_path, purity_path):
    """
    Follow the steps from your code snippet for a *single* DataFrame `df`, 
    creating `dir` if it does not exist:

      1) SNV subset with columns => ['mutation_chrom','position','alt_counts','ref_counts']
         renamed to => ['chromosome_index','position','alt_count','ref_count']
         then add 'total_cn' = df['major_cn'] + df['minor_cn']
         save to snv_path within `dir`.

      2) Re-assign total_cn = major_cn + minor_cn => save df again to cna_path 
         (In your snippet, just a direct copy with total_cn added.)

      3) Another subset => df[['mutation_chrom','startpos','endpos','major_cn','minor_cn']], 
         rename columns => 
            'mutation_chrom' -> 'chromosome_index'
            'startpos'       -> 'startpos'
            'endpos'         -> 'endpos'
            'major_cn'       -> 'major_cn'
            'minor_cn'       -> 'minor_cn'
         add total_cn, then save again to cna_path. 
         (Your snippet suggests overwriting or appending the same file.)

      4) Write df['tumour_purity'].iloc[0] to purity_path as a single scalar.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain at least:
          ['mutation_chrom','position','alt_counts','ref_counts',
           'major_cn','minor_cn','startpos','endpos','tumour_purity']
    dir : str
        Directory path where files will be saved. This function ensures 
        the directory is created if it doesn't exist.
    snv_path : str
        Filename for the SNV output inside `dir`. E.g. '9post.snv.txt'
    cna_path : str
        Filename for the CNA output inside `dir`. E.g. '9post.cna.txt'
    purity_path : str
        Filename for the purity scalar inside `dir`. E.g. '9post.purity.txt'
    """

    # Ensure the directory exists
    os.makedirs(dir, exist_ok=True)

    # ---------- 1) SNV subset / rename / add total_cn / save to snv_path ----------
    df = df.sort_values(by=['mutation_chrom', 'position'])
    df_snv = df[['mutation_chrom', 'position', 'alt_counts', 'ref_counts']].rename(
        columns={
            'mutation_chrom': 'chromosome_index',
            'position':       'position',
            'alt_counts':     'alt_count',
            'ref_counts':     'ref_count'
        }
    )
    # Save to snv_path within `dir`
    snv_fullpath = os.path.join(dir, snv_path)
    df_snv.to_csv(snv_fullpath, sep='\t', index=False)

    # ---------- 3) Another cna snippet => rename columns, add total_cn, 
    #     and save to the same cna_path again (overwriting).
    df_cna_2 = df[['mutation_chrom','startpos','endpos','major_cn','minor_cn', 'total_cn']].rename(
        columns={
            'mutation_chrom': 'chromosome_index',
            'startpos':       'startpos',
            'endpos':         'endpos',
            'major_cn':       'major_cn',
            'minor_cn':       'minor_cn',
            'total_cn' : 'total_cn'
        }
    )
    df_cna_2['total_cn'] = df_cna_2['major_cn'] + df_cna_2['minor_cn']

    # Overwrite or append the same cna_path: 
    cna_fullpath = os.path.join(dir, cna_path)
    df_cna_2.to_csv(cna_fullpath, sep='\t', index=False)

    # ---------- 4) Write tumour_purity scalar to purity_path ----------
    purity_fullpath = os.path.join(dir, purity_path)
    purity_val = df['tumour_purity'].iloc[0]
    with open(purity_fullpath, 'w') as f:
        f.write(f"{purity_val}\n")
##############################################################################
# 1) Utility functions (translated from your R script)
##############################################################################

def theta(w, bv, cv, cn, purity):
    """
    Same as in your original code: 
    theta = (exp(w)*bv)/[ (1+exp(w))*cn*(1-purity) + (1+exp(w))*cv*purity ]
    """
    return (np.exp(w) * bv) / (
        (1.0 + np.exp(w)) * cn * (1.0 - purity) +
        (1.0 + np.exp(w)) * cv * purity
    )

def LinearEvaluate(x, slope, intercept):
    return slope * x + intercept

def LinearApproximate(bv, cv, cn, purity, diag_plot=False):
    """
    A modified version that FIXES the piecewise breakpoints at w_cut_low=-1.8, w_cut_high=1.8,
    and calculates the corresponding piecewise coefficients and 'diff'.

    Returns:
       {
         'w.cut': array of length 2,  e.g. [-1.8, 1.8],
         'diff': float,               # max absolute difference
         'coef': array of length 6    # [slope1, intercept1, slope2, intercept2, slope3, intercept3]
       }
    """

    # 1) Generate w_vec from -4..4 in steps of 0.1 (same as original code).
    w_vec = np.arange(-4.0, 4.1, 0.1)
    actual_theta = theta(w_vec, bv, cv, cn, purity)
    Nw = len(w_vec)

    # 2) Identify the indices in w_vec that correspond to -1.8 and +1.8.
    #    Because of the step=0.1, 
    #    -1.8 is 2.2 away from -4.0 => index = 22 
    #    +1.8 is 5.8 away from -4.0 => index = 58 
    # But let's do it programmatically:
    def find_index_for_cut(cut_val):
        # find the index in w_vec that is closest to cut_val
        diff_arr = np.abs(w_vec - cut_val)
        return np.argmin(diff_arr)

    i = find_index_for_cut(-1.8)  # e.g. 22
    j = find_index_for_cut( 1.8)  # e.g. 58

    # 3) We'll break the piecewise segments at w_vec[i], w_vec[j].
    #    That means:
    #    segment1 => indices [0..i]
    #    segment2 => indices [i+1..j]
    #    segment3 => indices [j+1..(Nw-1)]

    w_cut_low  = w_vec[i]   # ~ -1.8
    w_cut_high = w_vec[j]   # ~  1.8

    # -- Segment1: slope & intercept from (w[0], theta[0]) to (w[i], theta[i])
    slope1 = (actual_theta[i] - actual_theta[0]) / (w_vec[i] - w_vec[0])
    intercept1 = actual_theta[0] - w_vec[0]*slope1

    # -- Segment2: slope & intercept from (w[i],theta[i]) to (w[j],theta[j])
    slope2 = (actual_theta[j] - actual_theta[i]) / (w_vec[j] - w_vec[i])
    intercept2 = actual_theta[i] - w_vec[i]*slope2

    # -- Segment3: slope & intercept from (w[j],theta[j]) to (w[-1],theta[-1])
    slope3 = (actual_theta[-1] - actual_theta[j]) / (w_vec[-1] - w_vec[j])
    intercept3 = actual_theta[-1] - w_vec[-1]*slope3

    # 4) Build piecewise approximation
    approx_vals = np.zeros(Nw)
    approx_vals[0:(i+1)]      = LinearEvaluate(w_vec[0:(i+1)], slope1, intercept1)
    approx_vals[(i+1):(j+1)]  = LinearEvaluate(w_vec[(i+1):(j+1)], slope2, intercept2)
    approx_vals[(j+1):Nw]     = LinearEvaluate(w_vec[(j+1):Nw], slope3, intercept3)

    # 5) Compute local_diff => max |actual - approx|
    local_diff = np.max(np.abs(actual_theta - approx_vals))

    coefs = np.array([slope1, intercept1, slope2, intercept2, slope3, intercept3])

    # 6) Return the "best_rec" dict with the forced breakpoints
    return {
        'w.cut': np.array([w_cut_low, w_cut_high]),  # e.g. [-1.8, 1.8]
        'diff': local_diff,
        'coef': coefs
    }

def CombineReasons(chrom, pos, indices, reason):
    """
    Mirroring your R CombineReasons function, building lines like:
      chrom   pos   reason
    If len(indices)==0 => return None or empty list.
    """
    results = []
    for i in indices:
        results.append(f"{chrom[i]}\t{pos[i]}\t{reason}")
    return results

##############################################################################
# 2) Main function (translated from your R script) 
##############################################################################

def preprocess(
    snv_file, cn_file, purity_file, sample_id, output_prefix, 
    drop_data=True
):
    """
    Python translation of your R code snippet, with the ability to skip row dropping.
    
    Parameters
    ----------
    snv_file : str
        Path to SNV file (e.g. '/AVPC/ACC9/9-post.snv.txt')
    cn_file : str
        Path to CNV file (e.g. '/AVPC/ACC9/9-post.cna.txt')
    purity_file : str
        Path to purity file (e.g. '/AVPC/ACC9/9-post.purity.txt')
    sample_id : str
        Sample ID for reference (not heavily used in this function except for log messages)
    output_prefix : str
        Directory/path prefix where outputs should be written
    drop_data : bool
        If True (default), rows failing filters (negative reads, invalid CN, etc.) are dropped.
        If False, all data is retained, and any "failures" are merely logged but not removed.
    
    The function performs the following steps:
      1) Check existence of input files.
      2) Create output directory if not existing.
      3) Read purity, read SNV, read CNV.
      4) Possibly drop certain rows based on R logic (if drop_data=True).
      5) Attempt the piecewise linear approximation.
      6) Write results to output files: r.txt, n.txt, minor.txt, total.txt, multiplicity.txt, etc.
         Also outputs excluded_SNVs.txt (logged failures).
    """

    print("SNV file:     ", snv_file)
    print("CNV file:     ", cn_file)
    print("Purity file:  ", purity_file)
    print("Sample ID:    ", sample_id)
    print("Output dir:   ", output_prefix)
    print("drop_data:    ", drop_data)

    # 1) Check existence of input files
    if not os.path.exists(snv_file):
        sys.exit(f"The Input SNV file: {snv_file} does not exist.")
    if not os.path.exists(cn_file):
        sys.exit(f"The Input CNV file: {cn_file} does not exist.")
    if not os.path.exists(purity_file):
        sys.exit(f"The Input Purity file: {purity_file} does not exist.")

    # 2) Create output directory if it does not exist
    if not os.path.isdir(output_prefix):
        os.makedirs(output_prefix, exist_ok=True)

    # 3) Define utility functions
    def theta(w, bv, cv, cn, purity):
        """
        Equivalent to the R function:
           theta = (exp(w)*bv) / ((1+exp(w))*cn*(1-purity) + (1+exp(w))*cv*purity)
        """
        return (np.exp(w)*bv) / (
            (1 + np.exp(w))*cn*(1 - purity) + (1 + np.exp(w))*cv*purity
        )

    def LinearEvaluate(x, a, b):
        """
        Equivalent to the R function:
           LinearEvaluate(x, a, b) = a*x + b
        """
        return a*x + b

    def LinearApproximate(bv, cv, cn, purity, diag_plot=False):
        """
        Equivalent to the R function LinearApproximate.
        Attempts to approximate the function theta(w, bv, cv, cn, purity)
        with 3 line segments, returning the combination that yields the
        minimum maximum difference from the true function.
        """
        w_arr = np.arange(-40, 41) / 10.0  # -4.0 to +4.0 in steps of 0.1
        actual_theta = theta(w_arr, bv, cv, cn, purity)

        No_w = len(w_arr)
        # We'll store results in arrays or lists
        total_combos = (No_w - 3) * (No_w - 2) // 2
        diff_vals = np.zeros(total_combos, dtype=float)
        coef_vals = np.zeros((total_combos, 6), dtype=float)
        w_cut_vals = np.zeros((total_combos, 2), dtype=float)
        approximations = [None]*total_combos

        k = 0
        # The R loops: for(i in 2:(length(w)-2)) for(j in (i+1):(length(w)-1))
        # Because R is 1-based, we want i in [2..(No_w-2)], j in [i+1..(No_w-1)].
        for i in range(2, No_w-1):        # i in [2..No_w-2]
            for j in range(i+1, No_w):    # j in [i+1..No_w-1]
                # Coefficients for line segments
                denom_1 = (w_arr[i] - w_arr[1])
                a1 = (actual_theta[i] - actual_theta[1]) / denom_1
                b1 = actual_theta[1] - w_arr[1]*a1

                denom_2 = (w_arr[j] - w_arr[i])
                a2 = (actual_theta[j] - actual_theta[i]) / denom_2
                b2 = actual_theta[i] - w_arr[i]*a2

                denom_3 = (w_arr[No_w-1] - w_arr[j])
                a3 = (actual_theta[No_w-1] - actual_theta[j]) / denom_3
                b3 = actual_theta[No_w-1] - w_arr[No_w-1]*a3

                # Build the piecewise approximation
                seg1_x = w_arr[1:i+1]
                seg1_y = LinearEvaluate(seg1_x, a1, b1)

                seg2_x = w_arr[i+1:j+1]
                seg2_y = LinearEvaluate(seg2_x, a2, b2)

                seg3_x = w_arr[j+1:No_w]
                seg3_y = LinearEvaluate(seg3_x, a3, b3)

                full_approx = np.zeros(No_w, dtype=float)
                full_approx[1:i+1] = seg1_y
                full_approx[i+1:j+1] = seg2_y
                full_approx[j+1:No_w] = seg3_y

                the_diff = np.max(np.abs(actual_theta[1:] - full_approx[1:]))

                diff_vals[k] = the_diff
                coef_vals[k, :] = [a1, b1, a2, b2, a3, b3]
                w_cut_vals[k, :] = [w_arr[i], w_arr[j]]
                approximations[k] = full_approx
                k += 1

        # Find best difference
        best_idx = np.argmin(diff_vals)
        best_cut = w_cut_vals[best_idx, :]
        best_diff = diff_vals[best_idx]
        best_coef = coef_vals[best_idx, :]

        if diag_plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(w_arr, actual_theta, label='Actual theta')
            plt.plot(w_arr, approximations[best_idx], label='Approx theta', color='red')
            plt.ylim([0,1])
            plt.xlim([-5,5])
            plt.legend()
            plt.show()

        return {
            'w_cut': best_cut,
            'diff': best_diff,
            'coef': best_coef
        }

    def CombineReasons(chrom_list, pos_list, ind_list, reason):
        """
        Equivalent to R's CombineReasons function:
          'chrom pos reason' lines for each index in ind_list.
        """
        results = []
        for idx in ind_list:
            results.append(f"{chrom_list[idx]}\t{pos_list[idx]}\t{reason}")
        return results

    # 4) Read purity, read SNV, read CNA
    pp_table = pd.read_csv(purity_file, header=None, sep="\t")
    purity = pp_table.iloc[0,0]  # first row, first column

    # Set threshold to check if we have enough valid SNVs
    VALID_CONT = 0

    # Initialize arrays to store meta
    case_store     = []
    cutbeta_store  = []
    valid_store    = []
    invalid_store  = []
    dropped_SNV    = []

    # Read SNV file
    snv_df = pd.read_csv(snv_file, header=0, sep="\t")
    mutation_chrom = snv_df["chromosome_index"].values.astype(float)
    mutation_pos   = snv_df["position"].values.astype(float)
    minor_read     = snv_df["alt_count"].values.astype(float)
    ref_read       = snv_df["ref_count"].values.astype(float)
    total_read     = minor_read + ref_read

    # 5) Possibly drop data (or not) based on filters

    #--- Filter 1: sex chromosomes (where chromosome_index is NaN) ---
    # Indices for passing or failing
    valid_ind_1 = np.where(~np.isnan(mutation_chrom))[0]
    drop_ind_1  = np.where(np.isnan(mutation_chrom))[0]
    dropped_SNV.extend(CombineReasons(mutation_chrom, mutation_pos, drop_ind_1,
                                      "The SNV is on sex chromosomes."))

    if drop_data:
        if len(valid_ind_1) < VALID_CONT:
            sys.exit(f"Fewer than {VALID_CONT} SNVs on autosomes.")
        # Keep only valid
        mutation_chrom = mutation_chrom[valid_ind_1]
        mutation_pos   = mutation_pos[valid_ind_1]
        minor_read     = minor_read[valid_ind_1]
        total_read     = total_read[valid_ind_1]

    #--- Filter 2: negative reads ---
    if drop_data:
        valid_ind_2 = np.where((minor_read >= 0) & (total_read >= 0))[0]
        drop_ind_2  = np.setdiff1d(np.arange(len(minor_read)), valid_ind_2)
        dropped_SNV.extend(CombineReasons(mutation_chrom, mutation_pos, drop_ind_2,
                                          "The SNV has negative reads."))

        if len(valid_ind_2) < VALID_CONT:
            sys.exit(f"Fewer than {VALID_CONT} SNVs with non-negative reads.")
        # Keep only valid
        mutation_chrom = mutation_chrom[valid_ind_2]
        mutation_pos   = mutation_pos[valid_ind_2]
        minor_read     = minor_read[valid_ind_2]
        total_read     = total_read[valid_ind_2]

    # (Recompute No_mutations after filters so far)
    No_mutations = len(mutation_chrom)

    # 6) Process copy number
    cn_df = pd.read_csv(cn_file, header=0, sep="\t")
    # remove rows with NA in 'minor_cn' (R code)
    cn_df = cn_df.dropna(subset=["minor_cn"])
    if cn_df.shape[0] == 0:
        sys.exit("No valid copy number entries (all minor_cn were NaN?).")

    cn_chr   = cn_df["chromosome_index"].values
    # Adjust column names as needed for your real data:
    cn_start = cn_df["startpos"].values
    cn_end   = cn_df["endpos"].values
    cn_minor = cn_df["minor_cn"].values
    cn_major = cn_df["major_cn"].values
    cn_total = cn_df["total_cn"].values

    #--- For each mutation, find segment covering that position ---
    mut_cna_id = []
    for x in range(No_mutations):
        found_segment = -1
        for i in range(cn_df.shape[0]):
            if (mutation_chrom[x] == cn_chr[i] and
                mutation_pos[x] >= cn_start[i] and
                mutation_pos[x] <= cn_end[i]):
                found_segment = i
                break
        mut_cna_id.append(found_segment)

    mut_cna_id = np.array(mut_cna_id)

    #--- Filter 3: invalid CN segments ---
    valid_ind_3 = np.where(mut_cna_id >= 0)[0]
    drop_ind_3  = np.setdiff1d(np.arange(No_mutations), valid_ind_3)
    dropped_SNV.extend(CombineReasons(mutation_chrom, mutation_pos, drop_ind_3,
                                      "The SNV does not have valid copy number."))

    if drop_data:
        if len(valid_ind_3) < VALID_CONT:
            sys.exit(f"Fewer than {VALID_CONT} SNVs with valid CN status.")
        # Keep only valid
        mutation_chrom = mutation_chrom[valid_ind_3]
        mutation_pos   = mutation_pos[valid_ind_3]
        minor_read     = minor_read[valid_ind_3]
        total_read     = total_read[valid_ind_3]
        mut_cna_id     = mut_cna_id[valid_ind_3]
        No_mutations   = len(valid_ind_3)

    # Now gather copy number info for *all* or valid subset
    if drop_data:
        minor_copy_lim = np.array([max(cn_minor[idx], cn_major[idx]) for idx in mut_cna_id])
        total_count    = cn_total[mut_cna_id]
    else:
        # If not dropping data, construct the arrays for the entire dataset:
        minor_copy_lim = np.array([
            max(cn_minor[idx], cn_major[idx]) if idx >= 0 else np.nan 
            for idx in mut_cna_id
        ])
        total_count = np.array([
            cn_total[idx] if idx >= 0 else np.nan
            for idx in mut_cna_id
        ])

    # 7) Calculate multiplicity
    multiplicity = np.round(
        (minor_read / total_read) / purity * (total_count * purity + (1 - purity)*2)
    )

    #--- Filter 4: clamp multiplicity and ensure minor_count>0 ---
    minor_count = np.minimum(minor_copy_lim, multiplicity)
    # Force minor_count to at least 1
    minor_count[minor_count < 1] = 1

    if drop_data:
        valid_ind_4 = np.where((minor_count > 0) & (total_count > 0))[0]
        drop_ind_4  = np.setdiff1d(np.arange(len(minor_read)), valid_ind_4)
        dropped_SNV.extend(CombineReasons(mutation_chrom, mutation_pos, drop_ind_4,
                                          "The SNV has negative or zero multiplicities."))

        if len(valid_ind_4) < VALID_CONT:
            sys.exit(f"Fewer than {VALID_CONT} SNVs that pass multiplicity checks.")
        # Keep only valid
        mutation_chrom = mutation_chrom[valid_ind_4]
        mutation_pos   = mutation_pos[valid_ind_4]
        minor_read     = minor_read[valid_ind_4]
        total_read     = total_read[valid_ind_4]
        minor_count    = minor_count[valid_ind_4]
        total_count    = total_count[valid_ind_4]
        No_mutations   = len(valid_ind_4)

    # 8) Piecewise approximation
    sample_coef    = np.zeros((No_mutations, 6), dtype=float)
    sample_cutbeta = np.zeros((No_mutations, 2), dtype=float)
    sample_diff    = np.zeros(No_mutations, dtype=float)

    # We track (cn=2, total_count[m], minor_count[m]) as a key
    case_store     = []
    cutbeta_store  = []
    valid_store    = []
    invalid_store  = []

    for m in range(No_mutations):
        # If not dropping, watch out for any NaNs:
        if np.isnan(total_count[m]) or np.isnan(minor_count[m]):
            # If there's invalid CN, skip approximation
            sample_diff[m] = 9999
            continue

        key = f"2_{total_count[m]}_{minor_count[m]}"
        if key in valid_store:
            idx_key = valid_store.index(key)
            sample_coef[m,:]    = case_store[idx_key]
            sample_cutbeta[m,:] = cutbeta_store[idx_key]
            sample_diff[m]      = 0.0
        elif key in invalid_store:
            sample_diff[m] = 1.0
        else:
            res = LinearApproximate(minor_count[m], total_count[m], 2, purity, diag_plot=False)
            if res['diff'] <= 0.1:
                sample_coef[m,:]    = res['coef']
                sample_cutbeta[m,:] = res['w_cut']
                sample_diff[m]      = res['diff']

                valid_store.append(key)
                case_store.append(res['coef'])
                cutbeta_store.append(res['w_cut'])
            else:
                sample_diff[m] = 1.0
                invalid_store.append(key)

    #--- Filter 5: sample_diff <= 0.1 ---
    if drop_data:
        valid_ind_5 = np.where(sample_diff <= 0.1)[0]
        drop_ind_5  = np.setdiff1d(np.arange(No_mutations), valid_ind_5)
        dropped_SNV.extend(CombineReasons(mutation_chrom, mutation_pos, drop_ind_5,
                                          "The copy numbers for the SNV are not stable enough to approximate."))

        if len(valid_ind_5) < VALID_CONT:
            sys.exit(f"Fewer than {VALID_CONT} SNVs that have valid approximated theta.")
        mutation_chrom = mutation_chrom[valid_ind_5]
        mutation_pos   = mutation_pos[valid_ind_5]
        minor_read     = minor_read[valid_ind_5]
        total_read     = total_read[valid_ind_5]
        minor_count    = minor_count[valid_ind_5]
        total_count    = total_count[valid_ind_5]
        sample_coef    = sample_coef[valid_ind_5, :]
        sample_cutbeta = sample_cutbeta[valid_ind_5, :]
        sample_diff    = sample_diff[valid_ind_5]
        No_mutations   = len(valid_ind_5)

    # 9) Calculate phi and filter outliers
    phi = 2.0 / ((minor_count / (minor_read / total_read)) - total_count + 2.0)

    # Outliers = phi > 1.5
    clonal_ind = np.where(phi > 1.5)[0]
    if drop_data:
        if len(clonal_ind) > 0:
            outlier_higherEnd = np.column_stack((
                mutation_chrom[clonal_ind],
                mutation_pos[clonal_ind],
                total_count[clonal_ind],
                minor_count[clonal_ind]
            ))
            pd.DataFrame(outlier_higherEnd).to_csv(
                os.path.join(output_prefix, "outPosition.txt"),
                header=False, index=False, sep="\t"
            )

    #--- Filter 6: keep 0 < phi <= 1.5 ---
    if drop_data:
        valid_ind_6 = np.where((phi <= 1.5) & (phi > 0))[0]
        drop_ind_6  = np.setdiff1d(np.arange(No_mutations), valid_ind_6)
        dropped_SNV.extend(CombineReasons(mutation_chrom, mutation_pos, drop_ind_6,
                                          "The empirical CP is off, possibly due to incorrect CN or super cluster(s)."))

        if len(valid_ind_6) < VALID_CONT:
            sys.exit(f"Fewer than {VALID_CONT} SNVs with valid empirical phi.")
        mutation_chrom = mutation_chrom[valid_ind_6]
        mutation_pos   = mutation_pos[valid_ind_6]
        minor_read     = minor_read[valid_ind_6]
        total_read     = total_read[valid_ind_6]
        minor_count    = minor_count[valid_ind_6]
        total_count    = total_count[valid_ind_6]
        sample_coef    = sample_coef[valid_ind_6, :]
        sample_cutbeta = sample_cutbeta[valid_ind_6, :]
        phi            = phi[valid_ind_6]
        No_mutations   = len(valid_ind_6)

    # 10) Prepare output
    # index => (chrom, pos, total_count, minor_count)
    index_matrix = np.column_stack((mutation_chrom, mutation_pos, total_count, minor_count))

    # Build output file names
    output_r        = os.path.join(output_prefix, "r.txt")
    output_n        = os.path.join(output_prefix, "n.txt")
    output_minor    = os.path.join(output_prefix, "minor.txt")
    output_total    = os.path.join(output_prefix, "total.txt")
    output_index    = os.path.join(output_prefix, "multiplicity.txt")
    output_pp       = os.path.join(output_prefix, "purity_ploidy.txt")
    output_coef     = os.path.join(output_prefix, "coef.txt")
    output_cutbeta  = os.path.join(output_prefix, "cutbeta.txt")
    output_dropped  = os.path.join(output_prefix, "excluded_SNVs.txt")

    # Write outputs
    np.savetxt(output_r, minor_read,    fmt="%.0f", delimiter="\t")
    np.savetxt(output_n, total_read,    fmt="%.0f", delimiter="\t")
    np.savetxt(output_minor, minor_count, fmt="%.0f", delimiter="\t")
    np.savetxt(output_total, total_count, fmt="%.0f", delimiter="\t")
    np.savetxt(output_index, index_matrix, fmt="%.6g", delimiter="\t")
    # Purity
    np.savetxt(output_pp, [purity], fmt="%.5f", delimiter="\t")
    # Coefficients
    np.savetxt(output_coef, sample_coef, fmt="%.6g", delimiter="\t")
    np.savetxt(output_cutbeta, sample_cutbeta, fmt="%.6g", delimiter="\t")

    with open(output_dropped, 'w') as f:
        for line in dropped_SNV:
            f.write(line + "\n")

    print("Finished processing. Results saved under:", output_prefix)
    if not drop_data:
        print("Note: drop_data=False, so no rows were actually removed. "
              "All data are included in the output files.")
