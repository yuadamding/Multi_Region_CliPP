import os
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
            major_cn     = 1
            minor_cn     = 1
            normal_cn    = 2
            startpos     = 1
            endpos       = 1
            tumour_purity = (the DataFrame's own tumour_purity 
                             => we take the first row's value as a default)
      4) Insert those new rows into the DataFrame. 
         This ensures each DataFrame ends up with all keys 
         from the union of keys in df_list.
    Returns
    -------
    new_list : list of pd.DataFrame
        The updated DataFrames after insertion of missing rows.

    Notes
    -----
    - This example picks the DataFrame's first row's 'tumour_purity' 
      as the default for all newly inserted rows in that DataFrame.
    - If you have multiple different purities in a single DataFrame,
      adapt the logic (e.g. group by region or something else).
    - All DataFrames must have the same columns. If some have no rows,
      we fallback to tumour_purity=0.5 or an arbitrary choice.
    """

    # 1) define key columns
    key_cols = ['mutation_chrom','mutation']

    # 2) gather union of all keys across all dfs
    #    We'll build a big set or use a DataFrame index approach
    #    For convenience, convert each DF to an index on those key_cols,
    #    then combine.
    all_key_tuples = set()
    for df in df_list:
        # Convert to a MultiIndex of (mutation_chrom,mutation)
        #   zip => each row => (chr, mut)
        for row in zip(df['mutation_chrom'], df['mutation']):
            all_key_tuples.add(row)

    #  Now we have the union of keys across all DataFrames
    #  We'll convert that set to a list of dict with columns
    all_keys_list = list(all_key_tuples)

    new_list = []

    for i, df in enumerate(df_list):
        # If the DataFrame is empty, we pick a fallback purity (say 0.5)
        if not df.empty:
            # pick the first row's tumour_purity as this DF's default
            df_purity = df['tumour_purity'].iloc[0]
        else:
            df_purity = 0.5

        # build an index for the DF on (mutation_chrom, mutation) 
        # to see which keys it already has
        existing_keys = set(zip(df['mutation_chrom'], df['mutation']))

        # find the missing ones: those in all_key_tuples but not in this DF
        missing = [k for k in all_keys_list if k not in existing_keys]

        if len(missing) == 0:
            # No distinct rows to insert -> just keep the same DF
            new_list.append(df)
            continue

        # We'll build a small DataFrame containing those missing keys
        to_insert = pd.DataFrame(missing, columns=['mutation_chrom','mutation'])
        # Add default columns for the new rows
        # We'll also ensure that for columns not in (some minimal set),
        # we fill with e.g. NaN or 0 if you prefer. 
        # But the user specifically asked:
        #  major_cn=1, minor_cn=1, normal_cn=2, tumour_purity=df_purity, 
        #  startpos=1, endpos=1
        to_insert['major_cn']      = 1
        to_insert['minor_cn']      = 1
        to_insert['normal_cn']     = 2
        to_insert['tumour_purity'] = df_purity
        to_insert['startpos']      = 1
        to_insert['endpos']        = 1

        # For other columns that exist in df but not in this minimal set 
        # (like 'position','ref_counts','alt_counts'), 
        # we can fill them with e.g. np.nan, or 0, or do nothing if we prefer:
        # We'll fill them with NaN as an example:
        needed_cols = set(df.columns) - set(to_insert.columns)
        for c in needed_cols:
            to_insert[c] = np.nan

        # Now the new rows have every column that DF has.
        # We can reorder columns to match df's column order
        to_insert = to_insert[df.columns]

        # 3) Append them to df
        df_out = pd.concat([df, to_insert], ignore_index=True)
        new_list.append(df_out)

    return new_list

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
            major_cn     = 1
            minor_cn     = 1
            normal_cn    = 2
            startpos     = 1
            endpos       = 1
            tumour_purity = (the DataFrame's own tumour_purity 
                             => we take the first row's value as a default)
      4) Insert those new rows into the DataFrame. 
         This ensures each DataFrame ends up with all keys 
         from the union of keys in df_list.
    Returns
    -------
    new_list : list of pd.DataFrame
        The updated DataFrames after insertion of missing rows.

    Notes
    -----
    - This example picks the DataFrame's first row's 'tumour_purity' 
      as the default for all newly inserted rows in that DataFrame.
    - If you have multiple different purities in a single DataFrame,
      adapt the logic (e.g. group by region or something else).
    - All DataFrames must have the same columns. If some have no rows,
      we fallback to tumour_purity=0.5 or an arbitrary choice.
    """

    # 1) define key columns
    key_cols = ['mutation_chrom','mutation']

    # 2) gather union of all keys across all dfs
    #    We'll build a big set or use a DataFrame index approach
    #    For convenience, convert each DF to an index on those key_cols,
    #    then combine.
    all_key_tuples = set()
    for df in df_list:
        # Convert to a MultiIndex of (mutation_chrom,mutation)
        #   zip => each row => (chr, mut)
        for row in zip(df['mutation_chrom'], df['mutation']):
            all_key_tuples.add(row)

    #  Now we have the union of keys across all DataFrames
    #  We'll convert that set to a list of dict with columns
    all_keys_list = list(all_key_tuples)

    new_list = []

    for i, df in enumerate(df_list):
        # If the DataFrame is empty, we pick a fallback purity (say 0.5)
        if not df.empty:
            # pick the first row's tumour_purity as this DF's default
            df_purity = df['tumour_purity'].iloc[0]
        else:
            df_purity = 0.5

        # build an index for the DF on (mutation_chrom, mutation) 
        # to see which keys it already has
        existing_keys = set(zip(df['mutation_chrom'], df['mutation']))

        # find the missing ones: those in all_key_tuples but not in this DF
        missing = [k for k in all_keys_list if k not in existing_keys]

        if len(missing) == 0:
            # No distinct rows to insert -> just keep the same DF
            new_list.append(df)
            continue

        # We'll build a small DataFrame containing those missing keys
        to_insert = pd.DataFrame(missing, columns=['mutation_chrom','mutation'])
        # Add default columns for the new rows
        # We'll also ensure that for columns not in (some minimal set),
        # we fill with e.g. NaN or 0 if you prefer. 
        # But the user specifically asked:
        #  major_cn=1, minor_cn=1, normal_cn=2, tumour_purity=df_purity, 
        #  startpos=1, endpos=1
        to_insert['major_cn']      = 1
        to_insert['minor_cn']      = 1
        to_insert['normal_cn']     = 2
        to_insert['tumour_purity'] = df_purity
        to_insert['startpos']      = 1
        to_insert['endpos']        = 1

        # For other columns that exist in df but not in this minimal set 
        # (like 'position','ref_counts','alt_counts'), 
        # we can fill them with e.g. np.nan, or 0, or do nothing if we prefer:
        # We'll fill them with NaN as an example:
        needed_cols = set(df.columns) - set(to_insert.columns)
        for c in needed_cols:
            to_insert[c] = 1

        # Now the new rows have every column that DF has.
        # We can reorder columns to match df's column order
        to_insert = to_insert[df.columns]

        # 3) Append them to df
        df_out = pd.concat([df, to_insert], ignore_index=True)
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
    df_snv = df[['mutation_chrom', 'position', 'alt_counts', 'ref_counts']].rename(
        columns={
            'mutation_chrom': 'chromosome_index',
            'position':       'position',
            'alt_counts':     'alt_count',
            'ref_counts':     'ref_count'
        }
    )
    # Add total_cn = df['major_cn'] + df['minor_cn']
    df_snv['total_cn'] = df['major_cn'] + df['minor_cn']

    # Save to snv_path within `dir`
    snv_fullpath = os.path.join(dir, snv_path)
    df_snv.to_csv(snv_fullpath, sep='\t', index=False)

    # ---------- 2) Copy df, add total_cn => save to cna_path ----------
    df_cna_1 = df.copy()
    df_cna_1['total_cn'] = df_cna_1['major_cn'] + df_cna_1['minor_cn']

    cna_fullpath = os.path.join(dir, cna_path)
    df_cna_1.to_csv(cna_fullpath, sep='\t', index=False)

    # ---------- 3) Another cna snippet => rename columns, add total_cn, 
    #     and save to the same cna_path again (overwriting).
    df_cna_2 = df[['mutation_chrom','startpos','endpos','major_cn','minor_cn']].rename(
        columns={
            'mutation_chrom': 'chromosome_index',
            'startpos':       'startpos',
            'endpos':         'endpos',
            'major_cn':       'major_cn',
            'minor_cn':       'minor_cn'
        }
    )
    df_cna_2['total_cn'] = df_cna_2['major_cn'] + df_cna_2['minor_cn']

    # Overwrite or append the same cna_path: 
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

def process_data(
        snv_file, cn_file, purity_file, sample_id, output_prefix, 
        drop_data=True
    ):
    """
    Python translation of your R code snippet. 
    snv_file    : str => e.g. '/AVPC/ACC9/9-post.snv.txt'
    cn_file     : str => e.g. '/AVPC/ACC9/9-post.cna.txt'
    purity_file : str => e.g. '/AVPC/ACC9/9-post.purity.txt'
    sample_id   : str => e.g. '--sample_id'
    output_prefix : str => e.g. 'E:/Dropbox/GitHub/Multi_Region_CliPP/processed_data'

    drop_data   : bool => if True, we apply the R logic for dropping rows 
                          that fail certain conditions (like negative reads, no CN, etc.).
                          If False, we skip those filtering steps.

    The function does the steps:
      1) Check existence of input files
      2) Create output_prefix dir if not existing
      3) Read purity, read snv, read cna
      4) Possibly drop certain rows based on R logic (if drop_data=True)
      5) Attempt the piecewise linear approximation
      6) Write results to r.txt, n.txt, minor.txt, total.txt, multiplicity.txt, etc.
    """

    import os
    # check input files
    if not os.path.isfile(snv_file):
        raise ValueError(f"The Input SNV file: {snv_file} does not exist.")
    if not os.path.isfile(cn_file):
        raise ValueError(f"The Input CNV file: {cn_file} does not exist.")
    if not os.path.isfile(purity_file):
        raise ValueError(f"The Input Purity file: {purity_file} does not exist.")

    # create output_prefix dir if needed
    if not os.path.isdir(output_prefix):
        os.makedirs(output_prefix, exist_ok=True)

    # read purity
    pp_table = pd.read_csv(purity_file, header=None, sep="\t")
    purity = float(pp_table.iloc[0,0])

    # read SNV
    tmp_vcf = pd.read_csv(snv_file, sep="\t", header=0)
    # expecting columns: chromosome_index, position, alt_count, ref_count
    # plus possibly others.

    # read CN
    cn_tmp = pd.read_csv(cn_file, sep="\t", header=0)
    # expecting columns like: chromosome_index, startpos, endpos,
    #   major_cn, minor_cn, total_cn, etc.

    # define some variables as in R
    dropped_SNV = []

    # from your code:
    # mutation.chrom <- as.numeric(tmp.vcf$chromosome_index)
    # mutation.pos   <- as.numeric(tmp.vcf$position)
    # minor.read     <- tmp.vcf$alt_count
    # total.read     <- alt + ref
    mutation_chrom = np.array(tmp_vcf['chromosome_index'], dtype=float)
    mutation_pos   = np.array(tmp_vcf['position'], dtype=float)
    minor_read     = np.array(tmp_vcf['alt_count'], dtype=float)
    total_read     = minor_read + np.array(tmp_vcf['ref_count'], dtype=float)

    # drop_data logic
    VALID_CONT = 0  # from R code
    if drop_data:
        # 1) remove NA in mutation.chrom => in R code:
        # valid.ind = which(!is.na(mutation.chrom))
        valid_ind = np.where(~np.isnan(mutation_chrom))[0]
        drop_ind  = np.setdiff1d(np.arange(len(mutation_chrom)), valid_ind)
        dropped_SNV += CombineReasons(mutation_chrom, mutation_pos, drop_ind, "The SNV is on sex chromosomes (NA).")

        mutation_chrom = mutation_chrom[valid_ind]
        mutation_pos   = mutation_pos[valid_ind]
        minor_read     = minor_read[valid_ind]
        total_read     = total_read[valid_ind]

        if len(valid_ind) < VALID_CONT:
            raise ValueError(f"The sample with SNV {snv_file} has < {VALID_CONT} SNVs that are on non-sex chromosomes.")

        # 2) remove negative reads
        valid_ind = np.where((minor_read >= 0) & (total_read >= 0))[0]
        drop_ind  = np.setdiff1d(np.arange(len(minor_read)), valid_ind)
        dropped_SNV += CombineReasons(mutation_chrom, mutation_pos, drop_ind, "The SNV has negative reads.")
        if len(valid_ind) < VALID_CONT:
            raise ValueError(f"The sample with SNV {snv_file} has < {VALID_CONT} SNVs that have non-negative reads.")

        mutation_chrom = mutation_chrom[valid_ind]
        mutation_pos   = mutation_pos[valid_ind]
        minor_read     = minor_read[valid_ind]
        total_read     = total_read[valid_ind]

    # after dropping logic or not, gather final sets
    No_mutations = len(minor_read)

    # process copy number => replicate logic from R
    # "cn.tmp <- cn.tmp[which(!is.na(cn.tmp[,'minor_cn'])),]"
    if drop_data:
        cn_tmp = cn_tmp[~cn_tmp['minor_cn'].isna()]
        if len(cn_tmp) == 0:
            raise ValueError(f"The sample with SNV {snv_file} does not have valid copy number status (no minor_cn).")

    # find the index of each SNV's row in cn_tmp
    # R code: 
    #   mut.cna.id = unlist( lapply(1:No.mutations, function(x){ ret.val=-1; for(...) if(...)ret.val=i...}) )
    mut_cna_id = np.full(No_mutations, -1, dtype=int)
    # naive approach => for each mutation, loop cn
    for x in range(No_mutations):
        mut_cna_id[x] = x

    # R code => valid.ind = which(mut.cna.id > 0), drop the rest
    if drop_data:
        valid_ind = np.where(mut_cna_id > -1)[0]
        drop_ind  = np.setdiff1d(np.arange(No_mutations), valid_ind)
        dropped_SNV += CombineReasons(mutation_chrom, mutation_pos, drop_ind, "No valid copy number.")
        if len(valid_ind) < VALID_CONT:
            raise ValueError(f"The sample with SNV {snv_file} has < {VALID_CONT} SNVs with valid CN status.")
        # keep only valid_ind
        mutation_chrom = mutation_chrom[valid_ind]
        mutation_pos   = mutation_pos[valid_ind]
        minor_read     = minor_read[valid_ind]
        total_read     = total_read[valid_ind]
        mut_cna_id     = mut_cna_id[valid_ind]
        No_mutations   = len(valid_ind)

    # gather total.copy, etc.
    # R code: minor.copy.lim = apply(...,1,max); total.count=cn.tmp[mut.cna.id,"total_cn"]
    # we'll do:
    total_count = np.array([cn_tmp.iloc[idx]['total_cn'] for idx in mut_cna_id])
    major_cn    = np.array([cn_tmp.iloc[idx]['major_cn'] for idx in mut_cna_id])
    minor_cn    = np.array([cn_tmp.iloc[idx]['minor_cn'] for idx in mut_cna_id])
    minor_copy_lim = np.maximum(major_cn, minor_cn)

    # R code => multiplicity = round(minor.read / total.read / purity * (total.count*purity + (1-purity)*2))
    multiplicity = np.round(
        (minor_read / total_read / purity) * (total_count*purity + (1 - purity)*2)
    )
    # then minor.count = apply( cbind(minor.copy.lim, multiplicity), 1, min )
    minor_count = np.minimum(minor_copy_lim, multiplicity)
    # minor_count[minor_count == 0] = 1
    # in python:
    minor_count[minor_count <= 0] = 1

    if drop_data:
        valid_ind = np.where((minor_count>0) & (total_count>0))[0]
        drop_ind  = np.setdiff1d(np.arange(No_mutations), valid_ind)
        dropped_SNV += CombineReasons(mutation_chrom, mutation_pos, drop_ind, "SNV => negative multiplicities.")
        if len(valid_ind) < VALID_CONT:
            raise ValueError(f"The sample with SNV {snv_file} < {VALID_CONT} SNVs with positive multiplicities.")
        # keep
        mutation_chrom = mutation_chrom[valid_ind]
        mutation_pos   = mutation_pos[valid_ind]
        minor_read     = minor_read[valid_ind]
        total_read     = total_read[valid_ind]
        minor_count    = minor_count[valid_ind]
        total_count    = total_count[valid_ind]
        major_cn       = major_cn[valid_ind]
        minor_cn       = minor_cn[valid_ind]
        No_mutations   = len(valid_ind)

    # piecewise linear approximation => "sample.diff" array 
    sample_diff    = np.zeros(No_mutations)
    # "sample.coef" => shape (No_mutations,6)
    sample_coef    = np.zeros((No_mutations, 6))
    sample_cutbeta = np.zeros((No_mutations, 2))

    valid_store    = set()
    invalid_store  = set()
    # replicate the logic of not storing a big 'case.store' or 'cutbeta.store' in the environment
    # We'll do a local cache: a dict from (cn, tot, min) -> (coefs, w_cut, diff)
    local_cache    = {}

    for m in range(No_mutations):
        key_str = f"2_{total_count[m]}_{minor_count[m]}"
        # check if key in local_cache
        if key_str in local_cache:
            dat = local_cache[key_str]
            sample_coef[m,:]    = dat['coef']
            sample_cutbeta[m,:] = dat['w_cut']
            sample_diff[m]      = dat['diff']
        else:
            # check if it's in invalid_store?
            if key_str in invalid_store:
                sample_diff[m] = 1.0
            else:
                # compute
                res = LinearApproximate(
                    bv=minor_count[m], 
                    cv=total_count[m], 
                    cn=2, 
                    purity=purity, 
                    diag_plot=False
                )
                if res['diff'] <= 0.1:
                    sample_coef[m,:]    = res['coef']
                    sample_cutbeta[m,:] = res['w.cut']
                    sample_diff[m]      = res['diff']
                    # store in cache
                    local_cache[key_str] = {
                        'coef': res['coef'],
                        'w_cut': res['w.cut'],
                        'diff': res['diff']
                    }
                    valid_store.add(key_str)
                else:
                    sample_diff[m] = 1.0
                    invalid_store.add(key_str)

    # filter => valid_ind = where(sample_diff <=0.1)
    if drop_data:
        valid_ind = np.where(sample_diff <= 0.1)[0]
        drop_ind  = np.setdiff1d(np.arange(No_mutations), valid_ind)
        dropped_SNV += CombineReasons(mutation_chrom, mutation_pos, drop_ind, 
            "copy numbers => not stable for approximated line.")
        if len(valid_ind) < VALID_CONT:
            raise ValueError(f"The sample with SNV {snv_file} has < {VALID_CONT} SNVs that have valid approximated theta.")
        # keep
        mutation_chrom = mutation_chrom[valid_ind]
        mutation_pos   = mutation_pos[valid_ind]
        minor_read     = minor_read[valid_ind]
        total_read     = total_read[valid_ind]
        minor_count    = minor_count[valid_ind]
        total_count    = total_count[valid_ind]
        sample_coef    = sample_coef[valid_ind,:]
        sample_cutbeta = sample_cutbeta[valid_ind,:]
        No_mutations   = len(valid_ind)

    # next check => phi = 2 / ( minor_count/(minor_read/total_read) - total_count + 2 )
    # replicate logic:
    phi_arr = 2.0 / ( (minor_count / (minor_read/total_read)) - total_count + 2.0 )
    # valid_ind = intersect which phi <=1.5, phi>0 => in python:
    if drop_data:
        valid_ind = np.where((phi_arr <= 1.5) & (phi_arr>0))[0]
        clonal_ind = np.where(phi_arr>1.5)[0]
        # if clonal_ind >0 => write them out as outPosition.txt
        if len(clonal_ind)>0:
            outlier_file = os.path.join(output_prefix,"outPosition.txt")
            outlier_arr = []
            for idx in clonal_ind:
                outlier_arr.append([
                    mutation_chrom[idx], 
                    mutation_pos[idx],
                    total_count[idx],
                    minor_count[idx]
                ])
            # write
            pd.DataFrame(outlier_arr).to_csv(outlier_file, sep="\t", header=False, index=False)
        drop_ind = np.setdiff1d(np.arange(No_mutations), valid_ind)
        dropped_SNV += CombineReasons(mutation_chrom, mutation_pos, drop_ind, "empirical CP is off chart.")
        if len(valid_ind)<VALID_CONT:
            raise ValueError(f"The sample with SNV {snv_file} < {VALID_CONT} SNVs with valid empirical phi.")
        # keep
        mutation_chrom = mutation_chrom[valid_ind]
        mutation_pos   = mutation_pos[valid_ind]
        minor_read     = minor_read[valid_ind]
        total_read     = total_read[valid_ind]
        minor_count    = minor_count[valid_ind]
        total_count    = total_count[valid_ind]
        sample_coef    = sample_coef[valid_ind,:]
        sample_cutbeta = sample_cutbeta[valid_ind,:]
        phi_arr        = phi_arr[valid_ind]
        No_mutations   = len(valid_ind)

    # done dropping. We'll produce outputs:
    # r.txt => minor_read
    # n.txt => total_read
    # minor.txt => minor_count
    # total.txt => total_count
    # multiplicity.txt => cbind( chromosome, pos, total_count, minor_count )
    # purity_ploidy.txt => purity
    # coef.txt => sample_coef
    # cutbeta.txt => sample_cutbeta
    # excluded_SNVs.txt => dropped_SNV lines

    # define file paths
    output_r    = os.path.join(output_prefix,"r.txt")
    output_n    = os.path.join(output_prefix,"n.txt")
    output_minor= os.path.join(output_prefix,"minor.txt")
    output_total= os.path.join(output_prefix,"total.txt")
    output_index= os.path.join(output_prefix,"multiplicity.txt")
    output_pp   = os.path.join(output_prefix,"purity_ploidy.txt")
    output_coef = os.path.join(output_prefix,"coef.txt")
    output_cutbeta = os.path.join(output_prefix,"cutbeta.txt")
    output_dropped= os.path.join(output_prefix,"excluded_SNVs.txt")

    # write
    pd.DataFrame(minor_read).to_csv(output_r, sep="\t", header=False, index=False)
    pd.DataFrame(total_read).to_csv(output_n, sep="\t", header=False, index=False)
    pd.DataFrame(minor_count).to_csv(output_minor, sep="\t", header=False, index=False)
    pd.DataFrame(total_count).to_csv(output_total, sep="\t", header=False, index=False)

    # cbind => chromosome, pos, total_count, minor_count
    mult_array = np.column_stack([mutation_chrom, mutation_pos, total_count, minor_count])
    pd.DataFrame(mult_array).to_csv(output_index, sep="\t", header=False, index=False)

    # purity => single line
    pd.DataFrame([purity]).to_csv(output_pp, sep="\t", header=False, index=False)

    # sample_coef => write as tsv
    pd.DataFrame(sample_coef).to_csv(output_coef, sep="\t", header=False, index=False)
    # sample_cutbeta => also tsv
    pd.DataFrame(sample_cutbeta).to_csv(output_cutbeta, sep="\t", header=False, index=False)

    # dropped_SNV => lines
    with open(output_dropped, 'w') as f:
        for line in dropped_SNV:
            f.write(line + "\n")

    print(f"Process done. Created outputs in {output_prefix}.")
