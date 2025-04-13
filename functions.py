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
            to_insert[c] = np.nan

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