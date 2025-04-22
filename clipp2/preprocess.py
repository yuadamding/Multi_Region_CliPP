import os
import sys
import numpy as np
import pandas as pd


def process_files(snvfile, cnafile, purityfile):
    """
    Process SNV, CNA, and purity files into a DataFrame.
    - snvfile: path to SNV with ['chromosome_index','position','alt_count','ref_count']
    - cnafile: path to CNA with ['chromosome_index','start_position','end_position','major_cn','minor_cn','total_cn']
    - purityfile: single-value file with tumour purity
    Returns DataFrame with columns:
      ['chromosome_index','position','ref_counts','alt_counts',
       'major_cn','minor_cn','normal_cn','tumour_purity','start_position','end_position']
    """
    # Load input files
    df_snv = pd.read_csv(snvfile, sep='\t')
    df_cna = pd.read_csv(cnafile, sep='\t')
    with open(purityfile, 'r') as f:
        purity = float(f.read().strip())

    # Validate required columns
    req_snv = {'chromosome_index','position','alt_count','ref_count'}
    req_cna = {'chromosome_index','start_position','end_position','major_cn','minor_cn','total_cn'}
    missing = req_snv - set(df_snv.columns)
    assert not missing, f"Missing SNV columns: {missing}"
    missing = req_cna - set(df_cna.columns)
    assert not missing, f"Missing CNA columns: {missing}"
    assert 0 <= purity <= 1, f"Purity must be in [0,1], got {purity}"

    # Map each SNV to its CNA segment
    major_cn = []
    minor_cn = []
    start_pos = []
    end_pos = []
    for _, row in df_snv.iterrows():
        seg = df_cna[
            (df_cna['chromosome_index'] == row['chromosome_index']) &
            (df_cna['start_position'] <= row['position']) &
            (df_cna['end_position'] >= row['position'])
        ]
        if not seg.empty:
            major_cn.append(int(seg['major_cn'].iloc[0]))
            minor_cn.append(int(seg['minor_cn'].iloc[0]))
            start_pos.append(int(seg['start_position'].iloc[0]))
            end_pos.append(int(seg['end_position'].iloc[0]))
        else:
            # default copy number if no segment matches
            major_cn.append(1)
            minor_cn.append(1)
            start_pos.append(int(row['position'] - 1))
            end_pos.append(int(row['position'] + 1))

    # Construct output DataFrame
    df_out = pd.DataFrame({
        'chromosome_index':  df_snv['chromosome_index'],
        'position':        df_snv['position'],
        'ref_counts':      df_snv['ref_count'],
        'alt_counts':      df_snv['alt_count'],
        'major_cn':        major_cn,
        'minor_cn':        minor_cn,
        'normal_cn':       2,
        'tumour_purity':   purity,
        'start_position':  start_pos,
        'end_position':    end_pos
    })
    return df_out


def insert_distinct_rows_multi(df_list):
    """
    Ensure each DataFrame in df_list contains all mutation loci observed across the list.
    Inserts missing rows with default copy numbers and purity.
    Returns list of updated DataFrames.
    """
    # Collect all unique keys
    keys = set()
    for df in df_list:
        keys.update(zip(df['chromosome_index'], df['position']))

    # Insert missing rows into each DataFrame
    updated = []
    for df in df_list:
        present = set(zip(df['chromosome_index'], df['position']))
        missing = [k for k in keys if k not in present]
        if missing:
            to_add = pd.DataFrame(missing, columns=['chromosome_index','position'])
            to_add['ref_counts']    = 0
            to_add['alt_counts']    = 0
            to_add['major_cn']      = 1
            to_add['minor_cn']      = 1
            to_add['normal_cn']     = 2
            to_add['tumour_purity'] = df['tumour_purity'].iat[0]
            to_add['start_position']= to_add['position'] - 1
            to_add['end_position']  = to_add['position'] + 1
            to_add = to_add[df.columns]
            df = pd.concat([df, to_add], ignore_index=True)
        updated.append(df)
    return updated


def export_snv_cna_and_purity(df, dir, snv_path, cna_path, purity_path):
    """
    Export standardized SNV, CNA, and purity files from df.
    """
    os.makedirs(dir, exist_ok=True)
    # SNV file
    snv_df = df[['chromosome_index','position','alt_counts','ref_counts']].rename(
        columns={
            'chromosome_index': 'chromosome_index',
            'alt_counts':     'alt_count',
            'ref_counts':     'ref_count'
        }
    )
    snv_df.to_csv(os.path.join(dir, snv_path), sep='\t', index=False)
    # CNA file
    cna_df = df[['chromosome_index','start_position','end_position','major_cn','minor_cn']].rename(
        columns={'chromosome_index': 'chromosome_index'}
    )
    # Include total_cn 
    cna_df['total_cn'] = cna_df['major_cn'] + cna_df['minor_cn']
    cna_df.to_csv(os.path.join(dir, cna_path), sep='\t', index=False)
    # Purity file
    purity_val = df['tumour_purity'].iat[0]
    with open(os.path.join(dir, purity_path), 'w') as f:
        f.write(f"{purity_val}\n")


def combine(chrom_list, pos_list, indices, reason):
    """
    Build list of "chromosome<TAB>position<TAB>reason" for each index.
    """
    return [f"{chrom_list[i]}\t{pos_list[i]}\t{reason}" for i in indices]


def preprocess(
    snv_file, cn_file, purity_file, sample_id, output_prefix,
    drop_data=True
):
    """
    Full CLiPP2 preprocessing for a single sample, applying all filters,
    computing multiplicity, performing piecewise linear approximation,
    calculating phi, and writing output files.
    """
    # 1) Validate input paths
    for path in (snv_file, cn_file, purity_file):
        if not os.path.exists(path):
            sys.exit(f"Missing file: {path}")
    os.makedirs(output_prefix, exist_ok=True)

    # 2) Define helper functions for approximation
    def theta(w, bv, cv, cn, pur):
        return (np.exp(w)*bv)/((1+np.exp(w))*cn*(1-pur)+(1+np.exp(w))*cv*pur)

    def linear_approximation(bv, cv, cn, pur, diag_plot=False):
        w = np.arange(-4.0, 4.1, 0.1)
        act = theta(w, bv, cv, cn, pur)
        i = np.argmin(np.abs(w + 1.8))
        j = np.argmin(np.abs(w - 1.8))
        # slopes & intercepts
        a1 = (act[i] - act[0]) / (w[i] - w[0]); b1 = act[0] - a1*w[0]
        a2 = (act[j] - act[i]) / (w[j] - w[i]); b2 = act[i] - a2*w[i]
        a3 = (act[-1] - act[j]) / (w[-1] - w[j]); b3 = act[-1] - a3*w[-1]
        # piecewise values
        appr = np.zeros_like(w)
        appr[:i+1]   = a1*w[:i+1]   + b1
        appr[i+1:j+1] = a2*w[i+1:j+1] + b2
        appr[j+1:]   = a3*w[j+1:]   + b3
        diff = np.max(np.abs(act - appr))
        coef = np.array([a1, b1, a2, b2, a3, b3])
        cuts = np.array([w[i], w[j]])
        return diff, coef, cuts

    # 3) Load data
    purity = float(pd.read_csv(purity_file, header=None, sep='\t').iloc[0, 0])
    snv_df = pd.read_csv(snv_file, sep='\t')
    cn_df  = pd.read_csv(cn_file, sep='\t').dropna(subset=['minor_cn'])

    # Extract arrays
    chrom = snv_df['chromosome_index'].astype(float).values
    pos   = snv_df['position'].astype(float).values
    alt   = snv_df['alt_count'].astype(float).values
    ref   = snv_df['ref_count'].astype(float).values
    total_read = alt + ref
    cn_chr   = cn_df['chromosome_index'].astype(float).values
    cn_start = cn_df['start_position'].astype(float).values
    cn_end   = cn_df['end_position'].astype(float).values
    cn_minor = cn_df['minor_cn'].astype(float).values
    cn_major = cn_df['major_cn'].astype(float).values
    cn_total = cn_df['total_cn'].astype(float).values

    dropped = []

    # FILTER 1: autosomes only
    valid = ~np.isnan(chrom)
    dropped += combine(chrom, pos, np.where(~valid)[0], "SNV on sex chromosome or missing")
    if drop_data:
        chrom, pos, alt, ref, total_read = chrom[valid], pos[valid], alt[valid], ref[valid], total_read[valid]

    # FILTER 2: non-negative reads
    valid = (alt >= 0) & (total_read >= 0)
    dropped += combine(chrom, pos, np.where(~valid)[0], "Negative read counts")
    if drop_data:
        chrom, pos, alt, ref, total_read = chrom[valid], pos[valid], alt[valid], ref[valid], total_read[valid]
    N = len(chrom)

    # Map SNVs to CNA segments
    seg_ids = np.full(N, -1, dtype=int)
    for i in range(N):
        hits = np.where(
            (cn_chr == chrom[i]) &
            (cn_start <= pos[i]) &
            (cn_end >= pos[i])
        )[0]
        if hits.size > 0:
            seg_ids[i] = hits[0]
    # FILTER 3: valid CNA segment
    valid = seg_ids >= 0
    dropped += combine(chrom, pos, np.where(~valid)[0], "No matching CNA segment")
    if drop_data:
        mask = valid
        chrom, pos, alt, ref, total_read, seg_ids = (
            chrom[mask], pos[mask], alt[mask], ref[mask], total_read[mask], seg_ids[mask]
        )
    N = len(chrom)
    seg_total = cn_total[seg_ids]
    seg_minor = np.maximum(cn_minor[seg_ids], cn_major[seg_ids])

    # multiplicity calculation
    multiplicity = np.round(
        (alt / total_read) / purity * (seg_total * purity + (1 - purity) * 2)
    )
    minor_count = np.minimum(seg_minor, multiplicity)
    minor_count[minor_count < 1] = 1

    # FILTER 4: positive minor and total CN
    valid = (minor_count > 0) & (seg_total > 0)
    dropped += combine(chrom, pos, np.where(~valid)[0], "Invalid multiplicity or CN")
    if drop_data:
        mask = valid
        chrom, pos, alt, ref, total_read, seg_total, minor_count = (
            chrom[mask], pos[mask], alt[mask], ref[mask], total_read[mask], seg_total[mask], minor_count[mask]
        )
    N = len(chrom)

    # Piecewise linear approximation for each SNV
    coefs = np.zeros((N, 6), dtype=float)
    cuts  = np.zeros((N, 2), dtype=float)
    diffs = np.zeros(N, dtype=float)
    cache = {}
    for i in range(N):
        key = (seg_total[i], minor_count[i])
        if key in cache:
            diffs[i], coefs[i], cuts[i] = cache[key]
        else:
            d, c, ct = linear_approximation(minor_count[i], seg_total[i], 2, purity)
            diffs[i], coefs[i], cuts[i] = d, c, ct
            cache[key] = (d, c, ct)
    # FILTER 5: approximation error ≤ 0.1
    valid = diffs <= 0.1
    dropped += combine(chrom, pos, np.where(~valid)[0], "Approximation error > 0.1")
    if drop_data:
        mask = valid
        chrom, pos, alt, ref, total_read, seg_total, minor_count, coefs, cuts = (
            chrom[mask], pos[mask], alt[mask], ref[mask], total_read[mask], seg_total[mask], minor_count[mask], coefs[mask], cuts[mask]
        )
    N = len(chrom)

    # Calculate phi and FILTER 6: 0 < phi ≤ 1.5
    phi = 2.0 / ((minor_count / (alt / total_read)) - seg_total + 2.0)
    out_hi = np.where(phi > 1.5)[0]
    if drop_data and out_hi.size > 0:
        pd.DataFrame(np.column_stack((chrom[out_hi], pos[out_hi], seg_total[out_hi], minor_count[out_hi]))).to_csv(
            os.path.join(output_prefix, 'outPosition.txt'), sep='\t', index=False, header=False
        )
    valid = (phi > 0) & (phi <= 1.5)
    dropped += combine(chrom, pos, np.where(~valid)[0], "Phi out of bounds")
    if drop_data:
        mask = valid
        chrom, pos, alt, ref, total_read, seg_total, minor_count, phi, coefs, cuts = (
            chrom[mask], pos[mask], alt[mask], ref[mask], total_read[mask], seg_total[mask], minor_count[mask], phi[mask], coefs[mask], cuts[mask]
        )
    # 7) Write outputs
    os.makedirs(output_prefix, exist_ok=True)
    np.savetxt(os.path.join(output_prefix, 'r.txt'), alt, fmt='%.0f', delimiter='\t')
    np.savetxt(os.path.join(output_prefix, 'n.txt'), total_read, fmt='%.0f', delimiter='\t')
    np.savetxt(os.path.join(output_prefix, 'minor.txt'), minor_count, fmt='%.0f', delimiter='\t')
    np.savetxt(os.path.join(output_prefix, 'total.txt'), seg_total, fmt='%.0f', delimiter='\t')
    np.savetxt(os.path.join(output_prefix, 'multiplicity.txt'), np.column_stack((chrom, pos, seg_total, minor_count)), fmt='%.6g', delimiter='\t')
    np.savetxt(os.path.join(output_prefix, 'purity_ploidy.txt'), [purity], fmt='%.5f', delimiter='\t')
    np.savetxt(os.path.join(output_prefix, 'coef.txt'), coefs, fmt='%.6g', delimiter='\t')
    np.savetxt(os.path.join(output_prefix, 'cutbeta.txt'), cuts, fmt='%.6g', delimiter='\t')
    with open(os.path.join(output_prefix, 'excluded_SNVs.txt'), 'w') as f:
        f.write("\n".join(dropped))
    print(f"Preprocessing complete for {sample_id}: retained {len(chrom)} SNVs.")