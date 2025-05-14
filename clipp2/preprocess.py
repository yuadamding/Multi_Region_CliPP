import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_snv_cna_purity(root_dir):
    """
    Traverse each subdirectory of `root_dir`, loading:
      - the first '*snv.txt'  as a DataFrame
      - the first '*cna.txt'  as a DataFrame
      - the first '*purity.txt' as a float
    Returns three dicts: snv_dict, cna_dict, purity_dict keyed by subdir name.
    """
    snv_dict    = {}
    cna_dict    = {}
    purity_dict = {}

    for name in os.listdir(root_dir):
        subdir = os.path.join(root_dir, name)
        if not os.path.isdir(subdir):
            continue

        # look *inside* this subdir for each file type
        snv_files    = glob.glob(os.path.join(subdir, '*snv.txt'))
        cna_files    = glob.glob(os.path.join(subdir, '*cna.txt'))
        purity_files = glob.glob(os.path.join(subdir, '*purity.txt'))

        if snv_files:
            snv_dict[name] = pd.read_csv(snv_files[0], sep='\t', comment='#')
        if cna_files:
            cna_dict[name] = pd.read_csv(cna_files[0], sep='\t', comment='#')

        # --- purity, per‐sample ---
        if purity_files:
            purity_path = purity_files[0]
            with open(purity_path, 'r') as f:
                txt = f.read().strip()
            try:
                purity_dict[name] = float(txt)
            except ValueError:
                raise ValueError(f"Could not parse purity value '{txt}' in {purity_path}")
        else:
            # No purity file in this subdir: set None or raise
            purity_dict[name] = 0.0

    return snv_dict, cna_dict, purity_dict


def combine_snv_cna(snv_df: pd.DataFrame,
                    cna_df: pd.DataFrame,
                    purity: float,
                    default_cn: tuple = (1, 1, 2)
                   ) -> pd.DataFrame:
    """
    Merge an SNV DataFrame and a CNA DataFrame, carrying along a single purity value.

    Parameters
    ----------
    snv_df : pd.DataFrame
        Columns: ['chromosome_index','position','alt_count','ref_count']
    cna_df : pd.DataFrame
        Columns: ['chromosome_index','start_position','end_position',
                  'major_cn','minor_cn','total_cn']
    purity : float
        Tumor purity to annotate every SNV.
    default_cn : (int,int,int), optional
        Fallback (major, minor, total) copy‐number if no CNA segment contains the SNV.

    Returns
    -------
    merged : pd.DataFrame
        Columns: ['chromosome_index','position','alt_count','ref_count',
                  'major_cn','minor_cn','total_cn','purity']
    """
    # bucket CNA segments by chromosome for fast lookup
    cna_groups = {}
    for _, row in cna_df.iterrows():
        chrom = row['chromosome_index']
        cna_groups.setdefault(chrom, []).append(row)

    records = []
    for _, snv in snv_df.iterrows():
        chrom = snv['chromosome_index']
        pos   = snv['position']

        maj, mino, tot = default_cn
        for seg in cna_groups.get(chrom, []):
            if seg['start_position'] <= pos <= seg['end_position']:
                maj, mino, tot = seg['major_cn'], seg['minor_cn'], seg['total_cn']
                break

        records.append({
            'chromosome_index': chrom,
            'position':         pos,
            'alt_count':        snv['alt_count'],
            'ref_count':        snv['ref_count'],
            'major_cn':         maj,
            'minor_cn':         mino,
            'total_cn':         tot,
            'purity':           purity
        })

    return pd.DataFrame.from_records(records)

def combine_patient_samples(root):
    # === Step 1: load everything ===
    snv_dict, cna_dict, purity_dict = load_snv_cna_purity(root)

    # === Step 2: per‐sample merge ===
    merged_dict = {}
    for sample, snv_df in snv_dict.items():
        cna_df = cna_dict.get(
            sample,
            pd.DataFrame(columns=[
                'chromosome_index','start_position','end_position',
                'major_cn','minor_cn','total_cn'
            ])
        )
        purity = purity_dict.get(sample, 1.0)
        merged_dict[sample] = combine_snv_cna(snv_df, cna_df, purity)

    # === Step 3: get the union of all (chrom, pos) pairs ===
    all_pairs = pd.concat(
        [df[['chromosome_index','position']] for df in merged_dict.values()],
        ignore_index=True
    ).drop_duplicates()

    # === Step 4: re‐index each sample to that union ===
    filled_dfs = []
    for sample, df in merged_dict.items():
        # drop purity/sample so they don't interfere with merge
        df_core = df.drop(columns=['purity','sample'], errors='ignore')

        # ensure every pair is present
        full = all_pairs.merge(
            df_core,
            on=['chromosome_index','position'],
            how='left'
        )

        # fill zeros for the count/CN columns
        for col in ['ref_count','major_cn','minor_cn','total_cn']:
            full[col] = full[col].fillna(1).astype(int)
        full['alt_count'] = full['alt_count'].fillna(0).astype(int)
        # re‐add purity & sample
        full['purity'] = purity_dict.get(sample, 1.0)
        full['sample'] = sample

        filled_dfs.append(full)


    # === Step 5: concatenate into one final DataFrame ===
    final_df = pd.concat(filled_dfs, ignore_index=True)
    final_df = final_df.sort_values(['chromosome_index', 'position']).reset_index(drop=True)
    
    # --- ENSURE minor_cn ≤ major_cn ---
    minor = final_df[['minor_cn','major_cn']].min(axis=1)
    major = final_df[['minor_cn','major_cn']].max(axis=1)
    final_df['minor_cn'] = minor.astype(int)
    final_df['major_cn'] = major.astype(int)
    
    return final_df

def add_to_df(df):
    # pull out vectors
    alt = df['alt_count'].to_numpy()
    ref = df['ref_count'].to_numpy()
    n   = alt + ref
    rho = df['purity'].fillna(0).to_numpy()
    tot = df['total_cn'].to_numpy()
    cN  = 2

    # ensure minor_cn ≤ major_cn
    maj = df['major_cn'].to_numpy()
    mino = df['minor_cn'].to_numpy()
    major = np.maximum(maj, mino)
    minor = np.minimum(maj, mino)

    # multiplicity: safe divide
    denom = n * rho
    ratio = np.zeros_like(alt, dtype=float)
    np.divide(alt, denom, out=ratio, where=denom>0)

    raw = ratio * (rho * tot + (1 - rho) * cN)
    # no NaNs/inf because ratio is 0 where denom==0
    mult = np.minimum(np.round(raw).astype(int), np.maximum(major, minor))
    mult = np.where(mult == 0, 1, mult)

    # linear‐approx parameters vectorized
    w    = np.array([-4.0, -1.8, 1.8, 4.0])
    expw = np.exp(w)                   # shape (4,)
    onep = 1 + expw                    # shape (4,)

    # build theta denominator (4 × N)
    theta_den = onep[:, None] * (cN*(1 - rho)[None, :] + tot[None, :] * rho[None, :])
    # compute act[k,i] = expw[k] * mult[i] / theta_den[k,i]
    act = expw[:, None] * mult[None, :] / theta_den

    # piecewise slopes & intercepts (N‐length each)
    a1 = (act[1] - act[0]) / (w[1] - w[0])
    b1 = act[0] - a1 * w[0]
    a2 = (act[2] - act[1]) / (w[2] - w[1])
    b2 = act[1] - a2 * w[1]
    a3 = (act[3] - act[2]) / (w[3] - w[2])
    b3 = act[2] - a3 * w[2]

    # stack into (N,6)
    params = np.stack([a1, b1, a2, b2, a3, b3], axis=1)
    breaks = np.array([-1.8, 1.8])

    # write back into a new DataFrame
    out = df.copy()
    out['minor_cn']          = minor.astype(int)
    out['major_cn']          = major.astype(int)
    out['multiplicity']      = mult
    out['lin_approx_params'] = list(params)
    out['lin_breakpoints']   = [breaks] * len(df)
    return out


def build_tensor_arrays(df):
    # 1) Determine sample order and unique mutation pairs
    samples = sorted(df['sample'].unique())
    pairs = (
        df[['chromosome_index','position']]
        .drop_duplicates()
        .sort_values(['chromosome_index','position'])
        .reset_index(drop=True)
    )
    No = len(pairs)
    M  = len(samples)

    # 2) Precompute n = alt + ref
    df = df.copy()
    df['n'] = df['alt_count'] + df['ref_count']

    # 3) Pivot each metric into a (No × M) table, then convert to numpy
    index = ['chromosome_index','position']
    r_df     = df.pivot(index=index, columns='sample', values='alt_count')
    n_df     = df.pivot(index=index, columns='sample', values='n')
    minor_df = df.pivot(index=index, columns='sample', values='multiplicity')
    total_df = df.pivot(index=index, columns='sample', values='total_cn')

    # Reindex to ensure the same ordering
    idx = pd.MultiIndex.from_frame(pairs)
    r_df     = r_df.reindex(index=idx, columns=samples)
    n_df     = n_df.reindex(index=idx, columns=samples)
    minor_df = minor_df.reindex(index=idx, columns=samples)
    total_df = total_df.reindex(index=idx, columns=samples)

    # convert to numpy and cast dtypes
    r     = r_df.to_numpy(dtype=int)
    n     = n_df.to_numpy(dtype=int)
    minor = minor_df.to_numpy(dtype=int)
    total = total_df.to_numpy(dtype=int)

    # 4) Purity array for each sample (float)
    pur_arr = np.array([
        df.loc[df['sample']==s, 'purity'].iat[0]
        for s in samples
    ], dtype=float)

    # 5) Build coef_list into a (No, M, 6) float array
    coef_df = df.pivot(index=index, columns='sample', values='lin_approx_params')
    coef_df = coef_df.reindex(index=idx, columns=samples)

    coef_arr = np.empty((No, M, 6), dtype=float)
    for i in range(No):
        for j in range(M):
            coef_arr[i, j, :] = coef_df.iat[i, j]

    coef_list = [ coef_arr[:, j, :] for j in range(M) ]

    # === Sanity checks ===
    assert isinstance(pairs, pd.DataFrame) and pairs.shape == (No, 2)
    assert isinstance(samples, list) and len(samples) == M

    assert isinstance(r, np.ndarray) and r.dtype == np.int64
    assert r.shape == (No, M)
    assert n.shape == (No, M) and n.dtype == np.int64
    assert minor.shape == (No, M) and minor.dtype == np.int64
    assert total.shape == (No, M) and total.dtype == np.int64

    assert isinstance(pur_arr, np.ndarray) and pur_arr.dtype == np.float64
    assert pur_arr.shape == (M,)

    return {
        'pairs': pairs,
        'samples': samples,
        'r': r,
        'n': n,
        'minor': minor,
        'total': total,
        'pur_arr': pur_arr,
        'coef_list': coef_list
    }