from clipp2.preprocess import *
from clipp2.core import *
import argparse
import numpy as np
import pandas as pd
import os
import sys

def run(args):
    root           = args.input_dir
    subsample_rate = args.subsample_rate
    Lambda         = args.Lambda

    # ► EARLY SANITY CHECK ◀
    if not os.path.isdir(root):
        print(f"Error: input directory not found: {root!r}")
        sys.exit(1)

    print(f"[1/7] Loading and combining patient samples from: {root}")
    df = combine_patient_samples(root)

    print("[2/7] Adding computed columns to DataFrame")
    df = add_to_df(df)

    print("[3/7] Building tensor arrays (r, n, minor, total, purity, coefficients)")
    dic = build_tensor_arrays(df)
    r, n, minor, total = dic['r'], dic['n'], dic['minor'], dic['total']
    pur_arr, coef_list  = dic['pur_arr'], dic['coef_list']

    print("[4/7] Computing φ̂ (phi_hat) with error‐safe division")
    with np.errstate(divide='ignore', invalid='ignore'):
        frac    = r / (n + 1e-12)
        phi_hat = frac * ((2 - 2*pur_arr) + (pur_arr * total) / (minor + 1e-12))

    if subsample_rate < 1.0:
        print(f"[5/7] Subsampling mutations at rate {subsample_rate:.2f}")
        M    = r.shape[0]
        keep = int(M * subsample_rate)
        idxs = np.random.choice(M, keep, replace=False)
        r, n, minor, total = [arr[idxs] for arr in (r, n, minor, total)]
        coef_list          = [mat[idxs] for mat in coef_list]
        for arr in (r, n, minor, total):
            arr[np.isnan(arr)] = 1
    else:
        print("[5/7] No subsampling applied")
        idxs = range(r.shape[0])

    print(f"[6/7] Running CLiPP2 with Lambda={Lambda}, device='cuda', dtype='float64'")
    res = clipp2(
        r, n, minor, total, pur_arr, coef_list,
        Lambda=Lambda, device='cuda', dtype='float64'
    )
    labels, phi_cent = res['label'], res['phi']

    print("[7/7] Building result DataFrame and writing to 'res.tsv'")
    pairs   = dic['pairs']
    samples = dic['samples']
    orig2sub = {orig: sub for sub, orig in enumerate(idxs)}

    rows = []
    No, M = phi_hat.shape
    for mut_idx in range(No):
        chrom, pos = pairs.iloc[mut_idx]
        for j, region in enumerate(samples):
            phi_hat_val = phi_hat[mut_idx, j]
            if mut_idx in orig2sub:
                sub_i  = orig2sub[mut_idx]
                lab    = labels[sub_i]
                phi    = phi_cent[sub_i, j]
                dropped= 0
            else:
                lab    = np.nan
                phi    = np.nan
                dropped= 1

            rows.append({
                'chromosome_index': chrom,
                'position':         pos,
                'region':           region,
                'label':            lab,
                'phi':              phi,
                'phi_hat':          phi_hat_val,
                'dropped':          dropped
            })

    result_df = pd.DataFrame(rows)
    result_df.to_csv('res.tsv', sep='\t', index=False)
    print("Done. Output written to res.tsv")

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Multi-sample CLiPP2 pipeline'
    )
    parser.add_argument('--input_dir',       required=True,
                        help="Root directory of processed samples")
    parser.add_argument('--Lambda', type=float, default=0.1,
                        help="Regularization strength for CLiPP2")
    parser.add_argument('--subsample_rate', type=float, default=1.0,
                        help="Fraction of mutations to keep (for speed/debug)")
    args = parser.parse_args()
    run(args)
