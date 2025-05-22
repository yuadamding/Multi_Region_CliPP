from clipp2.preprocess import *
from clipp2.core import *
import argparse
import numpy as np
import pandas as pd
import os
import sys
import ray

def run(args):
    root           = args.input_dir
    subsample_rate = args.subsample_rate
    outdir         = args.output_dir
    dtype          = args.dtype
    # ► EARLY SANITY CHECK ◀
    if not os.path.isdir(root):
        print(f"Error: input directory not found: {root!r}")
        sys.exit(1)

    df = combine_patient_samples(root)

    df = add_to_df(df)

    dic = build_tensor_arrays(df)
    r, n, minor, total = dic['r'], dic['n'], dic['minor'], dic['total']
    pur_arr, coef_list  = dic['pur_arr'], dic['coef_list']

    with np.errstate(divide='ignore', invalid='ignore'):
        frac    = r / (n + 1e-12)
        phi_hat = frac * ((2 - 2*pur_arr) + (pur_arr * total) / (minor + 1e-12))

    if subsample_rate < 1.0:
        M    = r.shape[0]
        keep = int(M * subsample_rate)
        idxs = np.random.choice(M, keep, replace=False)
        r, n, minor, total = [arr[idxs] for arr in (r, n, minor, total)]
        coef_list          = [mat[idxs] for mat in coef_list]
        for arr in (r, n, minor, total):
            arr[np.isnan(arr)] = 1
    else:
        idxs = range(r.shape[0])

    Lambda_lst = np.linspace(0.01, 0.5, 20)
    for Lambda in Lambda_lst:
        res = clipp2(
            r, n, minor, total, pur_arr, coef_list,
            Lambda=Lambda, device='cuda', dtype=dtype
        )
        labels, phi_cent, aic, bic = res['label'], res['phi'], res['aic'], res['bic']

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
                    aic    = aic
                    bic    = bic
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
                    'aic':              aic,
                    'bic':              bic,
                    'dropped':          dropped
                })

        result_df = pd.DataFrame(rows)
        os.makedirs(outdir, exist_ok=True)
        suboutput = os.path.join(outdir, root)
        os.makedirs(suboutput, exist_ok=True)
        result_df.to_csv(f'{suboutput}/lambda_{Lambda}.tsv', sep='\t', index=False)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Multi-sample CLiPP2 pipeline'
    )
    parser.add_argument('--input_dir',       required=True,
                        help="Root directory of processed samples")
    parser.add_argument('--output_dir',       default='output',
                        help="Root directory of processed samples")
    parser.add_argument('--subsample_rate', type=float, default=1.0,
                        help="Fraction of mutations to keep (for speed/debug)")
    parser.add_argument('--dtype', default='float16')
    args = parser.parse_args()
    run(args)