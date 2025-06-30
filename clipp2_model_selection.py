from clipp2.preprocess import combine_patient_samples, add_to_df, build_tensor_arrays
from clipp2.core import clipp2 

import argparse
import numpy as np
import pandas as pd
import os
import sys
import torch

def run(args):
    """
    Main execution function for the CLiPP2 pipeline.
    
    This version correctly uses the warm-start pipeline to get results for
    a full lambda path and saves a separate, vectorized output file for each lambda.
    """
    root, subsample_rate, outdir, dtype = args.input_dir, args.subsample_rate, args.output_dir, args.dtype

    if not os.path.isdir(root):
        # print(f"Error: input directory not found: {root!r}", file=sys.stderr)
        sys.exit(1)

    # print(f"1. Loading and preprocessing data from: {root}")
    df = combine_patient_samples(root)
    df = add_to_df(df)
    dic = build_tensor_arrays(df)

    r, n, minor, total = dic['r'], dic['n'], dic['minor'], dic['total']
    pur_arr, coef_list = dic['pur_arr'], dic['coef_list']

    with np.errstate(divide='ignore', invalid='ignore'):
        phi_hat = (r / (n + 1e-12)) * ((2 - 2 * pur_arr) + (pur_arr * total)) / (minor + 1e-12)
        phi_hat = np.nan_to_num(phi_hat, nan=0.0)

    No_orig, M_regions = phi_hat.shape
    
    if subsample_rate < 1.0:
        # print(f"2. Subsampling mutations to {subsample_rate * 100:.1f}%")
        keep = int(No_orig * subsample_rate)
        idxs = np.random.choice(No_orig, keep, replace=False)
        r_sub, n_sub, minor_sub, total_sub = [arr[idxs] for arr in (r, n, minor, total)]
        coef_list_sub = [mat[idxs] for mat in coef_list]
    else:
        # print("2. Using all mutations (no subsampling).")
        idxs = np.arange(No_orig)
        r_sub, n_sub, minor_sub, total_sub = r, n, minor, total
        coef_list_sub = coef_list
        
    lambda_seq = np.linspace(0.01, 0.5, 20)
    # print(f"3. Running CLiPP2 warm-start pipeline for {len(lambda_seq)} lambda values...")

    # --- SINGLE, EFFICIENT PIPELINE CALL ---
    # This call now correctly maps to the updated clipp2 function.
    list_of_results = clipp2(
        r_sub, n_sub, minor_sub, total_sub, pur_arr, coef_list_sub,
        lambda_seq=lambda_seq, # The argument name 'lambda_seq' matches.
        device='cuda' if torch.cuda.is_available() else 'cpu',
        dtype=dtype
    )
    # print("   Pipeline finished. Formatting and saving results...")

    output_base_dir = os.path.join(outdir, os.path.basename(os.path.normpath(root)))
    os.makedirs(output_base_dir, exist_ok=True)
    
    pairs = dic['pairs']
    samples = dic['samples']

    # --- LOOP OVER RESULTS (NOT THE PIPELINE) ---
    # This loop correctly consumes the list of dictionaries returned by clipp2.
    for res in list_of_results:
        lambda_val = res['lambda']

        # VECTORIZED DATAFRAME CREATION
        labels_full = np.full(No_orig, np.nan)
        phi_res_full = np.full((No_orig, M_regions), np.nan)
        dropped_full = np.ones(No_orig, dtype=int)

        labels_full[idxs] = res['label']
        phi_res_full[idxs, :] = res['phi']
        dropped_full[idxs] = 0

        data_for_df = {
            'chromosome_index': np.repeat(pairs['chromosome_index'].values, M_regions),
            'position': np.repeat(pairs['position'].values, M_regions),
            'region': np.tile(samples, No_orig),
            'lambda': lambda_val,
            'label': np.repeat(labels_full, M_regions),
            'phi': phi_res_full.flatten(),
            'phi_hat': phi_hat.flatten(),
            'aic': res['aic'],
            'bic': res['bic'],
            'wasserstein_distance': res['wasserstein_distance'], # The key exists.
            'dropped': np.repeat(dropped_full, M_regions)
        }
        
        result_df = pd.DataFrame(data_for_df)
        
        # Correction to handle the 'wasserstein_distance' being a single value
        result_df['wasserstein_distance'] = res['wasserstein_distance']
        
        output_file = os.path.join(output_base_dir, f'lambda_{lambda_val:.4f}.tsv')
        result_df.to_csv(output_file, sep='\t', index=False, na_rep='NA')

    # print(f"\n4. Success! All {len(list_of_results)} result files saved in: {output_base_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run the CLiPP2 multi-sample subclone reconstruction pipeline for a range of lambdas.'
    )
    parser.add_argument('--input_dir', required=True, help="Root directory of a patient's processed samples.")
    parser.add_argument('--output_dir', default='output', help="Directory to save the series of result TSV files.")
    parser.add_argument('--subsample_rate', type=float, default=1.0, help="Fraction of mutations to keep for the run (e.g., 0.5 for 50%%). For speed/debugging.")
    parser.add_argument('--dtype', default='float32', help="PyTorch dtype to use ('float32' is recommended for stability).")
    args = parser.parse_args()
    run(args)