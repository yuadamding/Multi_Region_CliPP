#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General multi-sample CLiPP2 pipeline with timestamped logging and OpenMP handling.
Scans an input root directory for sample subdirectories each containing
snv.txt, cna.txt, and purity.txt. Identifies all distinct mutation loci,
aligns and processes data, runs CLiPP2 clustering, and outputs results
with per-locus, per-region CCF and cluster assignments, preserving the
locus ordering from the first sample's sorted SNV list.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import datetime
import warnings
import shutil
import pickle

import numpy as np
import pandas as pd

from clipp2.core import clipp2, group_all_regions_for_ADMM
from clipp2.preprocess import (
    process_files,
    insert_distinct_rows_multi,
    export_snv_cna_and_purity,
    preprocess
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

def log(msg):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts} - {msg}")

def list_subdirs(root):
    return [os.path.join(root, d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))]

def find_file_by_suffix(dirpath, suffix):
    fl = [f for f in os.listdir(dirpath) if f.lower().endswith(suffix.lower())]
    if not fl:
        raise FileNotFoundError(f"No '{suffix}' in {dirpath}")
    if len(fl)>1:
        raise ValueError(f"Multiple '{suffix}' in {dirpath}: {fl}")
    return os.path.join(dirpath, fl[0])

def run_clipp2(args):
    log("Starting multi-sample CLiPP2 pipeline")
    samples = sorted(list_subdirs(args.input_dir))
    if not samples:
        raise ValueError(f"No samples found under {args.input_dir}")
    log(f"Found {len(samples)} samples")

    # Step 1: process each sample into DataFrame
    dfs = []
    for sd in samples:
        log(f"Processing sample {sd}")
        dfs.append(process_files(
            find_file_by_suffix(sd,'snv.txt'),
            find_file_by_suffix(sd,'cna.txt'),
            find_file_by_suffix(sd,'purity.txt')
        ))
    aligned = insert_distinct_rows_multi(dfs)

    # Step 2: determine loci_list from first sample sorted SNV
    first_sorted = aligned[0].sort_values(['chromosome_index','position'])
    loci_list = list(zip(first_sorted['chromosome_index'], first_sorted['position']))
    loci_df = pd.DataFrame(loci_list, columns=['chromosome_index','position'])
    log(f"Using {len(loci_list)} loci from first sample for ordering")

    # Step 3: export & preprocess each aligned sample
    work = os.path.join(args.output_root,'work')
    inter = os.path.join(args.output_root,'intermediate')
    os.makedirs(work,exist_ok=True)
    os.makedirs(inter,exist_ok=True)
    region_names = []
    for idx, df in enumerate(aligned, start=1):
        name = f'sample{idx}'
        region_names.append(name)
        df_sorted = df.sort_values(['chromosome_index','position'])
        export_snv_cna_and_purity(df_sorted, work,
            f'{name}.snv.txt', f'{name}.cna.txt', f'{name}.purity.txt')
        preprocess(
            os.path.join(work,f'{name}.snv.txt'),
            os.path.join(work,f'{name}.cna.txt'),
            os.path.join(work,f'{name}.purity.txt'),
            name, os.path.join(inter,name), drop_data=False
        )
    loci_df.to_csv(os.path.join(inter,'all_loci.txt'),sep='\t',index=False)

    # Step 4: clustering
    log("Extracting matrices for clustering")
    r,n,minor,total,pur_arr,coef_list,_,_ = group_all_regions_for_ADMM(inter)
    pur_arr = np.asarray(pur_arr, dtype=float)
    if args.subsample_rate < 1.0:
        log("Subsampling rows")
        M = r.shape[0]; keep = int(M * args.subsample_rate)
        idxs = np.random.choice(M, keep, replace=False)
        r,n,minor,total = [arr[idxs] for arr in (r,n,minor,total)]
        coef_list = [mat[idxs] for mat in coef_list]
        for arr in (r,n,minor,total): arr[np.isnan(arr)] = 1
    log("Running CliPP2 clustering")
    try:
        res = clipp2(r,n,minor,total,pur_arr,coef_list,
                     Lambda=args.Lambda, device=args.device)
    except Exception as e:
        return log(f"Clustering error: {e}")
    labels, phi_cent = res['label'], res['phi']

    # Step 5: compute phi_hat
    log("Computing raw cp values")
    with np.errstate(divide='ignore',invalid='ignore'):
        frac = r / (n + 1e-12)
        phi_hat = frac * ((2 - 2*pur_arr) + (pur_arr * total)/(minor + 1e-12))

    # Step 6: build result with ordering from loci_list
    rows = []
    R = len(region_names)
    for j, (chrom, pos) in enumerate(loci_list):
        for i, region in enumerate(region_names):
            val = phi_hat[j, i] 
            lab = labels[j]
            phi = phi_cent[j, i] 
            rows.append({
                'chromosome_index': chrom,
                'position': pos,
                'region': region,
                'label': lab,
                'phi': phi,
                'phi_hat': val,
                'dropped': 0
            })
    result_df = pd.DataFrame(rows)
    result_df.to_csv(os.path.join(args.output_root,'result.txt'),
                     sep='\t', index=False)

    # Step 7: cleanup
    log("Cleaning up work directory")
    shutil.rmtree(work, ignore_errors=True)
    log("Pipeline completed successfully")

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Multi-sample CLiPP2 pipeline')
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_root', default='output')
    parser.add_argument('--Lambda', type=float, default=0.1)
    parser.add_argument('--subsample_rate', type=float, default=1.0)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    run_clipp2(args)