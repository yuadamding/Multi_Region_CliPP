#!/usr/bin/env python3
"""
split_simulation_data.py

Reorganize multi-sample simulation outputs into per-sample folders.
For each replicate base (e.g. "CliPP2Sim4k_purity0.3_cna0.1_depth100_K2_0"),
creates:
  <output_dir>/<base>/sample<j>/
    sample<j>_snv.txt       (chromosome_index, position, alt_count, ref_count)
    sample<j>_cna.txt       (chromosome_index, start, end, major_cn, minor_cn, total_cn)
    sample<j>_purity.txt    (scalar purity)

Assumes:
  - <base>_obs.csv contains columns: mutation_id, sample_id, depth, reads, minor_true, total_true
  - <base>_params.pkl contains a dict with key 'rho' for purity

Usage:
  python split_simulation_data.py <input_dir> <output_dir>
"""
import os
import glob
import pickle
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Split simulation obs.csv into per-sample folders')
    parser.add_argument('input_dir', help='Directory containing *_obs.csv and *_params.pkl')
    parser.add_argument('output_dir', help='Root directory for per-replicate outputs')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # find all obs csv files
    obs_files = glob.glob(str(input_dir / '*_obs.csv'))
    for obs_path in obs_files:
        obs_file = Path(obs_path)
        base = obs_file.stem.replace('_obs','')  # replicate base name
        # corresponding params file
        params_path = input_dir / f"{base}_params.pkl"
        if not params_path.exists():
            print(f"Missing params for {base}, skipping.")
            continue
        # load data
        df = pd.read_csv(obs_file)
        params = pickle.load(open(params_path, 'rb'))
        purity = params.get('rho', None)
        # create replicate dir
        rep_dir = output_root / base
        rep_dir.mkdir(parents=True, exist_ok=True)
        # for each sample in this replicate
        sample_ids = sorted(df['sample_id'].unique())
        for sid in sample_ids:
            sdf = df[df['sample_id'] == sid]
            # create sample dir
            samp_name = f"sample{sid+1}"
            samp_dir = rep_dir / samp_name
            samp_dir.mkdir(exist_ok=True)
            # SNV file
            snv_df = pd.DataFrame({
                'chromosome_index': np.zeros(len(sdf), dtype=int),
                'position': sdf['mutation_id'].astype(int),
                'alt_count': sdf['reads'].astype(int),
                'ref_count': (sdf['depth'] - sdf['reads']).astype(int)
            })
            snv_df.to_csv(samp_dir / f"{samp_name}_snv.txt", sep='\t', index=False)
            # CNA file
            major_cn = sdf['total_true'] - sdf['minor_est']
            cna_df = pd.DataFrame({
                'chromosome_index': np.zeros(len(sdf), dtype=int),
                'start_position': sdf['mutation_id'].astype(int),
                'end_position': sdf['mutation_id'].astype(int),
                'major_cn': major_cn.astype(int),
                'minor_cn': sdf['minor_est'].astype(int),
                'total_cn': sdf['total_true'].astype(int)
            })
            cna_df.to_csv(samp_dir / f"{samp_name}_cna.txt", sep='\t', index=False)
            # purity file
            with open(samp_dir / f"{samp_name}_purity.txt", 'w') as pf:
                pf.write(str(purity))

    print("Reorganization complete.")

if __name__ == '__main__':
    main()